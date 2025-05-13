#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
import sys
from datetime import datetime

from call_connection_manager import CallConfigManager, SessionManager
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import AudioRawFrame, EndFrame, EndTaskFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

class SilenceDetector(FrameProcessor):
    """Processor that detects silence and prompts user after threshold."""
    
    def __init__(self, silence_threshold=10, max_unanswered_prompts=3):
        super().__init__()
        self.silence_threshold = silence_threshold
        self.max_unanswered_prompts = max_unanswered_prompts
        self.last_audio_time = None
        self.silence_events = 0
        self.unanswered_prompts = 0
        self.call_start_time = datetime.now()
        
    async def process_frame(self, frame):
        # Reset silence timer on any audio frame
        if isinstance(frame, AudioRawFrame):
            self.last_audio_time = datetime.now()
            await self.push_frame(frame)
        elif isinstance(frame, TextFrame):
            # Reset unanswered prompts counter when user responds
            self.unanswered_prompts = 0
            await self.push_frame(frame)
        else:
            # Check for silence
            if self.last_audio_time and (datetime.now() - self.last_audio_time).total_seconds() > self.silence_threshold:
                if self.unanswered_prompts < self.max_unanswered_prompts:
                    self.silence_events += 1
                    self.unanswered_prompts += 1
                    prompt = "Are you still there? Please respond if you'd like to continue."
                    await self.push_frame(TextFrame(prompt))
                    self.last_audio_time = datetime.now()  # Reset timer
                else:
                    await self.push_frame(TextFrame("We'll end the call now. Goodbye!"))
                    await self.push_frame(EndFrame())
            else:
                await self.push_frame(frame)
    
    def get_stats(self):
        return {
            "duration": (datetime.now() - self.call_start_time).total_seconds(),
            "silence_events": self.silence_events,
            "unanswered_prompts": self.unanswered_prompts
        }

async def main(
    room_url: str,
    token: str,
    body: dict,
):
    # Create config manager
    call_config_manager = CallConfigManager.from_json_string(body) if body else CallConfigManager()
    test_mode = call_config_manager.is_test_mode()
    dialin_settings = call_config_manager.get_dialin_settings()

    # Set up transport
    if test_mode:
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )
    else:
        daily_dialin_settings = DailyDialinSettings(
            call_id=dialin_settings.get("call_id"), 
            call_domain=dialin_settings.get("call_domain")
        )
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            dialin_settings=daily_dialin_settings,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )

    transport = DailyTransport(
        room_url,
        token,
        "Enhanced Dial-in Bot",
        transport_params,
    )

    # Initialize services
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",
    )

    # Silence detection setup
    silence_detector = SilenceDetector(
        silence_threshold=10,
        max_unanswered_prompts=3
    )

    # LLM setup
    system_instruction = """You are Chatbot, a friendly helpful assistant. Respond naturally to users. 
    If the user ends the conversation, call terminate_call."""
    
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [{"role": "system", "content": system_instruction}]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Function definitions
    async def terminate_call(params: FunctionCallParams):
        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    terminate_call_function = FunctionSchema(
        name="terminate_call",
        description="Call this to end the call.",
        properties={},
        required=[],
    )
    tools = ToolsSchema(standard_tools=[terminate_call_function])
    llm.register_function("terminate_call", terminate_call)

    # Build pipeline
    pipeline = Pipeline([
        transport.input(),
        silence_detector,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant()
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Event handlers
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.debug(f"Call stats: {silence_detector.get_stats()}")
        await task.cancel()

    # Run pipeline
    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Dial-in Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.token, args.body))