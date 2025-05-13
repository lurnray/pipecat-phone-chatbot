import argparse
import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import AudioRawFrame, EndFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

class SilenceDetector(FrameProcessor):
    """Enhanced silence detection with stats tracking"""
    
    def __init__(self, silence_threshold=10, max_unanswered_prompts=3):
        super().__init__()
        self.silence_threshold = silence_threshold
        self.max_unanswered_prompts = max_unanswered_prompts
        self.last_audio_time = None
        self.silence_events = 0
        self.unanswered_prompts = 0
        self.call_start_time = datetime.now()
        
    async def process_frame(self, frame, direction):
        if isinstance(frame, AudioRawFrame):
            self.last_audio_time = datetime.now()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TextFrame):
            self.unanswered_prompts = 0
            await self.push_frame(frame, direction)
        else:
            # Check for silence
            if self.last_audio_time and (datetime.now() - self.last_audio_time).total_seconds() > self.silence_threshold:
                if self.unanswered_prompts < self.max_unanswered_prompts:
                    self.silence_events += 1
                    self.unanswered_prompts += 1
                    await self.push_frame(TextFrame("Are you still there? Please respond if you'd like to continue."), direction)
                    self.last_audio_time = datetime.now()
                else:
                    await self.push_frame(TextFrame("We'll end the call now. Goodbye!"), direction)
                    await self.push_frame(EndFrame(), direction)
            else:
                await self.push_frame(frame, direction)
    
    def get_stats(self):
        return {
            "call_start": self.call_start_time.isoformat(),
            "duration_seconds": (datetime.now() - self.call_start_time).total_seconds(),
            "silence_events": self.silence_events,
            "unanswered_prompts": self.unanswered_prompts,
            "termination_reason": "timeout" if self.unanswered_prompts >=3 else "normal"
        }

async def main(room_url: str, token: str, body: dict):
    # Transport setup
    transport = DailyTransport(
        room_url,
        token,
        "Enhanced Phone Bot",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer()
        )
    )

    # Services
    tts = CartesiaTTSService(api_key=os.getenv("CARTESIA_API_KEY"))
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    silence_detector = SilenceDetector()

    # Pipeline
    pipeline = Pipeline([
        transport.input(),
        silence_detector,
        llm,
        tts,
        transport.output()
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Call stats: {silence_detector.get_stats()}")
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON config")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.token, args.body))