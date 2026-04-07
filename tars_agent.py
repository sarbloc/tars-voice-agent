"""
TARS Voice Agent — Real-time voice frontend for OpenClaw TARS agent.

Uses:
- Silero VAD for voice activity detection
- LiveKit turn detector for natural conversation flow
- faster-whisper (self-hosted) for speech-to-text
- OpenClaw cooper agent for LLM (same brain as Telegram TARS)
- Kokoro (self-hosted) for text-to-speech with am_onyx voice

STT transcript → OpenClaw (cooper/TARS) → text response → Kokoro TTS
"""

import asyncio
import json
import os
import logging
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AgentServer,
    JobContext,
    ModelSettings,
    cli,
    llm,
)
from livekit.plugins import openai, silero, anthropic
from livekit.plugins.turn_detector.english import EnglishModel

load_dotenv()

logger = logging.getLogger("tars-agent")
logger.setLevel(logging.INFO)

# Read config from environment
WHISPER_BASE_URL = os.getenv("WHISPER_BASE_URL", "http://192.168.50.13:8001/v1")
KOKORO_BASE_URL = os.getenv("KOKORO_BASE_URL", "http://192.168.50.13:8002/v1")
TARS_VOICE = os.getenv("TARS_VOICE", "am_onyx")
OPENCLAW_BIN = os.getenv("OPENCLAW_BIN", os.path.expanduser("~/.npm-global/bin/openclaw"))

# Voice-only instructions — personality/memory/tools are handled by OpenClaw
VOICE_INSTRUCTIONS = """This is a voice conversation. Keep responses concise and natural for speech.
No bullet points, no markdown, no formatting — pure spoken language.
Short sentences. Natural pauses. Like a real conversation."""


async def openclaw_ask(message: str) -> str:
    """Call OpenClaw cooper agent via CLI and return the text response."""
    proc = await asyncio.create_subprocess_exec(
        OPENCLAW_BIN, "agent",
        "--agent", "cooper",
        "--message", message,
        "--session-id", "voice",
        "--thinking", "off",
        "--json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.error("openclaw agent failed: %s", stderr.decode())
        return "Sorry, I had trouble thinking about that. Try again."

    try:
        data = json.loads(stdout.decode())
        payloads = data.get("result", {}).get("payloads", [])
        texts = [p["text"] for p in payloads if isinstance(p, dict) and "text" in p]
        return " ".join(texts) if texts else "I don't have a response for that."
    except (json.JSONDecodeError, KeyError) as e:
        logger.error("failed to parse openclaw response: %s", e)
        return "Sorry, something went wrong on my end."


class TARSAgent(Agent):
    """Voice frontend for the OpenClaw TARS agent."""

    def __init__(self):
        super().__init__(instructions=VOICE_INSTRUCTIONS)

    async def on_enter(self):
        """Called when TARS becomes the active agent in a session."""
        # Use say() instead of generate_reply() — sends text directly to TTS
        # without needing the LLM pipeline
        self.session.say("Hey Sarbloc, what's going on?")

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> str:
        """Route LLM calls through OpenClaw instead of a direct API."""
        # Extract the latest user message from chat context
        # ChatMessage.content is always a list[ChatContent] where
        # ChatContent = ImageContent | AudioContent | Instructions | str
        user_message = ""
        for msg in reversed(chat_ctx.items):
            if not hasattr(msg, "role") or msg.role != "user":
                continue
            parts = []
            for part in msg.content:
                if isinstance(part, str):
                    parts.append(part)
                elif hasattr(part, "text"):
                    parts.append(part.text)
            user_message = " ".join(parts)
            break

        if not user_message:
            return "I didn't catch that. Could you say it again?"

        logger.info("sending to openclaw: %s", user_message[:100])
        response = await openclaw_ask(user_message)
        logger.info("openclaw response: %s", response[:100])
        return response


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Entry point — starts the TARS voice pipeline."""
    session = AgentSession(
        # Voice Activity Detection — detects when user is speaking
        vad=silero.VAD.load(),

        # Turn detection — knows when user has finished their turn
        turn_detection=EnglishModel(),

        # Speech-to-Text — self-hosted faster-whisper via OpenAI-compatible API
        stt=openai.STT(
            model="Systran/faster-whisper-large-v3",
            base_url=WHISPER_BASE_URL,
            api_key="not-needed",
            language="en",
        ),

        # LLM — placeholder to satisfy framework check; actual calls go through
        # TARSAgent.llm_node override which routes to OpenClaw CLI
        llm=anthropic.LLM(model="claude-haiku-4-5-20251001"),

        # Text-to-Speech — self-hosted Kokoro via OpenAI-compatible API
        # model="tts-1" forces the AudioChunkedStream path (raw audio download)
        # instead of SSE streaming which Kokoro doesn't support
        tts=openai.TTS(
            model="tts-1",
            voice=TARS_VOICE,
            base_url=KOKORO_BASE_URL,
            api_key="not-needed",
            response_format="wav",
        ),

        # Wait 800ms of silence before assuming user is done
        min_consecutive_speech_delay=0.8,
    )

    await session.start(agent=TARSAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
