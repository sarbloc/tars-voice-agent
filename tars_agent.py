"""
TARS Voice Agent — Real-time voice frontend for OpenClaw TARS agent.

Uses:
- Silero VAD for voice activity detection
- LiveKit turn detector for natural conversation flow
- faster-whisper (self-hosted) for speech-to-text
- OpenClaw cooper agent via Chat Completions HTTP API (streaming)
- Kokoro (self-hosted) for text-to-speech with am_onyx voice

STT transcript → OpenClaw HTTP streaming → Kokoro TTS (chunk-by-chunk)
"""

import json
import os
import logging
from collections.abc import AsyncIterable
from dotenv import load_dotenv
import httpx

from livekit.agents import (
    Agent,
    AgentSession,
    AgentServer,
    JobContext,
    ModelSettings,
    cli,
    llm,
)
from livekit.plugins import openai, silero

load_dotenv()

logger = logging.getLogger("tars-agent")
logger.setLevel(logging.INFO)

# Read config from environment
WHISPER_BASE_URL = os.getenv("WHISPER_BASE_URL", "http://192.168.50.13:8001/v1")
KOKORO_BASE_URL = os.getenv("KOKORO_BASE_URL", "http://192.168.50.13:8002/v1")
TARS_VOICE = os.getenv("TARS_VOICE", "am_onyx")
OPENCLAW_URL = os.getenv("OPENCLAW_URL", "http://127.0.0.1:18789/v1")
OPENCLAW_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")

# Voice-only instructions — sent as system message to OpenClaw with every request
VOICE_INSTRUCTIONS = """This is a live voice call. You MUST respond with plain text only.
No voice notes, no audio, no markdown, no bullet points, no formatting.
Keep responses short and conversational — 1-3 sentences max.
Natural spoken language only. Like a real phone call."""

# Shared httpx client for OpenClaw streaming calls
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0))
    return _http_client


async def _stream_openclaw(message: str, session_id: str) -> AsyncIterable[str]:
    """Stream a response from OpenClaw Chat Completions API, yielding text chunks."""
    client = _get_http_client()
    async with client.stream(
        "POST",
        f"{OPENCLAW_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENCLAW_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openclaw:cooper",
            "messages": [
                {"role": "system", "content": VOICE_INSTRUCTIONS},
                {"role": "user", "content": message},
            ],
            "user": session_id,
            "stream": True,
        },
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


class TARSAgent(Agent):
    """Voice frontend for the OpenClaw TARS agent."""

    def __init__(self):
        super().__init__(instructions=VOICE_INSTRUCTIONS)
        # Stable session — avoids expensive agent re-initialization on each connection
        self._session_id = "sarbloc-voice"

    async def on_enter(self):
        self.session.say("Hey Sarbloc, what's going on?")

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str]:
        """Stream LLM responses from OpenClaw — only sends the user message.

        OpenClaw manages its own conversation context via the session user ID,
        so we don't forward LiveKit's chat history.
        """
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
            yield "I didn't catch that. Could you say it again?"
            return

        logger.info("sending to openclaw: %s", user_message[:100])
        async for chunk in _stream_openclaw(user_message, self._session_id):
            yield chunk


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Entry point — starts the TARS voice pipeline."""
    session = AgentSession(
        vad=silero.VAD.load(),

        # Silence-based turn detection: wait 0.6s of silence before treating
        # the user's turn as complete. The ONNX-based EnglishModel fails in the
        # inference subprocess on this setup, so we use fixed-delay instead.
        turn_handling={
            "turn_detection": "vad",
            "endpointing": {"min_delay": 0.6, "max_delay": 1.5},
        },

        # STT — self-hosted faster-whisper
        stt=openai.STT(
            model="Systran/faster-whisper-large-v3",
            base_url=WHISPER_BASE_URL,
            api_key="not-needed",
            language="en",
        ),

        # LLM — placeholder so the framework activates the LLM pipeline;
        # actual calls are handled by TARSAgent.llm_node override above
        llm=openai.LLM(
            model="openclaw:cooper",
            base_url=OPENCLAW_URL,
            api_key=OPENCLAW_TOKEN,
        ),

        # TTS — self-hosted Kokoro
        tts=openai.TTS(
            model="tts-1",
            voice=TARS_VOICE,
            base_url=KOKORO_BASE_URL,
            api_key="not-needed",
            response_format="wav",
        ),

    )

    await session.start(agent=TARSAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
