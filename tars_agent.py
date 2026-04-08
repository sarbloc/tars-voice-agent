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

# Python 3.14 defaults to forkserver which breaks LiveKit's turn detector
# ONNX inference subprocess IPC. Force fork before any LiveKit imports.
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import asyncio
import json
import os
import logging
import random
import re
import time
from collections.abc import AsyncIterable
from dotenv import load_dotenv
import httpx

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AgentServer,
    JobContext,
    ModelSettings,
    cli,
    llm,
)
from livekit.agents.types import FlushSentinel
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.english import EnglishModel

load_dotenv()

logger = logging.getLogger("tars-agent")
logger.setLevel(logging.INFO)

# Read config from environment
WHISPER_BASE_URL = os.getenv("WHISPER_BASE_URL", "http://192.168.50.13:8001/v1")
KOKORO_BASE_URL = os.getenv("KOKORO_BASE_URL", "http://192.168.50.13:8002/v1")
TARS_VOICE = os.getenv("TARS_VOICE", "am_onyx")
OPENCLAW_URL = os.getenv("OPENCLAW_URL", "http://127.0.0.1:18789/v1")
OPENCLAW_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")

# Voice-only instructions — sent as system message to OpenClaw with every request.
# Output is fed directly to a TTS engine that relies on punctuation for pacing.
VOICE_INSTRUCTIONS = """This is a live voice call. Your response will be read aloud by a text-to-speech engine.
You MUST respond with plain text only — no markdown, no bullet points, no numbered lists, no formatting.
Write your response exactly as a human would speak it, like a radio transcript.
When listing items, group them in natural phrases: "You've got Mercury, Venus, Earth, and Mars for the inner planets. Then Jupiter, Saturn, Uranus, and Neptune for the outer ones."
Never list items one per line or one per sentence. Group related items together in spoken phrases.
Use short sentences with periods between them for pacing.
Keep responses short and conversational — 3-5 sentences max.
Natural spoken language only. Like a real phone call."""

# Filler phrases spoken while waiting for slow LLM responses.
# Tone: dry, no-nonsense — consistent with TARS personality.
FILLER_PHRASES = [
    "Let me check on that.",
    "Hold on a sec.",
    "One moment.",
    "Checking now.",
    "Working on it.",
    "Give me a second.",
]

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


def _preprocess_tts_text(text: str) -> str:
    """Clean markdown and improve punctuation for Kokoro TTS prosody."""
    # Strip bold/italic markers
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    # Strip inline code
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Strip heading markers, add period after heading text
    text = re.sub(r"^#{1,6}\s+(.+)$", r"\1.", text, flags=re.MULTILINE)
    # Replace em dashes with commas
    text = text.replace("—", ", ").replace("–", ", ")
    # Convert numbered list items: "1. foo" -> "First, foo", etc.
    ordinals = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"]
    def _replace_numbered(m: re.Match) -> str:
        n = int(m.group(1))
        prefix = ordinals[n - 1] + ", " if 1 <= n <= len(ordinals) else str(n) + ", "
        return prefix
    text = re.sub(r"^\s*(\d{1,2})\.\s+", _replace_numbered, text, flags=re.MULTILINE)
    # Convert bullet list items: "- foo" or "* foo" -> sentence separator
    text = re.sub(r"^\s*[-*]\s+", "\n", text, flags=re.MULTILINE)
    # Add commas after common transitional words if missing
    for word in ("However", "Also", "Additionally", "Furthermore", "Moreover", "Finally", "Meanwhile", "Otherwise"):
        text = re.sub(rf"\b{word}\s(?!,)", f"{word}, ", text)
    # Ensure each line ends with punctuation before collapsing
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if stripped and stripped[-1] not in ".!?,;:":
            lines[i] = stripped + "."
    text = " ".join(lines)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    # Clean up double punctuation from replacements
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r",\s*,", ",", text)
    return text.strip()


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

        t0 = time.monotonic()
        logger.info("sending to openclaw: %s", user_message[:100])

        # Filler phrases while waiting for the first LLM token.
        # Each fires at a cumulative timeout; FlushSentinel forces immediate TTS.
        #   1.5s — "um..."
        #   3.0s — random filler phrase
        #   6.0s — second random filler phrase
        FILLER_SCHEDULE = [
            (1.5, "um... "),
            (1.5, None),  # None = pick random from FILLER_PHRASES
            (3.0, None),
        ]

        stream = _stream_openclaw(user_message, self._session_id).__aiter__()
        next_task = asyncio.ensure_future(anext(stream))
        first_chunk = None

        for timeout, phrase in FILLER_SCHEDULE:
            try:
                first_chunk = await asyncio.wait_for(asyncio.shield(next_task), timeout=timeout)
                logger.info("first token at %.2fs", time.monotonic() - t0)
                break
            except asyncio.TimeoutError:
                text = phrase if phrase else random.choice(FILLER_PHRASES) + " "
                logger.info("%.2fs, speaking filler: %s", time.monotonic() - t0, text.strip())
                yield text
                yield FlushSentinel()
            except StopAsyncIteration:
                yield "I didn't catch that. Could you say it again?"
                return

        # If all fillers exhausted, wait indefinitely for the response
        if first_chunk is None:
            try:
                first_chunk = await next_task
                logger.info("first token at %.2fs", time.monotonic() - t0)
            except StopAsyncIteration:
                return

        yield first_chunk
        async for chunk in stream:
            yield chunk


    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Preprocess LLM text to improve Kokoro TTS prosody, then synthesize."""
        async def _preprocess_stream() -> AsyncIterable[str]:
            buf = ""
            async for chunk in text:
                if isinstance(chunk, FlushSentinel):
                    if buf:
                        yield _preprocess_tts_text(buf)
                        buf = ""
                    yield chunk
                    continue
                buf += chunk
            # Preprocess and yield the entire response as one chunk —
            # the framework's StreamAdapter handles sentence-level TTS splitting,
            # and Kokoro uses the punctuation within for pacing.
            if buf:
                yield _preprocess_tts_text(buf)

        return Agent.default.tts_node(self, _preprocess_stream(), model_settings)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Entry point — starts the TARS voice pipeline."""
    session = AgentSession(
        vad=silero.VAD.load(),
        turn_detection=EnglishModel(),

        # STT — self-hosted Speaches (faster-whisper)
        stt=openai.STT(
            model="Systran/faster-whisper-large-v3",
            base_url=WHISPER_BASE_URL,
            api_key="not-needed",
            language="en",
            prompt="TARS is an AI assistant. Sarbloc is speaking to TARS about OpenClaw, Qdrant, and Home Assistant.",
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
