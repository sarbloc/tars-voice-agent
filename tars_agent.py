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
SESSION_ID = os.getenv("SESSION_ID", "voice-user")
GREETING = os.getenv("GREETING_MESSAGE", "Hey, what's going on?")
STT_PROMPT = os.getenv("STT_VOCABULARY_HINT", "")

# Voice-only instructions — sent as system message to OpenClaw with every request.
# Output is fed directly to a TTS engine that relies on punctuation for pacing.
VOICE_INSTRUCTIONS = """This is a live voice call. Your response will be read aloud by a text-to-speech engine.
Write like a radio presenter reading a transcript. Natural spoken English, not written English.
Use commas and periods liberally — they control pacing in speech. A comma creates a short pause, a period creates a longer one.
Use ellipses (…) when you want a dramatic pause or moment of thought.
Use semicolons when you need a breath between two related ideas.
Keep sentences short — two clauses maximum. Break longer thoughts into separate sentences.
Write time ranges with 'to' — say '11am to 1pm', not '11am - 1pm'. The dash works but 'to' sounds more natural spoken aloud.
Spell out abbreviations: 'Doctor', not 'Dr.'; 'versus', not 'vs.'.
Write small numbers as words: 'three meetings', not '3 meetings'. Use digits for larger numbers: '150 people'.
Use 'and' before the last item in any list.
Never use colons to introduce a list. Instead say 'You have three meetings today. First… Second… And finally…'
Never use markdown formatting — no asterisks, no hashes, no backticks, no bullet points.
Never use tables or structured data formats. Describe data conversationally.
Keep responses short and conversational — 3-5 sentences max."""

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
    """Clean text for Kokoro TTS — strip markdown artifacts, preserve all punctuation.

    Kokoro uses periods, commas, question marks, exclamation marks, semicolons,
    ellipses, hyphens/dashes, colons, and parentheses for prosody and pacing.
    All of these pass through untouched.
    """
    # --- Strip markdown formatting ---
    # Bold/italic: **text**, *text*, __text__, _text_
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)
    # Inline code and code blocks
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Heading markers (strip #, keep the text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bullet list markers at start of line only (- item or * item)
    # NOT mid-sentence hyphens like "11am - 1pm"
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    # Numbered list markers at start of line (1. item, 2. item)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Strip emojis (Unicode emoji ranges)
    text = re.sub(
        r"[\U0001F600-\U0001F64F"  # emoticons
        r"\U0001F300-\U0001F5FF"   # symbols & pictographs
        r"\U0001F680-\U0001F6FF"   # transport & map
        r"\U0001F1E0-\U0001F1FF"   # flags
        r"\U00002702-\U000027B0"   # dingbats
        r"\U0000FE00-\U0000FE0F"   # variation selectors
        r"\U0001F900-\U0001F9FF"   # supplemental symbols
        r"\U0001FA00-\U0001FA6F"   # chess symbols
        r"\U0001FA70-\U0001FAFF"   # symbols extended-A
        r"\U00002600-\U000026FF"   # misc symbols
        r"]+", "", text)

    # --- Normalize things Kokoro struggles with ---
    # Common abbreviations
    abbreviations = {
        "Dr.": "Doctor", "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miss",
        "vs.": "versus", "etc.": "et cetera", "approx.": "approximately",
        "dept.": "department", "govt.": "government",
    }
    for abbr, expansion in abbreviations.items():
        text = text.replace(abbr, expansion)

    # --- Clean up whitespace ---
    # Collapse blank lines and extra whitespace from stripped markers
    text = re.sub(r"\n\s*\n", "\n", text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = " ".join(lines)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    return text.strip()


class TARSAgent(Agent):
    """Voice frontend for the OpenClaw TARS agent."""

    def __init__(self):
        super().__init__(instructions=VOICE_INSTRUCTIONS)
        # Stable session — avoids expensive agent re-initialization on each connection
        self._session_id = SESSION_ID

    async def on_enter(self):
        self.session.say(GREETING)

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
            prompt=STT_PROMPT or "TARS is an AI voice assistant.",
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
