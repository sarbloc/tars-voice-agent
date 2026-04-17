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
import uuid
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
SESSION_ID_PREFIX = os.getenv("SESSION_ID_PREFIX", "voice")
_greetings_env = os.getenv("GREETING_MESSAGES") or os.getenv("GREETING_MESSAGE", "")
if _greetings_env:
    GREETINGS = [g.strip() for g in _greetings_env.split("|") if g.strip()]
else:
    GREETINGS = [
        "Hey, what's going on?",
        "Hey, what's up?",
        "Hi. What do you need?",
        "Yeah, I'm here. What's up?",
        "Hey. What's on your mind?",
    ]
STT_PROMPT = os.getenv("STT_VOCABULARY_HINT", "")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid float for %s=%r, using default %s", name, raw, default)
        return default


# VAD + interruption tuning — higher thresholds reject distant/cross-room speech
# and brief background noise. Override via env vars.
VAD_ACTIVATION_THRESHOLD = _env_float("VAD_ACTIVATION_THRESHOLD", 0.7)
VAD_MIN_SPEECH_DURATION = _env_float("VAD_MIN_SPEECH_DURATION", 0.2)
VAD_MIN_SILENCE_DURATION = _env_float("VAD_MIN_SILENCE_DURATION", 0.55)
INTERRUPTION_MIN_DURATION = _env_float("INTERRUPTION_MIN_DURATION", 0.8)
# Extra delay after the turn detector decides you're done before committing the turn.
TURN_ENDPOINTING_MIN_DELAY = _env_float("TURN_ENDPOINTING_MIN_DELAY", 0.4)
# Kokoro playback speed multiplier — drop below 1.0 for more natural pacing.
KOKORO_SPEED = _env_float("KOKORO_SPEED", 0.92)

# Voice-only instructions — sent as system message to OpenClaw with every request.
# Output is fed directly to a TTS engine that relies on punctuation for pacing.
VOICE_INSTRUCTIONS = """This is a live voice call. Your response is read aloud by a text-to-speech engine that uses punctuation for pacing, so every comma, period, dash, and ellipsis becomes a pause. Write the way you'd actually speak — not the way you'd write.

=== Cadence ===
Every sentence should sound like one natural breath. Aim for 8–14 words; split anything longer.
Use ellipses (…) inside a sentence to mark the beat of a real pause — the kind a thoughtful person takes before the key word. Example: "It's there… on the second shelf." "Yeah… that's the one."
Use em-dashes (—) for a quick aside or a self-correction. Example: "The meeting's at three — no, three-thirty."
Use commas liberally to break up noun phrases and clauses. "So, looking at the calendar, you've got two things today." Three commas in a sentence is fine.
Semicolons are okay for two closely linked thoughts, but prefer a period and a fresh sentence.

=== Openers ===
Start responses the way a person actually starts talking: "Yeah," "So," "Okay," "Right," "Alright," "Hm," "Honestly," "Look." Mix them up; don't open two responses in a row the same way.
Never open with "I would," "I can help you with," "Certainly," or anything that sounds like a chatbot.

=== Voice ===
Contractions always — "you're," "it's," "that's," "I'll," "can't." Never "you are," "it is" unless you're emphasising.
Avoid hedging filler like "feel free to," "please note," "it's worth mentioning." Just say the thing.
When stating a fact, land it cleanly. Don't wrap it in qualifiers.

=== Formatting ===
Never use markdown — no asterisks, hashes, backticks, bullets, tables. Describe any list conversationally: "First… second… and finally…"
Spell out abbreviations: "Doctor," not "Dr."; "versus," not "vs."
Small numbers as words ("three meetings"), larger as digits ("150 people").
Say time ranges with "to": "eleven to one," not "11-1."
Use "and" before the last item in a list.

=== Length ===
3–5 sentences. If the answer needs more, break it into a first-reply and offer to go deeper."""

# Short filler words — spoken first while waiting for LLM response.
SHORT_FILLERS = ["um... ", "ok... ", "yep... ", "hmm... "]

# Longer filler phrases — spoken if the LLM is still not responding.
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
        self._session_id = f"{SESSION_ID_PREFIX}-{uuid.uuid4().hex[:8]}"
        logger.info("new session: %s", self._session_id)

    async def on_enter(self):
        self.session.say(random.choice(GREETINGS), allow_interruptions=False)

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
        # A "tars.filler" attribute tells the web UI to keep showing "thinking"
        # instead of "speaking" while fillers play.
        #   1.0s — random short filler ("um...", "ok...", etc.)
        #   2.0s — random longer filler phrase
        #   3.0s — second random filler phrase
        FILLER_SCHEDULE = [
            (1.0, None, SHORT_FILLERS),
            (2.0, None, FILLER_PHRASES),
            (3.0, None, FILLER_PHRASES),
        ]

        room = self.session.room_io.room
        stream = _stream_openclaw(user_message, self._session_id).__aiter__()
        next_task = asyncio.ensure_future(anext(stream))
        first_chunk = None
        used_filler = False

        for timeout, phrase, pool in FILLER_SCHEDULE:
            try:
                first_chunk = await asyncio.wait_for(asyncio.shield(next_task), timeout=timeout)
                logger.info("first token at %.2fs", time.monotonic() - t0)
                break
            except asyncio.TimeoutError:
                if not used_filler:
                    await room.local_participant.set_attributes({"tars.filler": "true"})
                    used_filler = True
                text = phrase if phrase else random.choice(pool) + " "
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

        # Clear filler flag — real content is starting
        if used_filler:
            await room.local_participant.set_attributes({"tars.filler": ""})

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
    logger.info(
        "vad config: activation_threshold=%.2f min_speech=%.2fs min_silence=%.2fs; "
        "interruption min_duration=%.2fs; turn endpointing min_delay=%.2fs; "
        "kokoro speed=%.2f",
        VAD_ACTIVATION_THRESHOLD,
        VAD_MIN_SPEECH_DURATION,
        VAD_MIN_SILENCE_DURATION,
        INTERRUPTION_MIN_DURATION,
        TURN_ENDPOINTING_MIN_DELAY,
        KOKORO_SPEED,
    )

    session = AgentSession(
        vad=silero.VAD.load(
            activation_threshold=VAD_ACTIVATION_THRESHOLD,
            min_speech_duration=VAD_MIN_SPEECH_DURATION,
            min_silence_duration=VAD_MIN_SILENCE_DURATION,
        ),
        turn_handling={
            "turn_detection": EnglishModel(),
            "interruption": {"min_duration": INTERRUPTION_MIN_DURATION},
            "endpointing": {"min_delay": TURN_ENDPOINTING_MIN_DELAY},
        },

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
            speed=KOKORO_SPEED,
        ),

    )

    await session.start(agent=TARSAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
