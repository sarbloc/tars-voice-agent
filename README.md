# TARS Voice Agent

A real-time voice assistant built on [LiveKit Agents](https://docs.livekit.io/agents/), using self-hosted speech-to-text and text-to-speech models. Talk to any LLM through a browser — no cloud STT/TTS services required.

## How it works

```
Browser mic → LiveKit → Whisper STT → LLM (Chat Completions API) → Kokoro TTS → Browser speaker
```

1. **You speak** into the browser. LiveKit streams your audio to the agent.
2. **Silero VAD** detects voice activity; the **LiveKit turn detector** decides when you've finished speaking.
3. **Whisper** (self-hosted via [Speaches](https://github.com/speaches-ai/speaches)) transcribes your speech to text.
4. The transcript is sent to any **OpenAI-compatible Chat Completions API** — could be OpenAI, Ollama, LiteLLM, vLLM, or a custom gateway.
5. The LLM response streams back and is synthesized to speech by **Kokoro TTS** (self-hosted via [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)).
6. Audio plays back in your browser in real time.

The agent includes filler phrases ("Hold on a sec...") while waiting for slow LLM responses, and a TTS preprocessor that strips markdown and optimizes text for natural-sounding speech.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Browser     │◄───►│  LiveKit      │◄───►│  TARS Agent      │
│  (web UI)    │     │  Server       │     │  (tars_agent.py) │
└─────────────┘     └──────────────┘     └────────┬─────────┘
                                                   │
                                    ┌──────────────┼──────────────┐
                                    ▼              ▼              ▼
                              ┌──────────┐  ┌──────────┐  ┌──────────┐
                              │ Whisper  │  │ LLM API  │  │ Kokoro   │
                              │ (STT)    │  │          │  │ (TTS)    │
                              └──────────┘  └──────────┘  └──────────┘
```

## Prerequisites

- **Python 3.12+**
- **LiveKit Server** — [install guide](https://docs.livekit.io/home/self-hosting/local/)
- **Whisper STT server** — any OpenAI-compatible STT endpoint. Recommended: [Speaches](https://github.com/speaches-ai/speaches) (faster-whisper)
- **Kokoro TTS server** — any OpenAI-compatible TTS endpoint. Recommended: [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)
- **An LLM** — anything with an OpenAI-compatible Chat Completions API (OpenAI, Ollama, LiteLLM, vLLM, etc.)

### GPU recommendations

The STT and TTS servers run best with a GPU. A single consumer GPU (RTX 3060+) can handle both. CPU-only is possible but adds noticeable latency.

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USER/tars-voice-agent.git
cd tars-voice-agent
python -m venv venv
source venv/bin/activate
pip install "livekit-agents[openai,silero,turn-detector]" python-dotenv httpx
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual values
```

See `.env.example` for all available settings and descriptions.

### 3. Start LiveKit server

```bash
# Development mode (no auth needed)
livekit-server --dev
```

### 4. Start inference servers

Start your Whisper and Kokoro servers. Example with Docker:

```bash
# Speaches (faster-whisper STT) on port 8001
docker run -d --gpus all -p 8001:8001 \
  ghcr.io/speaches-ai/speaches:latest \
  --model Systran/faster-whisper-large-v3

# Kokoro-FastAPI (TTS) on port 8002
docker run -d --gpus all -p 8002:8002 \
  ghcr.io/remsky/kokoro-fastapi:latest
```

### 5. Start the agent

```bash
# Terminal 1: Token server + web UI
python token_server.py

# Terminal 2: Voice agent
python tars_agent.py dev
```

### 6. Connect

Open **http://localhost:8080** in your browser and click **Connect**.

> **Note:** Microphone access requires HTTPS on non-localhost origins. If you want to access the UI from another device (phone, tablet), you'll need a reverse proxy with TLS. See `Caddyfile.example` for a template using [Caddy](https://caddyserver.com/).

## Project structure

```
tars_agent.py        # Voice agent — VAD, STT, LLM streaming, TTS pipeline
token_server.py      # Web server — serves UI + issues LiveKit JWT tokens
web/index.html       # Browser client — connects to LiveKit, plays agent audio
Caddyfile.example    # Optional HTTPS reverse proxy config template
.env.example         # Environment variable template with all settings
tests/               # Unit + Playwright tests
```

## Customization

### Change the voice

Set `TARS_VOICE` in `.env` to any voice your Kokoro server supports. Default is `am_onyx`.

### Change the greeting

Set `GREETING_MESSAGE` in `.env`. This is what the agent says when you connect.

### Improve speech recognition

Set `STT_VOCABULARY_HINT` in `.env` with names, jargon, or product names that the speech model might mishear. Example:

```
STT_VOCABULARY_HINT=TARS is an AI assistant. Alice is speaking about Kubernetes and PostgreSQL.
```

### Use a different LLM

Point `OPENCLAW_URL` to any OpenAI-compatible endpoint:

```bash
# Ollama
OPENCLAW_URL=http://localhost:11434/v1
OPENCLAW_GATEWAY_TOKEN=not-needed

# OpenAI
OPENCLAW_URL=https://api.openai.com/v1
OPENCLAW_GATEWAY_TOKEN=sk-...
```

## Running tests

```bash
pip install pytest
python -m pytest tests/

# Web UI tests require Playwright
pip install pytest-playwright
playwright install chromium
python -m pytest tests/test_web_ui.py
```

## License

MIT
