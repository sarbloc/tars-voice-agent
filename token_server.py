"""Combined web + token server for the TARS voice client.

Serves the web UI, the /token endpoint, and a /tts proxy to Kokoro on port 8080.
"""

import os
import json
import urllib.request
import urllib.error
from http.server import HTTPServer, SimpleHTTPRequestHandler
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

load_dotenv()

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")
LIVEKIT_IDENTITY = os.getenv("LIVEKIT_IDENTITY", "user")
KOKORO_BASE_URL = os.getenv("KOKORO_BASE_URL", "http://192.168.50.13:8002/v1")
WHISPER_BASE_URL = os.getenv("WHISPER_BASE_URL", "http://192.168.50.13:8001/v1")
TARS_VOICE = os.getenv("TARS_VOICE", "am_onyx")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def _probe(self, base_url: str) -> dict:
        url = f"{base_url.rstrip('/')}/models"
        try:
            req = urllib.request.Request(
                url, headers={"Authorization": "Bearer not-needed"}
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                return {"ok": 200 <= resp.status < 500, "status": resp.status}
        except urllib.error.HTTPError as e:
            return {"ok": True, "status": e.code}
        except Exception as e:
            return {"ok": False, "error": str(e)[:120]}

    def do_GET(self):
        if self.path == "/health":
            whisper = self._probe(WHISPER_BASE_URL)
            kokoro = self._probe(KOKORO_BASE_URL)
            body = json.dumps({
                "ok": whisper["ok"] and kokoro["ok"],
                "whisper": whisper,
                "kokoro": kokoro,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/token":
            token = (
                AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
                .with_identity(LIVEKIT_IDENTITY)
                .with_grants(VideoGrants(room_join=True, room="tars-room"))
                .to_jwt()
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"token": token}).encode())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/tts":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return

        text = (body.get("text") or "").strip()
        if not text:
            self.send_error(400, "text required")
            return

        voice = body.get("voice") or TARS_VOICE
        fmt = body.get("format") or "wav"
        payload = json.dumps({
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "response_format": fmt,
        }).encode()

        req = urllib.request.Request(
            f"{KOKORO_BASE_URL.rstrip('/')}/audio/speech",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer not-needed",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                audio = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            print(f"[tts] kokoro error {e.code}: {detail}")
            self.send_error(502, f"kokoro error {e.code}")
            return
        except Exception as e:
            print(f"[tts] proxy error: {e}")
            self.send_error(502, "kokoro unreachable")
            return

        mime = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/ogg",
            "flac": "audio/flac",
        }.get(fmt, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), Handler)
    print("TARS web + token server on http://0.0.0.0:8080")
    server.serve_forever()
