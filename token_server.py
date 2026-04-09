"""Combined web + token server for the TARS voice client.

Serves the web UI and the /token endpoint on port 8080.
"""

import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

load_dotenv()

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")
LIVEKIT_IDENTITY = os.getenv("LIVEKIT_IDENTITY", "user")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
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

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), Handler)
    print("TARS web + token server on http://0.0.0.0:8080")
    server.serve_forever()
