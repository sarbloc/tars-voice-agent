"""Tests for the token server — verifies JWT issuance and static file serving."""

import json
import sys
import os
import threading
from http.client import HTTPConnection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set required env vars before import
os.environ.setdefault("LIVEKIT_API_KEY", "testkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "testsecretthatis32byteslong0000")

from token_server import Handler, HTTPServer

_PORT = 18901


def _start_server():
    server = HTTPServer(("127.0.0.1", _PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class TestTokenEndpoint:
    server = None

    @classmethod
    def setup_class(cls):
        cls.server = _start_server()

    @classmethod
    def teardown_class(cls):
        if cls.server:
            cls.server.shutdown()

    def _get(self, path: str) -> tuple[int, bytes]:
        conn = HTTPConnection("127.0.0.1", _PORT)
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read()

    def test_token_returns_200(self):
        status, _ = self._get("/token")
        assert status == 200

    def test_token_returns_json_with_token_key(self):
        _, body = self._get("/token")
        data = json.loads(body)
        assert "token" in data
        assert isinstance(data["token"], str)
        assert len(data["token"]) > 0

    def test_token_is_valid_jwt_format(self):
        _, body = self._get("/token")
        token = json.loads(body)["token"]
        # JWT has three base64-encoded parts separated by dots
        parts = token.split(".")
        assert len(parts) == 3

    def test_index_returns_200(self):
        status, body = self._get("/")
        assert status == 200
        assert b"TARS" in body

    def test_nonexistent_path_returns_404(self):
        status, _ = self._get("/nonexistent")
        assert status == 404
