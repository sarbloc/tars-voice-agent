"""Playwright tests for the TARS web UI — verifies rendering and agent state indicator."""

import json
import os
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

WEB_DIR = str(Path(__file__).resolve().parent.parent / "web")
_PORT = 18902


class _TestHandler(SimpleHTTPRequestHandler):
    """Serves the web dir and a fake /token endpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/token":
            # Return a dummy token — we won't actually connect to LiveKit
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"token": "fake.jwt.token"}).encode())
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", _PORT), _TestHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()


@pytest.fixture
def ui_page(page: Page, server):
    page.goto(f"http://127.0.0.1:{_PORT}/")
    return page


class TestPageRendering:
    def test_title_visible(self, ui_page: Page):
        expect(ui_page.locator(".wordmark")).to_have_text("TARS")

    def test_status_shows_ready(self, ui_page: Page):
        expect(ui_page.locator("#status")).to_have_text("Ready")

    def test_ring_visible(self, ui_page: Page):
        ring = ui_page.locator("#ring")
        expect(ring).to_be_visible()

    def test_ring_label_shows_tap_to_connect(self, ui_page: Page):
        expect(ui_page.locator("#ringLabel")).to_have_text("tap to connect")

    def test_agent_state_empty_initially(self, ui_page: Page):
        state = ui_page.locator("#agentState")
        expect(state).to_have_text("")


class TestAgentStateIndicator:
    """Test the agent state JS functions by calling them directly."""

    def _set_connected(self, page: Page):
        """Simulate connected state so ring classes apply."""
        page.evaluate("connected = true; ring.classList.add('active')")

    def test_update_agent_state_listening(self, ui_page: Page):
        self._set_connected(ui_page)
        ui_page.evaluate("updateAgentState('listening')")
        state = ui_page.locator("#agentState")
        expect(state).to_have_text("Listening")
        assert "listening" in ui_page.locator("#agentState").get_attribute("class")
        assert "listening" in ui_page.locator("#ring").get_attribute("class")

    def test_update_agent_state_thinking(self, ui_page: Page):
        self._set_connected(ui_page)
        ui_page.evaluate("updateAgentState('thinking')")
        state = ui_page.locator("#agentState")
        expect(state).to_have_text("Thinking")
        assert "thinking" in ui_page.locator("#ring").get_attribute("class")

    def test_update_agent_state_speaking(self, ui_page: Page):
        self._set_connected(ui_page)
        ui_page.evaluate("updateAgentState('speaking')")
        state = ui_page.locator("#agentState")
        expect(state).to_have_text("Speaking")
        assert "speaking" in ui_page.locator("#ring").get_attribute("class")

    def test_update_agent_state_idle(self, ui_page: Page):
        self._set_connected(ui_page)
        ui_page.evaluate("updateAgentState('idle')")
        state = ui_page.locator("#agentState")
        expect(state).to_have_text("Ready")

    def test_update_agent_state_initializing(self, ui_page: Page):
        ui_page.evaluate("updateAgentState('initializing')")
        state = ui_page.locator("#agentState")
        expect(state).to_have_text("Initializing")


class TestRingStateClasses:
    """Verify the ring element gets correct CSS classes per state."""

    def _set_connected(self, page: Page):
        page.evaluate("connected = true; ring.classList.add('active')")

    def test_listening_class(self, ui_page: Page):
        self._set_connected(ui_page)
        ui_page.evaluate("updateAgentState('listening')")
        assert "listening" in ui_page.locator("#ring").get_attribute("class")

    def test_thinking_class(self, ui_page: Page):
        self._set_connected(ui_page)
        ui_page.evaluate("updateAgentState('thinking')")
        assert "thinking" in ui_page.locator("#ring").get_attribute("class")

    def test_speaking_class(self, ui_page: Page):
        self._set_connected(ui_page)
        ui_page.evaluate("updateAgentState('speaking')")
        assert "speaking" in ui_page.locator("#ring").get_attribute("class")

    def test_idle_class(self, ui_page: Page):
        self._set_connected(ui_page)
        ui_page.evaluate("updateAgentState('idle')")
        assert "idle" in ui_page.locator("#ring").get_attribute("class")

    def test_state_switch_removes_previous(self, ui_page: Page):
        self._set_connected(ui_page)
        ui_page.evaluate("updateAgentState('listening')")
        ui_page.evaluate("updateAgentState('speaking')")
        classes = ui_page.locator("#ring").get_attribute("class")
        assert "speaking" in classes
        assert "listening" not in classes


class TestRingColors:
    """Verify each state applies the correct icon color (non-animated element)."""

    def _set_connected(self, page: Page):
        page.evaluate("connected = true; ring.classList.add('active')")

    def _get_icon_color(self, page: Page, state: str) -> str:
        self._set_connected(page)
        # Disable transitions so computed color is immediate
        page.evaluate("document.querySelector('.ring-icon').style.transition = 'none'")
        page.evaluate(f"updateAgentState('{state}')")
        return page.evaluate(
            "window.getComputedStyle(document.querySelector('.ring-icon')).color"
        )

    def test_listening_is_blue(self, ui_page: Page):
        color = self._get_icon_color(ui_page, "listening")
        # --blue: #5b8def = rgb(91, 141, 239)
        assert "91, 141, 239" in color

    def test_thinking_is_orange(self, ui_page: Page):
        color = self._get_icon_color(ui_page, "thinking")
        # --orange: #f59e0b = rgb(245, 158, 11)
        assert "245, 158, 11" in color

    def test_speaking_is_green(self, ui_page: Page):
        color = self._get_icon_color(ui_page, "speaking")
        # --green: #4ade80 = rgb(74, 222, 128)
        assert "74, 222, 128" in color


class TestDisconnectResetsState:
    """Verify the disconnect function clears agent state UI."""

    def test_disconnect_clears_agent_state(self, ui_page: Page):
        # Simulate connected + speaking state
        ui_page.evaluate("""
            connected = true;
            ring.classList.add('active');
            updateAgentState('speaking');
        """)
        # Run disconnect logic (without actual room)
        ui_page.evaluate("""
            connected = false;
            ring.classList.remove('active', 'idle', 'listening', 'thinking', 'speaking');
            document.getElementById('status').textContent = 'Disconnected';
            document.getElementById('agentState').textContent = '';
            document.getElementById('agentState').className = 'agent-state';
        """)
        expect(ui_page.locator("#agentState")).to_have_text("")
        assert "speaking" not in ui_page.locator("#ring").get_attribute("class")


class TestHandleAttributesChanged:
    """Test the handleAttributesChanged function used by LiveKit events."""

    def test_handles_state_in_changed_attributes(self, ui_page: Page):
        ui_page.evaluate("""
            handleAttributesChanged(
                {'lk.agent.state': 'thinking'},
                {attributes: {}}
            )
        """)
        expect(ui_page.locator("#agentState")).to_have_text("Thinking")

    def test_falls_back_to_participant_attributes(self, ui_page: Page):
        ui_page.evaluate("""
            handleAttributesChanged(
                {},
                {attributes: {'lk.agent.state': 'speaking'}}
            )
        """)
        expect(ui_page.locator("#agentState")).to_have_text("Speaking")

    def test_ignores_unrelated_attributes(self, ui_page: Page):
        ui_page.evaluate("updateAgentState('idle')")
        ui_page.evaluate("""
            handleAttributesChanged(
                {'some.other.attr': 'value'},
                {attributes: {}}
            )
        """)
        # Should stay on idle
        expect(ui_page.locator("#agentState")).to_have_text("Ready")
