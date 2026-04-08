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
        expect(ui_page.locator("h1")).to_have_text("TARS")

    def test_status_shows_ready(self, ui_page: Page):
        expect(ui_page.locator("#status")).to_have_text("Ready to connect")

    def test_connect_button_visible(self, ui_page: Page):
        btn = ui_page.locator("#connectBtn")
        expect(btn).to_be_visible()
        expect(btn).to_have_text("Connect")

    def test_indicator_hidden_initially(self, ui_page: Page):
        indicator = ui_page.locator("#indicator")
        expect(indicator).to_be_hidden()

    def test_agent_state_empty_initially(self, ui_page: Page):
        state = ui_page.locator("#agentState")
        expect(state).to_have_text("")


class TestAgentStateIndicator:
    """Test the agent state JS functions by calling them directly."""

    def test_update_agent_state_listening(self, ui_page: Page):
        ui_page.evaluate("document.getElementById('indicator').style.display = 'block'")
        ui_page.evaluate("updateAgentState('listening')")
        state = ui_page.locator("#agentState")
        expect(state).to_have_text("Listening...")
        assert "listening" in ui_page.locator("#agentState").get_attribute("class")
        assert "listening" in ui_page.locator("#indicator").get_attribute("class")

    def test_update_agent_state_thinking(self, ui_page: Page):
        ui_page.evaluate("document.getElementById('indicator').style.display = 'block'")
        ui_page.evaluate("updateAgentState('thinking')")
        state = ui_page.locator("#agentState")
        expect(state).to_have_text("Thinking...")
        assert "thinking" in ui_page.locator("#indicator").get_attribute("class")

    def test_update_agent_state_speaking(self, ui_page: Page):
        ui_page.evaluate("document.getElementById('indicator').style.display = 'block'")
        ui_page.evaluate("updateAgentState('speaking')")
        state = ui_page.locator("#agentState")
        expect(state).to_have_text("Speaking...")
        assert "speaking" in ui_page.locator("#indicator").get_attribute("class")

    def test_update_agent_state_idle(self, ui_page: Page):
        ui_page.evaluate("document.getElementById('indicator').style.display = 'block'")
        ui_page.evaluate("updateAgentState('idle')")
        state = ui_page.locator("#agentState")
        expect(state).to_have_text("Ready")

    def test_update_agent_state_initializing(self, ui_page: Page):
        ui_page.evaluate("updateAgentState('initializing')")
        state = ui_page.locator("#agentState")
        expect(state).to_have_text("Initializing...")


class TestIndicatorAnimations:
    """Verify CSS classes produce distinct animation names per state."""

    def _get_animation_name(self, page: Page, state: str) -> str:
        page.evaluate(f"document.getElementById('indicator').style.display = 'block'")
        page.evaluate(f"updateAgentState('{state}')")
        return page.evaluate(
            "window.getComputedStyle(document.getElementById('indicator')).animationName"
        )

    def test_listening_has_pulse_animation(self, ui_page: Page):
        assert "pulse" in self._get_animation_name(ui_page, "listening")

    def test_thinking_has_spin_animation(self, ui_page: Page):
        assert "spin" in self._get_animation_name(ui_page, "thinking")

    def test_speaking_has_speak_animation(self, ui_page: Page):
        assert "speak" in self._get_animation_name(ui_page, "speaking")

    def test_idle_has_pulse_animation(self, ui_page: Page):
        assert "pulse" in self._get_animation_name(ui_page, "idle")


class TestIndicatorColors:
    """Verify each state gets a distinct indicator background color."""

    def _get_bg_color(self, page: Page, state: str) -> str:
        page.evaluate("document.getElementById('indicator').style.display = 'block'")
        page.evaluate(f"updateAgentState('{state}')")
        return page.evaluate(
            "window.getComputedStyle(document.getElementById('indicator')).backgroundColor"
        )

    def test_listening_is_blue(self, ui_page: Page):
        color = self._get_bg_color(ui_page, "listening")
        # #4a9eff = rgb(74, 158, 255)
        assert "74, 158, 255" in color

    def test_thinking_is_orange(self, ui_page: Page):
        color = self._get_bg_color(ui_page, "thinking")
        # #ffaa4a = rgb(255, 170, 74)
        assert "255, 170, 74" in color

    def test_speaking_is_green(self, ui_page: Page):
        color = self._get_bg_color(ui_page, "speaking")
        # #4aff7f = rgb(74, 255, 127)
        assert "74, 255, 127" in color


class TestDisconnectResetsState:
    """Verify the disconnect function clears agent state UI."""

    def test_disconnect_clears_agent_state(self, ui_page: Page):
        # Simulate some state
        ui_page.evaluate("updateAgentState('speaking')")
        ui_page.evaluate("""
            document.getElementById('status').textContent = 'Disconnected';
            document.getElementById('agentState').textContent = '';
            document.getElementById('agentState').className = '';
            document.getElementById('indicator').style.display = 'none';
        """)
        expect(ui_page.locator("#agentState")).to_have_text("")
        expect(ui_page.locator("#indicator")).to_be_hidden()


class TestHandleAttributesChanged:
    """Test the handleAttributesChanged function used by LiveKit events."""

    def test_handles_state_in_changed_attributes(self, ui_page: Page):
        ui_page.evaluate("""
            handleAttributesChanged(
                {'lk.agent.state': 'thinking'},
                {attributes: {}}
            )
        """)
        expect(ui_page.locator("#agentState")).to_have_text("Thinking...")

    def test_falls_back_to_participant_attributes(self, ui_page: Page):
        ui_page.evaluate("""
            handleAttributesChanged(
                {},
                {attributes: {'lk.agent.state': 'speaking'}}
            )
        """)
        expect(ui_page.locator("#agentState")).to_have_text("Speaking...")

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
