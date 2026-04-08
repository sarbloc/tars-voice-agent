"""Tests for _preprocess_tts_text — ensures Kokoro TTS gets well-punctuated input."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tars_agent import _preprocess_tts_text


class TestNumberedLists:
    def test_simple_numbered_list(self):
        text = "1. Check the lights\n2. Lock the door\n3. Set the alarm"
        result = _preprocess_tts_text(text)
        assert "First, Check the lights." in result
        assert "Second, Lock the door." in result
        assert "Third, Set the alarm." in result

    def test_all_ordinals(self):
        lines = "\n".join(f"{i}. Item {i}" for i in range(1, 11))
        result = _preprocess_tts_text(lines)
        for ordinal in ["First", "Second", "Third", "Fourth", "Fifth",
                        "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"]:
            assert ordinal in result

    def test_number_beyond_ten_uses_digit(self):
        text = "11. Something else"
        result = _preprocess_tts_text(text)
        assert "11," in result
        assert "Something else" in result

    def test_numbered_with_leading_spaces(self):
        text = "  1. Indented item"
        result = _preprocess_tts_text(text)
        assert "First," in result


class TestBulletLists:
    def test_dash_bullets(self):
        text = "- Apples\n- Bananas\n- Cherries"
        result = _preprocess_tts_text(text)
        assert "Apples." in result
        assert "Bananas." in result
        assert "Cherries." in result

    def test_asterisk_bullets(self):
        text = "* One\n* Two\n* Three"
        result = _preprocess_tts_text(text)
        assert "One." in result
        assert "Two." in result

    def test_no_leading_period_on_first_bullet(self):
        result = _preprocess_tts_text("- First item")
        assert not result.startswith(".")


class TestMarkdownStripping:
    def test_bold(self):
        assert "important" in _preprocess_tts_text("This is **important**")
        assert "**" not in _preprocess_tts_text("This is **important**")

    def test_italic(self):
        assert "urgent" in _preprocess_tts_text("This is *urgent*")
        assert _preprocess_tts_text("*urgent*").count("*") == 0

    def test_bold_italic(self):
        result = _preprocess_tts_text("***critical***")
        assert "critical" in result
        assert "***" not in result

    def test_inline_code(self):
        result = _preprocess_tts_text("Run the `deploy` command")
        assert "deploy" in result
        assert "`" not in result

    def test_headings(self):
        for level in range(1, 7):
            hashes = "#" * level
            result = _preprocess_tts_text(f"{hashes} Title")
            assert "Title" in result
            assert "#" not in result

    def test_heading_gets_period(self):
        result = _preprocess_tts_text("## Settings\nHere are your options")
        assert "Settings." in result


class TestEmDashes:
    def test_em_dash_replaced(self):
        result = _preprocess_tts_text("The system — works")
        assert "—" not in result
        assert "," in result

    def test_en_dash_replaced(self):
        result = _preprocess_tts_text("The system – works")
        assert "–" not in result
        assert "," in result


class TestTransitionalWords:
    def test_however_gets_comma(self):
        result = _preprocess_tts_text("However the system is down.")
        assert "However," in result

    def test_also_gets_comma(self):
        result = _preprocess_tts_text("Also it crashed.")
        assert "Also," in result

    def test_already_has_comma_not_doubled(self):
        result = _preprocess_tts_text("However, the system is down.")
        assert "However," in result
        assert "However,," not in result

    def test_all_transitional_words(self):
        for word in ("Additionally", "Furthermore", "Moreover",
                     "Finally", "Meanwhile", "Otherwise"):
            result = _preprocess_tts_text(f"{word} something happened.")
            assert f"{word}," in result


class TestWhitespaceCollapse:
    def test_newlines_collapsed(self):
        result = _preprocess_tts_text("Line one.\n\n\nLine two.")
        assert "\n" not in result
        assert "Line one." in result and "Line two." in result

    def test_multiple_spaces_collapsed(self):
        result = _preprocess_tts_text("Too   many    spaces.")
        assert "  " not in result


class TestPunctuationCleanup:
    def test_double_periods_collapsed(self):
        result = _preprocess_tts_text("End of sentence.. Next.")
        assert ".." not in result

    def test_double_commas_collapsed(self):
        result = _preprocess_tts_text("One,, two")
        assert ",," not in result


class TestPassthrough:
    def test_plain_text_unchanged_content(self):
        text = "Everything is fine. No issues here."
        result = _preprocess_tts_text(text)
        assert "Everything is fine." in result
        assert "No issues here." in result

    def test_empty_string(self):
        assert _preprocess_tts_text("") == ""

    def test_single_word(self):
        result = _preprocess_tts_text("Hello")
        assert "Hello" in result


class TestMixedContent:
    def test_numbered_list_with_markdown(self):
        text = "1. **Turn on** the lights\n2. Check the *thermostat*"
        result = _preprocess_tts_text(text)
        assert "First," in result
        assert "Turn on" in result
        assert "**" not in result
        assert "thermostat" in result
        assert "*" not in result

    def test_bullets_with_transitionals(self):
        text = "- Also check doors\n- However close windows"
        result = _preprocess_tts_text(text)
        assert "Also," in result
        assert "However," in result

    def test_full_mixed_scenario(self):
        text = (
            "1. **Turn on** the lights\n"
            "2. Check the *thermostat*\n"
            "- Also check doors\n"
            "\n"
            "However the garage — which is open — needs closing."
        )
        result = _preprocess_tts_text(text)
        assert "First," in result
        assert "**" not in result
        assert "*" not in result
        assert "Also," in result
        assert "However," in result
        assert "—" not in result
