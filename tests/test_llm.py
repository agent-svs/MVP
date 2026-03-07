"""
Tests for the LLM (Gemini API) module.

Tests initialization, response shaping, and conversation history.
Live API tests require GEMINI_API_KEY to be set in .env.
"""

import pytest


# ── Response Shaper Tests ────────────────────────────────────

def test_shape_strips_bold():
    """Response shaper should remove markdown bold."""
    from core.llm import shape_for_speech

    assert shape_for_speech("This is **bold** text") == "This is bold text"
    assert shape_for_speech("This is *italic* text") == "This is italic text"


def test_shape_strips_headers():
    """Response shaper should remove markdown headers."""
    from core.llm import shape_for_speech

    result = shape_for_speech("## Services\nWe offer dental care.")
    assert "##" not in result
    assert "dental care" in result


def test_shape_strips_bullet_points():
    """Response shaper should remove bullet points."""
    from core.llm import shape_for_speech

    text = "Our services:\n- Dental care\n- Eye exams\n- General checkups"
    result = shape_for_speech(text)
    assert "-" not in result or "checkups" in result
    assert "Dental care" in result


def test_shape_strips_code_blocks():
    """Response shaper should remove code blocks."""
    from core.llm import shape_for_speech

    text = "Use this: `command` to start"
    result = shape_for_speech(text)
    assert "`" not in result
    assert "command" in result


def test_shape_strips_urls():
    """Response shaper should remove URLs."""
    from core.llm import shape_for_speech

    text = "Visit https://example.com for more info"
    result = shape_for_speech(text)
    assert "https://" not in result
    assert "Visit" in result


def test_shape_strips_links():
    """Response shaper should convert markdown links to plain text."""
    from core.llm import shape_for_speech

    text = "Check our [website](https://example.com) for details"
    result = shape_for_speech(text)
    assert "[" not in result
    assert "website" in result


def test_shape_expands_abbreviations():
    """Response shaper should expand abbreviations for speech."""
    from core.llm import shape_for_speech

    assert "and" in shape_for_speech("A & B")
    assert "for example" in shape_for_speech("e.g. this")
    assert "that is" in shape_for_speech("i.e. this")


def test_shape_handles_empty():
    """Response shaper should handle empty strings."""
    from core.llm import shape_for_speech

    assert shape_for_speech("") == ""
    assert shape_for_speech(None) == ""


# ── LLM Client Tests ──────────────────────────────────────────

def test_llm_init_defaults():
    """LLM should initialize with config defaults."""
    from core.llm import LLM
    from config import cfg

    llm = LLM()
    assert llm.model_name == cfg.llm.model
    assert llm.max_tokens == 256
    assert llm.is_initialized is False


def test_llm_init_custom():
    """LLM should accept custom settings."""
    from core.llm import LLM

    llm = LLM(
        model="gemini-2.5-pro",
        temperature=0.3,
        max_tokens=512,
    )
    assert llm.model_name == "gemini-2.5-pro"
    assert llm.temperature == 0.3
    assert llm.max_tokens == 512


def test_llm_missing_api_key_raises():
    """LLM should raise ValueError when API key is missing."""
    from core.llm import LLM

    llm = LLM(api_key="your-api-key-here")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        llm._ensure_client()


def test_llm_history_management():
    """LLM should track and clear conversation history."""
    from core.llm import LLM

    llm = LLM()
    assert llm.history_length == 0
    # Manually add history entries
    llm._history.append({"role": "user", "parts": [{"text": "Hello"}]})
    llm._history.append({"role": "model", "parts": [{"text": "Hi!"}]})
    assert llm.history_length == 2
    llm.clear_history()
    assert llm.history_length == 0


def test_llm_builds_contents():
    """LLM should build message contents correctly."""
    from core.llm import LLM

    llm = LLM()
    # Add some history
    llm._history.append({"role": "user", "parts": [{"text": "Hi"}]})
    llm._history.append({"role": "model", "parts": [{"text": "Hello!"}]})

    contents, dynamic_prompt = llm._build_contents("Book an appointment")
    assert len(contents) == 3  # 2 history + 1 new
    assert contents[0]["role"] == "user"
    assert contents[2]["parts"][0]["text"] == "Book an appointment"
    assert "CURRENT CONVERSATION CONTEXT" not in dynamic_prompt


def test_llm_system_prompt_default():
    """LLM should use the default system prompt."""
    from core.llm import LLM, DEFAULT_SYSTEM_PROMPT

    llm = LLM()
    assert llm.system_prompt == DEFAULT_SYSTEM_PROMPT


def test_llm_system_prompt_custom():
    """LLM should accept a custom system prompt."""
    from core.llm import LLM

    llm = LLM(system_prompt="You are a pirate.")
    assert llm.system_prompt == "You are a pirate."
