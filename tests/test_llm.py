"""LLM client — must never raise, must retry the right errors."""

from __future__ import annotations

import pytest

from core.llm import LLM, LLMResponse, _is_retryable, build_system_prompt, shape_for_speech


# ── retry classification ────────────────────────────────────
@pytest.mark.parametrize("msg", [
    "429 RESOURCE_EXHAUSTED", "503 UNAVAILABLE high demand",
    "500 internal error", "deadline exceeded", "connection reset",
])
def test_transient_errors_are_retryable(msg):
    """Only the literal string '429' was matched before, so a routine 503
    propagated and killed the process."""
    assert _is_retryable(Exception(msg)) is True


@pytest.mark.parametrize("msg", ["400 INVALID_ARGUMENT", "401 unauthorized", "404 not found"])
def test_client_errors_are_not_retryable(msg):
    assert _is_retryable(Exception(msg)) is False


# ── no-raise contract ───────────────────────────────────────
def test_missing_api_key_returns_a_fallback_instead_of_raising():
    r = LLM(api_key="").generate("hello")
    assert isinstance(r, LLMResponse)
    assert r.ok is False and r.error == "missing_api_key" and r.text


def test_api_failure_degrades_gracefully(monkeypatch):
    llm = LLM(api_key="test-key")

    class Boom:
        class models:
            @staticmethod
            def generate_content(**_):
                raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr(llm, "_ensure_client", lambda: None)
    monkeypatch.setattr(llm, "_client", Boom(), raising=False)
    monkeypatch.setattr("config.cfg.llm.max_retries", 1)

    r = llm.generate("hello")
    assert r.ok is False and r.text and "503" in (r.error or "")


def test_empty_model_response_degrades(monkeypatch):
    llm = LLM(api_key="test-key")

    class EmptyResp:
        candidates = []

        @property
        def text(self):
            raise ValueError("no parts")

    class Client:
        class models:
            @staticmethod
            def generate_content(**_):
                return EmptyResp()

    monkeypatch.setattr(llm, "_ensure_client", lambda: None)
    monkeypatch.setattr(llm, "_client", Client(), raising=False)
    monkeypatch.setattr(llm, "_build_config", lambda p: None)

    r = llm.generate("hello")
    assert r.ok is False and r.text


def test_result_unpacks_as_a_tuple_for_legacy_callers():
    text, elapsed = LLM(api_key="").generate("hi")
    assert isinstance(text, str) and isinstance(elapsed, float)


# ── prompt + shaping ────────────────────────────────────────
def test_system_prompt_is_grounded_and_config_driven():
    from config import cfg

    prompt = build_system_prompt()
    assert cfg.business.name in prompt
    assert cfg.business.hours_human in prompt
    assert "Dr. Jones" in prompt
    assert "GROUNDING CONTRACT" in prompt
    assert "911" in prompt


def test_shape_for_speech_strips_markdown():
    out = shape_for_speech("**Bold** and `code` and [link](http://x.com)\n\n- bullet")
    for junk in ("*", "`", "[", "](", "http"):
        assert junk not in out


def test_history_is_trimmed():
    llm = LLM(api_key="test-key")
    for i in range(40):
        llm._history.append({"role": "user", "parts": [{"text": str(i)}]})
    llm._trim_history()
    assert llm.history_length <= 12
