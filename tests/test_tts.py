"""TTS — cache policy and sentence splitting."""

from __future__ import annotations

import pytest

from core.tts import _MAX_CHARS, TTS, _is_cacheable


# ── the PII-on-disk regression ──────────────────────────────
@pytest.mark.parametrize("text", [
    "Done. You're booked for Monday at 3 PM, John Smith.",
    "Thank you, Priya Sharma. Your identity is verified.",
    "Your number is 555 123 4567.",
    "Welcome, Rahul Mehta!",
])
def test_caller_specific_phrases_are_never_cached(text):
    """Personalised responses were written to disk as audio, forever, for a
    near-zero hit rate."""
    assert _is_cacheable(text) is False


@pytest.mark.parametrize("text", [
    "Which day works best for you?",
    "One moment please.",
    "What time would you prefer?",
    "Hello! Welcome to Apex Medical Clinic. How can I help you today?",
])
def test_generic_phrases_are_cached(text):
    assert _is_cacheable(text) is True


def test_cache_can_be_disabled(monkeypatch):
    monkeypatch.setattr("config.cfg.tts.cache_enabled", False)
    assert _is_cacheable("One moment please.") is False


# ── sentence splitting ──────────────────────────────────────
def test_splits_on_sentence_boundaries():
    assert TTS._split_into_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_no_chunk_exceeds_the_xtts_limit():
    """XTTS is unstable past ~250 characters; a comma-less run used to pass
    straight through."""
    long_text = "word " * 200
    for chunk in TTS._split_into_sentences(long_text):
        assert len(chunk) <= _MAX_CHARS


def test_long_sentence_with_commas_splits_cleanly():
    text = ", ".join(["this is a reasonably long clause"] * 20)
    chunks = TTS._split_into_sentences(text)
    assert len(chunks) > 1
    assert all(len(c) <= _MAX_CHARS for c in chunks)


def test_empty_text_yields_nothing():
    assert TTS._split_into_sentences("") == []
    assert TTS._split_into_sentences("   ") == []
