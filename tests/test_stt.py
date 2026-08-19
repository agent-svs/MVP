"""STT — gating, noise filtering, disfluency cleanup."""

from __future__ import annotations

import numpy as np
import pytest

from config import cfg
from core.stt import _NOISE_RE, STT, clean_disfluencies


# ── the short-answer regression ─────────────────────────────
def test_gates_are_configurable_and_permit_short_answers():
    """Hardcoded 0.05 amplitude / 0.8s duration gates discarded every reply to
    a clarification question ("yes", "3 pm", "Monday")."""
    assert cfg.stt.min_duration_s <= 0.3
    assert cfg.stt.min_amplitude <= 0.02


def test_short_loud_utterance_is_not_dropped(monkeypatch):
    stt = STT()
    captured = {}

    class FakeInfo:
        language, language_probability = "en", 0.99

    class FakeSeg:
        text = "yes"

    def fake_transcribe(audio, **kw):
        captured["called"] = True
        return [FakeSeg()], FakeInfo()

    monkeypatch.setattr(stt, "_model", type("M", (), {"transcribe": staticmethod(fake_transcribe)})())
    # 0.4s of clear speech — shorter than the old 0.8s floor.
    audio = (np.random.default_rng(0).standard_normal(int(0.4 * cfg.audio.sample_rate)) * 0.2).astype(np.float32)
    text, _ = stt.transcribe(audio)
    assert captured.get("called") is True
    assert text == "yes"


def test_silence_is_still_gated(monkeypatch):
    stt = STT()
    monkeypatch.setattr(stt, "_model", object())
    text, _ = stt.transcribe(np.zeros(cfg.audio.sample_rate, dtype=np.float32))
    assert text == ""


# ── the phone-number regression ─────────────────────────────
@pytest.mark.parametrize("text,is_noise", [
    ("555 123 4567", False),   # was discarded: the old pattern required letters
    ("9876543210", False),
    ("hello", False),
    ("....", True),
    ("!!! ???", True),
])
def test_noise_filter_keeps_digits(text, is_noise):
    assert bool(_NOISE_RE.match(text)) is is_noise


def test_empty_transcript_is_handled_by_the_caller_guard():
    """The pattern matches "" by design; transcribe() guards with `if text`."""
    assert _NOISE_RE.match("")


# ── disfluencies ────────────────────────────────────────────
def test_fillers_removed():
    assert "um" not in clean_disfluencies("um I would like uh an appointment")


def test_meaningful_repetition_preserved():
    """A blanket repetition collapse would turn 'no no no' into 'no'."""
    assert clean_disfluencies("no no no") == "no no no"


def test_stuttered_word_collapsed():
    assert clean_disfluencies("book book book an appointment") == "book an appointment"
