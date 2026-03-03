"""
Tests for the STT (Faster Whisper) module.

Tests model loading, transcription, and disfluency filtering.
"""

import numpy as np
import pytest


def test_disfluency_filter_removes_fillers():
    """Disfluency filter should remove um, uh, like, etc."""
    from core.stt import clean_disfluencies

    assert clean_disfluencies("um I want to book") == "I want to book"
    assert clean_disfluencies("uh can you help me") == "can you help me"
    assert clean_disfluencies("I uhh need help") == "I need help"


def test_disfluency_filter_removes_hedges():
    """Filter should remove hedge phrases like 'you know', 'I mean'."""
    from core.stt import clean_disfluencies

    assert clean_disfluencies("you know I want to book") == "I want to book"
    assert clean_disfluencies("I basically want to cancel") == "I want to cancel"


def test_disfluency_filter_removes_repetitions():
    """Filter should remove word repetitions (false starts)."""
    from core.stt import clean_disfluencies

    assert clean_disfluencies("I I want to book") == "I want to book"
    assert clean_disfluencies("can can you help") == "can you help"


def test_disfluency_filter_preserves_clean_text():
    """Filter should not alter already-clean text."""
    from core.stt import clean_disfluencies

    clean = "I want to book an appointment for tomorrow at 3 PM"
    assert clean_disfluencies(clean) == clean


def test_disfluency_filter_handles_empty():
    """Filter should handle empty string gracefully."""
    from core.stt import clean_disfluencies

    assert clean_disfluencies("") == ""
    assert clean_disfluencies("   ") == ""


def test_stt_init_defaults():
    """STT should initialize with config defaults."""
    from core.stt import STT

    stt = STT()
    assert stt.model_size == "base"
    assert stt.language == "en"
    assert stt.device == "cpu"  # MPS falls back to CPU for CTranslate2
    assert stt.is_loaded is False


def test_stt_init_custom():
    """STT should accept custom model size and settings."""
    from core.stt import STT

    stt = STT(model_size="tiny", language="fr", beam_size=3)
    assert stt.model_size == "tiny"
    assert stt.language == "fr"
    assert stt.beam_size == 3


def test_stt_mps_fallback():
    """STT should fall back to CPU when device is MPS."""
    from core.stt import STT

    stt = STT(device="mps")
    assert stt.device == "cpu"
    assert stt.compute_type == "int8"


def test_stt_loads_model():
    """STT should load the Whisper model successfully."""
    from core.stt import STT

    stt = STT(model_size="tiny")  # Tiny for fast test
    stt.load()
    assert stt.is_loaded is True


def test_stt_transcribes_silence():
    """STT should return empty or near-empty text for silence."""
    from core.stt import STT

    stt = STT(model_size="tiny")
    stt.load()

    # Generate 1 second of silence
    silence = np.zeros(16000, dtype=np.float32)
    text, elapsed = stt.transcribe(silence)
    # Silence should produce empty or very short text
    assert len(text) < 20  # Might produce artifacts like "[silence]" etc.


def test_stt_transcribes_tone():
    """STT should handle a simple tone (non-speech) without crashing."""
    from core.stt import STT

    stt = STT(model_size="tiny")
    stt.load()

    # Generate a 440Hz tone (1 second)
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    text, elapsed = stt.transcribe(tone)
    # Should not crash, text content is unpredictable for non-speech
    assert isinstance(text, str)
    assert elapsed > 0
