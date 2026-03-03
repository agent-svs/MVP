"""
Tests for the TTS (Coqui XTTS v2) module.

Tests initialization, caching, and synthesis.
Note: Full model tests require the XTTS v2 model (~1.8GB download on first run).
"""

import numpy as np
import os
import pytest
import tempfile
from unittest.mock import patch


def test_tts_init_defaults():
    """TTS should initialize with config defaults."""
    from core.tts import TTS

    tts = TTS()
    assert tts.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"
    assert tts.language == "en"
    assert tts.is_loaded is False


def test_tts_init_custom():
    """TTS should accept custom settings."""
    from core.tts import TTS

    tts = TTS(language="es", device="cpu")
    assert tts.language == "es"
    assert tts.device == "cpu"


def test_tts_cache_key_deterministic():
    """Same text should produce same cache key."""
    from core.tts import TTS

    tts = TTS()
    key1 = tts._cache_key("Hello world")
    key2 = tts._cache_key("Hello world")
    assert key1 == key2


def test_tts_cache_key_varies_with_text():
    """Different text should produce different cache keys."""
    from core.tts import TTS

    tts = TTS()
    key1 = tts._cache_key("Hello")
    key2 = tts._cache_key("Goodbye")
    assert key1 != key2


def test_tts_cache_save_and_load():
    """Cache should save and retrieve audio correctly."""
    from core.tts import TTS

    with tempfile.TemporaryDirectory() as tmpdir:
        tts = TTS()
        tts._cache_dir = type(tts._cache_dir)(tmpdir)

        # Generate fake audio
        fake_audio = np.random.randn(16000).astype(np.float32)
        tts._save_cache("test phrase", fake_audio)

        # Retrieve from cache
        cached = tts._check_cache("test phrase")
        assert cached is not None
        np.testing.assert_array_almost_equal(cached, fake_audio)


def test_tts_cache_miss():
    """Cache should return None for uncached text."""
    from core.tts import TTS

    with tempfile.TemporaryDirectory() as tmpdir:
        tts = TTS()
        tts._cache_dir = type(tts._cache_dir)(tmpdir)

        assert tts._check_cache("never seen before") is None


def test_tts_synthesize_empty_text():
    """Synthesize should handle empty text gracefully."""
    from core.tts import TTS

    tts = TTS()
    audio, sr, elapsed = tts.synthesize("")
    assert len(audio) == 0
    assert sr == 24000

    audio, sr, elapsed = tts.synthesize("   ")
    assert len(audio) == 0


def test_tts_clear_cache():
    """Clear cache should remove all cached files."""
    from core.tts import TTS

    with tempfile.TemporaryDirectory() as tmpdir:
        tts = TTS()
        tts._cache_dir = type(tts._cache_dir)(tmpdir)

        # Save some fake cached files
        for i in range(3):
            fake_audio = np.random.randn(1000).astype(np.float32)
            tts._save_cache(f"phrase {i}", fake_audio)

        count = tts.clear_cache()
        assert count == 3
        assert tts._check_cache("phrase 0") is None
