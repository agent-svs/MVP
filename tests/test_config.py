"""
Tests for the Config module.

Verifies .env loading, device detection, validation, and defaults.
"""

import os
import pytest
from unittest.mock import patch


def test_config_loads_from_env():
    """Config should load values from environment variables."""
    from config import Config

    with patch.dict(os.environ, {
        "DEVICE": "cpu",
        "COMPUTE_TYPE": "float32",
        "STT_MODEL_SIZE": "small",
        "STT_LANGUAGE": "en",
        "STT_BEAM_SIZE": "3",
        "TTS_LANGUAGE": "en",
        "GEMINI_API_KEY": "test-key-123",
        "GEMINI_MODEL": "gemini-2.0-flash",
        "GEMINI_MAX_TOKENS": "512",
        "GEMINI_TEMPERATURE": "0.5",
        "AUDIO_SAMPLE_RATE": "16000",
        "VAD_THRESHOLD": "0.6",
        "LOG_LEVEL": "DEBUG",
    }):
        config = Config.from_env()

    assert config.device == "cpu"
    assert config.compute_type == "float32"
    assert config.stt.model_size == "small"
    assert config.stt.beam_size == 3
    assert config.llm.api_key == "test-key-123"
    assert config.llm.max_tokens == 512
    assert config.llm.temperature == 0.5
    assert config.vad.threshold == 0.6
    assert config.log_level == "DEBUG"


def test_config_defaults():
    """Config should have sensible defaults when env vars are missing."""
    from config import Config

    with patch.dict(os.environ, {}, clear=True):
        config = Config.from_env()

    assert config.stt.model_size == "base"
    assert config.stt.language == "en"
    assert config.llm.model == "gemini-2.5-flash"
    assert config.llm.max_tokens == 256
    assert config.audio.sample_rate == 16000
    assert config.vad.min_speech_ms == 250


def test_config_device_auto_detection():
    """Device='auto' should resolve to cpu, mps, or cuda."""
    from config import Config

    with patch.dict(os.environ, {"DEVICE": "auto"}):
        config = Config.from_env()

    # On most CI/test machines, should be cpu or mps (Mac)
    assert config.device in ("cpu", "mps", "cuda")


def test_config_validate_missing_api_key():
    """Validation should warn when GEMINI_API_KEY is not set."""
    from config import Config

    with patch.dict(os.environ, {"GEMINI_API_KEY": "your-api-key-here"}):
        config = Config.from_env()

    issues = config.validate()
    assert any("GEMINI_API_KEY" in issue for issue in issues)


def test_config_validate_invalid_device():
    """Validation should catch invalid device and fall back to cpu."""
    from config import Config

    with patch.dict(os.environ, {"DEVICE": "tpu"}):
        config = Config.from_env()

    issues = config.validate()
    assert any("Unknown device" in issue for issue in issues)
    assert config.device == "cpu"  # Should have been corrected


def test_config_audio_chunk_samples():
    """chunk_samples should be computed correctly from sample rate and duration."""
    from config import Config

    with patch.dict(os.environ, {
        "AUDIO_SAMPLE_RATE": "16000",
        "AUDIO_CHUNK_DURATION_MS": "30",
    }):
        config = Config.from_env()

    # 16000 * 30 / 1000 = 480
    assert config.audio.chunk_samples == 480


def test_config_summary_output():
    """Config summary should return a non-empty formatted string."""
    from config import Config

    with patch.dict(os.environ, {"DEVICE": "cpu"}):
        config = Config.from_env()

    summary = config.summary()
    assert "Device:" in summary
    assert "STT Model:" in summary
    assert "faster-whisper" in summary
