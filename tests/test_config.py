"""Config — env loading, device policy, validation."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from config import Config


def _fresh(**env) -> Config:
    with patch.dict(os.environ, env, clear=False):
        return Config.from_env()


def test_loads_values_from_env():
    c = _fresh(
        DEVICE="cpu", COMPUTE_TYPE="float32", STT_MODEL_SIZE="small",
        STT_BEAM_SIZE="3", GEMINI_API_KEY="test-key-123",
        GEMINI_MAX_TOKENS="512", GEMINI_TEMPERATURE="0.5",
        VAD_THRESHOLD="0.6", LOG_LEVEL="DEBUG",
    )
    assert c.device == "cpu"
    assert c.stt.model_size == "small" and c.stt.beam_size == 3
    assert c.llm.api_key == "test-key-123" and c.llm.max_tokens == 512
    assert c.llm.temperature == 0.5 and c.vad.threshold == 0.6
    assert c.log_level == "DEBUG"


def test_defaults_are_sane():
    c = Config()
    assert c.audio.sample_rate == 16000
    assert c.audio.chunk_samples == 480
    assert c.vad.min_speech_ms <= 300, "must not discard short answers"
    assert c.stt.min_duration_s <= 0.3
    assert c.llm.timeout_s > 0


# ── the settings that used to bypass Config ─────────────────
def test_intent_settings_are_part_of_config():
    """INTENT_FINE_TUNED / INTENT_MODEL were read via raw os.getenv inside
    intent.py and were absent from config.py entirely."""
    c = _fresh(INTENT_FINE_TUNED="true", INTENT_MODEL="my/model")
    assert c.nlu.intent_fine_tuned is True
    assert c.nlu.intent_model == "my/model"


def test_business_rules_come_from_config():
    c = _fresh(BUSINESS_HOURS_START="9", BUSINESS_HOURS_END="17")
    assert c.business.hours_start == 9 and c.business.hours_end == 17
    assert "9:00 AM" in c.business.hours_human


# ── device policy ───────────────────────────────────────────
@pytest.mark.parametrize("device,hf,ct", [
    ("cpu", -1, "cpu"),
    ("mps", "mps", "cpu"),   # CTranslate2 has no MPS support
    ("cuda", 0, "cuda"),
])
def test_one_device_policy_for_every_backend(device, hf, ct):
    c = _fresh(DEVICE=device)
    assert c.hf_pipeline_device == hf
    assert c.stt_device[0] == ct


# ── validation ──────────────────────────────────────────────
def test_missing_api_key_is_flagged():
    c = _fresh(GEMINI_API_KEY="your-api-key-here")
    assert any("GEMINI_API_KEY" in i for i in c.validate())


def test_invalid_device_falls_back_to_cpu():
    c = _fresh(DEVICE="tpu")
    assert any("Unknown device" in i for i in c.validate())
    assert c.device == "cpu"


def test_vad_min_speech_that_would_eat_short_answers_is_flagged():
    """The live .env had VAD_MIN_SPEECH_MS=1000, which silently discarded
    every reply to a clarification question."""
    c = _fresh(VAD_MIN_SPEECH_MS="1000")
    assert any("VAD_MIN_SPEECH_MS" in i for i in c.validate())


def test_unsupported_sample_rate_is_corrected():
    c = _fresh(AUDIO_SAMPLE_RATE="44100")
    assert any("AUDIO_SAMPLE_RATE" in i for i in c.validate())
    assert c.audio.sample_rate == 16000


def test_out_of_range_vad_threshold_is_corrected():
    c = _fresh(VAD_THRESHOLD="5.0")
    c.validate()
    assert c.vad.threshold == 0.5


def test_inverted_business_hours_are_corrected():
    c = _fresh(BUSINESS_HOURS_START="18", BUSINESS_HOURS_END="8")
    c.validate()
    assert c.business.hours_start == 8 and c.business.hours_end == 18


def test_malformed_numbers_fall_back_to_defaults():
    c = _fresh(GEMINI_MAX_TOKENS="not-a-number", VAD_THRESHOLD="abc")
    assert c.llm.max_tokens == 256 and c.vad.threshold == 0.5


def test_summary_is_renderable():
    s = Config().summary()
    for expected in ("Device:", "STT Model:", "Hours:", "faster-whisper"):
        assert expected in s


def test_coqui_licence_is_pre_accepted():
    """Coqui prompts on stdin for its CPML licence, which hangs a voice loop."""
    assert os.environ.get("COQUI_TOS_AGREED") == "1"
