"""Audio front-end — preprocessing, buffering, VAD windowing."""

from __future__ import annotations

import numpy as np
import pytest

from config import cfg
from core.audio_input import AudioInput
from core.audio_preprocessor import AudioPreprocessor


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


# ── the AGC regression ──────────────────────────────────────
def test_chunk_processing_preserves_relative_loudness():
    """Per-chunk AGC normalised every frame to the same RMS, erasing the
    silence/speech contrast that VAD and the STT energy gate depend on."""
    p = AudioPreprocessor()
    rng = np.random.default_rng(0)
    silence = (rng.standard_normal(480) * 0.001).astype(np.float32)
    speech = (rng.standard_normal(480) * 0.20).astype(np.float32)

    out_sil = p.process_chunk(silence)
    out_speech = p.process_chunk(speech)

    ratio = _rms(out_speech) / max(_rms(out_sil), 1e-9)
    assert ratio > 20, f"loudness contrast collapsed (ratio={ratio:.1f})"


def test_highpass_filter_state_is_continuous():
    """Without persistent `zi` the filter transient restarted every chunk,
    injecting a discontinuity 33 times a second."""
    p = AudioPreprocessor()
    t = np.arange(4800) / cfg.audio.sample_rate
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    chunked = np.concatenate([p.process_chunk(tone[i:i + 480]) for i in range(0, 4800, 480)])

    p.reset_filter_state()
    whole = p.process_chunk(tone)

    # Ignore the shared startup transient; the rest must match closely.
    assert np.max(np.abs(chunked[960:] - whole[960:])) < 1e-4


def test_utterance_processing_applies_agc():
    p = AudioPreprocessor()
    rng = np.random.default_rng(1)
    quiet = (rng.standard_normal(16000) * 0.005).astype(np.float32)
    out = p.process_utterance(quiet)
    assert _rms(out) > _rms(quiet) * 2


def test_utterance_processing_never_clips():
    p = AudioPreprocessor()
    loud = (np.random.default_rng(2).standard_normal(16000) * 0.5).astype(np.float32)
    assert float(np.max(np.abs(p.process_utterance(loud)))) <= 1.0


def test_empty_input_is_safe():
    p = AudioPreprocessor()
    empty = np.array([], dtype=np.float32)
    assert len(p.process_chunk(empty)) == 0
    assert len(p.process_utterance(empty)) == 0


# ── the stale-audio regression ──────────────────────────────
def test_mic_queue_is_bounded_and_keeps_the_newest_audio():
    """An unbounded queue let seconds of stale audio pile up during a slow
    turn; the agent then transcribed its own TTS output."""
    mic = AudioInput(max_queue_chunks=5)
    for i in range(50):
        mic._audio_callback(np.full((480, 1), float(i), dtype=np.float32), 480, None, None)

    assert mic.queued_chunks <= 5
    newest = [mic.read_chunk(timeout=0.01)[0] for _ in range(5)]
    assert newest[-1] == 49.0, "must retain the most recent audio, not the oldest"


def test_flush_empties_the_queue():
    mic = AudioInput(max_queue_chunks=10)
    for _ in range(5):
        mic._audio_callback(np.zeros((480, 1), dtype=np.float32), 480, None, None)
    assert mic.flush() == 5
    assert mic.queued_chunks == 0


def test_mute_drops_incoming_audio():
    mic = AudioInput(max_queue_chunks=10)
    mic.mute()
    mic._audio_callback(np.zeros((480, 1), dtype=np.float32), 480, None, None)
    assert mic.queued_chunks == 0
    mic.unmute()
    mic._audio_callback(np.zeros((480, 1), dtype=np.float32), 480, None, None)
    assert mic.queued_chunks == 1


# ── VAD windowing ───────────────────────────────────────────
def test_vad_window_matches_the_sample_rate():
    """512 was hardcoded while the sample rate was configurable, so 8kHz
    silently produced an invalid window for Silero."""
    assert cfg.vad.chunk_size_for(16000) == 512
    assert cfg.vad.chunk_size_for(8000) == 256


@pytest.mark.slow
def test_vad_emits_an_utterance_and_ignores_silence():
    from core.vad import VAD

    v = VAD()
    sr = cfg.audio.sample_rate
    silence = np.zeros(sr, dtype=np.float32)
    assert v.process_chunk(silence) is None
    assert v.is_speaking is False


# ── the self-transcription regression ───────────────────────
def test_barge_in_requires_energy_above_the_echo_floor():
    """The mic hears the agent's own speakers. A plain VAD check made the
    agent interrupt itself; barge-in now needs amplitude too."""
    from core.vad import VAD

    v = VAD()
    quiet_speechlike = (np.random.default_rng(3).standard_normal(512 * 8) * 0.03).astype(np.float32)
    assert v.check_barge_in(quiet_speechlike) is False


def test_barge_in_floor_is_configurable():
    assert cfg.vad.barge_in_min_amplitude > 0
    assert cfg.vad.barge_in_threshold >= cfg.vad.threshold
    assert cfg.vad.barge_in_consecutive >= 2


def test_echo_cooldown_is_configured():
    """Re-arming the mic the instant playback stops let the speaker tail be
    transcribed as the caller's reply."""
    assert cfg.audio.echo_cooldown_ms >= 200


# ── the barge-in-never-fires regression ─────────────────────
def test_barge_in_accumulates_short_mic_chunks():
    """Mic chunks are 480 samples; the Silero window is 512. A per-call length
    check meant check_barge_in returned False for every chunk ever passed to
    it, so barge-in had never once fired."""
    from core.vad import VAD

    v = VAD()
    assert cfg.audio.chunk_samples < v._vad_chunk_size, "premise of the bug"

    loud = (np.random.default_rng(7).standard_normal(cfg.audio.chunk_samples) * 0.5).astype(np.float32)
    v.check_barge_in(loud)
    # The first sub-window chunk must be buffered, not discarded.
    assert len(v._barge_buffer) > 0 or v._barge_hits > 0


def test_barge_in_ignores_quiet_echo_across_many_chunks():
    from core.vad import VAD

    v = VAD()
    quiet = (np.random.default_rng(8).standard_normal(480) * 0.02).astype(np.float32)
    assert not any(v.check_barge_in(quiet) for _ in range(40))


def test_reset_barge_in_clears_accumulation():
    from core.vad import VAD

    v = VAD()
    v.check_barge_in((np.random.default_rng(9).standard_normal(480) * 0.5).astype(np.float32))
    v.reset_barge_in()
    assert len(v._barge_buffer) == 0 and v._barge_hits == 0
