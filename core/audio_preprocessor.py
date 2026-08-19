"""
Audio Preprocessor — signal conditioning.

Split into two stages, deliberately:

  process_chunk(chunk)        — runs on every 30ms frame, feeding VAD.
                                High-pass ONLY, with persistent filter state.
  process_utterance(audio)    — runs once on a complete utterance, feeding STT.
                                Noise reduction + AGC.

Why the split: the previous version ran spectral-gating noise reduction and
AGC on every 30ms chunk. Three problems, all fixed here.

  1. `sosfilt` was called without carrying filter state, so the Butterworth
     transient restarted every chunk — a discontinuity 33 times a second.
  2. `noisereduce` was handed 480 samples while its STFT window is 2048; the
     output was meaningless and the cost was paid in the hot loop.
  3. AGC normalised every chunk independently to a fixed RMS. That erases the
     loudness difference between silence and speech, which is precisely the
     signal Silero VAD and the STT energy gate rely on.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from config import cfg

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Real-time audio preprocessor for voice input."""

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        agc_target_rms: Optional[float] = None,
        noise_reduce_enabled: Optional[bool] = None,
    ) -> None:
        pc = cfg.preprocess
        self.sample_rate = sample_rate if sample_rate is not None else cfg.audio.sample_rate
        self.agc_target_rms = (
            agc_target_rms if agc_target_rms is not None else pc.agc_target_rms
        )
        self.agc_enabled = pc.agc_enabled
        self.agc_max_gain = pc.agc_max_gain
        self.noise_reduce_enabled = (
            noise_reduce_enabled if noise_reduce_enabled is not None else pc.denoise_enabled
        )
        self.denoise_strength = pc.denoise_strength

        nyquist = self.sample_rate / 2.0
        self._hpf_sos = butter(
            pc.highpass_order,
            pc.highpass_hz / nyquist,
            btype="high",
            output="sos",
        )
        # Persistent filter state — without this the filter rings on every chunk.
        self._hpf_zi = sosfilt_zi(self._hpf_sos) * 0.0

        # Rolling noise profile, collected from leading silence.
        self._noise_profile: Optional[np.ndarray] = None
        self._noise_samples_needed = int(0.5 * self.sample_rate)
        # The barge-in monitor thread calls this concurrently with the main loop.
        self._lock = threading.Lock()

        logger.info(
            f"AudioPreprocessor ready: HPF={pc.highpass_hz}Hz order={pc.highpass_order}, "
            f"denoise={self.noise_reduce_enabled}, agc={self.agc_enabled} "
            f"(target_rms={self.agc_target_rms}, max_gain={self.agc_max_gain})"
        )

    # ── per-chunk (VAD path) ────────────────────────────────
    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Condition one chunk for VAD. High-pass only; loudness preserved."""
        if audio_chunk is None or len(audio_chunk) == 0:
            return audio_chunk
        with self._lock:
            filtered, self._hpf_zi = sosfilt(
                self._hpf_sos, audio_chunk, zi=self._hpf_zi
            )
            self._collect_noise(audio_chunk)
        return filtered.astype(np.float32)

    # Backwards-compatible alias — older call sites used .process().
    process = process_chunk

    def _collect_noise(self, chunk: np.ndarray) -> None:
        """Accumulate a leading-silence noise profile (caller holds the lock)."""
        if not self.noise_reduce_enabled:
            return
        if self._noise_profile is not None and len(self._noise_profile) >= self._noise_samples_needed:
            return
        if self._noise_profile is None:
            self._noise_profile = chunk.astype(np.float32).copy()
        else:
            self._noise_profile = np.concatenate([self._noise_profile, chunk])

    # ── per-utterance (STT path) ────────────────────────────
    def process_utterance(self, audio: np.ndarray) -> np.ndarray:
        """Clean a complete utterance before transcription."""
        if audio is None or len(audio) == 0:
            return audio
        out = self._reduce_noise(audio)
        out = self._apply_agc(out)
        return out.astype(np.float32)

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        if not self.noise_reduce_enabled:
            return audio
        # noisereduce needs a window at least as long as its STFT frame.
        if len(audio) < 2048:
            return audio
        try:
            import noisereduce as nr

            with self._lock:
                profile = (
                    self._noise_profile.copy()
                    if self._noise_profile is not None
                    and len(self._noise_profile) >= 1024
                    else None
                )
            kwargs = dict(
                y=audio,
                sr=self.sample_rate,
                stationary=True,
                prop_decrease=self.denoise_strength,
            )
            if profile is not None:
                kwargs["y_noise"] = profile
            return nr.reduce_noise(**kwargs).astype(np.float32)
        except Exception as e:
            logger.debug(f"Noise reduction skipped: {e}")
            return audio

    def _apply_agc(self, audio: np.ndarray) -> np.ndarray:
        """Normalise the whole utterance to a target RMS."""
        if not self.agc_enabled:
            return audio
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < 1e-6:
            return audio
        gain = float(np.clip(self.agc_target_rms / rms, 1.0 / self.agc_max_gain, self.agc_max_gain))
        out = audio * gain
        # Prevent clipping after gain.
        peak = float(np.max(np.abs(out))) if len(out) else 0.0
        if peak > 0.99:
            out = out * (0.99 / peak)
        return out.astype(np.float32)

    def reset_noise_profile(self) -> None:
        """Re-estimate the noise profile (call when the environment changes)."""
        with self._lock:
            self._noise_profile = None
        logger.info("Noise profile reset")

    def reset_filter_state(self) -> None:
        """Clear the high-pass filter memory (call between calls/sessions)."""
        with self._lock:
            self._hpf_zi = sosfilt_zi(self._hpf_sos) * 0.0
