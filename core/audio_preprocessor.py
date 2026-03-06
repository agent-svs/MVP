"""
Audio Preprocessor — Noise Suppression + Signal Conditioning

Cleans raw microphone audio before VAD/STT processing:
- Spectral gating noise suppression (noisereduce)
- High-pass filter to remove low-frequency rumble (80Hz cutoff)
- Automatic Gain Control — normalize RMS to target level
"""

import logging
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt

from config import cfg

logger = logging.getLogger(__name__)

# ── High-Pass Filter Design ────────────────────────────────
# 80Hz cutoff, 4th-order Butterworth — removes room rumble, HVAC, etc.
_HPF_CUTOFF = 80  # Hz
_HPF_ORDER = 4


def _design_highpass(sample_rate: int) -> np.ndarray:
    """Design a high-pass Butterworth filter."""
    nyquist = sample_rate / 2.0
    normalized_cutoff = _HPF_CUTOFF / nyquist
    return butter(_HPF_ORDER, normalized_cutoff, btype='high', output='sos')


class AudioPreprocessor:
    """
    Real-time audio preprocessor for voice input.

    Pipeline: raw audio → high-pass filter → noise suppression → AGC
    """

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        agc_target_rms: float = 0.1,
        noise_reduce_enabled: bool = True,
    ):
        self.sample_rate = sample_rate or cfg.audio.sample_rate
        self.agc_target_rms = agc_target_rms
        self.noise_reduce_enabled = noise_reduce_enabled

        # Design high-pass filter
        self._hpf_sos = _design_highpass(self.sample_rate)

        # Noise profile (estimated from initial silence)
        self._noise_profile: Optional[np.ndarray] = None
        self._noise_frames_collected = 0
        self._noise_frames_needed = 10  # ~300ms of noise profile at 30ms chunks

        logger.info(
            f"AudioPreprocessor ready: HPF={_HPF_CUTOFF}Hz, "
            f"AGC target={agc_target_rms}, noise_reduce={noise_reduce_enabled}"
        )

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Process a single audio chunk through the full pipeline.

        Args:
            audio_chunk: float32 mono audio samples

        Returns:
            Cleaned float32 audio chunk (same length)
        """
        if len(audio_chunk) == 0:
            return audio_chunk

        # 1. High-pass filter — remove low-frequency rumble
        filtered = self._apply_highpass(audio_chunk)

        # 2. Noise suppression — spectral gating
        if self.noise_reduce_enabled:
            filtered = self._reduce_noise(filtered)

        # 3. AGC — normalize volume
        filtered = self._apply_agc(filtered)

        return filtered

    def _apply_highpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply high-pass Butterworth filter."""
        return sosfilt(self._hpf_sos, audio).astype(np.float32)

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply spectral gating noise reduction."""
        import noisereduce as nr

        # Collect noise profile from initial silence
        if self._noise_frames_collected < self._noise_frames_needed:
            self._noise_frames_collected += 1
            if self._noise_profile is None:
                self._noise_profile = audio.copy()
            else:
                self._noise_profile = np.concatenate([self._noise_profile, audio])
            # Don't apply NR until we have enough noise profile
            return audio

        # Apply stationary noise reduction
        try:
            reduced = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                y_noise=self._noise_profile,
                stationary=True,
                prop_decrease=0.75,  # How much to reduce noise (0-1)
            )
            return reduced.astype(np.float32)
        except Exception:
            # If noise reduction fails, return original
            return audio

    def _apply_agc(self, audio: np.ndarray) -> np.ndarray:
        """Automatic Gain Control — normalize to target RMS."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6:
            return audio  # Silence, don't amplify noise

        gain = self.agc_target_rms / rms
        # Limit gain to prevent extreme amplification (max 10x) or reduction
        gain = np.clip(gain, 0.1, 10.0)
        return (audio * gain).astype(np.float32)

    def reset_noise_profile(self) -> None:
        """Reset noise profile — call when environment changes."""
        self._noise_profile = None
        self._noise_frames_collected = 0
        logger.info("Noise profile reset — will re-estimate from next silence")
