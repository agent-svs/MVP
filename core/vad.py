"""
Voice Activity Detection — Silero VAD wrapper.

Accumulates speech and emits complete utterances on trailing silence.

Silero's model is a stateful RNN. The pipeline touches it from both the main
loop and the barge-in monitor thread, so every call is serialised behind a
lock — previously they raced and corrupted each other's hidden state.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import torch

from config import cfg

logger = logging.getLogger(__name__)


class VAD:
    """Voice Activity Detection using Silero VAD."""

    def __init__(
        self,
        threshold: Optional[float] = None,
        min_speech_ms: Optional[int] = None,
        min_silence_ms: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        self.threshold = threshold if threshold is not None else cfg.vad.threshold
        self.min_speech_ms = min_speech_ms if min_speech_ms is not None else cfg.vad.min_speech_ms
        self.min_silence_ms = (
            min_silence_ms if min_silence_ms is not None else cfg.vad.min_silence_ms
        )
        self.sample_rate = sample_rate if sample_rate is not None else cfg.audio.sample_rate

        # Silero requires an exact window per sample rate (512@16k, 256@8k).
        # This used to be hardcoded to 512 while the rate was configurable.
        self._vad_chunk_size = cfg.vad.chunk_size_for(self.sample_rate)

        self._speech_buffer: list[np.ndarray] = []
        self._is_speaking = False
        self._silence_chunks = 0
        self._speech_chunks = 0
        self._audio_buffer = np.array([], dtype=np.float32)
        self._last_speech_prob = 0.0
        # Barge-in works on its own buffer: mic chunks (480 samples at
        # 30ms) are SHORTER than the Silero window (512), so a per-call
        # length check rejected every chunk and barge-in never fired.
        self._barge_buffer = np.array([], dtype=np.float32)
        self._barge_hits = 0
        # Pre-roll so the first phoneme isn't clipped off the utterance.
        self._preroll_chunks = max(1, int(0.2 * self.sample_rate / self._vad_chunk_size))
        self._preroll: list[np.ndarray] = []

        self._lock = threading.RLock()

        logger.info("Loading Silero VAD model...")
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()
        logger.info(
            f"Silero VAD ready (window={self._vad_chunk_size} @ {self.sample_rate}Hz, "
            f"threshold={self.threshold})"
        )

    # ── state ───────────────────────────────────────────────
    def reset(self) -> None:
        """Reset all VAD state, including the model's hidden state."""
        with self._lock:
            self._speech_buffer.clear()
            self._preroll.clear()
            self._is_speaking = False
            self._silence_chunks = 0
            self._speech_chunks = 0
            self._audio_buffer = np.array([], dtype=np.float32)
            self._barge_buffer = np.array([], dtype=np.float32)
            self._barge_hits = 0
            try:
                self._model.reset_states()
            except Exception as e:
                logger.debug(f"VAD reset_states failed: {e}")

    def _speech_prob(self, chunk: np.ndarray) -> float:
        """Run the model on exactly one window (caller holds the lock)."""
        tensor = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))
        with torch.inference_mode():
            return float(self._model(tensor, self.sample_rate).item())

    # ── streaming ───────────────────────────────────────────
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Feed a chunk; returns a complete utterance when speech ends."""
        if audio_chunk is None or len(audio_chunk) == 0:
            return None
        with self._lock:
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])
            while len(self._audio_buffer) >= self._vad_chunk_size:
                window = self._audio_buffer[: self._vad_chunk_size]
                self._audio_buffer = self._audio_buffer[self._vad_chunk_size:]
                result = self._process_window(window)
                if result is not None:
                    return result
            return None

    def _process_window(self, window: np.ndarray) -> Optional[np.ndarray]:
        prob = self._speech_prob(window)
        self._last_speech_prob = prob

        if prob >= self.threshold:
            self._silence_chunks = 0
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_chunks = 0
                # Prepend the pre-roll so the utterance starts cleanly.
                self._speech_buffer.extend(self._preroll)
                logger.debug("Speech started")
            self._speech_buffer.append(window)
            self._speech_chunks += 1
            return None

        # Silence.
        if not self._is_speaking:
            self._preroll.append(window)
            if len(self._preroll) > self._preroll_chunks:
                self._preroll.pop(0)
            return None

        self._speech_buffer.append(window)
        self._silence_chunks += 1
        silence_ms = self._silence_chunks * self._vad_chunk_size / self.sample_rate * 1000
        if silence_ms < self.min_silence_ms:
            return None

        speech_ms = self._speech_chunks * self._vad_chunk_size / self.sample_rate * 1000
        if speech_ms >= self.min_speech_ms:
            utterance = np.concatenate(self._speech_buffer)
            logger.info(
                f"Utterance complete: {len(utterance)} samples "
                f"({speech_ms:.0f}ms speech, {silence_ms:.0f}ms trailing silence)"
            )
            self.reset()
            return utterance

        logger.debug(f"Discarding short speech: {speech_ms:.0f}ms < {self.min_speech_ms}ms")
        self.reset()
        return None

    # ── barge-in ────────────────────────────────────────────
    def check_barge_in(self, audio_chunk: np.ndarray, consecutive: Optional[int] = None) -> bool:
        """Speech check during playback, gated on BOTH probability and energy.

        The microphone hears the agent's own speakers, so a plain VAD check
        makes the agent interrupt itself mid-sentence. Real barge-in must clear
        a higher speech probability AND an absolute amplitude floor that the
        acoustic echo does not reach. (This is only meaningful because AGC no
        longer runs per chunk — absolute amplitude is now trustworthy.)
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return False

        needed = consecutive if consecutive is not None else cfg.vad.barge_in_consecutive
        floor = cfg.vad.barge_in_min_amplitude
        prob_floor = cfg.vad.barge_in_threshold

        with self._lock:
            self._barge_buffer = np.concatenate([self._barge_buffer, audio_chunk])
            while len(self._barge_buffer) >= self._vad_chunk_size:
                window = self._barge_buffer[: self._vad_chunk_size]
                self._barge_buffer = self._barge_buffer[self._vad_chunk_size:]

                peak = float(np.max(np.abs(window)))
                if peak < floor:
                    # Too quiet to be someone talking over the speaker.
                    self._barge_hits = 0
                    continue

                prob = self._speech_prob(window)
                self._last_speech_prob = prob
                if prob >= prob_floor:
                    self._barge_hits += 1
                    if self._barge_hits >= needed:
                        logger.info(
                            f"Barge-in confirmed: prob={prob:.2f} peak={peak:.3f}"
                        )
                        self._barge_hits = 0
                        self._barge_buffer = np.array([], dtype=np.float32)
                        return True
                else:
                    self._barge_hits = 0
            return False

    def reset_barge_in(self) -> None:
        """Clear barge-in accumulation between utterances."""
        with self._lock:
            self._barge_buffer = np.array([], dtype=np.float32)
            self._barge_hits = 0

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def speech_prob(self) -> float:
        return self._last_speech_prob
