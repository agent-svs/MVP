"""
Voice Activity Detection — Silero VAD Wrapper

Detects speech segments in streaming audio using the Silero VAD model.
Accumulates audio during speech and emits complete utterances on silence.
"""

import logging
import time
from typing import Optional

import numpy as np
import torch

from config import cfg

logger = logging.getLogger(__name__)


class VAD:
    """
    Voice Activity Detection using Silero VAD.

    Processes audio chunks and detects speech/silence transitions.
    Accumulates speech segments and emits complete utterances when
    silence is detected after speech.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        min_speech_ms: Optional[int] = None,
        min_silence_ms: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ):
        self.threshold = threshold or cfg.vad.threshold
        self.min_speech_ms = min_speech_ms or cfg.vad.min_speech_ms
        self.min_silence_ms = min_silence_ms or cfg.vad.min_silence_ms
        self.sample_rate = sample_rate or cfg.audio.sample_rate

        # State
        self._speech_buffer: list[np.ndarray] = []
        self._is_speaking = False
        self._silence_start: Optional[float] = None
        self._speech_start: Optional[float] = None
        self._audio_buffer = np.array([], dtype=np.float32)

        # Silero VAD requires exactly 512 samples per chunk at 16kHz
        self._vad_chunk_size = 512

        # Load model (using silero-vad package — no internet needed)
        logger.info("Loading Silero VAD model...")
        from silero_vad import load_silero_vad
        self._model = load_silero_vad()
        logger.info("Silero VAD model loaded")

    def reset(self) -> None:
        """Reset VAD state for a new utterance."""
        self._speech_buffer.clear()
        self._is_speaking = False
        self._silence_start = None
        self._speech_start = None
        self._audio_buffer = np.array([], dtype=np.float32)
        self._model.reset_states()

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process an audio chunk through VAD.

        Args:
            audio_chunk: numpy float32 array of audio samples (mono, 16kHz)

        Returns:
            Complete utterance as numpy array when speech ends, else None.
        """
        # Accumulate audio until we have enough for VAD
        self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])

        # Process all complete 512-sample chunks in the buffer
        result = None
        while len(self._audio_buffer) >= self._vad_chunk_size:
            vad_chunk = self._audio_buffer[:self._vad_chunk_size]
            self._audio_buffer = self._audio_buffer[self._vad_chunk_size:]
            result = self._process_vad_chunk(vad_chunk)
            if result is not None:
                return result

        return None

    def _process_vad_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process a single 512-sample chunk through Silero VAD."""
        # Convert to torch tensor
        tensor = torch.from_numpy(audio_chunk).float()

        # Get speech probability
        speech_prob = self._model(tensor, self.sample_rate).item()

        now = time.time()

        if speech_prob >= self.threshold:
            # Speech detected
            self._silence_start = None

            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start = now
                logger.debug("Speech started")

            self._speech_buffer.append(audio_chunk)

        else:
            # Silence detected
            if self._is_speaking:
                self._speech_buffer.append(audio_chunk)

                if self._silence_start is None:
                    self._silence_start = now
                else:
                    silence_duration_ms = (now - self._silence_start) * 1000

                    if silence_duration_ms >= self.min_silence_ms:
                        # Check minimum speech duration
                        speech_duration_ms = (now - self._speech_start) * 1000
                        if speech_duration_ms >= self.min_speech_ms:
                            # Emit complete utterance
                            utterance = np.concatenate(self._speech_buffer)
                            logger.info(
                                f"Utterance complete: {len(utterance)} samples "
                                f"({speech_duration_ms:.0f}ms speech)"
                            )
                            self.reset()
                            return utterance
                        else:
                            # Too short — likely noise, discard
                            logger.debug(
                                f"Discarding short speech: {speech_duration_ms:.0f}ms"
                            )
                            self.reset()

        return None

    def check_barge_in(self, audio_chunk: np.ndarray) -> bool:
        """
        Lightweight speech check for barge-in detection during playback.

        Unlike process_chunk(), this doesn't accumulate speech or track
        utterance boundaries — it just checks if the user is speaking.

        Args:
            audio_chunk: raw mic audio (any length)

        Returns:
            True if speech is detected (potential barge-in)
        """
        # Need at least 512 samples for Silero
        if len(audio_chunk) < self._vad_chunk_size:
            return False

        # Use only the last 512 samples for a quick check
        chunk = audio_chunk[-self._vad_chunk_size:]
        tensor = torch.from_numpy(chunk).float()
        prob = self._model(tensor, self.sample_rate).item()
        self._last_speech_prob = prob

        if prob >= self.threshold:
            logger.debug(f"Barge-in detected: speech_prob={prob:.2f}")
            return True
        return False

    @property
    def is_speaking(self) -> bool:
        """Whether the user is currently speaking."""
        return self._is_speaking

    @property
    def speech_prob(self) -> float:
        """Last computed speech probability."""
        return getattr(self, '_last_speech_prob', 0.0)


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    from core.audio_input import AudioInput

    cfg.setup_logging()
    print("🎙️  Testing VAD — speak into your microphone")
    print(f"   Threshold:     {cfg.vad.threshold}")
    print(f"   Min speech:    {cfg.vad.min_speech_ms}ms")
    print(f"   Min silence:   {cfg.vad.min_silence_ms}ms")
    print(f"   Press Ctrl+C to stop")
    print()

    vad = VAD()
    utterance_count = 0

    try:
        with AudioInput() as mic:
            for chunk in mic.read_chunks():
                utterance = vad.process_chunk(chunk)
                if utterance is not None:
                    utterance_count += 1
                    duration = len(utterance) / cfg.audio.sample_rate
                    print(f"  ✅ Utterance #{utterance_count}: {duration:.2f}s")
                elif vad.is_speaking:
                    print("\r  🔴 Speaking...", end="")
                else:
                    print("\r  ⚪ Silence    ", end="")
    except KeyboardInterrupt:
        print(f"\n\nStopped. Detected {utterance_count} utterances.")
