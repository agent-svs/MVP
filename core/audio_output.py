"""
Audio Output — Speaker Playback Module

Plays audio through the system speakers using sounddevice.
Supports streaming playback and mid-stream interruption (for barge-in).
"""

import logging
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from config import cfg

logger = logging.getLogger(__name__)


class AudioOutput:
    """Play audio through the system speakers."""

    def __init__(self, sample_rate: Optional[int] = None):
        self.sample_rate = sample_rate or cfg.audio.sample_rate
        self._playing = False
        self._interrupt_flag = threading.Event()

    def play(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> None:
        """
        Play an audio array through the speakers (blocking).

        Args:
            audio: numpy float32 array of audio samples
            sample_rate: override sample rate, defaults to config
        """
        sr = sample_rate or self.sample_rate
        self._playing = True
        self._interrupt_flag.clear()

        try:
            logger.debug(f"Playing audio: {len(audio)} samples at {sr}Hz")
            sd.play(audio, samplerate=sr)
            sd.wait()
        except Exception as e:
            logger.error(f"Playback error: {e}")
        finally:
            self._playing = False

    def play_streaming(
        self, audio_chunks, sample_rate: Optional[int] = None
    ) -> None:
        """
        Play audio chunks as they arrive (streaming playback).

        Args:
            audio_chunks: iterable of numpy float32 arrays
            sample_rate: override sample rate, defaults to config
        """
        sr = sample_rate or self.sample_rate
        self._playing = True
        self._interrupt_flag.clear()

        try:
            for chunk in audio_chunks:
                if self._interrupt_flag.is_set():
                    logger.info("Playback interrupted (barge-in)")
                    break
                sd.play(chunk, samplerate=sr)
                sd.wait()
        except Exception as e:
            logger.error(f"Streaming playback error: {e}")
        finally:
            self._playing = False

    def interrupt(self) -> None:
        """Stop current playback immediately (for barge-in)."""
        self._interrupt_flag.set()
        sd.stop()
        self._playing = False
        logger.info("Playback interrupted")

    @property
    def is_playing(self) -> bool:
        return self._playing


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    cfg.setup_logging()
    print("🔊 Testing speaker output...")
    print(f"   Sample rate: {cfg.audio.sample_rate} Hz")
    print()

    # Generate a simple test tone (440Hz sine wave, 1 second)
    duration = 1.0
    t = np.linspace(0, duration, int(cfg.audio.sample_rate * duration), dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)

    player = AudioOutput()
    print("  Playing 440Hz test tone for 1 second...")
    player.play(tone)
    print("✅ Playback complete")
