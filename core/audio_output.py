"""
Audio Output — speaker playback with real interruption.

Uses a dedicated OutputStream and writes in small blocks, so `interrupt()`
actually stops playback within one block (~20ms).

The previous implementation used the module-level sd.play()/sd.wait()/sd.stop()
helpers, which share one global stream. `interrupt()` called sd.stop() from a
monitor thread while play() was blocked inside sd.wait() — a documented race —
and the interrupt flag was only honoured in a code path nothing called.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from config import cfg

logger = logging.getLogger(__name__)

# Playback block size — small enough that a barge-in stops mid-word.
_BLOCK = 1024


class AudioOutput:
    """Play audio through the system speakers."""

    def __init__(self, sample_rate: Optional[int] = None) -> None:
        self.sample_rate = sample_rate if sample_rate is not None else cfg.audio.sample_rate
        self._playing = False
        self._interrupt = threading.Event()
        self._lock = threading.Lock()

    def play(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> bool:
        """Play an array (blocking). Returns True if it finished uninterrupted."""
        if audio is None or len(audio) == 0:
            return True

        sr = sample_rate or self.sample_rate
        data = np.asarray(audio, dtype=np.float32).reshape(-1, 1)

        # Serialise playback so a filler and the main response never overlap.
        with self._lock:
            self._interrupt.clear()
            self._playing = True
            completed = True
            stream = None
            try:
                stream = sd.OutputStream(
                    samplerate=sr, channels=1, dtype="float32", blocksize=_BLOCK
                )
                stream.start()
                for start in range(0, len(data), _BLOCK):
                    if self._interrupt.is_set():
                        completed = False
                        logger.info("Playback interrupted (barge-in)")
                        break
                    stream.write(data[start:start + _BLOCK])
            except Exception as e:
                logger.error(f"Playback error: {e}")
                completed = False
            finally:
                if stream is not None:
                    try:
                        if not completed:
                            stream.abort()
                        stream.stop()
                        stream.close()
                    except Exception as e:
                        logger.debug(f"Error closing output stream: {e}")
                self._playing = False
            return completed

    def interrupt(self) -> None:
        """Stop current playback as soon as the next block boundary is reached."""
        if self._playing:
            logger.info("Playback interrupt requested")
        self._interrupt.set()

    def reset(self) -> None:
        """Clear a pending interrupt before the next utterance."""
        self._interrupt.clear()

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def was_interrupted(self) -> bool:
        return self._interrupt.is_set()
