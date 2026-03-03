"""
Audio Input — Microphone Capture Module

Captures audio from the system microphone using sounddevice.
Yields raw PCM audio chunks at the configured sample rate.
"""

import logging
import queue
import threading
from typing import Optional, Callable

import numpy as np
import sounddevice as sd

from config import cfg

logger = logging.getLogger(__name__)


class AudioInput:
    """Stream audio from the system microphone."""

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        chunk_duration_ms: Optional[int] = None,
    ):
        self.sample_rate = sample_rate or cfg.audio.sample_rate
        self.channels = channels or cfg.audio.channels
        self.chunk_duration_ms = chunk_duration_ms or cfg.audio.chunk_duration_ms
        self.chunk_samples = int(self.sample_rate * self.chunk_duration_ms / 1000)

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._running = False

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """Called by sounddevice for each audio chunk."""
        if status:
            logger.warning(f"Audio input status: {status}")
        # Copy the data — indata buffer is reused
        self._audio_queue.put(indata[:, 0].copy() if self.channels == 1 else indata.copy())

    def start(self) -> None:
        """Start capturing audio from the microphone."""
        if self._running:
            logger.warning("AudioInput already running")
            return

        logger.info(
            f"Starting mic capture: {self.sample_rate}Hz, "
            f"{self.channels}ch, {self.chunk_duration_ms}ms chunks"
        )
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_samples,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._running = True
        logger.info("Microphone capture started")

    def stop(self) -> None:
        """Stop capturing audio."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        # Drain the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Microphone capture stopped")

    def read_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Read one audio chunk from the microphone.

        Returns:
            numpy array of float32 audio samples, or None if timeout.
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_chunks(self):
        """Generator that yields audio chunks continuously."""
        while self._running:
            chunk = self.read_chunk(timeout=0.5)
            if chunk is not None:
                yield chunk

    @property
    def is_running(self) -> bool:
        return self._running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    import time

    cfg.setup_logging()
    print("🎙️  Testing microphone capture (5 seconds)...")
    print(f"   Sample rate: {cfg.audio.sample_rate} Hz")
    print(f"   Chunk size:  {cfg.audio.chunk_duration_ms} ms")
    print()

    collected_chunks = []
    with AudioInput() as mic:
        start = time.time()
        while time.time() - start < 5.0:
            chunk = mic.read_chunk(timeout=0.5)
            if chunk is not None:
                collected_chunks.append(chunk)
                # Print level meter
                level = np.abs(chunk).mean()
                bars = int(level * 200)
                print(f"\r  Level: {'█' * bars}{' ' * (40 - bars)} {level:.4f}", end="")
        print()

    total_samples = sum(len(c) for c in collected_chunks)
    duration = total_samples / cfg.audio.sample_rate
    print(f"\n✅ Captured {len(collected_chunks)} chunks, ~{duration:.2f}s of audio")
