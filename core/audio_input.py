"""
Audio Input — microphone capture.

Streams float32 mono chunks from the system microphone.

The queue is BOUNDED. Previously it was unbounded, so a slow turn (LLM call +
TTS synthesis takes seconds) let the callback pile up seconds of audio; the
next loop iteration then read stale frames — frequently the agent's own TTS
output — and transcribed them as a new caller utterance.
"""

from __future__ import annotations

import logging
import queue
from typing import Optional

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
        max_queue_chunks: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate if sample_rate is not None else cfg.audio.sample_rate
        self.channels = channels if channels is not None else cfg.audio.channels
        self.chunk_duration_ms = (
            chunk_duration_ms if chunk_duration_ms is not None else cfg.audio.chunk_duration_ms
        )
        self.chunk_samples = int(self.sample_rate * self.chunk_duration_ms / 1000)
        maxsize = max_queue_chunks if max_queue_chunks is not None else cfg.audio.max_queue_chunks

        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=maxsize)
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._muted = False
        self._dropped = 0

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """sounddevice callback — must stay non-blocking."""
        if status:
            logger.debug(f"Audio input status: {status}")
        if self._muted:
            return
        chunk = indata[:, 0].copy() if self.channels == 1 else indata.copy()
        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            # Drop the OLDEST frame so we always hold the most recent audio.
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(chunk)
            except (queue.Empty, queue.Full):
                pass
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning(
                    f"Mic queue full — dropped {self._dropped} stale chunks "
                    "(consumer is falling behind)"
                )

    def start(self) -> None:
        """Start capturing from the microphone."""
        if self._running:
            logger.warning("AudioInput already running")
            return
        logger.info(
            f"Starting mic capture: {self.sample_rate}Hz {self.channels}ch "
            f"{self.chunk_duration_ms}ms chunks"
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

    def stop(self) -> None:
        """Stop capturing and drain."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug(f"Error closing input stream: {e}")
            self._stream = None
        self._running = False
        self.flush()
        logger.info("Microphone capture stopped")

    def flush(self) -> int:
        """Discard everything buffered. Call after the agent finishes speaking
        so its own audio and any stale frames never reach the VAD."""
        dropped = 0
        while True:
            try:
                self._audio_queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        if dropped:
            logger.debug(f"Flushed {dropped} buffered mic chunks")
        return dropped

    def mute(self) -> None:
        """Stop enqueuing frames (used while the agent speaks)."""
        self._muted = True

    def unmute(self) -> None:
        self._muted = False

    def read_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read one chunk, or None on timeout."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_chunks(self):
        """Yield chunks continuously while running."""
        while self._running:
            chunk = self.read_chunk(timeout=0.5)
            if chunk is not None:
                yield chunk

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queued_chunks(self) -> int:
        return self._audio_queue.qsize()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
