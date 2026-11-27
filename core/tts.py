"""
Text-to-Speech — Coqui XTTS v2 wrapper.

Two behaviours worth calling out:

  * `synthesize_stream` is genuinely pipelined. A background thread synthesises
    sentence N+1 while the caller is still playing sentence N. The previous
    version synthesised, yielded, then waited for playback to finish before
    starting the next sentence — so it carried the complexity of streaming with
    none of the latency benefit.

  * The phrase cache only stores text with no caller data in it. Caching
    personalised responses wrote patient names to disk as audio files, forever,
    for a near-zero hit rate.
"""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import re
import threading
import time
from typing import Iterator, Optional

import numpy as np

from config import cfg

logger = logging.getLogger(__name__)

# XTTS is unstable above ~250 characters per call.
_MAX_CHARS = 200

# A phrase is cacheable only if it looks fully generic. Anything containing a
# digit (dates, times, phone numbers) or a capitalised word that isn't a known
# safe word is treated as caller-specific and never written to disk.
_SAFE_CAPITALISED = {
    "i", "i'm", "i'll", "i've", "ok", "okay", "sure", "hello", "hi", "hey",
    "good", "great", "thanks", "thank", "welcome", "bye", "goodbye", "sorry",
    "what", "which", "when", "where", "who", "how", "could", "would", "may",
    "and", "the", "a", "an", "is", "it", "no", "yes", "just", "one", "let",
    "do", "does", "did", "can", "shall", "we", "you", "your", "our", "my",
    "that", "this", "there", "here", "for", "to", "of", "on", "at", "in",
}


def _is_cacheable(text: str) -> bool:
    """True only when `text` contains no caller-specific information."""
    if not cfg.tts.cache_enabled:
        return False
    if any(ch.isdigit() for ch in text):
        return False
    words = re.findall(r"[A-Za-z']+", text)
    for i, w in enumerate(words):
        if not w[:1].isupper():
            continue
        if i == 0:
            continue  # sentence-initial capital is fine
        if w.lower() in _SAFE_CAPITALISED:
            continue
        if w.lower() in cfg.business.name.lower().split():
            continue
        return False  # a proper noun we don't recognise → probably a name
    return True


class TTS:
    """Text-to-Speech using Coqui XTTS v2."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        speaker: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or cfg.tts.model
        self.language = language or cfg.tts.language
        self.device = device or cfg.device
        self.speaker = speaker or cfg.tts.speaker

        raw = speaker_wav or cfg.tts.speaker_wav
        resolved = (
            str(cfg.project_root / raw) if raw and not os.path.isabs(raw) else (raw or "")
        )
        self.speaker_wav = resolved if resolved and os.path.exists(resolved) else None

        self._cache_dir = cfg.project_root / "assets" / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._sample_rate: int = 24000  # replaced with the model's real rate on load
        self._lock = threading.Lock()

    # ── loading ─────────────────────────────────────────────
    def load(self) -> None:
        """Load the XTTS model. Call once at startup."""
        from TTS.api import TTS as CoquiTTS

        logger.info(f"Loading TTS model: {self.model_name} (device={self.device})")
        t0 = time.time()
        self._model = CoquiTTS(model_name=self.model_name, progress_bar=False)
        try:
            self._model.to(self.device)
        except Exception as e:
            logger.warning(f"Could not move TTS to {self.device} ({e}); staying on CPU")
            self.device = "cpu"

        # Read the real output rate instead of assuming 24000.
        for attr in ("output_sample_rate", "sample_rate"):
            rate = getattr(getattr(self._model, "synthesizer", None), attr, None)
            if isinstance(rate, int) and rate > 0:
                self._sample_rate = rate
                break

        mode = f"voice cloning from {self.speaker_wav}" if self.speaker_wav else \
               f"built-in speaker '{self.speaker}'"
        logger.info(f"TTS loaded in {time.time() - t0:.1f}s @ {self._sample_rate}Hz — {mode}")

    # ── cache ───────────────────────────────────────────────
    def _cache_key(self, text: str) -> str:
        voice_id = self.speaker_wav or self.speaker
        return hashlib.sha256(
            f"{text}:{self.language}:{voice_id}:{self._sample_rate}".encode()
        ).hexdigest()[:32]

    def _check_cache(self, text: str) -> Optional[np.ndarray]:
        path = self._cache_dir / f"{self._cache_key(text)}.npy"
        if path.exists():
            try:
                logger.debug(f"TTS cache hit: {text[:40]!r}")
                return np.load(str(path), allow_pickle=False)
            except Exception as e:
                logger.debug(f"Corrupt TTS cache entry {path.name}: {e}")
                path.unlink(missing_ok=True)
        return None

    def _save_cache(self, text: str, audio: np.ndarray) -> None:
        try:
            self._evict_if_needed()
            np.save(str(self._cache_dir / f"{self._cache_key(text)}.npy"), audio)
        except Exception as e:
            logger.debug(f"Could not write TTS cache: {e}")

    def _evict_if_needed(self) -> None:
        files = sorted(self._cache_dir.glob("*.npy"), key=lambda f: f.stat().st_mtime)
        excess = len(files) - cfg.tts.cache_max_files
        for f in files[:max(0, excess)]:
            f.unlink(missing_ok=True)

    def clear_cache(self) -> int:
        count = 0
        for f in self._cache_dir.glob("*.npy"):
            f.unlink(missing_ok=True)
            count += 1
        logger.info(f"Cleared {count} cached TTS files")
        return count

    # ── synthesis ───────────────────────────────────────────
    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, int, float]:
        """Convert text to speech. Returns (audio, sample_rate, elapsed)."""
        if not text or not text.strip():
            return np.array([], dtype=np.float32), self._sample_rate, 0.0

        cacheable = use_cache and _is_cacheable(text)
        if cacheable:
            cached = self._check_cache(text)
            if cached is not None:
                return cached, self._sample_rate, 0.0

        if self._model is None:
            self.load()

        lang = language or self.language
        t0 = time.time()
        try:
            with self._lock:
                kwargs = {"text": text, "language": lang}
                if self.speaker_wav:
                    kwargs["speaker_wav"] = self.speaker_wav
                else:
                    kwargs["speaker"] = self.speaker
                audio_list = self._model.tts(**kwargs)
        except Exception as e:
            logger.error(f"TTS synthesis failed for {text[:60]!r}: {e}")
            return np.array([], dtype=np.float32), self._sample_rate, time.time() - t0

        elapsed = time.time() - t0
        audio = np.asarray(audio_list, dtype=np.float32)

        logger.info(
            f"TTS ({elapsed:.2f}s): {text[:50]!r} "
            f"[{len(audio)/max(self._sample_rate,1):.1f}s audio]"
        )
        if cacheable and len(audio):
            self._save_cache(text, audio)
        return audio, self._sample_rate, elapsed

    def synthesize_stream(
        self,
        text: str,
        language: Optional[str] = None,
        use_cache: bool = True,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """Yield (audio, sample_rate) per sentence, synthesising ahead.

        A worker thread stays one or two sentences in front of the consumer, so
        playback of sentence N overlaps synthesis of N+1.
        """
        if not text or not text.strip():
            return
        sentences = self._split_into_sentences(text)
        if not sentences:
            return
        if len(sentences) == 1:
            audio, sr, _ = self.synthesize(sentences[0], language, use_cache)
            if len(audio):
                yield audio, sr
            return

        out_q: "queue.Queue" = queue.Queue(maxsize=2)
        _SENTINEL = object()

        def worker() -> None:
            try:
                for i, sentence in enumerate(sentences):
                    audio, sr, elapsed = self.synthesize(sentence, language, use_cache)
                    if len(audio):
                        logger.debug(
                            f"Stream chunk {i+1}/{len(sentences)} ready ({elapsed:.2f}s)"
                        )
                        out_q.put((audio, sr))
            except Exception as e:
                logger.error(f"TTS stream worker failed: {e}")
            finally:
                out_q.put(_SENTINEL)

        thread = threading.Thread(target=worker, name="tts-stream", daemon=True)
        thread.start()
        try:
            while True:
                item = out_q.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            # Drain so the worker can exit if the consumer stopped early
            # (barge-in), then let the daemon thread finish.
            while not out_q.empty():
                try:
                    out_q.get_nowait()
                except queue.Empty:
                    break

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split into speakable chunks, respecting XTTS's length limit."""
        parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
        result: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= _MAX_CHARS:
                result.append(part)
                continue
            # Split long sentences on commas, then hard-wrap on word
            # boundaries so a comma-less run can never exceed the XTTS limit.
            buf = ""
            for piece in re.split(r',\s+', part):
                candidate = f"{buf}, {piece}" if buf else piece
                if len(candidate) <= _MAX_CHARS:
                    buf = candidate
                    continue
                if buf:
                    result.append(buf)
                    buf = ""
                while len(piece) > _MAX_CHARS:
                    cut = piece.rfind(" ", 0, _MAX_CHARS)
                    if cut <= 0:
                        cut = _MAX_CHARS
                    result.append(piece[:cut].strip())
                    piece = piece[cut:].strip()
                buf = piece
            if buf:
                result.append(buf)
        return [r for r in result if r.strip()]

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
