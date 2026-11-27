"""
Speech-to-Text — Faster Whisper wrapper.

CTranslate2 has no Apple MPS support, so on a Mac this always runs on CPU;
the device/compute policy comes from cfg.stt_device.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

import numpy as np

from config import cfg

logger = logging.getLogger(__name__)

# ── Disfluency patterns ─────────────────────────────────────
DISFLUENCY_PATTERNS = [
    r"\b(um+|uh+|uh-huh|mm+|hmm+|er+|ah+)\b",
    r"\b(you know|i mean|basically)\b",
]
_DISFLUENCY_RE = [re.compile(p, re.IGNORECASE) for p in DISFLUENCY_PATTERNS]
_MULTI_SPACE_RE = re.compile(r"\s{2,}")

# A transcript with no letters AND no digits is noise. Digits must be kept —
# the old pattern (^[^a-zA-Z]*$) discarded spoken phone numbers entirely.
_NOISE_RE = re.compile(r"^[^a-zA-Z0-9]*$")

# Canned phrases Whisper emits when fed near-silence. Only treated as
# hallucinations when the audio was genuinely quiet.
WHISPER_HALLUCINATIONS = frozenset({
    "thanks for watching", "thank you for watching", "please subscribe",
    "see you next time", "subscribe to my channel", "like and subscribe",
    "thanks for watching!", "you", ".", "...", "bye.",
    "in the name of allah", "in the name of allah, the most gracious, the most merciful",
    "pooja salil", "pooja salil karo", "built up in america",
    "thanks for coming", "i'll see you later", "translated by",
    "subtitles by", "amara.org", "www.mooji.org",
})


def clean_disfluencies(text: str) -> str:
    """Remove filler words and immediate word repetitions."""
    cleaned = text
    for pattern in _DISFLUENCY_RE:
        cleaned = pattern.sub("", cleaned)
    # "I I want" → "I want", but never collapse meaningful repeats like "no no".
    cleaned = re.sub(r"\b(\w{3,})(\s+\1)+\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    return cleaned


class STT:
    """Speech-to-Text using Faster Whisper."""

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        beam_size: Optional[int] = None,
    ) -> None:
        self.model_size = model_size or cfg.stt.model_size
        self.language = language or cfg.stt.language
        self.beam_size = beam_size if beam_size is not None else cfg.stt.beam_size

        resolved_device, resolved_compute = cfg.stt_device
        self.device = device or resolved_device
        self.compute_type = compute_type or resolved_compute
        if self.device == "mps":
            logger.info("CTranslate2 has no MPS support — using CPU/int8 for STT")
            self.device, self.compute_type = "cpu", "int8"

        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the Whisper model. Call once at startup."""
        from faster_whisper import WhisperModel

        logger.info(
            f"Loading Faster Whisper: {self.model_size} "
            f"(device={self.device}, compute={self.compute_type})"
        )
        t0 = time.time()
        self._model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )
        logger.info(f"Faster Whisper loaded in {time.time() - t0:.1f}s")

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        clean: bool = True,
    ) -> tuple[str, float]:
        """Transcribe an utterance. Returns (text, elapsed_seconds)."""
        if audio is None or len(audio) == 0:
            return "", 0.0
        if self._model is None:
            self.load()

        audio = np.asarray(audio, dtype=np.float32)
        max_amp = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        duration_s = len(audio) / cfg.audio.sample_rate

        logger.debug(
            f"Audio: dur={duration_s:.2f}s max_amp={max_amp:.4f} rms={rms:.4f}"
        )

        # Gates are configurable. The old hardcoded 0.05 / 0.8s pair discarded
        # every short reply ("yes", "3 pm", "Monday") before transcription.
        if max_amp < cfg.stt.min_amplitude:
            logger.info(f"STT skipped — too quiet (max_amp={max_amp:.4f})")
            return "", 0.0
        if duration_s < cfg.stt.min_duration_s:
            logger.info(f"STT skipped — too short ({duration_s:.2f}s)")
            return "", 0.0

        lang = language or self.language
        t0 = time.time()
        try:
            with self._lock:
                segments, info = self._model.transcribe(
                    audio,
                    language=lang,
                    beam_size=self.beam_size,
                    vad_filter=False,               # we run our own VAD
                    without_timestamps=True,
                    condition_on_previous_text=False,  # stops runaway repetition
                    temperature=[0.0, 0.2, 0.4],       # fallback on low confidence
                    no_speech_threshold=0.6,
                )
                text_parts = [seg.text.strip() for seg in segments]
        except Exception as e:
            logger.error(f"STT failed: {e}")
            return "", time.time() - t0

        raw_text = " ".join(text_parts).strip()
        elapsed = time.time() - t0

        lower = raw_text.lower().strip(".!?, ")
        if max_amp < cfg.stt.hallucination_amplitude and (
            lower in WHISPER_HALLUCINATIONS or lower.startswith("in the name of allah")
        ):
            logger.info(f"STT hallucination blocked (max_amp={max_amp:.3f}): {raw_text!r}")
            return "", elapsed

        text = clean_disfluencies(raw_text) if (clean and raw_text) else raw_text

        if text and _NOISE_RE.match(text):
            logger.info(f"STT ({elapsed:.2f}s): filtered noise {text!r}")
            return "", elapsed

        logger.info(
            f"STT ({elapsed:.2f}s): {text!r} "
            f"[lang={getattr(info, 'language', '?')} "
            f"p={getattr(info, 'language_probability', 0):.2f}]"
        )
        return text, elapsed

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
