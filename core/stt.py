"""
Speech-to-Text — Faster Whisper Wrapper

Transcribes audio segments using Faster Whisper (CTranslate2 backend).
Configurable model size, language, and beam size via .env.

Note: CTranslate2 does NOT support Apple MPS — always runs on CPU on Mac.
For GPU acceleration, set DEVICE=cuda (requires NVIDIA GPU).
"""

import logging
import re
import time
from typing import Optional

import numpy as np

from config import cfg

logger = logging.getLogger(__name__)

# ── Disfluency patterns ────────────────────────────────────
# Regex patterns to remove speech disfluencies that hurt NLU accuracy
DISFLUENCY_PATTERNS = [
    # Filler words (whole words only)
    r"\b(um+|uh+|uh-huh|mm+|hmm+|er+|ah+)\b",
    # Hedge phrases
    r"\b(you know|i mean|basically|sort of|kind of)\b",
]

# Pre-compile for performance
_DISFLUENCY_RE = [re.compile(p, re.IGNORECASE) for p in DISFLUENCY_PATTERNS]
# Collapse multiple spaces
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
# Pattern to detect transcriptions that are just noise (dots, punctuation, no words)
_NOISE_RE = re.compile(r'^[^a-zA-Z]*$')


def clean_disfluencies(text: str) -> str:
    """Remove speech disfluencies (um, uh, like, false starts) from text."""
    cleaned = text
    for pattern in _DISFLUENCY_RE:
        cleaned = pattern.sub("", cleaned)
    # Handle word repetitions separately to keep one copy: "I I want" → "I want"
    cleaned = re.sub(r"\b(\w+)(\s+\1)+\b", r"\1", cleaned, flags=re.IGNORECASE)
    # Clean up whitespace
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
    # Fix punctuation spacing
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    return cleaned


class STT:
    """
    Speech-to-Text using Faster Whisper.

    Wraps the WhisperModel with env-configurable settings.
    CTranslate2 runs on CPU on Mac (MPS not supported).
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        beam_size: Optional[int] = None,
    ):
        self.model_size = model_size or cfg.stt.model_size
        self.language = language or cfg.stt.language
        self.beam_size = beam_size or cfg.stt.beam_size

        # CTranslate2 doesn't support MPS — force CPU on Mac
        raw_device = device or cfg.device
        if raw_device == "mps":
            self.device = "cpu"
            self.compute_type = "int8"  # Best CPU performance
            logger.info("CTranslate2 doesn't support MPS — using CPU with int8")
        elif raw_device == "cuda":
            self.device = "cuda"
            self.compute_type = compute_type or cfg.compute_type
        else:
            self.device = "cpu"
            self.compute_type = compute_type or "int8"

        self._model = None

    def load(self) -> None:
        """Load the Whisper model. Call once at startup."""
        from faster_whisper import WhisperModel

        logger.info(
            f"Loading Faster Whisper model: {self.model_size} "
            f"(device={self.device}, compute={self.compute_type})"
        )
        t0 = time.time()
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        elapsed = time.time() - t0
        logger.info(f"Faster Whisper model loaded in {elapsed:.1f}s")

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        clean: bool = True,
    ) -> tuple[str, float]:
        """
        Transcribe an audio segment to text.

        Args:
            audio: numpy float32 array of audio samples (mono, 16kHz)
            language: override language, defaults to config
            clean: whether to apply disfluency filtering

        Returns:
            Tuple of (transcribed_text, elapsed_seconds)
        """
        if self._model is None:
            self.load()

        # Audio energy and duration gates to prevent Whisper hallucination
        max_amp = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        duration_s = len(audio) / 16000
        
        # Log the levels so we can tune them
        logger.debug(f"Audio check: duration={duration_s:.2f}s, max_amp={max_amp:.4f}, rms={rms:.4f}")
        
        # Strict gate for silence/static
        if max_amp < 0.05 or duration_s < 0.8:
            logger.info(f"STT skipped — too short/quiet (max={max_amp:.4f}, rms={rms:.4f}, dur={duration_s:.2f}s)")
            return "", 0.0

        lang = language or self.language
        t0 = time.time()

        segments, info = self._model.transcribe(
            audio,
            language=lang,
            beam_size=self.beam_size,
            vad_filter=False,  # We do our own VAD
            without_timestamps=True,
        )

        # Collect all segment texts
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        raw_text = " ".join(text_parts).strip()
        elapsed = time.time() - t0

        # Hallucination blocklist
        WHISPER_HALLUCINATIONS = {
            "thanks for watching", "thank you for watching",
            "thank you", "thanks", "bye", "you",
            "please subscribe", "see you next time",
            "in the name of allah", "in the name of allah, the most gracious, the most merciful",
            "pooja salil", "pooja salil karo", "built up in america",
            "thank you very much", "thanks for coming", "i'll see you later"
        }
        
        lower_stripped = raw_text.lower().strip('.!?, ')
        if lower_stripped in WHISPER_HALLUCINATIONS or lower_stripped.startswith("in the name of allah"):
            logger.info(f"STT hallucination blocked: '{raw_text}'")
            return "", elapsed

        # Apply disfluency filter
        if clean and raw_text:
            cleaned = clean_disfluencies(raw_text)
            if cleaned != raw_text:
                logger.debug(f"Disfluency filter: '{raw_text}' → '{cleaned}'")
            text = cleaned
        else:
            text = raw_text

        # Filter out noise transcriptions (just dots, punctuation, no real words)
        if text and _NOISE_RE.match(text):
            logger.info(f"STT ({elapsed:.2f}s): filtered noise \"{text}\"")
            return "", elapsed

        logger.info(
            f"STT ({elapsed:.2f}s): \"{text}\" "
            f"[lang={info.language}, prob={info.language_probability:.2f}]"
        )
        return text, elapsed

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    from core.audio_input import AudioInput
    from core.vad import VAD

    cfg.setup_logging()
    print("🎙️  Testing STT — speak into your microphone")
    print(f"   Model:    faster-whisper ({cfg.stt.model_size})")
    print(f"   Language: {cfg.stt.language}")
    print(f"   Device:   cpu (CTranslate2 doesn't support MPS)")
    print(f"   Press Ctrl+C to stop")
    print()

    # Initialize
    stt = STT()
    stt.load()
    vad = VAD()

    utterance_count = 0

    try:
        with AudioInput() as mic:
            print("Listening...\n")
            for chunk in mic.read_chunks():
                utterance = vad.process_chunk(chunk)
                if utterance is not None:
                    utterance_count += 1
                    text, elapsed = stt.transcribe(utterance)
                    print(f"  [{utterance_count}] ({elapsed:.2f}s) → \"{text}\"")
                elif vad.is_speaking:
                    print("\r  🔴 Speaking...", end="", flush=True)
                else:
                    print("\r  ⚪ Listening  ", end="", flush=True)
    except KeyboardInterrupt:
        print(f"\n\nStopped. Transcribed {utterance_count} utterances.")
