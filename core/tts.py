"""
Text-to-Speech — Coqui XTTS v2 Wrapper

Converts text to natural speech using XTTS v2.
Supports both built-in speakers (58 voices) and voice cloning from a WAV file.
Includes audio phrase caching for common responses (< 5ms for cached phrases).

Configurable via .env:
  - TTS_MODEL: model identifier
  - TTS_SPEAKER: built-in speaker name (e.g. "Daisy Studious")
  - TTS_SPEAKER_WAV: reference voice WAV for cloning (overrides TTS_SPEAKER)
  - TTS_LANGUAGE: output language
  - DEVICE: cpu / mps / cuda
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import cfg

logger = logging.getLogger(__name__)

# Default built-in speaker when no speaker_wav is provided
DEFAULT_SPEAKER = "Daisy Studious"


class TTS:
    """
    Text-to-Speech using Coqui XTTS v2.

    Two modes:
    1. Built-in speaker: Set TTS_SPEAKER in .env (no WAV needed)
    2. Voice cloning: Set TTS_SPEAKER_WAV to a 6-15s reference WAV

    If TTS_SPEAKER_WAV exists, voice cloning is used. Otherwise, falls back
    to the built-in speaker specified by TTS_SPEAKER.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        speaker: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or cfg.tts.model
        self.language = language or cfg.tts.language
        self.device = device or cfg.device
        self.speaker = speaker or cfg.tts.speaker

        # Resolve speaker wav path relative to project root
        raw_speaker_wav = speaker_wav or cfg.tts.speaker_wav
        if raw_speaker_wav and not os.path.isabs(raw_speaker_wav):
            resolved = str(cfg.project_root / raw_speaker_wav)
        else:
            resolved = raw_speaker_wav or ""

        # Only use speaker_wav if the file actually exists
        self.speaker_wav = resolved if resolved and os.path.exists(resolved) else None

        # Cache directory
        self._cache_dir = cfg.project_root / "assets" / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Model (lazy loaded)
        self._model = None
        self._sample_rate: int = 24000  # XTTS v2 default output rate

    def load(self) -> None:
        """Load the XTTS v2 model. Call once at startup."""
        from TTS.api import TTS as CoquiTTS

        logger.info(f"Loading TTS model: {self.model_name} (device={self.device})")
        t0 = time.time()

        self._model = CoquiTTS(
            model_name=self.model_name,
            progress_bar=False,
        )
        self._model.to(self.device)

        elapsed = time.time() - t0

        # Log which voice mode we're using
        if self.speaker_wav:
            logger.info(f"TTS loaded in {elapsed:.1f}s — voice cloning from {self.speaker_wav}")
        else:
            logger.info(f"TTS loaded in {elapsed:.1f}s — using built-in speaker: {self.speaker}")

    def _cache_key(self, text: str) -> str:
        """Generate a cache key from text + voice config."""
        voice_id = self.speaker_wav or self.speaker
        return hashlib.md5(
            f"{text}:{self.language}:{voice_id}".encode()
        ).hexdigest()

    def _check_cache(self, text: str) -> Optional[np.ndarray]:
        """Check if synthesized audio is cached."""
        key = self._cache_key(text)
        cache_path = self._cache_dir / f"{key}.npy"
        if cache_path.exists():
            logger.debug(f"TTS cache hit: '{text[:40]}...'")
            return np.load(str(cache_path))
        return None

    def _save_cache(self, text: str, audio: np.ndarray) -> None:
        """Save synthesized audio to cache."""
        key = self._cache_key(text)
        cache_path = self._cache_dir / f"{key}.npy"
        np.save(str(cache_path), audio)
        logger.debug(f"TTS cached: '{text[:40]}...'")

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, int, float]:
        """
        Convert text to speech audio.

        Args:
            text: Text to synthesize
            language: Override language
            use_cache: Whether to use/save phrase cache

        Returns:
            Tuple of (audio_array, sample_rate, elapsed_seconds)
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32), self._sample_rate, 0.0

        # Check cache first
        if use_cache:
            cached = self._check_cache(text)
            if cached is not None:
                return cached, self._sample_rate, 0.0

        if self._model is None:
            self.load()

        lang = language or self.language
        t0 = time.time()

        # Use speaker_wav for voice cloning, or built-in speaker name
        if self.speaker_wav:
            audio_list = self._model.tts(
                text=text,
                language=lang,
                speaker_wav=self.speaker_wav,
            )
        else:
            audio_list = self._model.tts(
                text=text,
                language=lang,
                speaker=self.speaker,
            )

        elapsed = time.time() - t0
        audio = np.array(audio_list, dtype=np.float32)

        logger.info(
            f"TTS ({elapsed:.2f}s): \"{text[:50]}\" "
            f"[{len(audio)} samples, {len(audio)/self._sample_rate:.1f}s audio]"
        )

        if use_cache:
            self._save_cache(text, audio)

        return audio, self._sample_rate, elapsed

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def clear_cache(self) -> int:
        """Clear all cached audio files. Returns number of files removed."""
        count = 0
        for f in self._cache_dir.glob("*.npy"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} cached TTS files")
        return count


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    from core.audio_output import AudioOutput

    cfg.setup_logging()
    print("🔊 Testing TTS — Coqui XTTS v2")
    print(f"   Model:    {cfg.tts.model}")
    print(f"   Speaker:  {cfg.tts.speaker}")
    print(f"   Language: {cfg.tts.language}")
    print(f"   Device:   {cfg.device}")
    print()

    tts = TTS()
    tts.load()

    player = AudioOutput()

    test_phrases = [
        "Hello Swayam Vyas! Welcome to our office. How can I help you today?",
        "Your appointment is confirmed for tomorrow at 3 PM.",
        "Let me check that for you. One moment please.",
    ]

    for i, phrase in enumerate(test_phrases, 1):
        print(f"\n  [{i}] Synthesizing: \"{phrase}\"")
        audio, sr, elapsed = tts.synthesize(phrase)
        print(f"      Generated {len(audio)/sr:.1f}s audio in {elapsed:.2f}s")
        print(f"      Playing...")
        player.play(audio, sample_rate=sr)

    # Test cache hit
    print(f"\n  [Cache test] Re-synthesizing first phrase...")
    audio, sr, elapsed = tts.synthesize(test_phrases[0])
    print(f"      Cache hit! Elapsed: {elapsed:.4f}s")
    player.play(audio, sample_rate=sr)

    print("\n✅ TTS test complete")
