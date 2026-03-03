"""
AI Voice Receptionist — Configuration Module

Loads all settings from .env file and exposes them as a typed Config dataclass.
Handles device auto-detection (CUDA > MPS > CPU).
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).parent
load_dotenv(_PROJECT_ROOT / ".env")


def _detect_device() -> str:
    """Auto-detect the best available compute device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_env(key: str, default: str = "") -> str:
    """Read an env var, stripping whitespace."""
    return os.getenv(key, default).strip()


def _get_env_float(key: str, default: float = 0.0) -> float:
    val = _get_env(key, str(default))
    try:
        return float(val)
    except ValueError:
        return default


def _get_env_int(key: str, default: int = 0) -> int:
    val = _get_env(key, str(default))
    try:
        return int(val)
    except ValueError:
        return default


@dataclass
class STTConfig:
    """Faster Whisper configuration."""
    model_size: str = "base"
    language: str = "en"
    beam_size: int = 5


@dataclass
class TTSConfig:
    """Coqui XTTS v2 configuration."""
    model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    speaker: str = "Daisy Studious"
    speaker_wav: str = "assets/reference_voice.wav"
    language: str = "en"


@dataclass
class LLMConfig:
    """Google Gemini API configuration."""
    api_key: str = ""
    model: str = "gemini-2.5-flash"
    max_tokens: int = 256
    temperature: float = 0.7


@dataclass
class AudioConfig:
    """Audio I/O configuration."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 30

    @property
    def chunk_samples(self) -> int:
        """Number of samples per chunk."""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 700


@dataclass
class NLUConfig:
    """NLU routing thresholds."""
    confidence_high: float = 0.85
    confidence_low: float = 0.60


@dataclass
class Config:
    """Master configuration — loads everything from .env."""
    device: str = "cpu"
    compute_type: str = "int8"
    log_level: str = "INFO"
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)

    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    nlu: NLUConfig = field(default_factory=NLUConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Build Config from environment variables."""
        # Device detection
        device_env = _get_env("DEVICE", "auto").lower()
        if device_env == "auto":
            device = _detect_device()
        else:
            device = device_env

        compute_type = _get_env("COMPUTE_TYPE", "int8")
        log_level = _get_env("LOG_LEVEL", "INFO").upper()

        # STT
        stt = STTConfig(
            model_size=_get_env("STT_MODEL_SIZE", "base"),
            language=_get_env("STT_LANGUAGE", "en"),
            beam_size=_get_env_int("STT_BEAM_SIZE", 5),
        )

        # TTS
        tts = TTSConfig(
            model=_get_env("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"),
            speaker=_get_env("TTS_SPEAKER", "Daisy Studious"),
            speaker_wav=_get_env("TTS_SPEAKER_WAV", "assets/reference_voice.wav"),
            language=_get_env("TTS_LANGUAGE", "en"),
        )

        # LLM
        llm = LLMConfig(
            api_key=_get_env("GEMINI_API_KEY", ""),
            model=_get_env("GEMINI_MODEL", "gemini-2.5-flash"),
            max_tokens=_get_env_int("GEMINI_MAX_TOKENS", 256),
            temperature=_get_env_float("GEMINI_TEMPERATURE", 0.7),
        )

        # Audio
        audio = AudioConfig(
            sample_rate=_get_env_int("AUDIO_SAMPLE_RATE", 16000),
            channels=_get_env_int("AUDIO_CHANNELS", 1),
            chunk_duration_ms=_get_env_int("AUDIO_CHUNK_DURATION_MS", 30),
        )

        # VAD
        vad = VADConfig(
            threshold=_get_env_float("VAD_THRESHOLD", 0.5),
            min_speech_ms=_get_env_int("VAD_MIN_SPEECH_MS", 250),
            min_silence_ms=_get_env_int("VAD_MIN_SILENCE_MS", 700),
        )

        # NLU
        nlu = NLUConfig(
            confidence_high=_get_env_float("NLU_CONFIDENCE_HIGH", 0.85),
            confidence_low=_get_env_float("NLU_CONFIDENCE_LOW", 0.60),
        )

        return cls(
            device=device,
            compute_type=compute_type,
            log_level=log_level,
            stt=stt,
            tts=tts,
            llm=llm,
            audio=audio,
            vad=vad,
            nlu=nlu,
        )

    def validate(self) -> list[str]:
        """Validate config, return list of warnings/errors."""
        issues = []
        if not self.llm.api_key or self.llm.api_key == "your-api-key-here":
            issues.append("GEMINI_API_KEY is not set — LLM features will not work")
        if self.device not in ("cpu", "mps", "cuda"):
            issues.append(f"Unknown device '{self.device}', falling back to cpu")
            self.device = "cpu"
        if self.stt.model_size not in ("tiny", "base", "small", "medium", "large-v3"):
            issues.append(f"Unknown STT model size '{self.stt.model_size}'")
        return issues

    def setup_logging(self) -> None:
        """Configure root logger based on LOG_LEVEL."""
        logging.basicConfig(
            level=getattr(logging, self.log_level, logging.INFO),
            format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )

    def summary(self) -> str:
        """Print a human-readable config summary."""
        return (
            f"\n{'='*60}\n"
            f"  AI Voice Receptionist — Configuration\n"
            f"{'='*60}\n"
            f"  Device:       {self.device}\n"
            f"  Compute Type: {self.compute_type}\n"
            f"  STT Model:    faster-whisper ({self.stt.model_size})\n"
            f"  TTS Model:    {self.tts.model}\n"
            f"  LLM Model:    {self.llm.model}\n"
            f"  Sample Rate:  {self.audio.sample_rate} Hz\n"
            f"  VAD Thresh:   {self.vad.threshold}\n"
            f"  Log Level:    {self.log_level}\n"
            f"{'='*60}\n"
        )


# Singleton instance — import and use: from config import cfg
cfg = Config.from_env()
