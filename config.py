"""
AI Voice Receptionist — Configuration Module

Single source of truth for every tunable in the system. Loads from .env and
exposes a typed Config dataclass.

Design rules enforced here:
  * ONE device policy (torch device, HF pipeline device, CTranslate2 device)
    derived from DEVICE — modules must not pick their own.
  * ONE business-hours definition — guardrails, slot extraction and the LLM
    system prompt all read it from here.
  * Every magic threshold that used to be hardcoded (STT energy gate, VAD
    chunk size, LLM timeout) is configurable and validated.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# MPS (Apple Silicon GPU) doesn't support conv1d with >65536 output channels.
# Must be set BEFORE torch is ever imported anywhere in the project.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Coqui XTTS prompts for CPML licence agreement on stdin the first time it
# loads a model. In a voice loop / container that is an unrecoverable hang, so
# accept it up front. (Set COQUI_TOS_AGREED=0 in the env to opt out.)
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# Keep tokenizers from forking threads under our own thread usage.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_PROJECT_ROOT = Path(__file__).parent

try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - dotenv is a hard dependency in practice
    pass


# ── env helpers ─────────────────────────────────────────────
def _get_env(key: str, default: str = "") -> str:
    """Read an env var, stripping whitespace and inline comments."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip()


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


def _get_env_bool(key: str, default: bool = False) -> bool:
    val = _get_env(key, "true" if default else "false").lower()
    return val in ("1", "true", "yes", "on")


def _detect_device() -> str:
    """Auto-detect the best available compute device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # torch missing or backend probe failed
        pass
    return "cpu"


# ── Component configs ───────────────────────────────────────
@dataclass
class STTConfig:
    """Faster Whisper configuration."""

    model_size: str = "base"
    language: str = "en"
    beam_size: int = 5

    # Energy / duration gates that decide whether an utterance is worth
    # transcribing at all. These used to be hardcoded at 0.05 / 0.8s, which
    # silently discarded every short answer ("yes", "3 pm", "Monday") — i.e.
    # every reply to a clarification question.
    min_amplitude: float = 0.015
    min_duration_s: float = 0.25
    # Below this peak amplitude a canned Whisper phrase ("thank you", "bye")
    # is treated as a silence hallucination rather than real speech.
    hallucination_amplitude: float = 0.06


@dataclass
class TTSConfig:
    """Coqui XTTS v2 configuration."""

    model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    speaker: str = "Daisy Studious"
    speaker_wav: str = "assets/reference_voice.wav"
    language: str = "en"
    # Only cache phrases that contain no caller data. Caching personalised
    # responses writes patient names to disk as audio, forever, for a ~0% hit
    # rate. See core/tts.py::_is_cacheable.
    cache_enabled: bool = True
    cache_max_files: int = 256


@dataclass
class LLMConfig:
    """Google Gemini API configuration."""

    api_key: str = ""
    model: str = "gemini-2.5-flash"
    max_tokens: int = 256
    temperature: float = 0.7
    # Hard ceiling on a single request. Without this a hung connection blocks
    # the voice loop forever and the caller hears silence.
    timeout_s: float = 12.0
    max_retries: int = 3
    retry_base_delay_s: float = 1.0
    # gemini-2.5-* enable "thinking" by default, which both adds latency and
    # consumes max_output_tokens. 0 disables it for supported models.
    thinking_budget: int = 0


@dataclass
class AudioConfig:
    """Audio I/O configuration."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 30
    # Bounded mic queue. Unbounded, a slow turn (LLM + TTS) lets seconds of
    # stale audio pile up and the agent then transcribes its own voice.
    max_queue_chunks: int = 100
    # Silence after the agent stops speaking, so the speaker tail and room
    # reverb decay before the mic is re-armed. Without it the agent
    # transcribes the end of its own sentence as the caller's reply.
    echo_cooldown_ms: int = 350

    @property
    def chunk_samples(self) -> int:
        """Number of samples per chunk."""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""

    threshold: float = 0.5
    min_speech_ms: int = 200
    min_silence_ms: int = 600
    # Barge-in must clear a HIGHER bar than normal speech detection, because
    # the mic hears the agent's own speakers. Without an energy floor the
    # agent interrupts itself on its own output.
    barge_in_threshold: float = 0.85
    barge_in_min_amplitude: float = 0.12
    barge_in_consecutive: int = 4
    # Ignore the first moments of playback, where the previous turn's room
    # reverb is still decaying.
    barge_in_grace_ms: int = 400
    # Silero requires an exact window: 512 samples @16k, 256 @8k.
    @staticmethod
    def chunk_size_for(sample_rate: int) -> int:
        return 512 if sample_rate >= 16000 else 256


@dataclass
class PreprocessConfig:
    """Audio preprocessing.

    Chunk-level work must stay cheap and must NOT alter relative loudness,
    because VAD and the STT energy gate both depend on it. Noise reduction and
    AGC therefore run once per complete utterance instead of per 30ms chunk.
    """

    highpass_hz: int = 80
    highpass_order: int = 4
    denoise_enabled: bool = True
    denoise_strength: float = 0.75
    agc_enabled: bool = True
    agc_target_rms: float = 0.08
    agc_max_gain: float = 8.0


@dataclass
class NLUConfig:
    """NLU model selection and routing thresholds."""

    confidence_high: float = 0.80
    confidence_low: float = 0.50
    joint_nlu_enabled: bool = True
    joint_nlu_model_path: str = "models/joint_nlu/"
    joint_nlu_hub_id: str = "agents-svs/voice-receptionist-joint-nlu"
    # Legacy separate-model path (zero-shot or fine-tuned classifier).
    intent_fine_tuned: bool = False
    intent_model: str = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
    ner_model: str = "dslim/bert-base-NER"
    # Minimum slot confidence to accept an extracted slot value.
    slot_confidence_threshold: float = 0.75


@dataclass
class BusinessConfig:
    """Clinic business rules — the ONE definition used everywhere.

    Previously duplicated across slots.py (8-20), guardrails.py (8-18) and the
    LLM system prompt (8am-6pm), which had already drifted apart.
    """

    name: str = "Apex Medical Clinic"
    address: str = "123 Healthway Drive, Suite 100"
    hours_start: int = 8  # 8 AM
    hours_end: int = 18  # 6 PM
    # 0=Mon .. 6=Sun
    open_days: tuple[int, ...] = (0, 1, 2, 3, 4)
    max_booking_days_ahead: int = 180

    @property
    def hours_human(self) -> str:
        def fmt(h: int) -> str:
            suffix = "AM" if h < 12 else "PM"
            hour = h if 1 <= h <= 12 else (h - 12 if h > 12 else 12)
            return f"{hour}:00 {suffix}"

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if self.open_days == (0, 1, 2, 3, 4):
            days = "Monday to Friday"
        else:
            days = ", ".join(day_names[d] for d in self.open_days)
        return f"{days}, {fmt(self.hours_start)} to {fmt(self.hours_end)}"


@dataclass
class LoggingConfig:
    """Conversation logging + PII handling."""

    level: str = "INFO"
    # Turn-by-turn NLU logging to logs/nlu_data.jsonl. Contains conversation
    # content, so it is redacted by default and can be switched off entirely.
    nlu_log_enabled: bool = True
    nlu_log_redact_pii: bool = True
    nlu_log_max_bytes: int = 5 * 1024 * 1024
    nlu_log_backups: int = 3


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
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    nlu: NLUConfig = field(default_factory=NLUConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)
    logging_cfg: LoggingConfig = field(default_factory=LoggingConfig)

    # ── Device policy (single source of truth) ──────────────
    @property
    def torch_device(self) -> str:
        """Device string for torch.device() — used by VAD and the joint NLU."""
        return self.device

    @property
    def hf_pipeline_device(self) -> int | str:
        """Device argument for a transformers pipeline().

        transformers wants -1 for CPU, a CUDA ordinal, or the literal "mps".
        """
        if self.device == "cuda":
            return 0
        if self.device == "mps":
            return "mps"
        return -1

    @property
    def stt_device(self) -> tuple[str, str]:
        """(device, compute_type) for CTranslate2, which has no MPS support."""
        if self.device == "cuda":
            return "cuda", (self.compute_type if self.compute_type != "int8" else "float16")
        return "cpu", "int8"

    @classmethod
    def from_env(cls) -> "Config":
        """Build Config from environment variables."""
        device_env = _get_env("DEVICE", "auto").lower()
        device = _detect_device() if device_env == "auto" else device_env

        stt = STTConfig(
            model_size=_get_env("STT_MODEL_SIZE", "base"),
            language=_get_env("STT_LANGUAGE", "en"),
            beam_size=_get_env_int("STT_BEAM_SIZE", 5),
            min_amplitude=_get_env_float("STT_MIN_AMPLITUDE", 0.015),
            min_duration_s=_get_env_float("STT_MIN_DURATION_S", 0.25),
            hallucination_amplitude=_get_env_float("STT_HALLUCINATION_AMPLITUDE", 0.06),
        )

        tts = TTSConfig(
            model=_get_env("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"),
            speaker=_get_env("TTS_SPEAKER", "Daisy Studious"),
            speaker_wav=_get_env("TTS_SPEAKER_WAV", "assets/reference_voice.wav"),
            language=_get_env("TTS_LANGUAGE", "en"),
            cache_enabled=_get_env_bool("TTS_CACHE_ENABLED", True),
            cache_max_files=_get_env_int("TTS_CACHE_MAX_FILES", 256),
        )

        llm = LLMConfig(
            api_key=_get_env("GEMINI_API_KEY", ""),
            model=_get_env("GEMINI_MODEL", "gemini-2.5-flash"),
            max_tokens=_get_env_int("GEMINI_MAX_TOKENS", 256),
            temperature=_get_env_float("GEMINI_TEMPERATURE", 0.7),
            timeout_s=_get_env_float("GEMINI_TIMEOUT_S", 12.0),
            max_retries=_get_env_int("GEMINI_MAX_RETRIES", 3),
            retry_base_delay_s=_get_env_float("GEMINI_RETRY_BASE_DELAY_S", 1.0),
            thinking_budget=_get_env_int("GEMINI_THINKING_BUDGET", 0),
        )

        audio = AudioConfig(
            sample_rate=_get_env_int("AUDIO_SAMPLE_RATE", 16000),
            channels=_get_env_int("AUDIO_CHANNELS", 1),
            chunk_duration_ms=_get_env_int("AUDIO_CHUNK_DURATION_MS", 30),
            max_queue_chunks=_get_env_int("AUDIO_MAX_QUEUE_CHUNKS", 100),
            echo_cooldown_ms=_get_env_int("AUDIO_ECHO_COOLDOWN_MS", 350),
        )

        vad = VADConfig(
            threshold=_get_env_float("VAD_THRESHOLD", 0.5),
            min_speech_ms=_get_env_int("VAD_MIN_SPEECH_MS", 200),
            min_silence_ms=_get_env_int("VAD_MIN_SILENCE_MS", 600),
            barge_in_threshold=_get_env_float("VAD_BARGE_IN_THRESHOLD", 0.85),
            barge_in_min_amplitude=_get_env_float("VAD_BARGE_IN_MIN_AMPLITUDE", 0.12),
            barge_in_consecutive=_get_env_int("VAD_BARGE_IN_CONSECUTIVE", 4),
            barge_in_grace_ms=_get_env_int("VAD_BARGE_IN_GRACE_MS", 400),
        )

        preprocess = PreprocessConfig(
            highpass_hz=_get_env_int("PREPROCESS_HIGHPASS_HZ", 80),
            highpass_order=_get_env_int("PREPROCESS_HIGHPASS_ORDER", 4),
            denoise_enabled=_get_env_bool("PREPROCESS_DENOISE", True),
            denoise_strength=_get_env_float("PREPROCESS_DENOISE_STRENGTH", 0.75),
            agc_enabled=_get_env_bool("PREPROCESS_AGC", True),
            agc_target_rms=_get_env_float("PREPROCESS_AGC_TARGET_RMS", 0.08),
            agc_max_gain=_get_env_float("PREPROCESS_AGC_MAX_GAIN", 8.0),
        )

        nlu = NLUConfig(
            confidence_high=_get_env_float("NLU_CONFIDENCE_HIGH", 0.80),
            confidence_low=_get_env_float("NLU_CONFIDENCE_LOW", 0.50),
            joint_nlu_enabled=_get_env_bool("JOINT_NLU_ENABLED", True),
            joint_nlu_model_path=_get_env("JOINT_NLU_MODEL_PATH", "models/joint_nlu/"),
            joint_nlu_hub_id=_get_env("JOINT_NLU_HUB_ID", "agents-svs/voice-receptionist-joint-nlu"),
            intent_fine_tuned=_get_env_bool("INTENT_FINE_TUNED", False),
            intent_model=_get_env(
                "INTENT_MODEL", "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
            ),
            ner_model=_get_env("NER_MODEL", "dslim/bert-base-NER"),
            slot_confidence_threshold=_get_env_float("NLU_SLOT_CONFIDENCE", 0.75),
        )

        business = BusinessConfig(
            name=_get_env("BUSINESS_NAME", "Apex Medical Clinic"),
            address=_get_env("BUSINESS_ADDRESS", "123 Healthway Drive, Suite 100"),
            hours_start=_get_env_int("BUSINESS_HOURS_START", 8),
            hours_end=_get_env_int("BUSINESS_HOURS_END", 18),
            max_booking_days_ahead=_get_env_int("BUSINESS_MAX_DAYS_AHEAD", 180),
        )

        log_level = _get_env("LOG_LEVEL", "INFO").upper()
        logging_cfg = LoggingConfig(
            level=log_level,
            nlu_log_enabled=_get_env_bool("NLU_LOG_ENABLED", True),
            nlu_log_redact_pii=_get_env_bool("NLU_LOG_REDACT_PII", True),
            nlu_log_max_bytes=_get_env_int("NLU_LOG_MAX_BYTES", 5 * 1024 * 1024),
            nlu_log_backups=_get_env_int("NLU_LOG_BACKUPS", 3),
        )

        return cls(
            device=device,
            compute_type=_get_env("COMPUTE_TYPE", "int8"),
            log_level=log_level,
            stt=stt,
            tts=tts,
            llm=llm,
            audio=audio,
            vad=vad,
            preprocess=preprocess,
            nlu=nlu,
            business=business,
            logging_cfg=logging_cfg,
        )

    # ── Validation ──────────────────────────────────────────
    def validate(self) -> list[str]:
        """Validate config; returns a list of human-readable issues.

        Mutates self to a safe value where a sane fallback exists.
        """
        issues: list[str] = []

        if not self.llm.api_key or self.llm.api_key == "your-api-key-here":
            issues.append(
                "GEMINI_API_KEY is not set — open-ended questions will fall back to a "
                "canned reply instead of the LLM"
            )
        if self.device not in ("cpu", "mps", "cuda"):
            issues.append(f"Unknown device '{self.device}', falling back to cpu")
            self.device = "cpu"
        if self.stt.model_size not in (
            "tiny", "base", "small", "medium", "large-v2", "large-v3",
            "distil-small.en", "distil-medium.en", "distil-large-v3",
        ):
            issues.append(f"Unknown STT model size '{self.stt.model_size}'")
        if self.audio.sample_rate not in (8000, 16000):
            issues.append(
                f"AUDIO_SAMPLE_RATE={self.audio.sample_rate} is unsupported by Silero VAD "
                "(needs 8000 or 16000) — forcing 16000"
            )
            self.audio.sample_rate = 16000
        if self.vad.min_speech_ms > 500:
            issues.append(
                f"VAD_MIN_SPEECH_MS={self.vad.min_speech_ms} will discard short answers "
                "like 'yes' or '3 pm' — 200-300 recommended"
            )
        if self.stt.min_duration_s > 0.5:
            issues.append(
                f"STT_MIN_DURATION_S={self.stt.min_duration_s} will discard short answers "
                "— 0.25 recommended"
            )
        if not 0.0 < self.vad.threshold < 1.0:
            issues.append(f"VAD_THRESHOLD={self.vad.threshold} out of range — forcing 0.5")
            self.vad.threshold = 0.5
        if self.business.hours_start >= self.business.hours_end:
            issues.append("BUSINESS_HOURS_START must be before BUSINESS_HOURS_END — forcing 8-18")
            self.business.hours_start, self.business.hours_end = 8, 18
        if self.llm.timeout_s <= 0:
            issues.append("GEMINI_TIMEOUT_S must be > 0 — forcing 12")
            self.llm.timeout_s = 12.0
        return issues

    def setup_logging(self) -> None:
        """Configure the root logger based on LOG_LEVEL."""
        logging.basicConfig(
            level=getattr(logging, self.log_level, logging.INFO),
            format="%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )
        # These are chatty at INFO and drown out our own turn logs.
        for noisy in ("urllib3", "httpx", "httpcore", "numba", "TTS", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    def summary(self) -> str:
        """Human-readable config summary."""
        stt_dev, stt_ct = self.stt_device
        return (
            f"\n{'='*62}\n"
            f"  {self.business.name} — Voice Receptionist\n"
            f"{'='*62}\n"
            f"  Device:       {self.device}\n"
            f"  Compute Type: {self.compute_type}\n"
            f"  STT Model:    faster-whisper ({self.stt.model_size}) on {stt_dev}/{stt_ct}\n"
            f"  TTS Model:    {self.tts.model}\n"
            f"  LLM Model:    {self.llm.model} (timeout {self.llm.timeout_s}s)\n"
            f"  NLU:          {'Joint DeBERTa' if self.nlu.joint_nlu_enabled else 'zero-shot + NER'}\n"
            f"  Sample Rate:  {self.audio.sample_rate} Hz\n"
            f"  VAD:          thresh={self.vad.threshold} "
            f"min_speech={self.vad.min_speech_ms}ms min_silence={self.vad.min_silence_ms}ms\n"
            f"  Hours:        {self.business.hours_human}\n"
            f"  Log Level:    {self.log_level}\n"
            f"{'='*62}\n"
        )


# Singleton instance — import and use: from config import cfg
cfg = Config.from_env()
