"""
NLU turn logger.

Writes one JSON object per turn for offline analysis and threshold tuning.

Privacy: conversation content is caller PII. This logger therefore
  * redacts names, phone numbers and free text by default
    (NLU_LOG_REDACT_PII=true),
  * can be switched off entirely (NLU_LOG_ENABLED=false),
  * rotates at a size cap so it cannot grow without bound,
  * is thread-safe.

The log directory is created lazily on first write rather than at import time,
so importing this module no longer has a filesystem side effect.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from config import cfg

logger = logging.getLogger(__name__)

# Slots whose values are caller PII.
_PII_SLOTS = {"patient_name", "phone", "email", "address", "dob", "reason"}
# Fields that carry raw conversation text.
_TEXT_FIELDS = {"text", "response", "repair_message"}

_DIGITS_RE = re.compile(r"\d")


def _pseudonymise(value: Any) -> str:
    """Stable, non-reversible token so a caller can still be followed across
    turns without storing who they are."""
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:10]
    return f"<redacted:{digest}>"


def _redact_text(text: Any) -> str:
    """Keep the shape of an utterance, drop its content."""
    s = str(text or "")
    words = len(s.split())
    digits = len(_DIGITS_RE.findall(s))
    return f"<text: {words}w{f' {digits}d' if digits else ''}>"


class NLULogger:
    """Structured, size-capped, optionally-redacted turn logging."""

    def __init__(self, log_file: Path | None = None) -> None:
        self._log_file = log_file
        self._lock = threading.Lock()

    @property
    def log_file(self) -> Path:
        if self._log_file is None:
            self._log_file = cfg.project_root / "logs" / "nlu_data.jsonl"
        return self._log_file

    # ── redaction ───────────────────────────────────────────
    def _redact(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in turn.items():
            if key in _TEXT_FIELDS:
                out[key] = _redact_text(value)
            elif key in ("slots", "slot_meta"):
                out[key] = {
                    k: (_pseudonymise(v) if k in _PII_SLOTS else v)
                    for k, v in (value or {}).items()
                }
            else:
                out[key] = value
        return out

    # ── rotation ────────────────────────────────────────────
    def _rotate_if_needed(self, path: Path) -> None:
        try:
            if not path.exists() or path.stat().st_size < cfg.logging_cfg.nlu_log_max_bytes:
                return
            backups = cfg.logging_cfg.nlu_log_backups
            oldest = path.with_suffix(path.suffix + f".{backups}")
            if oldest.exists():
                oldest.unlink()
            for i in range(backups - 1, 0, -1):
                src = path.with_suffix(path.suffix + f".{i}")
                if src.exists():
                    src.rename(path.with_suffix(path.suffix + f".{i + 1}"))
            path.rename(path.with_suffix(path.suffix + ".1"))
        except Exception as e:
            logger.debug(f"Log rotation failed: {e}")

    # ── public API ──────────────────────────────────────────
    def log_turn(self, turn_data: Dict[str, Any]) -> None:
        """Append one turn record. Never raises."""
        if not cfg.logging_cfg.nlu_log_enabled:
            return
        try:
            payload = dict(turn_data)
            # slot_meta holds raw spans and canonical ids — too detailed to log.
            payload.pop("slot_meta", None)
            if cfg.logging_cfg.nlu_log_redact_pii:
                payload = self._redact(payload)

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "redacted": cfg.logging_cfg.nlu_log_redact_pii,
                **payload,
            }

            path = self.log_file
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed(path)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log NLU turn: {e}")


nlu_logger = NLULogger()
