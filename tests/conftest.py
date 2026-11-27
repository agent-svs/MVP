"""Shared fixtures.

Also guarantees the project root is importable regardless of how pytest is
invoked — previously this relied on rootdir inference, which broke when pytest
walked up and adopted ~/pyproject.toml as its config.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Tests must never write to the real conversation log.
os.environ.setdefault("NLU_LOG_ENABLED", "false")


@pytest.fixture(autouse=True)
def _isolate_logs(tmp_path, monkeypatch):
    """Redirect any turn logging into a temp dir."""
    from core.nlu import logger as logger_mod

    monkeypatch.setattr(logger_mod.nlu_logger, "_log_file", tmp_path / "nlu.jsonl")
    yield


@pytest.fixture
def legacy_nlu(monkeypatch):
    """Force the separate NER + zero-shot path (JOINT_NLU_ENABLED=false)."""
    from config import cfg

    monkeypatch.setattr(cfg.nlu, "joint_nlu_enabled", False)
    yield cfg


@pytest.fixture(scope="session")
def joint_engine():
    """The real joint NLU model. Skips if the weights aren't available."""
    from config import cfg
    from core.nlu.joint_nlu import JointNLU, JointNLUError

    engine = JointNLU()
    try:
        engine.load(cfg.nlu.joint_nlu_model_path)
    except (JointNLUError, Exception) as e:  # noqa: BLE001 - any load failure skips
        pytest.skip(f"Joint NLU model unavailable: {e}")
    return engine


@pytest.fixture
def joint_dst(monkeypatch):
    """A DST wired to the joint path with a scripted fake engine."""
    from config import cfg
    from core.nlu.dst import DialogueStateTracker

    monkeypatch.setattr(cfg.nlu, "joint_nlu_enabled", True)

    class FakeEngine:
        """Returns whatever the test queued, in order."""

        def __init__(self) -> None:
            self.script: list[dict] = []

        def queue(self, intent, confidence=0.95, slots=None):
            self.script.append({
                "intent": intent,
                "intent_confidence": confidence,
                "slots": slots or {},
                "raw_nlu_intent": intent,
            })
            return self

        def predict(self, context, utterance):
            if self.script:
                return self.script.pop(0)
            return {"intent": None, "intent_confidence": 0.9,
                    "slots": {}, "raw_nlu_intent": "slot_answer"}

    return DialogueStateTracker(), FakeEngine()


def slot(value, confidence: float = 0.95) -> dict:
    """Build a slot payload in the shape the NLU emits."""
    return {"value": value, "confidence": confidence}
