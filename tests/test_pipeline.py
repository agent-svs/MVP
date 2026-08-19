"""Pipeline — turn isolation, barge-in wiring, action commit."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def pipeline(monkeypatch):
    """A VoicePipeline with every I/O and model boundary stubbed."""
    import core.pipeline as pipeline_mod

    for name in ("AudioInput", "AudioOutput", "AudioPreprocessor", "VAD", "STT", "TTS"):
        monkeypatch.setattr(pipeline_mod, name, MagicMock)
    monkeypatch.setattr(pipeline_mod, "LLM", MagicMock)

    p = pipeline_mod.VoicePipeline()
    p.audio_in, p.audio_out = MagicMock(), MagicMock()
    p.preprocessor, p.vad = MagicMock(), MagicMock()
    p.stt, p.tts, p.llm = MagicMock(), MagicMock(), MagicMock()
    p.dst = MagicMock()

    p.audio_in.read_chunk.return_value = np.zeros(480, dtype=np.float32)
    p.preprocessor.process_chunk.return_value = np.zeros(480, dtype=np.float32)
    p.preprocessor.process_utterance.return_value = np.zeros(16000, dtype=np.float32)
    p.vad.process_chunk.return_value = np.zeros(16000, dtype=np.float32)
    p.stt.transcribe.return_value = ("book an appointment", 0.3)
    p.tts.synthesize_stream.return_value = iter([(np.zeros(100, dtype=np.float32), 24000)])
    p.tts.synthesize.return_value = (np.zeros(100, dtype=np.float32), 24000, 0.1)
    p.audio_out.play.return_value = True
    return p


def _nlu(**over):
    base = {
        "intent": "book_appointment", "_active_intent": "book_appointment",
        "confidence": 0.9, "slots": {}, "state": "collecting",
        "missing_slots": ["date"], "turn_count": 1, "failed_turns": 0,
        "repair_message": None, "pending_slot": None,
        "completed_action": None, "declined_confirmation": False,
    }
    base.update(over)
    return base


# ── the crash regression ────────────────────────────────────
def test_one_bad_turn_does_not_end_the_call(pipeline):
    """An LLM error (a routine Gemini 503) propagated out of the loop and
    killed the process."""
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("503 UNAVAILABLE")
        pipeline._running = False

    pipeline._process_one_turn = boom
    pipeline._speak = MagicMock()

    pipeline.run()  # must return normally

    assert calls["n"] == 2, "the loop must continue after a failed turn"
    assert pipeline._speak.called


def test_repeated_failures_stop_the_loop(pipeline):
    pipeline._process_one_turn = MagicMock(side_effect=RuntimeError("boom"))
    pipeline._speak = MagicMock()
    pipeline.run()
    assert pipeline._process_one_turn.call_count == 5


def test_llm_failure_still_speaks_something(pipeline, monkeypatch):
    import core.pipeline as pipeline_mod
    from core.llm import LLMResponse

    pipeline.dst.process_turn.return_value = _nlu(
        intent="out_of_domain", _active_intent="out_of_domain", missing_slots=[]
    )
    pipeline.llm.generate.return_value = LLMResponse(
        "Sorry, I'm having trouble.", 0.2, ok=False, error="503"
    )
    spoken = []
    monkeypatch.setattr(pipeline_mod.VoicePipeline, "_speak_with_barge_in",
                        lambda self, t: spoken.append(t))

    pipeline._process_one_turn()

    assert spoken and spoken[0]


# ── action commit ───────────────────────────────────────────
def test_action_is_committed_exactly_once(pipeline, monkeypatch):
    import core.pipeline as pipeline_mod

    pipeline.dst.process_turn.return_value = _nlu(state="executing", missing_slots=[])
    monkeypatch.setattr(pipeline_mod.VoicePipeline, "_speak_with_barge_in", lambda self, t: None)

    pipeline._process_one_turn()

    pipeline.dst.mark_action_complete.assert_called_once_with("book_appointment")


def test_clarify_sets_the_expected_slot(pipeline, monkeypatch):
    import core.pipeline as pipeline_mod

    pipeline.dst.process_turn.return_value = _nlu(missing_slots=["date"])
    monkeypatch.setattr(pipeline_mod.VoicePipeline, "_speak_with_barge_in", lambda self, t: None)

    pipeline._process_one_turn()

    assert pipeline.dst.set_expectation.called
    assert pipeline.dst.set_expectation.call_args[0][0] == "date"


def test_empty_transcript_is_skipped(pipeline):
    pipeline.stt.transcribe.return_value = ("", 0.1)
    pipeline._process_one_turn()
    pipeline.dst.process_turn.assert_not_called()


# ── barge-in wiring ─────────────────────────────────────────
def test_monitor_is_gated_on_the_speaking_flag(pipeline):
    """The monitor used to loop on audio_out.is_playing, which was still False
    when the thread started — it exited at once and barge-in never ran."""
    pipeline.audio_in.read_chunk.return_value = None
    pipeline._speaking.set()

    t = threading.Thread(target=pipeline._barge_in_monitor, daemon=True)
    t.start()
    t.join(timeout=0.3)
    assert t.is_alive(), "monitor must stay alive while _speaking is set"

    pipeline._speaking.clear()
    t.join(timeout=1.0)
    assert not t.is_alive()


def test_speaking_mutes_and_then_restores_the_mic(pipeline):
    pipeline._speak_with_barge_in("Hello there.")
    assert pipeline.audio_in.mute.called
    assert pipeline.audio_in.flush.called
    assert pipeline.audio_in.unmute.called
    assert pipeline.vad.reset.called


def test_only_one_monitor_thread_at_a_time(pipeline):
    pipeline.audio_in.read_chunk.return_value = None
    pipeline._speaking.set()
    pipeline._start_monitor()
    first = pipeline._monitor
    pipeline._speaking.set()
    pipeline._start_monitor()
    assert pipeline._monitor is not first
    assert not first.is_alive(), "the previous monitor must be joined"
    pipeline._speaking.clear()
    pipeline._stop_monitor()


def test_stop_is_idempotent(pipeline):
    pipeline.stop()
    pipeline.stop()


# ── echo / barge-in interaction ─────────────────────────────
def test_monitor_is_not_starved_by_muting(pipeline):
    """Muting during playback stopped the mic callback enqueuing, so the
    barge-in monitor read from an empty queue and could never fire."""
    pipeline.tts.synthesize_stream.return_value = iter(
        [(np.zeros(100, dtype=np.float32), 24000)]
    )
    pipeline._speak_with_barge_in("Hello there.")

    mute_calls = [c for c in pipeline.audio_in.method_calls if c[0] == "mute"]
    # The only mute must come from _resume_listening (post-playback), never
    # before/during playback while the monitor needs the queue.
    assert pipeline.audio_in.flush.called
    assert len(mute_calls) <= 1


def test_resume_listening_drains_waits_then_unmutes(pipeline):
    pipeline._resume_listening()
    names = [c[0] for c in pipeline.audio_in.method_calls]
    assert names.index("mute") < names.index("unmute")
    assert names.count("flush") >= 2, "flush before AND after the cooldown"
    assert pipeline.vad.reset.called
    assert pipeline.preprocessor.reset_filter_state.called


def test_speak_mutes_before_synthesising(pipeline):
    """TTS takes seconds; audio captured during it is never the caller's turn.
    Muting after synthesis overflowed the queue at startup."""
    order = []
    pipeline.audio_in.mute.side_effect = lambda: order.append("mute")
    pipeline.tts.synthesize.side_effect = lambda *a, **k: (
        order.append("synthesize"), (np.zeros(10, dtype=np.float32), 24000, 0.1)
    )[1]
    pipeline._speak("Hello")
    assert order[:2] == ["mute", "synthesize"]
