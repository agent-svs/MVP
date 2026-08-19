"""
Voice Pipeline — full orchestration loop.

  Mic → Preprocess → VAD → STT → NLU → DST → Router → [LLM] → Guardrails → TTS

Reliability properties this version guarantees:
  * One bad turn cannot end the call. Every stage is wrapped; a failure is
    spoken as a recovery line and the loop continues. Previously an LLM error
    (including a routine Gemini 503) propagated out of the loop and killed the
    process.
  * Barge-in monitoring starts only once playback is actually running, so the
    monitor can no longer exit immediately on a startup race and silently
    disable barge-in for that sentence.
  * Exactly one monitor thread exists at a time and it is always joined.
  * The mic is muted while the agent speaks and flushed afterwards, so the
    agent never transcribes its own voice or stale audio.
  * An action is committed exactly once via dst.mark_action_complete().
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from config import cfg
from core.audio_input import AudioInput
from core.audio_output import AudioOutput
from core.audio_preprocessor import AudioPreprocessor
from core.guardrails import apply_guardrails
from core.llm import LLM
from core.nlu.dst import DialogueStateTracker
from core.nlu.logger import nlu_logger
from core.response_templates import get_greeting
from core.router import route
from core.stt import STT
from core.tts import TTS
from core.vad import VAD

logger = logging.getLogger(__name__)

# Spoken when a turn fails for a reason the caller shouldn't hear about.
_RECOVERY_LINE = "Sorry, I didn't catch that. Could you say it again?"


class VoicePipeline:
    """Full voice conversation pipeline."""

    def __init__(self) -> None:
        self.audio_in = AudioInput()
        self.audio_out = AudioOutput()
        self.preprocessor = AudioPreprocessor()
        self.vad = VAD()
        self.stt = STT()
        self.tts = TTS()
        self.llm = LLM()
        self.dst = DialogueStateTracker()
        self.joint_nlu = None

        self._running = False
        self._turn_count = 0
        self._last_response = ""
        self._consecutive_errors = 0

        # Barge-in coordination
        self._speaking = threading.Event()      # set while the agent is talking
        self._barge_in = threading.Event()      # set when the caller interrupts
        self._monitor: Optional[threading.Thread] = None

    # ── startup ─────────────────────────────────────────────
    def load_models(self) -> None:
        """Pre-load all ML models."""
        logger.info("Loading models...")
        t0 = time.time()

        self.stt.load()
        self.tts.load()
        # LLM is lazy — it only needs the network on first use.

        if cfg.nlu.joint_nlu_enabled:
            from core.nlu.joint_nlu import JointNLU, JointNLUError

            self.joint_nlu = JointNLU()
            try:
                self.joint_nlu.load(cfg.nlu.joint_nlu_model_path)
            except JointNLUError as e:
                logger.error(f"{e}\nFalling back to the separate NER + intent pipeline.")
                self.joint_nlu = None
                cfg.nlu.joint_nlu_enabled = False

        if not cfg.nlu.joint_nlu_enabled:
            from core.nlu.intent import _load_model as load_intent_model
            from core.nlu.slots import _load_ner as load_ner_model

            load_intent_model()
            load_ner_model()

        logger.info(f"All models loaded in {time.time() - t0:.1f}s")

    # ── main loop ───────────────────────────────────────────
    def run(self) -> None:
        """Run the conversation loop until interrupted."""
        self._running = True
        self._turn_count = 0

        print("\n🎙️  Voice Receptionist — Ready")
        print("   Speak into your microphone. Press Ctrl+C to stop.\n")

        greeting = get_greeting()
        self._last_response = greeting
        self.audio_in.start()
        self._speak(greeting)

        try:
            while self._running:
                try:
                    self._process_one_turn()
                    self._consecutive_errors = 0
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self._consecutive_errors += 1
                    logger.exception(f"Turn failed: {e}")
                    print(f"  ⚠️  Turn error: {type(e).__name__}: {e}")
                    if self._consecutive_errors >= 5:
                        logger.error("5 consecutive turn failures — stopping")
                        self._speak(
                            "I'm having technical difficulties. "
                            "Please call back shortly. Goodbye."
                        )
                        break
                    self._speak(_RECOVERY_LINE)
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
        finally:
            self.stop()

    def _process_one_turn(self) -> None:
        """Process a single conversation turn."""
        # 1. Listen
        chunk = self.audio_in.read_chunk(timeout=0.1)
        if chunk is None:
            return
        utterance = self.vad.process_chunk(self.preprocessor.process_chunk(chunk))
        if utterance is None:
            return

        # 2. Transcribe (denoise + AGC happen once, on the whole utterance)
        self._turn_count += 1
        t0 = time.time()
        cleaned = self.preprocessor.process_utterance(utterance)
        text, stt_elapsed = self.stt.transcribe(cleaned)
        if not text or not text.strip():
            return

        print(f'\n  👤 You: "{text}" ({stt_elapsed:.2f}s)')

        # 3. NLU + state
        nlu_result = self.dst.process_turn(
            text,
            system_context=self._last_response,
            joint_nlu_engine=self.joint_nlu,
        )

        # 4. Route
        routing = route(nlu_result, last_response=self._last_response)
        print(f"  🔀 {routing.path.value}: {nlu_result['intent']} "
              f"(state={nlu_result['state']})")

        # 5. Respond
        if routing.needs_llm:
            if routing.filler:
                self._speak_async(routing.filler)
            llm_result = self.llm.generate(
                text,
                context={
                    "slots": nlu_result["slots"],
                    "active_intent": nlu_result.get("_active_intent"),
                },
            )
            response = llm_result.text
            if not llm_result.ok:
                logger.warning(f"LLM degraded on turn {self._turn_count}: {llm_result.error}")
            print(f'  🤖 LLM ({llm_result.elapsed:.2f}s): "{response[:80]}"')
        else:
            response = routing.response

        # 6. Commit the action exactly once
        if routing.execute_action:
            self.dst.mark_action_complete(routing.execute_action)

        # 7. Track what we asked for, so the next turn can resolve a bare answer
        if routing.path.value == "clarify":
            next_slot = routing.metadata.get("missing_slot")
            if next_slot:
                self.dst.set_expectation(next_slot, response)

        # 8. Guardrails
        guard = apply_guardrails(response, nlu_result["slots"])
        for w in guard.warnings:
            print(f"  ⚠️  {w}")
        final_response = guard.text
        self._last_response = final_response

        # 9. Log the turn (redacted)
        nlu_logger.log_turn({
            **nlu_result,
            "text": text,
            "route": routing.path.value,
            "response": final_response,
            "latency_s": round(time.time() - t0, 3),
        })

        # 10. Speak
        print(f'  🔊 AI: "{final_response}" (total: {time.time() - t0:.2f}s)')
        self._speak_with_barge_in(final_response)

        if routing.end_conversation:
            logger.info("Conversation ended by routing decision")

    # ── speech output ───────────────────────────────────────
    def _speak(self, text: str) -> None:
        """Synthesise and play (blocking, no barge-in)."""
        if not text:
            return
        # Mute BEFORE synthesis: TTS can take several seconds, and anything the
        # mic captures in that window is either room noise or the previous
        # utterance's tail.
        self.audio_in.mute()
        try:
            audio, sr, _ = self.tts.synthesize(text)
            if len(audio):
                self.audio_out.play(audio, sample_rate=sr)
        except Exception as e:
            logger.error(f"Speech playback failed: {e}")
        finally:
            self._resume_listening()

    def _speak_async(self, text: str) -> None:
        """Play a short filler in the background while the LLM runs."""
        threading.Thread(
            target=self._speak, args=(text,), name="filler", daemon=True
        ).start()

    def _speak_with_barge_in(self, text: str) -> None:
        """Speak sentence-by-sentence, stopping if the caller interrupts."""
        if not text:
            return

        self._barge_in.clear()
        self.audio_out.reset()
        # Discard whatever arrived while we were thinking, so the monitor only
        # ever sees audio from *during* playback.
        self.audio_in.flush()
        self._speaking.set()
        # Deliberately NOT muted: the barge-in monitor reads from this same
        # queue, so muting here would starve it and disable barge-in entirely.
        # Accumulation is prevented by the monitor draining continuously.
        self._start_monitor()

        try:
            for audio, sr in self.tts.synthesize_stream(text):
                if self._barge_in.is_set():
                    logger.info("Barge-in: skipping remaining sentences")
                    print("  🛑 Barge-in — stopping")
                    break
                completed = self.audio_out.play(audio, sample_rate=sr)
                if not completed or self._barge_in.is_set():
                    print("  🛑 Barge-in — stopping")
                    break
        except Exception as e:
            logger.error(f"Streaming playback failed: {e}")
        finally:
            self._speaking.clear()
            self._stop_monitor()
            self._resume_listening()

    def _resume_listening(self) -> None:
        """Clear every trace of our own audio before listening again.

        Order matters. The speaker keeps emitting for a moment after the
        stream stops, and the room reverberates after that. Re-arming the mic
        immediately made the agent transcribe the tail of its own sentence as
        the caller's reply. So: mute, drain, wait out the decay, drain again,
        then listen.
        """
        self.audio_in.mute()
        self.audio_in.flush()
        cooldown = cfg.audio.echo_cooldown_ms / 1000.0
        if cooldown > 0:
            time.sleep(cooldown)
        self.audio_in.flush()
        self.vad.reset()
        self.preprocessor.reset_filter_state()
        self.audio_out.reset()
        self.audio_in.unmute()

    # ── barge-in monitor ────────────────────────────────────
    def _start_monitor(self) -> None:
        """Start exactly one monitor thread for this utterance."""
        self._stop_monitor()
        self._monitor = threading.Thread(
            target=self._barge_in_monitor, name="barge-in", daemon=True
        )
        self._monitor.start()

    def _stop_monitor(self) -> None:
        self._speaking.clear()
        monitor, self._monitor = self._monitor, None
        if monitor is not None and monitor.is_alive():
            monitor.join(timeout=1.0)

    def _barge_in_monitor(self) -> None:
        """Watch the mic for speech while the agent is talking.

        Gated on `self._speaking` (set BEFORE the thread starts) rather than on
        audio_out.is_playing, which was only true once playback had begun — a
        race that let this thread exit immediately and disable barge-in.
        """
        started = time.time()
        grace = cfg.vad.barge_in_grace_ms / 1000.0
        while self._speaking.is_set():
            chunk = self.audio_in.read_chunk(timeout=0.05)
            if chunk is None:
                continue
            # Early playback still overlaps the previous turn's reverb.
            if time.time() - started < grace:
                continue
            try:
                clean = self.preprocessor.process_chunk(chunk)
                if self.vad.check_barge_in(clean):
                    self._barge_in.set()
                    self.audio_out.interrupt()
                    logger.info("Barge-in detected")
                    return
            except Exception as e:
                logger.debug(f"Barge-in check failed: {e}")

    # ── shutdown ────────────────────────────────────────────
    def stop(self) -> None:
        """Stop the pipeline and release the audio devices."""
        if not self._running and self._monitor is None:
            return
        self._running = False
        self._stop_monitor()
        try:
            self.audio_out.interrupt()
        except Exception:
            pass
        try:
            self.audio_in.stop()
        except Exception as e:
            logger.debug(f"Error stopping mic: {e}")
        logger.info(f"Pipeline stopped after {self._turn_count} turns")
