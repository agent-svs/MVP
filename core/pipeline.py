"""
Voice Pipeline — Full Orchestration Loop (v2)

Wires all layers together:
  Mic → Preprocessor → VAD → STT → NLU → Router → [LLM] → Guardrails → TTS → Speaker

Features:
- Audio preprocessing (noise suppression, HPF, AGC)
- Filler injection for LLM latency masking
- Barge-in support — monitors mic during TTS playback
- Streaming TTS — plays sentence-by-sentence
- Graceful error handling per turn
"""

import logging
import threading
import time
from typing import Optional

from config import cfg
from core.audio_input import AudioInput
from core.audio_output import AudioOutput
from core.audio_preprocessor import AudioPreprocessor
from core.vad import VAD
from core.stt import STT
from core.tts import TTS
from core.llm import LLM
from core.nlu.dst import DialogueStateTracker
from core.router import route, Path
from core.guardrails import apply_guardrails
from core.response_templates import get_filler

logger = logging.getLogger(__name__)


class VoicePipeline:
    """
    Full voice conversation pipeline (v2).

    Orchestrates: Audio → Preprocess → VAD → STT → NLU → Router
                  → [LLM] → Guard → Streaming TTS → Speaker
    With barge-in monitoring during TTS playback.
    """

    def __init__(self):
        self.audio_in = AudioInput()
        self.audio_out = AudioOutput()
        self.preprocessor = AudioPreprocessor()
        self.vad = VAD()
        self.stt = STT()
        self.tts = TTS()
        self.llm = LLM()
        self.dst = DialogueStateTracker()

        self._running = False
        self._turn_count = 0
        self._is_speaking_response = False  # True while TTS is playing
        self._barge_in_detected = False

    def load_models(self) -> None:
        """Pre-load all ML models at startup."""
        logger.info("Loading all models...")
        t0 = time.time()

        self.stt.load()
        self.tts.load()
        # LLM is lazy-loaded on first call

        # Pre-load NLU models (DeBERTa intent + BERT NER)
        from core.nlu.intent import _load_model as load_intent_model
        from core.nlu.slots import _load_ner as load_ner_model
        load_intent_model()
        load_ner_model()

        elapsed = time.time() - t0
        logger.info(f"All models loaded in {elapsed:.1f}s")

    def run(self) -> None:
        """Run the voice conversation loop."""
        self._running = True
        self._turn_count = 0

        print("\n🎙️  AI Voice Receptionist — Ready")
        print("   Speak into your microphone. Press Ctrl+C to stop.\n")

        # Play greeting
        self._speak("Hello! Welcome. How can I help you today?")

        self.audio_in.start()

        try:
            while self._running:
                self._process_one_turn()
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
        finally:
            self.audio_in.stop()
            self._running = False

    def _process_one_turn(self) -> None:
        """Process a single conversation turn."""
        # 1. Listen for speech via VAD (with preprocessing)
        chunk = self.audio_in.read_chunk(timeout=0.1)
        if chunk is None:
            return

        # Preprocess audio before VAD
        clean_chunk = self.preprocessor.process(chunk)
        utterance = self.vad.process_chunk(clean_chunk)
        if utterance is None:
            return  # No complete utterance yet

        # 2. Transcribe speech
        self._turn_count += 1
        t0 = time.time()

        text, stt_elapsed = self.stt.transcribe(utterance)
        if not text or not text.strip():
            return

        print(f"\n  👤 You: \"{text}\" ({stt_elapsed:.2f}s)")

        # 3. NLU: classify intent + extract slots
        nlu_result = self.dst.process_turn(text)
        intent = nlu_result["intent"]
        slots = nlu_result["slots"]

        # 4. Route to appropriate path
        routing = route(nlu_result)
        print(f"  🔀 Path {routing.path.value}: {intent}")

        # 5. Generate response
        if routing.needs_llm:
            # Play filler while LLM processes
            if routing.filler:
                print(f"  💬 Filler: \"{routing.filler}\"")
                self._speak_async(routing.filler)

            # Call LLM
            response, llm_elapsed = self.llm.generate(text)
            print(f"  🤖 LLM ({llm_elapsed:.2f}s): \"{response[:80]}\"")
        else:
            response = routing.response

        # 6. Apply guardrails
        guard_result = apply_guardrails(response, slots)
        if guard_result.warnings:
            for w in guard_result.warnings:
                print(f"  ⚠️  {w}")

        final_response = guard_result.text

        # 7. Speak response (with barge-in monitoring)
        total_elapsed = time.time() - t0
        print(f"  🔊 AI: \"{final_response}\" (total: {total_elapsed:.2f}s)")
        self._speak_with_barge_in(final_response)

    def _speak(self, text: str) -> None:
        """Synthesize and play speech (blocking, no barge-in)."""
        if not text:
            return
        audio, sr, _ = self.tts.synthesize(text)
        if len(audio) > 0:
            self.audio_out.play(audio, sample_rate=sr)

    def _speak_with_barge_in(self, text: str) -> None:
        """
        Speak with barge-in monitoring using streaming TTS.

        Synthesizes text sentence-by-sentence and plays each chunk,
        checking for barge-in between sentences. If the user starts
        speaking, stops playback immediately.
        """
        if not text:
            return

        self._is_speaking_response = True
        self._barge_in_detected = False

        try:
            for audio, sr in self.tts.synthesize_stream(text):
                if self._barge_in_detected:
                    logger.info("Barge-in: stopping remaining sentences")
                    print("  🛑 Barge-in detected — stopping")
                    break

                # Start barge-in monitor thread during playback
                monitor = threading.Thread(
                    target=self._barge_in_monitor, daemon=True
                )
                monitor.start()

                # Play this sentence chunk
                self.audio_out.play(audio, sample_rate=sr)

                # Check if barge-in was detected during playback
                if self._barge_in_detected:
                    self.audio_out.interrupt()
                    break

        finally:
            self._is_speaking_response = False

    def _barge_in_monitor(self) -> None:
        """Monitor mic for speech during TTS playback."""
        while self._is_speaking_response and self.audio_out.is_playing:
            chunk = self.audio_in.read_chunk(timeout=0.05)
            if chunk is None:
                continue

            # Preprocess before VAD check
            clean = self.preprocessor.process(chunk)
            if self.vad.check_barge_in(clean):
                self._barge_in_detected = True
                self.audio_out.interrupt()
                logger.info("Barge-in: user started speaking during response")
                return

    def _speak_async(self, text: str) -> None:
        """Synthesize and play speech in background thread."""
        def _play():
            self._speak(text)
        thread = threading.Thread(target=_play, daemon=True)
        thread.start()

    def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
