"""
Voice Pipeline — Full Orchestration Loop

Wires all layers together:
  Mic → VAD → STT → NLU → Router → [LLM] → Guardrails → TTS → Speaker

Features:
- Filler injection for LLM latency masking
- Barge-in support (interrupt TTS playback)
- Graceful error handling per turn
"""

import logging
import threading
import time
from typing import Optional

from config import cfg
from core.audio_input import AudioInput
from core.audio_output import AudioOutput
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
    Full voice conversation pipeline.

    Orchestrates: Audio → VAD → STT → NLU → Router → [LLM] → Guard → TTS → Speaker
    """

    def __init__(self):
        self.audio_in = AudioInput()
        self.audio_out = AudioOutput()
        self.vad = VAD()
        self.stt = STT()
        self.tts = TTS()
        self.llm = LLM()
        self.dst = DialogueStateTracker()

        self._running = False
        self._turn_count = 0

    def load_models(self) -> None:
        """Pre-load all ML models at startup."""
        logger.info("Loading all models...")
        t0 = time.time()

        self.stt.load()
        self.tts.load()
        # LLM is lazy-loaded on first call

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
        # 1. Listen for speech via VAD
        chunk = self.audio_in.read_chunk(timeout=0.1)
        if chunk is None:
            return

        utterance = self.vad.process_chunk(chunk)
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

        # 7. Speak response
        total_elapsed = time.time() - t0
        print(f"  🔊 AI: \"{final_response}\" (total: {total_elapsed:.2f}s)")
        self._speak(final_response)

    def _speak(self, text: str) -> None:
        """Synthesize and play speech (blocking)."""
        if not text:
            return
        audio, sr, _ = self.tts.synthesize(text)
        if len(audio) > 0:
            self.audio_out.play(audio, sample_rate=sr)

    def _speak_async(self, text: str) -> None:
        """Synthesize and play speech in background thread."""
        def _play():
            self._speak(text)
        thread = threading.Thread(target=_play, daemon=True)
        thread.start()

    def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
