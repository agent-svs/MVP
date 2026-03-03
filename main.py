"""
AI Voice Receptionist — Main Entry Point

Usage:
    python main.py

Runs the full voice conversation pipeline:
    Mic → VAD → STT → NLU → Router → [LLM] → Guardrails → TTS → Speaker
"""

import sys
import time

from config import cfg


def main():
    cfg.setup_logging()

    print("=" * 60)
    print("  🎙️  AI Voice Receptionist")
    print("=" * 60)
    print(f"  Device:    {cfg.device}")
    print(f"  STT:       Faster Whisper ({cfg.stt.model_size})")
    print(f"  TTS:       XTTS v2 ({cfg.tts.speaker})")
    print(f"  LLM:       Gemini ({cfg.llm.model})")
    print(f"  VAD:       Silero (threshold={cfg.vad.threshold})")
    print("=" * 60)
    print()

    # Validate config
    issues = cfg.validate()
    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
        print()

    from core.pipeline import VoicePipeline

    pipeline = VoicePipeline()

    print("⏳ Loading models (first run may download ~2GB)...")
    t0 = time.time()
    pipeline.load_models()
    print(f"✅ Models loaded in {time.time() - t0:.1f}s\n")

    try:
        pipeline.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        print("✅ Voice receptionist stopped.\n")


if __name__ == "__main__":
    main()
