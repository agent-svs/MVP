"""
AI Voice Receptionist — entry point.

    python main.py            run the voice loop
    python main.py --check    validate config and load models, then exit
    python main.py --text     text-only REPL (no microphone or speakers)
"""

from __future__ import annotations

import sys
import time


def _banner(cfg) -> None:
    print(cfg.summary())
    issues = cfg.validate()
    if issues:
        print("  Configuration warnings:")
        for issue in issues:
            print(f"    ⚠️  {issue}")
        print()


def run_text_mode(cfg) -> int:
    """Text REPL — exercises NLU → DST → Router → LLM → Guardrails, no audio."""
    from core.guardrails import apply_guardrails
    from core.llm import LLM
    from core.nlu.dst import DialogueStateTracker
    from core.router import route

    joint = None
    if cfg.nlu.joint_nlu_enabled:
        from core.nlu.joint_nlu import JointNLU, JointNLUError

        joint = JointNLU()
        try:
            joint.load(cfg.nlu.joint_nlu_model_path)
        except JointNLUError as e:
            print(f"  ⚠️  {e}\n  Falling back to the separate NLU pipeline.")
            joint, cfg.nlu.joint_nlu_enabled = None, False

    dst, llm = DialogueStateTracker(), LLM()
    last = ""
    print("\n💬 Text mode — type 'quit' to exit, 'reset' to start a new call.\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            return 0
        if text.lower() == "reset":
            dst.reset()
            llm.clear_history()
            last = ""
            print("  [call reset]\n")
            continue

        try:
            nlu = dst.process_turn(text, system_context=last, joint_nlu_engine=joint)
            routing = route(nlu, last_response=last)
            if routing.needs_llm:
                result = llm.generate(
                    text,
                    context={"slots": nlu["slots"], "active_intent": nlu.get("_active_intent")},
                )
                response = result.text
            else:
                response = routing.response
            if routing.execute_action:
                dst.mark_action_complete(routing.execute_action)
            if routing.path.value == "clarify" and routing.metadata.get("missing_slot"):
                dst.set_expectation(routing.metadata["missing_slot"], response)

            guard = apply_guardrails(response, nlu["slots"])
            last = guard.text
            print(
                f"  [{nlu['intent']}/{nlu['_active_intent']} state={nlu['state']} "
                f"path={routing.path.value} missing={nlu['missing_slots']}]"
            )
            print(f"AI : {last}\n")
        except Exception as e:
            print(f"  ⚠️  Turn failed: {type(e).__name__}: {e}\n")


def run_check(cfg) -> int:
    """Validate configuration and load every model, then exit."""
    from core.pipeline import VoicePipeline

    print("Loading models...")
    t0 = time.time()
    pipeline = VoicePipeline()
    pipeline.load_models()
    print(f"\n✅ All models loaded in {time.time() - t0:.1f}s")
    print(f"   NLU path : {'joint DeBERTa' if cfg.nlu.joint_nlu_enabled else 'zero-shot + NER'}")
    print(f"   LLM      : {'configured' if __import__('core.llm', fromlist=['LLM']).LLM().is_configured else 'NOT configured'}")
    return 0


def main() -> int:
    from config import cfg

    cfg.setup_logging()
    args = set(sys.argv[1:])

    print("=" * 62)
    print(f"  🎙️  {cfg.business.name} — AI Voice Receptionist")
    print("=" * 62)
    _banner(cfg)

    if "--check" in args:
        return run_check(cfg)
    if "--text" in args:
        return run_text_mode(cfg)

    from core.pipeline import VoicePipeline

    print("⏳ Loading models (first run downloads ~2GB)...")
    t0 = time.time()
    pipeline = VoicePipeline()
    try:
        pipeline.load_models()
    except Exception as e:
        print(f"\n❌ Failed to load models: {type(e).__name__}: {e}")
        return 1
    print(f"✅ Models loaded in {time.time() - t0:.1f}s\n")

    try:
        pipeline.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
