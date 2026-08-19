# AI Voice Receptionist

A voice agent for a medical clinic. Speech in, speech out, over a local microphone.

```
Mic → Preprocess → VAD → STT → Joint NLU → DST → Router → [LLM] → Guardrails → TTS → Speaker
```

| Layer | Implementation |
|---|---|
| VAD | Silero VAD (streaming, 512-sample windows) |
| STT | faster-whisper (CTranslate2; CPU on macOS — no MPS support) |
| NLU | Fine-tuned joint DeBERTa — intent + BIO slot tags in one forward pass |
| Dialogue | Finite state machine: `collecting → confirming → executing → done` |
| Routing | 5 paths: direct / clarify / LLM / FAQ+LLM / escalate |
| LLM | Google Gemini, grounded to a fixed knowledge block |
| TTS | Coqui XTTS v2 (pipelined, sentence-by-sentence, with barge-in) |

---

## Quick start

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # add -r requirements-dev.txt to run tests

cp .env.example .env                     # then set GEMINI_API_KEY
python main.py --check                   # validate config + load every model
python main.py                           # start the voice loop
```

Requires **Python 3.11**. First run downloads ~2 GB of model weights.

### Modes

| Command | What it does |
|---|---|
| `python main.py` | Full voice loop (microphone + speakers) |
| `python main.py --text` | Text REPL — same NLU/DST/router/LLM path, no audio hardware |
| `python main.py --check` | Validate config, load all models, exit |

`--text` is the fastest way to exercise the conversation logic without a mic.

---

## Configuration

Everything lives in `.env`; `config.py` is the single source of truth and
validates on startup. See `.env.example` for the annotated list.

Settings worth knowing about:

| Key | Why it matters |
|---|---|
| `VAD_MIN_SPEECH_MS` | Keep at ~200. Above ~300 the agent discards short replies like "yes" or "3 pm" — i.e. every answer to a clarifying question. |
| `STT_MIN_DURATION_S` | Same reason. 0.25 is right; 0.8 silently drops short answers. |
| `GEMINI_THINKING_BUDGET` | 0 disables Gemini 2.5 "thinking". Thinking adds ~1s of latency and consumes `max_output_tokens`, which can leave nothing for the reply. |
| `GEMINI_TIMEOUT_S` | Hard per-request ceiling. Without it a hung call stalls the whole loop. |
| `NLU_LOG_REDACT_PII` | On by default. Turn logs contain caller names and phone numbers. |
| `JOINT_NLU_ENABLED` | `true` uses the fine-tuned joint model (production). `false` falls back to zero-shot intent + BERT NER, which is far slower and less accurate. |

`python main.py --check` prints warnings for any setting that would degrade the
conversation.

---

## Architecture notes

**Slot validation vs. guardrails are separate concerns.**
`core/nlu/validation.py` checks business rules (opening hours, roster,
past dates) and returns a *repair* — a spoken explanation plus the slot to
clear — so the dialogue keeps moving. `core/guardrails.py` only sanitises
outbound text (PII redaction, markdown stripping, length cap). Mixing the two
deadlocks the call: an invalid slot replaces the agent's next question with a
refusal and the caller can never correct it.

**Actions require explicit confirmation.** Filling the last slot moves the DST
to `CONFIRMING`, which reads the booking back. Only an explicit yes advances to
`EXECUTING`, and the pipeline calls `dst.mark_action_complete()` exactly once so
the action can never re-fire.

**Audio preprocessing is split by stage.** `process_chunk()` runs per 30 ms
frame and applies only a stateful high-pass filter — it must not alter relative
loudness, because VAD and the STT energy gate depend on it. `process_utterance()`
runs once per complete utterance and applies noise reduction and AGC.

**The mic is muted while the agent speaks** and flushed afterwards, so the agent
never transcribes its own output. The queue is bounded and drops oldest-first.

---

## Testing

```bash
pytest                      # everything
pytest -m "not slow"        # skip tests that load real model weights
pytest tests/test_dst.py -v # one file
```

`pyproject.toml` pins the pytest rootdir to this project. Without it pytest
walks up the filesystem and can adopt an unrelated config from your home
directory.

---

## Known limitations

These are design boundaries, not defects:

- **No backend.** "Your appointment is booked" is a sentence, not a
  transaction. There is no database or calendar; nothing is persisted or
  retrievable.
- **Closed intent set.** The NLU handles the 30 intents it was trained on.
  Anything outside that falls through to the LLM.
- **No retrieval.** The LLM is grounded by a fixed knowledge block in the
  system prompt, with an explicit refuse-if-unknown contract. That prevents
  invention but is not a real knowledge base — `Path D` is named `RAG_LLM`
  aspirationally.
- **Local microphone only.** No telephony (SIP/WebRTC) integration.
- **Latency.** A turn costs roughly VAD endpointing + STT + NLU + optional LLM
  + TTS. Expect a few seconds end to end on CPU, not human turn-taking speed.

---

## Repository layout

```
config.py               all settings, device policy, business rules
main.py                 entry point (voice / --text / --check)
core/
  pipeline.py           orchestration loop, barge-in, error isolation
  audio_input.py        bounded mic capture
  audio_output.py       playback with real interruption
  audio_preprocessor.py per-chunk HPF, per-utterance denoise + AGC
  vad.py                Silero VAD (thread-safe)
  stt.py                faster-whisper
  tts.py                XTTS v2, pipelined, PII-safe cache
  llm.py                Gemini client with retries + grounding contract
  router.py             5-path routing
  guardrails.py         outbound response sanitisation
  response_templates.py deterministic replies
  nlu/
    joint_nlu.py        joint intent + slot inference
    joint_model.py      the DeBERTa model definition
    dst.py              dialogue state machine
    validation.py       business-rule slot validation
    entities.py         canonical doctor/service resolution
    slots.py            date/time/phone/service extraction
    confidence.py       routing confidence
    logger.py           redacted, rotating turn log
scripts/                dataset generation + training
tests/                  pytest suite
```
