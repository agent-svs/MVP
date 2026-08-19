> [!CAUTION]
> **STALE — superseded.** This document describes an earlier build in which the
> NLU was regex-based and barge-in was missing. Both were replaced: intent and
> slots now come from a fine-tuned joint DeBERTa model (`core/nlu/joint_nlu.py`)
> and barge-in is implemented in `core/pipeline.py`. Treat the layer-by-layer
> scores below as historical. See `README.md` for the current architecture and
> `claudeIMPRo.txt` for the current strategic assessment.

---

# Reference Architecture vs Implementation Gap Analysis

> [!NOTE]
> Detailed comparison of every component in `random.txt` against the actual codebase.
> **Summary:** 11 of 45 components fully implemented. 15 partially done. 19 completely missing.

> [!WARNING]
> **Major Architectural Deviations:** The most critical gaps are in the **NLU Brain** (using regex instead of ML) and **Action Execution** (missing entirely).

## 🚨 Biggest Gaps (Ordered by Impact)

1. **Intent Classifier is regex, not ML** — The plan calls for a trained classifier; we used keyword+pattern matching.
2. **Slot Extractor is regex, not NER** — No spaCy/transformer NER despite spaCy being in `requirements.txt`.
3. **No Action Execution** — The pipeline returns template strings but never calls real APIs.
4. **No RAG/Knowledge Base** — FAQ questions go to plain LLM with no retrieved context.
5. **No Coreference / Sentiment / Ambiguity** — "Cancel it", frustration detection, and multi-intent are all missing.
6. **No Streaming** — Both STT and TTS are batch; plan calls for streaming in both.
7. **No Infrastructure** — No Redis, no user profiles, no observability beyond basic logging.

---

## Detailed Layer Analysis

### Layer 0 — Ingestion & Transport
* **Score:** 1 Done, 1 Partial, 2 Missing*

#### ⚠️ WebRTC/SIP Gateway
- **Plan:** PSTN, VoIP, browser calls via Opus → 16kHz PCM
- **Implemented:** Local mic via `sounddevice` (`AudioInput`)
- **Gap:** Simplified — direct mic only, no telephony

#### ✅ VAD
- **Plan:** Detect speech, prevent STT on silence, barge-in
- **Implemented:** Silero VAD with speech/silence detection
- **Gap:** Mostly done

#### ❌ Barge-in Detector
- **Plan:** Detect user speaking during TTS, stop playback
- **Implemented:** Pipeline has `_speak_async` but **no actual barge-in** — TTS plays fully
- **Gap:** Missing

#### ❌ Acoustic Preprocessor
- **Plan:** Noise suppression, echo cancellation, auto gain
- **Implemented:** Not implemented
- **Gap:** Missing

---

### Layer 1 — Speech-to-Text
* **Score:** 1 Done, 2 Partial, 1 Missing*

#### ⚠️ Streaming STT
- **Plan:** Partial transcripts every 200-300ms (Deepgram/Whisper Live)
- **Implemented:** Batch STT — full utterance transcribed after VAD silence
- **Gap:** Batch, not streaming

#### ⚠️ Endpointing
- **Plan:** Silence detection + semantic completeness
- **Implemented:** VAD silence-only endpointing
- **Gap:** Partial — no semantic completeness

#### ❌ Domain Vocabulary Boost
- **Plan:** Inject business terms (doctor names, products)
- **Implemented:** Not implemented
- **Gap:** Missing

#### ✅ Disfluency Filter
- **Plan:** Remove "um", "uh", "like"
- **Implemented:** `clean_disfluencies()` in `stt.py`
- **Gap:** Done

---

### Layer 2 — The Brain (NLU + DST)
* **Score:** 2 Done, 3 Partial, 4 Missing*

#### ❌ Fast Intent Classifier
- **Plan:** ML-based classifier (the doc implies a trained model)
- **Implemented:** Rule-based regex + keyword matching in `intent.py`
- **Gap:** Not as planned — uses regex, not ML

#### ⚠️ Intents Covered
- **Plan:** book, cancel, reschedule, hours, pricing, transfer, complaint
- **Implemented:** book, cancel, reschedule, check_status, faq, greeting, goodbye
- **Gap:** Partial — missing transfer, complaint, pricing as distinct intents

#### ❌ Slot Extractor (NER)
- **Plan:** ML NER for date, time, phone, party_size, reference_ID
- **Implemented:** Regex-based extractor for date, time, name, phone, service_type
- **Gap:** Regex, not NER — missing party_size, reference_ID

#### ✅ Dialogue State Tracker
- **Plan:** Track slot completion per intent
- **Implemented:** FSM with states: IDLE → GREETING → COLLECTING → CONFIRMING → EXECUTING → DONE
- **Gap:** Done

#### ✅ Confidence Scorer
- **Plan:** 3-tier thresholds: >0.85 / 0.60-0.85 / <0.60
- **Implemented:** Weighted combo (80% intent + 20% slot fill), configurable thresholds
- **Gap:** Done (thresholds differ slightly)

#### ⚠️ Context Manager
- **Plan:** Session memory: intent, slots, history, caller ID, metadata
- **Implemented:** In-memory `_history` list in DST
- **Gap:** Basic — no caller ID, no metadata, no persistence

#### ❌ Coreference Resolver
- **Plan:** "Cancel **it**" → resolves to previous appointment
- **Implemented:** Not implemented
- **Gap:** Missing

#### ⚠️ Ambiguity Detector
- **Plan:** Multiple intents, vague requests, out-of-domain
- **Implemented:** Basic `out_of_domain` fallback only
- **Gap:** Minimal — no multi-intent detection

#### ❌ Sentiment & Urgency Detector
- **Plan:** Detect frustration, anger, urgency → escalation
- **Implemented:** Not implemented
- **Gap:** Missing

---

### Layer 3 — Decision Engine
* **Score:** 3 Done, 3 Partial, 0 Missing*

#### ✅ Path A — DST Happy Path
- **Plan:** conf >0.85 + all slots → template response (~400ms)
- **Implemented:** `Path.A_DIRECT` — high confidence + all slots
- **Gap:** Done

#### ✅ Path B — Clarification
- **Plan:** Intent known, missing slots
- **Implemented:** `Path.B_CLARIFY` — asks for first missing slot
- **Gap:** Done

#### ✅ Path C — LLM Edge Case
- **Plan:** conf <0.85, compound/complex language
- **Implemented:** `Path.C_LLM` — low confidence → Gemini
- **Gap:** Done

#### ⚠️ Path D — RAG Knowledge
- **Plan:** Vector search → retrieve chunks → LLM response
- **Implemented:** `Path.D_RAG_LLM` exists but falls through to plain LLM — no vector DB
- **Gap:** Stub only — no RAG

#### ⚠️ Path E — Human Escalation
- **Plan:** User requests human, repeated failures, frustration
- **Implemented:** Escalation after 3 failed turns
- **Gap:** Partial — no frustration detection, no "transfer" keyword

#### ⚠️ Routing Inputs
- **Plan:** Intent, confidence, slot completion, state, sentiment
- **Implemented:** Intent, confidence, slots, state, failed_turns
- **Gap:** No sentiment input

---

### Layer 4 — LLM Generation Engine
* **Score:** 1 Done, 2 Partial, 2 Missing*

#### ⚠️ Prompt Composer
- **Plan:** System persona + business rules + history + RAG + user query
- **Implemented:** System prompt + history + user query
- **Gap:** Partial — no business rules injection, no RAG context

#### ❌ History Compressor
- **Plan:** Summarize older turns to reduce tokens
- **Implemented:** Not implemented — full history sent
- **Gap:** Missing

#### ⚠️ Streaming LLM
- **Plan:** Stream tokens to TTS immediately
- **Implemented:** `generate_stream()` exists but pipeline uses `generate()` (blocking)
- **Gap:** Available but unused

#### ❌ Slot Backfill Extractor
- **Plan:** Extract slots from LLM output ("Friday evening" → date+time)
- **Implemented:** Not implemented — NLU only runs on user input
- **Gap:** Missing

#### ✅ Response Shaper
- **Plan:** "Dr. Smith at 3pm" → "Doctor Smith at three PM"
- **Implemented:** `shape_for_speech()` — strips markdown, expands abbreviations
- **Gap:** Done

---

### Layer 5 — Guardrails & Supervisor
* **Score:** 1 Done, 2 Partial, 1 Missing*

#### ⚠️ Output Safety Filter
- **Plan:** Block PII leakage, hallucinated data, off-brand
- **Implemented:** PII detection (SSN, credit card) + scrubbing
- **Gap:** PII only — no hallucination or brand check

#### ⚠️ Business Rule Validator
- **Plan:** No past bookings, no capacity overflow, no invalid cancels
- **Implemented:** Date validation (past dates, weekends) + time validation (business hours)
- **Gap:** Partial — no capacity/availability check

#### ✅ Loop & Failure Detector
- **Plan:** Repeated failures → escalation
- **Implemented:** `failed_turns >= 3` triggers Path E
- **Gap:** Done

#### ❌ Latency Monitor
- **Plan:** P50/P95 tracking, component delays, SLA fallback
- **Implemented:** Not implemented — only per-turn elapsed time printed
- **Gap:** Missing

---

### Layer 6 — Action Execution
* **Score:** 0 Done, 0 Partial, 4 Missing*

#### ❌ Action Dispatcher
- **Plan:** Map intent → API calls (book, cancel, lookup, SMS, transfer)
- **Implemented:** Template responses only — no real API calls
- **Gap:** Missing

#### ❌ Tool Registry
- **Plan:** Define tool schemas for deterministic + LLM function calling
- **Implemented:** Not implemented
- **Gap:** Missing

#### ❌ Async Action Queue
- **Plan:** Side-effects: SMS, CRM update, analytics
- **Implemented:** Not implemented
- **Gap:** Missing

#### ❌ Action Result Handler
- **Plan:** API result → spoken response
- **Implemented:** Not implemented
- **Gap:** Missing

---

### Layer 7 — Text-to-Speech
* **Score:** 2 Done, 1 Partial, 1 Missing*

#### ❌ SSML Processor
- **Plan:** Pauses, emphasis, speaking rate adjustments
- **Implemented:** Not implemented — plain text to TTS
- **Gap:** Missing

#### ⚠️ Streaming TTS
- **Plan:** Stream audio chunks immediately
- **Implemented:** Batch TTS — full synthesis then play
- **Gap:** Batch, not streaming

#### ✅ Audio Phrase Cache
- **Plan:** Pre-render common phrases (~5ms latency)
- **Implemented:** `.npy` file cache in `assets/cache/`
- **Gap:** Done

#### ✅ Filler Injector
- **Plan:** Play filler if processing >500ms
- **Implemented:** `_speak_async()` plays filler while LLM generates
- **Gap:** Done

---

### Layer 8 — Infrastructure & Memory
* **Score:** 0 Done, 1 Partial, 4 Missing*

#### ❌ Session Store (Redis)
- **Plan:** call_id, intent, slots, history, sentiment, timestamps
- **Implemented:** In-memory `_history` list — lost on restart
- **Gap:** No persistence

#### ❌ User Profile Store
- **Plan:** Caller history, preferences, name, VIP, last appointment
- **Implemented:** Not implemented
- **Gap:** Missing

#### ❌ Knowledge Base (RAG)
- **Plan:** FAQs, policies, product info, documentation
- **Implemented:** Not implemented
- **Gap:** Missing

#### ⚠️ Observability Stack
- **Plan:** STT latency, intent confidence, routing decisions, LLM tokens, TTS latency
- **Implemented:** Basic `logger.info()` per component
- **Gap:** Logging only — no metrics/dashboards

#### ❌ Feedback Loop
- **Plan:** Failed interactions → retrain classifier, slot extractor
- **Implemented:** Not implemented
- **Gap:** Missing
