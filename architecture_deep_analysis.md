# Voice Receptionist Brain Architecture — Deep Analysis

> This document dissects every layer, component, routing path, edge case, and tech choice in the proposed 8-layer voice brain architecture. For each, I cover: **what it does**, **how smart it is**, **how practical it is**, **realistic latency**, and **what changes when we swap to open-source self-hosted models**.

---

## TL;DR Verdict

| Aspect | Rating | Commentary |
|---|---|---|
| **Architecture Design** | ⭐⭐⭐⭐⭐ | Production-grade 8-layer pipeline with deterministic happy path + LLM fallback. This is exactly how companies like Bland AI, Vapi, and Retell do it |
| **Practicality** | ⭐⭐⭐⭐ | ~80% practical as-is. Some components (coreference resolver, negation scope parser) are harder to build than stated |
| **Latency Claims** | ⭐⭐⭐½ | Happy path 400ms is realistic with APIs. Self-hosted open-source will be **600–900ms** happy path, **1.5–2.5s** LLM path |
| **Open-Source Feasibility** | ⭐⭐⭐⭐ | Fully achievable, but GPU requirements jump significantly. Budget 2× A10/L4 GPUs minimum |

---

## Latency Budget Summary

The architecture claims:

| Path | Claimed | Realistic (APIs) | Realistic (Self-Hosted Open Source) |
|---|---|---|---|
| **Happy Path E2E** | ~400ms | **350–500ms** ✅ | **600–900ms** ⚠️ |
| **LLM Edge Case** | ~900ms | **800–1200ms** ✅ | **1.5–2.5s** ⚠️ |
| **Perceived (filler UX)** | <500ms | **<500ms** ✅ | **<600ms** with filler injection ✅ |
| **STT TTFB** | ~200ms | **200ms** with Deepgram | **400–800ms** with Whisper variants ⚠️ |
| **TTS TTFB** | ~80ms | **80ms** with Cartesia | **200–500ms** with Piper/XTTS ⚠️ |

> [!CAUTION]
> **Self-hosted open-source adds 200–600ms** to both STT and TTS compared to API-based services. The happy path goes from 400ms to ~700ms. The LLM path can exceed 2 seconds. Filler injection becomes **mandatory**, not optional.

---

## Layer-by-Layer Breakdown

---

### Layer 0 — Ingestion & Transport (4 components)

**What it does**: Handles raw audio from the telephone/browser — WebRTC/SIP gateway, voice activity detection, noise cancellation, and barge-in (interruption) detection.

#### Component Analysis

| Component | Smartness | Practicality | Latency | Open-Source Swap |
|---|---|---|---|---|
| **WebRTC/SIP Gateway** | Standard | ✅ Very practical — Twilio/LiveKit handle this | ~0ms (transport layer) | **LiveKit** (open-source SFU), or **Janus** Gateway. Self-hostable on any VM |
| **VAD (Silero VAD)** | Clever — gates STT to save cost | ✅ Already open-source. Runs on CPU in <2ms | **<2ms** | **Already open-source** — Silero VAD is free, MIT-licensed, PyTorch. No swap needed |
| **Acoustic Pre-processor** | Essential but unglamorous | ✅ WebRTC audio engine is built into every browser | **<1ms** | **WebRTC native** or **SpeexDSP** (open-source). RNNoise for denoising is excellent and free |
| **Barge-in Detector** | Very smart UX decision | ⚠️ Moderate — requires careful coordination between TTS output and VAD | **<2ms** detection, but TTS interrupt logic adds complexity | Same VAD on output channel. The hard part is the coordination logic, not the models |

> [!NOTE]
> **Layer 0 is already largely open-source.** Silero VAD, WebRTC, and RNNoise are all free. The real complexity is the **orchestration** — stopping TTS mid-stream, flushing audio buffers, and re-entering listen mode without clicks or artifacts. This is custom engineering, not a model problem.

**Layer 0 Verdict**: ⭐⭐⭐⭐⭐ Smart and practical. No model swaps needed.

---

### Layer 1 — Speech to Text / Streaming (4 components)

**What it does**: Converts streaming audio to text in real-time with partial transcripts, smart endpointing, domain vocabulary boosting, and disfluency filtering.

#### Component Analysis

| Component | Smartness | Practicality | Latency | Open-Source Swap |
|---|---|---|---|---|
| **Streaming STT Engine** | Core component, must be fast | ✅ With Deepgram API. ⚠️ Harder self-hosted | **200ms TTFB** (Deepgram) vs **400–800ms** (self-hosted Whisper) | **Whisper.cpp** (C++, fast), **Faster-Whisper** (CTranslate2), or **whisper-streaming** (RealtimeSTT). Self-hosted needs GPU (T4 minimum) |
| **Endpointing Logic** | Very smart — semantic completeness check beyond silence | ⚠️ Hard to build well. Semantic completeness needs a small classifier | **10–30ms** for the classifier | Train a tiny binary classifier (is-sentence-complete) on labeled data. Or use a rule-based approach: detect trailing verbs without objects |
| **Domain Vocabulary Boost** | Practical and impactful | ✅ Deepgram supports this natively. ⚠️ Whisper doesn't have built-in keyword boosting | **0ms** (preprocessing step) | For Whisper: use **prompt conditioning** (prefix the prompt with domain terms). Less effective than Deepgram's native boost but helps. Alternative: **Whisper + hotword plugin** |
| **Disfluency Filter** | Smart for NLU accuracy | ✅ Very practical — regex + simple rules | **<2ms** | Already model-free. Regex: remove "um", "uh", "like", "you know" + false start patterns. No swap needed |

> [!IMPORTANT]
> **The STT layer is the #1 latency bottleneck for self-hosting.** Deepgram Nova-3 gives 200ms TTFB with true streaming. Self-hosted Whisper variants typically give 400–800ms on a T4 GPU. **This single component can add 200-600ms to your total latency.**
>
> **Recommendation**: Use **Faster-Whisper** with CTranslate2 on an L4 GPU with VAD-segmented chunks. Target the `small` or `medium` model (not `large`) for latency. Expected: ~400ms TTFB for 2-3 second chunks.

**Layer 1 Verdict**: ⭐⭐⭐⭐ Smart design. Self-hosted STT is the biggest latency risk. Budget a dedicated GPU.

---

### Layer 2 — The Brain: NLU + DST + Routing (8 components)

> This is the **core brain** — the most component-heavy and architecturally significant layer.

**What it does**: Classifies intent, extracts slots (entities), tracks dialogue state, resolves pronouns, detects ambiguity and sentiment, and scores confidence to decide routing.

#### Component Analysis

| Component | Smartness | Practicality | Latency | Open-Source Swap |
|---|---|---|---|---|
| **Fast Intent Classifier** (DistilBERT) | ✅ Right approach — small fine-tuned model | ✅ Very practical with ONNX export | **15ms** on GPU, **30ms** on CPU with ONNX | **Already open-source**. DistilBERT is Hugging Face, MIT. Fine-tune on your intents, export to ONNX. Consider **MiniLM-L6-v2** (22M params, ~15ms) for even faster inference |
| **Slot Extractor** (spaCy/NER) | Good — parallel with intent | ✅ Practical. Regex for structured data + spaCy for names | **5–10ms** | **spaCy** is open-source (MIT). Use `en_core_web_sm` for speed. Add **dateparser** (open-source) for natural date/time parsing. **GLiNER** for zero-shot NER if you need flexibility |
| **Dialogue State Tracker** (FSM) | ✅ Very smart — deterministic, no LLM | ✅ This is your **money-saver**. Finite state machine per intent with required/optional slots | **<1ms** (pure logic) | No model involved. This is your own Python code + Redis. Already "open-source" by nature |
| **Confidence Scorer** | Smart gating system | ✅ Practical — just threshold on softmax scores | **<1ms** (arithmetic) | Already model-free. The key is tuning thresholds: 0.85 (high), 0.6–0.85 (clarify), <0.6 (LLM). Tune these on real call data |
| **Context Manager** (Redis) | Standard and essential | ✅ Redis is battle-tested | **2–5ms** per read/write | **Redis** is open-source (BSD). Self-host with Docker. Alternative: **KeyDB** (faster Redis fork, open-source) or **DragonflyDB** |
| **Coreference Resolver** | ⭐ Most impressive sub-component | ⚠️ **Hard to build well.** Rule-based works for simple cases ("cancel it"), but a transformer is needed for complex ones | **5–15ms** rule-based, **30–50ms** with small transformer | Rule-based first: map "it/that/them" to the last-mentioned entity of matching type. For a transformer approach: **SpanBERT** or a fine-tuned small model. This is the **hardest component to get right** |
| **Ambiguity Detector** | Smart — handles compound intents | ⚠️ Medium difficulty. Multi-intent detection needs training data | **5–10ms** | Train a binary classifier: single-intent vs multi-intent. Or use heuristic: presence of "and", "also", "plus" between verb phrases. Open-source: fine-tune a tiny BERT on your data |
| **Sentiment/Urgency Detector** | Nice-to-have for UX quality | ✅ Lightweight and practical | **3–5ms** | **Open-source options**: fine-tuned DistilBERT for sentiment, or **cardiffnlp/twitter-roberta-base-sentiment** (Hugging Face, free). Run on CPU with ONNX |

> [!WARNING]
> **Coreference resolution** is the **most underestimated component** in this architecture. The note says "rule-based + small transformer" casually, but properly resolving "cancel it", "move that to Friday", "call them back" across a multi-turn conversation is a hard NLP problem. For v1, use **rule-based only**: map pronouns to the most recent entity of matching type. Add transformer-based resolution in v2 only if accuracy is insufficient.

> [!TIP]
> **Parallel execution is critical.** The architecture correctly states that intent classification and slot extraction should run in parallel. In Python, use `asyncio.gather()` or `concurrent.futures`. This saves 10–15ms per turn.

**Layer 2 Verdict**: ⭐⭐⭐⭐⭐ The smartest layer. 6 of 8 components are already open-source or model-free. Coreference and ambiguity detection are the hardest to build well.

---

### Layer 3 — Decision Engine & Routing (6 components / 5 paths)

**What it does**: The traffic cop — routes each turn to one of five execution paths based on intent, confidence, slot-fill state, and turn count.

#### Path Analysis

| Path | When Triggered | Latency | % of Calls | How Smart | Practicality |
|---|---|---|---|---|---|
| **Path A: DST Direct** | High confidence (>0.85) + all slots filled | **~400ms E2E** | **60–70%** | ✅ Brilliant — zero LLM, pure templates | ✅ Easiest to build. The 80/20 rule at work |
| **Path B: Clarification** | High confidence + missing slots | **~450ms E2E** | **15–20%** | ✅ Smart — asks only for what's missing | ✅ Template-based prompts: "What date?" not "Can you repeat?" |
| **Path C: LLM Edge Case** | Low confidence (0.6–0.85) or compound intent | **~900ms** claimed, **1.5–2.5s** self-hosted | **5–10%** | ✅ Correct fallback strategy | ⚠️ Latency is the concern here with self-hosted LLM |
| **Path D: RAG + LLM** | Knowledge/FAQ questions | **~800ms** claimed, **1.5–2s** self-hosted | **3–5%** | ✅ Grounds LLM on real data | ⚠️ Needs vector DB + embedding model + LLM — triple stack |
| **Path E: Escalation** | Loop detected, high frustration, or OOD | **Immediate** | **2–5%** | ✅ Very smart — escalation is a feature | ✅ Simple logic: count failed turns, check sentiment score |

> [!IMPORTANT]
> **The genius of this architecture is Path A and B handling 75–90% of all calls deterministically.** This means your open-source LLM (which will be slower than Groq/GPT-4o-mini) only needs to fire on 10–25% of turns. That's the whole game — make the exceptions rare, and you can tolerate higher latency on the LLM path.

**Flow Router itself**: Pure rule engine — `if intent == X and confidence > Y and slots_filled == all: return Path_A`. Runs in **microseconds**. No model, no swap needed.

**Layer 3 Verdict**: ⭐⭐⭐⭐⭐ The most important architectural decision. This 5-path routing is what makes the system production-viable.

---

### Layer 4 — LLM Generation Engine (5 components)

**What it does**: When the deterministic path can't handle it, this layer assembles a prompt, compresses history, calls the LLM, extracts any new slots from the response, and formats output for speech.

#### Component Analysis

| Component | Smartness | Practicality | Latency | Open-Source Swap |
|---|---|---|---|---|
| **Prompt Composer** | Smart — structured, not free-form | ✅ Practical. Pre-templated prompt with slot injection | **<1ms** (string assembly) | No model needed. Your own template engine |
| **History Compressor** | Smart cost/latency optimization | ⚠️ If done with an LLM, adds latency. If done with a summarizer, needs another model | **10–30ms** rule-based, **200ms+** if LLM-based | **Recommended**: Don't use a model. Use structured compression — convert history to slot-fill summary format programmatically. E.g., `"User: booking, name=John, date=Friday, phone=collected"` |
| **LLM (Streaming)** | Core fallback engine | ✅ With Groq/APIs. ⚠️ Self-hosted needs GPU | **100–300ms TTFT** (Groq) vs **500ms–1.5s** (self-hosted) | **Open-source LLMs**: see detailed table below |
| **Slot Backfill Extractor** | Very smart — keeps state clean | ✅ Practical. Run slot extractor on LLM output | **5–10ms** (reuses Layer 2 slot extractor) | Same spaCy/regex from Layer 2. No additional model needed |
| **Response Shaper** | Essential for voice quality | ✅ Practical — regex transformations | **<2ms** | No model. Rules: strip markdown, expand abbreviations, add SSML pauses. Pure string manipulation |

#### Open-Source LLM Options (replacing GPT-4o-mini / Claude Haiku / Groq)

| Model | Params | VRAM Needed | TTFT (A10 GPU) | TTFT (L4 GPU) | Quality for Receptionist | Recommendation |
|---|---|---|---|---|---|---|
| **Llama 3.1 8B** (via vLLM) | 8B | ~16GB | **150–300ms** | **200–400ms** | ✅ Good for structured conversations | ✅ **Best balance** |
| **Mistral 7B v0.3** | 7B | ~14GB | **130–280ms** | **180–350ms** | ✅ Good, fast | ✅ Strong alternative |
| **Qwen 2.5 7B** | 7B | ~14GB | **140–300ms** | **190–380ms** | ✅ Good structured output | Good alternative |
| **Phi-3.5-mini** | 3.8B | ~8GB | **80–150ms** | **100–200ms** | ⚠️ Decent but weaker on nuance | Best if GPU-constrained |
| **Llama 3.1 70B** (4-bit GPTQ) | 70B | ~35GB | **400–800ms** | ❌ Won't fit | ⭐ Excellent quality | Only with 2× A10 or A100 |
| **Gemma 2 9B** | 9B | ~18GB | **160–320ms** | **220–440ms** | ✅ Good quality | Solid option |

> [!TIP]
> **For self-hosted, use vLLM as the inference server** — it handles batching, continuous batching, and PagedAttention automatically. Serve via OpenAI-compatible API endpoint. Your code doesn't change whether you point it at Groq or vLLM.
>
> **Recommended setup**: **Llama 3.1 8B** on **vLLM** with an **L4 or A10 GPU**. Expected TTFT: 200–400ms. Total LLM path: 500ms–1s + STT/TTS time.

**Layer 4 Verdict**: ⭐⭐⭐⭐ Smart design. Self-hosted LLM adds 300–500ms vs API. History compression should be rule-based, not model-based.

---

### Layer 5 — Guardrails & Supervisor (4 components)

**What it does**: Safety net — blocks PII leakage, validates business rules, detects conversation loops, and monitors latency SLAs.

#### Component Analysis

| Component | Smartness | Practicality | Latency | Open-Source Swap |
|---|---|---|---|---|
| **Output Safety Filter** | Essential for production | ✅ Regex + lightweight classifier. Correct: NO LLM here | **<5ms** | **Already model-free** mostly. If you want a classifier: **distilbert-base-uncased** fine-tuned on toxicity data. But regex handles 95% of cases |
| **Business Rule Validator** | Standard but critical | ✅ Very practical — hard-coded constraints | **<1ms** | Pure Python/logic. No model. "Can't book in the past", "can't exceed capacity" — just if/else |
| **Loop & Failure Detector** | Smart UX protection | ✅ Simple counter logic | **<1ms** | Counter: `if same_intent_count >= 3 and not resolved: escalate()`. No model |
| **Latency SLA Monitor** | Production-essential | ✅ Standard observability | **0ms** (async/background) | **OpenTelemetry** (open-source) for traces. **Prometheus** + **Grafana** for metrics. All open-source |

> [!NOTE]
> **Layer 5 is entirely model-free.** This is correct. Using an LLM for guardrails would add 300ms+ to every single turn and defeat the purpose. The architecture correctly uses regex, rules, and counters.

**Layer 5 Verdict**: ⭐⭐⭐⭐⭐ Smart, fast, already open-source. No model swaps needed.

---

### Layer 6 — Action Execution (4 components)

**What it does**: Translates confirmed intents + slots into real API calls (book appointment, cancel, lookup, send SMS), handles results, and queues async side-effects.

#### Component Analysis

| Component | Smartness | Practicality | Latency |
|---|---|---|---|
| **Action Dispatcher** | Standard microservice pattern | ✅ Practical — async execution while TTS streams | Depends on external API (50–500ms) |
| **Tool Registry** | Smart — enables LLM function calling | ✅ Define tools with JSON schemas. LLM path uses function calling; DST path calls directly | **0ms** (configuration) |
| **Async Action Queue** | Smart — don't block response for side-effects | ✅ Use Python `asyncio` task queue or **Celery/Redis Queue** | **0ms** (fire-and-forget) |
| **Action Result Handler** | Essential UX quality | ✅ Map API errors to friendly spoken messages | **<1ms** |

> [!TIP]
> The clever part: **async action queue** means you can fire the "send confirmation SMS" action while already speaking the confirmation to the user. The user never waits for the SMS API.

**Layer 6 Verdict**: ⭐⭐⭐⭐ Standard but well-designed. All model-free. Use **Celery + Redis** or just `asyncio.create_task()` for the async queue.

---

### Layer 7 — TTS & Audio Output (4 components)

**What it does**: Converts text response to streaming audio with SSML prosody, caching of common phrases, and filler injection for latency masking.

#### Component Analysis

| Component | Smartness | Practicality | Latency | Open-Source Swap |
|---|---|---|---|---|
| **SSML Processor** | Good UX polish | ✅ Rule-based text transformation | **<1ms** | No model. String processing to insert `<break>`, `<emphasis>`, and `<prosody>` tags |
| **Streaming TTS Engine** | Must be fast | ✅ With Cartesia/ElevenLabs APIs. ⚠️ Harder self-hosted | **80ms** (Cartesia) vs **200–500ms** (self-hosted) | See table below |
| **Audio Phrase Cache** | ⭐ Brilliant optimization | ✅ Pre-render common phrases, serve in <5ms | **<5ms** for cached phrases | Generate at deployment time. Store as WAV/OGG files. Match by hash of text. Can cut TTS calls by 30–40% |
| **Filler Injector** | ⭐ Highest-ROI UX technique | ✅ Very practical — cached filler audio | **<5ms** (pre-cached) | Pre-record 10–15 filler phrases. Play immediately when processing exceeds 500ms. Resets perceived latency clock |

#### Open-Source TTS Options (replacing Cartesia Sonic / ElevenLabs)

| Model | TTFB (GPU) | Quality | Streaming | VRAM | Recommendation |
|---|---|---|---|---|---|
| **Piper TTS** | **50–100ms** | Good (not natural) | ✅ Chunk-based | **CPU only** | ✅ **Fastest, lightest** — great for templates/fillers |
| **Coqui XTTS v2** | **300–500ms** | Very good (natural) | ⚠️ Chunked | ~4GB | Good for LLM-generated responses |
| **StyleTTS2** | **200–400ms** | Excellent | ⚠️ Not true streaming | ~4GB | Best quality, but not realtime-friendly |
| **Bark** | **2–5s** | Natural with emotion | ❌ Batch | ~6GB | ❌ Too slow for realtime |
| **OpenVoice v2** | **300–500ms** | Good with voice cloning | ⚠️ | ~4GB | If voice cloning needed |
| **Fish Speech 1.5** | **200–400ms** | Very good | ✅ Streaming | ~4GB | ✅ **Best open-source streaming TTS** |

> [!WARNING]
> **Self-hosted TTS is the #2 latency risk** (after STT). Cartesia's 80ms TTFB is nearly impossible to match with open-source in 2024–2025. Best strategy: use **Piper TTS** for cached/template responses (50ms, CPU) and **Fish Speech or XTTS v2** for LLM-generated dynamic responses (300–500ms, GPU).
>
> **Dual-TTS strategy**: Fast TTS (Piper) for deterministic paths, quality TTS (XTTS/Fish) for LLM paths. This is a clever optimization the architecture doesn't explicitly mention.

**Layer 7 Verdict**: ⭐⭐⭐⭐ Good design. Audio caching and filler injection are the real winners. Self-hosted TTS adds 200–400ms latency.

---

### Layer 8 — Infrastructure, Memory & Observability (5 components)

| Component | Smartness | Practicality | Open-Source Options |
|---|---|---|---|
| **Session Store (Redis)** | Standard, essential | ✅ | Redis (BSD), KeyDB, DragonflyDB (all open-source) |
| **User Profile Store (PostgreSQL)** | Smart personalization | ✅ | PostgreSQL is open-source. Standard stuff |
| **Knowledge Base / RAG** | Good for FAQ grounding | ⚠️ Start with JSON lookup, graduate to vector | **pgvector** (PostgreSQL extension, open-source), **Qdrant** (open-source), **ChromaDB** (open-source). Embedding: **nomic-embed-text** or **all-MiniLM-L6-v2** (both free) |
| **Observability Stack** | Production-essential | ✅ | **Langfuse** (open-source), **OpenTelemetry** (open-source), **Prometheus** + **Grafana** |
| **Feedback Loop / RLHF Queue** | Smart continuous improvement | ⚠️ Nice-to-have for v2 | Log failed turns to PostgreSQL. Human labels in a simple admin UI. Re-fine-tune intent classifier monthly |

**Layer 8 Verdict**: ⭐⭐⭐⭐ All standard infrastructure. Every component has mature open-source alternatives.

---

## Routing Decision Matrix Analysis

| Path | Condition | Claimed Latency | Self-Hosted Latency | Smart? |
|---|---|---|---|---|
| **A: DST Direct** | Intent >0.85 + all slots filled | ~400ms | **~600ms** | ✅ Handles 60-70% of calls with zero LLM |
| **B: Clarify** | Intent >0.85 + slots missing | ~450ms | **~650ms** | ✅ Template-based, still no LLM |
| **C: LLM** | Intent 0.6–0.85 or compound | ~900ms | **~1.5–2.5s** | ✅ Correct fallback, but hide latency with fillers |
| **D: RAG + LLM** | FAQ/knowledge question | ~800ms | **~1.5–2s** | ✅ Grounds LLM, prevents hallucination |
| **E: Escalate** | Loop/frustration/OOD | Immediate | **Immediate** | ✅ Escalation is a feature, not failure |

> [!IMPORTANT]
> The **threshold values** (0.85, 0.6) are starting points. In production, you **must** tune these on real call data. Start conservative (high thresholds) and lower them as your intent classifier improves with fine-tuning.

---

## Edge Case Taxonomy — Analysis

| Edge Case | Handling Strategy | Difficulty to Implement | Verdict |
|---|---|---|---|
| **Pronoun reference** ("Cancel it") | Coreference resolver | ⚠️ **Hard** — rule-based covers 70%, need transformer for rest | Smart but hard. Start with "last mentioned entity" heuristic |
| **Implicit date/time** ("tomorrow morning") | Temporal NER with ISO normalization | ✅ **Easy** — use `dateparser` library | Solved problem, open-source library handles it |
| **Compound intent** ("Cancel and reschedule") | Split into sequential intents | ⚠️ **Medium** — needs intent-aware splitting | Use LLM path for these. Don't try to split deterministically in v1 |
| **Partial information** ("Book at 3") | DST tracks missing slots | ✅ **Easy** — this is what DST is built for | Core functionality of Layer 2 DST |
| **Contradiction/correction** ("Monday... actually Tuesday") | Latest value wins + explicit confirmation | ✅ **Easy** — slot override logic | Standard DST behavior |
| **Negation scope** ("Not the morning one") | Negation scope parser | ⚠️ **Hard** — requires understanding semantic scope | ⚠️ Start with keyword detection ("not", "don't") and route to LLM path. Full negation parsing is research-level NLP |
| **Ambiguous affirmation** ("Yeah", "Uh-huh") | Context-aware resolver | ✅ **Medium** — confirm last proposed action | Track `last_proposed_action` in context. "Yeah" confirms it |
| **Out-of-domain** ("What's the weather?") | OOD classifier + polite redirect | ✅ **Easy** — add OOD as an intent class | Add "out_of_domain" as a catch-all intent in your classifier |
| **Self-correction** ("Wait, Smith not Jones") | Repair detector + slot re-extraction | ⚠️ **Medium** — detect "wait", "actually", "I meant" | Keyword triggers + re-run slot extraction on correction utterance |
| **Emotional escalation** | Sentiment spike → escalation | ✅ **Easy** — sentiment threshold | Sentiment > threshold → Path E. Simple and effective |

---

## Complete Open-Source Tech Stack Replacement

| Layer | Original Pick | Open-Source Replacement | Notes |
|---|---|---|---|
| **STT** | Deepgram Nova-3 | **Faster-Whisper** (medium model, CTranslate2) | GPU required (T4/L4). ~400ms TTFB. Use VAD-segmented chunks |
| **VAD** | Silero VAD | **Silero VAD** (already open-source) | No change needed. MIT license |
| **Intent NLU** | Fine-tuned DistilBERT | **Fine-tuned DistilBERT/MiniLM** (ONNX) | Already open-source. Fine-tune on your data |
| **NER / Slots** | spaCy + custom NER | **spaCy** + **dateparser** + regex | Already open-source. MIT license |
| **DST State** | Redis + custom FSM | **Redis** (BSD) + your Python FSM | Already open-source |
| **LLM Fast** | Groq Llama3-8b | **Llama 3.1 8B via vLLM** (self-hosted) | Needs L4/A10 GPU. ~200–400ms TTFT |
| **LLM Smart** | GPT-4o / Claude Sonnet | **Llama 3.1 70B 4-bit via vLLM** | Needs A100/2×A10. ~400–800ms TTFT |
| **RAG Vector DB** | pgvector | **pgvector** (already open-source) | PostgreSQL extension. Free |
| **Embeddings** | text-embedding-3-small | **nomic-embed-text** or **all-MiniLM-L6-v2** | Free, self-hosted, ~5ms inference |
| **TTS** | Cartesia Sonic | **Piper** (fast/templates) + **Fish Speech** (quality/dynamic) | Dual strategy. CPU + GPU |
| **Session Store** | Redis Cluster | **Redis** (self-hosted Docker) | Already open-source |
| **Observability** | Langfuse + OpenTelemetry | **Langfuse** (self-hosted) + **OpenTelemetry** | Both open-source |
| **Call Transport** | Twilio Media Streams | **LiveKit** (open-source SFU) | Self-hostable. Apache 2.0 license |
| **Orchestration** | Python FastAPI | **FastAPI** (already open-source) | MIT license. No change |

---

## GPU Requirements for Full Open-Source Stack

| Component | GPU | VRAM | Can Share GPU? |
|---|---|---|---|
| **STT (Faster-Whisper medium)** | L4 or T4 | ~4GB | ✅ Yes — bursty workload |
| **LLM (Llama 3.1 8B vLLM)** | L4 or A10 | ~16GB | ❌ Needs dedicated GPU |
| **TTS (Fish Speech / XTTS v2)** | T4 or L4 | ~4GB | ✅ Yes — bursty workload |
| **NLU models** (intent, sentiment, NER) | **CPU only** | N/A | ✅ ONNX on CPU is fast enough |
| **Embedding model** | **CPU only** | N/A | ✅ MiniLM is tiny |

**Minimum setup**: **2 GPUs** — 1× L4/A10 (dedicated to LLM) + 1× T4/L4 (shared STT + TTS)

**Ideal setup**: **3 GPUs** — 1× for LLM, 1× for STT, 1× for TTS (full isolation, no contention)

**Monthly cost estimate** (cloud):
- 2× L4 GPUs on GCP: ~$600–800/month
- 1× A10 + 1× T4: ~$500–700/month
- Plus compute (CPU instances for brain/Redis/etc): ~$100–200/month
- **Total: ~$700–1000/month** for full self-hosted open-source

> [!NOTE]
> Compare this to the original plan's **~$500–550/month with API-based services** (Deepgram + Groq + Cartesia). Self-hosting is **40–80% more expensive** but gives you **full control, no rate limits, data privacy, and no vendor lock-in**.

---

## Realistic End-to-End Latency (Self-Hosted Open Source)

```
HAPPY PATH (Path A — 60-70% of turns):
┌─────────────────────────────────────────────────────────────────────┐
│ STT (Whisper)     Intent+Slots     DST      Template    TTS (Piper)│
│   400ms      +      30ms       +  <1ms   +   <1ms    +   100ms    │
│                                                                     │
│ Total: ~530ms                                                       │
│ With cached TTS: ~430ms ✅                                          │
└─────────────────────────────────────────────────────────────────────┘

CLARIFICATION PATH (Path B — 15-20%):
┌─────────────────────────────────────────────────────────────────────┐
│ STT (Whisper)     Intent+Slots     DST      Template    TTS (Piper)│
│   400ms      +      30ms       +  <1ms   +   <1ms    +   100ms    │
│                                                                     │
│ Total: ~530ms                                                       │
│ With cached TTS: ~430ms ✅                                          │
└─────────────────────────────────────────────────────────────────────┘

LLM PATH (Path C — 5-10%):
┌──────────────────────────────────────────────────────────────────────────┐
│ STT (Whisper)   NLU    LLM (Llama 3.1 8B)   Backfill   TTS (Fish)     │
│   400ms      + 30ms +    500–1000ms        +  10ms   +  300ms          │
│                                                                          │
│ Total: ~1.2–1.7s ⚠️                                                     │
│ With filler injection: perceived ~600ms ✅                               │
└──────────────────────────────────────────────────────────────────────────┘

RAG + LLM PATH (Path D — 3-5%):
┌───────────────────────────────────────────────────────────────────────────────┐
│ STT (Whisper)   NLU    Embed+Search   LLM (Llama 3.1 8B)   TTS (Fish)      │
│   400ms      + 30ms +    50ms      +    500–1000ms        +  300ms          │
│                                                                               │
│ Total: ~1.3–1.8s ⚠️                                                          │
│ With filler injection: perceived ~600ms ✅                                    │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## What's Missing From the Architecture

| Gap | Importance | Suggestion |
|---|---|---|
| **Multi-language support** | ⚠️ Important for India | Add language detection at STT layer. Use multilingual Whisper. Intent classifier needs multilingual training data |
| **Call recording & compliance** | ✅ Critical for businesses | Record all calls to object storage. GDPR/privacy compliance. Consent at call start |
| **A/B testing framework** | Nice-to-have | Compare template vs LLM responses on similar intents. Track resolution rate per path |
| **Warm-up / pre-loading** | ⚠️ Important | Pre-load all models at startup. Cold starts on models add 5–30 seconds. Keep models in GPU memory |
| **Graceful degradation** | ✅ Critical | If LLM GPU is down → expand template coverage. If STT fails → offer to call back. Always have fallback paths |
| **Cost per call tracking** | Nice-to-have | Track: STT minutes + LLM tokens + TTS characters per call. Bill per call or per minute |

---

## Final Verdict

> [!IMPORTANT]
> **This architecture is production-grade, well-thought-out, and follows the same patterns used by commercial voice AI platforms.** The 8-layer separation, 5-path routing, and deterministic-first / LLM-fallback philosophy are exactly right.

### Strengths
1. **Hybrid deterministic + LLM approach** — 70–85% of calls never touch an LLM. This is the key insight that makes it viable
2. **Filler injection** — Buy time during LLM processing. Single highest-ROI UX technique
3. **Audio phrase caching** — Pre-rendered greetings/confirmations in <5ms is brilliant
4. **Edge case taxonomy** — 10 failure modes with specific handling strategies. Shows deep product thinking
5. **Parallel execution** — Intent + slot extraction running in parallel. Correct

### Risks
1. **Self-hosted STT latency** — Whisper is 2× slower than Deepgram for streaming. This is the biggest latency bottleneck
2. **Coreference resolution** — Harder than it sounds. Start rule-based, iterate
3. **LLM path latency** — 1.2–1.7s self-hosted. **Must** use filler injection or it breaks UX
4. **GPU costs** — 2–3 GPUs for full self-hosting. More expensive than API-based but gives full control
5. **Negation scope parsing** — Research-level NLP. Route to LLM path instead of building a parser

### Overall: **Build it. It's smart, practical, and achievable with open-source.**
