# Joint NLU Model — Revised Implementation Plan (v2)

## Status: Phase 1 ✅ Complete → Phases 2–4 Remaining

Phase 1 (Dataset Creation) is **done**. `dataset.jsonl` has **4,326 entries** across **30 intents** with **6 slot types**, including upsampled bottom intents, adversarial examples, ASR-error patterns, and negation edge cases.

---

## Why This Approach (Unchanged — Still Best)

| Criteria | Joint NLU + Context | Current: Separate NER + Intent | Previous: Regex |
|---|---|---|---|
| **"Okay" problem** | Context resolves ambiguity natively | Needs `pending_slot` hack | Fails |
| **One-word answers** | Context tells model it's a `slot_answer` | `_raw_text_to_slot()` fallback (brittle) | Fails |
| **Scalability** | Add intents by retraining only | Must update DST rules, router, templates | Must update regex |
| **Production standard** | Used by Rasa DIET, Dialogflow, Alexa | Common in prototypes | Legacy |

The joint model replaces `_try_fill_expected_slot()`, `_raw_text_to_slot()`, `pending_slot`, and the dual-low-confidence logic in `dst.py` with a **single learned forward pass**.

---

## Architecture

```
[SYS]: What service do you need?       ← previous system turn (context)
[USR]: Cardiology                       ← current user utterance
                    ↓
        ┌─────────────────────────────────────────┐
        │   Shared DeBERTa-v3-base Encoder        │
        │   Input: "[CLS] [SYS] context [SEP]     │
        │           [USR] utterance [SEP]"         │
        └──────────┬──────────────┬───────────────┘
                   │              │
        ┌──────────▼──────┐ ┌────▼────────────────┐
        │  Intent Head    │ │  Slot Head (NER)     │
        │  Linear → 30   │ │  Linear → BIO tags   │
        │  classes        │ │  per token           │
        └──────────┬──────┘ └────┬────────────────┘
                   │              │
         intent: slot_answer  B-SERVICE_TYPE: "cardiology"
                   ↓
            DST Update (simplified)
```

**Key decisions:**
- **Base model**: `microsoft/deberta-v3-base` (86M params, 768-dim) — fast CPU inference (~80ms)
- **Input format**: `[CLS] [SYS] {context} [SEP] [USR] {utterance} [SEP]`
- **Intent head**: Linear on `[CLS]` → 30 classes
- **Slot head**: Linear per token → 13 BIO tags (6 entity types × 2 B/I + O)
- **Joint loss**: `L = 0.6 × L_intent + 0.4 × L_slot`
- **Fallback**: The existing v3 pipeline (`intent.py` + `slots.py` + `dst.py`) stays intact as fallback if `JOINT_NLU_ENABLED=false` in config

---

## Phase 1: Dataset ✅ DONE

- **File**: [dataset.jsonl](file:///Users/swayamvyas7/Developer/Agents/MVP/dataset.jsonl) — 4,326 entries
- **Format**: `{"context": "...", "utterance": "...", "intent": "...", "slots": {...}}`
- **30 intents** balanced (lowest: 73 examples, highest: 856)
- **6 slot types**: `date`, `time`, `patient_name`, `doctor_name`, `phone`, `service_type`
- Includes adversarial examples, ASR errors, negation traps, ambiguous "Okay" cases

---

## Phase 2: Dataset Processing & BIO Annotation

> [!IMPORTANT]
> This phase transforms the raw `dataset.jsonl` into a tokenized, BIO-annotated format ready for model training. This is the **most error-prone step** — subword alignment must be pixel-perfect.

### Step 2.1: Define BIO Tag Schema

```
O               — Outside (no entity)
B-SERVICE_TYPE  — Beginning of service mention
I-SERVICE_TYPE  — Inside/continuation
B-DATE / I-DATE
B-TIME / I-TIME
B-PATIENT_NAME / I-PATIENT_NAME
B-DOCTOR_NAME  / I-DOCTOR_NAME
B-PHONE / I-PHONE
```

**Total: 13 BIO tags** (6 entity types × 2 + O)

> [!NOTE]
> The original plan had 15 tags (with B-DEPARTMENT/I-DEPARTMENT). We drop DEPARTMENT since `service_type` already covers it — having both creates label confusion.

### Step 2.2: Label Maps Module

#### [NEW] [label_maps.py](file:///Users/swayamvyas7/Developer/Agents/MVP/core/nlu/label_maps.py)

Intent label list (30 intents) + BIO slot label list (13 tags) + bidirectional ID maps. This is the single source of truth referenced by training, inference, and tests.

### Step 2.3: BIO Annotation Script

#### [NEW] [create_joint_dataset.py](file:///Users/swayamvyas7/Developer/Agents/MVP/scripts/create_joint_dataset.py)

Reads `dataset.jsonl` and produces `data/joint_nlu_train.jsonl`, `data/joint_nlu_val.jsonl`, `data/joint_nlu_test.jsonl` (80/10/10 split).

The script must:
1. **Tokenize** each utterance (whitespace-first, then verify alignment with DeBERTa tokenizer)
2. **Locate slot spans** in the utterance text using the `slots` dict values and character-level matching
3. **Assign BIO tags** per token — `B-*` for the first token of a slot span, `I-*` for subsequent tokens, `O` for everything else
4. **Handle edge cases**:
   - Multi-word slots: `"John Smith"` → `["B-PATIENT_NAME", "I-PATIENT_NAME"]`
   - Slots not found in utterance text (e.g., normalized values) — use fuzzy matching
   - Empty slots dict → all `O` tags
5. **Validate** every processed example: assert `len(tokens) == len(slot_labels)`
6. **Split** 80/10/10 with stratified sampling on intent

**Output format:**
```json
{
  "context": "What service do you need?",
  "utterance": "Cardiology",
  "intent": "slot_answer",
  "tokens": ["Cardiology"],
  "slot_labels": ["B-SERVICE_TYPE"]
}
```

---

## Phase 3: Model Architecture & Training

### Step 3.1: Joint Model Class

#### [NEW] [joint_model.py](file:///Users/swayamvyas7/Developer/Agents/MVP/core/nlu/joint_model.py)

`JointDeBERTa(DebertaV2PreTrainedModel)` with:
- Shared DeBERTa encoder
- Intent classifier: `Linear(hidden_size → num_intents)` on `[CLS]`
- Slot classifier: `Linear(hidden_size → num_slot_labels)` per token
- Joint loss: weighted combo of `CrossEntropyLoss` (intent) + `CrossEntropyLoss` with `ignore_index=-100` (slots)
- `user_token_mask` to ignore context tokens during slot loss

### Step 3.2: Dataset Loader

#### [NEW] [joint_dataset_loader.py](file:///Users/swayamvyas7/Developer/Agents/MVP/scripts/joint_dataset_loader.py)

PyTorch `Dataset` that:
1. Formats input as `[CLS] [SYS] {context} [SEP] [USR] {utterance} [SEP]`
2. Tokenizes with DeBERTa tokenizer (handles subword splitting)
3. **Aligns BIO labels to subwords** — only first subword gets `B-`/`I-` tag, rest get `-100`
4. Builds `user_token_mask`: `1` for user-utterance tokens, `0` for context/special tokens
5. Sets `slot_labels = -100` for all non-user tokens

### Step 3.3: Training Script

#### [NEW] [train_joint_nlu.py](file:///Users/swayamvyas7/Developer/Agents/MVP/scripts/train_joint_nlu.py)

Based on the existing [train_intent.py](file:///Users/swayamvyas7/Developer/Agents/MVP/scripts/train_intent.py) structure but extended for joint training:

```python
# Key hyperparameters
BASE_MODEL = "microsoft/deberta-v3-base"
EPOCHS = 10
BATCH_SIZE = 16
LR = 2e-5
WARMUP_RATIO = 0.1
MAX_SEQ_LEN = 128
INTENT_LOSS_WEIGHT = 0.6
OUTPUT_DIR = "models/joint_nlu/"
```

**Flow:**
1. Load tokenizer, add `[SYS]` and `[USR]` special tokens
2. Load model, resize embeddings
3. Load train/val/test splits
4. Train with HuggingFace `Trainer` + custom `compute_metrics` (intent acc + slot F1 + joint F1)
5. Save best model + tokenizer
6. Evaluate on test set, print final metrics

**Target metrics (minimum acceptable):**
- Intent Accuracy: ≥ 0.90
- Slot F1 (entity-level, seqeval): ≥ 0.85
- Joint F1 (average): ≥ 0.87

---

## Phase 4: Inference Integration

### Step 4.1: Joint NLU Inference Engine

#### [NEW] [joint_nlu.py](file:///Users/swayamvyas7/Developer/Agents/MVP/core/nlu/joint_nlu.py)

```python
class JointNLU:
    def load(self, model_path="models/joint_nlu/"):
        """Load model + tokenizer once."""
    
    def predict(self, context: str, utterance: str) -> dict:
        """
        Single forward pass → intent + slots.
        
        Returns:
            {"intent": str|None, "intent_confidence": float,
             "slots": {"slot_name": {"value": str, "confidence": float}}}
        """
```

Key implementation details:
- Builds `user_token_mask` by finding the `[USR]` token position
- Decodes BIO tags → slot dict with confidence scores
- If `intent == "slot_answer"`, returns `intent = None` (DST keeps current intent)

### Step 4.2: Config Flag

#### [MODIFY] [config.py](file:///Users/swayamvyas7/Developer/Agents/MVP/config.py)

Add:
```python
JOINT_NLU_ENABLED = bool(os.getenv("JOINT_NLU_ENABLED", "false").lower() == "true")
JOINT_NLU_MODEL_PATH = os.getenv("JOINT_NLU_MODEL_PATH", "models/joint_nlu/")
```

This allows toggling between v3 (separate NER + intent) and v4 (joint model) without code changes.

### Step 4.3: Simplified DST

#### [MODIFY] [dst.py](file:///Users/swayamvyas7/Developer/Agents/MVP/core/nlu/dst.py)

**What changes:**
- Add a `_process_turn_joint()` method that takes `(text, system_context)` and uses `JointNLU.predict()`
- In `process_turn()`, check `cfg.JOINT_NLU_ENABLED`:
  - If `True` → call `_process_turn_joint()` (single forward pass, no pending_slot logic needed)
  - If `False` → existing v3 pipeline (unchanged)
- `_process_turn_joint()` replaces: `_try_fill_expected_slot()`, `_raw_text_to_slot()`, `pending_slot`, and the separate `classify_intent()` + `extract_slots()` calls

**What stays the same:**
- `State` enum, `REQUIRED_SLOTS`, `_update_state()`, `_is_new_conversation()`, `_is_slot_correction()`, `_build_result()`, `missing_slots()`, `all_slots_filled()`, `history()`
- The state machine logic is intact — only the NLU input pipeline changes

### Step 4.4: Pipeline Context Passing

#### [MODIFY] [pipeline.py](file:///Users/swayamvyas7/Developer/Agents/MVP/core/pipeline.py)

Add `self._last_response = ""` to `__init__`. In `_process_one_turn()`:
```python
nlu_result = self.dst.process_turn(text, system_context=self._last_response)
# ...
self._last_response = final_response
```

And in `load_models()`, conditionally load the joint model:
```python
if cfg.JOINT_NLU_ENABLED:
    from core.nlu.joint_nlu import JointNLU
    self.joint_nlu = JointNLU()
    self.joint_nlu.load()
```

### Step 4.5: Router Updates

#### [MODIFY] [router.py](file:///Users/swayamvyas7/Developer/Agents/MVP/core/router.py)

Minimal changes — the router already works with a `nlu_result` dict. The joint model produces the same dict structure. Only change needed: remove the `set_expectation()` call since the joint model doesn't need it (context does the work).

---

## File Summary

| File | Status | Purpose |
|---|---|---|
| `dataset.jsonl` | ✅ DONE | Raw dataset (4,326 entries) |
| `core/nlu/label_maps.py` | **NEW** | Intent + BIO slot label definitions |
| `scripts/create_joint_dataset.py` | **NEW** | BIO annotation + train/val/test split |
| `core/nlu/joint_model.py` | **NEW** | `JointDeBERTa` model class |
| `scripts/joint_dataset_loader.py` | **NEW** | PyTorch dataset with subword alignment |
| `scripts/train_joint_nlu.py` | **NEW** | Training script (HF Trainer) |
| `core/nlu/joint_nlu.py` | **NEW** | Inference engine (predict from context + utterance) |
| `config.py` | **MODIFY** | Add `JOINT_NLU_ENABLED` flag |
| `core/nlu/dst.py` | **MODIFY** | Add `_process_turn_joint()`, keep v3 as fallback |
| `core/pipeline.py` | **MODIFY** | Pass `system_context`, load joint model |
| `core/router.py` | **MODIFY** | Remove `set_expectation()` when joint mode active |
| `tests/test_joint_nlu.py` | **NEW** | Tests for joint model inference + integration |
| `core/nlu/intent.py` | **KEEP** | Fallback when `JOINT_NLU_ENABLED=false` |
| `core/nlu/slots.py` | **KEEP** | Fallback when `JOINT_NLU_ENABLED=false` |

---

## Verification Plan

### Automated Tests

#### 1. BIO Annotation Validation (`scripts/create_joint_dataset.py`)
```bash
python scripts/create_joint_dataset.py --validate
```
- Checks every example has `len(tokens) == len(slot_labels)`
- Checks every slot value in `slots` dict has a corresponding BIO span in `slot_labels`
- Reports annotation coverage %

#### 2. Training Metrics (`scripts/train_joint_nlu.py`)
```bash
python scripts/train_joint_nlu.py
```
- Outputs intent accuracy, slot F1, and joint F1 on test set
- Target: intent acc ≥ 0.90, slot F1 ≥ 0.85, joint F1 ≥ 0.87
- Saves model to `models/joint_nlu/`

#### 3. Unit Tests (`tests/test_joint_nlu.py`)
```bash
python -m pytest tests/test_joint_nlu.py -v
```

**New test cases to write:**

| Test | Input Context | Utterance | Expected Intent | Expected Slots |
|---|---|---|---|---|
| `test_joint_book_appointment` | "How can I help?" | "I want to book cardiology" | `book_appointment` | `service_type: cardiology` |
| `test_joint_slot_answer_service` | "What service?" | "Cardiology" | `None` (slot_answer) | `service_type: cardiology` |
| `test_joint_slot_answer_name` | "Your name?" | "John Smith" | `None` (slot_answer) | `patient_name: John Smith` |
| `test_joint_confirmation_okay` | "Is Thursday at 3pm fine?" | "Okay" | `confirmation` | none |
| `test_joint_denial_with_slot` | "Is Thursday fine?" | "No, make it Friday" | `denial` | `date: Friday` |
| `test_joint_negation_trap` | "Cancel your appointment?" | "I do NOT want to cancel" | `denial` | none |
| `test_joint_greeting_no_context` | "" | "Hello" | `greeting` | none |
| `test_joint_emergency` | "How can I help?" | "My father is having a heart attack" | `emergency_request` | none |
| `test_joint_phone_asr` | "Your phone?" | "nine eight seven six five four three two one zero" | `None` (slot_answer) | `phone: 9876543210` |

#### 4. Existing Test Regression
```bash
python -m pytest tests/test_nlu.py -v
```
- All 40+ existing tests in [test_nlu.py](file:///Users/swayamvyas7/Developer/Agents/MVP/tests/test_nlu.py) must still pass (v3 fallback mode)

### Manual Verification

> [!IMPORTANT]
> After all automated tests pass, the user should test the interactive DST:

```bash
JOINT_NLU_ENABLED=true python -m core.nlu.dst
```

Test this conversation flow manually:
1. Type `"Hello"` → should get `greeting` intent
2. Type `"I want to book an appointment"` → should get `book_appointment`
3. (AI asks "What service?") → Type `"Cardiology"` → should fill `service_type` without breaking intent
4. (AI asks "What date?") → Type `"Tomorrow"` → should fill `date`
5. (AI asks "What time?") → Type `"3pm"` → should fill `time`
6. Type `"Actually, make it Friday"` → should correct `date` to Friday
7. Type `"Bye"` → should get `goodbye`

---

## Execution Order

> [!IMPORTANT]
> Follow this exact order. Each phase depends on the previous.

| Phase | Work | Est. Time | Files |
|---|---|---|---|
| ~~1~~ | ~~Dataset Creation~~ | ~~✅ Done~~ | ~~`dataset.jsonl`~~ |
| **2** | BIO Annotation + Split | ~3-4 hours | `label_maps.py`, `create_joint_dataset.py` |
| **3** | Model + Training | ~4-6 hours (GPU) | `joint_model.py`, `joint_dataset_loader.py`, `train_joint_nlu.py` |
| **4** | Integration + Tests | ~3-4 hours | `joint_nlu.py`, `dst.py`, `pipeline.py`, `config.py`, `router.py`, `test_joint_nlu.py` |

**Total remaining: ~10-14 hours of dev time**

The existing v3 pipeline continues operating as-is until Phase 4 is complete and the `JOINT_NLU_ENABLED` flag is set to `true`. **Zero breaking changes until explicit switchover.**
