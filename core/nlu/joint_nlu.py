"""
Joint NLU — single-pass intent classification + slot tagging.

One DeBERTa forward pass produces both the intent label and BIO slot tags over
the user utterance, conditioned on the previous system turn:

    [CLS] [SYS] <system context> [SEP] [USR] <user utterance> [SEP]

This is the production NLU path (JOINT_NLU_ENABLED=true).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Optional

import torch
from transformers import AutoConfig, AutoTokenizer

from config import cfg
from core.nlu.entities import entity_store
from core.nlu.joint_model import JointDeBERTa
from core.nlu.label_maps import ID_TO_INTENT, ID_TO_TAG, NUM_INTENTS, NUM_SLOT_LABELS
from core.nlu.slots import _extract_date, _extract_time, _extract_phone

logger = logging.getLogger(__name__)

# Must match the training max_seq_len (scripts/train_joint_nlu.py).
MAX_SEQ_LEN = 128

# Tag suffix → canonical slot key. Kept explicit so a change to SLOT_TAGS
# surfaces here instead of silently breaking I- continuation matching.
_TAG_TO_SLOT = {
    "date": "date",
    "time": "time",
    "phone": "phone",
    "doctor_name": "doctor_name",
    "patient_name": "patient_name",
    "name": "patient_name",
    "service_type": "service_type",
}


def _normalize_slot(name: str, raw_val: str) -> str:
    """Normalise a raw span to the same canonical form the legacy path emits.

    Without this the two NLU paths returned different shapes for the same slot
    (joint: "555-123-4567", legacy: "5551234567").
    """
    raw_val = (raw_val or "").strip()
    if not raw_val:
        return raw_val
    if name == "date":
        res = _extract_date(raw_val)
        return res[0] if res else raw_val
    if name == "time":
        res = _extract_time(raw_val)
        return res[0] if res else raw_val
    if name == "phone":
        res = _extract_phone(raw_val)
        return res[0] if res else re.sub(r"\D", "", raw_val) or raw_val
    if name in ("patient_name", "doctor_name"):
        return re.sub(r"\s+", " ", raw_val).strip(" .,")
    return raw_val


class JointNLUError(RuntimeError):
    """Raised when the joint NLU model cannot be loaded or run."""


class JointNLU:
    """Single-pass intent + slot model."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_path: Optional[str] = None
        self.device = torch.device(cfg.torch_device)
        # The model is stateless per call, but tokenizer/model access is
        # serialised because the pipeline touches NLU from more than one thread.
        self._lock = threading.Lock()

    # ── loading ─────────────────────────────────────────────
    def load(self, model_path: Optional[str] = None) -> None:
        """Load model + tokenizer once.

        Falls back to the published Hub repo when the local directory is
        missing. Raises JointNLUError (not a bare exception) on failure so the
        caller can degrade instead of crashing with a stack trace.
        """
        model_path = model_path or cfg.nlu.joint_nlu_model_path
        resolved = model_path
        if not os.path.isdir(model_path):
            resolved = cfg.nlu.joint_nlu_hub_id
            logger.info(
                f"Local joint NLU model not found at '{model_path}' — "
                f"falling back to Hub repo '{resolved}'"
            )

        logger.info(f"Loading joint NLU from {resolved} onto {self.device}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(resolved)
            if not tokenizer.is_fast:
                raise JointNLUError(
                    f"Tokenizer for '{resolved}' is not a fast tokenizer; "
                    "offset mapping (required for slot span extraction) is unavailable."
                )

            config = AutoConfig.from_pretrained(resolved)
            config.num_intents = NUM_INTENTS
            config.num_slot_labels = NUM_SLOT_LABELS

            model = JointDeBERTa.from_pretrained(resolved, config=config)
            model.to(self.device)
            model.eval()
        except JointNLUError:
            raise
        except Exception as e:
            raise JointNLUError(
                f"Could not load joint NLU model from '{resolved}': {e}"
            ) from e

        self.tokenizer, self.model, self.model_path = tokenizer, model, resolved
        logger.info("Joint NLU ready")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    # ── inference ───────────────────────────────────────────
    def predict(self, context: str, utterance: str) -> dict[str, Any]:
        """Single forward pass → intent + slots.

        Returns:
            {"intent": str|None, "intent_confidence": float,
             "slots": {slot: {"value","confidence",...}},
             "raw_nlu_intent": str}
        """
        if not self.is_loaded:
            raise JointNLUError("Model not loaded. Call load() first.")

        utterance = utterance or ""
        context = context or ""
        if not utterance.strip():
            return {"intent": None, "intent_confidence": 0.0, "slots": {},
                    "raw_nlu_intent": None}

        with self._lock:
            return self._predict_locked(context, utterance)

    def _predict_locked(self, context: str, utterance: str) -> dict[str, Any]:
        tok = self.tokenizer

        context_ids = tok(context, add_special_tokens=False)["input_ids"]
        utt_enc = tok(utterance, add_special_tokens=False, return_offsets_mapping=True)
        utterance_ids = utt_enc["input_ids"]
        offsets = utt_enc["offset_mapping"]

        # Sequence = [CLS][SYS] ctx [SEP][USR] utt [SEP] → ctx + utt + 5 tokens.
        # Keep the utterance whole (it carries the slots); trim context first.
        budget = MAX_SEQ_LEN - 5
        if len(utterance_ids) > budget:
            utterance_ids = utterance_ids[:budget]
            offsets = offsets[:budget]
            context_ids = []
        else:
            max_context = budget - len(utterance_ids)
            context_ids = context_ids[-max_context:] if max_context > 0 else []

        sys_id = tok.convert_tokens_to_ids("[SYS]")
        usr_id = tok.convert_tokens_to_ids("[USR]")
        if sys_id == tok.unk_token_id:
            sys_id = tok.cls_token_id
        if usr_id == tok.unk_token_id:
            usr_id = tok.cls_token_id

        input_ids = (
            [tok.cls_token_id, sys_id]
            + context_ids
            + [tok.sep_token_id, usr_id]
            + utterance_ids
            + [tok.sep_token_id]
        )
        attention_mask = [1] * len(input_ids)

        ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        mask_t = torch.tensor([attention_mask], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            out = self.model(input_ids=ids_t, attention_mask=mask_t)
            intent_logits = out["intent_logits"][0]
            slot_logits = out["slot_logits"][0]

        intent_probs = torch.softmax(intent_logits, dim=-1)
        intent_id = int(torch.argmax(intent_probs).item())
        intent_conf = float(intent_probs[intent_id].item())
        raw_intent = ID_TO_INTENT.get(intent_id)

        # "slot_answer" means "this is a reply to my question, not a new
        # intent" — the DST keeps the active task intent.
        intent_str = None if raw_intent == "slot_answer" else raw_intent

        slot_probs = torch.softmax(slot_logits, dim=-1)
        slot_preds = torch.argmax(slot_probs, dim=-1).cpu().tolist()

        # [CLS][SYS] + ctx + [SEP][USR]  →  utterance starts at 4 + len(ctx)
        start_idx = 4 + len(context_ids)
        spans = self._decode_bio(
            utterance, utterance_ids, offsets, slot_preds, slot_probs, start_idx
        )

        return {
            "intent": intent_str,
            "intent_confidence": intent_conf,
            "slots": spans,
            "raw_nlu_intent": raw_intent,
        }

    @staticmethod
    def _decode_bio(utterance, utterance_ids, offsets, slot_preds, slot_probs, start_idx):
        """Decode BIO tags into {slot: {value, confidence, canonical_*}}."""
        found: dict[str, dict[str, Any]] = {}
        cur_name: Optional[str] = None
        cur_start = cur_end = -1
        cur_conf: list[float] = []

        def flush() -> None:
            nonlocal cur_name, cur_start, cur_end, cur_conf
            if cur_name and cur_start != -1 and cur_conf:
                raw = utterance[cur_start:cur_end]
                value = _normalize_slot(cur_name, raw)
                entry: dict[str, Any] = {
                    "value": value,
                    "confidence": sum(cur_conf) / len(cur_conf),
                    "raw_text": raw.strip(),
                }
                # The legacy path canonicalises via the entity store; the joint
                # path previously did not, so doctor/service IDs never resolved
                # in production.
                canonical = entity_store.resolve_entity(cur_name, value)
                if canonical:
                    entry["canonical_id"] = canonical["id"]
                    entry["canonical_name"] = canonical["canonical_name"]
                    entry["value"] = canonical["canonical_name"]
                    # A canonical hit is independent evidence. The tagger often
                    # over-extends a span ("to book a checkup"), which drags the
                    # mean token confidence below the accept threshold even
                    # though we know exactly which service was named.
                    entry["confidence"] = max(
                        entry["confidence"],
                        float(canonical.get("match_confidence", 1.0)) * 0.95,
                    )
                found[cur_name] = entry
            cur_name, cur_start, cur_end, cur_conf = None, -1, -1, []

        for i in range(len(utterance_ids)):
            seq_idx = start_idx + i
            if seq_idx >= len(slot_preds):
                break
            tag_id = slot_preds[seq_idx]
            tag = ID_TO_TAG.get(tag_id, "O")
            conf = float(slot_probs[seq_idx][tag_id].item())

            start_c, end_c = offsets[i]
            if start_c == end_c:  # special / zero-width token
                continue

            if tag.startswith("B-"):
                flush()
                cur_name = _TAG_TO_SLOT.get(tag[2:].lower(), tag[2:].lower())
                cur_start, cur_end, cur_conf = start_c, end_c, [conf]
            elif (
                tag.startswith("I-")
                and cur_name is not None
                and _TAG_TO_SLOT.get(tag[2:].lower(), tag[2:].lower()) == cur_name
            ):
                if cur_start != -1:
                    cur_end = end_c
                    cur_conf.append(conf)
            else:
                flush()

        flush()
        return found
