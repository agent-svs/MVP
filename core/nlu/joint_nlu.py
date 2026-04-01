import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoConfig
from core.nlu.joint_model import JointDeBERTa
from core.nlu.label_maps import ID_TO_INTENT, ID_TO_TAG, NUM_INTENTS, NUM_SLOT_LABELS
from core.nlu.slots import _extract_date, _extract_time

def _normalize_slot(name: str, raw_val: str) -> str:
    """Normalize specific slots using slots.py heuristics."""
    if name == "date":
        res = _extract_date(raw_val)
        if res: return res[0]
    elif name == "time":
        res = _extract_time(raw_val)
        if res: return res[0]
    return raw_val

class JointNLU:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    def load(self, model_path="models/joint_nlu/"):
        """Load model + tokenizer once."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Joint NLU model not found at {model_path}. Please train it first.")
            
        print(f"Loading Joint NLU from {model_path} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        config = AutoConfig.from_pretrained(model_path)
        config.num_intents = NUM_INTENTS
        config.num_slot_labels = NUM_SLOT_LABELS
        
        self.model = JointDeBERTa.from_pretrained(model_path, config=config)
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, context: str, utterance: str) -> dict:
        """
        Single forward pass → intent + slots.
        
        Returns:
            {"intent": str|None, "intent_confidence": float,
             "slots": {"slot_name": {"value": str, "confidence": float}}}
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        # Tokenize context
        context_enc = self.tokenizer(context, add_special_tokens=False)
        context_ids = context_enc["input_ids"]
        
        # Tokenize utterance with offsets to map tokens back to characters
        utterance_enc = self.tokenizer(utterance, add_special_tokens=False, return_offsets_mapping=True)
        utterance_ids = utterance_enc["input_ids"]
        word_ids = utterance_enc.word_ids()
        offsets = utterance_enc["offset_mapping"]

        sys_token_id = self.tokenizer.convert_tokens_to_ids("[SYS]")
        usr_token_id = self.tokenizer.convert_tokens_to_ids("[USR]")
        
        # Fallbacks if special tokens are somehow missing
        if sys_token_id == self.tokenizer.unk_token_id:
            sys_token_id = self.tokenizer.cls_token_id
        if usr_token_id == self.tokenizer.unk_token_id:
            usr_token_id = self.tokenizer.cls_token_id
            
        # Assemble sequence: [CLS] [SYS] context [SEP] [USR] utterance [SEP]
        input_ids = (
            [self.tokenizer.cls_token_id, sys_token_id] + 
            context_ids + 
            [self.tokenizer.sep_token_id, usr_token_id] + 
            utterance_ids + 
            [self.tokenizer.sep_token_id]
        )
        
        attention_mask = [1] * len(input_ids)
        
        # Convert to tensors
        input_ids_t = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attn_mask_t = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids_t, attention_mask=attn_mask_t)
            intent_logits = outputs["intent_logits"][0]
            slot_logits = outputs["slot_logits"][0]
            
        # Intent classification
        intent_probs = torch.softmax(intent_logits, dim=-1)
        intent_id = torch.argmax(intent_probs).item()
        intent_conf = intent_probs[intent_id].item()
        intent_str = ID_TO_INTENT.get(intent_id)
        
        # If intent is slot_answer, we return None so DST maintains current intent
        if intent_str == "slot_answer":
            intent_str = None
            
        # Slot classification
        slot_probs = torch.softmax(slot_logits, dim=-1)
        slot_preds = torch.argmax(slot_probs, dim=-1).cpu().numpy()
        
        # Calculate exactly where the utterance subwords start in the sequence
        utterance_start_idx = 2 + len(context_ids) + 2
        
        slots_found = {}
        current_slot_name = None
        current_slot_start = -1
        current_slot_end = -1
        current_slot_conf = []
        
        for i, subword_idx in enumerate(range(utterance_start_idx, utterance_start_idx + len(utterance_ids))):
            tag_id = slot_preds[subword_idx]
            tag = ID_TO_TAG.get(tag_id, "O")
            conf = slot_probs[subword_idx][tag_id].item()
            
            # offsets[i] gives us the (start_char, end_char) in the raw utterance string
            start_c, end_c = offsets[i]
            if start_c == end_c:
                continue # special/space token
            
            if tag.startswith("B-"):
                # Save previous slot if exists
                if current_slot_name and current_slot_start != -1:
                    raw_val = utterance[current_slot_start:current_slot_end]
                    slots_found[current_slot_name] = {
                        "value": _normalize_slot(current_slot_name, raw_val.strip()),
                        "confidence": sum(current_slot_conf) / len(current_slot_conf)
                    }
                    
                raw_slot_name = tag[2:].lower()
                current_slot_name = "patient_name" if raw_slot_name == "name" else raw_slot_name
                current_slot_start = start_c
                current_slot_end = end_c
                current_slot_conf = [conf]
                
            elif tag.startswith("I-") and current_slot_name == tag[2:].lower():
                # Continue slot
                if current_slot_start != -1:
                    current_slot_end = end_c
                    current_slot_conf.append(conf)
            else:
                # O tag or mismatching I- tag
                if current_slot_name and current_slot_start != -1:
                    raw_val = utterance[current_slot_start:current_slot_end]
                    slots_found[current_slot_name] = {
                        "value": _normalize_slot(current_slot_name, raw_val.strip()),
                        "confidence": sum(current_slot_conf) / len(current_slot_conf)
                    }
                    current_slot_name = None
                    current_slot_start = -1
                    current_slot_end = -1
                    current_slot_conf = []
                    
        # Catch any trailing slot
        if current_slot_name and current_slot_start != -1:
            raw_val = utterance[current_slot_start:current_slot_end]
            slots_found[current_slot_name] = {
                "value": _normalize_slot(current_slot_name, raw_val.strip()),
                "confidence": sum(current_slot_conf) / len(current_slot_conf)
            }
            
        return {
            "intent": intent_str,
            "intent_confidence": intent_conf,
            "slots": slots_found,
            "raw_nlu_intent": ID_TO_INTENT.get(intent_id) # useful for debugging
        }
