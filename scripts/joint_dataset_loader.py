import json
import torch
from torch.utils.data import Dataset
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.nlu.label_maps import INTENT_TO_ID, TAG_TO_ID

class JointNLUDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []
        
        with open(data_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        context = item.get("context", "")
        utterance = item.get("utterance", "")
        intent = item.get("intent", "out_of_domain")
        
        words = item.get("tokens", [])
        slot_labels_str = item.get("slot_labels", [])
        
        # Tokenize context
        context_enc = self.tokenizer(context, add_special_tokens=False)
        context_ids = context_enc["input_ids"]
        
        # Tokenize utterance (is_split_into_words=True for aligning slot targets)
        word_enc = self.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
            truncation=False
        )
        utterance_ids = word_enc["input_ids"]
        word_ids = word_enc.word_ids()
        
        # Align labels
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                label_str = slot_labels_str[word_idx]
                aligned_labels.append(TAG_TO_ID.get(label_str, 0))
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx
            
        sys_token_id = self.tokenizer.convert_tokens_to_ids("[SYS]")
        usr_token_id = self.tokenizer.convert_tokens_to_ids("[USR]")
        
        # Handle cases where special tokens weren't added to vocab yet
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
        
        # Labels should be -100 for context and special tokens
        slot_labels = (
            [-100, -100] + 
            [-100] * len(context_ids) + 
            [-100, -100] + 
            aligned_labels + 
            [-100]
        )
        
        # Truncate
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len-1] + [self.tokenizer.sep_token_id]
            slot_labels = slot_labels[:self.max_seq_len-1] + [-100]
            
        attention_mask = [1] * len(input_ids)
        
        # Pad
        padding_length = self.max_seq_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            slot_labels = slot_labels + [-100] * padding_length
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "intent_labels": torch.tensor(INTENT_TO_ID.get(intent, 0), dtype=torch.long),
            "slot_labels": torch.tensor(slot_labels, dtype=torch.long)
        }
