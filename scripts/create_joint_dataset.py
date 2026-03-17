import json
import re
import os
import argparse
from collections import defaultdict
import random

# Read intent and tag info
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.nlu.label_maps import INTENT_LABELS, SLOT_TAGS

def fuzzy_match_span(utterance, slot_val):
    u = utterance.lower()
    s = str(slot_val).lower()
    
    # Exact match first
    if s in u:
        idx = u.find(s)
        return idx, idx + len(s)
        
    if not s.strip():
        return None
        
    s_words = re.findall(r'\w+', s)
    if not s_words:
        return None
        
    u_words = list(re.finditer(r'\w+', u))
    if not u_words:
        return None
        
    best_match = None
    best_score = 0
    
    # Try sliding window of length len(s_words) over u_words
    for i in range(len(u_words)):
        for length in range(1, len(s_words) + 3):  # try lengths from 1 to len(s_words)+2
            if i + length > len(u_words):
                break
            
            window_u = u_words[i:i+length]
            window_u_strs = [w.group() for w in window_u]
            
            # Count how many words in s_words are in window
            score = 0
            for sw in s_words:
                if any(sw in uw or uw in sw for uw in window_u_strs):
                    score += 1
            
            # Penalty for window being too long compared to matching words
            ratio = score / length
            if score > best_score and ratio >= 0.5 and score >= len(s_words) * 0.5:
                best_score = score
                best_match = (window_u[0].start(), window_u[-1].end())
                
    if best_match:
        return best_match
    return None

def find_spans(utterance, slots):
    spans = []
    for slot_name, slot_val in slots.items():
        if isinstance(slot_val, dict):
            slot_val = slot_val.get("value", "")
            
        span = fuzzy_match_span(utterance, slot_val)
        if span:
            spans.append({"start": span[0], "end": span[1], "slot": slot_name.upper()})
    return sorted(spans, key=lambda x: x["start"])

def annotate_example(utterance, slots):
    spans = find_spans(utterance, slots)
    
    tokens = []
    slot_labels = []
    
    # Tokenize words and punctuation
    for match in re.finditer(r"\w+|[^\w\s]", utterance):
        tok_str = match.group()
        tok_start = match.start()
        tok_end = match.end()
        
        tokens.append(tok_str)
        
        assigned_label = "O"
        # Find which span this token falls into
        for span in spans:
            # Overlap check
            if tok_start >= span["start"] and tok_end <= span["end"]:
                # If we're at the very start of the matched span (or the first token that overlaps it)
                # Actually, check if the previous token was already labeled with this slot
                # A simple heuristic: if it's the first token in the list with this slot type, it's B-
                is_first = True
                for prev_label in reversed(slot_labels):
                    if prev_label == "O": continue
                    if prev_label.endswith(span["slot"]):
                        is_first = False
                        break
                
                if is_first:
                    assigned_label = f"B-{span['slot']}"
                else:
                    assigned_label = f"I-{span['slot']}"
                break
        
        # Verify valid label
        if assigned_label not in SLOT_TAGS:
            # fallback to O if invalid (e.g. unknown slot)
            assigned_label = "O"
            
        slot_labels.append(assigned_label)
        
    return tokens, slot_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="dataset.jsonl")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    data = []
    with open(args.input, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
                
    annotated_data = []
    valid_count = 0
    dropped = 0
    for i, item in enumerate(data):
        utterance = item["utterance"]
        slots = item.get("slots", {})
        intent = item.get("intent", "out_of_domain")
        
        tokens, slot_labels = annotate_example(utterance, slots)
        
        if len(tokens) == len(slot_labels):
            valid_count += 1
            
        # Check if we aligned all expected slots
        matched_slots = set([l.split("-")[1] for l in slot_labels if l != "O"])
        expected_slots = set([k.upper() for k in slots.keys()])
        
        if expected_slots and not expected_slots.issubset(matched_slots):
            dropped += 1
            continue
            
        item["tokens"] = tokens
        item["slot_labels"] = slot_labels
        annotated_data.append(item)
        
    if args.validate:
        print(f"Validated {valid_count}/{len(data)} examples with matching token/label counts.")
        if valid_count != len(data):
            print("ERROR: Mismatch in token/label lengths.")
            sys.exit(1)
        
        # Check that we actually caught some slots
        has_slot_count = sum(1 for x in annotated_data if any(l != "O" for l in x["slot_labels"]))
        expected_slots_count = sum(1 for x in annotated_data if x.get("slots"))
        print(f"Coverage: {has_slot_count} / {expected_slots_count} examples with slots successfully aligned. Dropped {dropped} unaligned examples.")
        sys.exit(0)

    # Split dataset based on intents (stratified roughly)
    intent_bins = defaultdict(list)
    for item in annotated_data:
        intent_bins[item["intent"]].append(item)
        
    train_data = []
    val_data = []
    test_data = []
    
    # 80/10/10 split
    for opt_intent, items in intent_bins.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train+n_val])
        test_data.extend(items[n_train+n_val:])
        
    # shuffle everything
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    def write_jsonl(path, dataset):
        with open(path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
                
    write_jsonl(os.path.join(args.out_dir, "joint_nlu_train.jsonl"), train_data)
    write_jsonl(os.path.join(args.out_dir, "joint_nlu_val.jsonl"), val_data)
    write_jsonl(os.path.join(args.out_dir, "joint_nlu_test.jsonl"), test_data)
    
    print(f"Saved {len(train_data)} train, {len(val_data)} val, {len(test_data)} test to {args.out_dir}/")

if __name__ == "__main__":
    # set seed for reproducible split
    random.seed(42)
    main()
