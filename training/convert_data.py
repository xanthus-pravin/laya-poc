"""Convert the prepared JSONL splits to Laya's expected format.

Input format (from prep_data.py):
  {"text": "...", "questions": {...}, "labels": {"burnout_risk": true, ...}}

Output format (Laya expects):
  {"state": "...", "questions": {...}, "gold": {"burnout_risk": {"probabilities": {"true": 1.0, "false": 0.0}}, ...}}
"""
from __future__ import annotations

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CONVERTED_DIR = os.path.join(DATA_DIR, "laya_format")
os.makedirs(CONVERTED_DIR, exist_ok=True)


def convert_row(row: dict) -> dict:
    """Convert a row from 'text'+'labels' to 'state'+'gold'."""
    labels = row["labels"]
    questions = row["questions"]
    gold = {}

    for qid, qdef in questions.items():
        qtype = qdef["type"]
        label_val = labels.get(qid)
        crit = qdef["criteria"]

        if qtype == "choice":
            keys = list(crit.keys())
            probs = {k: 0.0 for k in keys}
            if label_val in keys:
                probs[label_val] = 1.0
            elif label_val is not None:
                for k in keys:
                    if str(k) == str(label_val):
                        probs[k] = 1.0
                        break
            gold[qid] = {"probabilities": probs}

        elif qtype == "noul":
            probs = {"false": 0.0, "true": 0.0}
            if label_val is True:
                probs["true"] = 1.0
            elif label_val is False:
                probs["false"] = 1.0
            else:
                probs["true"] = 1.0
            gold[qid] = {"probabilities": probs}

        elif qtype == "score":
            n = len(crit)
            probs = {str(i): 0.0 for i in range(n)}
            idx = int(label_val) if label_val is not None else 0
            if 0 <= idx < n:
                probs[str(idx)] = 1.0
            else:
                probs["0"] = 1.0
            gold[qid] = {"probabilities": probs}

    return {
        "state": row["text"],
        "questions": questions,
        "gold": gold,
    }


def convert_split(input_path: str, output_path: str) -> int:
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            converted = convert_row(row)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    print(f"  {os.path.basename(output_path)}: {count} records")
    return count


def main():
    print("Converting data to Laya format...")
    for name in ["train", "calibrate", "test"]:
        inp = os.path.join(DATA_DIR, f"{name}.jsonl")
        out = os.path.join(CONVERTED_DIR, f"{name}.jsonl")
        if os.path.exists(inp):
            convert_split(inp, out)
        else:
            print(f"  SKIP: {inp} not found")
    print(f"\nConverted data in: {CONVERTED_DIR}")


if __name__ == "__main__":
    main()
