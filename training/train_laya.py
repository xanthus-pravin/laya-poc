"""Convert the prepared JSONL splits to Laya's expected format and run training.

Input format (from prep_data.py):
  {"text": "...", "questions": {...}, "labels": {"burnout_risk": true, ...}}

Output format (Laya expects):
  {"state": "...", "questions": {...}, "gold": {"burnout_risk": {"probabilities": {"true": 1.0, "false": 0.0}}, ...}}

We convert hard labels to one-hot probability distributions so RLCD training works.
"""
from __future__ import annotations

import json
import os
import sys

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "laya_finetuned")

# Laya question type order: choice=0, score=1, noul=2
# From train.py QTYPES mapping
QTYPE_MAP = {"choice": 0, "score": 1, "noul": 2}


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
            # crit is a dict like {"overload_and_deadlines": "...", "lack_of_support_and_isolation": "...", "none_healthy": "..."}
            keys = list(crit.keys())
            probs = {k: 0.0 for k in keys}
            if label_val in keys:
                probs[label_val] = 1.0
            elif label_val is not None:
                # Try string match
                for k in keys:
                    if str(k) == str(label_val):
                        probs[k] = 1.0
                        break
            gold[qid] = {"probabilities": probs}

        elif qtype == "noul":
            # Boolean: true/false
            probs = {"false": 0.0, "true": 0.0}
            if label_val is True:
                probs["true"] = 1.0
            elif label_val is False:
                probs["false"] = 1.0
            else:
                probs["true"] = 1.0  # default
            gold[qid] = {"probabilities": probs}

        elif qtype == "score":
            # Ordinal levels: 0, 1, 2
            n = len(crit)
            probs = {str(i): 0.0 for i in range(n)}
            idx = int(label_val) if label_val is not None else 0
            if 0 <= idx < n:
                probs[str(idx)] = 1.0
            else:
                probs["0"] = 1.0  # fallback
            gold[qid] = {"probabilities": probs}

    return {
        "state": row["text"],
        "questions": questions,
        "gold": gold,
    }


def convert_split(input_path: str, output_path: str) -> int:
    """Read input JSONL, convert each row, write to output. Returns count."""
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
    print("=== Converting data to Laya format ===")

    splits = {
        "train": "train.jsonl",
        "calibrate": "calibrate.jsonl",
        "test": "test.jsonl",
    }

    converted_dir = os.path.join(DATA_DIR, "laya_format")
    os.makedirs(converted_dir, exist_ok=True)

    for split_name, filename in splits.items():
        inp = os.path.join(DATA_DIR, filename)
        out = os.path.join(converted_dir, f"{split_name}.jsonl")
        if os.path.exists(inp):
            convert_split(inp, out)
        else:
            print(f"  SKIP: {inp} not found")

    # Now run the training
    print("\n=== Starting Laya fine-tuning ===")
    from laya.train import finetune, TrainConfig

    train_path = os.path.join(converted_dir, "train.jsonl")
    calib_path = os.path.join(converted_dir, "calibrate.jsonl")

    config = TrainConfig(
        epochs=4,
        micro_batch=4,
        grad_accum=4,
        encoder_lr=2.5e-5,
        head_lr=1e-4,
        loss="rlcd",
        shuffle_options=("choice",),
        seed=42,
    )

    # Use the convaiinnovations/laya base checkpoint (multilingual)
    base_model = "convaiinnovations/laya"

    print(f"  Training data : {train_path}")
    print(f"  Calibration   : {calib_path}")
    print(f"  Base model    : {base_model}")
    print(f"  Output        : {OUTPUT_DIR}")
    print(f"  Epochs        : {config.epochs}")
    print(f"  Loss          : {config.loss}")
    print()

    summary = finetune(
        data=train_path,
        model_dir=base_model,
        output_dir=OUTPUT_DIR,
        config=config,
        device="auto",
    )

    print("\n=== Training Complete ===")
    print(f"  Output: {summary['output_dir']}")
    print(f"  Train items: {summary['train_items']}")
    print(f"  Calibration items: {summary['calibration_items']}")
    print(f"  Epoch losses: {[f'{l:.4f}' for l in summary['epoch_loss']]}")
    print(f"  Temperatures (choice, score, noul): {[round(t, 4) for t in summary['temperature']]}")
    if summary["skipped"]:
        print(f"  Skipped: {summary['skipped']}")


if __name__ == "__main__":
    main()
