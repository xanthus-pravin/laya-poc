"""Run Laya fine-tuning on the prepared burnout dataset."""
from __future__ import annotations

import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "laya_format")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "laya_finetuned")

from laya.train import finetune, TrainConfig


def main() -> None:
    train_path = os.path.join(DATA_DIR, "train.jsonl")

    is_cuda = torch.cuda.is_available()
    micro_batch = 16 if is_cuda else 4
    device_name = "cuda" if is_cuda else ("mps" if torch.backends.mps.is_available() else "cpu")

    config = TrainConfig(
        epochs=4,
        micro_batch=micro_batch,
        grad_accum=4,
        encoder_lr=2.5e-5,
        head_lr=1e-4,
        loss="rlcd",
        shuffle_options=("choice",),
        seed=42,
    )

    base_model = "convaiinnovations/laya"

    print(f"Training data : {train_path}")
    print(f"Base model    : {base_model}")
    print(f"Output        : {OUTPUT_DIR}")
    print(f"Device        : {device_name}")
    print(f"Micro batch   : {config.micro_batch}  (grad_accum={config.grad_accum} → effective {config.micro_batch * config.grad_accum})")
    print(f"Epochs        : {config.epochs}")
    print(f"Loss          : {config.loss}")
    print(f"Shuffle opts  : {config.shuffle_options}")
    print()

    summary = finetune(
        data=train_path,
        model_dir=base_model,
        output_dir=OUTPUT_DIR,
        config=config,
        device=device_name,
    )

    print("\n=== Training Complete ===")
    print(f"  Output: {summary['output_dir']}")
    print(f"  Train items: {summary['train_items']}")
    print(f"  Calibration items: {summary['calibration_items']}")
    print(f"  Epoch losses: {[f'{l:.4f}' for l in summary['epoch_loss']]}")
    print(f"  Temperatures (choice, score, noul): {[round(t, 4) for t in summary['temperature']]}")
    if summary["skipped"]:
        print(f"  Skipped: {summary['skipped']}")

    cal = summary.get("calibration", {})
    for qtype, entry in cal.items():
        print(f"  [{qtype}] temp={entry['temperature']:.4f} acc={entry.get('accuracy', 'N/A')} "
              f"confidence={entry.get('mean_confidence', 'N/A')} "
              f"items={entry['items']} issues={entry['issues']}")


if __name__ == "__main__":
    main()
