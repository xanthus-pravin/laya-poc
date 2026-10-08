"""Score windows with Laya.

Writes out/predictions_laya.json in the same shape as run_keyword.py.

Four things this gets deliberately right:

- Preloads only english + multilingual. `Router(preload=True)` loads all three
  checkpoints including typed-decisions, which is tuned for Laya's own invoice/security/
  support workflows and is dead weight here: 2.26 GB downloaded instead of ~1.5 GB and
  about 6 s added to every cold start.
- Records routing.model and routing.reason per window. When a slice scores badly the
  first thing to check is whether those windows were answered by the wrong checkpoint.
- Persists answer_confidence, not confidence. On a choice question `confidence` is one
  minus normalised entropy -- a spread measure, not a probability. `answer_confidence` is
  the calibrated probability of the reported answer and is what any threshold must read.
- Passes min_confidence even at 0.0, because Laya only emits `abstention` and
  `abstention_threshold` when a gate is set, and 0.0 counts as set. That lets the report
  sweep thresholds from a single run.

Usage:
    python src\\run_laya.py
    python src\\run_laya.py --default multilingual --batch-size 8
    python src\\run_laya.py --model multilingual --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from questions import (
    CHOICE_QUESTIONS,
    NOUL_QUESTIONS,
    NOUL_THRESHOLD,
    QUESTIONS,
    pss_from_safety,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract(answers: Dict[str, Any]) -> Dict[str, Any]:
    """Project one Laya answer map onto the POC prediction schema."""
    row: Dict[str, Any] = {"confidence": {}, "abstention": {}}

    for qid in CHOICE_QUESTIONS:
        a = answers.get(qid) or {}
        probabilities = a.get("probabilities") or {}
        row[qid] = a.get("choice")
        row[f"{qid}_probabilities"] = probabilities
        row["confidence"][qid] = a.get("answer_confidence")
        row["abstention"][qid] = a.get("abstention")

    for qid in NOUL_QUESTIONS:
        a = answers.get(qid) or {}
        p = a.get("noul")
        row[f"{qid}_p"] = p
        row[qid] = None if p is None else bool(p >= NOUL_THRESHOLD)
        row["confidence"][qid] = a.get("answer_confidence")
        row["abstention"][qid] = a.get("abstention")

    row["pss"] = pss_from_safety(row.get("safety_probabilities") or {})
    return row


def versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in ("laya", "torch", "transformers", "tokenizers"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=PROJECT_ROOT / "training" / "data" / "windows.json")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "out" / "predictions_laya.json")
    parser.add_argument("--default", default="multilingual",
                        choices=["english", "multilingual"],
                        help="checkpoint for text the detector cannot place; romanized "
                             "Hindi and Hinglish look Latin and land here")
    parser.add_argument("--model", default=None,
                        help="pin a checkpoint for every request instead of routing")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sort-by-length", action="store_true", default=True)
    parser.add_argument("--no-sort-by-length", dest="sort_by_length", action="store_false")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-len", type=int, default=None,
                        help="per-request token budget; multilingual defaults to 1024 "
                             "and silently truncates longer windows")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.windows.exists():
        raise SystemExit(f"windows not found: {args.windows}. Run build_windows.py first.")

    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    from laya import Router

    windows: List[Dict[str, Any]] = json.loads(args.windows.read_text(encoding="utf-8"))["windows"]
    if args.limit:
        windows = windows[: args.limit]
    if not windows:
        raise SystemExit("no windows to score")

    t0 = time.perf_counter()
    router = Router(default=args.default)
    router.preload(["english", "multilingual"])
    cold_load = time.perf_counter() - t0
    print(f"cold load {cold_load:.1f}s (english + multilingual)")

    requests: List[Dict[str, Any]] = []
    for w in windows:
        req: Dict[str, Any] = {"state": w["text"], "questions": QUESTIONS}
        if args.model:
            req["model"] = args.model
        if args.max_len:
            req["max_len"] = args.max_len
        requests.append(req)

    t0 = time.perf_counter()
    results = router.predict_batch(
        requests,
        batch_size=args.batch_size,
        sort_by_length=args.sort_by_length,
        min_confidence=args.min_confidence,
    )
    elapsed = time.perf_counter() - t0

    predictions: List[Dict[str, Any]] = []
    for w, result in zip(windows, results):
        row = extract(result.get("answers") or {})
        row["window_id"] = w["window_id"]
        row["routing"] = result.get("routing") or {}
        row["usage"] = result.get("usage") or {}
        predictions.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "scorer": "laya",
                "meta": {
                    "windows": len(predictions),
                    "cold_load_seconds": round(cold_load, 3),
                    "total_seconds": round(elapsed, 3),
                    "seconds_per_window": round(elapsed / len(predictions), 4),
                    "batch_size": args.batch_size,
                    "sort_by_length": args.sort_by_length,
                    "default_model": args.default,
                    "pinned_model": args.model,
                    "min_confidence": args.min_confidence,
                    "max_len": args.max_len,
                    "questions": list(QUESTIONS),
                    "noul_threshold": NOUL_THRESHOLD,
                    "versions": versions(),
                    "source": str(args.windows),
                },
                "predictions": predictions,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    routed: Dict[str, int] = {}
    for row in predictions:
        model = (row.get("routing") or {}).get("model", "unknown")
        routed[model] = routed.get(model, 0) + 1
    print(f"scored {len(predictions)} windows in {elapsed:.3f}s "
          f"({elapsed / len(predictions) * 1000:.0f} ms/window) -> {args.output}")
    print(f"  routed: {routed}")


if __name__ == "__main__":
    main()
