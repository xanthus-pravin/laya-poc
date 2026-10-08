"""Score windows with the incumbent keyword scorer.

Writes out/predictions_keyword.json in the same shape as run_laya.py, so compare.py can
read both without special-casing either.

Usage:
    python src\\run_keyword.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from keyword_baseline import score_message, to_poc_labels

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=PROJECT_ROOT / "training" / "data" / "windows.json")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "out" / "predictions_keyword.json")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N windows")
    args = parser.parse_args()

    if not args.windows.exists():
        raise SystemExit(f"windows not found: {args.windows}. Run build_windows.py first.")

    windows: List[Dict[str, Any]] = json.loads(args.windows.read_text(encoding="utf-8"))["windows"]
    if args.limit:
        windows = windows[: args.limit]

    predictions: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for w in windows:
        t = time.perf_counter()
        analysis = score_message(w["text"])
        row = to_poc_labels(analysis)
        row["window_id"] = w["window_id"]
        row["latency_ms"] = round((time.perf_counter() - t) * 1000, 3)
        # kept for the report's signal lists, which laya cannot produce
        row["signals"] = analysis["sentiment_analysis"]
        predictions.append(row)
    elapsed = time.perf_counter() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "scorer": "keyword",
                "meta": {
                    "windows": len(predictions),
                    "total_seconds": round(elapsed, 3),
                    "source": str(args.windows),
                },
                "predictions": predictions,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"scored {len(predictions)} windows in {elapsed:.3f}s -> {args.output}")


if __name__ == "__main__":
    main()
