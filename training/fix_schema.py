"""Fix Laya question schemas in training/dataset.jsonl.

Changes applied to every record:
  - root_cause:     "options" -> "criteria" (choice dict, label -> description)
  - psychological_safety: "levels" -> "criteria" (score list of strings)

Usage:
    .venv/Scripts/python training/fix_schema.py
"""
from __future__ import annotations

import json

PATH = "training/dataset.jsonl"


def main() -> None:
    with open(PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for r in rows:
        # choice: options -> criteria
        qc = r["questions"]["root_cause"]
        if "options" in qc:
            qc["criteria"] = {o: o.replace("_", " ") for o in qc["options"]}
            del qc["options"]

        # score: levels -> criteria
        qs = r["questions"]["psychological_safety"]
        if "levels" in qs:
            qs["criteria"] = qs["levels"]
            del qs["levels"]

    with open(PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Fixed {len(rows)} records in {PATH}")


if __name__ == "__main__":
    main()
