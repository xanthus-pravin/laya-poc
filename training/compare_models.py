"""Compare base model vs fine-tuned model on the held-out test set."""
from __future__ import annotations

import json
import os
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL = "convaiinnovations/laya"
FINETUNED_MODEL = os.path.join(SCRIPT_DIR, "laya_finetuned")
TEST_PATH = os.path.join(SCRIPT_DIR, "data", "test.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Questions shared across all test rows (same schema)
QUESTIONS = {
    "burnout_risk": {
        "type": "noul",
        "instructions": "Does the conversation contain explicit signs of employee burnout, exhaustion, or being overwhelmed?",
        "criteria": {
            "true": "team members express exhaustion",
            "false": "team members sound comfortable"
        }
    },
    "root_cause": {
        "type": "choice",
        "instructions": "What is the primary negative driver expressed by the team in this chat?",
        "criteria": {
            "overload_and_deadlines": "overload and deadlines",
            "lack_of_support_and_isolation": "lack of support and isolation",
            "none_healthy": "none healthy"
        }
    },
    "psychological_safety": {
        "type": "score",
        "instructions": "Rate the psychological safety of the team based on their communication.",
        "criteria": ["Low: Afraid to speak up", "Medium: Frustrated but vocal", "High: Healthy collaboration"]
    }
}


def load_test_cases():
    cases = []
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cases.append({"text": row["text"], "labels": row["labels"]})
    return cases


def predict_label(agent, text):
    """Run predict and return simple predicted labels for each question."""
    result = agent.predict(text, QUESTIONS)
    answers = result["answers"]

    # noul: the "noul" field is probability of true. Threshold 0.5.
    burnout_noul = answers["burnout_risk"]["noul"]
    burnout_pred = "true" if burnout_noul > 0.5 else "false"
    burnout_conf = answers["burnout_risk"]["answer_confidence"]

    # choice: direct choice field
    root_cause_pred = answers["root_cause"]["choice"]
    root_cause_conf = answers["root_cause"]["answer_confidence"]

    # score: round the float score to nearest int index
    psych_score = answers["psychological_safety"]["score"]
    psych_pred = str(round(psych_score))
    psych_conf = answers["psychological_safety"]["answer_confidence"]

    return {
        "burnout_risk": {"pred": burnout_pred, "conf": burnout_conf, "noul": burnout_noul},
        "root_cause":    {"pred": root_cause_pred, "conf": root_cause_conf},
        "psychological_safety": {"pred": psych_pred, "conf": psych_conf, "raw_score": psych_score},
    }


def evaluate(model_label, cases, model_path):
    from laya import load

    print(f"\n{'='*70}")
    print(f"  {model_label}")
    print(f"  Path: {model_path}")
    print(f"{'='*70}")

    start = time.time()
    agent = load(model_path)
    load_time = time.time() - start
    print(f"  Model loaded in {load_time:.1f}s")

    results = {qid: {"correct": 0, "total": 0, "confs": [], "wrong": []}
               for qid in QUESTIONS}

    print(f"\n  {'#':<3} {'QID':<22} {'Predicted':<28} {'Expected':<28} {'Conf':<8} {'OK'}")
    print(f"  {'-'*3} {'-'*22} {'-'*28} {'-'*28} {'-'*8} {'-'*3}")

    for i, case in enumerate(cases):
        labels = case["labels"]
        preds = predict_label(agent, case["text"])

        for qid in QUESTIONS:
            pred_info = preds[qid]
            raw_expected = str(labels[qid])
            # Normalize: labels may be "True"/"False", predictions are "true"/"false"
            expected = raw_expected.lower()
            predicted = pred_info["pred"]
            conf = pred_info["conf"]
            r = results[qid]
            r["total"] += 1
            r["confs"].append(conf)

            correct = (predicted == expected)
            if correct:
                r["correct"] += 1
            else:
                r["wrong"].append({
                    "case": i + 1,
                    "predicted": predicted,
                    "expected": expected,
                    "conf": conf,
                    "text_preview": case["text"][:80]
                })

            ok = "Y" if correct else "N"
            pred_display = str(predicted)[:27]
            exp_display = str(expected)[:27]
            print(f"  {i+1:<3} {qid:<22} {pred_display:<28} {exp_display:<28} {conf:<8.4f} {ok}")

    # Summary
    print(f"\n  --- Per-question accuracy ---")
    total_correct = 0
    total_items = 0
    for qid, r in results.items():
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        avg_conf = sum(r["confs"]) / len(r["confs"]) if r["confs"] else 0
        total_correct += r["correct"]
        total_items += r["total"]
        print(f"  {qid:<22} acc={acc:.0%} ({r['correct']}/{r['total']})  avg_conf={avg_conf:.4f}")
    overall = total_correct / total_items if total_items > 0 else 0
    print(f"\n  OVERALL: {total_correct}/{total_items} = {overall:.0%}")
    return results


def main():
    cases = load_test_cases()
    print(f"Loaded {len(cases)} test cases from {TEST_PATH}")

    base_results = evaluate("BASE MODEL (not trained)", cases, BASE_MODEL)
    ft_results = evaluate("FINE-TUNED MODEL", cases, FINETUNED_MODEL)

    # Comparison
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Question':<22} {'Base Acc':>10} {'FT Acc':>10} {'Base Conf':>12} {'FT Conf':>10} {'Delta':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")

    for qid in QUESTIONS:
        b = base_results[qid]
        f = ft_results[qid]
        b_acc = b["correct"] / b["total"] if b["total"] > 0 else 0
        f_acc = f["correct"] / f["total"] if f["total"] > 0 else 0
        b_conf = sum(b["confs"]) / len(b["confs"]) if b["confs"] else 0
        f_conf = sum(f["confs"]) / len(f["confs"]) if f["confs"] else 0
        delta = f_acc - b_acc
        print(f"  {qid:<22} {b_acc:>10.0%} {f_acc:>10.0%} {b_conf:>12.4f} {f_conf:>10.4f} {delta:>+8.0%}")

    b_total = sum(base_results[q]["correct"] for q in base_results)
    f_total = sum(ft_results[q]["correct"] for q in ft_results)
    b_n = sum(base_results[q]["total"] for q in base_results)
    f_n = sum(ft_results[q]["total"] for q in ft_results)
    b_acc = b_total / b_n
    f_acc = f_total / f_n
    delta = f_acc - b_acc
    print(f"\n  {'Overall':<22} {b_acc:>10.0%} {f_acc:>10.0%} {'':>12} {'':>10} {delta:>+8.0%}")
    print(f"  Base: {b_total}/{b_n}  |  Fine-tuned: {f_total}/{f_n}")


if __name__ == "__main__":
    main()
