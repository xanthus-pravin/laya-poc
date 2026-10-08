"""The incumbent scorer, copied verbatim from OrgPulse-API.

`_NEGATIVE_KEYWORDS`, `_POSITIVE_KEYWORDS`, `score_message` and `burnout_numeric` are
byte-for-byte the production logic in `services/cliq_service.py` (`CliqService._score_message`
and `CliqService._burnout_numeric`).

Do not improve this file. The POC measures what production does today; a tuned baseline
would flatter the incumbent and invalidate the comparison. If OrgPulse's scorer changes,
re-copy it rather than editing here.

Only `to_poc_labels` is new, and it does nothing but project the existing output onto the
POC label space using the thresholds already in production.
"""
from typing import Any, Dict, Optional

from questions import NO_BASELINE, pss_from_label

# ---------------------------------------------------------------------------
# verbatim from OrgPulse-API/services/cliq_service.py
# ---------------------------------------------------------------------------

# Risk keywords for lightweight per-message scoring
_NEGATIVE_KEYWORDS = [
    "frustrated", "unfair", "ignored", "overwhelmed", "stressed",
    "burned out", "burnout", "exhausted", "quit", "resign", "toxic",
    "micromanage", "pressure", "deadline", "overloaded", "unclear",
    "confused", "blocked", "behind", "late", "angry", "disappointed",
    "unmotivated", "unsupported", "isolated"
]
_POSITIVE_KEYWORDS = [
    "great", "excellent", "thanks", "appreciate", "well done", "proud",
    "motivated", "excited", "collaboration", "support", "clear",
    "achieved", "success", "celebration", "progress", "improved"
]


def score_message(text: str) -> Dict[str, Any]:
    """
    Lightweight keyword-based analysis for a single message.
    Returns a dict compatible with AnalysisResult fields.
    """
    lower = text.lower()
    neg_count = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in lower)
    pos_count = sum(1 for kw in _POSITIVE_KEYWORDS if kw in lower)

    base_score = 70.0
    score = base_score - (neg_count * 8) + (pos_count * 4)
    score = max(10.0, min(100.0, score))

    if score >= 70:
        safety_meter = "green"
    elif score >= 45:
        safety_meter = "yellow"
    else:
        safety_meter = "red"

    detected_negatives = [kw for kw in _NEGATIVE_KEYWORDS if kw in lower]
    detected_positives = [kw for kw in _POSITIVE_KEYWORDS if kw in lower]

    return {
        "psychological_safety_score": round(score, 1),
        "safety_meter": safety_meter,
        "sentiment_analysis": {
            "overall": "positive" if score > 65 else "negative" if score < 50 else "neutral",
            "positive_signals": detected_positives,
            "negative_signals": detected_negatives,
        },
        "risk_indicators": {
            "burnout_signals": detected_negatives,
            "conflict_signals": [kw for kw in detected_negatives if kw in ["unfair", "ignored", "angry"]],
        },
        "burnout_indicators": {
            "overall_risk": "high" if neg_count >= 3 else "medium" if neg_count >= 1 else "low",
            "key_signals": detected_negatives,
        },
        "communication_patterns": {},
        "leadership_impact": {},
        "silence_risk": {},
        "interceptor_alerts": [],
        "leadership_suggestions": [],
        "recommendations": [],
        "anonymized_participants": [],
    }


def burnout_numeric(analysis: Dict[str, Any]) -> float:
    bi = analysis.get("burnout_indicators") or {}
    overall = bi.get("overall_risk")
    if isinstance(overall, (int, float)):
        return float(max(0.0, min(100.0, overall)))
    if isinstance(overall, str):
        return {"high": 75.0, "medium": 45.0, "low": 20.0}.get(overall.lower(), 20.0)
    signals = bi.get("key_signals") or []
    if signals:
        return min(90.0, 20.0 + len(signals) * 12.0)
    return 20.0


# ---------------------------------------------------------------------------
# POC-only: project the production output onto the POC label space
# ---------------------------------------------------------------------------

# Production thresholds, unchanged. Tuning these would flatter the baseline.
_METER_TO_SAFETY = {"green": "safe", "yellow": "guarded", "red": "unsafe"}
_RISK_TO_BURNOUT = {"high": True, "medium": True, "low": False}


def to_poc_labels(analysis: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    """Map a `score_message` result onto the POC label space.

    `conflict` and `exclusion` come back None: the keyword scorer has no comparable
    output, which is a capability gap rather than a wrong answer. Scoring it as False
    would silently credit the baseline for a decision it never makes.
    """
    meter = analysis.get("safety_meter", "yellow")
    risk = (analysis.get("burnout_indicators") or {}).get("overall_risk", "low")

    safety = _METER_TO_SAFETY.get(meter, "guarded")
    out: Dict[str, Optional[Any]] = {
        "safety": safety,
        "safety_probabilities": None,       # the scorer emits no distribution
        "burnout": _RISK_TO_BURNOUT.get(str(risk).lower(), False),
        "burnout_p": round(burnout_numeric(analysis) / 100.0, 4),
        "pss": analysis.get("psychological_safety_score", pss_from_label(safety)),
        "confidence": {},                   # no confidence of any kind
    }
    for qid in NO_BASELINE:
        out[qid] = None
        out[f"{qid}_p"] = None
    return out
