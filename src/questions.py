"""Typed question schema for the OrgPulse Laya POC.

Single source of truth for the label space. The labelling tool, both scorers and the
comparison all import from here, so a change to a label propagates everywhere at once.

Design constraints, from Laya's documented limits:

- Every `noul` carries explicit true/false criteria. Issue #156: the English checkpoint
  returns a confident "no" for clearly positive input when criteria are omitted, because
  the bare `false:`/`true:` option labels carry the whole decision. Keys other than
  "true"/"false" are rejected by the library.
- No `score` questions. It is Laya's weakest primitive (SST-5 0.372) and
  laya-multilingual has a position bias that makes it rarely pick the first level.
  The 0-100 PSS number is composed from choice probabilities instead.
- Four options maximum. Options share a fixed head_max_len token budget; past roughly
  head_max_len/4 options each label gets too few tokens to stay distinct.
"""
from typing import Any, Dict, Mapping

QUESTIONS: Dict[str, Dict[str, Any]] = {
    "safety": {
        "type": "choice",
        "instructions": "How psychologically safe does this conversation feel?",
        "criteria": {
            "safe": "people disagree openly, ask questions, and admit mistakes without fear",
            "guarded": "people are polite but withholding, avoiding the real issue",
            "unsafe": "people are silenced, blamed, dismissed, or afraid to speak",
        },
    },
    "burnout": {
        "type": "noul",
        "instructions": "Does anyone show signs of exhaustion or sustained overload?",
        "criteria": {
            "true": "someone sounds exhausted, overloaded, or unable to keep up",
            "false": "nobody expresses exhaustion or overload",
        },
    },
    "conflict": {
        "type": "noul",
        "instructions": "Is there interpersonal conflict or hostility here?",
        "criteria": {
            "true": "people are blaming, dismissing, or attacking each other",
            "false": "the tone is cooperative or neutral",
        },
    },
    "exclusion": {
        "type": "noul",
        "instructions": "Is someone being shut down, talked over, or excluded?",
        "criteria": {
            "true": "a participant's input is ignored, overridden, or dismissed",
            "false": "participants are heard and responded to",
        },
    },
}

CHOICE_QUESTIONS = tuple(qid for qid, q in QUESTIONS.items() if q["type"] == "choice")
NOUL_QUESTIONS = tuple(qid for qid, q in QUESTIONS.items() if q["type"] == "noul")

SAFETY_LABELS = ("safe", "guarded", "unsafe")

# Decision threshold applied to a noul P(true) to get a boolean. A POC default, not a
# fitted value: the real cut comes from the calibration split.
NOUL_THRESHOLD = 0.5

# Anchors for composing the 0-100 psychological safety score as an expected value over
# the safety distribution. These are placeholders and MUST be fitted so the resulting
# distribution matches what CliqService._score_message currently produces -- the
# dashboard has trend history on the old scale, and an unanchored swap renders as an
# org-wide event that never happened.
SAFETY_ANCHORS: Dict[str, float] = {"safe": 100.0, "guarded": 55.0, "unsafe": 20.0}

# Laya-only capabilities: the keyword scorer produces no comparable output, so these are
# reported standalone rather than scored as a win against it.
NO_BASELINE = ("conflict", "exclusion")


def pss_from_safety(probabilities: Mapping[str, float]) -> float:
    """Expected 0-100 safety score over the safety distribution."""
    if not probabilities:
        return 0.0
    total = sum(probabilities.values())
    if total <= 0:
        return 0.0
    weighted = sum(SAFETY_ANCHORS[label] * p for label, p in probabilities.items())
    return round(weighted / total, 1)


def pss_from_label(label: str) -> float:
    """Fallback for a scorer that emits a bare label with no distribution."""
    return SAFETY_ANCHORS.get(label, 55.0)
