"""Flask REST server wrapping Laya Router.

POST /predict   – single text in, answers out
POST /predict/batch – array of texts in, answers out (preserves order)
GET  /health     – model status and versions

Payload shape:
{
  "text": "<window text>",
  "questions": { ... },          // optional; falls back to env LAYA_QUESTIONS path
  "model": "english",            // optional; pins checkpoint, disables routing
  "max_len": 1024,               // optional; token budget per request
  "min_confidence": 0.0          // optional; abstention gate
}

Batch payload: same fields wrapped in an array under the "requests" key,
or pass {"texts": [...]} for text-only batches (questions come from default).

Environment variables:
  LAYA_DEFAULT_MODEL  – english | multilingual (default: multilingual)
  LAYA_QUESTIONS      – path to a JSON file containing the questions dict
  LAYA_PRELOAD        – comma-separated models to preload (default: english,multilingual)
  LAYA_PORT           – server port (default: 8000)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = os.environ.get("LAYA_DEFAULT_MODEL", "multilingual")
QUESTIONS_PATH = os.environ.get("LAYA_QUESTIONS", str(PROJECT_ROOT / "src" / "questions.py"))
PRELOAD_MODELS = [m.strip() for m in os.environ.get("LAYA_PRELOAD", "english,multilingual").split(",") if m.strip()]
PORT = int(os.environ.get("LAYA_PORT", "8000"))

# ---------------------------------------------------------------------------
# Load questions
# ---------------------------------------------------------------------------
def _load_questions() -> Dict[str, Any]:
    path = Path(QUESTIONS_PATH) if QUESTIONS_PATH else PROJECT_ROOT / "src" / "questions.py"
    if path.suffix == ".py":
        # Import the QUESTIONS dict from the questions module
        import importlib.util
        spec = importlib.util.spec_from_file_location("questions", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.QUESTIONS  # type: ignore[attr-defined]
    # JSON file
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    return data

QUESTIONS: Dict[str, Any] = _load_questions()

# ---------------------------------------------------------------------------
# Laya router (module-level, initialised once, shared across requests)
# ---------------------------------------------------------------------------
from laya import Router

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

t0 = time.perf_counter()
router = Router(default=DEFAULT_MODEL)
router.preload(PRELOAD_MODELS)
print(f"Laya ready: {PRELOAD_MODELS} preloaded in {time.perf_counter() - t0:.1f}s")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


def _versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in ("laya", "torch", "transformers", "tokenizers"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "laya_version": __import__("laya").__version__,
        "default_model": DEFAULT_MODEL,
        "preloaded_models": PRELOAD_MODELS,
        "questions_loaded": list(QUESTIONS.keys()),
        "versions": _versions(),
    })


@app.post("/predict")
def predict():
    body = request.get_json(force=True)

    if not body or "text" not in body:
        return jsonify({"error": "request body must include 'text'"}), 400

    text: str = body["text"]
    questions: Dict[str, Any] = body.get("questions") or QUESTIONS
    print(questions)
    model: Optional[str] = body.get("model")
    max_len: Optional[int] = body.get("max_len")
    min_confidence: Optional[float] = body.get("min_confidence")

    predict_kwargs: Dict[str, Any] = {}
    if model:
        predict_kwargs["model"] = model
    if max_len:
        predict_kwargs["max_len"] = max_len
    if min_confidence is not None:
        predict_kwargs["min_confidence"] = min_confidence

    t0 = time.perf_counter()
    result = router.predict(text, questions, **predict_kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return jsonify({
        "answers": result.get("answers") or {},
        "routing": result.get("routing") or {},
        "usage": result.get("usage") or {},
        "elapsed_ms": round(elapsed_ms, 2),
    })


@app.post("/predict/batch")
def predict_batch():
    body = request.get_json(force=True)

    if not body or "requests" not in body:
        return jsonify({"error": "request body must include 'requests' (array)"}), 400

    requests_in: List[Dict[str, Any]] = body["requests"]
    if not isinstance(requests_in, list) or not requests_in:
        return jsonify({"error": "'requests' must be a non-empty array"}), 400

    shared_questions = body.get("questions") or QUESTIONS

    # Normalise to Router predict_batch format
    router_requests: List[Dict[str, Any]] = []
    for item in requests_in:
        if not isinstance(item, dict) or "text" not in item:
            return jsonify({"error": "each request must include 'text'"}), 400
        req: Dict[str, Any] = {"state": item["text"], "questions": shared_questions}
        if "model" in item:
            req["model"] = item["model"]
        if "max_len" in item:
            req["max_len"] = item["max_len"]
        router_requests.append(req)

    batch_kwargs: Dict[str, Any] = {"batch_size": body.get("batch_size", 8)}
    if "min_confidence" in body:
        batch_kwargs["min_confidence"] = body["min_confidence"]

    t0 = time.perf_counter()
    results = router.predict_batch(router_requests, **batch_kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    answers_list = []
    for r in results:
        answers_list.append({
            "answers": r.get("answers") or {},
            "routing": r.get("routing") or {},
            "usage": r.get("usage") or {},
        })

    return jsonify({
        "results": answers_list,
        "count": len(answers_list),
        "elapsed_ms": round(elapsed_ms, 2),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
