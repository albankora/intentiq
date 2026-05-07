"""
app.py
------
Flask REST API — the only backend service.
React calls this directly; no Express layer needed.

Start:
    python app.py            (default port 5001)
    NLU_PORT=8080 python app.py

Endpoints:
    POST /predict   { "text": "..." }  →  full NLU result
    GET  /health                       →  { "status": "ok" }
    GET  /intents                      →  list of known intents
    GET  /history                      →  last 20 predictions
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request
from flask_cors import CORS
from predict import predict

app = Flask(__name__)
CORS(app)   # allow React (localhost:5173) to call us

_history: list[dict] = []
MAX_HISTORY = 20

INTENTS = [
    "reschedule_appointment",
    "cancel_appointment",
    "send_email",
    "book_meeting",
    "create_reminder",
    "update_calendar",
]


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "intent-nlu", "uptime": int(time.time())})


@app.get("/intents")
def intents():
    return jsonify({"intents": INTENTS})


@app.get("/history")
def history():
    return jsonify({"history": _history, "total": len(_history)})


@app.post("/predict")
def predict_route():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required"}), 400
    if len(text) > 500:
        return jsonify({"error": "Text must be ≤ 500 characters"}), 400

    t0     = time.time()
    result = predict(text)
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)

    # save to in-memory history
    _history.insert(0, {
        "input":      text,
        "intent":     result["intent"],
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"],
    })
    if len(_history) > MAX_HISTORY:
        _history.pop()

    return jsonify(result)


# ── start ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("NLU_PORT", 5001))
    print(f"\n✓ NLU Flask API  →  http://localhost:{port}")
    print(f"  POST /predict")
    print(f"  GET  /health")
    print(f"  GET  /intents")
    print(f"  GET  /history\n")
    app.run(host="0.0.0.0", port=port, debug=False)
