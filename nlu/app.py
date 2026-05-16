"""
app.py
------
Flask REST API — the only backend service for Clarix NLU.
React calls this directly on port 5001.

Start:
    python app.py
    NLU_PORT=8080 python app.py

Endpoints:
    POST /predict   { "text": "..." }  →  full NLU result
    GET  /health                       →  { "status": "ok" }
    GET  /intents                      →  list of known intents grouped by category
    GET  /history                      →  last 20 predictions
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request
from flask_cors import CORS
from predict import predict

app = Flask(__name__)
CORS(app)

_history: list[dict] = []
MAX_HISTORY = 20

INTENTS = {
    "appointment_lifecycle": [
        "book_appointment",
        "confirm_appointment",
        "reschedule_appointment",
        "cancel_appointment",
        "check_appointment_status",
        "no_show_appointment",
    ],
    "calendar_and_meetings": [
        "book_meeting",
        "update_calendar",
        "create_reminder",
    ],
    "communication": [
        "send_email",
    ],
    "payments": [
        "make_payment",
        "request_refund",
        "check_payment_status",
    ],
    "billing": [
        "get_invoice",
        "get_payment_history",
        "update_billing_details",
        "dispute_charge",
    ],
}


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "clarix-nlu", "uptime": int(time.time())})


@app.get("/intents")
def intents():
    flat = [i for group in INTENTS.values() for i in group]
    return jsonify({"intents": flat, "grouped": INTENTS, "total": len(flat)})


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

    try:
        t0     = time.time()
        result = predict(text)
        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    _history.insert(0, {
        "input":      text,
        "intent":     result["intent"],
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"],
    })
    if len(_history) > MAX_HISTORY:
        _history.pop()

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("NLU_PORT", 5001))
    total = sum(len(v) for v in INTENTS.values())
    print(f"\n✓ Clarix NLU API  →  http://localhost:{port}")
    print(f"  {total} intents across {len(INTENTS)} categories")
    print(f"  POST /predict  |  GET /health  |  GET /intents  |  GET /history\n")
    app.run(host="0.0.0.0", port=port, debug=False)
