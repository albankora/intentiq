"""
predict.py
----------
Production inference module for Clarix NLU.
Supports 17 intents across the full appointment + payment lifecycle.
"""

import os, pickle, re
from typing import Any
from nltk.stem import PorterStemmer

BASE       = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "models", "intent_classifier.pkl")

_cache: dict[str, Any] = {}
_stemmer = PorterStemmer()

# ── preprocessing (must match train.py exactly) ───────────────────────────────
STOP = {
    "i","me","my","we","our","you","your","he","she","it","the","a","an",
    "and","or","but","is","are","was","were","be","been","have","has","do",
    "does","to","of","in","on","at","for","with","can","could","would","will",
    "please","need","want","like","let","get","s","t",
}

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s@.]", " ", text)
    tokens = [_stemmer.stem(t) for t in text.split() if t not in STOP and len(t) > 1]
    return " ".join(tokens)


# ── model loader ──────────────────────────────────────────────────────────────
def _model() -> dict:
    if not _cache:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. "
                "Run 'python train.py' first."
            )
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        _cache["pipeline"] = data["pipeline"]
        _cache["intents"]  = data["intents"]
    return _cache


# ── entity extraction ─────────────────────────────────────────────────────────
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

_DATE = re.compile(
    r"\b(today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|next\s+\w+|this\s+\w+|\d{1,2}[\/\-]\d{1,2}(?:[\/\-]\d{2,4})?|"
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december(?:\s+\d{1,2}(?:st|nd|rd|th)?)?)\b",
    re.I,
)

_TIME = re.compile(
    r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)|\d{1,2}:\d{2}|"
    r"noon|midnight|morning|afternoon|evening)\b",
    re.I,
)

_PHONE = re.compile(
    r"\b(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}\b"
)

_LOCATION = re.compile(
    r"\b(london|new york|paris|berlin|tokyo|sydney|toronto|chicago|"
    r"los angeles|san francisco|boston|seattle|austin|dubai|singapore)\b",
    re.I,
)

_CURRENCY = re.compile(
    r"(\£|\$|\€|GBP|USD|EUR)\s?[\d,]+(?:\.\d{2})?|"
    r"[\d,]+(?:\.\d{2})?\s?(\£|\$|\€|GBP|USD|EUR)",
    re.I,
)

def _extract_phone(text: str):
    for match in _PHONE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if len(digits) >= 7:
            return match.group().strip()
    return None

def extract_entities(text: str) -> dict:
    emails    = _EMAIL.findall(text)
    dates     = _DATE.findall(text)
    times     = _TIME.findall(text)
    locations = _LOCATION.findall(text)
    amounts   = _CURRENCY.findall(text)
    return {
        "email":    emails[0]                       if emails    else None,
        "date":     dates[0]                        if dates     else None,
        "time":     times[0]                        if times     else None,
        "phone":    _extract_phone(text),
        "location": locations[0]                    if locations else None,
        "amount":   next((a for t in amounts for a in t if a), None),
    }


# ── action descriptions ───────────────────────────────────────────────────────
_ACTIONS = {
    # Appointment lifecycle
    "book_appointment":        "Create a new appointment booking",
    "confirm_appointment":     "Confirm and lock in the pending appointment",
    "reschedule_appointment":  "Move the appointment to a new date or time",
    "cancel_appointment":      "Cancel the appointment and remove it from the calendar",
    "check_appointment_status":"Look up the details or status of the appointment",
    "no_show_appointment":     "Record a missed or unattended appointment",
    # Calendar & meetings
    "book_meeting":            "Schedule a new meeting or call",
    "update_calendar":         "Add an event or block time on the calendar",
    "create_reminder":         "Set a timed reminder or alert",
    # Communication
    "send_email":              "Compose and send an email",
    # Payments
    "make_payment":            "Process a payment or charge",
    "request_refund":          "Initiate a refund to the original payment method",
    "check_payment_status":    "Look up the status of a transaction or balance",
    # Billing
    "get_invoice":             "Retrieve or send a billing document or receipt",
    "get_payment_history":     "Return a list of past transactions",
    "update_billing_details":  "Change the payment method or billing information",
    "dispute_charge":          "Raise a formal dispute about an incorrect charge",
}


# ── public API ────────────────────────────────────────────────────────────────
def predict(text: str) -> dict:
    """
    Classify intent and extract entities from a natural language string.

    Args:
        text: Raw user input (max 500 chars).

    Returns:
        dict with keys: intent, confidence, scores, entities, tokens, action.

    Raises:
        ValueError:   if text is empty or not a string.
        RuntimeError: if the model file has not been generated yet.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    text = text.strip()[:500]

    m       = _model()
    proc    = preprocess(text)
    intent  = m["pipeline"].predict([proc])[0]
    proba   = m["pipeline"].predict_proba([proc])[0]
    classes = m["pipeline"].classes_

    return {
        "intent":     intent,
        "confidence": round(float(max(proba)), 4),
        "scores":     {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "entities":   extract_entities(text),
        "tokens":     re.sub(r"[^\w\s]", "", text.lower()).split(),
        "action":     _ACTIONS.get(intent, "Execute the requested action"),
    }


if __name__ == "__main__":
    samples = [
        ("Book me an appointment with the doctor on Friday",       "book_appointment"),
        ("Please confirm my appointment for Monday morning",       "confirm_appointment"),
        ("I need to reschedule my appointment to next Thursday",   "reschedule_appointment"),
        ("Cancel my 3pm appointment tomorrow",                     "cancel_appointment"),
        ("What time is my appointment on Wednesday",               "check_appointment_status"),
        ("I missed my appointment this morning",                   "no_show_appointment"),
        ("Set up a team meeting for Friday at 2pm",                "book_meeting"),
        ("Block my calendar for the Berlin conference in June",    "update_calendar"),
        ("Remind me at 8am to take my medication",                 "create_reminder"),
        ("Send an email to boss@company.com about the deadline",   "send_email"),
        ("I want to pay my outstanding balance of £75",            "make_payment"),
        ("Please refund the payment I made last week",             "request_refund"),
        ("Did my payment go through yesterday",                    "check_payment_status"),
        ("Send me an invoice for my last three sessions",          "get_invoice"),
        ("Show me all payments I have made this year",             "get_payment_history"),
        ("Please update my credit card details",                   "update_billing_details"),
        ("I was charged twice and want to dispute it",             "dispute_charge"),
    ]
    print(f"\n{'─'*72}")
    print(f"  {'INPUT':<48} {'PREDICTED':<28} CONF")
    print(f"{'─'*72}")
    passed = 0
    for text, expected in samples:
        r    = predict(text)
        ok   = r["intent"] == expected
        passed += ok
        mark = "✓" if ok else "✗"
        entities = {k:v for k,v in r["entities"].items() if v}
        print(f"  {mark} {text[:46]:<48} {r['intent']:<28} {r['confidence']*100:.0f}%")
        if entities:
            print(f"    entities: {entities}")
    print(f"{'─'*72}")
    print(f"  Passed: {passed}/{len(samples)}\n")
