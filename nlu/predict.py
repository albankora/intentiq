"""
predict.py
----------
Production inference module for Clarix NLU.
Loads the trained model and exposes predict() for use by app.py.
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

# Requires a realistic phone pattern: optional +, country/area code, min 7 digits
_PHONE = re.compile(
    r"\b(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}\b"
)

_LOCATION = re.compile(
    r"\b(london|new york|paris|berlin|tokyo|sydney|toronto|chicago|"
    r"los angeles|san francisco|boston|seattle|austin|dubai|singapore)\b",
    re.I,
)

def _extract_phone(text: str):
    """Return a phone match only if it looks like a real number (7+ digits)."""
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
    return {
        "email":    emails[0]    if emails    else None,
        "date":     dates[0]     if dates     else None,
        "time":     times[0]     if times     else None,
        "phone":    _extract_phone(text),
        "location": locations[0] if locations else None,
    }


# ── action descriptions ───────────────────────────────────────────────────────
_ACTIONS = {
    "reschedule_appointment": "Reschedule the appointment to the new date/time",
    "cancel_appointment":     "Cancel the appointment and remove it from the calendar",
    "send_email":             "Compose and send an email to the specified recipient",
    "book_meeting":           "Create a new meeting event in the calendar",
    "create_reminder":        "Set a reminder notification for the specified time",
    "update_calendar":        "Update calendar availability or add an event block",
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
        ValueError: if text is empty or not a string.
        RuntimeError: if the model file has not been generated yet.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    text = text.strip()[:500]   # hard cap — matches app.py validation

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
        "I want to reschedule my appointment to next Monday at 2pm",
        "Send an email to boss@company.com about the Q3 deadline",
        "Cancel tomorrow's 3pm meeting with the marketing team",
        "Book a meeting with John in London on Friday at 10am",
        "Remind me at 8am to take my medication",
        "Block my calendar for the Berlin conference in June",
    ]
    for s in samples:
        r = predict(s)
        entities = {k: v for k, v in r["entities"].items() if v}
        print(f"  [{r['confidence']*100:.0f}%] {r['intent']:<30} {s}")
        if entities:
            print(f"         entities: {entities}")