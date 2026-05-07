"""
train.py
--------
Train the intent classifier:
  TF-IDF (unigrams + bigrams) → Logistic Regression

Run:
    python train.py

Outputs:
    models/intent_classifier.pkl
"""

import json, os, pickle, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE   = os.path.dirname(__file__)
DATA   = os.path.join(BASE, "data", "intents.json")
MODEL  = os.path.join(BASE, "models", "intent_classifier.pkl")
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)

# ── stop words ────────────────────────────────────────────────────────────────
STOP = {
    "i","me","my","we","our","you","your","he","she","it","the","a","an",
    "and","or","but","is","are","was","were","be","been","have","has","do",
    "does","to","of","in","on","at","for","with","can","could","would","will",
    "please","need","want","like","let","get","s","t",
}

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s@.]", " ", text)
    return " ".join(t for t in text.split() if t not in STOP and len(t) > 1)

# ── load ──────────────────────────────────────────────────────────────────────
with open(DATA) as f:
    data = json.load(f)

texts  = [preprocess(d["text"]) for d in data]
labels = [d["intent"]           for d in data]

print(f"Loaded {len(texts)} examples · {len(set(labels))} intents")
for intent in sorted(set(labels)):
    print(f"  {intent:<30} {labels.count(intent)} examples")

# ── pipeline ──────────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
    ("clf",   LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs")),
])

# ── train / eval ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
pipeline.fit(X_train, y_train)
print("\nClassification report (held-out 20%):")
print(classification_report(y_test, pipeline.predict(X_test)))

pipeline.fit(texts, labels)   # final fit on all data

# ── save ──────────────────────────────────────────────────────────────────────
with open(MODEL, "wb") as f:
    pickle.dump({"pipeline": pipeline, "intents": sorted(set(labels))}, f)
print(f"Model saved → {MODEL}")

# ── smoke test ────────────────────────────────────────────────────────────────
SMOKE = [
    ("Reschedule my appointment to next Thursday",  "reschedule_appointment"),
    ("Send an email to boss@work.com",              "send_email"),
    ("Cancel tomorrow's 3pm meeting",              "cancel_appointment"),
    ("Book a team meeting next Friday at 2pm",      "book_meeting"),
    ("Remind me at 8am to take my medication",      "create_reminder"),
    ("Block my calendar for the Berlin conference", "update_calendar"),
]
print("\nSmoke tests:")
for text, expected in SMOKE:
    pred = pipeline.predict([preprocess(text)])[0]
    conf = max(pipeline.predict_proba([preprocess(text)])[0]) * 100
    ok   = "✓" if pred == expected else "✗"
    print(f"  {ok} [{conf:5.1f}%]  {pred:<30}  {text}")
