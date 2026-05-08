"""
train_experiment.py
-------------------
Development / research trainer for Clarix NLU.
Run this when you want to:
  - Benchmark multiple models head-to-head
  - Run grid search to find better hyperparameters
  - Validate a new dataset before committing to production

NOT used in production. Results feed back into train.py's hardcoded config.

Run:
    python train_experiment.py
"""

import json, os, pickle, re, time, warnings
warnings.filterwarnings("ignore")

from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE  = os.path.dirname(__file__)
DATA  = os.path.join(BASE, "data", "intents.json")
MODEL = os.path.join(BASE, "models", "intent_classifier.pkl")
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)

_stemmer = PorterStemmer()

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


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Converts sparse TF-IDF matrix to dense array for MLP."""
    def fit(self, X, y=None): return self
    def transform(self, X):   return X.toarray()


# ── load data ─────────────────────────────────────────────────────────────────
with open(DATA) as f:
    data = json.load(f)

texts  = [preprocess(d["text"]) for d in data]
labels = [d["intent"]           for d in data]

print(f"\n{'='*60}")
print(f"  CLARIX — Experiment Trainer")
print(f"{'='*60}")
print(f"\nDataset: {len(texts)} examples · {len(set(labels))} intents")
for intent in sorted(set(labels)):
    print(f"  {intent:<30} {labels.count(intent)} examples")

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.15, random_state=42, stratify=labels
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── model zoo ─────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=2000, solver="lbfgs")),
    ]),
    "LinearSVC": Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("clf",   CalibratedClassifierCV(LinearSVC(max_iter=2000))),
    ]),
    "MLP Neural Net": Pipeline([
        ("tfidf",  TfidfVectorizer(sublinear_tf=True, max_features=3000)),
        ("dense",  DenseTransformer()),
        ("clf",    MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                  early_stopping=False, random_state=42)),
    ]),
}

# ── benchmark ─────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  BENCHMARK: 5-fold cross-validation")
print(f"{'─'*60}")

results = {}
for name, pipeline in models.items():
    t0     = time.time()
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                              scoring="accuracy", n_jobs=1)
    elapsed = time.time() - t0
    results[name] = {"mean": scores.mean(), "std": scores.std(), "time": elapsed}
    bar = "█" * int(scores.mean() * 40)
    print(f"\n  {name}")
    print(f"  CV Accuracy : {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%  [{elapsed:.1f}s]")
    print(f"  [{bar:<40}]")
    print(f"  Folds       : {' '.join(f'{s*100:.1f}%' for s in scores)}")

# ── winner ────────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]["mean"])
print(f"\n{'─'*60}")
print(f"  WINNER: {best_name}  ({results[best_name]['mean']*100:.2f}%)")
print(f"{'─'*60}")

# ── grid search ───────────────────────────────────────────────────────────────
print(f"\n  Grid search on {best_name}…")

param_grids = {
    "Logistic Regression": {
        "tfidf__ngram_range":  [(1,1),(1,2),(1,3)],
        "tfidf__max_features": [5000, 8000, None],
        "tfidf__sublinear_tf": [True, False],
        "clf__C":              [0.5, 1.0, 5.0, 10.0],
    },
    "LinearSVC": {
        "tfidf__ngram_range":    [(1,1),(1,2),(1,3)],
        "tfidf__max_features":   [5000, 8000, None],
        "tfidf__sublinear_tf":   [True, False],
        "clf__estimator__C":     [0.1, 0.5, 1.0, 5.0],
    },
    "MLP Neural Net": {
        "tfidf__ngram_range":        [(1,1),(1,2)],
        "tfidf__max_features":       [3000, 5000],
        "clf__hidden_layer_sizes":   [(64,),(128,64),(256,128)],
        "clf__alpha":                [0.0001, 0.001],
    },
}

grid = GridSearchCV(
    models[best_name], param_grids[best_name],
    cv=cv, scoring="accuracy", n_jobs=1, verbose=0
)
grid.fit(X_train, y_train)
print(f"  Best params : {grid.best_params_}")
print(f"  Best CV acc : {grid.best_score_*100:.2f}%")

# ── test set ──────────────────────────────────────────────────────────────────
best_pipeline = grid.best_estimator_
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

print(f"\n{'─'*60}")
print(f"  TEST SET  ({len(X_test)} held-out examples)")
print(f"{'─'*60}")
print(f"  Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"\n{classification_report(y_test, y_pred)}")

# ── smoke tests ───────────────────────────────────────────────────────────────
SMOKE = [
    ("Reschedule my appointment to next Thursday 2pm",  "reschedule_appointment"),
    ("Send an email to boss@work.com about the report", "send_email"),
    ("Cancel tomorrow's 3pm meeting",                  "cancel_appointment"),
    ("Book a team meeting next Friday at 10am",         "book_meeting"),
    ("Remind me at 8am to take my medication",          "create_reminder"),
    ("Block my calendar for the Berlin conference",     "update_calendar"),
    ("Can we push the appointment to next week",        "reschedule_appointment"),
    ("Drop a message to the client about the delay",    "send_email"),
    ("I need to call off the Monday standup",           "cancel_appointment"),
    ("Set up a sync with the design team on Tuesday",   "book_meeting"),
    ("Alert me before the client call starts",          "create_reminder"),
    ("Put the product launch date on my calendar",      "update_calendar"),
]

print(f"\n{'─'*60}")
print("  SMOKE TESTS")
print(f"{'─'*60}")
passed = 0
for text, expected in SMOKE:
    proc  = preprocess(text)
    pred  = best_pipeline.predict([proc])[0]
    proba = best_pipeline.predict_proba([proc])[0]
    conf  = max(proba) * 100
    ok    = pred == expected
    passed += ok
    print(f"  {'✓' if ok else '✗'} [{conf:5.1f}%]  {pred:<30}  {text}")

print(f"\n  Result: {passed}/{len(SMOKE)} smoke tests passed")
print(f"\n{'─'*60}")
print(f"  → Copy best params into train.py's build_pipeline() to lock them in.")
print(f"{'='*60}\n")
