"""
train.py
--------
Production trainer for Clarix NLU.
Best configuration (from train_experiment.py grid search):
  - Porter stemming
  - TF-IDF  ngram_range=(1,1), sublinear_tf=True
  - LinearSVC C=1.0 wrapped in CalibratedClassifierCV

Run:
    python train.py              # train on all data, save model
    python train.py --eval       # also print accuracy on a held-out split

To benchmark alternative models/params use train_experiment.py.

Output:
    models/intent_classifier.pkl
"""

import argparse, json, os, pickle, re, time, warnings
warnings.filterwarnings("ignore")

from nltk.stem import PorterStemmer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
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


def build_pipeline() -> Pipeline:
    """Best config from grid search on 650-example, 17-intent dataset."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 1),
            sublinear_tf=True,
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000)
        )),
    ])


def train(eval_mode: bool = False) -> None:
    with open(DATA) as f:
        data = json.load(f)

    texts   = [preprocess(d["text"]) for d in data]
    labels  = [d["intent"]           for d in data]
    intents = sorted(set(labels))

    print(f"[train] {len(texts)} examples · {len(intents)} intents")

    if eval_mode:
        X_tr, X_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=0.15, random_state=42, stratify=labels
        )
        pipeline = build_pipeline()
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        print(f"[eval]  Held-out accuracy: {accuracy_score(y_te, y_pred)*100:.2f}%")
        print(classification_report(y_te, y_pred))

    t0       = time.time()
    pipeline = build_pipeline()
    pipeline.fit(texts, labels)
    elapsed  = time.time() - t0

    with open(MODEL, "wb") as f:
        pickle.dump({"pipeline": pipeline, "intents": intents}, f)

    print(f"[train] Done in {elapsed:.2f}s → {MODEL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clarix NLU trainer")
    parser.add_argument("--eval", action="store_true",
                        help="Print accuracy on a held-out split before saving")
    args = parser.parse_args()
    train(eval_mode=args.eval)
