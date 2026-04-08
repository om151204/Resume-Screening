"""
src/model.py
------------
Builds, trains, evaluates, and persists the ML pipeline.

Pipeline structure:
  Step 1: TfidfVectorizer  (feature_engineering.py config)
  Step 2: LogisticRegression (multi-class, one-vs-rest)

Using a sklearn Pipeline ensures:
  - No data leakage: vectorizer is fit only on training data
  - Single object to serialize/load for inference
"""

import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from src.feature_engineering import build_tfidf_vectorizer

# ── Path ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "resume_pipeline.pkl"


# ── Pipeline Builder ──────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Build an untrained sklearn Pipeline.

    Returns
    -------
    Pipeline: TfidfVectorizer → LogisticRegression
    """
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=5.0,                   # Regularization strength
            class_weight="balanced", # Handle class imbalance
            n_jobs=-1,
        )),
    ])


# ── Train & Evaluate ──────────────────────────────────────────────────────

def train(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict]:
    """
    Train the pipeline and return the model + evaluation metrics.

    Parameters
    ----------
    X : array-like of cleaned resume texts
    y : array-like of category labels
    test_size : fraction of data for evaluation
    random_state : reproducibility seed

    Returns
    -------
    (fitted_pipeline, metrics_dict)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[Model] Train size: {len(X_train)} | Test size: {len(X_test)}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    print(f"[Model] Test Accuracy: {acc:.4f}")
    print("\n[Model] Classification Report:")
    print(classification_report(y_test, y_pred))

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": list(pipeline.classes_),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }

    return pipeline, metrics


# ── Serialization ─────────────────────────────────────────────────────────

def save_pipeline(pipeline: Pipeline, path: Path = MODEL_PATH) -> Path:
    """Save the trained pipeline to disk using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[Model] Pipeline saved → {path}")
    return path


def load_pipeline(path: Path = MODEL_PATH) -> Pipeline:
    """Load a previously saved pipeline from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at: {path}\n"
            "Run 'python train.py' first to train and save the model."
        )
    return joblib.load(path)
