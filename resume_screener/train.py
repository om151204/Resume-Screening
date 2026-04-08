"""
train.py
--------
Entry point for training the resume classification pipeline.

Run:
    python train.py

What it does:
  1. Loads the raw CSV dataset from data/raw/
  2. Cleans & preprocesses all resume texts
  3. Trains TF-IDF + Logistic Regression pipeline
  4. Evaluates on held-out test set (prints metrics)
  5. Saves the pipeline to models/resume_pipeline.pkl
  6. Saves a confusion matrix plot to data/processed/
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from src.data_loader import load_raw_dataset, save_processed
from src.preprocessor import preprocess_series
from src.model import train, save_pipeline

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm, classes: list[str], save_path: Path):
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Resume Category Classifier", fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Train] Confusion matrix saved → {save_path}")


def plot_category_distribution(y: pd.Series, save_path: Path):
    counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Resume Category Distribution", fontsize=13)
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Train] Category distribution saved → {save_path}")


def main():
    print("=" * 60)
    print("  Resume Screener — Training Pipeline")
    print("=" * 60)

    # 1. Load data
    df = load_raw_dataset()
    plot_category_distribution(df["Category"], PROCESSED_DIR / "category_distribution.png")

    # 2. Preprocess
    print("\n[Train] Cleaning resume texts (this may take a minute)...")
    df["cleaned_resume"] = preprocess_series(df["Resume"])
    save_processed(df)

    # 3. Train
    print("\n[Train] Training pipeline...")
    pipeline, metrics = train(df["cleaned_resume"], df["Category"])

    # 4. Save confusion matrix
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        metrics["classes"],
        PROCESSED_DIR / "confusion_matrix.png",
    )

    # 5. Save pipeline
    save_pipeline(pipeline)

    print("\n" + "=" * 60)
    print(f"    Training complete! Accuracy: {metrics['accuracy']:.4f}")
    print("  Model saved to: models/resume_pipeline.pkl")
    print("  Run the app:    streamlit run app/streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
