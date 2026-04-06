"""
src/data_loader.py
------------------
Loads and validates the raw resume dataset (CSV format).
Expected columns: 'Resume' (raw text) and 'Category' (job label).

Dataset source:
  https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset
  File: UpdatedResumeDataSet.csv
"""

import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_CSV = RAW_DATA_DIR / "UpdatedResumeDataSet.csv"


def load_raw_dataset(filepath: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """
    Load raw CSV resume dataset.

    Parameters
    ----------
    filepath : path to the CSV file

    Returns
    -------
    pd.DataFrame with columns ['Resume', 'Category']
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Download from: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset\n"
            "and place 'UpdatedResumeDataSet.csv' inside data/raw/"
        )

    df = pd.read_csv(filepath)

    # Validate expected columns
    required_cols = {"Resume", "Category"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must have columns {required_cols}. Found: {list(df.columns)}"
        )

    df = df[["Resume", "Category"]].dropna()
    df["Resume"] = df["Resume"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()

    print(f"[DataLoader] Loaded {len(df)} records | {df['Category'].nunique()} categories")
    return df


def get_category_distribution(df: pd.DataFrame) -> pd.Series:
    """Return value counts of job categories."""
    return df["Category"].value_counts()


def save_processed(df: pd.DataFrame, filename: str = "processed_resumes.csv") -> Path:
    """Save the processed DataFrame to data/processed/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"[DataLoader] Saved processed data → {out_path}")
    return out_path


def load_processed(filename: str = "processed_resumes.csv") -> pd.DataFrame:
    """Load a previously saved processed DataFrame."""
    path = PROCESSED_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_csv(path)
