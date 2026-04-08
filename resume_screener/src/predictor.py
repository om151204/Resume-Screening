"""
src/predictor.py
-----------------
Inference module: given raw resume text, return predicted job category
and per-class confidence scores.

Also handles PDF and TXT file parsing.
"""

import io
from pathlib import Path

import pdfplumber
from sklearn.pipeline import Pipeline

from src.preprocessor import clean_text
from src.model import load_pipeline, MODEL_PATH


# ── File Parsing ──────────────────────────────────────────────────────────

def extract_text_from_pdf(file) -> str:
    """
    Extract text from a PDF file object (supports file path or bytes-like).

    Parameters
    ----------
    file : str | Path | bytes | file-like object

    Returns
    -------
    Extracted text string
    """
    text_parts = []

    # Handle bytes input (from Streamlit uploader)
    if isinstance(file, bytes):
        file = io.BytesIO(file)

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts)


def extract_text_from_txt(file) -> str:
    """
    Extract text from a plain-text file object.

    Parameters
    ----------
    file : bytes or file-like object

    Returns
    -------
    Decoded string
    """
    if isinstance(file, bytes):
        return file.decode("utf-8", errors="ignore")
    return file.read().decode("utf-8", errors="ignore")


def parse_uploaded_file(file_bytes: bytes, filename: str) -> str:
    """
    Route to the correct parser based on file extension.

    Parameters
    ----------
    file_bytes : raw bytes from Streamlit uploader
    filename   : original filename (used to detect extension)

    Returns
    -------
    Raw extracted text string
    """
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == ".txt":
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Upload a PDF or TXT file.")


# ── Predictor Class ───────────────────────────────────────────────────────

class ResumePredictor:
    """
    Wraps the trained pipeline for easy inference.

    Usage
    -----
    predictor = ResumePredictor()
    result = predictor.predict_from_text(raw_text)
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self.pipeline: Pipeline = load_pipeline(model_path)
        self.classes: list[str] = list(self.pipeline.classes_)

    def predict_from_text(self, raw_text: str) -> dict:
        """
        Predict job category from raw (uncleaned) resume text.

        Parameters
        ----------
        raw_text : str — raw resume content

        Returns
        -------
        dict with keys:
          - predicted_category : str
          - confidence         : float (0–1)
          - all_scores         : dict {category: probability}
          - cleaned_text       : str (for debugging)
        """
        cleaned = clean_text(raw_text)
        proba = self.pipeline.predict_proba([cleaned])[0]
        predicted_idx = proba.argmax()

        all_scores = dict(zip(self.classes, proba.tolist()))
        # Sort descending by score
        all_scores = dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True))

        return {
            "predicted_category": self.classes[predicted_idx],
            "confidence": float(proba[predicted_idx]),
            "all_scores": all_scores,
            "cleaned_text": cleaned,
        }

    def predict_from_file(self, file_bytes: bytes, filename: str) -> dict:
        """
        Parse file and predict job category.

        Parameters
        ----------
        file_bytes : raw bytes
        filename   : original filename

        Returns
        -------
        Same dict as predict_from_text, plus 'raw_text' key.
        """
        raw_text = parse_uploaded_file(file_bytes, filename)
        result = self.predict_from_text(raw_text)
        result["raw_text"] = raw_text
        return result
