"""
src/preprocessor.py
--------------------
Text cleaning and NLP preprocessing for resume text.

Steps:
  1. Lowercase
  2. Remove URLs, emails, phone numbers
  3. Remove special characters & digits
  4. Tokenize
  5. Remove stopwords
  6. Lemmatize
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── Download NLTK resources (idempotent) ──────────────────────────────────
import nltk


def _download_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


_download_nltk_resources()

# ── Constants ─────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ── Core Cleaning Functions ───────────────────────────────────────────────

def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_emails(text: str) -> str:
    return re.sub(r"\S+@\S+", " ", text)


def remove_phone_numbers(text: str) -> str:
    return re.sub(r"\+?\d[\d\s\-().]{7,}\d", " ", text)


def remove_special_characters(text: str) -> str:
    # Keep only alphabetic characters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text)


def tokenize(text: str) -> list[str]:
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def lemmatize(tokens: list[str]) -> list[str]:
    return [LEMMATIZER.lemmatize(t) for t in tokens]


# ── Main Pipeline Function ────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline.

    Parameters
    ----------
    text : raw resume string

    Returns
    -------
    Cleaned, lemmatized string ready for TF-IDF vectorization.
    """
    text = text.lower()
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = remove_special_characters(text)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    return " ".join(tokens)


def preprocess_series(series) -> "pd.Series":
    """Apply clean_text to a pandas Series."""
    return series.apply(clean_text)
