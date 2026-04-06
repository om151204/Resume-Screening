"""
src/feature_engineering.py
---------------------------
TF-IDF feature extraction configuration.

This module defines the TfidfVectorizer settings used inside the sklearn
Pipeline so that the same parameters are shared between training and inference.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# ── TF-IDF Configuration ─────────────────────────────────────────────────

TFIDF_PARAMS = dict(
    max_features=5000,        # Top 5000 terms by TF-IDF score
    ngram_range=(1, 2),       # Unigrams + bigrams
    sublinear_tf=True,        # Apply log normalization to TF
    min_df=2,                 # Ignore terms in fewer than 2 docs
    max_df=0.95,              # Ignore terms in more than 95% of docs
    strip_accents="unicode",
    analyzer="word",
)


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Return a configured (unfitted) TF-IDF vectorizer."""
    return TfidfVectorizer(**TFIDF_PARAMS)
