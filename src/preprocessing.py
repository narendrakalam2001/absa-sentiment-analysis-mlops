# ============================================================
# PREPROCESSING — Aspect-Based Sentiment Analysis (ABSA)
# ============================================================
# Pipeline:
#   1. TextCleaner     — lowercase, punct strip, stopwords
#   2. AspectContextTransformer — aspect-aware context embedding
#   3. TF-IDF + numeric feature ColumnTransformer (dual pipeline)
#   4. clone(pipeline) pattern for shared-object safety
# ============================================================

import re
import logging
import numpy as np
import pandas as pd

from typing import List, Tuple, Optional

from sklearn.base             import BaseEstimator, TransformerMixin, clone
from sklearn.compose          import ColumnTransformer
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    TFIDF_MAX_FEATURES, TFIDF_SUBLINEAR, NGRAM_RANGE, RANDOM_STATE
)

logger = logging.getLogger(__name__)


# ============================================================
# TEXT CLEANER — sklearn-compatible transformer
# ============================================================

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans raw review text:
      - lowercase
      - remove URLs, emails, special chars
      - collapse whitespace
      - optional stopword removal
      - optional lemmatization (if spacy available)

    Preserves negation context (not, n't kept).
    """

    NEGATION_SAFE = {"not", "no", "never", "n't", "nor", "neither",
                     "nothing", "nobody", "nowhere", "hardly", "barely"}

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize:        bool = False,
    ):
        self.remove_stopwords = remove_stopwords
        self.lemmatize        = lemmatize

    def fit(self, X, y=None):
        if self.remove_stopwords:
            self._stopwords = self._load_stopwords()
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.Series):
            X = X.tolist()
        return [self._clean(t) for t in X]

    def _clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)        # URLs
        text = re.sub(r"\S+@\S+",        " ", text)        # emails
        text = re.sub(r"[^a-z0-9'!\?\s]"," ", text)       # special chars (keep ' for n't)
        text = re.sub(r"\s+",            " ", text).strip()

        if self.remove_stopwords:
            words = text.split()
            # Keep negation words always
            words = [
                w for w in words
                if w in self.NEGATION_SAFE or w not in self._stopwords
            ]
            text  = " ".join(words)

        return text

    def _load_stopwords(self) -> set:
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words("english")) - self.NEGATION_SAFE
        except Exception:
            # Minimal fallback list
            return {
                "the", "a", "an", "is", "it", "its", "was", "were",
                "are", "i", "we", "they", "he", "she", "to", "of",
                "for", "in", "on", "at", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "will",
                "would", "should", "could", "may", "might", "this",
                "that", "these", "those", "my", "your", "our", "their"
            } - self.NEGATION_SAFE

    def get_feature_names_out(self, input_features=None):
        return np.array(["cleaned_text"], dtype=object)


# ============================================================
# ASPECT CONTEXT TRANSFORMER
# ============================================================

class AspectContextTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts a window of ±N words around the aspect term.
    If aspect not found verbatim, uses full text.
    Input: Series/list of strings in format 'aspect [SEP] sentence'
    Output: list of context strings
    """

    def __init__(self, window: int = 5, cleaner: Optional[TextCleaner] = None):
        self.window  = window
        self.cleaner = cleaner or TextCleaner()

    def fit(self, X, y=None):
        self.cleaner.fit(X)
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.Series):
            X = X.tolist()
        results = []
        for text in X:
            aspect, sentence = self._split(text)
            ctx = self._context(sentence, aspect)
            results.append(ctx)
        return results

    def _split(self, text: str) -> Tuple[str, str]:
        if "[SEP]" in str(text):
            parts    = str(text).split("[SEP]", 1)
            aspect   = parts[0].strip()
            sentence = parts[1].strip() if len(parts) > 1 else ""
        else:
            aspect, sentence = "GENERAL", str(text)
        return aspect, sentence

    def _context(self, sentence: str, aspect: str) -> str:
        words = sentence.lower().split()
        a_words = aspect.lower().split()
        for i in range(len(words) - len(a_words) + 1):
            if words[i: i + len(a_words)] == a_words:
                start = max(0, i - self.window)
                end   = min(len(words), i + len(a_words) + self.window)
                return " ".join(words[start:end])
        return sentence  # fallback

    def get_feature_names_out(self, input_features=None):
        return np.array(["aspect_context"], dtype=object)


# ============================================================
# TFIDF PIPELINE BUILDER — for text columns
# ============================================================

def build_tfidf_pipeline(
    max_features: int  = TFIDF_MAX_FEATURES,
    ngram_range:  tuple = NGRAM_RANGE,
    sublinear_tf: bool  = TFIDF_SUBLINEAR,
) -> Pipeline:
    """
    Text → clean → TF-IDF
    Stateless (cleaner fitted inside pipeline).
    """
    return Pipeline([
        ("cleaner", TextCleaner(remove_stopwords=True)),
        ("tfidf",   TfidfVectorizer(
            max_features = max_features,
            ngram_range  = ngram_range,
            sublinear_tf = sublinear_tf,
            min_df       = 2,
            max_df       = 0.95,
            analyzer     = "word",
            token_pattern= r"(?u)\b\w+\b",
        ))
    ])


# ============================================================
# FULL FEATURE PREPROCESSOR
# ============================================================

def build_preprocessors(
    text_cols: List[str],
    num_cols:  List[str],
    bin_cols:  List[str],
    X_train:   pd.DataFrame,
) -> Tuple["ColumnTransformer", "ColumnTransformer", List[str]]:
    """
    Builds two ColumnTransformers:
        preprocessor_main     — TF-IDF + scaled numerics  (for all models)
        preprocessor_context  — context TF-IDF only       (ablation / fast models)

    Returns:
        preprocessor_main, preprocessor_context, feature_order
    """

    # Primary text column = 'combined_text'
    primary_text_col = "combined_text" if "combined_text" in text_cols else text_cols[0]

    # ── Scaled numeric transformer ────────────────────────────
    num_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    # ── Main preprocessor: primary TF-IDF + numeric ───────────
    main_transformers = [
        ("tfidf_combined", build_tfidf_pipeline(), primary_text_col),
    ]

    # Context TF-IDF if column exists
    if "aspect_in_context" in text_cols and "aspect_in_context" in X_train.columns:
        main_transformers.append(
            ("tfidf_context", build_tfidf_pipeline(max_features=5000, ngram_range=(1, 2)),
             "aspect_in_context")
        )

    if num_cols:
        main_transformers.append(("numeric", num_transformer, num_cols))

    if bin_cols:
        main_transformers.append(("binary", "passthrough", bin_cols))

    preprocessor_main = ColumnTransformer(
        transformers = main_transformers,
        remainder    = "drop",
        sparse_threshold = 0.0,    # dense output for consistency
    )

    # ── Context-only preprocessor (for tree/fast models) ──────
    # Use clone to prevent shared fitted state between the two
    context_transformers = [
        ("tfidf_combined", build_tfidf_pipeline(), primary_text_col),
    ]
    if num_cols:
        context_transformers.append(("numeric", Pipeline([("scaler", MinMaxScaler())]), num_cols))
    if bin_cols:
        context_transformers.append(("binary", "passthrough", bin_cols))

    preprocessor_context = ColumnTransformer(
        transformers = context_transformers,
        remainder    = "drop",
        sparse_threshold = 0.0,
    )

    feature_order = ([primary_text_col] +
                     (["aspect_in_context"] if "aspect_in_context" in text_cols else []) +
                     num_cols + bin_cols)

    logger.info("Preprocessors built  |  feature_order=%s", feature_order)

    return preprocessor_main, preprocessor_context, feature_order


# ============================================================
# SAFE FEATURE COUNT — prevents SelectKBest crash
# ============================================================

def safe_k(requested_k: int, preprocessor: ColumnTransformer, X_sample: pd.DataFrame) -> int:
    """
    Returns min(requested_k, n_output_features_of_preprocessor).
    """
    try:
        fitted = clone(preprocessor).fit(X_sample)
        from scipy.sparse import issparse
        arr = fitted.transform(X_sample[:5])
        if issparse(arr):
            arr = arr.toarray()
        n_features = arr.shape[1]
    except Exception as e:
        logger.warning("safe_k estimation failed: %s — using requested_k=%d", e, requested_k)
        return requested_k

    k = min(requested_k, max(1, n_features))
    logger.info("safe_k: requested=%d  available=%d  using=%d", requested_k, n_features, k)
    return k
