"""
ml/features.py
Shared feature extraction for training and inference.
Both train_model.py and anomaly_detection.py import from here
so they are guaranteed to apply identical transformations.
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

LEVEL_MAP = {'DEBUG': 0, 'INFO': 1, 'WARN': 2, 'ERROR': 3}

# Fixed vocabulary matched exactly against the actual word forms
# that appear in the log messages.
#
# Bug fixed (was broken before):
#   'crash'    → 'crashed'      (messages say "crashed unexpectedly")
#   'success'  → 'successfully' (messages say "logged in successfully")
#   'login'    → 'logged'       (messages say "User logged in")
#   'initializ'→ 'initializing' (messages say "Initializing module")
#   'error'    → removed        (the word "error" never appears in any
#                                message body — it is the log LEVEL, not
#                                the message text)
#
# 'authentication', 'connection', 'failed', 'timeout', 'memory',
# 'disk', 'slow', 'threshold', 'completed', 'started' were already correct.
VOCAB = [
    'crashed', 'failed', 'timeout', 'authentication',
    'connection', 'memory', 'disk', 'slow', 'threshold',
    'successfully', 'completed', 'started', 'logged', 'initializing',
]


def parse_timestamp(ts_str):
    """
    Return hour-of-day integer from a timestamp string.
    Returns -1 on any parse failure so the caller can decide what to do.
    """
    if not isinstance(ts_str, str):
        return -1
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            return pd.to_datetime(ts_str, format=fmt).hour
        except (ValueError, TypeError):
            pass
    # Fallback: pandas flexible parser
    try:
        return pd.to_datetime(ts_str).hour
    except Exception:
        return -1


def build_vectorizer():
    """
    Return a fitted-ready TfidfVectorizer restricted to the fixed vocabulary.
    Calling fit() or fit_transform() on this object always uses the same
    feature space, so train and inference produce columns in the same order.
    Vocabulary has 14 words (down from 15 — 'error' was removed because
    the word never appears in message bodies, only in the level field).
    """
    return TfidfVectorizer(
        vocabulary=VOCAB,
        analyzer='word',
        token_pattern=r'[a-zA-Z]+',
        lowercase=True,
    )


def extract_features(df, vectorizer, fit=False):
    """
    Given a DataFrame with columns [timestamp, level, message],
    return a numpy array of shape (n_samples, n_features).

    Parameters
    ----------
    df         : pd.DataFrame  Must contain 'level', 'timestamp', 'message'.
    vectorizer : TfidfVectorizer  The shared vectorizer from the saved pipeline.
    fit        : bool  True during training (calls fit_transform), False during inference.

    Returns
    -------
    np.ndarray of shape (n_valid, 16)  — level_code, hour, 14 tfidf columns.
    valid_mask : boolean Series aligned with df.index indicating kept rows.
    """
    if df.empty:
        empty = np.empty((0, len(VOCAB) + 2), dtype=float)
        return empty, pd.Series(dtype=bool)

    df = df.copy()

    # --- Level ---
    df['level_code'] = df['level'].map(LEVEL_MAP)

    # --- Hour ---
    df['hour'] = df['timestamp'].apply(parse_timestamp)

    # Drop rows we cannot use: unknown level or unparseable timestamp
    valid = df['level_code'].notna() & (df['hour'] >= 0)
    df = df[valid]

    if df.empty:
        empty = np.empty((0, len(VOCAB) + 2), dtype=float)
        return empty, valid

    # --- Message text ---
    messages = df['message'].fillna('').astype(str)
    if fit:
        tfidf = vectorizer.fit_transform(messages).toarray()
    else:
        tfidf = vectorizer.transform(messages).toarray()

    structured = df[['level_code', 'hour']].values.astype(float)
    features = np.hstack([structured, tfidf])
    return features, valid
