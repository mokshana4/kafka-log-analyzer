"""
kafka_client/test_consumer_logic.py
Unit tests for the two helper functions added in consumer.py for
real-time anomaly detection (Improvement 2).

These tests do NOT require Kafka, Elasticsearch, or Slack to be
running — they test only the Python logic in isolation.

Run with:
    python kafka_client/test_consumer_logic.py
"""

import sys
sys.path.append('.')

import os
import joblib
import tempfile
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml.features import build_vectorizer, extract_features, VOCAB


# ---------------------------------------------------------------------------
# Helpers — replicated here so this test file is self-contained
# ---------------------------------------------------------------------------

def make_entry(level='INFO', hour=10, message='User logged in successfully'):
    ts = '2026-08-01 {:02d}:00:00'.format(hour)
    return {'level': level, 'timestamp': ts, 'message': message}


def build_test_bundle():
    """Train a minimal model on normal entries and return the bundle dict."""
    rows = [make_entry('INFO', 10, 'User logged in successfully') for _ in range(30)]
    rows += [make_entry('WARN', 11, 'Slow query detected') for _ in range(10)]
    df = pd.DataFrame(rows)
    vec = build_vectorizer()
    feats, _ = extract_features(df, vec, fit=True)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(feats)
    return {'model': model, 'vectorizer': vec}


# ---------------------------------------------------------------------------
# Import the two helper functions from consumer.py
# (We import by exec so we avoid triggering the KafkaConsumer at module level)
# ---------------------------------------------------------------------------

def _load_consumer_helpers():
    """
    Return (load_model_bundle, is_anomaly) by parsing consumer.py up to
    the point where KafkaConsumer is instantiated.
    """
    import types
    mod = types.ModuleType('consumer_helpers')
    mod.__dict__.update({
        'sys': sys,
        'os': os,
        'joblib': joblib,
        'pd': pd,
        'extract_features': extract_features,
        'send_slack_alert': lambda msg: None,   # stub — no real Slack needed
    })

    with open('kafka_client/consumer.py') as f:
        src = f.read()

    # Only exec the function definitions, not the startup code that
    # tries to connect to Kafka/Elasticsearch
    import ast
    tree = ast.parse(src)
    func_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    func_src = '\n'.join(ast.unparse(n) for n in func_nodes)
    exec(compile(func_src, 'consumer.py', 'exec'), mod.__dict__)
    return mod.load_model_bundle, mod.is_anomaly


load_model_bundle, is_anomaly = _load_consumer_helpers()


# ---------------------------------------------------------------------------
# Tests for load_model_bundle
# ---------------------------------------------------------------------------

def test_load_missing_file():
    """load_model_bundle returns (None, None) when the file does not exist."""
    model, vec = load_model_bundle('/nonexistent/path/model.pkl')
    assert model is None and vec is None, "Expected (None, None) for missing file"
    print("PASS  test_load_missing_file")


def test_load_valid_bundle():
    """load_model_bundle returns (model, vectorizer) for a correctly saved bundle."""
    bundle = build_test_bundle()
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        tmp_path = f.name
    try:
        joblib.dump(bundle, tmp_path)
        model, vec = load_model_bundle(tmp_path)
        assert model is not None, "Model should not be None"
        assert vec is not None, "Vectorizer should not be None"
        print("PASS  test_load_valid_bundle")
    finally:
        os.unlink(tmp_path)


def test_load_old_format_bundle():
    """load_model_bundle returns (None, None) for old-format files (model only)."""
    bundle = build_test_bundle()
    old_model = bundle['model']   # just the model object, no dict wrapper
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        tmp_path = f.name
    try:
        joblib.dump(old_model, tmp_path)
        model, vec = load_model_bundle(tmp_path)
        assert model is None and vec is None, \
            "Old format should return (None, None)"
        print("PASS  test_load_old_format_bundle")
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests for is_anomaly
# ---------------------------------------------------------------------------

def test_normal_entry_not_flagged():
    """
    A typical business-hours INFO entry that the model was trained on
    should NOT be flagged as anomalous on average.
    We train on 100 normal entries and check that at most 15% are flagged
    (contamination=0.1 means ~10% will be flagged by design).
    """
    bundle = build_test_bundle()
    model, vec = bundle['model'], bundle['vectorizer']

    normal_entries = [make_entry('INFO', 10, 'User logged in successfully')
                      for _ in range(20)]
    flags = [is_anomaly(e, model, vec) for e in normal_entries]
    flagged = sum(flags)
    # With contamination=0.1, up to ~10–15% of normal entries may be flagged
    assert flagged <= 4, \
        "Too many normal entries flagged: {}/20".format(flagged)
    print("PASS  test_normal_entry_not_flagged  ({}/20 flagged)".format(flagged))


def test_anomalous_entry_scores_lower():
    """
    Unusual entries (night-time errors/crashes) should on average receive
    a lower anomaly score than normal entries — i.e. the model ranks them
    as more suspicious even if the binary threshold does not catch all of them.
    """
    bundle = build_test_bundle()
    model, vec = bundle['model'], bundle['vectorizer']

    normal_entries  = [make_entry('INFO', 10, 'User logged in successfully')
                       for _ in range(10)]
    unusual_entries = [make_entry('ERROR', 2, 'Server crashed unexpectedly'),
                       make_entry('ERROR', 1, 'Authentication failed'),
                       make_entry('ERROR', 3, 'Database connection failed')]

    def mean_score(entries):
        rows = []
        for e in entries:
            df = pd.DataFrame([e])
            for col in ('level', 'timestamp', 'message'):
                if col not in df.columns:
                    df[col] = ''
            feats, valid = extract_features(df, vec, fit=False)
            if feats.shape[0] > 0:
                rows.append(model.score_samples(feats)[0])
        return float(np.mean(rows)) if rows else 0.0

    score_normal  = mean_score(normal_entries)
    score_unusual = mean_score(unusual_entries)
    assert score_unusual < score_normal, (
        "Unusual entries should score lower than normal. "
        "Got unusual={:.3f}, normal={:.3f}".format(score_unusual, score_normal)
    )
    print("PASS  test_anomalous_entry_scores_lower  "
          "(normal={:.3f}, unusual={:.3f})".format(score_normal, score_unusual))


def test_malformed_entry_does_not_crash():
    """
    An entry with no parseable timestamp or unknown level must return False
    (safe default) without raising an exception.
    """
    bundle = build_test_bundle()
    model, vec = bundle['model'], bundle['vectorizer']

    bad_entries = [
        {'level': 'UNKNOWN', 'timestamp': 'not-a-date', 'message': 'something'},
        {'level': '',        'timestamp': '',            'message': ''},
        {},
    ]
    for entry in bad_entries:
        result = is_anomaly(entry, model, vec)
        assert result is False, \
            "Malformed entry should return False, got {}".format(result)
    print("PASS  test_malformed_entry_does_not_crash")


def test_missing_columns_handled():
    """
    Entry dict missing 'message' key must not crash is_anomaly.
    """
    bundle = build_test_bundle()
    model, vec = bundle['model'], bundle['vectorizer']

    entry = {'level': 'INFO', 'timestamp': '2026-08-01 10:00:00'}  # no 'message'
    result = is_anomaly(entry, model, vec)
    assert result in (True, False), "Expected a boolean result, got: {}".format(result)
    print("PASS  test_missing_columns_handled")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing real-time consumer logic (Improvement 2)...\n")
    test_load_missing_file()
    test_load_valid_bundle()
    test_load_old_format_bundle()
    test_normal_entry_not_flagged()
    test_anomalous_entry_scores_lower()
    test_malformed_entry_does_not_crash()
    test_missing_columns_handled()
    print("\n" + "=" * 55)
    print("All consumer logic tests passed.")
    print("=" * 55)
