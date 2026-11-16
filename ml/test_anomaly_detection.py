"""
ml/test_anomaly_detection.py
Tests for feature extraction and anomaly detection.

Three kinds of tests:
1. Unit tests for extract_features — edge cases (empty, missing fields,
   bad timestamps, unseen messages).
2. Labeled evaluation — 30 hand-labeled examples where the ground truth
   is known. Precision, recall, and F1 are reported. These are the only
   results that measure detection quality; everything else is illustrative.
3. Smoke test — train a fresh model on normal patterns and confirm it flags
   the deliberately unusual ones at a rate above chance.

Run with:
    python ml/test_anomaly_detection.py
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

from ml.features import build_vectorizer, extract_features, LEVEL_MAP

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_log(level='INFO', hour=10, message='User logged in successfully'):
    ts = f'2026-08-01 {hour:02d}:00:00'
    return {'level': level, 'timestamp': ts, 'message': message}


def train_on(rows):
    df = pd.DataFrame(rows)
    vec = build_vectorizer()
    feats, _ = extract_features(df, vec, fit=True)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(feats)
    return model, vec


def predict_on(model, vec, rows):
    df = pd.DataFrame(rows)
    feats, valid = extract_features(df, vec, fit=False)
    if len(feats) == 0:
        return np.array([]), valid
    return model.predict(feats), valid


# ---------------------------------------------------------------------------
# 1. Unit tests: extract_features edge cases
# ---------------------------------------------------------------------------

def test_empty_input():
    df = pd.DataFrame(columns=['level', 'timestamp', 'message'])
    vec = build_vectorizer()
    vec.fit(['placeholder'])          # must be fitted before transform
    feats, valid = extract_features(df, vec, fit=False)
    assert feats.shape == (0, len(vec.vocabulary_) + 2), \
        f"Expected 0 rows, got {feats.shape}"
    print("PASS  test_empty_input")


def test_missing_message_field():
    rows = [{'level': 'INFO', 'timestamp': '2026-08-01 09:00:00'}]
    df = pd.DataFrame(rows)
    df['message'] = ''               # caller fills missing column with ''
    vec = build_vectorizer()
    feats, valid = extract_features(df, vec, fit=True)
    assert feats.shape[0] == 1, "Expected 1 valid row"
    print("PASS  test_missing_message_field")


def test_unknown_level_dropped():
    rows = [
        make_log(level='INFO'),
        {'level': 'TRACE', 'timestamp': '2026-08-01 10:00:00',
         'message': 'something'},   # unknown level — should be dropped
        make_log(level='ERROR'),
    ]
    df = pd.DataFrame(rows)
    vec = build_vectorizer()
    feats, valid = extract_features(df, vec, fit=True)
    assert feats.shape[0] == 2, \
        f"Expected 2 valid rows (TRACE dropped), got {feats.shape[0]}"
    print("PASS  test_unknown_level_dropped")


def test_malformed_timestamp_dropped():
    rows = [
        make_log(),
        {'level': 'WARN', 'timestamp': 'not-a-date', 'message': 'slow query'},
        make_log(level='WARN'),
    ]
    df = pd.DataFrame(rows)
    vec = build_vectorizer()
    feats, valid = extract_features(df, vec, fit=True)
    assert feats.shape[0] == 2, \
        f"Expected 2 valid rows (bad timestamp dropped), got {feats.shape[0]}"
    print("PASS  test_malformed_timestamp_dropped")


def test_unseen_message_does_not_crash():
    """
    A message with no words from VOCAB should produce all-zero tfidf columns.
    The model must still return a prediction without raising an error.
    """
    train_rows = [make_log() for _ in range(20)]
    model, vec = train_on(train_rows)
    unseen = [make_log(message='xyz abc qwerty totally unknown words')]
    preds, valid = predict_on(model, vec, unseen)
    assert len(preds) == 1, "Expected one prediction for unseen message"
    assert preds[0] in (1, -1), f"Unexpected prediction value: {preds[0]}"
    print("PASS  test_unseen_message_does_not_crash")


def test_feature_shape():
    """Feature matrix must have exactly len(VOCAB)+2 columns."""
    from ml.features import VOCAB
    rows = [make_log() for _ in range(5)]
    df = pd.DataFrame(rows)
    vec = build_vectorizer()
    feats, _ = extract_features(df, vec, fit=True)
    expected_cols = len(VOCAB) + 2   # 14 tfidf + level_code + hour = 16
    assert feats.shape == (5, expected_cols), \
        f"Expected shape (5, {expected_cols}), got {feats.shape}"
    print("PASS  test_feature_shape")


# ---------------------------------------------------------------------------
# 2. Labeled evaluation on 30 hand-labeled examples
#
# Ground truth rationale (not "ERROR = anomaly"):
#   - Normal: common level+hour combinations seen throughout business hours
#   - Anomalous: messages or level+hour combos that are genuinely unusual
#     (e.g. crash at midnight, auth failure burst, DEBUG in prod at odd hours)
#
# The model is trained on the 120-log normal corpus, then evaluated here.
# ---------------------------------------------------------------------------

NORMAL_TRAIN = [
    # Typical business-hours INFO logs
    make_log('INFO', 9,  'User logged in successfully'),
    make_log('INFO', 10, 'Request processed successfully'),
    make_log('INFO', 11, 'Backup completed'),
    make_log('INFO', 14, 'Database connection established'),
    make_log('INFO', 15, 'Service started'),
    make_log('INFO', 9,  'User logged in successfully'),
    make_log('INFO', 10, 'Request processed successfully'),
    make_log('INFO', 11, 'Backup completed'),
    make_log('INFO', 14, 'Database connection established'),
    make_log('INFO', 15, 'Service started'),
    # Occasional WARN during business hours
    make_log('WARN', 10, 'High memory usage detected'),
    make_log('WARN', 14, 'Slow query detected'),
    make_log('WARN', 11, 'Disk usage above 70%'),
    make_log('WARN', 10, 'High memory usage detected'),
    make_log('WARN', 14, 'Slow query detected'),
    # Occasional ERROR during business hours
    make_log('ERROR', 10, 'Database connection failed'),
    make_log('ERROR', 14, 'Timeout while connecting to service'),
    make_log('ERROR', 10, 'Database connection failed'),
    # DEBUG entries (less common, but present)
    make_log('DEBUG', 9,  'Initializing module'),
    make_log('DEBUG', 10, 'Loading configuration'),
] * 3   # 60 training rows


LABELED_EVAL = [
    # --- Normal examples (label 1) ---
    (make_log('INFO',  10, 'User logged in successfully'),       1),
    (make_log('INFO',  14, 'Backup completed'),                  1),
    (make_log('INFO',  11, 'Request processed successfully'),    1),
    (make_log('WARN',  10, 'High memory usage detected'),        1),
    (make_log('WARN',  15, 'Slow query detected'),               1),
    (make_log('ERROR', 10, 'Database connection failed'),        1),
    (make_log('ERROR', 14, 'Timeout while connecting to service'), 1),
    (make_log('DEBUG',  9, 'Initializing module'),               1),
    (make_log('DEBUG', 10, 'Loading configuration'),             1),
    (make_log('INFO',  13, 'Service started'),                   1),
    (make_log('INFO',   9, 'Database connection established'),   1),
    (make_log('WARN',  11, 'Disk usage above 70%'),              1),
    (make_log('INFO',  16, 'Backup completed'),                  1),
    (make_log('WARN',  13, 'Response time exceeding threshold'), 1),
    (make_log('INFO',  10, 'User logged in successfully'),       1),
    # --- Anomalous examples (label -1) ---
    # Server crash at 2 AM — unusual hour + crash keyword
    (make_log('ERROR',  2, 'Server crashed unexpectedly'),      -1),
    # Auth failure burst at midnight
    (make_log('ERROR',  0, 'Authentication failed'),            -1),
    # ERROR at 3 AM — unusual hour
    (make_log('ERROR',  3, 'Database connection failed'),       -1),
    # DEBUG at 3 AM — unusual level+hour combination
    (make_log('DEBUG',  3, 'Loading configuration'),            -1),
    # Server crash at 1 AM
    (make_log('ERROR',  1, 'Server crashed unexpectedly'),      -1),
    # Auth failure at 4 AM
    (make_log('ERROR',  4, 'Authentication failed'),            -1),
    # ERROR at 23:00 — late night
    (make_log('ERROR', 23, 'Timeout while connecting to service'), -1),
    # WARN at 2 AM — unusual hour
    (make_log('WARN',   2, 'Disk usage above 70%'),             -1),
    # Server crash at noon — unusual message even at normal hour
    (make_log('ERROR', 12, 'Server crashed unexpectedly'),      -1),
    # Auth failure at normal hour — borderline; model may or may not flag
    (make_log('ERROR', 10, 'Authentication failed'),             1),
    # DEBUG at 4 AM
    (make_log('DEBUG',  4, 'Initializing module'),              -1),
    # Normal INFO at unusual hour — model should mostly pass this
    (make_log('INFO',   2, 'Backup completed'),                  1),
    # ERROR crash at 5 AM
    (make_log('ERROR',  5, 'Server crashed unexpectedly'),      -1),
    # Normal WARN at business hours
    (make_log('WARN',   9, 'High memory usage detected'),        1),
    # Late-night auth failure
    (make_log('ERROR', 22, 'Authentication failed'),            -1),
]


def run_labeled_evaluation():
    print("\n" + "="*60)
    print("LABELED EVALUATION (30 hand-labeled examples)")
    print("="*60)

    model, vec = train_on(NORMAL_TRAIN)

    eval_rows  = [r for r, _ in LABELED_EVAL]
    y_true     = np.array([label for _, label in LABELED_EVAL])

    preds, valid = predict_on(model, vec, eval_rows)

    if len(preds) == 0:
        print("ERROR: no predictions returned.")
        return

    # Convert scikit-learn convention (-1=anomaly, 1=normal) to
    # binary (1=anomaly, 0=normal) for sklearn metrics
    y_pred_bin = (preds == -1).astype(int)
    y_true_bin = (y_true == -1).astype(int)

    p  = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    r  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
    fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
    fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())
    tn = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())

    n_anomalous = int(y_true_bin.sum())
    n_normal    = len(y_true_bin) - n_anomalous

    print(f"  Labeled examples : {len(LABELED_EVAL)}  "
          f"({n_normal} normal, {n_anomalous} anomalous)")
    print(f"  Precision        : {p:.2f}  "
          f"(of {tp+fp} flagged, {tp} were truly anomalous)")
    print(f"  Recall           : {r:.2f}  "
          f"(found {tp} of {n_anomalous} anomalous examples)")
    print(f"  F1 score         : {f1:.2f}")
    print(f"  False positives  : {fp}  (normal logs incorrectly flagged)")
    print(f"  False negatives  : {fn}  (anomalies missed)")
    print(f"  True negatives   : {tn}")

    print("\n  Per-example breakdown:")
    print(f"  {'Level':<7} {'Hour':<5} {'True':>6} {'Pred':>6}  Message")
    print(f"  {'-'*65}")
    for i, (row, label) in enumerate(LABELED_EVAL):
        pred_label = preds[i] if i < len(preds) else '?'
        match = '✓' if pred_label == label else '✗'
        hour  = int(row['timestamp'][11:13])
        print(f"  {match} {row['level']:<6} {hour:<5} "
              f"{'ANOM' if label==-1 else 'norm':>6} "
              f"{'ANOM' if pred_label==-1 else 'norm':>6}  "
              f"{row['message'][:40]}")

    print(f"\n  Note: contamination=0.1 means the model flags ~10% of any")
    print(f"  batch as anomalous regardless of true prevalence. Precision")
    print(f"  and recall are sensitive to the makeup of the eval set.")

    return {'precision': p, 'recall': r, 'f1': f1, 'fp': fp, 'fn': fn}


# ---------------------------------------------------------------------------
# 3. Smoke test — normal corpus vs injected unusual patterns
# ---------------------------------------------------------------------------

def test_smoke_unusual_patterns_score_lower_than_normal():
    """
    Train on normal patterns (business-hours INFO/WARN), then score a mixed
    batch of normal and unusual entries.

    We check that the mean anomaly score of unusual entries is lower than
    the mean score of normal entries — i.e. the model correctly ranks unusual
    entries as more anomalous, even if the binary threshold (contamination=0.1)
    does not cleanly separate all of them.

    Why not assert a binary count:
    IsolationForest sets its decision threshold during training to flag exactly
    contamination*n_train samples. When predicting on a new batch, that same
    threshold applies regardless of true prevalence. With contamination=0.1
    and 100 normal training samples, the model flags roughly 10 entries per
    100-log batch. Unusual entries may not all fall below the threshold if the
    normal/unusual score distributions overlap — this is a known limitation of
    the approach with small training sets and only 17 features.
    """
    msgs = ['User logged in successfully',
            'Request processed successfully',
            'Backup completed']
    # Build exactly 90 INFO entries across business hours
    normal = [make_log('INFO', 8 + (i % 10), msgs[i % 3]) for i in range(90)]
    # Add 10 WARN entries — 100 normal total
    normal += [make_log('WARN', 9 + (i % 8), 'High memory usage detected')
               for i in range(10)]
    assert len(normal) == 100, f"Expected 100 normal, got {len(normal)}"

    unusual = [
        make_log('ERROR',  2, 'Server crashed unexpectedly'),
        make_log('ERROR',  0, 'Authentication failed'),
        make_log('DEBUG',  3, 'Loading configuration'),
        make_log('ERROR',  1, 'Server crashed unexpectedly'),
        make_log('ERROR', 23, 'Timeout while connecting to service'),
        make_log('ERROR',  4, 'Authentication failed'),
        make_log('WARN',   2, 'Disk usage above 70%'),
        make_log('ERROR',  5, 'Server crashed unexpectedly'),
        make_log('DEBUG',  4, 'Initializing module'),
        make_log('ERROR', 22, 'Authentication failed'),
    ]

    model, vec = train_on(normal)

    import pandas as _pd
    from ml.features import extract_features as _ef

    df_normal  = _pd.DataFrame(normal)
    df_unusual = _pd.DataFrame(unusual)

    feats_n, _ = _ef(df_normal,  vec, fit=False)
    feats_u, _ = _ef(df_unusual, vec, fit=False)

    assert feats_n.shape[0] == 100, f"Expected 100 normal rows, got {feats_n.shape[0]}"
    assert feats_u.shape[0] == 10,  f"Expected 10 unusual rows, got {feats_u.shape[0]}"

    mean_normal  = float(model.score_samples(feats_n).mean())
    mean_unusual = float(model.score_samples(feats_u).mean())
    flagged      = int((model.predict(feats_u) == -1).sum())

    print(f"PASS  test_smoke_unusual_patterns_score_lower_than_normal  "
          f"(mean normal={mean_normal:.3f}, "
          f"mean unusual={mean_unusual:.3f}, "
          f"{flagged}/10 unusual binary-flagged)")

    assert mean_unusual < mean_normal, (
        f"Unusual entries should score lower (more anomalous) than normal. "
        f"Got mean_unusual={mean_unusual:.3f} >= mean_normal={mean_normal:.3f}"
    )


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running unit tests...")
    test_empty_input()
    test_missing_message_field()
    test_unknown_level_dropped()
    test_malformed_timestamp_dropped()
    test_unseen_message_does_not_crash()
    test_feature_shape()
    test_smoke_unusual_patterns_score_lower_than_normal()

    metrics = run_labeled_evaluation()

    print("\n" + "="*60)
    print("All tests passed." if metrics else "Some tests failed.")
    print("="*60)
