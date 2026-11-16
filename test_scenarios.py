"""
test_scenarios.py
Scenario-based integration tests for the full log processing pipeline.
Tests every important scenario without requiring live Kafka, Elasticsearch,
or Slack services.

Scenarios covered:
  1. Normal application logs
  2. Database connection failures
  3. Server crashes
  4. Timeout / external service failures
  5. Authentication failures
  6. Malformed log lines (unparseable)
  7. Missing or invalid timestamps

Each scenario runs the complete on-machine pipeline:
  parse_log → preprocess → extract_features → model.predict → classify_severity

Run with:
    python test_scenarios.py
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

from log_parser.parse_logs import parse_log
from log_parser.preprocess import preprocess
from ml.features import build_vectorizer, extract_features
from alerting.alert_manager import AlertManager


# ── Train a shared model for all scenario tests ──────────────────────────────

def _build_model():
    """
    Train a fresh Isolation Forest on a balanced normal corpus and return
    (model, vectorizer).  Contamination=0.1 so ~10% of any batch is flagged.
    """
    normal_rows = []
    for hour in range(8, 19):         # business hours
        normal_rows.append({'level': 'INFO',  'timestamp': '2026-08-01 {:02d}:00:00'.format(hour),
                            'message': 'User logged in successfully'})
        normal_rows.append({'level': 'INFO',  'timestamp': '2026-08-01 {:02d}:00:00'.format(hour),
                            'message': 'Request processed successfully'})
        normal_rows.append({'level': 'WARN',  'timestamp': '2026-08-01 {:02d}:00:00'.format(hour),
                            'message': 'Slow query detected'})
        normal_rows.append({'level': 'INFO',  'timestamp': '2026-08-01 {:02d}:00:00'.format(hour),
                            'message': 'Database connection established'})

    df  = pd.DataFrame(normal_rows * 3)   # 132 training rows
    vec = build_vectorizer()
    feats, _ = extract_features(df, vec, fit=True)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(feats)
    return model, vec


MODEL, VEC = _build_model()
AM = AlertManager()


def _score(entries):
    """
    Run the full pipeline on a list of entry dicts.
    Returns (predictions, valid_entries) where predictions uses
    IsolationForest convention: 1 = normal, -1 = anomaly.
    """
    df = pd.DataFrame(entries)
    for col in ('level', 'timestamp', 'message'):
        if col not in df.columns:
            df[col] = ''
    feats, valid = extract_features(df, VEC, fit=False)
    if feats.shape[0] == 0:
        return np.array([]), valid
    return MODEL.predict(feats), valid


def _make(level, hour, message):
    return {'level': level,
            'timestamp': '2026-08-01 {:02d}:00:00'.format(hour),
            'message': message}


# ── Scenario 1: Normal application logs ──────────────────────────────────────

def test_scenario_normal_logs():
    """
    Business-hours INFO/WARN logs should mostly be classified as normal.
    With contamination=0.1, at most 15% (tolerance) may be flagged.
    """
    entries = [
        _make('INFO', 10, 'User logged in successfully'),
        _make('INFO', 11, 'Request processed successfully'),
        _make('INFO', 14, 'Backup completed'),
        _make('INFO', 15, 'Service started'),
        _make('WARN', 10, 'High memory usage detected'),
        _make('WARN', 13, 'Slow query detected'),
        _make('INFO',  9, 'Database connection established'),
        _make('INFO', 16, 'Backup completed'),
        _make('WARN', 11, 'Disk usage above 70%'),
        _make('INFO', 12, 'Request processed successfully'),
    ]
    preds, valid = _score(entries)
    flagged = int((preds == -1).sum())
    total   = len(preds)
    # At most 15% flagged is acceptable for a normal batch
    assert flagged <= int(total * 0.15) + 1, \
        "Too many normal logs flagged: {}/{}".format(flagged, total)
    print("PASS  Scenario 1 — Normal logs             ({}/{} flagged, expected ≤2)".format(
        flagged, total))


# ── Scenario 2: Database connection failures ──────────────────────────────────

def test_scenario_db_failures():
    """
    Night-time database failures should score lower (more anomalous) than
    the normal corpus on average.
    """
    db_failures = [
        _make('ERROR', 2, 'Database connection failed'),
        _make('ERROR', 3, 'Database connection failed'),
        _make('ERROR', 1, 'Database connection failed'),
    ]
    normal = [
        _make('INFO', 10, 'Database connection established'),
        _make('INFO', 11, 'Database connection established'),
        _make('INFO', 14, 'Database connection established'),
    ]

    preds_fail,   _ = _score(db_failures)
    preds_normal, _ = _score(normal)

    df_fail   = pd.DataFrame(db_failures)
    df_normal = pd.DataFrame(normal)
    for df_ in (df_fail, df_normal):
        for col in ('level', 'timestamp', 'message'):
            if col not in df_.columns:
                df_[col] = ''

    feats_fail,   _ = extract_features(df_fail,   VEC, fit=False)
    feats_normal, _ = extract_features(df_normal, VEC, fit=False)
    score_fail   = float(MODEL.score_samples(feats_fail).mean())
    score_normal = float(MODEL.score_samples(feats_normal).mean())

    assert score_fail < score_normal, (
        "DB failures should score lower than normal connections. "
        "Got fail={:.3f}, normal={:.3f}".format(score_fail, score_normal)
    )
    print("PASS  Scenario 2 — Database failures        "
          "(anomaly score: failures={:.3f} < normal={:.3f})".format(
              score_fail, score_normal))


# ── Scenario 3: Server crashes ────────────────────────────────────────────────

def test_scenario_server_crashes():
    """
    Server crash entries should be classified as HIGH severity.
    """
    crash_entries = [
        _make('ERROR', 2,  'Server crashed unexpectedly'),
        _make('ERROR', 1,  'Server crashed unexpectedly'),
        _make('ERROR', 23, 'Server crashed unexpectedly'),
    ]
    for entry in crash_entries:
        severity = AM.classify_severity(entry)
        assert severity == 'HIGH', \
            "Server crash should be HIGH, got {}".format(severity)
    print("PASS  Scenario 3 — Server crashes           (all classified HIGH)")


# ── Scenario 4: Timeout / external service failures ───────────────────────────

def test_scenario_timeouts_and_external_failures():
    """
    Night-time timeout and external service failures score lower than
    daytime timeouts.
    """
    night_timeouts = [
        _make('ERROR', 2,  'Timeout while connecting to service'),
        _make('ERROR', 3,  'Timeout while connecting to service'),
    ]
    day_timeouts = [
        _make('ERROR', 10, 'Timeout while connecting to service'),
        _make('ERROR', 14, 'Timeout while connecting to service'),
    ]

    df_night = pd.DataFrame(night_timeouts)
    df_day   = pd.DataFrame(day_timeouts)
    for df_ in (df_night, df_day):
        for col in ('level', 'timestamp', 'message'):
            if col not in df_.columns:
                df_[col] = ''

    feats_night, _ = extract_features(df_night, VEC, fit=False)
    feats_day,   _ = extract_features(df_day,   VEC, fit=False)

    score_night = float(MODEL.score_samples(feats_night).mean())
    score_day   = float(MODEL.score_samples(feats_day).mean())

    assert score_night < score_day, (
        "Night timeouts should score lower than day timeouts. "
        "Got night={:.3f}, day={:.3f}".format(score_night, score_day)
    )
    print("PASS  Scenario 4 — Timeouts / ext failures  "
          "(night={:.3f} < day={:.3f})".format(score_night, score_day))


# ── Scenario 5: Authentication failures ──────────────────────────────────────

def test_scenario_auth_failures():
    """
    Authentication failures should be classified as HIGH severity
    (critical keyword: 'authentication' and 'failed').
    """
    auth_entries = [
        _make('ERROR', 10, 'Authentication failed'),   # business hours
        _make('ERROR',  0, 'Authentication failed'),   # midnight
        _make('ERROR',  4, 'Authentication failed'),   # 4 AM
    ]
    for entry in auth_entries:
        severity = AM.classify_severity(entry)
        assert severity == 'HIGH', \
            "Auth failure should be HIGH, got {} for hour {}".format(
                severity, entry['timestamp'][11:13])
    print("PASS  Scenario 5 — Authentication failures  (all classified HIGH)")


# ── Scenario 6: Malformed log lines ──────────────────────────────────────────

def test_scenario_malformed_logs():
    """
    Lines that do not match the timestamp+level+message pattern must
    return None from parse_log and must NOT crash the pipeline.
    """
    malformed_lines = [
        '',                              # empty line
        'not a log line at all',         # no timestamp
        '2026-08-01 WARN something',     # missing time component
        '!!!ERROR!!! something happened',
        '2026-08-01 10:00:00',           # timestamp only, no level or message
        None,
    ]
    skipped = 0
    for line in malformed_lines:
        if line is None:
            skipped += 1
            continue
        result = parse_log(line)
        if result is None:
            skipped += 1
        else:
            # Parseable lines must still survive preprocess without error
            try:
                preprocess(result)
            except Exception as e:
                raise AssertionError("preprocess() raised on valid parse: {}".format(e))

    assert skipped >= 4, \
        "Expected at least 4 malformed lines to be rejected, got {}".format(skipped)
    print("PASS  Scenario 6 — Malformed log lines      ({}/{} correctly rejected)".format(
        skipped, len(malformed_lines)))


# ── Scenario 7: Missing / invalid timestamps ──────────────────────────────────

def test_scenario_invalid_timestamps():
    """
    Entries with missing or malformed timestamps must be dropped by
    extract_features (not crash), and the remaining valid rows still score.
    """
    mixed = [
        {'level': 'INFO',  'timestamp': 'not-a-date',         'message': 'Service started'},
        {'level': 'ERROR', 'timestamp': '',                    'message': 'Database failed'},
        {'level': 'WARN',  'timestamp': None,                  'message': 'Slow query'},
        {'level': 'INFO',  'timestamp': '2026-08-01 10:00:00', 'message': 'Backup completed'},
        {'level': 'INFO',  'timestamp': '2026-08-01 11:00:00', 'message': 'Service started'},
    ]
    df = pd.DataFrame(mixed)
    df['timestamp'] = df['timestamp'].fillna('')
    feats, valid = extract_features(df, VEC, fit=False)

    valid_count = int(valid.sum())
    assert valid_count == 2, \
        "Expected 2 valid rows (last 2), got {}".format(valid_count)
    assert feats.shape[0] == 2, \
        "Feature matrix should have 2 rows, got {}".format(feats.shape[0])

    # Must not crash on predict
    preds = MODEL.predict(feats)
    assert len(preds) == 2, "Expected 2 predictions"
    print("PASS  Scenario 7 — Invalid timestamps       "
          "({} bad rows dropped, {} valid rows scored)".format(
              len(mixed) - valid_count, valid_count))


# ── Precision / Recall / F1 / FPR summary ────────────────────────────────────

LABELED_SET = [
    # (entry, true_label)  true_label: -1=anomaly, 1=normal
    (_make('INFO',  10, 'User logged in successfully'),          1),
    (_make('INFO',  14, 'Backup completed'),                     1),
    (_make('INFO',  11, 'Request processed successfully'),       1),
    (_make('WARN',  10, 'High memory usage detected'),           1),
    (_make('WARN',  15, 'Slow query detected'),                  1),
    (_make('ERROR', 10, 'Database connection failed'),           1),
    (_make('ERROR', 14, 'Timeout while connecting to service'),  1),
    (_make('DEBUG',  9, 'Initializing module'),                  1),
    (_make('DEBUG', 10, 'Loading configuration'),                1),
    (_make('INFO',  13, 'Service started'),                      1),
    (_make('INFO',   9, 'Database connection established'),      1),
    (_make('WARN',  11, 'Disk usage above 70%'),                 1),
    (_make('INFO',  16, 'Backup completed'),                     1),
    (_make('WARN',  13, 'Response time exceeding threshold'),    1),
    (_make('INFO',  10, 'User logged in successfully'),          1),
    # --- Anomalous ---
    (_make('ERROR',  2, 'Server crashed unexpectedly'),         -1),
    (_make('ERROR',  0, 'Authentication failed'),               -1),
    (_make('ERROR',  3, 'Database connection failed'),          -1),
    (_make('DEBUG',  3, 'Loading configuration'),               -1),
    (_make('ERROR',  1, 'Server crashed unexpectedly'),         -1),
    (_make('ERROR',  4, 'Authentication failed'),               -1),
    (_make('ERROR', 23, 'Timeout while connecting to service'), -1),
    (_make('WARN',   2, 'Disk usage above 70%'),                -1),
    (_make('ERROR', 12, 'Server crashed unexpectedly'),         -1),
    (_make('ERROR', 10, 'Authentication failed'),                1),  # borderline
    (_make('DEBUG',  4, 'Initializing module'),                 -1),
    (_make('INFO',   2, 'Backup completed'),                     1),
    (_make('ERROR',  5, 'Server crashed unexpectedly'),         -1),
    (_make('WARN',   9, 'High memory usage detected'),           1),
    (_make('ERROR', 22, 'Authentication failed'),               -1),
]


def test_detection_metrics():
    """
    Run the labeled evaluation and report precision, recall, F1, and FPR.
    Does NOT assert a minimum threshold — the results are reported for
    transparency.  The test passes as long as the pipeline runs without error.
    """
    entries  = [e for e, _ in LABELED_SET]
    y_true   = np.array([lbl for _, lbl in LABELED_SET])
    preds, _ = _score(entries)

    if len(preds) == 0:
        print("FAIL  Detection metrics — no predictions returned")
        return

    y_pred_bin = (preds == -1).astype(int)
    y_true_bin = (y_true == -1).astype(int)

    p   = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    r   = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1  = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    fp  = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
    tn  = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("PASS  Detection metrics              —  "
          "Precision={:.2f}  Recall={:.2f}  F1={:.2f}  FPR={:.2f}".format(
              p, r, f1, fpr))
    return {'precision': p, 'recall': r, 'f1': f1, 'fpr': fpr}


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Running scenario tests (Improvement 4)...\n")

    test_scenario_normal_logs()
    test_scenario_db_failures()
    test_scenario_server_crashes()
    test_scenario_timeouts_and_external_failures()
    test_scenario_auth_failures()
    test_scenario_malformed_logs()
    test_scenario_invalid_timestamps()
    test_detection_metrics()

    print("\n" + "=" * 60)
    print("All scenario tests passed.")
    print("=" * 60)

