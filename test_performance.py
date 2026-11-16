"""
test_performance.py
Throughput and latency benchmarks for the full on-machine pipeline.

Measures the speed of:
  parse_log → preprocess → extract_features → model.predict

Does NOT require Kafka, Elasticsearch, or Slack.

Run with:
    python test_performance.py

Output example:
  Throughput  :  4,312 logs/sec
  Latency p50 :   0.23 ms/log
  Latency p95 :   0.41 ms/log
  Latency p99 :   0.58 ms/log
"""

import sys
sys.path.append('.')

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from log_parser.parse_logs import parse_log
from log_parser.preprocess import preprocess
from ml.features import build_vectorizer, extract_features

# ── Build model ───────────────────────────────────────────────────────────────

def _build_model():
    rows = []
    messages = [
        ('INFO',  'User logged in successfully'),
        ('INFO',  'Request processed successfully'),
        ('INFO',  'Backup completed'),
        ('INFO',  'Service started'),
        ('WARN',  'Slow query detected'),
        ('WARN',  'High memory usage detected'),
        ('WARN',  'Disk usage above 70%'),
        ('ERROR', 'Database connection failed'),
        ('ERROR', 'Timeout while connecting to service'),
        ('DEBUG', 'Initializing module'),
    ]
    for hour in range(8, 19):
        for level, msg in messages:
            rows.append({'level': level,
                         'timestamp': '2026-08-01 {:02d}:00:00'.format(hour),
                         'message': msg})
    df  = pd.DataFrame(rows)
    vec = build_vectorizer()
    feats, _ = extract_features(df, vec, fit=True)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(feats)
    return model, vec


MODEL, VEC = _build_model()

# ── Synthetic log lines (as raw strings, like Kafka delivers them) ─────────────

RAW_LINES = [
    '2026-08-01 10:00:00 INFO User logged in successfully',
    '2026-08-01 10:10:00 WARN High memory usage detected',
    '2026-08-01 10:20:00 ERROR Database connection failed',
    '2026-08-01 10:30:00 DEBUG Initializing module',
    '2026-08-01 10:40:00 INFO Request processed successfully',
    '2026-08-01 02:00:00 ERROR Server crashed unexpectedly',
    '2026-08-01 02:10:00 ERROR Authentication failed',
    '2026-08-01 11:00:00 WARN Slow query detected',
    '2026-08-01 15:00:00 INFO Backup completed',
    '2026-08-01 03:00:00 ERROR Timeout while connecting to service',
]


def _run_pipeline_single(raw_line):
    """Full pipeline for one raw log string. Returns prediction (1 or -1)."""
    entry = parse_log(raw_line)
    if entry is None:
        return None
    entry = preprocess(entry)
    df    = pd.DataFrame([entry])
    for col in ('level', 'timestamp', 'message'):
        if col not in df.columns:
            df[col] = ''
    feats, valid = extract_features(df, VEC, fit=False)
    if feats.shape[0] == 0:
        return None
    return MODEL.predict(feats)[0]


# ── Throughput benchmark ──────────────────────────────────────────────────────

def benchmark_throughput(n_logs=1000):
    """
    Process n_logs entries through the full pipeline and report logs/sec.
    Uses batch feature extraction for realistic throughput measurement.
    """
    # Build a batch of raw lines by cycling through RAW_LINES
    batch_lines = [RAW_LINES[i % len(RAW_LINES)] for i in range(n_logs)]

    start = time.perf_counter()

    # Step 1: parse + preprocess
    entries = []
    for line in batch_lines:
        entry = parse_log(line)
        if entry:
            entries.append(preprocess(entry))

    # Step 2: batch feature extraction
    df = pd.DataFrame(entries)
    for col in ('level', 'timestamp', 'message'):
        if col not in df.columns:
            df[col] = ''
    feats, valid = extract_features(df, VEC, fit=False)

    # Step 3: batch predict
    if feats.shape[0] > 0:
        preds = MODEL.predict(feats)

    elapsed = time.perf_counter() - start
    logs_per_sec = n_logs / elapsed if elapsed > 0 else float('inf')

    print("  Logs processed  : {:,}".format(n_logs))
    print("  Elapsed time    : {:.3f} s".format(elapsed))
    print("  Throughput      : {:,.0f} logs/sec".format(logs_per_sec))
    return logs_per_sec


# ── Latency benchmark ────────────────────────────────────────────────────────

def benchmark_latency(n_samples=200):
    """
    Measure the per-log latency (parse → preprocess → features → predict)
    for n_samples individual calls and report percentile statistics.
    """
    latencies = []
    for i in range(n_samples):
        line = RAW_LINES[i % len(RAW_LINES)]
        t0   = time.perf_counter()
        _run_pipeline_single(line)
        t1   = time.perf_counter()
        latencies.append((t1 - t0) * 1000)   # milliseconds

    lat = np.array(latencies)
    p50 = float(np.percentile(lat, 50))
    p95 = float(np.percentile(lat, 95))
    p99 = float(np.percentile(lat, 99))
    mean = float(lat.mean())

    print("  Samples         : {:,}".format(n_samples))
    print("  Mean latency    : {:.3f} ms/log".format(mean))
    print("  Latency p50     : {:.3f} ms/log".format(p50))
    print("  Latency p95     : {:.3f} ms/log".format(p95))
    print("  Latency p99     : {:.3f} ms/log".format(p99))
    return {'mean': mean, 'p50': p50, 'p95': p95, 'p99': p99}


# ── Assertions ───────────────────────────────────────────────────────────────

def test_throughput_acceptable(n_logs=1000, min_logs_per_sec=200):
    """
    Throughput should comfortably exceed min_logs_per_sec.
    The bar is set low (200 logs/sec) because this is a local dev machine
    with a small model — real throughput is typically 1,000-10,000+.
    """
    print("\nThroughput benchmark ({:,} logs):".format(n_logs))
    logs_per_sec = benchmark_throughput(n_logs)
    assert logs_per_sec >= min_logs_per_sec, \
        "Throughput {:.0f} logs/sec is below minimum {}".format(
            logs_per_sec, min_logs_per_sec)
    print("PASS  Throughput >= {:,} logs/sec".format(min_logs_per_sec))


def test_latency_acceptable(n_samples=200, max_p95_ms=50.0):
    """
    p95 latency per log should be below max_p95_ms milliseconds.
    50 ms is a generous upper bound for local processing — real p95
    is typically well under 5 ms.
    """
    print("\nLatency benchmark ({} samples):".format(n_samples))
    stats = benchmark_latency(n_samples)
    assert stats['p95'] < max_p95_ms, \
        "p95 latency {:.2f} ms exceeds maximum {:.0f} ms".format(
            stats['p95'], max_p95_ms)
    print("PASS  p95 latency < {:.0f} ms".format(max_p95_ms))


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Performance benchmarks (Improvement 4)\n")
    print("=" * 60)
    test_throughput_acceptable()
    test_latency_acceptable()
    print("\n" + "=" * 60)
    print("All performance benchmarks passed.")
    print("=" * 60)

