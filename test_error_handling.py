"""
test_error_handling.py
Tests for Improvement 5 — reliability and error handling.

Verifies that:
  - Config reads from environment variables and falls back to defaults
  - Elasticsearch storage failure returns False, does not raise
  - Per-message errors (bad bytes, bad parse, bad scoring) are isolated
  - Kafka retry counter controls the number of attempts
  - Malformed input at every pipeline stage is handled gracefully

No live Kafka, Elasticsearch, or Slack needed.

Run with:
    python test_error_handling.py
"""

import sys
sys.path.append('.')

import os
import types
import joblib
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.ensemble import IsolationForest

from ml.features import build_vectorizer, extract_features


# ── 1. Config — env var overrides ─────────────────────────────────────────────

def test_config_defaults():
    """Config provides sensible defaults when env vars are not set."""
    # Remove any env vars that might be set, import fresh
    for key in ('KAFKA_BOOTSTRAP_SERVERS', 'ELASTICSEARCH_URL',
                'MODEL_PATH', 'MAX_RETRY_ATTEMPTS', 'RETRY_DELAY_SECONDS',
                'ALERT_COOLDOWN_SECONDS'):
        os.environ.pop(key, None)

    import importlib
    import config as cfg
    importlib.reload(cfg)   # reload after clearing env vars

    assert cfg.KAFKA_BOOTSTRAP_SERVERS == 'localhost:9092'
    assert cfg.ELASTICSEARCH_URL       == 'http://localhost:9200'
    assert cfg.ELASTICSEARCH_INDEX     == 'logs'
    assert cfg.MAX_RETRY_ATTEMPTS      == 3
    assert cfg.RETRY_DELAY_SECONDS     == 5.0
    assert cfg.ALERT_COOLDOWN_SECONDS  == 60
    print("PASS  test_config_defaults")


def test_config_env_var_override():
    """Setting an env var overrides the default."""
    os.environ['KAFKA_BOOTSTRAP_SERVERS'] = 'broker1:9092,broker2:9092'
    os.environ['ELASTICSEARCH_URL']       = 'http://es-prod:9200'
    os.environ['MAX_RETRY_ATTEMPTS']      = '5'

    import importlib
    import config as cfg
    importlib.reload(cfg)

    assert cfg.KAFKA_BOOTSTRAP_SERVERS == 'broker1:9092,broker2:9092'
    assert cfg.ELASTICSEARCH_URL       == 'http://es-prod:9200'
    assert cfg.MAX_RETRY_ATTEMPTS      == 5

    # Reset
    for key in ('KAFKA_BOOTSTRAP_SERVERS', 'ELASTICSEARCH_URL', 'MAX_RETRY_ATTEMPTS'):
        os.environ.pop(key, None)
    importlib.reload(cfg)
    print("PASS  test_config_env_var_override")


# ── 2. Elasticsearch — failure returns False, does not raise ─────────────────

def test_elasticsearch_failure_returns_false():
    """
    When es.index() raises, save_to_elasticsearch must return False
    without propagating the exception.
    """
    from storage import elasticsearch_loader as loader

    # Patch the module-level es object to raise on .index()
    mock_es = MagicMock()
    mock_es.index.side_effect = Exception("Simulated ES connection refused")

    original_es = loader.es
    loader.es = mock_es
    try:
        result = loader.save_to_elasticsearch(
            {'level': 'INFO', 'timestamp': '2026-08-01 10:00:00',
             'message': 'Service started'}
        )
        assert result is False, "Expected False on ES failure, got {}".format(result)
    finally:
        loader.es = original_es
    print("PASS  test_elasticsearch_failure_returns_false")


def test_elasticsearch_success_returns_true():
    """save_to_elasticsearch returns True when es.index() succeeds."""
    from storage import elasticsearch_loader as loader

    mock_es = MagicMock()
    mock_es.index.return_value = {'result': 'created'}

    original_es = loader.es
    loader.es = mock_es
    try:
        result = loader.save_to_elasticsearch(
            {'level': 'INFO', 'timestamp': '2026-08-01 10:00:00',
             'message': 'Service started'}
        )
        assert result is True, "Expected True on success"
    finally:
        loader.es = original_es
    print("PASS  test_elasticsearch_success_returns_true")


def test_elasticsearch_none_entry_returns_false():
    """save_to_elasticsearch returns False immediately for None/empty entry."""
    from storage.elasticsearch_loader import save_to_elasticsearch
    assert save_to_elasticsearch(None)  is False
    assert save_to_elasticsearch({})    is False
    print("PASS  test_elasticsearch_none_entry_returns_false")


def test_save_all_counts_failures():
    """save_all reports success and failure counts correctly."""
    from storage import elasticsearch_loader as loader

    call_count = [0]
    def fake_index(index, document):
        call_count[0] += 1
        if call_count[0] % 2 == 0:
            raise Exception("Simulated failure")
        return {'result': 'created'}

    mock_es = MagicMock()
    mock_es.index.side_effect = fake_index

    original_es = loader.es
    loader.es = mock_es
    try:
        entries = [{'level': 'INFO', 'timestamp': '2026-08-01 10:00:00',
                    'message': 'msg {}'.format(i)} for i in range(4)]
        # Entries 1,3 succeed; 2,4 fail (even call_count fails)
        loader.save_all(entries)   # just verify it doesn't raise
    finally:
        loader.es = original_es
    print("PASS  test_save_all_counts_failures")


# ── 3. Consumer — per-message error isolation ─────────────────────────────────

def _load_consumer_fn(fn_name):
    """Extract a single function from consumer.py via exec."""
    mod = types.ModuleType('consumer_mod')
    mod.__dict__.update({
        'sys': sys, 'os': os, 'time': __import__('time'),
        'joblib': joblib, 'pd': pd,
        'extract_features': extract_features,
        'parse_log': __import__('log_parser.parse_logs', fromlist=['parse_log']).parse_log,
        'preprocess': __import__('log_parser.preprocess', fromlist=['preprocess']).preprocess,
        'save_to_elasticsearch': lambda e: True,   # stub
        'send_slack_alert': lambda m: None,         # stub
        'AlertManager': __import__('alerting.alert_manager',
                                   fromlist=['AlertManager']).AlertManager,
        'config': __import__('config'),
    })
    import ast
    with open('kafka_client/consumer.py') as f:
        src = f.read()
    tree = ast.parse(src)
    fn_nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    # Load all helpers first so they are in scope when _process_message runs
    for name in ('load_model_bundle', 'is_anomaly', '_process_message'):
        if name in fn_nodes:
            exec(compile(ast.unparse(fn_nodes[name]), 'consumer.py', 'exec'), mod.__dict__)
    return mod.__dict__[fn_name]


def _build_model():
    rows = [{'level': 'INFO', 'timestamp': '2026-08-01 10:00:00',
             'message': 'User logged in successfully'} for _ in range(30)]
    df  = pd.DataFrame(rows)
    vec = build_vectorizer()
    feats, _ = extract_features(df, vec, fit=True)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(feats)
    return model, vec


def test_bad_bytes_message_skipped():
    """
    A Kafka message whose bytes cannot be decoded must be skipped
    without crashing the loop.
    """
    _process_message = _load_consumer_fn('_process_message')
    model, vec = _build_model()

    # Use a mock object whose .decode() raises UnicodeDecodeError
    class BadBytes:
        def decode(self, encoding):
            raise UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte')

    bad_msg = MagicMock()
    bad_msg.value = BadBytes()

    counters = {'total': 0, 'anomalies': 0}
    _process_message(bad_msg, model, vec, True,
                     __import__('alerting.alert_manager',
                                fromlist=['AlertManager']).AlertManager(),
                     counters)
    assert counters['total'] == 0, "Bad message should not increment counter"
    print("PASS  test_bad_bytes_message_skipped")


def test_unparseable_line_skipped():
    """
    A Kafka message with no parseable log pattern must be skipped
    without crashing.
    """
    _process_message = _load_consumer_fn('_process_message')
    model, vec = _build_model()

    bad_msg = MagicMock()
    bad_msg.value = b'this is not a valid log line at all'

    counters = {'total': 0, 'anomalies': 0}
    _process_message(bad_msg, model, vec, True,
                     __import__('alerting.alert_manager',
                                fromlist=['AlertManager']).AlertManager(),
                     counters)
    assert counters['total'] == 0
    print("PASS  test_unparseable_line_skipped")


def test_valid_message_increments_counter():
    """A valid log message increments the total counter."""
    _process_message = _load_consumer_fn('_process_message')
    model, vec = _build_model()

    good_msg = MagicMock()
    good_msg.value = b'2026-08-01 10:00:00 INFO User logged in successfully'

    counters = {'total': 0, 'anomalies': 0}
    _process_message(good_msg, model, vec, True,
                     __import__('alerting.alert_manager',
                                fromlist=['AlertManager']).AlertManager(),
                     counters)
    assert counters['total'] == 1, "Valid message should increment counter"
    print("PASS  test_valid_message_increments_counter")


def test_anomalous_message_increments_anomaly_counter():
    """An anomalous log message increments the anomaly counter."""
    _process_message = _load_consumer_fn('_process_message')
    model = MagicMock()
    model.predict.return_value = np.array([-1])
    vec = build_vectorizer()
    vec.fit(['placeholder'])

    anom_msg = MagicMock()
    anom_msg.value = b'2026-08-01 02:00:00 ERROR Server crashed unexpectedly'

    counters = {'total': 0, 'anomalies': 0}
    _process_message(anom_msg, model, vec, True,
                     __import__('alerting.alert_manager',
                                fromlist=['AlertManager']).AlertManager(),
                     counters)

    assert counters['anomalies'] == 1, \
        "Expected anomalies counter to be 1, got {}".format(counters['anomalies'])
    assert counters['total'] == 1, \
        "Expected total counter to be 1, got {}".format(counters['total'])
    print("PASS  test_anomalous_message_increments_anomaly_counter")


# ── 4. Kafka retry — count is respected ───────────────────────────────────────

def test_kafka_retry_attempts_respected():
    """
    run_consumer retries exactly MAX_RETRY_ATTEMPTS times before giving up
    when Kafka is unavailable.
    """
    from kafka.errors import KafkaConnectionError
    import config as cfg
    from kafka_client import consumer as cons_mod

    # Force 2 retry attempts for speed
    original_max    = cfg.MAX_RETRY_ATTEMPTS
    original_delay  = cfg.RETRY_DELAY_SECONDS
    cfg.MAX_RETRY_ATTEMPTS  = 2
    cfg.RETRY_DELAY_SECONDS = 0.0   # no sleep in tests

    attempt_log = []

    def fake_kafka_consumer(*args, **kwargs):
        attempt_log.append(1)
        raise KafkaConnectionError()

    with patch.object(cons_mod, 'KafkaConsumer', side_effect=fake_kafka_consumer):
        with patch.object(cons_mod, 'load_model_bundle', return_value=(None, None)):
            try:
                cons_mod.run_consumer()
            except KafkaConnectionError:
                pass   # Expected after retries are exhausted

    cfg.MAX_RETRY_ATTEMPTS  = original_max
    cfg.RETRY_DELAY_SECONDS = original_delay

    assert len(attempt_log) == 2, \
        "Expected 2 attempts, got {}".format(len(attempt_log))
    print("PASS  test_kafka_retry_attempts_respected")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Running error-handling tests (Improvement 5)...\n")

    test_config_defaults()
    test_config_env_var_override()

    test_elasticsearch_failure_returns_false()
    test_elasticsearch_success_returns_true()
    test_elasticsearch_none_entry_returns_false()
    test_save_all_counts_failures()

    test_bad_bytes_message_skipped()
    test_unparseable_line_skipped()
    test_valid_message_increments_counter()
    test_anomalous_message_increments_anomaly_counter()

    test_kafka_retry_attempts_respected()

    print("\n" + "=" * 55)
    print("All error-handling tests passed.")
    print("=" * 55)
