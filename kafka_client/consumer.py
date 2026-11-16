"""
kafka_client/consumer.py
Reads messages from the Kafka topic, parses and preprocesses each log
entry, runs real-time ML anomaly detection, sends intelligent Slack
alerts, and stores every entry in Elasticsearch.

Improvement 2: anomaly detection integrated into the consumer loop.
Improvement 3: AlertManager for severity / dedup / cooldown.
Improvement 5: Kafka retry on disconnection, ES error handling,
               config from environment variables, per-message error
               isolation so one bad message never crashes the loop.
"""

import sys
sys.path.append('.')

import os
import time
import joblib
import pandas as pd

from kafka import KafkaConsumer
from kafka.errors import KafkaError, KafkaConnectionError

from log_parser.parse_logs import parse_log
from log_parser.preprocess import preprocess
from storage.elasticsearch_loader import save_to_elasticsearch
from ml.features import extract_features
from alerting.slack_alert import send_slack_alert
from alerting.alert_manager import AlertManager
import config


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_bundle(path):
    """
    Load the model + vectorizer bundle saved by train_model.py.

    Returns (model, vectorizer) on success.
    Returns (None, None) if the file is missing or in an old format so
    the consumer can still run in store-only mode without crashing.
    """
    if not os.path.exists(path):
        print("[WARNING] Model file not found at '{}'.".format(path))
        print("[WARNING] Running in store-only mode — anomaly detection disabled.")
        print("[WARNING] Run 'python ml/train_model.py' first to enable detection.")
        return None, None

    try:
        bundle = joblib.load(path)
    except Exception as e:
        print("[WARNING] Could not load model: {}".format(e))
        print("[WARNING] Running in store-only mode — anomaly detection disabled.")
        return None, None

    if isinstance(bundle, dict) and 'model' in bundle and 'vectorizer' in bundle:
        print("[INFO] Model and vectorizer loaded from '{}'.".format(path))
        return bundle['model'], bundle['vectorizer']

    print("[WARNING] Model file is in old format — no vectorizer found.")
    print("[WARNING] Re-run 'python ml/train_model.py' to save the new bundle.")
    print("[WARNING] Running in store-only mode — anomaly detection disabled.")
    return None, None


# ── Per-message scoring ───────────────────────────────────────────────────────

def is_anomaly(entry, model, vectorizer):
    """
    Score a single parsed log entry dict against the loaded model.

    Returns True  if the model flags the entry as anomalous.
    Returns False if it is normal or if feature extraction fails.
    """
    df = pd.DataFrame([entry])
    for col in ('level', 'timestamp', 'message'):
        if col not in df.columns:
            df[col] = ''

    features, valid = extract_features(df, vectorizer, fit=False)
    if features.shape[0] == 0:
        return False

    prediction = model.predict(features)[0]
    return bool(prediction == -1)


# ── Message processing loop ───────────────────────────────────────────────────

def _process_message(message, model, vectorizer, detection_active,
                     alert_manager, counters):
    """
    Handle one Kafka message end-to-end.
    All exceptions are caught here so a single bad message never kills
    the consumer loop.
    """
    try:
        log_line = message.value.decode('utf-8')
    except Exception as e:
        print("[SKIP] Could not decode message bytes: {}".format(e))
        return

    entry = parse_log(log_line)
    if not entry:
        print("[SKIP] Unparseable line: {}".format(log_line.strip()))
        return

    try:
        entry = preprocess(entry)
    except Exception as e:
        print("[SKIP] Preprocess error: {}".format(e))
        return

    # Store in Elasticsearch — save_to_elasticsearch returns True/False
    saved = save_to_elasticsearch(entry)
    if not saved:
        print("[WARN] Entry NOT saved to Elasticsearch: {}".format(entry.get('message', '')))
    counters['total'] += 1

    if not detection_active:
        print("[STORED] [{}] {} — {}".format(
            counters['total'], entry.get('level', ''), entry.get('message', '')))
        return

    try:
        flagged = is_anomaly(entry, model, vectorizer)
    except Exception as e:
        print("[WARN] Anomaly scoring failed for entry: {}".format(e))
        flagged = False

    if flagged:
        counters['anomalies'] += 1
        print("[ANOMALY] [{}] {} — {}".format(
            entry.get('level', ''), entry.get('timestamp', ''),
            entry.get('message', '')))

        try:
            alert_text = alert_manager.process_anomaly(entry)
            if alert_text:
                send_slack_alert(alert_text)
            else:
                print("[SUPPRESSED] Duplicate alert within cooldown window.")
        except Exception as e:
            print("[WARN] Alerting failed: {}".format(e))
    else:
        print("[OK]     [{}] {} — {}".format(
            counters['total'], entry.get('level', ''), entry.get('message', '')))


# ── Kafka consumer with retry ─────────────────────────────────────────────────

def run_consumer():
    """
    Start the Kafka consumer and run the processing loop.
    Retries the connection up to MAX_RETRY_ATTEMPTS times if Kafka
    is unavailable, with RETRY_DELAY_SECONDS between attempts.
    """
    model, vectorizer  = load_model_bundle(config.MODEL_PATH)
    detection_active   = model is not None
    alert_manager      = AlertManager(cooldown_seconds=config.ALERT_COOLDOWN_SECONDS)
    counters           = {'total': 0, 'anomalies': 0}

    max_attempts = config.MAX_RETRY_ATTEMPTS
    delay        = config.RETRY_DELAY_SECONDS

    for attempt in range(1, max_attempts + 1):
        try:
            consumer = KafkaConsumer(
                config.KAFKA_TOPIC,
                bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
                auto_offset_reset='earliest',
                group_id=config.KAFKA_GROUP_ID,
            )

            mode = "real-time detection + storage" if detection_active else "store-only"
            print("[INFO] Consumer started in {} mode (attempt {}/{}).".format(
                mode, attempt, max_attempts))
            print("[INFO] Waiting for messages on topic '{}'...\n".format(
                config.KAFKA_TOPIC))

            # Main loop — each message is fully isolated
            try:
                for message in consumer:
                    _process_message(message, model, vectorizer,
                                     detection_active, alert_manager, counters)
            except KeyboardInterrupt:
                print("\n[INFO] Consumer stopped by user (Ctrl+C).")

            break   # Clean exit

        except KafkaConnectionError:
            print("[ERROR] Kafka broker not available at '{}'.".format(
                config.KAFKA_BOOTSTRAP_SERVERS))
            if attempt < max_attempts:
                print("[INFO]  Retrying in {:.0f}s... ({}/{})".format(
                    delay, attempt, max_attempts))
                time.sleep(delay)
            else:
                print("[ERROR] Max retry attempts ({}) reached. Exiting.".format(
                    max_attempts))
                raise

        except KafkaError as e:
            print("[ERROR] Kafka error: {}".format(e))
            if attempt < max_attempts:
                print("[INFO]  Retrying in {:.0f}s... ({}/{})".format(
                    delay, attempt, max_attempts))
                time.sleep(delay)
            else:
                print("[ERROR] Max retry attempts ({}) reached. Exiting.".format(
                    max_attempts))
                raise

    print("\n[INFO] Consumer finished. Total: {} logs, {} anomalies.".format(
        counters['total'], counters['anomalies']))

    if counters['anomalies'] > 0:
        incidents = alert_manager.active_incidents()
        print("[INFO] Incident summary ({} distinct incident type(s)):".format(
            len(incidents)))
        for key, inc in incidents.items():
            print("         {} — {} occurrence(s), first: {}".format(
                key, inc['count'], inc['first_seen']))


if __name__ == '__main__':
    run_consumer()
