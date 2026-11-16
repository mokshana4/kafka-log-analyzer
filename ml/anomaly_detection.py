"""
ml/anomaly_detection.py
Loads the saved model bundle and detects anomalies in Elasticsearch logs.

Changes from original:
- Loads the bundle (model + vectorizer) saved by train_model.py.
- Feature extraction imported from ml/features.py — same code as training.
- Deprecated `body=` argument removed.
- Empty input and missing fields handled safely.
"""

import sys
sys.path.append('.')

import joblib
import pandas as pd
from elasticsearch import Elasticsearch

from ml.features import extract_features
from alerting.slack_alert import send_slack_alert
from alerting.email_alert import send_email_alert
import config

es = Elasticsearch(config.ELASTICSEARCH_URL)


def load_bundle(path='./ml/model.pkl'):
    bundle = joblib.load(path)
    # Support old-format model.pkl (model object only, not a dict)
    if isinstance(bundle, dict):
        return bundle['model'], bundle['vectorizer']
    # Old format — return model only; inference will lack text features
    print("Warning: old model format detected. Re-run train_model.py.")
    return bundle, None


def detect_anomalies():
    print("Fetching logs from Elasticsearch...")
    response = es.search(
        index="logs",
        query={"match_all": {}},
        size=1000,
    )
    logs = [hit['_source'] for hit in response['hits']['hits']]
    print(f"Fetched {len(logs)} logs.")

    if not logs:
        print("No logs to analyze.")
        return

    df = pd.DataFrame(logs)
    for col in ('level', 'timestamp', 'message'):
        if col not in df.columns:
            df[col] = ''

    model, vectorizer = load_bundle()

    if vectorizer is None:
        # Fallback: old two-feature path so existing model.pkl still works
        from ml.features import LEVEL_MAP
        df['level_code'] = df['level'].map(LEVEL_MAP)
        df['hour'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.hour
        df = df.dropna(subset=['level_code', 'hour'])
        features = df[['level_code', 'hour']].values
        valid_df = df
    else:
        features, valid = extract_features(df, vectorizer, fit=False)
        valid_df = df[valid.values].copy()

    if len(features) == 0:
        print("No valid rows after feature extraction.")
        return

    predictions = model.predict(features)
    valid_df = valid_df.copy()
    valid_df['anomaly'] = predictions
    anomalies = valid_df[valid_df['anomaly'] == -1]

    print(f"\nDetected {len(anomalies)} anomalies out of {len(valid_df)} logs.")
    print("\nAnomalous log entries:")
    for _, row in anomalies.iterrows():
        print(f"  [{row['level']}] {row['timestamp']} - {row['message']}")

    if len(anomalies) > 0:
        msg = f"Detected {len(anomalies)} anomalies in logs! Check immediately."
        send_slack_alert(msg)
        send_email_alert("Log Anomaly Detected", msg, "admin@example.com")


if __name__ == "__main__":
    detect_anomalies()
