"""
ml/train_model.py
Trains an IsolationForest on logs fetched from Elasticsearch.

Changes from original:
- Feature extraction is imported from ml/features.py (shared with inference).
- The vectorizer is saved together with the model so inference always uses
  the same transformation — previously they were duplicated and could drift.
- Message text is now included via TF-IDF on a fixed 15-word vocabulary.
- Empty result sets and missing fields are handled safely.
- Deprecated `body=` argument removed.
"""

import sys
sys.path.append('.')

import os
import joblib
import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.ensemble import IsolationForest

from ml.features import build_vectorizer, extract_features
import config

es = Elasticsearch(config.ELASTICSEARCH_URL)


def fetch_logs():
    try:
        response = es.search(
            index=config.ELASTICSEARCH_INDEX,
            query={"match_all": {}},
            size=1000,
        )
        hits = [hit['_source'] for hit in response['hits']['hits']]
        if hits:
            return hits
    except Exception as e:
        print(f"[INFO] Could not fetch logs from Elasticsearch: {e}")

    # Fallback: load directly from local sample logs
    sample_path = './logs/sample_logs.txt'
    if os.path.exists(sample_path):
        print(f"[INFO] Falling back to local dataset: '{sample_path}'")
        from log_parser.parse_logs import parse_log_file
        from log_parser.preprocess import preprocess_all
        entries = parse_log_file(sample_path)
        return preprocess_all(entries)

    return []


if __name__ == "__main__":
    print("Fetching logs from Elasticsearch...")
    logs = fetch_logs()
    print(f"Fetched {len(logs)} logs.")

    if not logs:
        print("No logs found. Exiting without training.")
        sys.exit(1)

    df = pd.DataFrame(logs)

    # Ensure required columns exist
    for col in ('level', 'timestamp', 'message'):
        if col not in df.columns:
            df[col] = ''

    vectorizer = build_vectorizer()
    features, valid = extract_features(df, vectorizer, fit=True)

    print(f"Valid rows for training: {valid.sum()} / {len(df)}")
    print(f"Feature matrix shape: {features.shape}  "
          f"(level_code, hour, + {features.shape[1] - 2} tfidf columns)")

    print("Training Isolation Forest model...")
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(features)

    # Save model and vectorizer together — inference must use this bundle
    bundle = {'model': model, 'vectorizer': vectorizer}
    joblib.dump(bundle, './ml/model.pkl')
    print("Model and vectorizer saved to ml/model.pkl")
