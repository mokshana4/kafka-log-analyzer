"""
config.py
Centralised configuration — every service URL, credential, and
tunable constant is read from environment variables with sensible
local-dev defaults.

Setting values (Windows PowerShell, before running the consumer):
    $env:KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
    $env:ELASTICSEARCH_URL       = "http://localhost:9200"
    $env:SLACK_WEBHOOK_URL       = "https://hooks.slack.com/services/..."

All env vars are optional.  The defaults make the project work
out-of-the-box on a local machine with the standard Docker setup.
"""

import os

# ── Kafka ─────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC             = os.environ.get('KAFKA_TOPIC',             'app-logs')
KAFKA_GROUP_ID          = os.environ.get('KAFKA_GROUP_ID',          'log-analyzer-group')

# ── Elasticsearch ─────────────────────────────────────────────────────────────
ELASTICSEARCH_URL   = os.environ.get('ELASTICSEARCH_URL',   'http://localhost:9200')
ELASTICSEARCH_INDEX = os.environ.get('ELASTICSEARCH_INDEX', 'logs')

# ── Slack ─────────────────────────────────────────────────────────────────────
# Never put the real URL here — set the environment variable instead.
SLACK_WEBHOOK_URL = os.environ.get('SLACK_WEBHOOK_URL', '')

# ── ML model ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get('MODEL_PATH', './ml/model.pkl')

# ── Retry / reliability ───────────────────────────────────────────────────────
MAX_RETRY_ATTEMPTS  = int(os.environ.get('MAX_RETRY_ATTEMPTS',  '3'))
RETRY_DELAY_SECONDS = float(os.environ.get('RETRY_DELAY_SECONDS', '5.0'))

# ── Alerting ──────────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = int(os.environ.get('ALERT_COOLDOWN_SECONDS', '60'))

