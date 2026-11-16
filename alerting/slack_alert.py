"""
alerting/slack_alert.py
Sends formatted alert messages to a Slack incoming webhook.

The webhook URL is read from the environment variable SLACK_WEBHOOK_URL
so it is never stored in source code.

To set it before running (Windows PowerShell):
    $env:SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."

To set it permanently, add it to your system environment variables or
to a .env file (and load it with python-dotenv if needed).
"""

import sys
sys.path.append('.')

import requests
import config

# URL comes from config which reads SLACK_WEBHOOK_URL env var.
# Never hardcode it here.
_WEBHOOK_URL = config.SLACK_WEBHOOK_URL


def send_slack_alert(message):
    """
    Post a plain-text message to the configured Slack webhook.

    Does nothing (and prints a warning once) if SLACK_WEBHOOK_URL is not set.
    Catches all network errors so a Slack outage never crashes the consumer.
    """
    if not _WEBHOOK_URL:
        print('[WARNING] SLACK_WEBHOOK_URL is not set — Slack alert skipped.')
        return

    payload = {'text': message}
    try:
        response = requests.post(_WEBHOOK_URL, json=payload, timeout=5)
        if response.status_code == 200:
            # Print only the first line of the message to keep the console clean
            first_line = message.splitlines()[0] if message else ''
            print('[SLACK] Alert sent: {}'.format(first_line))
        else:
            print('[SLACK] Unexpected status {}: {}'.format(
                response.status_code, response.text))
    except requests.exceptions.RequestException as e:
        print('[SLACK] Alert failed: {}'.format(e))
