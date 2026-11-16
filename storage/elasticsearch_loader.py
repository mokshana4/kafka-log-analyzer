"""
storage/elasticsearch_loader.py
Saves parsed log entries to Elasticsearch.

Improvement 5 changes:
  - Service URL read from config (no hardcoded 'localhost:9200').
  - save_to_elasticsearch wraps es.index() in try/except so a failed
    write logs the error and returns False instead of crashing.
  - save_all reports how many entries succeeded and how many failed.
"""

import sys
sys.path.append('.')

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import TransportError
import config

es = Elasticsearch(config.ELASTICSEARCH_URL)


def save_to_elasticsearch(log_entry):
    """
    Index one parsed log entry into Elasticsearch.

    Returns True on success, False if Elasticsearch is unreachable or
    returns an error — so the caller can decide how to handle the failure
    without catching an exception itself.
    """
    if not log_entry:
        return False
    try:
        es.index(index=config.ELASTICSEARCH_INDEX, document=log_entry)
        return True
    except TransportError as e:
        print('[ERROR] Elasticsearch storage failed: {}'.format(e))
        return False
    except Exception as e:
        print('[ERROR] Unexpected error saving to Elasticsearch: {}'.format(e))
        return False


def save_all(entries):
    """Index a list of entries, reporting success and failure counts."""
    success = 0
    failed  = 0
    for entry in entries:
        if save_to_elasticsearch(entry):
            success += 1
        else:
            failed += 1
    print('Saved {}/{} entries to Elasticsearch.'.format(
        success, success + failed))
    if failed:
        print('[WARNING] {} entr{} failed to save.'.format(
            failed, 'y' if failed == 1 else 'ies'))
