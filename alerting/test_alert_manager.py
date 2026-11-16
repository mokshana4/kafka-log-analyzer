"""
alerting/test_alert_manager.py
Tests for AlertManager — severity classification, deduplication,
cooldown, incident grouping, and message formatting.

Does NOT require Kafka, Elasticsearch, or Slack.

Run with:
    python alerting/test_alert_manager.py
"""

import sys
sys.path.append('.')

import time
from alerting.alert_manager import AlertManager, COOLDOWN_SECONDS


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_entry(level='ERROR', hour=2, message='Server crashed unexpectedly'):
    ts = '2026-08-01 {:02d}:00:00'.format(hour)
    return {'level': level, 'timestamp': ts, 'message': message}


# ── Severity classification ───────────────────────────────────────────────────

def test_severity_high_off_hours_error():
    """ERROR at night-time → HIGH."""
    am = AlertManager()
    entry = make_entry('ERROR', hour=2, message='Database connection failed')
    assert am.classify_severity(entry) == 'HIGH', "Night ERROR should be HIGH"
    print("PASS  test_severity_high_off_hours_error")


def test_severity_high_critical_keyword():
    """ERROR with a critical keyword at business hours → HIGH."""
    am = AlertManager()
    entry = make_entry('ERROR', hour=10, message='Server crashed unexpectedly')
    assert am.classify_severity(entry) == 'HIGH', \
        "'crashed' keyword should escalate to HIGH even at business hours"
    print("PASS  test_severity_high_critical_keyword")


def test_severity_high_auth_failure():
    """Authentication failure (critical keyword) → HIGH."""
    am = AlertManager()
    entry = make_entry('ERROR', hour=10, message='Authentication failed')
    assert am.classify_severity(entry) == 'HIGH', \
        "'authentication' keyword should escalate to HIGH"
    print("PASS  test_severity_high_auth_failure")


def test_severity_medium_error_business_hours():
    """ERROR at business hours with no critical keyword → MEDIUM."""
    am = AlertManager()
    entry = make_entry('ERROR', hour=10, message='Timeout while connecting to service')
    assert am.classify_severity(entry) == 'MEDIUM', \
        "Daytime ERROR without critical keyword should be MEDIUM"
    print("PASS  test_severity_medium_error_business_hours")


def test_severity_medium_warn_off_hours():
    """WARN at off-hours → MEDIUM."""
    am = AlertManager()
    entry = make_entry('WARN', hour=3, message='Slow query detected')
    assert am.classify_severity(entry) == 'MEDIUM', \
        "Night-time WARN should be MEDIUM"
    print("PASS  test_severity_medium_warn_off_hours")


def test_severity_low_warn_business_hours():
    """WARN at business hours → LOW."""
    am = AlertManager()
    entry = make_entry('WARN', hour=10, message='Disk usage above 70%')
    assert am.classify_severity(entry) == 'LOW', \
        "Business-hours WARN should be LOW"
    print("PASS  test_severity_low_warn_business_hours")


# ── First occurrence ──────────────────────────────────────────────────────────

def test_first_occurrence_returns_alert():
    """process_anomaly returns a non-empty string on first occurrence."""
    am = AlertManager()
    entry = make_entry('ERROR', hour=2, message='Server crashed unexpectedly')
    result = am.process_anomaly(entry)
    assert result is not None, "First occurrence should return an alert"
    assert len(result) > 0,    "Alert text should not be empty"
    print("PASS  test_first_occurrence_returns_alert")


def test_alert_contains_severity_and_level():
    """Alert text includes severity label and log level."""
    am = AlertManager()
    entry = make_entry('ERROR', hour=2, message='Server crashed unexpectedly')
    result = am.process_anomaly(entry)
    assert 'HIGH' in result,    "Alert should mention HIGH severity"
    assert 'ERROR' in result,   "Alert should include the log level"
    assert 'crashed' in result.lower() or 'Server crashed' in result, \
        "Alert should include the message"
    print("PASS  test_alert_contains_severity_and_level")


def test_alert_mentions_first_occurrence():
    """Alert text says 'First occurrence' when it is the first time."""
    am = AlertManager()
    entry = make_entry()
    result = am.process_anomaly(entry)
    assert 'First occurrence' in result, \
        "First alert should say 'First occurrence'"
    print("PASS  test_alert_mentions_first_occurrence")


# ── Deduplication & cooldown ─────────────────────────────────────────────────

def test_duplicate_within_cooldown_suppressed():
    """Second identical entry within cooldown window → None (suppressed)."""
    am = AlertManager(cooldown_seconds=60)
    entry = make_entry()
    am.process_anomaly(entry)        # first — goes through
    result = am.process_anomaly(entry)  # second — same fingerprint, same window
    assert result is None, "Duplicate within cooldown should be suppressed"
    print("PASS  test_duplicate_within_cooldown_suppressed")


def test_different_entries_not_suppressed():
    """Two entries with different messages are treated as separate incidents."""
    am = AlertManager(cooldown_seconds=60)
    entry_a = make_entry('ERROR', hour=2, message='Server crashed unexpectedly')
    entry_b = make_entry('ERROR', hour=2, message='Database connection failed')
    am.process_anomaly(entry_a)
    result_b = am.process_anomaly(entry_b)
    assert result_b is not None, \
        "Different message should not be suppressed by entry_a's cooldown"
    print("PASS  test_different_entries_not_suppressed")


def test_cooldown_expiry_allows_re_alert():
    """After the cooldown window, the same entry fires again."""
    am = AlertManager(cooldown_seconds=1)   # 1-second cooldown for speed
    entry = make_entry()
    am.process_anomaly(entry)               # first alert
    am.process_anomaly(entry)               # suppressed
    time.sleep(1.1)                         # wait for cooldown to expire
    result = am.process_anomaly(entry)      # should fire again
    assert result is not None, \
        "Alert should fire again after cooldown expires"
    print("PASS  test_cooldown_expiry_allows_re_alert")


# ── Incident tracking ─────────────────────────────────────────────────────────

def test_incident_count_increments():
    """Incident count increments even when alerts are suppressed."""
    am = AlertManager(cooldown_seconds=60)
    entry = make_entry()
    am.process_anomaly(entry)   # count = 1
    am.process_anomaly(entry)   # count = 2 (suppressed)
    am.process_anomaly(entry)   # count = 3 (suppressed)
    incidents = am.active_incidents()
    key = 'ERROR | Server crashed unexpectedly'
    assert key in incidents, "Incident should be tracked"
    assert incidents[key]['count'] == 3, \
        "Count should be 3, got {}".format(incidents[key]['count'])
    print("PASS  test_incident_count_increments")


def test_repeat_occurrence_text_in_re_alert():
    """After cooldown, the re-alert message shows occurrence count."""
    am = AlertManager(cooldown_seconds=1)
    entry = make_entry()
    am.process_anomaly(entry)       # first alert
    am.process_anomaly(entry)       # suppressed (count=2)
    time.sleep(1.1)
    result = am.process_anomaly(entry)   # re-alert (count=3)
    assert result is not None, "Re-alert should fire after cooldown"
    assert 'Occurrence #3' in result, \
        "Re-alert should show occurrence count. Got:\n{}".format(result)
    print("PASS  test_repeat_occurrence_text_in_re_alert")


def test_active_incidents_returns_all():
    """active_incidents reflects every tracked fingerprint."""
    am = AlertManager(cooldown_seconds=60)
    am.process_anomaly(make_entry('ERROR', 2,  'Server crashed unexpectedly'))
    am.process_anomaly(make_entry('ERROR', 10, 'Database connection failed'))
    am.process_anomaly(make_entry('WARN',  3,  'Slow query detected'))
    incidents = am.active_incidents()
    assert len(incidents) == 3, \
        "Expected 3 incidents, got {}".format(len(incidents))
    print("PASS  test_active_incidents_returns_all")


# ── Emoji and formatting ──────────────────────────────────────────────────────

def test_emoji_matches_severity():
    """HIGH alert uses red circle emoji."""
    am = AlertManager()
    entry = make_entry('ERROR', hour=2, message='Server crashed unexpectedly')
    result = am.process_anomaly(entry)
    assert '\U0001f534' in result, "HIGH alert should contain 🔴"
    print("PASS  test_emoji_matches_severity")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing AlertManager (Improvement 3)...\n")

    test_severity_high_off_hours_error()
    test_severity_high_critical_keyword()
    test_severity_high_auth_failure()
    test_severity_medium_error_business_hours()
    test_severity_medium_warn_off_hours()
    test_severity_low_warn_business_hours()

    test_first_occurrence_returns_alert()
    test_alert_contains_severity_and_level()
    test_alert_mentions_first_occurrence()

    test_duplicate_within_cooldown_suppressed()
    test_different_entries_not_suppressed()
    test_cooldown_expiry_allows_re_alert()

    test_incident_count_increments()
    test_repeat_occurrence_text_in_re_alert()
    test_active_incidents_returns_all()

    test_emoji_matches_severity()

    print("\n" + "=" * 55)
    print("All AlertManager tests passed.")
    print("=" * 55)

