"""
alerting/alert_manager.py
Intelligent alerting with severity classification, deduplication,
cooldown, and incident grouping (Improvement 3).

Responsibilities:
  - classify_severity : assign HIGH / MEDIUM / LOW to an anomalous entry
  - process_anomaly   : decide whether to fire an alert or suppress it
                        (deduplication + cooldown), and return a formatted
                        message string when firing
  - active_incidents  : return a summary of all tracked incidents
                        (useful for demos and debugging)

Design decisions:
  - State is kept in memory (a plain dict). That is intentional for a
    local project — no extra infrastructure needed.
  - The cooldown window and business-hours range are module-level
    constants so they are easy to adjust.
  - process_anomaly returns None when suppressed so the caller only
    needs to check "if alert_text:" before sending.
"""

import time
import sys
sys.path.append('.')
import config

# ── Tunable constants ────────────────────────────────────────────────────────

COOLDOWN_SECONDS = config.ALERT_COOLDOWN_SECONDS  # read from env / config
BUSINESS_HOURS   = range(8, 19) # 08:00 – 18:59 considered normal working hours

# Keywords in the message body that escalate an alert to HIGH severity.
# These match the fixed vocabulary in ml/features.py so they are consistent.
HIGH_SEVERITY_KEYWORDS = {
    'crashed', 'authentication', 'failed',
}

SEVERITY_EMOJI = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🔵'}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_hour(entry):
    """Return hour-of-day (int) from entry['timestamp'], or 12 on failure."""
    ts = entry.get('timestamp', '')
    try:
        return int(ts[11:13])
    except (ValueError, IndexError, TypeError):
        return 12


def _is_off_hours(hour):
    return hour not in BUSINESS_HOURS


def _has_critical_keyword(message):
    words = set(message.lower().split())
    return bool(words & HIGH_SEVERITY_KEYWORDS)


# ── AlertManager ─────────────────────────────────────────────────────────────

class AlertManager:
    """
    Tracks anomalous log entries, groups them into incidents, suppresses
    repeated alerts during the cooldown window, and formats rich Slack
    messages when an alert is allowed through.
    """

    def __init__(self, cooldown_seconds=COOLDOWN_SECONDS):
        self.cooldown = cooldown_seconds
        # fingerprint -> {count, first_seen, last_seen, last_sent}
        self._incidents = {}

    # ── Severity classification ───────────────────────────────────────────────

    def classify_severity(self, entry):
        """
        Return 'HIGH', 'MEDIUM', or 'LOW'.

        Rules (applied in priority order):
          HIGH   — ERROR at off-hours OR message contains a critical keyword
          MEDIUM — ERROR at business hours, OR any level at off-hours
          LOW    — anything else (WARN/DEBUG anomaly at normal hours)
        """
        level   = entry.get('level', '').upper()
        message = entry.get('message', '')
        hour    = _parse_hour(entry)
        off     = _is_off_hours(hour)
        crit    = _has_critical_keyword(message)

        if level == 'ERROR' and (off or crit):
            return 'HIGH'
        if level == 'ERROR' or off:
            return 'MEDIUM'
        return 'LOW'

    # ── Fingerprint & reason ─────────────────────────────────────────────────

    @staticmethod
    def _fingerprint(entry):
        """Deduplication key: (level, message) tuple."""
        return (entry.get('level', ''), entry.get('message', ''))

    def _build_reason(self, entry, severity):
        """Human-readable explanation of why this entry was flagged."""
        level   = entry.get('level', '')
        message = entry.get('message', '')
        hour    = _parse_hour(entry)

        parts = []
        if _is_off_hours(hour):
            parts.append('off-hours occurrence ({}:xx)'.format(hour))
        if _has_critical_keyword(message):
            parts.append('critical keyword in message')
        if not parts:
            if level == 'ERROR':
                parts.append('error-level log flagged by model')
            elif level == 'WARN':
                parts.append('warning-level log flagged by model')
            else:
                parts.append('flagged as unusual by anomaly model')
        return ', '.join(parts)

    # ── Core method ──────────────────────────────────────────────────────────

    def process_anomaly(self, entry):
        """
        Decide whether to send an alert for this anomalous log entry.

        Returns a formatted alert string if the alert should be sent.
        Returns None if the alert is suppressed by the cooldown window
        (the incident count is still incremented so nothing is lost).
        """
        fp       = self._fingerprint(entry)
        now      = time.time()
        severity = self.classify_severity(entry)

        if fp in self._incidents:
            inc = self._incidents[fp]
            inc['count']     += 1
            inc['last_seen']  = entry.get('timestamp', '')

            if now - inc['last_sent'] < self.cooldown:
                # Still in cooldown — suppress the alert
                return None

            # Cooldown has expired — allow the alert through
            inc['last_sent'] = now
        else:
            # First occurrence of this (level, message) combination
            self._incidents[fp] = {
                'first_seen': entry.get('timestamp', ''),
                'last_seen':  entry.get('timestamp', ''),
                'count':      1,
                'last_sent':  now,
            }

        return self._format_alert(entry, severity)

    # ── Formatting ───────────────────────────────────────────────────────────

    def _format_alert(self, entry, severity):
        """Return the full Slack message string for this alert."""
        fp  = self._fingerprint(entry)
        inc = self._incidents[fp]

        emoji  = SEVERITY_EMOJI[severity]
        reason = self._build_reason(entry, severity)
        sep    = '\u2501' * 38          # ━━━━━━━━━ divider line

        if inc['count'] == 1:
            occurrence = 'First occurrence'
        else:
            occurrence = 'Occurrence #{n}  (first seen: {first})'.format(
                n=inc['count'], first=inc['first_seen'])

        return (
            '{emoji} {severity} SEVERITY ANOMALY\n'
            '{sep}\n'
            'Level     : {level}\n'
            'Time      : {timestamp}\n'
            'Message   : {message}\n'
            'Reason    : {reason}\n'
            'Incident  : {occurrence}\n'
            '{sep}'
        ).format(
            emoji=emoji,
            severity=severity,
            sep=sep,
            level=entry.get('level', ''),
            timestamp=entry.get('timestamp', ''),
            message=entry.get('message', ''),
            reason=reason,
            occurrence=occurrence,
        )

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def active_incidents(self):
        """
        Return a summary dict of all tracked incidents.
        Useful for end-of-session reporting or debugging.

            key   : 'LEVEL | message text'
            value : {count, first_seen, last_seen}
        """
        return {
            '{} | {}'.format(fp[0], fp[1]): {
                'count':      inc['count'],
                'first_seen': inc['first_seen'],
                'last_seen':  inc['last_seen'],
            }
            for fp, inc in self._incidents.items()
        }

