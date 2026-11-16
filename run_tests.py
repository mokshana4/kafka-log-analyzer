"""
run_tests.py
Master test runner — executes every test suite in the project and
prints a consolidated pass/fail summary.

Run with:
    python run_tests.py

No live Kafka, Elasticsearch, or Slack connection is required.
"""

import sys
sys.path.append('.')

import importlib
import traceback

# ── Test suites ───────────────────────────────────────────────────────────────
# Each entry is (display_name, module_path, runner_function_name)

SUITES = [
    # ML feature extraction & anomaly detection (Improvement 1)
    ('ML — feature extraction & detection',
     'ml.test_anomaly_detection',
     None),                        # has its own __main__ block; we import and call

    # Real-time consumer logic (Improvement 2)
    ('Consumer — real-time detection logic',
     'kafka_client.test_consumer_logic',
     None),

    # Alert manager (Improvement 3)
    ('Alerting — severity / dedup / cooldown / incidents',
     'alerting.test_alert_manager',
     None),

    # Scenario tests (Improvement 4)
    ('Scenarios — 7 scenario types + detection metrics',
     'test_scenarios',
     None),

    # Performance benchmarks (Improvement 4)
    ('Performance — throughput & latency',
     'test_performance',
     None),

    # Reliability and error handling (Improvement 5)
    ('Reliability — config / retries / error handling',
     'test_error_handling',
     None),
]

# Functions to call per module (all tests in each module, in order)
SUITE_RUNNERS = {
    'ml.test_anomaly_detection': [
        'test_empty_input',
        'test_missing_message_field',
        'test_unknown_level_dropped',
        'test_malformed_timestamp_dropped',
        'test_unseen_message_does_not_crash',
        'test_feature_shape',
        'test_smoke_unusual_patterns_score_lower_than_normal',
        'run_labeled_evaluation',
    ],
    'kafka_client.test_consumer_logic': [
        'test_load_missing_file',
        'test_load_valid_bundle',
        'test_load_old_format_bundle',
        'test_normal_entry_not_flagged',
        'test_anomalous_entry_scores_lower',
        'test_malformed_entry_does_not_crash',
        'test_missing_columns_handled',
    ],
    'alerting.test_alert_manager': [
        'test_severity_high_off_hours_error',
        'test_severity_high_critical_keyword',
        'test_severity_high_auth_failure',
        'test_severity_medium_error_business_hours',
        'test_severity_medium_warn_off_hours',
        'test_severity_low_warn_business_hours',
        'test_first_occurrence_returns_alert',
        'test_alert_contains_severity_and_level',
        'test_alert_mentions_first_occurrence',
        'test_duplicate_within_cooldown_suppressed',
        'test_different_entries_not_suppressed',
        'test_cooldown_expiry_allows_re_alert',
        'test_incident_count_increments',
        'test_repeat_occurrence_text_in_re_alert',
        'test_active_incidents_returns_all',
        'test_emoji_matches_severity',
    ],
    'test_scenarios': [
        'test_scenario_normal_logs',
        'test_scenario_db_failures',
        'test_scenario_server_crashes',
        'test_scenario_timeouts_and_external_failures',
        'test_scenario_auth_failures',
        'test_scenario_malformed_logs',
        'test_scenario_invalid_timestamps',
        'test_detection_metrics',
    ],
    'test_performance': [
        'test_throughput_acceptable',
        'test_latency_acceptable',
    ],
    'test_error_handling': [
        'test_config_defaults',
        'test_config_env_var_override',
        'test_elasticsearch_failure_returns_false',
        'test_elasticsearch_success_returns_true',
        'test_elasticsearch_none_entry_returns_false',
        'test_save_all_counts_failures',
        'test_bad_bytes_message_skipped',
        'test_unparseable_line_skipped',
        'test_valid_message_increments_counter',
        'test_anomalous_message_increments_anomaly_counter',
        'test_kafka_retry_attempts_respected',
    ],
}

# ── Runner ────────────────────────────────────────────────────────────────────

def run_all():
    total_passed  = 0
    total_failed  = 0
    suite_results = []

    for display_name, module_path, _ in SUITES:
        print("\n" + "─" * 60)
        print("Suite: {}".format(display_name))
        print("─" * 60)

        try:
            mod = importlib.import_module(module_path)
        except Exception as e:
            print("[ERROR] Could not import {}: {}".format(module_path, e))
            traceback.print_exc()
            suite_results.append((display_name, 0, 1))
            total_failed += 1
            continue

        functions = SUITE_RUNNERS.get(module_path, [])
        passed = 0
        failed = 0

        for fn_name in functions:
            fn = getattr(mod, fn_name, None)
            if fn is None:
                print("[SKIP] {} not found in {}".format(fn_name, module_path))
                continue
            try:
                fn()
                passed += 1
            except AssertionError as e:
                print("[FAIL] {}: {}".format(fn_name, e))
                failed += 1
            except Exception as e:
                print("[ERROR] {} raised an unexpected exception: {}".format(fn_name, e))
                traceback.print_exc()
                failed += 1

        suite_results.append((display_name, passed, failed))
        total_passed += passed
        total_failed += failed

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, p, f in suite_results:
        status = "OK  " if f == 0 else "FAIL"
        print("  [{}]  {:>2} passed  {:>2} failed   {}".format(status, p, f, name))
    print("─" * 60)
    print("  TOTAL:  {} passed,  {} failed".format(total_passed, total_failed))
    print("=" * 60)

    if total_failed == 0:
        print("\nAll tests passed.")
    else:
        print("\n{} test(s) failed.".format(total_failed))
        sys.exit(1)


if __name__ == '__main__':
    run_all()

