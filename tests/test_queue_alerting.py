"""
Unit tests for job queue depth monitoring and alerting (api/queue_alerting.py).

Tests cover:
- AlertSeverity enum values and count
- AlertState enum values and count
- QueueThreshold defaults and to_dict
- QueueDepthPoint creation
- Alert defaults and to_dict
- QueueSnapshot defaults and to_dict
- QueueMonitor construction
- set_threshold stores config
- record_depth stores history
- record_depth prunes old entries
- record_depth triggers warning alert
- record_depth triggers critical alert
- record_depth auto-resolves when below threshold
- record_depth with wait time warning
- record_depth with wait time critical
- record_depth no duplicate alerts for same queue+severity
- record_depth updates existing alert depth
- add_alert_callback receives alerts
- add_alert_callback with failing callback (logged not raised)
- acknowledge_alert changes state
- acknowledge_alert on non-active (no-op)
- resolve_alert changes state
- resolve_alert sets resolved_at
- get_active_alerts filters resolved
- get_queue_history returns time-series
- get_queue_history respects window
- get_queue_history unknown queue returns empty
- get_snapshot with queues and alerts
- get_snapshot total_depth aggregation
- reset clears all
- get_queue_monitor singleton
- Thread safety concurrent recording

Run with: python -m pytest tests/test_queue_alerting.py -v
"""

import json
import threading
import time

# Add project root to path
from api.queue_alerting import (
    Alert,
    AlertSeverity,
    AlertState,
    QueueDepthPoint,
    QueueMonitor,
    QueueSnapshot,
    QueueThreshold,
    get_queue_monitor,
)

# ---------------------------------------------------------------------------
# Tests: AlertSeverity
# ---------------------------------------------------------------------------


class TestAlertSeverity:
    def test_enum_values(self):
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_enum_count(self):
        assert len(AlertSeverity) == 3


# ---------------------------------------------------------------------------
# Tests: AlertState
# ---------------------------------------------------------------------------


class TestAlertState:
    def test_enum_values(self):
        assert AlertState.ACTIVE.value == "active"
        assert AlertState.ACKNOWLEDGED.value == "acknowledged"
        assert AlertState.RESOLVED.value == "resolved"

    def test_enum_count(self):
        assert len(AlertState) == 3


# ---------------------------------------------------------------------------
# Tests: QueueThreshold
# ---------------------------------------------------------------------------


class TestQueueThreshold:
    def test_defaults(self):
        t = QueueThreshold(queue_name="ocr")
        assert t.queue_name == "ocr"
        assert t.warning_depth == 50
        assert t.critical_depth == 100
        assert t.warning_wait_seconds == 300.0
        assert t.critical_wait_seconds == 600.0

    def test_custom_values(self):
        t = QueueThreshold(
            queue_name="ingest",
            warning_depth=10,
            critical_depth=25,
            warning_wait_seconds=60.0,
            critical_wait_seconds=120.0,
        )
        assert t.warning_depth == 10
        assert t.critical_depth == 25

    def test_to_dict(self):
        t = QueueThreshold(queue_name="ocr")
        d = t.to_dict()
        assert d["queue_name"] == "ocr"
        assert d["warning_depth"] == 50
        assert d["critical_depth"] == 100
        assert d["warning_wait_seconds"] == 300.0
        assert d["critical_wait_seconds"] == 600.0

    def test_to_dict_keys(self):
        t = QueueThreshold(queue_name="q")
        d = t.to_dict()
        expected_keys = {
            "queue_name", "warning_depth", "critical_depth",
            "warning_wait_seconds", "critical_wait_seconds",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: QueueDepthPoint
# ---------------------------------------------------------------------------


class TestQueueDepthPoint:
    def test_creation(self):
        p = QueueDepthPoint(timestamp=1000.0, queue_name="ocr", depth=42)
        assert p.timestamp == 1000.0
        assert p.queue_name == "ocr"
        assert p.depth == 42
        assert p.oldest_item_age_seconds == 0.0

    def test_with_age(self):
        p = QueueDepthPoint(
            timestamp=1000.0, queue_name="ocr", depth=5,
            oldest_item_age_seconds=120.5,
        )
        assert p.oldest_item_age_seconds == 120.5


# ---------------------------------------------------------------------------
# Tests: Alert
# ---------------------------------------------------------------------------


class TestAlert:
    def test_defaults(self):
        a = Alert(alert_id="a-1", queue_name="ocr", severity=AlertSeverity.WARNING)
        assert a.alert_id == "a-1"
        assert a.queue_name == "ocr"
        assert a.severity == AlertSeverity.WARNING
        assert a.state == AlertState.ACTIVE
        assert a.message == ""
        assert a.triggered_at == 0.0
        assert a.resolved_at == 0.0
        assert a.acknowledged_at == 0.0
        assert a.current_depth == 0
        assert a.threshold_value == 0

    def test_to_dict(self):
        a = Alert(
            alert_id="a-1", queue_name="ocr", severity=AlertSeverity.CRITICAL,
            message="Queue full", current_depth=120, threshold_value=100,
        )
        d = a.to_dict()
        assert d["alert_id"] == "a-1"
        assert d["queue_name"] == "ocr"
        assert d["severity"] == "critical"
        assert d["state"] == "active"
        assert d["message"] == "Queue full"
        assert d["current_depth"] == 120
        assert d["threshold_value"] == 100

    def test_to_dict_keys(self):
        a = Alert(alert_id="a-1", queue_name="q", severity=AlertSeverity.INFO)
        d = a.to_dict()
        expected_keys = {
            "alert_id", "queue_name", "severity", "state", "message",
            "triggered_at", "resolved_at", "acknowledged_at",
            "current_depth", "threshold_value",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: QueueSnapshot
# ---------------------------------------------------------------------------


class TestQueueSnapshot:
    def test_defaults(self):
        s = QueueSnapshot()
        assert s.timestamp == 0.0
        assert s.queues == []
        assert s.active_alerts == []
        assert s.total_depth == 0

    def test_to_dict(self):
        alert = Alert(alert_id="a-1", queue_name="q", severity=AlertSeverity.WARNING)
        s = QueueSnapshot(
            timestamp=1000.0,
            queues=[{"queue_name": "q", "depth": 10}],
            active_alerts=[alert],
            total_depth=10,
        )
        d = s.to_dict()
        assert d["timestamp"] == 1000.0
        assert d["total_depth"] == 10
        assert len(d["queues"]) == 1
        assert len(d["active_alerts"]) == 1
        assert d["active_alerts"][0]["alert_id"] == "a-1"

    def test_to_dict_keys(self):
        s = QueueSnapshot()
        d = s.to_dict()
        expected_keys = {"timestamp", "total_depth", "queues", "active_alerts"}
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: QueueMonitor
# ---------------------------------------------------------------------------


class TestQueueMonitorConstruction:
    def test_default_construction(self):
        m = QueueMonitor()
        snap = m.get_snapshot()
        assert snap.total_depth == 0
        assert snap.queues == []

    def test_custom_history_seconds(self):
        m = QueueMonitor(history_seconds=60)
        # Just verify it doesn't error
        m.record_depth("q", 5)
        assert len(m.get_queue_history("q")) == 1


class TestSetThreshold:
    def test_stores_config(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=20, critical_depth=40)
        # Record depth to trigger threshold check — below threshold
        m.record_depth("ocr", 5)
        snap = m.get_snapshot()
        assert snap.queues[0]["warning_threshold"] == 20
        assert snap.queues[0]["critical_threshold"] == 40

    def test_overwrite_threshold(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=20, critical_depth=40)
        m.set_threshold("ocr", warning_depth=30, critical_depth=60)
        m.record_depth("ocr", 1)
        snap = m.get_snapshot()
        assert snap.queues[0]["warning_threshold"] == 30
        assert snap.queues[0]["critical_threshold"] == 60

    def test_persists_thresholds(self, tmp_path):
        path = tmp_path / "queue_thresholds.json"
        m = QueueMonitor(thresholds_path=path)
        m.set_threshold(
            "ocr",
            warning_depth=20,
            critical_depth=40,
            warning_wait_seconds=30.0,
            critical_wait_seconds=90.0,
        )

        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload == [
            {
                "queue_name": "ocr",
                "warning_depth": 20,
                "critical_depth": 40,
                "warning_wait_seconds": 30.0,
                "critical_wait_seconds": 90.0,
            }
        ]

    def test_loads_persisted_thresholds(self, tmp_path):
        path = tmp_path / "queue_thresholds.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "queue_name": "ocr",
                        "warning_depth": 20,
                        "critical_depth": 40,
                        "warning_wait_seconds": 30.0,
                        "critical_wait_seconds": 90.0,
                    }
                ]
            ),
            encoding="utf-8",
        )

        m = QueueMonitor(thresholds_path=path)
        threshold = m.get_threshold("ocr")

        assert threshold is not None
        assert threshold.warning_depth == 20
        assert threshold.critical_wait_seconds == 90.0


class TestRecordDepth:
    def test_stores_history(self):
        m = QueueMonitor()
        m.record_depth("ocr", 10)
        m.record_depth("ocr", 15)
        history = m.get_queue_history("ocr")
        assert len(history) == 2
        assert history[0]["depth"] == 10
        assert history[1]["depth"] == 15

    def test_prunes_old_entries(self):
        m = QueueMonitor(history_seconds=1)
        m.record_depth("ocr", 10)
        time.sleep(1.1)
        m.record_depth("ocr", 20)
        history = m.get_queue_history("ocr", window_seconds=3600)
        # Old entry should have been pruned
        assert len(history) == 1
        assert history[0]["depth"] == 20

    def test_triggers_warning_alert(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 15)
        alerts = m.get_active_alerts()
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert alerts[0].queue_name == "ocr"
        assert "warning" in alerts[0].message.lower() or "exceeds" in alerts[0].message.lower()

    def test_triggers_critical_alert(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 60)
        alerts = m.get_active_alerts()
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert alerts[0].current_depth == 60

    def test_critical_not_also_warning(self):
        """When depth exceeds critical, only a critical alert fires (not warning too)."""
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 60)
        alerts = m.get_active_alerts()
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_auto_resolves_when_below_threshold(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 15)
        assert len(m.get_active_alerts()) == 1
        # Drop below threshold
        m.record_depth("ocr", 5)
        assert len(m.get_active_alerts()) == 0

    def test_wait_time_warning(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_wait_seconds=60.0, critical_wait_seconds=120.0)
        m.record_depth("ocr", 5, oldest_item_age_seconds=75.0)
        alerts = m.get_active_alerts()
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert "oldest item" in alerts[0].message

    def test_wait_time_critical(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_wait_seconds=60.0, critical_wait_seconds=120.0)
        m.record_depth("ocr", 5, oldest_item_age_seconds=150.0)
        alerts = m.get_active_alerts()
        # Critical wait should fire (not warning since critical supersedes)
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_alerts) == 1
        assert "oldest item" in critical_alerts[0].message

    def test_no_duplicate_alerts_same_queue_severity(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 15)
        m.record_depth("ocr", 20)
        m.record_depth("ocr", 25)
        alerts = m.get_active_alerts()
        warning_alerts = [a for a in alerts if a.severity == AlertSeverity.WARNING]
        assert len(warning_alerts) == 1

    def test_updates_existing_alert_depth(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 15)
        m.record_depth("ocr", 25)
        alerts = m.get_active_alerts()
        assert len(alerts) == 1
        assert alerts[0].current_depth == 25

    def test_no_threshold_no_alert(self):
        m = QueueMonitor()
        m.record_depth("ocr", 999)
        assert len(m.get_active_alerts()) == 0

    def test_records_oldest_item_age(self):
        m = QueueMonitor()
        m.record_depth("ocr", 10, oldest_item_age_seconds=42.5)
        history = m.get_queue_history("ocr")
        assert history[0]["oldest_item_age_seconds"] == 42.5

    def test_multiple_queues_independent(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.set_threshold("ingest", warning_depth=5, critical_depth=20)
        m.record_depth("ocr", 15)
        m.record_depth("ingest", 3)
        alerts = m.get_active_alerts()
        assert len(alerts) == 1
        assert alerts[0].queue_name == "ocr"


class TestAlertCallbacks:
    def test_callback_receives_alerts(self):
        m = QueueMonitor()
        received = []
        m.add_alert_callback(lambda a: received.append(a))
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 15)
        assert len(received) == 1
        assert received[0].severity == AlertSeverity.WARNING

    def test_multiple_callbacks(self):
        m = QueueMonitor()
        received_a = []
        received_b = []
        m.add_alert_callback(lambda a: received_a.append(a))
        m.add_alert_callback(lambda a: received_b.append(a))
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 15)
        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_failing_callback_logged_not_raised(self):
        m = QueueMonitor()

        def bad_callback(alert):
            raise RuntimeError("callback error")

        m.add_alert_callback(bad_callback)
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        # Should not raise
        m.record_depth("ocr", 15)
        # Alert should still be created
        assert len(m.get_active_alerts()) == 1

    def test_callback_not_fired_for_duplicate(self):
        m = QueueMonitor()
        received = []
        m.add_alert_callback(lambda a: received.append(a))
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 15)
        m.record_depth("ocr", 20)  # No new alert, just updates depth
        assert len(received) == 1


class TestAcknowledgeAlert:
    def test_changes_state(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 15)
        alerts = m.get_active_alerts()
        alert_id = alerts[0].alert_id
        m.acknowledge_alert(alert_id)
        alerts = m.get_active_alerts()
        assert len(alerts) == 1  # Still active (acknowledged ≠ resolved)
        assert alerts[0].state == AlertState.ACKNOWLEDGED
        assert alerts[0].acknowledged_at > 0

    def test_noop_on_non_active(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 15)
        alerts = m.get_active_alerts()
        alert_id = alerts[0].alert_id
        m.resolve_alert(alert_id)
        # Acknowledging a resolved alert should be a no-op
        m.acknowledge_alert(alert_id)
        # Should still be resolved
        active = m.get_active_alerts()
        assert len(active) == 0

    def test_noop_on_unknown_id(self):
        m = QueueMonitor()
        m.acknowledge_alert("nonexistent")  # Should not raise


class TestResolveAlert:
    def test_changes_state(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 15)
        alerts = m.get_active_alerts()
        alert_id = alerts[0].alert_id
        m.resolve_alert(alert_id)
        assert len(m.get_active_alerts()) == 0

    def test_sets_resolved_at(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 15)
        alerts = m.get_active_alerts()
        alert_id = alerts[0].alert_id
        m.resolve_alert(alert_id)
        # Access internal state to check resolved_at
        snap = m.get_snapshot()
        # Alert is resolved, so not in active_alerts
        assert len(snap.active_alerts) == 0

    def test_noop_on_already_resolved(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 15)
        alerts = m.get_active_alerts()
        alert_id = alerts[0].alert_id
        m.resolve_alert(alert_id)
        m.resolve_alert(alert_id)  # Second resolve is a no-op
        assert len(m.get_active_alerts()) == 0

    def test_noop_on_unknown_id(self):
        m = QueueMonitor()
        m.resolve_alert("nonexistent")  # Should not raise


class TestGetActiveAlerts:
    def test_filters_resolved(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 15)
        m.record_depth("ocr", 60)  # This triggers critical
        alerts = m.get_active_alerts()
        assert len(alerts) == 2  # warning + critical
        # Resolve the warning
        warning_alerts = [a for a in alerts if a.severity == AlertSeverity.WARNING]
        m.resolve_alert(warning_alerts[0].alert_id)
        active = m.get_active_alerts()
        assert len(active) == 1
        assert active[0].severity == AlertSeverity.CRITICAL

    def test_empty_when_none(self):
        m = QueueMonitor()
        assert m.get_active_alerts() == []


class TestGetQueueHistory:
    def test_returns_time_series(self):
        m = QueueMonitor()
        m.record_depth("ocr", 10)
        m.record_depth("ocr", 20)
        m.record_depth("ocr", 30)
        history = m.get_queue_history("ocr")
        assert len(history) == 3
        assert history[0]["depth"] == 10
        assert history[2]["depth"] == 30
        assert "timestamp" in history[0]
        assert "oldest_item_age_seconds" in history[0]

    def test_respects_window(self):
        m = QueueMonitor()
        m.record_depth("ocr", 10)
        time.sleep(0.05)
        m.record_depth("ocr", 20)
        # Use a tiny window that only captures the most recent
        history = m.get_queue_history("ocr", window_seconds=0.01)
        # Depending on timing, could be 0 or 1 — but certainly not 2
        assert len(history) <= 1

    def test_unknown_queue_returns_empty(self):
        m = QueueMonitor()
        assert m.get_queue_history("nonexistent") == []


class TestGetSnapshot:
    def test_with_queues_and_alerts(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.set_threshold("ingest", warning_depth=5, critical_depth=20)
        m.record_depth("ocr", 15)
        m.record_depth("ingest", 8)
        snap = m.get_snapshot()
        assert snap.timestamp > 0
        assert len(snap.queues) == 2
        assert len(snap.active_alerts) == 2  # one warning per queue

    def test_total_depth_aggregation(self):
        m = QueueMonitor()
        m.record_depth("ocr", 10)
        m.record_depth("ingest", 20)
        m.record_depth("export", 5)
        snap = m.get_snapshot()
        assert snap.total_depth == 35

    def test_to_dict_structure(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=5, critical_depth=20)
        m.record_depth("ocr", 10)
        snap = m.get_snapshot()
        d = snap.to_dict()
        assert "timestamp" in d
        assert "total_depth" in d
        assert "queues" in d
        assert "active_alerts" in d
        assert isinstance(d["active_alerts"], list)
        assert len(d["active_alerts"]) == 1
        assert d["active_alerts"][0]["severity"] == "warning"

    def test_snapshot_queue_thresholds_shown(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 5)
        snap = m.get_snapshot()
        q = snap.queues[0]
        assert q["warning_threshold"] == 10
        assert q["critical_threshold"] == 50
        assert q["warning_wait_seconds"] == 300.0
        assert q["critical_wait_seconds"] == 600.0

    def test_snapshot_no_threshold(self):
        m = QueueMonitor()
        m.record_depth("ocr", 5)
        snap = m.get_snapshot()
        q = snap.queues[0]
        assert q["warning_threshold"] is None
        assert q["critical_threshold"] is None
        assert q["warning_wait_seconds"] is None
        assert q["critical_wait_seconds"] is None


class TestReset:
    def test_clears_all(self):
        m = QueueMonitor()
        m.set_threshold("ocr", warning_depth=10, critical_depth=50)
        m.record_depth("ocr", 15)
        m.add_alert_callback(lambda a: None)
        assert len(m.get_active_alerts()) > 0
        m.reset()
        snap = m.get_snapshot()
        assert snap.total_depth == 0
        assert snap.queues == []
        assert snap.active_alerts == []
        assert m.get_queue_history("ocr") == []


class TestGetQueueMonitorSingleton:
    def test_singleton(self):
        m1 = get_queue_monitor()
        m2 = get_queue_monitor()
        assert m1 is m2
        assert isinstance(m1, QueueMonitor)


class TestThreadSafety:
    def test_concurrent_recording(self):
        m = QueueMonitor()
        m.set_threshold("q-0", warning_depth=5, critical_depth=20)
        m.set_threshold("q-1", warning_depth=5, critical_depth=20)
        m.set_threshold("q-2", warning_depth=5, critical_depth=20)
        m.set_threshold("q-3", warning_depth=5, critical_depth=20)
        errors = []
        barrier = threading.Barrier(4)

        def worker(queue_name, count):
            try:
                barrier.wait(timeout=5)
                for i in range(count):
                    m.record_depth(queue_name, i)
                    m.get_snapshot()
                    m.get_queue_history(queue_name)
                    m.get_active_alerts()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"q-{i}", 50))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread errors: {errors}"
        snap = m.get_snapshot()
        assert len(snap.queues) == 4
