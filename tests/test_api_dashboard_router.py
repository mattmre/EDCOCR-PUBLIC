"""Tests for dashboard, fleet, alerts, and analytics API routers."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.analytics import get_analytics_store
from api.dashboard import get_collector
from api.database import get_engine, reset_engine
from api.fleet_status import get_fleet_tracker
from api.queue_alerting import get_queue_monitor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset global singletons between tests."""
    get_collector().reset()
    get_fleet_tracker().reset()
    get_queue_monitor().reset()
    get_analytics_store().reset()
    yield
    get_collector().reset()
    get_fleet_tracker().reset()
    get_queue_monitor().reset()
    get_analytics_store().reset()


@pytest.fixture()
def client(tmp_path):
    """FastAPI TestClient with dashboard endpoints enabled."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with (
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.job_manager.config") as mock_config,
        patch.dict("os.environ", {"ENABLE_DASHBOARD": "true"}),
    ):
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 1
        mock_config.MAX_CONCURRENT_JOBS = 64

        from api.main import create_app

        app = create_app()
        app.state.limiter.enabled = False
        app.state.limiter.reset()
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Dashboard endpoints
# ---------------------------------------------------------------------------


class TestDashboardSnapshot:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "throughput" in data
        assert "latency" in data
        assert "jobs" in data

    def test_with_window_param(self, client):
        resp = client.get("/api/v1/dashboard?window=HOUR_1")
        assert resp.status_code == 200

    def test_invalid_window_falls_back(self, client):
        resp = client.get("/api/v1/dashboard?window=INVALID")
        assert resp.status_code == 200

    def test_with_data(self, client):
        collector = get_collector()
        collector.record_throughput(pages=10, documents=1, bytes_processed=5000)
        collector.record_latency(job_id="j1", total_ms=150.0)
        collector.update_job_counts(total=5, active=2, completed=3)

        resp = client.get("/api/v1/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"]["total"] == 5
        assert data["jobs"]["completed"] == 3


class TestDashboardThroughput:
    def test_returns_200_empty(self, client):
        resp = client.get("/api/v1/dashboard/throughput")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_data(self, client):
        collector = get_collector()
        collector.record_throughput(pages=5, documents=1, bytes_processed=1000)

        resp = client.get("/api/v1/dashboard/throughput?window=HOUR_1&bucket_seconds=60")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        assert "pages" in data[0]


class TestDashboardLatency:
    def test_returns_200_empty(self, client):
        resp = client.get("/api/v1/dashboard/latency")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_data(self, client):
        collector = get_collector()
        collector.record_latency(job_id="j1", total_ms=100.0)

        resp = client.get("/api/v1/dashboard/latency?window=HOUR_1&bucket_seconds=60")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        assert "avg_ms" in data[0]


# ---------------------------------------------------------------------------
# Fleet endpoints
# ---------------------------------------------------------------------------


class TestFleetSnapshot:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/fleet")
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "gpu" in data
        assert "workers" in data

    def test_with_workers(self, client):
        tracker = get_fleet_tracker()
        tracker.register_worker("w1", hostname="host1", capabilities=["ocr"])
        tracker.register_worker("w2", hostname="host2", capabilities=["nlp"])

        resp = client.get("/api/v1/fleet")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["total_workers"] == 2


class TestFleetWorkerDetail:
    def test_returns_worker(self, client):
        tracker = get_fleet_tracker()
        tracker.register_worker("w1", hostname="host1", capabilities=["ocr"])

        resp = client.get("/api/v1/fleet/w1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["worker_id"] == "w1"
        assert data["hostname"] == "host1"

    def test_worker_not_found(self, client):
        resp = client.get("/api/v1/fleet/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["error"]


# ---------------------------------------------------------------------------
# Alerts endpoints
# ---------------------------------------------------------------------------


class TestAlertsSnapshot:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_depth" in data
        assert "active_alerts" in data


class TestActiveAlerts:
    def test_returns_empty_list(self, client):
        resp = client.get("/api/v1/alerts/active")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_active_alerts(self, client):
        monitor = get_queue_monitor()
        monitor.set_threshold("gpu", warning_depth=5, critical_depth=10)
        monitor.record_depth("gpu", 50)

        resp = client.get("/api/v1/alerts/active")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        assert data[0]["queue_name"] == "gpu"


class TestAcknowledgeAlert:
    def test_acknowledge_existing_alert(self, client):
        monitor = get_queue_monitor()
        monitor.set_threshold("gpu", warning_depth=5, critical_depth=10)
        monitor.record_depth("gpu", 50)

        # Get the alert ID
        alerts = monitor.get_active_alerts()
        assert len(alerts) > 0
        alert_id = alerts[0].alert_id

        resp = client.post(f"/api/v1/alerts/{alert_id}/acknowledge")
        assert resp.status_code == 200
        assert resp.json()["status"] == "acknowledged"

    def test_acknowledge_not_found(self, client):
        resp = client.post("/api/v1/alerts/nonexistent/acknowledge")
        assert resp.status_code == 404


class TestResolveAlert:
    def test_resolve_existing_alert(self, client):
        monitor = get_queue_monitor()
        monitor.set_threshold("gpu", warning_depth=5, critical_depth=10)
        monitor.record_depth("gpu", 50)

        alerts = monitor.get_active_alerts()
        assert len(alerts) > 0
        alert_id = alerts[0].alert_id

        resp = client.post(f"/api/v1/alerts/{alert_id}/resolve")
        assert resp.status_code == 200
        assert resp.json()["status"] == "resolved"

    def test_resolve_not_found(self, client):
        resp = client.post("/api/v1/alerts/nonexistent/resolve")
        assert resp.status_code == 404


class TestQueueHistory:
    def test_returns_empty(self, client):
        resp = client.get("/api/v1/queues/gpu/history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_history(self, client):
        monitor = get_queue_monitor()
        monitor.record_depth("gpu", 10)
        monitor.record_depth("gpu", 15)

        resp = client.get("/api/v1/queues/gpu/history?window_seconds=3600")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["depth"] == 10


# ---------------------------------------------------------------------------
# Analytics endpoints
# ---------------------------------------------------------------------------


class TestAnalyticsSummary:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_jobs" in data
        assert "success_rate" in data

    def test_with_data(self, client):
        store = get_analytics_store()
        store.record_job("j1", pages=10, duration_seconds=5.0, success=True, engine="paddle")
        store.record_job("j2", pages=5, duration_seconds=3.0, success=True, engine="tesseract")

        resp = client.get("/api/v1/analytics?hours=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_jobs"] == 2
        assert data["total_pages"] == 15


class TestAnalyticsTrends:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/analytics/trends")
        assert resp.status_code == 200
        data = resp.json()
        assert "current" in data
        assert "previous" in data
        assert "changes" in data


class TestAnalyticsSeries:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/analytics/series?hours=1&granularity=HOURLY")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_invalid_granularity_falls_back(self, client):
        resp = client.get("/api/v1/analytics/series?granularity=INVALID")
        assert resp.status_code == 200


class TestAnalyticsTopEngines:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/analytics/engines")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_with_data(self, client):
        store = get_analytics_store()
        store.record_job("j1", engine="paddle", pages=1)
        store.record_job("j2", engine="paddle", pages=1)
        store.record_job("j3", engine="tesseract", pages=1)

        resp = client.get("/api/v1/analytics/engines?hours=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        # paddle should be first (count=2)
        assert data[0]["engine"] == "paddle"
        assert data[0]["count"] == 2


class TestAnalyticsTopLanguages:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/analytics/languages")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_with_data(self, client):
        store = get_analytics_store()
        store.record_job("j1", language="en", pages=1)
        store.record_job("j2", language="en", pages=1)
        store.record_job("j3", language="fr", pages=1)

        resp = client.get("/api/v1/analytics/languages?hours=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2


class TestAnalyticsWorkerStats:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/analytics/workers")
        assert resp.status_code == 200

    def test_with_data(self, client):
        store = get_analytics_store()
        store.record_job("j1", worker_id="w1", pages=10, success=True, duration_seconds=5.0)
        store.record_job("j2", worker_id="w1", pages=5, success=False, duration_seconds=2.0)

        resp = client.get("/api/v1/analytics/workers?hours=1")
        assert resp.status_code == 200
        data = resp.json()
        assert "w1" in data
        assert data["w1"]["total_jobs"] == 2
        assert data["w1"]["successes"] == 1
        assert data["w1"]["failures"] == 1


# ---------------------------------------------------------------------------
# Feature gate tests
# ---------------------------------------------------------------------------


class TestDashboardFeatureGate:
    def test_disabled_by_default(self, tmp_path):
        """Dashboard endpoints are 404 when ENABLE_DASHBOARD is not set."""
        reset_engine()
        db_file = str(tmp_path / "gate_test.db")
        with (
            patch("api.config.DB_PATH", db_file),
            patch("api.database.DB_PATH", db_file),
            patch("api.config.SOURCE_FOLDER", str(tmp_path)),
            patch("api.config.OUTPUT_FOLDER", str(tmp_path)),
            patch.dict("os.environ", {"ENABLE_DASHBOARD": ""}, clear=False),
        ):
            reset_engine()
            get_engine(db_file)

            from api.main import create_app

            app = create_app()
            app.state.limiter.enabled = False
            test_client = TestClient(app)

            resp = test_client.get("/api/v1/dashboard")
            assert resp.status_code == 404

            resp = test_client.get("/api/v1/fleet")
            assert resp.status_code == 404

            resp = test_client.get("/api/v1/alerts")
            assert resp.status_code == 404

            resp = test_client.get("/api/v1/analytics")
            assert resp.status_code == 404

            reset_engine()
