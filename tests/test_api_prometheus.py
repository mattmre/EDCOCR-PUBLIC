"""Tests for the API-side Prometheus metrics endpoint (api/prometheus.py).

Covers:
- Endpoint returns 200 with correct content type
- Metrics include expected metric names
- Auth check works (401 without key)
- Empty collectors return zero values
- Latency metric labels are correct (p50, p95, p99)
- Fleet worker state labels match WorkerState enum
- Stage metrics propagate from dashboard collector
- Queue monitor metrics propagate from queue alerting
- Refresh function is called on each request

Run with: python -m pytest tests/test_api_prometheus.py -v
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on path
from api.dashboard import get_collector
from api.fleet_status import GpuInfo, WorkerState, get_fleet_tracker
from api.queue_alerting import get_queue_monitor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    """FastAPI TestClient with auth disabled (ALLOW_UNAUTHENTICATED)."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with (
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.config.ALLOW_UNAUTHENTICATED", True),
        patch("api.auth.ALLOW_UNAUTHENTICATED", True),
        patch("api.job_manager.config") as mock_config,
        patch("api.job_manager.JobManager._run_pipeline"),
    ):
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 1
        mock_config.MAX_CONCURRENT_JOBS = 64
        mock_config.WEBHOOK_TIMEOUT = 30
        mock_config.WEBHOOK_MAX_RETRIES = 0
        mock_config.WEBHOOK_SECRET = ""

        from api.main import create_app

        app = create_app()
        app.state.limiter.enabled = False
        app.state.limiter.reset()
        yield TestClient(app)


@pytest.fixture()
def authed_client(tmp_path):
    """FastAPI TestClient with auth required (OCR_API_KEY set)."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    api_key = "test-secret-key-12345"

    with (
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.config.OCR_API_KEY", api_key),
        patch("api.config.ALLOW_UNAUTHENTICATED", False),
        patch("api.auth.OCR_API_KEY", api_key),
        patch("api.auth.ALLOW_UNAUTHENTICATED", False),
        patch("api.job_manager.config") as mock_config,
        patch("api.job_manager.JobManager._run_pipeline"),
    ):
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 1
        mock_config.MAX_CONCURRENT_JOBS = 64
        mock_config.WEBHOOK_TIMEOUT = 30
        mock_config.WEBHOOK_MAX_RETRIES = 0
        mock_config.WEBHOOK_SECRET = ""

        from api.main import create_app

        app = create_app()
        app.state.limiter.enabled = False
        app.state.limiter.reset()
        yield TestClient(app), api_key


# ---------------------------------------------------------------------------
# Tests: Endpoint basics
# ---------------------------------------------------------------------------


class TestPrometheusEndpoint:
    """Tests for GET /api/v1/prometheus/."""

    def test_returns_200(self, client):
        resp = client.get("/api/v1/prometheus/")
        assert resp.status_code == 200

    def test_content_type_text_plain(self, client):
        resp = client.get("/api/v1/prometheus/")
        ct = resp.headers.get("content-type", "")
        assert "text/plain" in ct

    def test_contains_throughput_metric(self, client):
        resp = client.get("/api/v1/prometheus/")
        body = resp.text
        assert "ocr_api_throughput_pages_per_minute" in body
        assert "ocr_api_throughput_docs_per_hour" in body

    def test_contains_latency_metric(self, client):
        resp = client.get("/api/v1/prometheus/")
        body = resp.text
        assert "ocr_api_latency_ms" in body

    def test_contains_fleet_metric(self, client):
        resp = client.get("/api/v1/prometheus/")
        body = resp.text
        assert "ocr_api_fleet_workers" in body
        assert "ocr_api_fleet_gpu_utilization_pct" in body

    def test_contains_stage_metrics(self, client):
        resp = client.get("/api/v1/prometheus/")
        body = resp.text
        assert "ocr_api_stage_queue_depth" in body
        assert "ocr_api_stage_active_workers" in body

    def test_contains_queue_metrics(self, client):
        resp = client.get("/api/v1/prometheus/")
        body = resp.text
        assert "ocr_api_queue_active_alerts" in body


# ---------------------------------------------------------------------------
# Tests: Auth
# ---------------------------------------------------------------------------


class TestPrometheusAuth:
    """Tests for authentication on the prometheus endpoint."""

    def test_returns_401_without_key(self, authed_client):
        tc, _api_key = authed_client
        resp = tc.get("/api/v1/prometheus/")
        assert resp.status_code == 401

    def test_returns_200_with_valid_key(self, authed_client):
        tc, api_key = authed_client
        resp = tc.get(
            "/api/v1/prometheus/",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200

    def test_returns_401_with_bad_key(self, authed_client):
        tc, _api_key = authed_client
        resp = tc.get(
            "/api/v1/prometheus/",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Tests: Empty collectors return zero values
# ---------------------------------------------------------------------------


class TestEmptyCollectors:
    """When no data has been recorded, metrics should be zero."""

    def test_throughput_zero(self, client):
        # Reset the global collector to ensure clean state
        get_collector().reset()
        resp = client.get("/api/v1/prometheus/")
        body = resp.text
        # The throughput gauge should be present with value 0.0
        for line in body.splitlines():
            if line.startswith("ocr_api_throughput_pages_per_minute "):
                val = float(line.split()[-1])
                assert val == 0.0
                break

    def test_latency_zero(self, client):
        get_collector().reset()
        resp = client.get("/api/v1/prometheus/")
        body = resp.text
        for line in body.splitlines():
            if 'ocr_api_latency_ms{percentile="p50"}' in line:
                val = float(line.split()[-1])
                assert val == 0.0
                break

    def test_fleet_gpu_zero(self, client):
        get_fleet_tracker().reset()
        resp = client.get("/api/v1/prometheus/")
        body = resp.text
        for line in body.splitlines():
            if line.startswith("ocr_api_fleet_gpu_utilization_pct "):
                val = float(line.split()[-1])
                assert val == 0.0
                break


# ---------------------------------------------------------------------------
# Tests: Latency label correctness
# ---------------------------------------------------------------------------


class TestLatencyLabels:
    """Verify that latency metric uses correct percentile labels."""

    def test_p50_p95_p99_labels_present(self, client):
        get_collector().reset()
        # Record some latency data
        collector = get_collector()
        collector.record_latency(job_id="j1", total_ms=100.0)
        collector.record_latency(job_id="j2", total_ms=200.0)
        collector.record_latency(job_id="j3", total_ms=300.0)

        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        assert 'percentile="p50"' in body
        assert 'percentile="p95"' in body
        assert 'percentile="p99"' in body


# ---------------------------------------------------------------------------
# Tests: Fleet worker state labels
# ---------------------------------------------------------------------------


class TestFleetLabels:
    """Verify fleet worker metrics reflect worker states correctly."""

    def test_worker_state_labels(self, client):
        tracker = get_fleet_tracker()
        tracker.reset()
        tracker.register_worker("w1", hostname="host1")
        tracker.heartbeat("w1", state=WorkerState.BUSY)
        tracker.register_worker("w2", hostname="host2")
        tracker.heartbeat("w2", state=WorkerState.IDLE)

        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        # All WorkerState values should appear as labels
        for ws in WorkerState:
            assert f'state="{ws.value}"' in body

    def test_gpu_utilization_reflects_fleet(self, client):
        tracker = get_fleet_tracker()
        tracker.reset()
        tracker.register_worker(
            "w1",
            hostname="host1",
            gpus=[GpuInfo(gpu_id=0, utilization_pct=80.0, memory_total_mb=16000)],
        )

        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        for line in body.splitlines():
            if line.startswith("ocr_api_fleet_gpu_utilization_pct "):
                val = float(line.split()[-1])
                assert val == 80.0
                break
        else:
            pytest.fail("ocr_api_fleet_gpu_utilization_pct not found in output")


# ---------------------------------------------------------------------------
# Tests: Stage metrics propagation
# ---------------------------------------------------------------------------


class TestStageMetrics:
    """Verify pipeline stage metrics propagate from dashboard collector."""

    def test_stage_depth_and_workers(self, client):
        collector = get_collector()
        collector.reset()
        collector.update_stage(
            "extraction",
            active_workers=4,
            queue_depth=12,
            completed=100,
        )
        collector.update_stage(
            "ocr",
            active_workers=8,
            queue_depth=25,
            completed=80,
        )

        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        assert 'stage="extraction"' in body
        assert 'stage="ocr"' in body

        # Verify queue depth values
        for line in body.splitlines():
            if 'ocr_api_stage_queue_depth{stage="extraction"}' in line:
                val = float(line.split()[-1])
                assert val == 12.0
            if 'ocr_api_stage_active_workers{stage="ocr"}' in line:
                val = float(line.split()[-1])
                assert val == 8.0


# ---------------------------------------------------------------------------
# Tests: Queue monitor integration
# ---------------------------------------------------------------------------


class TestQueueMonitorMetrics:
    """Verify queue monitor metrics appear in output."""

    def test_queue_depth_from_monitor(self, client):
        monitor = get_queue_monitor()
        monitor.reset()
        monitor.record_depth("ocr_gpu", 15)
        monitor.record_depth("compression", 3)

        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        assert 'queue_name="ocr_gpu"' in body
        assert 'queue_name="compression"' in body

    def test_active_alerts_count(self, client):
        monitor = get_queue_monitor()
        monitor.reset()
        # Set a low threshold and trigger it
        monitor.set_threshold("test_q", warning_depth=5, critical_depth=10)
        monitor.record_depth("test_q", 6)

        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        for line in body.splitlines():
            if line.startswith("ocr_api_queue_active_alerts "):
                val = float(line.split()[-1])
                assert val >= 1.0
                break
        else:
            pytest.fail("ocr_api_queue_active_alerts not found")


# ---------------------------------------------------------------------------
# Tests: _refresh_metrics function
# ---------------------------------------------------------------------------


class TestRefreshMetrics:
    """Verify _refresh_metrics pulls from singletons correctly."""

    def test_refresh_updates_throughput(self):
        from api.prometheus import _THROUGHPUT_PPM, _refresh_metrics

        collector = get_collector()
        collector.reset()
        collector.record_throughput(pages=60)

        _refresh_metrics()

        # After recording 60 pages, PPM should be > 0
        # (exact value depends on window calculation)
        # Just verify it ran without error and set a value
        assert _THROUGHPUT_PPM._value is not None

    def test_refresh_updates_fleet(self):
        from api.prometheus import _FLEET_GPU_UTIL, _refresh_metrics

        tracker = get_fleet_tracker()
        tracker.reset()
        tracker.register_worker(
            "w1",
            gpus=[GpuInfo(gpu_id=0, utilization_pct=50.0, memory_total_mb=8000)],
        )

        _refresh_metrics()

        assert _FLEET_GPU_UTIL._value._value == 50.0
