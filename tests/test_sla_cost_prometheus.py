"""Tests for SLA monitoring and cost tracking Prometheus bridges.

Covers:
- SLA metrics appear in prometheus output after recording data
- Cost metrics appear after tracking usage
- Per-tenant label correctness
- Graceful degradation when modules fail to import
- _refresh_metrics propagates SLA and cost values correctly

Run with: python -m pytest tests/test_sla_cost_prometheus.py -v
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset SLA and cost singletons and clear Prometheus gauge label children."""
    from api.prometheus import (
        _COST_ESTIMATE_TOTAL,
        _SLA_AVAILABILITY_PCT,
        _SLA_COMPLIANCE_PCT,
        _SLA_P95_LATENCY_SECONDS,
        _SLA_VIOLATION_COUNT,
        _TENANT_GPU_SECONDS,
        _TENANT_STORAGE_BYTES,
    )
    from cost_tracking import reset_global_tracker
    from sla_monitoring import reset_global_monitor

    reset_global_monitor()
    reset_global_tracker()

    # Clear stale label children from prior tests so "no data" tests
    # don't see leftover labeled samples in the Prometheus registry.
    for gauge in (
        _SLA_COMPLIANCE_PCT,
        _SLA_VIOLATION_COUNT,
        _SLA_AVAILABILITY_PCT,
        _SLA_P95_LATENCY_SECONDS,
        _COST_ESTIMATE_TOTAL,
        _TENANT_GPU_SECONDS,
        _TENANT_STORAGE_BYTES,
    ):
        gauge._metrics.clear()

    yield
    reset_global_monitor()
    reset_global_tracker()


@pytest.fixture()
def client(tmp_path):
    """FastAPI TestClient with auth disabled."""
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_metric_value(body: str, metric_name: str, labels: dict | None = None) -> float | None:
    """Extract a metric value from Prometheus text output.

    Returns the float value if found, None otherwise.
    """
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        if metric_name not in line:
            continue
        # Check labels if provided
        if labels:
            match = True
            for k, v in labels.items():
                if f'{k}="{v}"' not in line:
                    match = False
                    break
            if not match:
                continue
        # Extract the value (last token on the line)
        parts = line.split()
        if parts:
            try:
                return float(parts[-1])
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Tests: SLA metrics in Prometheus output
# ---------------------------------------------------------------------------


class TestSLAPrometheusMetrics:
    """SLA monitoring metrics appear in Prometheus text output."""

    def test_sla_compliance_after_recording(self, client):
        """After recording SLA data, compliance metric should appear."""
        from sla_monitoring import get_monitor

        monitor = get_monitor()
        monitor.record_request("tenant-a", success=True, latency_seconds=1.0)
        monitor.record_request("tenant-a", success=True, latency_seconds=2.0)

        resp = client.get("/api/v1/prometheus/")
        assert resp.status_code == 200
        body = resp.text

        val = _get_metric_value(body, "ocr_sla_compliance_pct", {"tenant_id": "tenant-a"})
        assert val is not None
        assert val == 100.0  # all requests succeeded, all SLOs met

    def test_sla_violation_count(self, client):
        """SLA violations should be reported when SLOs are breached."""
        from sla_monitoring import SLODefinition, get_monitor

        monitor = get_monitor()
        # Set a very strict throughput SLO
        monitor.set_tenant_slos(
            "tenant-strict",
            [
                SLODefinition(
                    "Throughput",
                    "throughput",
                    1000.0,  # impossible target
                    "pages_per_minute",
                    "gte",
                ),
            ],
        )
        monitor.record_throughput("tenant-strict", 5.0)  # way below target

        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        val = _get_metric_value(body, "ocr_sla_violation_count", {"tenant_id": "tenant-strict"})
        assert val is not None
        assert val >= 1.0

    def test_sla_availability_metric(self, client):
        """Availability percentage should reflect success/failure ratio."""
        from sla_monitoring import get_monitor

        monitor = get_monitor()
        # 3 successes, 1 failure = 75% availability
        monitor.record_request("tenant-b", success=True, latency_seconds=1.0)
        monitor.record_request("tenant-b", success=True, latency_seconds=1.0)
        monitor.record_request("tenant-b", success=True, latency_seconds=1.0)
        monitor.record_request("tenant-b", success=False, latency_seconds=5.0)

        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        val = _get_metric_value(body, "ocr_sla_availability_pct", {"tenant_id": "tenant-b"})
        assert val is not None
        assert val == 75.0

    def test_sla_p95_latency_metric(self, client):
        """P95 latency should be reported in seconds."""
        from sla_monitoring import get_monitor

        monitor = get_monitor()
        for i in range(20):
            monitor.record_request("tenant-c", success=True, latency_seconds=float(i))

        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        val = _get_metric_value(body, "ocr_sla_p95_latency_seconds", {"tenant_id": "tenant-c"})
        assert val is not None
        assert val > 0.0  # should be close to 19s

    def test_sla_multiple_tenants(self, client):
        """Multiple tenants produce separate label sets."""
        from sla_monitoring import get_monitor

        monitor = get_monitor()
        monitor.record_request("alpha", success=True, latency_seconds=1.0)
        monitor.record_request("beta", success=False, latency_seconds=2.0)

        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        alpha_compliance = _get_metric_value(
            body, "ocr_sla_compliance_pct", {"tenant_id": "alpha"}
        )
        beta_compliance = _get_metric_value(
            body, "ocr_sla_compliance_pct", {"tenant_id": "beta"}
        )
        assert alpha_compliance is not None
        assert beta_compliance is not None

    def test_no_sla_data_no_labels(self, client):
        """When no SLA data is recorded, no tenant labels should appear."""
        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        # The HELP/TYPE lines should still appear but no labeled samples
        for line in body.splitlines():
            if line.startswith("#"):
                continue
            assert "ocr_sla_compliance_pct" not in line


# ---------------------------------------------------------------------------
# Tests: Cost metrics in Prometheus output
# ---------------------------------------------------------------------------


class TestCostPrometheusMetrics:
    """Cost tracking metrics appear in Prometheus text output."""

    def test_cost_estimate_after_tracking(self, client):
        """After recording cost data, the total cost metric should appear."""
        from cost_tracking import get_tracker

        tracker = get_tracker()
        tracker.record_pages("tenant-x", 100)
        tracker.record_gpu_time("tenant-x", 60.0)
        tracker.record_storage("tenant-x", 1024 * 1024 * 1024)  # 1 GB

        resp = client.get("/api/v1/prometheus/")
        assert resp.status_code == 200
        body = resp.text

        val = _get_metric_value(body, "ocr_cost_estimate_total", {"tenant_id": "tenant-x"})
        assert val is not None
        assert val > 0.0

    def test_gpu_seconds_metric(self, client):
        """GPU seconds should reflect recorded GPU time."""
        from cost_tracking import get_tracker

        tracker = get_tracker()
        tracker.record_gpu_time("tenant-y", 123.5)

        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        val = _get_metric_value(body, "ocr_tenant_gpu_seconds", {"tenant_id": "tenant-y"})
        assert val is not None
        assert val == 123.5

    def test_storage_bytes_metric(self, client):
        """Storage bytes should reflect recorded storage usage."""
        from cost_tracking import get_tracker

        tracker = get_tracker()
        tracker.record_storage("tenant-z", 5_000_000)

        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        val = _get_metric_value(body, "ocr_tenant_storage_bytes", {"tenant_id": "tenant-z"})
        assert val is not None
        assert val == 5_000_000.0

    def test_cost_multiple_tenants(self, client):
        """Multiple tenants produce separate cost label sets."""
        from cost_tracking import get_tracker

        tracker = get_tracker()
        tracker.record_pages("t1", 10)
        tracker.record_pages("t2", 20)

        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        t1_cost = _get_metric_value(body, "ocr_cost_estimate_total", {"tenant_id": "t1"})
        t2_cost = _get_metric_value(body, "ocr_cost_estimate_total", {"tenant_id": "t2"})
        assert t1_cost is not None
        assert t2_cost is not None
        assert t2_cost > t1_cost  # t2 processed more pages

    def test_no_cost_data_no_labels(self, client):
        """When no cost data is recorded, no tenant labels should appear."""
        resp = client.get("/api/v1/prometheus/")
        body = resp.text

        for line in body.splitlines():
            if line.startswith("#"):
                continue
            assert "ocr_cost_estimate_total{" not in line


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """When SLA/cost modules are broken, other metrics still work."""

    def test_sla_import_failure_does_not_break_endpoint(self, client):
        """If sla_monitoring import fails, endpoint still returns 200."""
        with patch(
            "api.prometheus.get_monitor",
            side_effect=ImportError("mock import failure"),
            create=True,
        ):
            # Force the import path to fail by patching the module-level import
            import builtins

            original_import = builtins.__import__

            def failing_import(name, *args, **kwargs):
                if name == "sla_monitoring":
                    raise ImportError("mocked")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=failing_import):
                resp = client.get("/api/v1/prometheus/")

            assert resp.status_code == 200
            # Other metrics should still be present
            assert "ocr_api_throughput_pages_per_minute" in resp.text

    def test_cost_import_failure_does_not_break_endpoint(self, client):
        """If cost_tracking import fails, endpoint still returns 200."""
        import builtins

        original_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "cost_tracking":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            resp = client.get("/api/v1/prometheus/")

        assert resp.status_code == 200
        assert "ocr_api_throughput_pages_per_minute" in resp.text


# ---------------------------------------------------------------------------
# Tests: _refresh_metrics direct invocation
# ---------------------------------------------------------------------------


class TestRefreshMetricsSLACost:
    """Verify _refresh_metrics correctly populates SLA and cost gauges."""

    def test_refresh_populates_sla_gauge(self):
        from api.prometheus import _SLA_COMPLIANCE_PCT, _refresh_metrics
        from sla_monitoring import get_monitor

        monitor = get_monitor()
        monitor.record_request("direct-test", success=True, latency_seconds=1.0)

        _refresh_metrics()

        # The labeled child should have a value
        val = _SLA_COMPLIANCE_PCT.labels(tenant_id="direct-test")._value._value
        assert val == 100.0

    def test_refresh_populates_cost_gauge(self):
        from api.prometheus import _COST_ESTIMATE_TOTAL, _refresh_metrics
        from cost_tracking import get_tracker

        tracker = get_tracker()
        tracker.record_pages("cost-test", 50)

        _refresh_metrics()

        val = _COST_ESTIMATE_TOTAL.labels(tenant_id="cost-test")._value._value
        assert val > 0.0

    def test_refresh_populates_gpu_seconds_gauge(self):
        from api.prometheus import _TENANT_GPU_SECONDS, _refresh_metrics
        from cost_tracking import get_tracker

        tracker = get_tracker()
        tracker.record_gpu_time("gpu-test", 42.0)

        _refresh_metrics()

        val = _TENANT_GPU_SECONDS.labels(tenant_id="gpu-test")._value._value
        assert val == 42.0

    def test_refresh_populates_storage_bytes_gauge(self):
        from api.prometheus import _TENANT_STORAGE_BYTES, _refresh_metrics
        from cost_tracking import get_tracker

        tracker = get_tracker()
        tracker.record_storage("storage-test", 99999)

        _refresh_metrics()

        val = _TENANT_STORAGE_BYTES.labels(tenant_id="storage-test")._value._value
        assert val == 99999.0

    def test_refresh_populates_sla_violation_gauge(self):
        from api.prometheus import _SLA_VIOLATION_COUNT, _refresh_metrics
        from sla_monitoring import SLODefinition, get_monitor

        monitor = get_monitor()
        # Set impossible SLOs to force violations
        monitor.set_tenant_slos(
            "viol-test",
            [
                SLODefinition("Avail", "availability", 99.9, "percent", "gte"),
                SLODefinition("Throughput", "throughput", 9999.0, "pages_per_minute", "gte"),
            ],
        )
        monitor.record_request("viol-test", success=False, latency_seconds=100.0)
        monitor.record_throughput("viol-test", 1.0)

        _refresh_metrics()

        val = _SLA_VIOLATION_COUNT.labels(tenant_id="viol-test")._value._value
        assert val >= 1.0
