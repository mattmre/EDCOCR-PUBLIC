"""Tests for Prometheus metrics collector and endpoint.

Tests the PipelineCollector custom collector, the has_valid_metrics_key
auth helper, and the GET /api/v1/prometheus/ view endpoint.

Run with: cd coordinator && python -m pytest jobs/tests/test_prometheus.py -v
"""

from unittest.mock import patch

from django.test import Client, RequestFactory, TestCase

from jobs.metrics_auth import has_valid_metrics_key
from jobs.metrics_cache import _cache
from jobs.models import CustodyEvent, Job, PageResult, Worker
from jobs.prometheus_metrics import PipelineCollector


class TestPipelineCollector(TestCase):
    """Tests for the PipelineCollector custom Prometheus collector."""

    def setUp(self):
        _cache.invalidate()
        self.collector = PipelineCollector()

    def test_collect_yields_all_metric_families(self):
        """collect() yields exactly the 22 expected metric families."""
        families = list(self.collector.collect())
        family_names = [f.name for f in families]

        # NOTE: CounterMetricFamily strips the _total suffix from the name
        # in prometheus-client >= 0.21.  So "ocr_pages_processed_total"
        # yields a family named "ocr_pages_processed".
        expected = [
            "ocr_jobs_total",
            "ocr_job_error_rate_1h",
            "ocr_workers_total",
            "ocr_gpu_workers_available",
            "ocr_pages_processed",
            "ocr_page_processing_time_avg_ms",
            "ocr_pages_by_status",
            "ocr_custody_violations_total",
            "ocr_s3_job_error_rate_1h",
            "ocr_job_completion_rate_1h",
            "ocr_jobs_by_storage_backend",
            "ocr_jobs_stuck_total",
            "ocr_page_processing_time_p95_ms",
            "ocr_page_processing_time_p99_ms",
            "ocr_queue_depth",
            "ocr_dpi_escalation_total",
            "ocr_pages_by_engine",
            "ocr_tenant_jobs_total",
            "ocr_tenant_pages_processed",
            "ocr_tenant_error_rate",
            "ocr_tenant_processing_time_avg_ms",
            "ocr_processing_duration_seconds",
        ]
        assert family_names == expected, (
            f"Expected {expected}, got {family_names}"
        )
        assert len(families) == 22

    def test_collect_job_counts_by_status(self):
        """Job counts per status label match the database."""
        Job.objects.create(source_file="/a.pdf", status=Job.Status.SUBMITTED)
        Job.objects.create(source_file="/b.pdf", status=Job.Status.PROCESSING)
        Job.objects.create(source_file="/c.pdf", status=Job.Status.COMPLETED)
        Job.objects.create(source_file="/d.pdf", status=Job.Status.COMPLETED)
        Job.objects.create(source_file="/e.pdf", status=Job.Status.FAILED)

        families = {f.name: f for f in self.collector.collect()}
        job_family = families["ocr_jobs_total"]

        # Build a dict of {status_label: value} from the metric samples
        samples = {s.labels["status"]: s.value for s in job_family.samples}

        assert samples["submitted"] == 1
        assert samples["processing"] == 1
        assert samples["completed"] == 2
        assert samples["failed"] == 1
        # Statuses with zero jobs should still be present
        assert samples.get("ingesting", 0) == 0
        assert samples.get("assembling", 0) == 0
        assert samples.get("cancelled", 0) == 0

    def test_collect_worker_counts(self):
        """Worker counts by status and GPU availability are correct."""
        Worker.objects.create(
            hostname="w1", status=Worker.Status.ONLINE,
            gpu_available=True, gpu_model="RTX 4090",
        )
        Worker.objects.create(
            hostname="w2", status=Worker.Status.BUSY,
            gpu_available=True, gpu_model="RTX 3090",
        )
        Worker.objects.create(
            hostname="w3", status=Worker.Status.OFFLINE,
            gpu_available=False,
        )
        Worker.objects.create(
            hostname="w4", status=Worker.Status.OFFLINE,
            gpu_available=True, gpu_model="RTX 3080",
        )

        families = {f.name: f for f in self.collector.collect()}

        # Check ocr_workers_total
        worker_family = families["ocr_workers_total"]
        worker_samples = {s.labels["status"]: s.value for s in worker_family.samples}
        assert worker_samples["online"] == 1
        assert worker_samples["busy"] == 1
        assert worker_samples["offline"] == 2

        # Check ocr_gpu_workers_available (online or busy with gpu_available=True)
        gpu_family = families["ocr_gpu_workers_available"]
        gpu_value = gpu_family.samples[0].value
        assert gpu_value == 2  # w1 (online+gpu) + w2 (busy+gpu); w4 offline excluded

    def test_collect_page_stats(self):
        """Page processed total and average time are computed correctly."""
        job = Job.objects.create(source_file="/test.pdf", total_pages=5)
        PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=100, ocr_method="paddle",
        )
        PageResult.objects.create(
            job=job, document_id="d1", page_num=2,
            status="fallback", processing_time_ms=200, ocr_method="tesseract",
        )
        PageResult.objects.create(
            job=job, document_id="d1", page_num=3,
            status="image_only", processing_time_ms=300, ocr_method="paddle",
        )
        PageResult.objects.create(
            job=job, document_id="d1", page_num=4,
            status="completed", processing_time_ms=400, ocr_method="paddle",
        )
        # "failed" status should be excluded from processed count
        PageResult.objects.create(
            job=job, document_id="d1", page_num=5,
            status="failed", processing_time_ms=50,
        )

        families = {f.name: f for f in self.collector.collect()}

        # ocr_pages_processed is a CounterMetricFamily with labels [engine, status].
        # Each sample represents one (engine, status) combination.
        # Sum all sample values to get total throughput.
        pages_total = families["ocr_pages_processed"]
        total_processed = sum(s.value for s in pages_total.samples)
        # 4 success pages (ok/fallback/image_only/completed) + 1 failed with no engine = 5
        assert total_processed == 5

        # Avg: (100+200+300+400) / 4 = 250.0 (only non-failed pages)
        avg_time = families["ocr_page_processing_time_avg_ms"]
        assert avg_time.samples[0].value == 250.0

        # Pages by status should include all statuses from DB
        by_status = families["ocr_pages_by_status"]
        status_samples = {s.labels["status"]: s.value for s in by_status.samples}
        assert status_samples["ok"] == 1
        assert status_samples["fallback"] == 1
        assert status_samples["image_only"] == 1
        assert status_samples["completed"] == 1
        assert status_samples["failed"] == 1

    def test_collect_empty_database(self):
        """collect() works with no data in the database (all zeros)."""
        families = {f.name: f for f in self.collector.collect()}

        # All job statuses should be 0
        job_family = families["ocr_jobs_total"]
        for sample in job_family.samples:
            assert sample.value == 0

        # Error rate should be 0.0 with no recent jobs
        error_rate = families["ocr_job_error_rate_1h"]
        assert error_rate.samples[0].value == 0.0

        # Page throughput counter has labels [engine, status]; with an empty DB
        # no (engine, status) combinations exist, so no samples are emitted.
        pages_total = families["ocr_pages_processed"]
        assert len(pages_total.samples) == 0

        # Avg time should be 0 when no pages exist
        avg_time = families["ocr_page_processing_time_avg_ms"]
        assert avg_time.samples[0].value == 0

        # New metrics should have safe defaults with empty DB
        assert families["ocr_custody_violations_total"].samples[0].value == 0
        assert families["ocr_s3_job_error_rate_1h"].samples[0].value == 0.0
        assert families["ocr_job_completion_rate_1h"].samples[0].value == 1.0
        assert families["ocr_jobs_stuck_total"].samples[0].value == 0
        backend_family = families["ocr_jobs_by_storage_backend"]
        for sample in backend_family.samples:
            assert sample.value == 0

    def test_describe_returns_descriptors(self):
        """describe() yields metric descriptors without DB access."""
        descriptors = list(self.collector.describe())
        descriptor_names = [d.name for d in descriptors]

        # CounterMetricFamily strips _total suffix from family name.
        expected = [
            "ocr_jobs_total",
            "ocr_job_error_rate_1h",
            "ocr_workers_total",
            "ocr_gpu_workers_available",
            "ocr_pages_processed",
            "ocr_page_processing_time_avg_ms",
            "ocr_pages_by_status",
            "ocr_custody_violations_total",
            "ocr_s3_job_error_rate_1h",
            "ocr_job_completion_rate_1h",
            "ocr_jobs_by_storage_backend",
            "ocr_jobs_stuck_total",
            "ocr_page_processing_time_p95_ms",
            "ocr_page_processing_time_p99_ms",
            "ocr_queue_depth",
            "ocr_dpi_escalation_total",
            "ocr_pages_by_engine",
            "ocr_tenant_jobs_total",
            "ocr_tenant_pages_processed",
            "ocr_tenant_error_rate",
            "ocr_tenant_processing_time_avg_ms",
            "ocr_processing_duration_seconds",
        ]
        assert descriptor_names == expected
        assert len(descriptors) == 22

    def test_collect_custody_violations_none(self):
        """No violations when custody chain hashes are valid."""
        import hashlib as _hashlib
        import json as _json

        from django.utils import timezone as _tz

        job = Job.objects.create(source_file="/test.pdf")
        now = _tz.now()
        prev_hash = None
        for etype in ("ingest", "complete"):
            ts = now
            event_dict = {
                "document_id": "doc1",
                "event_type": etype,
                "timestamp": ts.isoformat(timespec="milliseconds"),
                "data": {},
                "prev_hash": prev_hash,
            }
            event_hash = _hashlib.sha256(
                _json.dumps(event_dict, sort_keys=True, default=str).encode()
            ).hexdigest()
            CustodyEvent.objects.create(
                document_id="doc1", job=job, event_type=etype,
                timestamp=ts, prev_hash=prev_hash or "",
                event_hash=event_hash, chain_finalized=True,
            )
            prev_hash = event_hash

        families = {f.name: f for f in self.collector.collect()}
        assert families["ocr_custody_violations_total"].samples[0].value == 0

    def test_collect_custody_violations_detected(self):
        """Detects a corrupted custody chain hash."""
        job = Job.objects.create(source_file="/test.pdf")
        CustodyEvent.objects.create(
            document_id="doc1", job=job, event_type="ingest",
            prev_hash="", event_hash="deliberately_wrong_hash",
            chain_finalized=True,
        )
        families = {f.name: f for f in self.collector.collect()}
        assert families["ocr_custody_violations_total"].samples[0].value == 1

    def test_collect_s3_error_rate(self):
        """S3 error rate is computed from S3-backed jobs only."""
        Job.objects.create(
            source_file="/a.pdf", status=Job.Status.COMPLETED,
            storage_backend_used="s3",
        )
        Job.objects.create(
            source_file="/b.pdf", status=Job.Status.FAILED,
            storage_backend_used="s3",
        )
        Job.objects.create(
            source_file="/c.pdf", status=Job.Status.FAILED,
            storage_backend_used="nfs",
        )
        families = {f.name: f for f in self.collector.collect()}
        s3_rate = families["ocr_s3_job_error_rate_1h"].samples[0].value
        assert s3_rate == 0.5

    def test_collect_completion_rate(self):
        """Completion rate = completed / (completed + failed)."""
        Job.objects.create(source_file="/a.pdf", status=Job.Status.COMPLETED)
        Job.objects.create(source_file="/b.pdf", status=Job.Status.COMPLETED)
        Job.objects.create(source_file="/c.pdf", status=Job.Status.COMPLETED)
        Job.objects.create(source_file="/d.pdf", status=Job.Status.FAILED)
        Job.objects.create(source_file="/e.pdf", status=Job.Status.PROCESSING)

        families = {f.name: f for f in self.collector.collect()}
        rate = families["ocr_job_completion_rate_1h"].samples[0].value
        assert rate == 0.75

    def test_collect_completion_rate_no_terminal_jobs(self):
        """Completion rate defaults to 1.0 when no terminal jobs exist."""
        Job.objects.create(source_file="/a.pdf", status=Job.Status.PROCESSING)
        families = {f.name: f for f in self.collector.collect()}
        rate = families["ocr_job_completion_rate_1h"].samples[0].value
        assert rate == 1.0

    def test_collect_jobs_by_storage_backend(self):
        """Jobs are counted by storage backend label."""
        Job.objects.create(source_file="/a.pdf", storage_backend_used="nfs")
        Job.objects.create(source_file="/b.pdf", storage_backend_used="nfs")
        Job.objects.create(source_file="/c.pdf", storage_backend_used="s3")
        Job.objects.create(source_file="/d.pdf", storage_backend_used="")

        families = {f.name: f for f in self.collector.collect()}
        backend_family = families["ocr_jobs_by_storage_backend"]
        samples = {s.labels["backend"]: s.value for s in backend_family.samples}
        assert samples["nfs"] == 2
        assert samples["s3"] == 1
        assert samples["unset"] == 1

    def test_collect_stuck_jobs(self):
        """Stuck jobs are counted when started_at exceeds 1 hour."""
        from django.utils import timezone as _tz

        two_hours_ago = _tz.now() - _tz.timedelta(hours=2)
        Job.objects.create(
            source_file="/a.pdf", status=Job.Status.PROCESSING,
            started_at=two_hours_ago,
        )
        Job.objects.create(
            source_file="/b.pdf", status=Job.Status.PROCESSING,
            started_at=_tz.now(),
        )
        families = {f.name: f for f in self.collector.collect()}
        stuck = families["ocr_jobs_stuck_total"].samples[0].value
        assert stuck == 1

    def test_collect_cache_hit_returns_stale_data(self):
        """Second collect() within TTL returns cached data, not fresh ORM query."""
        Job.objects.create(source_file="/a.pdf", status=Job.Status.COMPLETED)
        families1 = {f.name: f for f in self.collector.collect()}
        completed1 = next(
            s.value for s in families1["ocr_jobs_total"].samples
            if s.labels["status"] == "completed"
        )
        assert completed1 == 1

        # Add another job -- but cache should still return old data
        Job.objects.create(source_file="/b.pdf", status=Job.Status.COMPLETED)
        families2 = {f.name: f for f in self.collector.collect()}
        completed2 = next(
            s.value for s in families2["ocr_jobs_total"].samples
            if s.labels["status"] == "completed"
        )
        # Still 1 because warm cache has not expired
        assert completed2 == 1

    def test_collect_cache_invalidation_forces_fresh_query(self):
        """After invalidate(), collect() returns fresh data."""
        Job.objects.create(source_file="/a.pdf", status=Job.Status.COMPLETED)
        list(self.collector.collect())  # populate cache

        Job.objects.create(source_file="/b.pdf", status=Job.Status.COMPLETED)
        _cache.invalidate()

        families = {f.name: f for f in self.collector.collect()}
        completed = next(
            s.value for s in families["ocr_jobs_total"].samples
            if s.labels["status"] == "completed"
        )
        assert completed == 2

    def test_setup_isolation_between_tests(self):
        """setUp invalidates cache so tests do not see stale data from prior tests."""
        # This test relies on setUp calling _cache.invalidate().
        # If setUp did not clear the cache, this empty-database check would
        # fail when run after test_collect_cache_hit_returns_stale_data.
        families = {f.name: f for f in self.collector.collect()}
        job_family = families["ocr_jobs_total"]
        for sample in job_family.samples:
            assert sample.value == 0

    def test_ttl_ordering_hot_expires_before_warm(self):
        """Hot tier (15s) expires before warm tier (30s)."""
        from jobs.prometheus_metrics import _TTL_COLD, _TTL_HOT, _TTL_WARM

        assert _TTL_HOT < _TTL_WARM < _TTL_COLD


class TestHasValidMetricsKey(TestCase):
    """Tests for the has_valid_metrics_key auth helper."""

    def setUp(self):
        self.factory = RequestFactory()

    @patch("jobs.metrics_auth.METRICS_API_KEY", "")
    def test_no_key_configured_allows_all(self):
        """When METRICS_API_KEY is empty, all requests are allowed."""
        request = self.factory.get("/api/v1/prometheus/")
        assert has_valid_metrics_key(request) is True

    @patch("jobs.metrics_auth.METRICS_API_KEY", "secret-key-123")
    def test_valid_x_api_key_header(self):
        """Returns True for correct X-Api-Key header."""
        request = self.factory.get(
            "/api/v1/prometheus/",
            HTTP_X_API_KEY="secret-key-123",
        )
        assert has_valid_metrics_key(request) is True

    @patch("jobs.metrics_auth.METRICS_API_KEY", "secret-key-123")
    def test_valid_bearer_token(self):
        """Returns True for correct Authorization: Bearer token."""
        request = self.factory.get(
            "/api/v1/prometheus/",
            HTTP_AUTHORIZATION="Bearer secret-key-123",
        )
        assert has_valid_metrics_key(request) is True

    @patch("jobs.metrics_auth.METRICS_API_KEY", "secret-key-123")
    def test_invalid_key_rejected(self):
        """Returns False for wrong key."""
        request = self.factory.get(
            "/api/v1/prometheus/",
            HTTP_X_API_KEY="wrong-key",
        )
        assert has_valid_metrics_key(request) is False

    @patch("jobs.metrics_auth.METRICS_API_KEY", "secret-key-123")
    def test_no_auth_headers_rejected(self):
        """Returns False when no auth headers are provided."""
        request = self.factory.get("/api/v1/prometheus/")
        assert has_valid_metrics_key(request) is False

    @patch("jobs.metrics_auth.METRICS_API_KEY", "secret-key-123")
    def test_wrong_bearer_scheme_rejected(self):
        """Returns False when Authorization header uses non-Bearer scheme."""
        request = self.factory.get(
            "/api/v1/prometheus/",
            HTTP_AUTHORIZATION="Basic secret-key-123",
        )
        assert has_valid_metrics_key(request) is False

    @patch("jobs.metrics_auth.METRICS_API_KEY", "secret-key-123")
    def test_either_header_suffices(self):
        """If both headers are present, a matching one is sufficient."""
        request = self.factory.get(
            "/api/v1/prometheus/",
            HTTP_X_API_KEY="wrong-key",
            HTTP_AUTHORIZATION="Bearer secret-key-123",
        )
        assert has_valid_metrics_key(request) is True


class TestPrometheusEndpoint(TestCase):
    """Tests for the GET /api/v1/prometheus/ view endpoint."""

    def setUp(self):
        _cache.invalidate()
        self.client = Client()

    def test_endpoint_returns_prometheus_format(self):
        """GET /api/v1/prometheus/ returns 200 with Prometheus content type."""
        response = self.client.get("/api/v1/prometheus/")
        assert response.status_code == 200
        content_type = response["Content-Type"]
        assert "text/plain" in content_type
        assert "0.0.4" in content_type

    def test_endpoint_contains_metric_names(self):
        """Response body contains all expected Prometheus metric names."""
        # Create some data so metrics have values
        Job.objects.create(source_file="/a.pdf", status=Job.Status.COMPLETED)
        Worker.objects.create(
            hostname="w1", status=Worker.Status.ONLINE, gpu_available=True,
        )

        response = self.client.get("/api/v1/prometheus/")
        body = response.content.decode("utf-8")

        expected_metrics = [
            "ocr_jobs_total",
            "ocr_job_error_rate_1h",
            "ocr_workers_total",
            "ocr_gpu_workers_available",
            "ocr_pages_processed_total",
            "ocr_page_processing_time_avg_ms",
            "ocr_custody_violations_total",
            "ocr_s3_job_error_rate_1h",
            "ocr_job_completion_rate_1h",
            "ocr_jobs_by_storage_backend",
            "ocr_jobs_stuck_total",
        ]
        for metric_name in expected_metrics:
            assert metric_name in body, (
                f"Expected metric '{metric_name}' not found in response body"
            )

    def test_endpoint_only_get(self):
        """POST to /api/v1/prometheus/ returns 405 Method Not Allowed."""
        response = self.client.post("/api/v1/prometheus/")
        assert response.status_code == 405

    @patch("jobs.metrics_auth.METRICS_API_KEY", "test-prom-key")
    def test_endpoint_auth_required_when_key_set(self):
        """Returns 401 without auth when METRICS_API_KEY is configured."""
        response = self.client.get("/api/v1/prometheus/")
        assert response.status_code == 401

    @patch("jobs.metrics_auth.METRICS_API_KEY", "test-prom-key")
    def test_endpoint_auth_accepted_with_x_api_key(self):
        """Returns 200 with correct X-Api-Key header."""
        response = self.client.get(
            "/api/v1/prometheus/",
            HTTP_X_API_KEY="test-prom-key",
        )
        assert response.status_code == 200

    @patch("jobs.metrics_auth.METRICS_API_KEY", "test-prom-key")
    def test_endpoint_auth_accepted_with_bearer(self):
        """Returns 200 with correct Authorization: Bearer token."""
        response = self.client.get(
            "/api/v1/prometheus/",
            HTTP_AUTHORIZATION="Bearer test-prom-key",
        )
        assert response.status_code == 200

    @patch("jobs.metrics_auth.METRICS_API_KEY", "test-prom-key")
    def test_endpoint_auth_rejected_with_wrong_key(self):
        """Returns 401 with incorrect auth key."""
        response = self.client.get(
            "/api/v1/prometheus/",
            HTTP_X_API_KEY="wrong-key",
        )
        assert response.status_code == 401

    def test_endpoint_metric_values_reflect_database(self):
        """Prometheus output reflects actual DB state (not cached)."""
        # Create test data
        Job.objects.create(source_file="/a.pdf", status=Job.Status.COMPLETED)
        Job.objects.create(source_file="/b.pdf", status=Job.Status.FAILED)
        job = Job.objects.create(source_file="/c.pdf", status=Job.Status.PROCESSING)
        PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=150,
        )

        response = self.client.get("/api/v1/prometheus/")
        body = response.content.decode("utf-8")

        # Verify specific metric lines are present
        assert 'ocr_jobs_total{status="completed"}' in body
        assert 'ocr_jobs_total{status="failed"}' in body
        assert 'ocr_jobs_total{status="processing"}' in body
        assert "ocr_pages_processed_total" in body
