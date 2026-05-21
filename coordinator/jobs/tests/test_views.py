"""Tests for coordinator API views (Phase M5 production hardening).

Tests the /api/v1/metrics/ endpoint for correct JSON structure,
job/worker/page counts, and HTTP method enforcement.

Run with: cd coordinator && python -m pytest jobs/tests/test_views.py -v
"""

import json
from unittest.mock import patch

from django.test import Client, TestCase

from jobs.metrics_cache import _cache
from jobs.models import Job, PageResult, Worker


class TestMetricsEndpoint(TestCase):
    """Tests for the GET /api/v1/metrics/ endpoint."""

    def setUp(self):
        _cache.invalidate()
        self.client = Client()

    def test_metrics_returns_json(self):
        """GET /api/v1/metrics/ returns 200 with correct top-level JSON keys."""
        response = self.client.get("/api/v1/metrics/")
        assert response.status_code == 200
        assert response["Content-Type"] == "application/json"

        data = json.loads(response.content)
        assert "jobs" in data
        assert "workers" in data
        assert "pages" in data
        assert "timestamp" in data

        # Verify nested structure
        assert "by_status" in data["jobs"]
        assert "total" in data["jobs"]
        assert "error_rate_1h" in data["jobs"]
        assert "by_status" in data["workers"]
        assert "total" in data["workers"]
        assert "gpu_available" in data["workers"]
        assert "total_processed" in data["pages"]
        assert "avg_processing_time_ms" in data["pages"]

    def test_metrics_job_counts(self):
        """Job counts by status are correctly reflected in the response."""
        Job.objects.create(source_file="/a.pdf", status=Job.Status.SUBMITTED)
        Job.objects.create(source_file="/b.pdf", status=Job.Status.PROCESSING)
        Job.objects.create(source_file="/c.pdf", status=Job.Status.COMPLETED)
        Job.objects.create(source_file="/d.pdf", status=Job.Status.COMPLETED)
        Job.objects.create(source_file="/e.pdf", status=Job.Status.FAILED)

        response = self.client.get("/api/v1/metrics/")
        data = json.loads(response.content)

        by_status = data["jobs"]["by_status"]
        assert by_status["submitted"] == 1
        assert by_status["processing"] == 1
        assert by_status["completed"] == 2
        assert by_status["failed"] == 1
        assert data["jobs"]["total"] == 5

    def test_metrics_worker_counts(self):
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

        response = self.client.get("/api/v1/metrics/")
        data = json.loads(response.content)

        by_status = data["workers"]["by_status"]
        assert by_status["online"] == 1
        assert by_status["busy"] == 1
        assert by_status["offline"] == 1
        assert data["workers"]["total"] == 3
        assert data["workers"]["gpu_available"] == 2  # online + busy with GPU

    def test_metrics_page_stats(self):
        """Page processing stats include active status taxonomy."""
        job = Job.objects.create(source_file="/test.pdf", total_pages=4)
        PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", processing_time_ms=100,
        )
        PageResult.objects.create(
            job=job, document_id="d1", page_num=2,
            status="fallback", processing_time_ms=200,
        )
        PageResult.objects.create(
            job=job, document_id="d1", page_num=3,
            status="image_only", processing_time_ms=300,
        )
        PageResult.objects.create(
            job=job, document_id="d1", page_num=4,
            status="failed", processing_time_ms=50,
        )

        response = self.client.get("/api/v1/metrics/")
        data = json.loads(response.content)

        assert data["pages"]["total_processed"] == 3  # ok/fallback/image_only
        assert data["pages"]["avg_processing_time_ms"] == 200.0  # (100+200+300)/3

    def test_metrics_only_get(self):
        """POST to /api/v1/metrics/ returns 405 Method Not Allowed."""
        response = self.client.post("/api/v1/metrics/")
        assert response.status_code == 405

    @patch("jobs.metrics_auth.METRICS_API_KEY", "test-secret-key")
    def test_metrics_requires_api_key_when_configured(self):
        """When METRICS_API_KEY is set, requests without key get 401."""
        response = self.client.get("/api/v1/metrics/")
        assert response.status_code == 401
        data = json.loads(response.content)
        assert data["error"] == "Unauthorized"

    @patch("jobs.metrics_auth.METRICS_API_KEY", "test-secret-key")
    def test_metrics_accepts_valid_api_key(self):
        """When METRICS_API_KEY is set, requests with correct key succeed."""
        response = self.client.get(
            "/api/v1/metrics/",
            HTTP_X_API_KEY="test-secret-key",
        )
        assert response.status_code == 200

    @patch("jobs.metrics_auth.METRICS_API_KEY", "test-secret-key")
    def test_metrics_rejects_wrong_api_key(self):
        """When METRICS_API_KEY is set, requests with wrong key get 401."""
        response = self.client.get(
            "/api/v1/metrics/",
            HTTP_X_API_KEY="wrong-key",
        )
        assert response.status_code == 401

    @patch("jobs.metrics_auth.METRICS_API_KEY", "test-secret-key")
    def test_metrics_accepts_valid_bearer_token(self):
        """When METRICS_API_KEY is set, bearer token auth is accepted."""
        response = self.client.get(
            "/api/v1/metrics/",
            HTTP_AUTHORIZATION="Bearer test-secret-key",
        )
        assert response.status_code == 200

    @patch("jobs.metrics_auth.METRICS_API_KEY", "test-secret-key")
    def test_metrics_rejects_wrong_bearer_token(self):
        """When METRICS_API_KEY is set, wrong bearer token gets 401."""
        response = self.client.get(
            "/api/v1/metrics/",
            HTTP_AUTHORIZATION="Bearer wrong-key",
        )
        assert response.status_code == 401

    @patch("jobs.metrics_auth.METRICS_API_KEY", "test-secret-key")
    def test_metrics_accepts_if_any_provided_key_matches(self):
        """If both auth headers are provided, a matching one is sufficient."""
        response = self.client.get(
            "/api/v1/metrics/",
            HTTP_X_API_KEY="wrong-key",
            HTTP_AUTHORIZATION="Bearer test-secret-key",
        )
        assert response.status_code == 200

    def test_metrics_error_rate(self):
        """Error rate is computed from recent jobs (last 1 hour)."""
        # Create 4 recent jobs: 1 failed, 3 completed
        Job.objects.create(source_file="/ok1.pdf", status=Job.Status.COMPLETED)
        Job.objects.create(source_file="/ok2.pdf", status=Job.Status.COMPLETED)
        Job.objects.create(source_file="/ok3.pdf", status=Job.Status.COMPLETED)
        Job.objects.create(source_file="/fail.pdf", status=Job.Status.FAILED)

        response = self.client.get("/api/v1/metrics/")
        data = json.loads(response.content)

        # 1 out of 4 recent jobs failed -> 0.25
        assert data["jobs"]["error_rate_1h"] == 0.25
