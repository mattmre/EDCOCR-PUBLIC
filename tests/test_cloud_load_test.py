"""Tests for scripts/cloud_load_test.py — load testing harness."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from scripts.cloud_load_test import (
    CloudLoadTester,
    JobResult,
    _create_synthetic_pdf,
    main,
)

# ---------------------------------------------------------------------------
# Synthetic PDF generation
# ---------------------------------------------------------------------------


class TestSyntheticPdf:
    """Tests for synthetic PDF generation."""

    def test_creates_valid_pdf_bytes(self):
        pdf = _create_synthetic_pdf()
        assert isinstance(pdf, bytes)
        assert pdf.startswith(b"%PDF-1.0")
        assert pdf.endswith(b"%%EOF")

    def test_page_count_param_accepted(self):
        pdf = _create_synthetic_pdf(page_count=5)
        assert isinstance(pdf, bytes)
        assert len(pdf) > 0


# ---------------------------------------------------------------------------
# JobResult
# ---------------------------------------------------------------------------


class TestJobResult:
    """Tests for JobResult data class."""

    def test_default_values(self):
        r = JobResult()
        assert r.job_id == ""
        assert r.status == "pending"
        assert r.submit_time == 0.0
        assert r.response_time == 0.0
        assert r.http_status == 0
        assert r.error is None

    def test_slots_defined(self):
        r = JobResult()
        assert hasattr(r, "__slots__")
        with pytest.raises(AttributeError):
            r.nonexistent_attr = "fail"


# ---------------------------------------------------------------------------
# CloudLoadTester — unit tests (no network)
# ---------------------------------------------------------------------------


class TestCloudLoadTester:
    """Tests for CloudLoadTester without network access."""

    def test_init_defaults(self):
        t = CloudLoadTester("https://example.com", "key123")
        assert t.api_url == "https://example.com"
        assert t.api_key == "key123"
        assert t.concurrency == 50
        assert t.request_timeout == 60

    def test_init_strips_trailing_slash(self):
        t = CloudLoadTester("https://example.com/", "key")
        assert t.api_url == "https://example.com"

    def test_init_min_concurrency(self):
        t = CloudLoadTester("https://example.com", "key", concurrency=0)
        assert t.concurrency == 1

    def test_stop_sets_event(self):
        t = CloudLoadTester("https://example.com", "key")
        assert not t._stopped.is_set()
        t.stop()
        assert t._stopped.is_set()

    def test_generate_report_empty(self):
        t = CloudLoadTester("https://example.com", "key")
        report = t.generate_report()
        assert "error" in report

    @patch("scripts.cloud_load_test.urllib.request.urlopen")
    def test_submit_job_success(self, mock_urlopen):
        """Test successful job submission."""
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 201
        mock_resp.read.return_value = json.dumps({"job_id": "test-123"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        t = CloudLoadTester("https://example.com", "key")
        result = t._submit_job(0)

        assert result.status == "submitted"
        assert result.job_id == "test-123"
        assert result.http_status == 201
        assert result.error is None
        assert result.response_time >= 0

    @patch("scripts.cloud_load_test.urllib.request.urlopen")
    def test_submit_job_network_error(self, mock_urlopen):
        """Test job submission with network error."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        t = CloudLoadTester("https://example.com", "key")
        result = t._submit_job(0)

        assert result.status == "error"
        assert result.error is not None
        assert "Connection refused" in result.error

    @patch("scripts.cloud_load_test.urllib.request.urlopen")
    def test_run_load_test_collects_results(self, mock_urlopen):
        """Test that run_load_test accumulates results."""
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 201
        mock_resp.read.return_value = json.dumps({"job_id": "j1"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        t = CloudLoadTester("https://example.com", "key", concurrency=2)
        results = t.run_load_test(num_jobs=5, duration_minutes=1)

        assert len(results) == 5
        assert all(r.status == "submitted" for r in results)

    @patch("scripts.cloud_load_test.urllib.request.urlopen")
    def test_generate_report_structure(self, mock_urlopen):
        """Test report structure after a load test."""
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 201
        mock_resp.read.return_value = json.dumps({"job_id": "j1"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        t = CloudLoadTester("https://example.com", "key", concurrency=1)
        t.run_load_test(num_jobs=3, duration_minutes=1)
        report = t.generate_report()

        assert "timestamp" in report
        assert "api_url" in report
        assert "summary" in report
        assert "latency" in report
        assert "status_codes" in report

        summary = report["summary"]
        assert summary["total_jobs"] == 3
        assert summary["successful"] == 3
        assert summary["success_rate"] == 100.0
        assert summary["duration_seconds"] >= 0
        assert summary["throughput_jobs_per_second"] >= 0

        latency = report["latency"]
        assert "min_ms" in latency
        assert "max_ms" in latency
        assert "mean_ms" in latency
        assert "median_ms" in latency
        assert "p95_ms" in latency
        assert "p99_ms" in latency
        assert "stdev_ms" in latency

    @patch("scripts.cloud_load_test.urllib.request.urlopen")
    def test_generate_report_with_failures(self, mock_urlopen):
        """Test report captures failures."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("timeout")

        t = CloudLoadTester("https://example.com", "key", concurrency=1)
        t.run_load_test(num_jobs=2, duration_minutes=1)
        report = t.generate_report()

        assert report["summary"]["failed"] == 2
        assert report["summary"]["success_rate"] == 0.0
        assert len(report["errors"]) == 2

    def test_duration_limit_respected(self):
        """Test that duration_minutes caps the test."""
        t = CloudLoadTester("https://example.com", "key", concurrency=1)

        def slow_submit(idx):
            time.sleep(0.5)
            r = JobResult()
            r.status = "submitted"
            r.response_time = 0.5
            with t._lock:
                t._results.append(r)
            return r

        t._submit_job = slow_submit
        # With duration 0 minutes, should submit very few
        t._start_time = time.monotonic()
        results = t.run_load_test(num_jobs=1000, duration_minutes=0)
        # Should have stopped quickly (well under 1000 jobs)
        assert len(results) < 100


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    """Tests for the CLI entry point."""

    @patch("scripts.cloud_load_test.CloudLoadTester")
    def test_main_requires_api_url_and_key(self, mock_cls):
        """Test that --api-url and --api-key are required."""
        with pytest.raises(SystemExit):
            main([])

    @patch("scripts.cloud_load_test.urllib.request.urlopen")
    def test_main_runs_to_completion(self, mock_urlopen):
        """Test CLI runs end to end with mocked network."""
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 201
        mock_resp.read.return_value = json.dumps({"job_id": "j1"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        exit_code = main([
            "--api-url", "https://example.com",
            "--api-key", "test-key",
            "--num-jobs", "2",
            "--concurrency", "1",
            "--duration-minutes", "1",
        ])
        assert exit_code == 0

    @patch("scripts.cloud_load_test.urllib.request.urlopen")
    def test_main_failure_exit_code(self, mock_urlopen):
        """Test CLI returns 1 when success rate is below 90%."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("fail")

        exit_code = main([
            "--api-url", "https://example.com",
            "--api-key", "test-key",
            "--num-jobs", "2",
            "--concurrency", "1",
            "--duration-minutes", "1",
        ])
        assert exit_code == 1

    @patch("scripts.cloud_load_test.urllib.request.urlopen")
    def test_main_output_to_file(self, mock_urlopen, tmp_path):
        """Test CLI writes report to file."""
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 201
        mock_resp.read.return_value = json.dumps({"job_id": "j1"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        output_file = str(tmp_path / "report.json")
        exit_code = main([
            "--api-url", "https://example.com",
            "--api-key", "test-key",
            "--num-jobs", "1",
            "--output", output_file,
        ])
        assert exit_code == 0
        with open(output_file) as f:
            report = json.load(f)
        assert "summary" in report
