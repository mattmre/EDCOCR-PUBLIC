"""Tests for process_document Celery retry logic.

Validates that transient failures (S3 timeout, DB drop, OCR errors) trigger
automatic retries with exponential backoff, and that permanent failures
properly mark the job as FAILED after retries are exhausted.

Run with: cd coordinator && python -m pytest jobs/tests/test_process_document_retry.py -v
"""

import shutil
import socket
import tempfile
import uuid
from unittest.mock import MagicMock, PropertyMock, patch

from celery.exceptions import MaxRetriesExceededError, Retry
from django.test import TestCase

from jobs.models import Job, Worker
from jobs.tasks import _ensure_job_dirs, process_document


class TestProcessDocumentRetryConfig(TestCase):
    """Tests that the process_document task has correct retry configuration."""

    def test_max_retries_is_three(self):
        assert process_document.max_retries == 3

    def test_retry_backoff_enabled(self):
        assert process_document.retry_backoff is True

    def test_retry_backoff_max_is_sixty(self):
        assert process_document.retry_backoff_max == 60

    def test_retry_jitter_enabled(self):
        assert process_document.retry_jitter is True

    def test_acks_late_enabled(self):
        assert process_document.acks_late is True

    def test_reject_on_worker_lost_enabled(self):
        assert process_document.reject_on_worker_lost is True


class TestProcessDocumentRetryOnFailure(TestCase):
    """Tests that transient processing failures trigger retries."""

    def _make_job_with_nfs(self, total_pages=2, **kwargs):
        tmpdir = tempfile.mkdtemp()
        _ensure_job_dirs(tmpdir)
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=total_pages,
            source_hash="abcdef1234567890" * 4,
            nfs_job_path=tmpdir,
            detected_language="en",
            **kwargs,
        )
        return job, tmpdir

    def _cleanup(self, tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("jobs.tasks._process_single_page")
    def test_retry_called_on_processing_exception(self, mock_process):
        """Processing failure should call self.retry() for transient errors."""
        mock_process.side_effect = RuntimeError("GPU OOM")
        job, tmpdir = self._make_job_with_nfs(total_pages=1)
        try:
            # In eager mode, self.retry() raises Retry
            with self.assertRaises(Retry):
                process_document.apply(
                    args=[str(job.job_id)],
                    throw=True,
                )
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks._process_single_page")
    def test_job_marked_failed_after_max_retries(self, mock_process):
        """After max retries exhausted, job status should be FAILED."""
        mock_process.side_effect = RuntimeError("persistent GPU failure")
        job, tmpdir = self._make_job_with_nfs(total_pages=1)

        try:
            # Patch retry on the task to simulate max retries exhausted
            with patch.object(
                process_document, "retry",
                side_effect=MaxRetriesExceededError(),
            ):
                # Also need to patch request.retries to return 3
                with patch.object(
                    type(process_document.request), "retries",
                    new_callable=PropertyMock, return_value=3,
                ):
                    result = process_document.run(str(job.job_id))

            assert result["status"] == "error"
            assert "persistent GPU failure" in result["message"]

            job.refresh_from_db()
            assert job.status == Job.Status.FAILED
            assert "persistent GPU failure" in job.error_message
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks._process_single_page")
    def test_retry_logs_warning_with_attempt_number(self, mock_process):
        """Retry attempt should log a warning with the attempt number."""
        mock_process.side_effect = RuntimeError("transient error")
        job, tmpdir = self._make_job_with_nfs(total_pages=1)

        try:
            with patch.object(
                process_document, "retry",
                side_effect=Retry(),
            ):
                with patch.object(
                    type(process_document.request), "retries",
                    new_callable=PropertyMock, return_value=1,
                ):
                    with self.assertRaises(Retry):
                        with self.assertLogs("jobs.tasks", level="WARNING") as cm:
                            process_document.run(str(job.job_id))

            # Check that retry warning was logged with attempt number
            retry_logs = [
                log for log in cm.output
                if "Retrying process_document" in log
            ]
            assert len(retry_logs) == 1
            assert "attempt 2/3" in retry_logs[0]
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks._process_single_page")
    def test_worker_failure_counter_incremented_on_final_failure(self, mock_process):
        """Worker tasks_failed should increment only after final retry exhausted."""
        mock_process.side_effect = RuntimeError("permanent error")
        hostname = socket.gethostname()
        Worker.objects.create(hostname=hostname, status=Worker.Status.ONLINE)
        job, tmpdir = self._make_job_with_nfs(total_pages=1)

        try:
            with patch.object(
                process_document, "retry",
                side_effect=MaxRetriesExceededError(),
            ):
                with patch.object(
                    type(process_document.request), "retries",
                    new_callable=PropertyMock, return_value=3,
                ):
                    process_document.run(str(job.job_id))

            worker = Worker.objects.get(hostname=hostname)
            assert worker.tasks_failed == 1
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks._process_single_page")
    def test_worker_status_reset_on_retry(self, mock_process):
        """Worker status should be reset to ONLINE when retrying."""
        mock_process.side_effect = RuntimeError("transient error")
        hostname = socket.gethostname()
        Worker.objects.create(hostname=hostname, status=Worker.Status.ONLINE)
        job, tmpdir = self._make_job_with_nfs(total_pages=1)

        try:
            with patch.object(
                process_document, "retry",
                side_effect=Retry(),
            ):
                with patch.object(
                    type(process_document.request), "retries",
                    new_callable=PropertyMock, return_value=0,
                ):
                    with self.assertRaises(Retry):
                        process_document.run(str(job.job_id))

            worker = Worker.objects.get(hostname=hostname)
            assert worker.status == Worker.Status.ONLINE
            assert worker.current_task_id == ""
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks._process_single_page")
    def test_job_not_marked_failed_on_retry(self, mock_process):
        """Job should NOT be marked FAILED while retries remain."""
        mock_process.side_effect = RuntimeError("transient error")
        job, tmpdir = self._make_job_with_nfs(total_pages=1)

        try:
            with patch.object(
                process_document, "retry",
                side_effect=Retry(),
            ):
                with patch.object(
                    type(process_document.request), "retries",
                    new_callable=PropertyMock, return_value=0,
                ):
                    with self.assertRaises(Retry):
                        process_document.run(str(job.job_id))

            job.refresh_from_db()
            # Job should still be PROCESSING, not FAILED
            assert job.status != Job.Status.FAILED
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks._process_single_page")
    def test_permanent_failure_logs_error(self, mock_process):
        """Permanent failure (after max retries) should log error."""
        mock_process.side_effect = RuntimeError("final error")
        job, tmpdir = self._make_job_with_nfs(total_pages=1)

        try:
            with patch.object(
                process_document, "retry",
                side_effect=MaxRetriesExceededError(),
            ):
                with patch.object(
                    type(process_document.request), "retries",
                    new_callable=PropertyMock, return_value=3,
                ):
                    with self.assertLogs("jobs.tasks", level="ERROR") as cm:
                        process_document.run(str(job.job_id))

            error_logs = [
                log for log in cm.output
                if "permanently failed" in log
            ]
            assert len(error_logs) == 1
            assert "3 retries" in error_logs[0]
        finally:
            self._cleanup(tmpdir)


class TestProcessDocumentNoRetryOnPermanent(TestCase):
    """Tests that permanent/non-retryable conditions are not retried."""

    def test_missing_job_returns_error_no_retry(self):
        """Job not found should return error immediately, no retry."""
        fake_id = str(uuid.uuid4())
        result = process_document.run(fake_id)
        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_cancelled_job_returns_cancelled_no_retry(self):
        """Cancelled job should return cancelled immediately, no retry."""
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.CANCELLED,
        )
        result = process_document.run(str(job.job_id))
        assert result["status"] == "cancelled"

    @patch("jobs.tasks._process_single_page")
    def test_keyboard_interrupt_not_retried(self, mock_process):
        """KeyboardInterrupt should propagate, not be retried."""
        mock_process.side_effect = KeyboardInterrupt()
        tmpdir = tempfile.mkdtemp()
        _ensure_job_dirs(tmpdir)
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=1,
            source_hash="abcdef1234567890" * 4,
            nfs_job_path=tmpdir,
            detected_language="en",
        )
        try:
            with self.assertRaises(KeyboardInterrupt):
                process_document.run(str(job.job_id))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("jobs.tasks._process_single_page")
    def test_system_exit_not_retried(self, mock_process):
        """SystemExit should propagate, not be retried."""
        mock_process.side_effect = SystemExit(1)
        tmpdir = tempfile.mkdtemp()
        _ensure_job_dirs(tmpdir)
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=1,
            source_hash="abcdef1234567890" * 4,
            nfs_job_path=tmpdir,
            detected_language="en",
        )
        try:
            with self.assertRaises(SystemExit):
                process_document.run(str(job.job_id))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestProcessDocumentS3DownloadRetry(TestCase):
    """Tests for S3 download failure retry behavior."""

    @patch("jobs.tasks._get_backend_for_job")
    def test_s3_download_failure_triggers_retry(self, mock_get_backend):
        """S3 download failure should trigger retry, not immediate FAILED."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.download_file.side_effect = ConnectionError("S3 timeout")
        mock_get_backend.return_value = backend

        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=1,
            source_hash="abcdef1234567890" * 4,
            detected_language="en",
        )

        with patch.object(
            process_document, "retry",
            side_effect=Retry(),
        ):
            with patch.object(
                type(process_document.request), "retries",
                new_callable=PropertyMock, return_value=0,
            ):
                with self.assertRaises(Retry):
                    process_document.run(str(job.job_id))

        # Job should NOT be marked FAILED yet (retry pending)
        job.refresh_from_db()
        assert job.status != Job.Status.FAILED

    @patch("jobs.tasks._get_backend_for_job")
    def test_s3_download_failure_marks_failed_after_max_retries(self, mock_get_backend):
        """S3 download permanently failing should mark job FAILED."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.download_file.side_effect = ConnectionError("S3 timeout")
        mock_get_backend.return_value = backend

        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=1,
            source_hash="abcdef1234567890" * 4,
            detected_language="en",
        )

        with patch.object(
            process_document, "retry",
            side_effect=MaxRetriesExceededError(),
        ):
            with patch.object(
                type(process_document.request), "retries",
                new_callable=PropertyMock, return_value=3,
            ):
                result = process_document.run(str(job.job_id))

        assert result["status"] == "error"
        assert "Download failed" in result["message"]

        job.refresh_from_db()
        assert job.status == Job.Status.FAILED
        assert "S3 source download failed" in job.error_message

    @patch("jobs.tasks._get_backend_for_job")
    def test_s3_download_retry_logs_warning(self, mock_get_backend):
        """S3 download retry should log a warning with context."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.download_file.side_effect = ConnectionError("S3 timeout")
        mock_get_backend.return_value = backend

        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=1,
            source_hash="abcdef1234567890" * 4,
            detected_language="en",
        )

        with patch.object(
            process_document, "retry",
            side_effect=Retry(),
        ):
            with patch.object(
                type(process_document.request), "retries",
                new_callable=PropertyMock, return_value=1,
            ):
                with self.assertRaises(Retry):
                    with self.assertLogs("jobs.tasks", level="WARNING") as cm:
                        process_document.run(str(job.job_id))

        retry_logs = [
            log for log in cm.output
            if "Retrying process_document" in log
            and "S3 download failure" in log
        ]
        assert len(retry_logs) == 1
        assert "attempt 2/3" in retry_logs[0]
