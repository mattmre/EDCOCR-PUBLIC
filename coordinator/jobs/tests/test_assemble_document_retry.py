"""Tests for assemble_document Celery retry logic.

Validates that transient S3 upload failures during assembly trigger automatic
retries with exponential backoff, and that permanent failures properly mark
the job as FAILED after retries are exhausted.  Also verifies idempotency
guards prevent duplicate custody events on retry.

Run with: cd coordinator && python -m pytest jobs/tests/test_assemble_document_retry.py -v
"""

import os
import shutil
import socket
import tempfile
import uuid
from unittest.mock import MagicMock, PropertyMock, patch

from celery.exceptions import MaxRetriesExceededError, Retry
from django.test import TestCase

from jobs.models import CustodyEvent, Job
from jobs.tasks import _ensure_job_dirs, assemble_document


class TestAssembleDocumentRetryConfig(TestCase):
    """Tests that the assemble_document task has correct retry configuration."""

    def test_max_retries_is_three(self):
        assert assemble_document.max_retries == 3

    def test_retry_backoff_enabled(self):
        assert assemble_document.retry_backoff is True

    def test_retry_backoff_max_is_sixty(self):
        assert assemble_document.retry_backoff_max == 60

    def test_retry_jitter_enabled(self):
        assert assemble_document.retry_jitter is True

    def test_queue_is_coordinator(self):
        assert assemble_document.queue == "coordinator"


class TestAssembleDocumentNoRetryOnPermanent(TestCase):
    """Tests that permanent/non-retryable conditions are not retried."""

    def test_missing_job_returns_error_no_retry(self):
        """Job not found should return error immediately, no retry."""
        fake_id = str(uuid.uuid4())
        result = assemble_document.run(fake_id)
        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_cancelled_job_returns_cancelled_no_retry(self):
        """Cancelled job should return cancelled immediately, no retry."""
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.CANCELLED,
        )
        result = assemble_document.run(str(job.job_id))
        assert result["status"] == "cancelled"


class TestAssembleDocumentS3UploadRetry(TestCase):
    """Tests for S3 upload failure retry behavior in assemble_document."""

    def _make_job_with_nfs(self, total_pages=2, **kwargs):
        """Create a job with NFS backend and page artifacts."""
        tmpdir = tempfile.mkdtemp()
        _ensure_job_dirs(tmpdir)
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=total_pages,
            source_hash="abcdef1234567890" * 4,
            nfs_job_path=tmpdir,
            detected_language="en",
            **kwargs,
        )
        # Create page artifacts in temp dir
        document_id = job.source_hash[:16]
        page_temp_dir = os.path.join(tmpdir, "temp", document_id)
        os.makedirs(page_temp_dir, exist_ok=True)
        for page_num in range(1, total_pages + 1):
            # Create minimal PDF-like files (fitz.open will handle them)
            pdf_path = os.path.join(page_temp_dir, f"{page_num}.pdf")
            txt_path = os.path.join(page_temp_dir, f"{page_num}.txt")
            with open(txt_path, "w") as f:
                f.write(f"Page {page_num} text")
            # Create a valid single-page PDF via fitz
            import fitz
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {page_num}")
            doc.save(pdf_path)
            doc.close()
        return job, tmpdir

    def _cleanup(self, tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("jobs.tasks._get_backend_for_job")
    def test_s3_upload_failure_triggers_retry(self, mock_get_backend):
        """S3 upload failure during assembly should trigger retry."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = True
        backend.download_file.return_value = None
        backend.upload_file.side_effect = ConnectionError("S3 timeout")
        mock_get_backend.return_value = backend

        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
            source_hash="abcdef1234567890" * 4,
            detected_language="en",
        )

        with patch.object(
            assemble_document, "retry",
            side_effect=Retry(),
        ):
            with patch.object(
                type(assemble_document.request), "retries",
                new_callable=PropertyMock, return_value=0,
            ):
                with self.assertRaises(Retry):
                    assemble_document.run(str(job.job_id))

        # Job should NOT be marked FAILED yet (retry pending)
        job.refresh_from_db()
        assert job.status != Job.Status.FAILED

    @patch("jobs.tasks._get_backend_for_job")
    def test_s3_upload_marks_failed_after_max_retries(self, mock_get_backend):
        """S3 upload permanently failing should mark job FAILED."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = True
        backend.download_file.return_value = None
        backend.upload_file.side_effect = ConnectionError("S3 timeout")
        mock_get_backend.return_value = backend

        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
            source_hash="abcdef1234567890" * 4,
            detected_language="en",
        )

        with patch.object(
            assemble_document, "retry",
            side_effect=MaxRetriesExceededError(),
        ):
            with patch.object(
                type(assemble_document.request), "retries",
                new_callable=PropertyMock, return_value=3,
            ):
                result = assemble_document.run(str(job.job_id))

        assert result["status"] == "error"
        assert "S3 upload failed" in result["message"]

        job.refresh_from_db()
        assert job.status == Job.Status.FAILED
        assert "S3 upload failed during assembly" in job.error_message

    @patch("jobs.tasks._get_backend_for_job")
    def test_s3_upload_retry_logs_warning(self, mock_get_backend):
        """S3 upload retry should log a warning with attempt number."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = True
        backend.download_file.return_value = None
        backend.upload_file.side_effect = ConnectionError("S3 timeout")
        mock_get_backend.return_value = backend

        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
            source_hash="abcdef1234567890" * 4,
            detected_language="en",
        )

        with patch.object(
            assemble_document, "retry",
            side_effect=Retry(),
        ):
            with patch.object(
                type(assemble_document.request), "retries",
                new_callable=PropertyMock, return_value=1,
            ):
                with self.assertRaises(Retry):
                    with self.assertLogs("jobs.tasks", level="WARNING") as cm:
                        assemble_document.run(str(job.job_id))

        retry_logs = [
            log for log in cm.output
            if "Retrying assemble_document" in log
            and "S3 upload failure" in log
        ]
        assert len(retry_logs) == 1
        assert "attempt 2/3" in retry_logs[0]

    @patch("jobs.tasks._get_backend_for_job")
    def test_permanent_failure_logs_error(self, mock_get_backend):
        """Permanent failure (after max retries) should log error."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = True
        backend.download_file.return_value = None
        backend.upload_file.side_effect = ConnectionError("final S3 error")
        mock_get_backend.return_value = backend

        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
            source_hash="abcdef1234567890" * 4,
            detected_language="en",
        )

        with patch.object(
            assemble_document, "retry",
            side_effect=MaxRetriesExceededError(),
        ):
            with patch.object(
                type(assemble_document.request), "retries",
                new_callable=PropertyMock, return_value=3,
            ):
                with self.assertLogs("jobs.tasks", level="ERROR") as cm:
                    assemble_document.run(str(job.job_id))

        error_logs = [
            log for log in cm.output
            if "permanently failed" in log
        ]
        assert len(error_logs) == 1
        assert "3 retries" in error_logs[0]


class TestAssembleDocumentNfsSuccess(TestCase):
    """Tests that NFS-mode assembly succeeds without retry interference."""

    def _make_job_with_nfs_pages(self, total_pages=2):
        """Create a job with NFS backend and valid page PDFs."""
        tmpdir = tempfile.mkdtemp()
        _ensure_job_dirs(tmpdir)
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=total_pages,
            source_hash="abcdef1234567890" * 4,
            nfs_job_path=tmpdir,
            detected_language="en",
        )
        # Create page artifacts
        document_id = job.source_hash[:16]
        page_temp_dir = os.path.join(tmpdir, "temp", document_id)
        os.makedirs(page_temp_dir, exist_ok=True)
        for page_num in range(1, total_pages + 1):
            pdf_path = os.path.join(page_temp_dir, f"{page_num}.pdf")
            txt_path = os.path.join(page_temp_dir, f"{page_num}.txt")
            with open(txt_path, "w") as f:
                f.write(f"Page {page_num} text")
            import fitz
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {page_num}")
            doc.save(pdf_path)
            doc.close()
        return job, tmpdir

    def _cleanup(self, tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("jobs.tasks.chord")
    def test_nfs_assembly_succeeds(self, mock_chord):
        """NFS-mode assembly should complete without triggering retry logic."""
        mock_chord.return_value = MagicMock()
        job, tmpdir = self._make_job_with_nfs_pages(total_pages=2)

        try:
            result = assemble_document.run(str(job.job_id))

            assert result["status"] == "assembled"
            assert result["pages_assembled"] == 2

            job.refresh_from_db()
            assert job.pages_completed == 2
        finally:
            self._cleanup(tmpdir)


class TestAssembleDocumentCustodyIdempotency(TestCase):
    """Tests that custody events are not duplicated on retry."""

    @patch("jobs.tasks.chord")
    def test_custody_event_not_duplicated_on_retry(self, mock_chord):
        """If assembly_complete custody event already exists, skip creation."""
        mock_chord.return_value = MagicMock()
        tmpdir = tempfile.mkdtemp()
        _ensure_job_dirs(tmpdir)
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
            source_hash="abcdef1234567890" * 4,
            nfs_job_path=tmpdir,
            detected_language="en",
        )
        document_id = job.source_hash[:16]

        # Create page artifacts
        page_temp_dir = os.path.join(tmpdir, "temp", document_id)
        os.makedirs(page_temp_dir, exist_ok=True)
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Page 1")
        doc.save(os.path.join(page_temp_dir, "1.pdf"))
        doc.close()
        with open(os.path.join(page_temp_dir, "1.txt"), "w") as f:
            f.write("Page 1 text")

        # Pre-create an assembly_complete custody event (simulating prior attempt)
        CustodyEvent.objects.create(
            document_id=document_id,
            job=job,
            event_type="assembly_complete",
            data={"pages_assembled": 1, "total_pages": 1, "output_pdf": "/old.pdf"},
            worker_hostname=socket.gethostname(),
        )

        try:
            result = assemble_document.run(str(job.job_id))
            assert result["status"] == "assembled"

            # Should still be exactly 1 assembly_complete event (not 2)
            event_count = CustodyEvent.objects.filter(
                job=job, document_id=document_id, event_type="assembly_complete"
            ).count()
            assert event_count == 1, (
                f"Expected 1 assembly_complete event but found {event_count}"
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("jobs.tasks.chord")
    def test_custody_event_created_on_first_attempt(self, mock_chord):
        """First successful assembly should create exactly one custody event."""
        mock_chord.return_value = MagicMock()
        tmpdir = tempfile.mkdtemp()
        _ensure_job_dirs(tmpdir)
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
            source_hash="abcdef1234567890" * 4,
            nfs_job_path=tmpdir,
            detected_language="en",
        )
        document_id = job.source_hash[:16]

        # Create page artifacts
        page_temp_dir = os.path.join(tmpdir, "temp", document_id)
        os.makedirs(page_temp_dir, exist_ok=True)
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Page 1")
        doc.save(os.path.join(page_temp_dir, "1.pdf"))
        doc.close()
        with open(os.path.join(page_temp_dir, "1.txt"), "w") as f:
            f.write("Page 1 text")

        try:
            result = assemble_document.run(str(job.job_id))
            assert result["status"] == "assembled"

            event_count = CustodyEvent.objects.filter(
                job=job, document_id=document_id, event_type="assembly_complete"
            ).count()
            assert event_count == 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
