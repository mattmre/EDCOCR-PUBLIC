"""Tests for Celery task definitions.

Tests the coordinator, OCR, and periodic tasks with all external
dependencies (PaddleOCR, Tesseract, filesystem, Celery broker) mocked.

For bind=True tasks, we use task.run(...) which auto-injects `self`.
For non-bind tasks, we call them directly as functions.

Run with: cd coordinator && python -m pytest jobs/tests/test_tasks.py -v
"""

import hashlib
import os
import shutil
import socket
import sys
import tempfile
import types
import uuid
from datetime import timedelta
from unittest.mock import MagicMock, PropertyMock, patch

from celery.exceptions import MaxRetriesExceededError
from django.conf import settings
from django.test import TestCase, override_settings
from django.utils import timezone

from jobs.models import CustodyEvent, Job, PageResult, Worker
from jobs.storage import NFSBackend
from jobs.tasks import (
    FANOUT_THRESHOLD,
    _compute_file_hash,
    _create_storage_backend,
    _ensure_job_dirs,
    _get_backend_for_job,
    _get_storage_backend,
    _job_storage_key,
    _nfs_job_path,
    _process_single_page,
    _record_custody_event,
    _reset_storage_backend,
    assemble_document,
    check_worker_heartbeats,
    chord_error_handler,
    cleanup_completed_jobs,
    cleanup_stale_jobs,
    compress_pdf,
    extract_entities,
    extract_pages,
    finalize_job,
    ingest_document,
    process_document,
    process_page,
    register_worker,
    unregister_worker,
)


class TestFanoutThreshold(TestCase):
    """Tests for the FANOUT_THRESHOLD constant."""

    def test_fanout_threshold_value(self):
        assert FANOUT_THRESHOLD == 20


class TestNfsJobPath(TestCase):
    """Tests for _nfs_job_path helper."""

    def test_returns_correct_path(self):
        job_id = uuid.uuid4()
        result = _nfs_job_path(job_id)
        expected = os.path.join(settings.NFS_ROOT, "jobs", str(job_id))
        assert result == expected

    @override_settings(STORAGE_BACKEND="s3")
    def test_returns_path_even_when_storage_backend_is_s3(self):
        job_id = uuid.uuid4()
        result = _nfs_job_path(job_id)
        expected = os.path.join(settings.NFS_ROOT, "jobs", str(job_id))
        assert result == expected


class TestEnsureJobDirs(TestCase):
    """Tests for _ensure_job_dirs helper."""

    def test_creates_all_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _ensure_job_dirs(tmpdir)
            expected_dirs = [
                "source",
                "temp",
                os.path.join("output", "EXPORT", "PDF"),
                os.path.join("output", "EXPORT", "TEXT"),
                os.path.join("output", "EXPORT", "STRUCTURE"),
                os.path.join("output", "EXPORT", "NER"),
                os.path.join("output", "EXPORT", "VALIDATION"),
                os.path.join("output", "EXPORT", "CUSTODY"),
            ]
            for subdir in expected_dirs:
                full_path = os.path.join(tmpdir, subdir)
                assert os.path.isdir(full_path), f"Missing directory: {subdir}"

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _ensure_job_dirs(tmpdir)
            _ensure_job_dirs(tmpdir)  # Should not raise
            assert os.path.isdir(os.path.join(tmpdir, "source"))


class TestComputeFileHash(TestCase):
    """Tests for _compute_file_hash helper."""

    def test_computes_sha256(self):
        # On Windows, NamedTemporaryFile must be closed before re-opening
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            tmp.write(b"Hello, World!")
            tmp.close()
            result = _compute_file_hash(tmp.name)
            expected = hashlib.sha256(b"Hello, World!").hexdigest()
            assert result == expected
        finally:
            os.unlink(tmp.name)

    def test_empty_file(self):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            tmp.close()
            result = _compute_file_hash(tmp.name)
            expected = hashlib.sha256(b"").hexdigest()
            assert result == expected
        finally:
            os.unlink(tmp.name)


class TestRecordCustodyEvent(TestCase):
    """Tests for _record_custody_event helper."""

    def test_creates_event(self):
        job = Job.objects.create(source_file="/test.pdf")
        _record_custody_event(
            job,
            document_id="doc123",
            event_type="file_ingested",
            data={"source_hash": "abc"},
            worker_hostname="worker-01",
        )
        events = CustodyEvent.objects.filter(job=job)
        assert events.count() == 1
        event = events.first()
        assert event.document_id == "doc123"
        assert event.event_type == "file_ingested"
        assert event.data == {"source_hash": "abc"}
        assert event.worker_hostname == "worker-01"

    def test_creates_event_with_empty_data(self):
        job = Job.objects.create(source_file="/test.pdf")
        _record_custody_event(job, "doc123", "test_event")
        event = CustodyEvent.objects.filter(job=job).first()
        assert event.data == {}
        assert event.worker_hostname == ""


class TestRegisterWorker(TestCase):
    """Tests for register_worker function."""

    def test_creates_new_worker(self):
        register_worker(
            hostname="new-worker",
            queues=["ocr_gpu", "cpu_general"],
            capabilities=["ocr", "compress"],
            gpu_available=True,
            gpu_model="RTX 4090",
            gpu_vram_mb=24576,
            cpu_cores=16,
            ram_mb=65536,
            pipeline_version="0.5.0",
        )
        worker = Worker.objects.get(hostname="new-worker")
        assert worker.status == Worker.Status.ONLINE
        assert worker.queues == ["ocr_gpu", "cpu_general"]
        assert worker.capabilities == ["ocr", "compress"]
        assert worker.gpu_available is True
        assert worker.gpu_model == "RTX 4090"
        assert worker.gpu_vram_mb == 24576
        assert worker.cpu_cores == 16
        assert worker.ram_mb == 65536
        assert worker.pipeline_version == "0.5.0"
        assert worker.last_heartbeat is not None

    def test_updates_existing_worker(self):
        Worker.objects.create(
            hostname="existing-worker",
            status=Worker.Status.OFFLINE,
        )
        register_worker(
            hostname="existing-worker",
            queues=["ocr_gpu"],
            gpu_available=True,
        )
        worker = Worker.objects.get(hostname="existing-worker")
        assert worker.status == Worker.Status.ONLINE
        assert worker.queues == ["ocr_gpu"]

    def test_defaults_for_optional_params(self):
        register_worker(hostname="minimal-worker")
        worker = Worker.objects.get(hostname="minimal-worker")
        assert worker.queues == []
        assert worker.capabilities == []
        assert worker.gpu_available is False


class TestUnregisterWorker(TestCase):
    """Tests for unregister_worker function."""

    def test_marks_worker_offline(self):
        Worker.objects.create(
            hostname="active-worker",
            status=Worker.Status.ONLINE,
            current_task_id="task-123",
        )
        unregister_worker("active-worker")
        worker = Worker.objects.get(hostname="active-worker")
        assert worker.status == Worker.Status.OFFLINE
        assert worker.current_task_id == ""

    def test_no_error_for_unknown_worker(self):
        # Should not raise even if worker doesn't exist
        unregister_worker("nonexistent-worker")


class TestIngestDocument(TestCase):
    """Tests for ingest_document task."""

    def test_missing_job_returns_error(self):
        fake_id = str(uuid.uuid4())
        # bind=True task: .run() auto-injects self
        result = ingest_document.run(fake_id)
        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_cancelled_job_skips_processing(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.CANCELLED,
        )
        result = ingest_document.run(str(job.job_id))
        assert result["status"] == "cancelled"


class TestFinalizeJob(TestCase):
    """Tests for finalize_job task."""

    def test_missing_job_returns_error(self):
        fake_id = str(uuid.uuid4())
        result = finalize_job.run(fake_id)
        assert result["status"] == "error"

    def test_cancelled_job_returns_cancelled(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.CANCELLED,
        )
        result = finalize_job.run(str(job.job_id))
        assert result["status"] == "cancelled"

    def test_computes_result_summary(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=4,
        )
        # Create page results
        PageResult.objects.create(
            job=job, document_id="d1", page_num=1,
            status="ok", ocr_confidence=0.95, ocr_method="PaddleOCR",
        )
        PageResult.objects.create(
            job=job, document_id="d1", page_num=2,
            status="ok", ocr_confidence=0.90, ocr_method="PaddleOCR",
        )
        PageResult.objects.create(
            job=job, document_id="d1", page_num=3,
            status="fallback", ocr_confidence=0.70, ocr_method="Tesseract",
        )
        PageResult.objects.create(
            job=job, document_id="d1", page_num=4,
            status="image_only", ocr_confidence=0.0, ocr_method="ImageOnly",
        )

        result = finalize_job.run(str(job.job_id))

        assert result["status"] == "completed"
        summary = result["summary"]
        assert summary["total_pages"] == 4
        assert summary["pages_ok"] == 2
        assert summary["pages_fallback"] == 1
        assert summary["pages_image_only"] == 1
        assert summary["pages_failed"] == 0
        # Average confidence: (0.95 + 0.90 + 0.70) / 3
        expected_avg = round((0.95 + 0.90 + 0.70) / 3, 4)
        assert summary["average_confidence"] == expected_avg

        job.refresh_from_db()
        assert job.status == Job.Status.COMPLETED
        assert job.completed_at is not None

    def test_finalize_with_no_page_results(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=0,
        )
        result = finalize_job.run(str(job.job_id))
        assert result["status"] == "completed"
        assert result["summary"]["average_confidence"] == 0.0

    def test_finalize_marks_job_failed_when_any_page_failed(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=2,
        )
        PageResult.objects.create(
            job=job,
            document_id="d2",
            page_num=1,
            status="ok",
            ocr_confidence=0.92,
            ocr_method="PaddleOCR",
        )
        PageResult.objects.create(
            job=job,
            document_id="d2",
            page_num=2,
            status="failed",
            ocr_confidence=0.0,
            ocr_method="PaddleOCR",
        )

        result = finalize_job.run(str(job.job_id))
        assert result["status"] == "failed"
        assert result["summary"]["pages_failed"] == 1

        job.refresh_from_db()
        assert job.status == Job.Status.FAILED


class TestCheckWorkerHeartbeats(TestCase):
    """Tests for check_worker_heartbeats periodic task."""

    def test_marks_stale_workers_offline(self):
        now = timezone.now()
        stale_time = now - timedelta(minutes=5)

        Worker.objects.create(
            hostname="stale-worker",
            status=Worker.Status.ONLINE,
            last_heartbeat=stale_time,
        )
        Worker.objects.create(
            hostname="fresh-worker",
            status=Worker.Status.BUSY,
            last_heartbeat=now,
        )

        result = check_worker_heartbeats()
        assert result["stale_workers"] == 1

        stale = Worker.objects.get(hostname="stale-worker")
        assert stale.status == Worker.Status.OFFLINE

        fresh = Worker.objects.get(hostname="fresh-worker")
        assert fresh.status == Worker.Status.BUSY

    def test_no_stale_workers(self):
        Worker.objects.create(
            hostname="active-worker",
            status=Worker.Status.ONLINE,
            last_heartbeat=timezone.now(),
        )
        result = check_worker_heartbeats()
        assert result["stale_workers"] == 0

    def test_ignores_offline_workers(self):
        stale_time = timezone.now() - timedelta(minutes=5)
        Worker.objects.create(
            hostname="already-offline",
            status=Worker.Status.OFFLINE,
            last_heartbeat=stale_time,
        )
        result = check_worker_heartbeats()
        assert result["stale_workers"] == 0


class TestCleanupStaleJobs(TestCase):
    """Tests for cleanup_stale_jobs periodic task."""

    def test_flags_stuck_jobs(self):
        stale_time = timezone.now() - timedelta(minutes=45)
        job = Job.objects.create(
            source_file="/stuck.pdf",
            status=Job.Status.PROCESSING,
            started_at=stale_time,
        )
        result = cleanup_stale_jobs()
        assert result["stale_jobs"] == 1

        job.refresh_from_db()
        assert "30 minutes" in job.error_message

    def test_does_not_flag_recent_jobs(self):
        Job.objects.create(
            source_file="/recent.pdf",
            status=Job.Status.PROCESSING,
            started_at=timezone.now() - timedelta(minutes=5),
        )
        result = cleanup_stale_jobs()
        assert result["stale_jobs"] == 0

    def test_flags_stuck_assembling_jobs(self):
        stale_time = timezone.now() - timedelta(minutes=45)
        Job.objects.create(
            source_file="/assembling.pdf",
            status=Job.Status.ASSEMBLING,
            started_at=stale_time,
        )
        result = cleanup_stale_jobs()
        assert result["stale_jobs"] == 1

    def test_ignores_completed_jobs(self):
        stale_time = timezone.now() - timedelta(minutes=45)
        Job.objects.create(
            source_file="/done.pdf",
            status=Job.Status.COMPLETED,
            started_at=stale_time,
        )
        result = cleanup_stale_jobs()
        assert result["stale_jobs"] == 0

    @override_settings(JOB_PROCESSING_TIMEOUT_MINUTES=10)
    def test_uses_default_timeout_setting(self):
        stale_time = timezone.now() - timedelta(minutes=15)
        job = Job.objects.create(
            source_file="/configured.pdf",
            status=Job.Status.PROCESSING,
            started_at=stale_time,
        )
        result = cleanup_stale_jobs()
        assert result["stale_jobs"] == 1

        job.refresh_from_db()
        assert "10 minutes" in job.error_message

    @override_settings(JOB_PROCESSING_TIMEOUT_MINUTES=30)
    def test_per_job_timeout_override_controls_stale_detection(self):
        early_time = timezone.now() - timedelta(minutes=15)
        late_time = timezone.now() - timedelta(minutes=45)
        fast_job = Job.objects.create(
            source_file="/fast.pdf",
            status=Job.Status.PROCESSING,
            started_at=early_time,
            settings_json={"processing_timeout_minutes": 5},
        )
        slow_job = Job.objects.create(
            source_file="/slow.pdf",
            status=Job.Status.PROCESSING,
            started_at=late_time,
            settings_json={"processing_timeout_minutes": 60},
        )

        result = cleanup_stale_jobs()
        assert result["stale_jobs"] == 1

        fast_job.refresh_from_db()
        slow_job.refresh_from_db()
        assert fast_job.status == Job.Status.FAILED
        assert "5 minutes" in fast_job.error_message
        assert slow_job.status == Job.Status.PROCESSING


class TestCompressPdf(TestCase):
    """Tests for compress_pdf task."""

    def test_missing_job_returns_error(self):
        fake_id = str(uuid.uuid4())
        result = compress_pdf.run(fake_id)
        assert result["status"] == "error"

    def test_no_pdf_file_returns_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job = Job.objects.create(
                source_file="/test.pdf",
                nfs_job_path=tmpdir,
            )
            _ensure_job_dirs(tmpdir)
            result = compress_pdf.run(str(job.job_id))
            assert result["status"] == "skipped"
            assert result["reason"] == "no_pdf"


class TestExtractEntities(TestCase):
    """Tests for extract_entities task."""

    def test_missing_job_returns_error(self):
        fake_id = str(uuid.uuid4())
        result = extract_entities.run(fake_id)
        assert result["status"] == "error"

    def test_no_text_file_returns_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job = Job.objects.create(
                source_file="/test.pdf",
                nfs_job_path=tmpdir,
            )
            _ensure_job_dirs(tmpdir)
            result = extract_entities.run(str(job.job_id))
            assert result["status"] == "skipped"
            assert result["reason"] == "no_text"

    def test_ner_module_unavailable_returns_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job = Job.objects.create(
                source_file="/document.pdf",
                nfs_job_path=tmpdir,
            )
            _ensure_job_dirs(tmpdir)
            # Create a text file
            text_dir = os.path.join(tmpdir, "output", "EXPORT", "TEXT")
            text_path = os.path.join(text_dir, "document.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write("Some OCR text content.")

            # Patch the ner import to raise ImportError
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "ner":
                    raise ImportError("No module named 'ner'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = extract_entities.run(str(job.job_id))
            assert result["status"] == "skipped"
            assert result["reason"] == "ner_unavailable"


# ---------------------------------------------------------------------------
# Step 5a: extract_pages tests
# ---------------------------------------------------------------------------

class TestExtractPages(TestCase):
    """Tests for extract_pages task (chord fan-out orchestration)."""

    def _make_job(self, total_pages=25, status=Job.Status.PROCESSING, **kwargs):
        return Job.objects.create(
            source_file="/docs/big_report.pdf",
            status=status,
            total_pages=total_pages,
            source_hash="abcdef1234567890" * 4,
            **kwargs,
        )

    @patch("jobs.tasks.chord")
    def test_creates_page_result_rows(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job = self._make_job(total_pages=3)
        extract_pages.run(str(job.job_id))
        assert PageResult.objects.filter(job=job).count() == 3
        for pn in range(1, 4):
            pr = PageResult.objects.get(job=job, page_num=pn)
            assert pr.status == "pending"
            assert pr.document_id == job.source_hash[:16]

    @patch("jobs.tasks.chord")
    def test_records_pages_extracted_custody_event(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job = self._make_job(total_pages=5)
        extract_pages.run(str(job.job_id))
        events = CustodyEvent.objects.filter(job=job, event_type="pages_extracted")
        assert events.count() == 1
        assert events.first().data["total_pages"] == 5

    @patch("jobs.tasks.chord")
    def test_dispatches_chord_with_correct_page_count(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job = self._make_job(total_pages=30)
        extract_pages.run(str(job.job_id))
        mock_chord.assert_called_once()
        page_tasks = mock_chord.call_args[0][0]
        assert len(page_tasks) == 30

    @patch("jobs.tasks.chord")
    def test_chord_wires_errback(self, mock_chord):
        mock_callback = MagicMock()
        mock_chord.return_value = mock_callback
        job = self._make_job(total_pages=3)
        extract_pages.run(str(job.job_id))
        page_tasks = mock_chord.call_args[0][0]
        assert all(task.options.get("link_error") for task in page_tasks)
        mock_callback.assert_called_once()

    def test_missing_job_returns_error(self):
        fake_id = str(uuid.uuid4())
        result = extract_pages.run(fake_id)
        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_cancelled_job_returns_cancelled(self):
        job = self._make_job(status=Job.Status.CANCELLED)
        result = extract_pages.run(str(job.job_id))
        assert result["status"] == "cancelled"

    @patch("jobs.tasks.chord")
    def test_returns_fanout_dispatched(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job = self._make_job(total_pages=25)
        result = extract_pages.run(str(job.job_id))
        assert result["status"] == "fanout_dispatched"
        assert result["page_tasks"] == 25

    @patch("jobs.tasks.chord")
    def test_zero_pages_creates_no_page_results(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job = self._make_job(total_pages=0)
        extract_pages.run(str(job.job_id))
        assert PageResult.objects.filter(job=job).count() == 0

    @patch("jobs.tasks.chord")
    def test_does_not_duplicate_page_results(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job = self._make_job(total_pages=2)
        doc_id = job.source_hash[:16]
        # Pre-create a page result (e.g. from a previous attempt)
        PageResult.objects.create(job=job, page_num=1, document_id=doc_id,
                                  status="failed")
        extract_pages.run(str(job.job_id))
        # Should still be 2 total (get_or_create doesn't duplicate)
        assert PageResult.objects.filter(job=job).count() == 2
        # Existing one should not be overwritten
        pr1 = PageResult.objects.get(job=job, page_num=1)
        assert pr1.status == "failed"

    @patch("jobs.tasks.chord")
    def test_document_id_fallback_when_no_hash(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=1,
            source_hash="",
        )
        extract_pages.run(str(job.job_id))
        pr = PageResult.objects.get(job=job, page_num=1)
        assert pr.document_id == str(job.job_id)[:16]


# ---------------------------------------------------------------------------
# Step 5b: process_page tests
# ---------------------------------------------------------------------------

class TestProcessPage(TestCase):
    """Tests for process_page task."""

    def _make_job_with_nfs(self, total_pages=5):
        """Create a job with a real temp directory for NFS paths."""
        tmpdir = tempfile.mkdtemp()
        _ensure_job_dirs(tmpdir)
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=total_pages,
            source_hash="abcdef1234567890" * 4,
            nfs_job_path=tmpdir,
            detected_language="en",
        )
        return job, tmpdir

    def _cleanup(self, tmpdir):
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("jobs.tasks._process_single_page")
    def test_successful_page_increments_progress(self, mock_process):
        mock_process.return_value = {
            "page_num": 1, "method": "PaddleOCR",
            "confidence": 0.95, "text_length": 100, "status": "ok",
        }
        job, tmpdir = self._make_job_with_nfs()
        try:
            result = process_page.run(str(job.job_id), 1)
            assert result["status"] == "page_processed"
            assert result["page_num"] == 1
            job.refresh_from_db()
            assert job.pages_completed == 1
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks._process_single_page")
    def test_failure_records_failed_page_result(self, mock_process):
        mock_process.side_effect = RuntimeError("OCR engine crashed")
        job, tmpdir = self._make_job_with_nfs()
        try:
            with self.assertRaises(RuntimeError):
                process_page.run(str(job.job_id), 3)
            pr = PageResult.objects.get(job=job, page_num=3)
            assert pr.status == "failed"
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks._process_single_page")
    def test_failure_records_custody_event(self, mock_process):
        mock_process.side_effect = ValueError("Bad image")
        job, tmpdir = self._make_job_with_nfs()
        try:
            with self.assertRaises(ValueError):
                process_page.run(str(job.job_id), 2)
            events = CustodyEvent.objects.filter(
                job=job, event_type="processing_failed"
            )
            assert events.count() == 1
            assert events.first().data["page_num"] == 2
            assert "Bad image" in events.first().data["error"]
        finally:
            self._cleanup(tmpdir)

    def test_missing_job_returns_error(self):
        fake_id = str(uuid.uuid4())
        result = process_page.run(fake_id, 1)
        assert result["status"] == "error"

    def test_cancelled_job_returns_cancelled(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.CANCELLED,
        )
        result = process_page.run(str(job.job_id), 1)
        assert result["status"] == "cancelled"

    @patch("jobs.tasks._process_single_page")
    def test_updates_worker_heartbeat(self, mock_process):
        mock_process.return_value = {
            "page_num": 1, "status": "ok", "method": "PaddleOCR",
        }
        job, tmpdir = self._make_job_with_nfs()
        hostname = __import__("socket").gethostname()
        Worker.objects.create(hostname=hostname, status=Worker.Status.ONLINE)
        try:
            process_page.run(str(job.job_id), 1)
            worker = Worker.objects.get(hostname=hostname)
            assert worker.last_heartbeat is not None
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks._process_single_page")
    def test_failure_re_raises_exception(self, mock_process):
        """process_page must re-raise so the chord detects the failure."""
        mock_process.side_effect = RuntimeError("Engine failed")
        job, tmpdir = self._make_job_with_nfs()
        try:
            with self.assertRaises(RuntimeError):
                process_page.run(str(job.job_id), 1)
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks._process_single_page")
    def test_failure_sets_worker_hostname(self, mock_process):
        mock_process.side_effect = RuntimeError("fail")
        job, tmpdir = self._make_job_with_nfs()
        try:
            with self.assertRaises(RuntimeError):
                process_page.run(str(job.job_id), 1)
            pr = PageResult.objects.get(job=job, page_num=1)
            assert pr.worker_hostname != ""
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks._process_single_page")
    def test_success_returns_page_num_in_result(self, mock_process):
        mock_process.return_value = {"page_num": 7, "status": "ok"}
        job, tmpdir = self._make_job_with_nfs()
        try:
            result = process_page.run(str(job.job_id), 7)
            assert result["page_num"] == 7
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks._process_single_page")
    def test_atomic_progress_update(self, mock_process):
        """pages_completed uses F() expression for atomic update."""
        mock_process.return_value = {"page_num": 1, "status": "ok"}
        job, tmpdir = self._make_job_with_nfs()
        job.pages_completed = 5
        job.save(update_fields=["pages_completed"])
        try:
            process_page.run(str(job.job_id), 1)
            job.refresh_from_db()
            assert job.pages_completed == 6
        finally:
            self._cleanup(tmpdir)


# ---------------------------------------------------------------------------
# Step 5c: process_document tests
# ---------------------------------------------------------------------------

class TestProcessDocument(TestCase):
    """Tests for process_document task (single-worker mode for small docs)."""

    def _make_job_with_nfs(self, total_pages=5, **kwargs):
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
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("jobs.tasks.assemble_document")
    @patch("jobs.tasks._process_single_page")
    def test_processes_all_pages_sequentially(self, mock_process, mock_assemble):
        mock_process.return_value = {"page_num": 1, "status": "ok"}
        mock_assemble.delay = MagicMock()
        job, tmpdir = self._make_job_with_nfs(total_pages=3)
        try:
            result = process_document.run(str(job.job_id))
            assert mock_process.call_count == 3
            assert result["status"] == "processed"
            assert result["pages_processed"] == 3
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.assemble_document")
    @patch("jobs.tasks._process_single_page")
    def test_dispatches_assemble_after_processing(self, mock_process, mock_assemble):
        mock_process.return_value = {"page_num": 1, "status": "ok"}
        mock_assemble.delay = MagicMock()
        job, tmpdir = self._make_job_with_nfs(total_pages=1)
        try:
            process_document.run(str(job.job_id))
            mock_assemble.delay.assert_called_once_with(str(job.job_id))
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.assemble_document")
    @patch("jobs.tasks._process_single_page")
    def test_updates_progress_after_each_page(self, mock_process, mock_assemble):
        mock_process.return_value = {"page_num": 1, "status": "ok"}
        mock_assemble.delay = MagicMock()
        job, tmpdir = self._make_job_with_nfs(total_pages=3)
        try:
            process_document.run(str(job.job_id))
            job.refresh_from_db()
            assert job.pages_completed == 3
        finally:
            self._cleanup(tmpdir)

    def test_missing_job_returns_error(self):
        fake_id = str(uuid.uuid4())
        result = process_document.run(fake_id)
        assert result["status"] == "error"

    def test_cancelled_job_returns_cancelled(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.CANCELLED,
        )
        result = process_document.run(str(job.job_id))
        assert result["status"] == "cancelled"

    @patch("jobs.tasks._process_single_page")
    def test_failure_sets_job_status_failed(self, mock_process):
        mock_process.side_effect = RuntimeError("GPU OOM")
        job, tmpdir = self._make_job_with_nfs(total_pages=2)
        try:
            # Simulate max retries exhausted so job goes to FAILED
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
            job.refresh_from_db()
            assert job.status == Job.Status.FAILED
            assert "GPU OOM" in job.error_message
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.assemble_document")
    @patch("jobs.tasks._process_single_page")
    def test_assigns_worker_hostname(self, mock_process, mock_assemble):
        mock_process.return_value = {"page_num": 1, "status": "ok"}
        mock_assemble.delay = MagicMock()
        job, tmpdir = self._make_job_with_nfs(total_pages=1)
        try:
            process_document.run(str(job.job_id))
            job.refresh_from_db()
            assert job.assigned_worker != ""
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.assemble_document")
    @patch("jobs.tasks._process_single_page")
    def test_updates_worker_status(self, mock_process, mock_assemble):
        mock_process.return_value = {"page_num": 1, "status": "ok"}
        mock_assemble.delay = MagicMock()
        hostname = __import__("socket").gethostname()
        Worker.objects.create(hostname=hostname, status=Worker.Status.ONLINE)
        job, tmpdir = self._make_job_with_nfs(total_pages=1)
        try:
            process_document.run(str(job.job_id))
            worker = Worker.objects.get(hostname=hostname)
            # After completion, worker status reset to ONLINE
            assert worker.status == Worker.Status.ONLINE
            assert worker.current_task_id == ""
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks._process_single_page")
    def test_failure_resets_worker_status(self, mock_process):
        mock_process.side_effect = RuntimeError("fail")
        hostname = __import__("socket").gethostname()
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
            assert worker.status == Worker.Status.ONLINE
            assert worker.current_task_id == ""
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.assemble_document")
    @patch("jobs.tasks._process_single_page")
    def test_zero_pages_still_dispatches_assemble(self, mock_process, mock_assemble):
        mock_assemble.delay = MagicMock()
        job, tmpdir = self._make_job_with_nfs(total_pages=0)
        try:
            result = process_document.run(str(job.job_id))
            assert result["status"] == "processed"
            mock_process.assert_not_called()
            mock_assemble.delay.assert_called_once()
        finally:
            self._cleanup(tmpdir)


# ---------------------------------------------------------------------------
# Step 5d: assemble_document tests
# ---------------------------------------------------------------------------

class TestAssembleDocument(TestCase):
    """Tests for assemble_document task (merge pages and finalize custody)."""

    def _make_job_with_pages(self, total_pages=3, pages_ok=None):
        """Create a job with NFS dirs and optional page PDFs."""
        tmpdir = tempfile.mkdtemp()
        _ensure_job_dirs(tmpdir)
        job = Job.objects.create(
            source_file="/document.pdf",
            status=Job.Status.PROCESSING,
            total_pages=total_pages,
            source_hash="abcdef1234567890" * 4,
            nfs_job_path=tmpdir,
        )
        document_id = job.source_hash[:16]

        # Create temp page files for specified pages
        if pages_ok is None:
            pages_ok = list(range(1, total_pages + 1))

        temp_dir = os.path.join(tmpdir, "temp", document_id)
        os.makedirs(temp_dir, exist_ok=True)

        for pn in pages_ok:
            self._create_page_pdf(temp_dir, pn)
            self._create_page_text(temp_dir, pn, f"Text for page {pn}")
            PageResult.objects.create(
                job=job, document_id=document_id, page_num=pn,
                status="ok", ocr_confidence=0.9,
            )

        return job, tmpdir, document_id

    def _create_page_pdf(self, temp_dir, page_num):
        """Create a minimal valid PDF file for a page."""
        import fitz
        doc = fitz.open()
        doc.new_page(width=100, height=100)
        doc.save(os.path.join(temp_dir, f"{page_num}.pdf"))
        doc.close()

    def _create_page_text(self, temp_dir, page_num, text):
        with open(os.path.join(temp_dir, f"{page_num}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(text)

    def _cleanup(self, tmpdir):
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("jobs.tasks.chord")
    def test_merges_all_page_pdfs(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, _ = self._make_job_with_pages(total_pages=3)
        try:
            result = assemble_document.run(str(job.job_id))
            assert result["status"] == "assembled"
            assert result["pages_assembled"] == 3
            # Check output PDF exists
            output_pdf = os.path.join(
                tmpdir, "output", "EXPORT", "PDF", "document.pdf"
            )
            assert os.path.isfile(output_pdf)
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_writes_merged_text(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, _ = self._make_job_with_pages(total_pages=2)
        try:
            assemble_document.run(str(job.job_id))
            output_text = os.path.join(
                tmpdir, "output", "EXPORT", "TEXT", "document.txt"
            )
            assert os.path.isfile(output_text)
            with open(output_text, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Text for page 1" in content
            assert "Text for page 2" in content
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_large_assembly_uses_periodic_checkpointing(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, document_id = self._make_job_with_pages(total_pages=2)
        checkpoint_path = os.path.join(tmpdir, "temp", document_id, "_merged_checkpoint.pdf")
        import jobs.tasks as tasks_module

        try:
            with patch("jobs.tasks.ASSEMBLY_STREAMING_THRESHOLD_PAGES", 1), \
                 patch("jobs.tasks.ASSEMBLY_CHECKPOINT_INTERVAL_PAGES", 1), \
                 patch("jobs.tasks._checkpoint_merged_pdf", wraps=tasks_module._checkpoint_merged_pdf) as mock_checkpoint:
                assemble_document.run(str(job.job_id))

            assert mock_checkpoint.call_count == 2
            assert not os.path.exists(checkpoint_path)
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_sets_assembling_status(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, _ = self._make_job_with_pages(total_pages=1)
        try:
            # We can't easily check mid-task status, but verify it changed
            assemble_document.run(str(job.job_id))
            job.refresh_from_db()
            assert job.pages_completed == 1
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_dispatches_post_processing_chord(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, _ = self._make_job_with_pages(total_pages=1)
        try:
            assemble_document.run(str(job.job_id))
            # chord is called with the list of post-processing tasks
            mock_chord.assert_called_once()
            post_tasks = mock_chord.call_args[0][0]
            assert len(post_tasks) == 3  # compress_pdf + extract_entities + extract_structured_data
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_chord_wires_callback_and_errback(self, mock_chord):
        mock_callback = MagicMock()
        mock_chord.return_value = mock_callback
        job, tmpdir, _ = self._make_job_with_pages(total_pages=1)
        try:
            assemble_document.run(str(job.job_id))
            post_tasks = mock_chord.call_args[0][0]
            assert all(task.options.get("link_error") for task in post_tasks)
            mock_callback.assert_called_once()
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_records_assembly_custody_event(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, doc_id = self._make_job_with_pages(total_pages=2)
        try:
            assemble_document.run(str(job.job_id))
            events = CustodyEvent.objects.filter(
                job=job, event_type="assembly_complete"
            )
            assert events.count() == 1
            assert events.first().data["pages_assembled"] == 2
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_handles_missing_page_pdfs(self, mock_chord):
        mock_chord.return_value = MagicMock()
        # Create job with 3 pages but only provide PDFs for pages 1 and 3
        job, tmpdir, _ = self._make_job_with_pages(total_pages=3, pages_ok=[1, 3])
        try:
            result = assemble_document.run(str(job.job_id))
            assert result["pages_assembled"] == 2  # only 2 pages had PDFs
        finally:
            self._cleanup(tmpdir)

    def test_missing_job_returns_error(self):
        fake_id = str(uuid.uuid4())
        result = assemble_document.run(fake_id)
        assert result["status"] == "error"

    def test_cancelled_job_returns_cancelled(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.CANCELLED,
        )
        result = assemble_document.run(str(job.job_id))
        assert result["status"] == "cancelled"

    @patch("jobs.tasks.chord")
    def test_finalizes_custody_chain(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, doc_id = self._make_job_with_pages(total_pages=1)
        # Add a pre-existing custody event
        _record_custody_event(job, doc_id, "file_ingested",
                              data={"hash": "abc"})
        try:
            assemble_document.run(str(job.job_id))
            # Verify events have been finalized (hash chain computed)
            finalized = CustodyEvent.objects.filter(
                job=job, chain_finalized=True
            )
            assert finalized.count() > 0
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_exports_custody_jsonl(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, doc_id = self._make_job_with_pages(total_pages=1)
        try:
            assemble_document.run(str(job.job_id))
            jsonl_path = os.path.join(
                tmpdir, "output", "EXPORT", "CUSTODY",
                f"{doc_id}.custody.jsonl"
            )
            assert os.path.isfile(jsonl_path)
        finally:
            self._cleanup(tmpdir)


# ---------------------------------------------------------------------------
# Step 5e: _process_single_page tests
# ---------------------------------------------------------------------------

def _valid_png_bytes():
    """Create a valid minimal PNG in memory for fitz insert_image."""
    import io

    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.new("RGB", (10, 10), "white").save(buf, format="PNG")
    return buf.getvalue()


def _make_ocr_utils_mock(**overrides):
    """Create a mock ocr_distributed.ocr_utils module for _process_single_page tests.

    Since ocr_distributed is not on sys.path from the coordinator/ directory,
    we inject a mock module via sys.modules before calling _process_single_page.
    """
    mock_mod = MagicMock()
    mock_mod.iter_source_images = MagicMock(return_value=[])
    mock_mod.create_paddle_engine = MagicMock()
    mock_mod.extract_paddle_lines = MagicMock(return_value=[])
    mock_mod.img_to_bytes = MagicMock(return_value=_valid_png_bytes())
    mock_mod.insert_text_line = MagicMock()
    mock_mod._resolve_text_font = MagicMock(return_value=("helv", None))
    for key, val in overrides.items():
        setattr(mock_mod, key, val)
    return mock_mod


class TestProcessSinglePage(TestCase):
    """Tests for _process_single_page helper function.

    _process_single_page does local imports from ocr_distributed.ocr_utils
    and fitz, which may not be on sys.path or may segfault on repeated import
    in test environments. We inject mock modules via sys.modules patches.
    """

    def _make_job_with_nfs(self):
        tmpdir = tempfile.mkdtemp()
        _ensure_job_dirs(tmpdir)
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=5,
            source_hash="abcdef1234567890" * 4,
            nfs_job_path=tmpdir,
            detected_language="en",
            source_type="pdf",
        )
        return job, tmpdir

    def _cleanup(self, tmpdir):
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def _make_fitz_mock(self, tmpdir):
        """Create a fitz mock that writes real temp PDFs via PIL."""
        mock_fitz = MagicMock()

        # Mock fitz.Rect
        mock_fitz.Rect = MagicMock(return_value=MagicMock())

        # Mock fitz.open() to return a document mock that writes a real file
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.new_page.return_value = mock_page
        mock_doc.page_count = 1

        def fake_save(path, **kwargs):
            # Write a minimal valid PDF
            with open(path, "wb") as f:
                f.write(b"%PDF-1.0\n1 0 obj<</Type/Catalog>>endobj\n%%EOF\n")

        mock_doc.save = fake_save
        mock_fitz.open.return_value = mock_doc
        return mock_fitz

    def _ocr_mods(self, mock_utils, mock_fitz):
        """Return a sys.modules dict patch for ocr_distributed and fitz."""
        mock_pkg = MagicMock()
        mock_pkg.ocr_utils = mock_utils
        mock_pytesseract = types.SimpleNamespace(
            image_to_string=MagicMock(return_value=""),
        )
        return {
            "ocr_distributed": mock_pkg,
            "ocr_distributed.ocr_utils": mock_utils,
            "fitz": mock_fitz,
            "pytesseract": mock_pytesseract,
        }

    def test_tesseract_fallback(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                return_value=MagicMock(predict=MagicMock(return_value=None))
            ),
            extract_paddle_lines=MagicMock(return_value=[]),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        try:
            with patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)), \
                 patch("jobs.tasks._gpu_available", return_value=False), \
                 patch("pytesseract.image_to_string", return_value="Hello Tess"):
                result = _process_single_page(
                    job, "/test.pdf", 1, job.source_hash[:16], "worker-01",
                )
            assert result["method"] == "Tesseract"
            assert result["status"] == "fallback"
            pr = PageResult.objects.get(job=job, page_num=1)
            assert pr.ocr_method == "Tesseract"
            assert pr.status == "fallback"
        finally:
            self._cleanup(tmpdir)

    def test_image_only_fallback(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                return_value=MagicMock(predict=MagicMock(return_value=None))
            ),
            extract_paddle_lines=MagicMock(return_value=[]),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        try:
            with patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)), \
                 patch("jobs.tasks._gpu_available", return_value=False), \
                 patch("pytesseract.image_to_string",
                       side_effect=RuntimeError("No tesseract")):
                result = _process_single_page(
                    job, "/test.pdf", 1, job.source_hash[:16], "worker-01",
                )
            assert result["method"] == "ImageOnly"
            assert result["status"] == "image_only"
            pr = PageResult.objects.get(job=job, page_num=1)
            assert pr.ocr_method == "ImageOnly"
            assert pr.status == "image_only"
        finally:
            self._cleanup(tmpdir)

    def test_paddle_success(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                return_value=MagicMock(predict=MagicMock(return_value="result"))
            ),
            extract_paddle_lines=MagicMock(return_value=[
                ("Hello world", [[0, 0], [100, 0], [100, 20], [0, 20]], 0.95),
            ]),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        try:
            with patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)), \
                 patch("jobs.tasks._gpu_available", return_value=False):
                result = _process_single_page(
                    job, "/test.pdf", 1, job.source_hash[:16], "worker-01",
                )
            assert result["method"] == "PaddleOCR"
            assert result["status"] == "ok"
            assert result["confidence"] == 0.95
            assert result["text_length"] == len("Hello world")
        finally:
            self._cleanup(tmpdir)

    def test_no_images_records_failed(self):
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[]),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        try:
            with patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)):
                result = _process_single_page(
                    job, "/test.pdf", 1, job.source_hash[:16], "worker-01",
                )
            assert result["status"] == "failed"
            pr = PageResult.objects.get(job=job, page_num=1)
            assert pr.status == "failed"
        finally:
            self._cleanup(tmpdir)

    def test_s3_owned_temp_dir_is_cleaned_when_no_images(self):
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[]),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        backend = MagicMock()
        backend.backend_name = "s3"

        owned_temp_dir = os.path.join(tempfile.gettempdir(), f"ocr_page_{uuid.uuid4().hex}")
        os.makedirs(owned_temp_dir, exist_ok=True)
        try:
            with patch("tempfile.mkdtemp", return_value=owned_temp_dir), \
                 patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)):
                result = _process_single_page(
                    job,
                    "/test.pdf",
                    1,
                    job.source_hash[:16],
                    "worker-01",
                    backend=backend,
                    storage_mode="s3",
                )
            assert result["status"] == "failed"
            assert not os.path.isdir(owned_temp_dir)
        finally:
            self._cleanup(tmpdir)
            shutil.rmtree(owned_temp_dir, ignore_errors=True)

    def test_records_custody_event(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                return_value=MagicMock(predict=MagicMock(return_value="r"))
            ),
            extract_paddle_lines=MagicMock(return_value=[
                ("text", [[0, 0], [10, 0], [10, 10], [0, 10]], 0.8),
            ]),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        doc_id = job.source_hash[:16]
        try:
            with patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)), \
                 patch("jobs.tasks._gpu_available", return_value=False):
                _process_single_page(job, "/test.pdf", 1, doc_id, "w1")
            events = CustodyEvent.objects.filter(job=job, event_type="ocr_primary")
            assert events.count() == 1
        finally:
            self._cleanup(tmpdir)

    def test_writes_temp_files(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                return_value=MagicMock(predict=MagicMock(return_value=None))
            ),
            extract_paddle_lines=MagicMock(return_value=[]),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        doc_id = job.source_hash[:16]
        try:
            with patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)), \
                 patch("jobs.tasks._gpu_available", return_value=False), \
                 patch("pytesseract.image_to_string", return_value="test text"):
                _process_single_page(job, "/test.pdf", 1, doc_id, "w1")
            temp_dir = os.path.join(tmpdir, "temp", doc_id)
            assert os.path.isfile(os.path.join(temp_dir, "1.pdf"))
            assert os.path.isfile(os.path.join(temp_dir, "1.txt"))
        finally:
            self._cleanup(tmpdir)

    def test_records_processing_time(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                return_value=MagicMock(predict=MagicMock(return_value="r"))
            ),
            extract_paddle_lines=MagicMock(return_value=[
                ("text", [[0, 0], [10, 0], [10, 10], [0, 10]], 0.9),
            ]),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        try:
            with patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)), \
                 patch("jobs.tasks._gpu_available", return_value=False):
                result = _process_single_page(
                    job, "/test.pdf", 1, job.source_hash[:16], "w1",
                )
            assert result["elapsed_ms"] >= 0
            pr = PageResult.objects.get(job=job, page_num=1)
            assert pr.processing_time_ms >= 0
        finally:
            self._cleanup(tmpdir)

    def test_confidence_averaging(self):
        from PIL import Image
        img = Image.new("RGB", (200, 200), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                return_value=MagicMock(predict=MagicMock(return_value="r"))
            ),
            extract_paddle_lines=MagicMock(return_value=[
                ("line1", [[0, 0], [100, 0], [100, 10], [0, 10]], 0.8),
                ("line2", [[0, 20], [100, 20], [100, 30], [0, 30]], 0.6),
            ]),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        try:
            with patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)), \
                 patch("jobs.tasks._gpu_available", return_value=False):
                result = _process_single_page(
                    job, "/test.pdf", 1, job.source_hash[:16], "w1",
                )
            assert result["confidence"] == (0.8 + 0.6) / 2
        finally:
            self._cleanup(tmpdir)

    def test_custody_event_type_for_fallback(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                return_value=MagicMock(predict=MagicMock(return_value=None))
            ),
            extract_paddle_lines=MagicMock(return_value=[]),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        try:
            with patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)), \
                 patch("jobs.tasks._gpu_available", return_value=False), \
                 patch("pytesseract.image_to_string", return_value="fallback"):
                _process_single_page(
                    job, "/test.pdf", 1, job.source_hash[:16], "w1",
                )
            events = CustodyEvent.objects.filter(job=job, event_type="ocr_fallback")
            assert events.count() == 1
        finally:
            self._cleanup(tmpdir)


# ---------------------------------------------------------------------------
# Step 5f: chord_error_handler tests
# ---------------------------------------------------------------------------

class TestChordErrorHandler(TestCase):
    """Tests for chord_error_handler task."""

    def _make_job(self):
        return Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=10,
            source_hash="abcdef1234567890" * 4,
        )

    def test_sets_job_failed(self):
        job = self._make_job()
        mock_request = MagicMock()
        mock_request.id = "task-123"
        chord_error_handler.run(
            mock_request, RuntimeError("page 5 failed"), None, str(job.job_id)
        )
        job.refresh_from_db()
        assert job.status == Job.Status.FAILED

    def test_preserves_error_message(self):
        job = self._make_job()
        mock_request = MagicMock()
        mock_request.id = "task-456"
        chord_error_handler.run(
            mock_request, ValueError("Bad data on page 3"), None, str(job.job_id)
        )
        job.refresh_from_db()
        assert "Bad data on page 3" in job.error_message

    def test_records_custody_event(self):
        job = self._make_job()
        mock_request = MagicMock()
        mock_request.id = "task-789"
        chord_error_handler.run(
            mock_request, RuntimeError("engine crash"), None, str(job.job_id)
        )
        events = CustodyEvent.objects.filter(job=job, event_type="chord_failed")
        assert events.count() == 1
        assert "engine crash" in events.first().data["error"]

    def test_missing_job_returns_error(self):
        fake_id = str(uuid.uuid4())
        mock_request = MagicMock()
        mock_request.id = "task-000"
        result = chord_error_handler.run(
            mock_request, RuntimeError("fail"), None, fake_id
        )
        assert result["status"] == "error"

    def test_handles_various_exception_types(self):
        job = self._make_job()
        mock_request = MagicMock()
        mock_request.id = "task-x"
        for exc_type in [RuntimeError, ValueError, OSError, TypeError]:
            exc = exc_type(f"Error from {exc_type.__name__}")
            chord_error_handler.run(mock_request, exc, None, str(job.job_id))
            job.refresh_from_db()
            assert job.status == Job.Status.FAILED
            # Reset for next iteration
            job.status = Job.Status.PROCESSING
            job.save(update_fields=["status"])


# ---------------------------------------------------------------------------
# Step 5g: Worker task counter tests
# ---------------------------------------------------------------------------

class TestWorkerTaskCounters(TestCase):
    """Tests for M3 worker task counter increments."""

    def test_process_document_increments_completed(self):
        """process_document increments tasks_completed on success."""
        worker = Worker.objects.create(
            hostname=socket.gethostname(),
            status=Worker.Status.ONLINE,
            tasks_completed=0,
        )
        job = Job.objects.create(
            source_file="test.pdf",
            source_hash="abc123",
            total_pages=1,
            nfs_job_path=tempfile.mkdtemp(),
        )
        _ensure_job_dirs(job.nfs_job_path)

        with patch("jobs.tasks._process_single_page"):
            with patch("jobs.tasks.assemble_document") as mock_assemble:
                mock_assemble.delay = MagicMock()
                process_document.run(str(job.job_id))

        worker.refresh_from_db()
        assert worker.tasks_completed == 1
        assert worker.tasks_failed == 0
        self._cleanup(job.nfs_job_path)

    def test_process_document_increments_failed_on_error(self):
        """process_document increments tasks_failed on exception (after retries exhausted)."""
        worker = Worker.objects.create(
            hostname=socket.gethostname(),
            status=Worker.Status.ONLINE,
            tasks_failed=0,
        )
        job = Job.objects.create(
            source_file="test.pdf",
            source_hash="abc123",
            total_pages=1,
            nfs_job_path=tempfile.mkdtemp(),
        )
        _ensure_job_dirs(job.nfs_job_path)

        with patch("jobs.tasks._process_single_page", side_effect=RuntimeError("OCR crash")):
            with patch.object(
                process_document, "retry",
                side_effect=MaxRetriesExceededError(),
            ):
                with patch.object(
                    type(process_document.request), "retries",
                    new_callable=PropertyMock, return_value=3,
                ):
                    result = process_document.run(str(job.job_id))

        worker.refresh_from_db()
        assert worker.tasks_failed == 1
        assert result["status"] == "error"
        self._cleanup(job.nfs_job_path)

    def test_process_page_increments_completed(self):
        """process_page increments tasks_completed on success."""
        worker = Worker.objects.create(
            hostname=socket.gethostname(),
            status=Worker.Status.ONLINE,
            tasks_completed=0,
        )
        job = Job.objects.create(
            source_file="test.pdf",
            source_hash="abc123",
            total_pages=5,
            nfs_job_path=tempfile.mkdtemp(),
        )
        _ensure_job_dirs(job.nfs_job_path)

        with patch("jobs.tasks._process_single_page"):
            result = process_page.run(str(job.job_id), 1)

        worker.refresh_from_db()
        assert worker.tasks_completed == 1
        assert result["status"] == "page_processed"
        self._cleanup(job.nfs_job_path)

    def test_process_page_increments_failed_on_error(self):
        """process_page increments tasks_failed on exception."""
        worker = Worker.objects.create(
            hostname=socket.gethostname(),
            status=Worker.Status.ONLINE,
            tasks_failed=0,
        )
        job = Job.objects.create(
            source_file="test.pdf",
            source_hash="abc123",
            total_pages=5,
            nfs_job_path=tempfile.mkdtemp(),
        )
        _ensure_job_dirs(job.nfs_job_path)
        # Create PageResult row for the page
        PageResult.objects.create(
            job=job, page_num=1, document_id="abc123abc123abc1",
        )

        with patch("jobs.tasks._process_single_page", side_effect=RuntimeError("fail")):
            with self.assertRaises(RuntimeError):
                process_page.run(str(job.job_id), 1)

        worker.refresh_from_db()
        assert worker.tasks_failed == 1
        self._cleanup(job.nfs_job_path)

    def test_compress_pdf_increments_completed(self):
        """compress_pdf increments tasks_completed on success."""
        worker = Worker.objects.create(
            hostname=socket.gethostname(),
            status=Worker.Status.ONLINE,
            tasks_completed=0,
        )
        job = Job.objects.create(
            source_file="test.pdf",
            source_hash="abc123",
            nfs_job_path=tempfile.mkdtemp(),
        )
        _ensure_job_dirs(job.nfs_job_path)
        # Create a dummy PDF file
        pdf_path = os.path.join(job.nfs_job_path, "output", "EXPORT", "PDF", "test.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.0 test content")

        mock_optimize_mod = MagicMock()
        with patch.dict("sys.modules", {"optimize_pdfs": mock_optimize_mod}):
            result = compress_pdf.run(str(job.job_id))

        worker.refresh_from_db()
        assert worker.tasks_completed == 1
        assert result["status"] == "compressed"
        self._cleanup(job.nfs_job_path)

    def test_extract_entities_increments_completed(self):
        """extract_entities increments tasks_completed on success."""
        worker = Worker.objects.create(
            hostname=socket.gethostname(),
            status=Worker.Status.ONLINE,
            tasks_completed=0,
        )
        job = Job.objects.create(
            source_file="test.pdf",
            source_hash="abc123",
            nfs_job_path=tempfile.mkdtemp(),
        )
        _ensure_job_dirs(job.nfs_job_path)
        # Create a dummy text file
        text_path = os.path.join(job.nfs_job_path, "output", "EXPORT", "TEXT", "test.txt")
        with open(text_path, "w") as f:
            f.write("Sample text for NER testing")

        mock_ner_mod = MagicMock()
        mock_ner_mod.extract_entities.return_value = [{"entity": "test", "type": "PERSON"}]
        with patch.dict("sys.modules", {"ner": mock_ner_mod}):
            result = extract_entities.run(str(job.job_id))

        worker.refresh_from_db()
        assert worker.tasks_completed == 1
        assert result["status"] == "extracted"
        self._cleanup(job.nfs_job_path)

    def _cleanup(self, path):
        try:
            shutil.rmtree(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Step 6: cleanup_completed_jobs periodic task tests
# ---------------------------------------------------------------------------

class TestCleanupCompletedJobs(TestCase):
    """Tests for cleanup_completed_jobs periodic task (Phase M5)."""

    def test_deletes_old_completed_jobs(self):
        """Old completed/failed/cancelled jobs are deleted after retention period."""
        j1 = Job.objects.create(source_file="/old1.pdf", status=Job.Status.COMPLETED)
        j2 = Job.objects.create(source_file="/old2.pdf", status=Job.Status.FAILED)
        old_date = timezone.now() - timezone.timedelta(days=60)
        Job.objects.filter(
            job_id__in=[j1.job_id, j2.job_id]
        ).update(created_at=old_date)

        result = cleanup_completed_jobs()

        assert result["deleted"] == 2
        assert Job.objects.filter(job_id=j1.job_id).count() == 0
        assert Job.objects.filter(job_id=j2.job_id).count() == 0

    def test_preserves_recent_jobs(self):
        """Recent completed jobs within retention period are not deleted."""
        job = Job.objects.create(
            source_file="/recent.pdf", status=Job.Status.COMPLETED
        )
        # created_at is auto_now_add (set to now), within 30-day retention

        result = cleanup_completed_jobs()

        assert result["deleted"] == 0
        assert Job.objects.filter(job_id=job.job_id).exists()

    def test_returns_count(self):
        """Task returns a dict with 'deleted' and 'nfs_cleaned' counts."""
        j1 = Job.objects.create(
            source_file="/c1.pdf", status=Job.Status.COMPLETED
        )
        j2 = Job.objects.create(
            source_file="/c2.pdf", status=Job.Status.CANCELLED
        )
        old_date = timezone.now() - timezone.timedelta(days=45)
        Job.objects.filter(
            job_id__in=[j1.job_id, j2.job_id]
        ).update(created_at=old_date)

        result = cleanup_completed_jobs()

        assert "deleted" in result
        assert "nfs_cleaned" in result
        assert result["deleted"] == 2
        assert result["nfs_cleaned"] == 0  # no nfs_job_path set


# ---------------------------------------------------------------------------
# Phase 7: Storage Backend Integration Tests
# ---------------------------------------------------------------------------

class TestStorageBackendHelpers(TestCase):
    """Tests for storage backend helper functions."""

    def setUp(self):
        _reset_storage_backend()

    def tearDown(self):
        _reset_storage_backend()

    def test_get_storage_backend_nfs_default(self):
        backend = _get_storage_backend()
        assert backend.backend_name == "nfs"
        assert isinstance(backend, NFSBackend)

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._create_storage_backend")
    def test_get_storage_backend_s3_uses_factory(self, mock_create_backend):
        mock_backend = MagicMock()
        mock_backend.backend_name = "s3"
        mock_create_backend.return_value = mock_backend
        backend = _get_storage_backend()
        assert backend.backend_name == "s3"
        mock_create_backend.assert_called_once()

    def test_singleton_returns_same_instance(self):
        """_get_storage_backend() returns the same object on repeated calls."""
        backend1 = _get_storage_backend()
        backend2 = _get_storage_backend()
        assert backend1 is backend2

    @patch("jobs.tasks._create_storage_backend")
    def test_singleton_creates_backend_only_once(self, mock_create):
        """Factory is called exactly once even with many callers."""
        mock_create.return_value = MagicMock(backend_name="nfs")
        for _ in range(10):
            _get_storage_backend()
        mock_create.assert_called_once()

    def test_singleton_thread_safety(self):
        """Concurrent threads all receive the same backend instance."""
        import threading as _threading

        results = []
        barrier = _threading.Barrier(8)

        def worker():
            barrier.wait()
            results.append(_get_storage_backend())

        threads = [_threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 8
        assert all(r is results[0] for r in results)

    def test_reset_clears_singleton(self):
        """_reset_storage_backend() allows a new instance on the next call."""
        backend1 = _get_storage_backend()
        _reset_storage_backend()
        backend2 = _get_storage_backend()
        assert backend1 is not backend2

    def test_create_storage_backend_returns_new_each_time(self):
        """_create_storage_backend() always returns a fresh instance."""
        b1 = _create_storage_backend()
        b2 = _create_storage_backend()
        assert b1 is not b2

    def test_job_storage_key_basic(self):
        job_id = uuid.uuid4()
        key = _job_storage_key(job_id)
        assert key == f"jobs/{job_id}"

    def test_job_storage_key_with_subpath(self):
        job_id = uuid.uuid4()
        key = _job_storage_key(job_id, "output/EXPORT/PDF/doc.pdf")
        assert key == f"jobs/{job_id}/output/EXPORT/PDF/doc.pdf"


class TestCleanupCompletedJobsS3(TestCase):
    """Tests for cleanup_completed_jobs with S3 backend."""

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._get_storage_backend")
    def test_deletes_s3_objects_for_old_jobs(self, mock_get_storage_backend):
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.list_objects.return_value = [
            "jobs/key-1/source/file.pdf",
            "jobs/key-1/output/EXPORT/PDF/result.pdf",
        ]
        # delete_many returns count of successfully deleted objects
        backend.delete_many.return_value = 2
        mock_get_storage_backend.return_value = backend

        # Create old jobs
        j1 = Job.objects.create(
            source_file="/test1.pdf",
            status=Job.Status.COMPLETED,
        )
        j2 = Job.objects.create(
            source_file="/test2.pdf",
            status=Job.Status.FAILED,
        )
        old_date = timezone.now() - timezone.timedelta(days=45)
        Job.objects.filter(
            job_id__in=[j1.job_id, j2.job_id]
        ).update(created_at=old_date)

        result = cleanup_completed_jobs()

        assert result["deleted"] == 2
        assert result["nfs_cleaned"] == 0
        assert result["s3_jobs_cleaned"] == 2
        assert result["s3_objects_cleaned"] == 4
        assert backend.list_objects.call_count == 2
        assert backend.delete_many.call_count == 2

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_cleanup_handles_exceptions(self, mock_get_storage_backend):
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.list_objects.side_effect = RuntimeError("S3 connection error")
        mock_get_storage_backend.return_value = backend

        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.COMPLETED,
        )
        old_date = timezone.now() - timezone.timedelta(days=45)
        Job.objects.filter(job_id=job.job_id).update(created_at=old_date)

        # Should not raise, just log error
        result = cleanup_completed_jobs()
        assert result["deleted"] == 1


class TestGetOcrQueue(TestCase):
    """Tests for _get_ocr_queue() routing helper."""

    def test_default_returns_ocr_gpu(self):
        """Default routing (no env var) returns ocr_gpu."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OCR_TASK_ROUTING", None)
            from jobs.tasks import _get_ocr_queue

            assert _get_ocr_queue() == "ocr_gpu"

    def test_gpu_mode_returns_ocr_gpu(self):
        """Explicit gpu routing returns ocr_gpu."""
        with patch.dict(os.environ, {"OCR_TASK_ROUTING": "gpu"}):
            from jobs.tasks import _get_ocr_queue

            assert _get_ocr_queue() == "ocr_gpu"

    def test_cpu_mode_returns_ocr_cpu(self):
        """Explicit cpu routing returns ocr_cpu."""
        with patch.dict(os.environ, {"OCR_TASK_ROUTING": "cpu"}):
            from jobs.tasks import _get_ocr_queue

            assert _get_ocr_queue() == "ocr_cpu"

    def test_auto_with_gpu_workers_returns_ocr_gpu(self):
        """Auto routing with online GPU workers returns ocr_gpu."""
        Worker.objects.create(
            hostname="gpu-1",
            status=Worker.Status.ONLINE,
            gpu_available=True,
            gpu_model="RTX 4090",
        )
        with patch.dict(os.environ, {"OCR_TASK_ROUTING": "auto"}):
            from jobs.tasks import _get_ocr_queue

            assert _get_ocr_queue() == "ocr_gpu"

    def test_auto_without_gpu_workers_returns_ocr_cpu(self):
        """Auto routing with no GPU workers returns ocr_cpu."""
        Worker.objects.create(
            hostname="cpu-1",
            status=Worker.Status.ONLINE,
            gpu_available=False,
        )
        with patch.dict(os.environ, {"OCR_TASK_ROUTING": "auto"}):
            from jobs.tasks import _get_ocr_queue

            assert _get_ocr_queue() == "ocr_cpu"

    def test_auto_with_offline_gpu_returns_ocr_cpu(self):
        """Auto routing with only offline GPU workers returns ocr_cpu."""
        Worker.objects.create(
            hostname="gpu-1",
            status=Worker.Status.OFFLINE,
            gpu_available=True,
            gpu_model="RTX 4090",
        )
        with patch.dict(os.environ, {"OCR_TASK_ROUTING": "auto"}):
            from jobs.tasks import _get_ocr_queue

            assert _get_ocr_queue() == "ocr_cpu"

    def test_case_insensitive(self):
        """Routing env var is case-insensitive."""
        with patch.dict(os.environ, {"OCR_TASK_ROUTING": "CPU"}):
            from jobs.tasks import _get_ocr_queue

            assert _get_ocr_queue() == "ocr_cpu"

    def test_unknown_value_defaults_to_gpu(self):
        """Unknown routing value defaults to ocr_gpu."""
        with patch.dict(os.environ, {"OCR_TASK_ROUTING": "unknown"}):
            from jobs.tasks import _get_ocr_queue

            assert _get_ocr_queue() == "ocr_gpu"


class TestIngestDocumentS3(TestCase):
    """Tests for ingest_document with S3 backend."""

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks.process_document.apply_async")
    @patch("jobs.tasks.extract_pages.apply_async")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_mode_uploads_source_file(
        self,
        mock_get_storage_backend,
        mock_extract_pages_apply,
        mock_process_document_apply,
    ):
        """S3 mode should upload source file and dispatch process_document for small docs."""
        backend = MagicMock()
        backend.backend_name = "s3"
        mock_get_storage_backend.return_value = backend

        fake_pkg = types.ModuleType("ocr_distributed")
        fake_ocr_utils = types.ModuleType("ocr_distributed.ocr_utils")
        fake_language = types.ModuleType("ocr_distributed.language")

        def _classify_source_file(_path, **_kw):
            return "pdf", None

        def _get_source_page_count(_path, _source_type):
            return 1

        class _FakeLanguageDetector:
            def __init__(self, _model_path):
                pass

            def detect_from_pdf(self, _source_path):
                return "en"

        fake_ocr_utils.classify_source_file = _classify_source_file
        fake_ocr_utils.get_source_page_count = _get_source_page_count
        fake_language.LanguageDetector = _FakeLanguageDetector
        fake_pkg.ocr_utils = fake_ocr_utils
        fake_pkg.language = fake_language

        with tempfile.TemporaryDirectory() as nfs_root:
            source_dir = os.path.join(nfs_root, "uploads")
            os.makedirs(source_dir, exist_ok=True)
            test_file = os.path.join(source_dir, "test.pdf")
            with open(test_file, "wb") as f:
                f.write(b"%PDF-1.4 test content")

            with override_settings(NFS_ROOT=nfs_root):
                job = Job.objects.create(
                    source_file=test_file,
                    status=Job.Status.SUBMITTED,
                )
                with patch.dict(
                    sys.modules,
                    {
                        "ocr_distributed": fake_pkg,
                        "ocr_distributed.ocr_utils": fake_ocr_utils,
                        "ocr_distributed.language": fake_language,
                    },
                ):
                    ingest_document.run(str(job.job_id))

                # Verify upload was called
                backend.upload_file.assert_called_once()
                # Small doc (1 page) should dispatch process_document, not fan-out
                mock_process_document_apply.assert_called_once_with(
                    args=[str(job.job_id)], queue="ocr_gpu", priority=5,
                )
                mock_extract_pages_apply.assert_not_called()
                job.refresh_from_db()
                assert job.status == Job.Status.PROCESSING

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks.process_document.apply_async")
    @patch("jobs.tasks.extract_pages.apply_async")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_mode_dispatches_fanout_for_large_docs(
        self,
        mock_get_storage_backend,
        mock_extract_pages_apply,
        mock_process_document_apply,
    ):
        """S3 mode should fan out when source page count exceeds threshold."""
        backend = MagicMock()
        backend.backend_name = "s3"
        mock_get_storage_backend.return_value = backend

        fake_pkg = types.ModuleType("ocr_distributed")
        fake_ocr_utils = types.ModuleType("ocr_distributed.ocr_utils")
        fake_language = types.ModuleType("ocr_distributed.language")

        def _classify_source_file(_path, **_kw):
            return "pdf", None

        def _get_source_page_count(_path, _source_type):
            return FANOUT_THRESHOLD + 5

        class _FakeLanguageDetector:
            def __init__(self, _model_path):
                pass

            def detect_from_pdf(self, _source_path):
                return "en"

        fake_ocr_utils.classify_source_file = _classify_source_file
        fake_ocr_utils.get_source_page_count = _get_source_page_count
        fake_language.LanguageDetector = _FakeLanguageDetector
        fake_pkg.ocr_utils = fake_ocr_utils
        fake_pkg.language = fake_language

        with tempfile.TemporaryDirectory() as nfs_root:
            source_dir = os.path.join(nfs_root, "uploads")
            os.makedirs(source_dir, exist_ok=True)
            test_file = os.path.join(source_dir, "large.pdf")
            with open(test_file, "wb") as f:
                f.write(b"%PDF-1.4 large test content")

            with override_settings(NFS_ROOT=nfs_root):
                job = Job.objects.create(
                    source_file=test_file,
                    status=Job.Status.SUBMITTED,
                )
                with patch.dict(
                    sys.modules,
                    {
                        "ocr_distributed": fake_pkg,
                        "ocr_distributed.ocr_utils": fake_ocr_utils,
                        "ocr_distributed.language": fake_language,
                    },
                ):
                    result = ingest_document.run(str(job.job_id))

                backend.upload_file.assert_called_once()
                mock_extract_pages_apply.assert_called_once_with(
                    args=[str(job.job_id)], priority=5,
                )
                mock_process_document_apply.assert_not_called()
                assert result["mode"] == "fanout"
                job.refresh_from_db()
                assert job.status == Job.Status.PROCESSING

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks.process_document.apply_async")
    @patch("jobs.tasks.extract_pages.delay")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_mode_rejects_source_outside_allowed_root(
        self,
        mock_get_storage_backend,
        mock_extract_pages_delay,
        mock_process_document_apply,
    ):
        backend = MagicMock()
        backend.backend_name = "s3"
        mock_get_storage_backend.return_value = backend

        with tempfile.TemporaryDirectory() as allowed_root, tempfile.TemporaryDirectory() as other_root:
            outside_file = os.path.join(other_root, "outside.pdf")
            with open(outside_file, "wb") as f:
                f.write(b"%PDF-1.4 outside content")

            with override_settings(NFS_ROOT=allowed_root):
                job = Job.objects.create(
                    source_file=outside_file,
                    status=Job.Status.SUBMITTED,
                )
                result = ingest_document.run(str(job.job_id))

            assert result["status"] == "error"
            assert "outside allowed directory" in result["message"]
            backend.upload_file.assert_not_called()
            mock_extract_pages_delay.assert_not_called()
            mock_process_document_apply.assert_not_called()


class TestProcessPageS3(TestCase):
    """Tests for process_page with S3 backend."""

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._get_storage_backend")
    @patch("jobs.tasks._process_single_page")
    def test_s3_mode_downloads_source_before_processing(self, mock_process, mock_get_storage_backend):
        """S3 mode should download source file before processing."""
        backend = MagicMock()
        backend.backend_name = "s3"
        mock_get_storage_backend.return_value = backend
        mock_process.return_value = {"page_num": 1, "status": "ok"}

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=5,
            source_hash="abcdef1234567890" * 4,
            detected_language="en",
        )

        process_page.run(str(job.job_id), 1)

        # Verify download was called
        backend.download_file.assert_called_once()
        # Verify processing was called
        mock_process.assert_called_once()

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_download_failure_marks_page_failed(self, mock_get_storage_backend):
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.download_file.side_effect = RuntimeError("download failed")
        mock_get_storage_backend.return_value = backend

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
            source_hash="abcdef1234567890" * 4,
        )

        try:
            process_page.run(str(job.job_id), 1)
            assert False, "Expected RuntimeError"
        except RuntimeError:
            page = PageResult.objects.get(job=job, page_num=1)
            assert page.status == "failed"


class TestAssembleDocumentS3(TestCase):
    """Tests for assemble_document with S3 backend."""

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._finalize_custody_chain_s3")
    @patch("jobs.tasks.chord")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_mode_uploads_final_artifacts(
        self, mock_get_storage_backend, mock_chord, mock_finalize_custody_s3
    ):
        """S3 mode should upload final PDF and text to backend and dispatch post-processing."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = False
        mock_get_storage_backend.return_value = backend
        
        # Mock chord to return a callable
        mock_chord_instance = MagicMock()
        mock_chord.return_value = mock_chord_instance

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=0,
            source_hash="abcdef1234567890" * 4,
        )

        assemble_document.run(str(job.job_id))

        # Verify upload was called for text output in S3 mode.
        backend.upload_file.assert_called()
        # S3 now dispatches post-processing chord like NFS
        mock_chord.assert_called_once()
        mock_chord_instance.assert_called_once()

    @override_settings(STORAGE_BACKEND="s3")
    @patch("tempfile.mkdtemp")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_mode_cleans_temp_dir_when_upload_fails(self, mock_get_storage_backend, mock_mkdtemp):
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = False
        backend.upload_file.side_effect = RuntimeError("upload failed")
        mock_get_storage_backend.return_value = backend

        work_dir = os.path.join(tempfile.gettempdir(), f"assemble_fail_{uuid.uuid4().hex}")
        os.makedirs(work_dir, exist_ok=True)
        mock_mkdtemp.return_value = work_dir

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=0,
            source_hash="abcdef1234567890" * 4,
        )

        try:
            with self.assertRaises(RuntimeError):
                assemble_document.run(str(job.job_id))
            assert not os.path.isdir(work_dir)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


class TestProcessDocumentS3(TestCase):
    """Tests for process_document with S3 backend."""

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks.assemble_document.delay")
    @patch("jobs.tasks._process_single_page")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_mode_downloads_source_and_cleans_up(
        self, mock_get_storage_backend, mock_process_single_page, mock_assemble_delay
    ):
        """S3 mode should download source, process, and clean up temp dir."""
        backend = MagicMock()
        backend.backend_name = "s3"
        mock_get_storage_backend.return_value = backend
        mock_process_single_page.return_value = {"status": "ok"}

        def _download_source(_key, local_path):
            with open(local_path, "wb") as f:
                f.write(b"%PDF-1.0 source")

        backend.download_file.side_effect = _download_source

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=2,
            source_hash="abcdef1234567890" * 4,
            detected_language="en",
        )

        process_document.run(str(job.job_id))

        # Verify source download was called
        backend.download_file.assert_called_once()
        source_path = backend.download_file.call_args[0][1]
        assert not os.path.isdir(os.path.dirname(source_path))
        # Verify processing was called for each page
        assert mock_process_single_page.call_count == 2
        # Verify assembly was dispatched
        mock_assemble_delay.assert_called_once_with(str(job.job_id))

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_download_failure_returns_error_and_cleans_up(self, mock_get_storage_backend):
        """S3 download failure should return error and clean up temp dir (after retries)."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.download_file.side_effect = RuntimeError("S3 download failed")
        mock_get_storage_backend.return_value = backend

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=2,
            source_hash="abcdef1234567890" * 4,
        )

        # Simulate max retries exhausted so the task returns error
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
        # Verify download was attempted
        backend.download_file.assert_called_once()
        source_path = backend.download_file.call_args[0][1]
        assert not os.path.isdir(os.path.dirname(source_path))


class TestCompressPdfS3(TestCase):
    """Tests for compress_pdf with S3 backend."""

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_mode_downloads_optimizes_uploads_and_cleans_up(
        self, mock_get_storage_backend
    ):
        """S3 mode should download PDF, optimize, upload, and clean up."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = True
        mock_get_storage_backend.return_value = backend

        def _download_pdf(_key, local_path):
            with open(local_path, "wb") as f:
                f.write(b"%PDF-1.0 test content")

        backend.download_file.side_effect = _download_pdf

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
            source_hash="abcdef1234567890" * 4,
        )

        mock_optimize_mod = types.ModuleType("optimize_pdfs")
        mock_optimize_mod.optimize_pdf = MagicMock()
        with patch.dict(sys.modules, {"optimize_pdfs": mock_optimize_mod}):
            result = compress_pdf.run(str(job.job_id))

        assert result["status"] == "compressed"
        # Verify exists check
        backend.exists.assert_called_once()
        # Verify download
        backend.download_file.assert_called_once()
        # Verify optimization
        mock_optimize_mod.optimize_pdf.assert_called_once()
        # Verify upload
        backend.upload_file.assert_called_once()
        pdf_path = backend.download_file.call_args[0][1]
        assert not os.path.isdir(os.path.dirname(pdf_path))

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_mode_no_pdf_returns_skipped_and_cleans_up(self, mock_get_storage_backend):
        """S3 mode with no PDF should return skipped and clean up."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = False
        mock_get_storage_backend.return_value = backend

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
        )

        result = compress_pdf.run(str(job.job_id))

        assert result["status"] == "skipped"
        assert result["reason"] == "no_pdf"
        backend.exists.assert_called_once()
        backend.download_file.assert_not_called()


class TestExtractEntitiesS3(TestCase):
    """Tests for extract_entities with S3 backend."""

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_mode_downloads_ner_uploads_and_cleans_up(
        self, mock_get_storage_backend
    ):
        """S3 mode should download text, run NER, upload results, and clean up."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = True
        mock_get_storage_backend.return_value = backend

        def _download_text(_key, local_path):
            with open(local_path, "w", encoding="utf-8") as f:
                f.write("Sample OCR text")

        backend.download_file.side_effect = _download_text

        # Need to mock the ner module import
        fake_ner = types.ModuleType("ner")
        fake_ner.extract_entities = MagicMock(return_value=[
            {"text": "John Doe", "label": "PERSON"},
            {"text": "New York", "label": "LOCATION"},
        ])

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
            source_hash="abcdef1234567890" * 4,
        )

        with patch.dict(sys.modules, {"ner": fake_ner}):
            result = extract_entities.run(str(job.job_id))

        assert result["status"] == "extracted"
        assert result["entity_count"] == 2
        # Verify exists check
        backend.exists.assert_called_once()
        # Verify download
        backend.download_file.assert_called_once()
        # Verify upload
        backend.upload_file.assert_called_once()
        text_path = backend.download_file.call_args[0][1]
        assert not os.path.isdir(os.path.dirname(text_path))

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_mode_no_text_returns_skipped_and_cleans_up(self, mock_get_storage_backend):
        """S3 mode with no text file should return skipped and clean up."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = False
        mock_get_storage_backend.return_value = backend

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
        )

        result = extract_entities.run(str(job.job_id))

        assert result["status"] == "skipped"
        assert result["reason"] == "no_text"
        backend.exists.assert_called_once()
        backend.download_file.assert_not_called()


class TestStorageBackendUsed(TestCase):
    """Tests for storage_backend_used field being set during ingest."""

    @patch("jobs.tasks.process_document.apply_async")
    @patch("jobs.tasks.extract_pages.delay")
    @patch("jobs.tasks._get_storage_backend")
    def test_ingest_sets_storage_backend_used_nfs(
        self,
        mock_get_storage_backend,
        mock_extract_pages_delay,
        mock_process_document_apply,
    ):
        """Ingest with NFS should set storage_backend_used to 'nfs'."""
        backend = MagicMock()
        backend.backend_name = "nfs"
        mock_get_storage_backend.return_value = backend

        fake_pkg = types.ModuleType("ocr_distributed")
        fake_ocr_utils = types.ModuleType("ocr_distributed.ocr_utils")
        fake_language = types.ModuleType("ocr_distributed.language")

        fake_ocr_utils.classify_source_file = lambda _path, **_kw: ("pdf", None)
        fake_ocr_utils.get_source_page_count = lambda _path, _type: 1

        class _FakeLanguageDetector:
            def __init__(self, _model_path):
                pass

            def detect_from_pdf(self, _source_path):
                return "en"

        fake_language.LanguageDetector = _FakeLanguageDetector
        fake_pkg.ocr_utils = fake_ocr_utils
        fake_pkg.language = fake_language

        with tempfile.TemporaryDirectory() as nfs_root:
            source_dir = os.path.join(nfs_root, "uploads")
            os.makedirs(source_dir, exist_ok=True)
            test_file = os.path.join(source_dir, "test.pdf")
            with open(test_file, "wb") as f:
                f.write(b"%PDF-1.4 test content")

            with override_settings(NFS_ROOT=nfs_root):
                job = Job.objects.create(
                    source_file=test_file,
                    status=Job.Status.SUBMITTED,
                )
                with patch.dict(
                    sys.modules,
                    {
                        "ocr_distributed": fake_pkg,
                        "ocr_distributed.ocr_utils": fake_ocr_utils,
                        "ocr_distributed.language": fake_language,
                    },
                ):
                    ingest_document.run(str(job.job_id))

                job.refresh_from_db()
                assert job.storage_backend_used == "nfs"

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks.process_document.apply_async")
    @patch("jobs.tasks.extract_pages.delay")
    @patch("jobs.tasks._get_storage_backend")
    def test_ingest_sets_storage_backend_used_s3(
        self,
        mock_get_storage_backend,
        mock_extract_pages_delay,
        mock_process_document_apply,
    ):
        """Ingest with S3 should set storage_backend_used to 's3'."""
        backend = MagicMock()
        backend.backend_name = "s3"
        mock_get_storage_backend.return_value = backend

        fake_pkg = types.ModuleType("ocr_distributed")
        fake_ocr_utils = types.ModuleType("ocr_distributed.ocr_utils")
        fake_language = types.ModuleType("ocr_distributed.language")

        fake_ocr_utils.classify_source_file = lambda _path, **_kw: ("pdf", None)
        fake_ocr_utils.get_source_page_count = lambda _path, _type: 1

        class _FakeLanguageDetector:
            def __init__(self, _model_path):
                pass

            def detect_from_pdf(self, _source_path):
                return "en"

        fake_language.LanguageDetector = _FakeLanguageDetector
        fake_pkg.ocr_utils = fake_ocr_utils
        fake_pkg.language = fake_language

        with tempfile.TemporaryDirectory() as nfs_root:
            source_dir = os.path.join(nfs_root, "uploads")
            os.makedirs(source_dir, exist_ok=True)
            test_file = os.path.join(source_dir, "test.pdf")
            with open(test_file, "wb") as f:
                f.write(b"%PDF-1.4 test content")

            with override_settings(NFS_ROOT=nfs_root):
                job = Job.objects.create(
                    source_file=test_file,
                    status=Job.Status.SUBMITTED,
                )
                with patch.dict(
                    sys.modules,
                    {
                        "ocr_distributed": fake_pkg,
                        "ocr_distributed.ocr_utils": fake_ocr_utils,
                        "ocr_distributed.language": fake_language,
                    },
                ):
                    ingest_document.run(str(job.job_id))

                job.refresh_from_db()
                assert job.storage_backend_used == "s3"


class TestGetBackendForJob(TestCase):
    """Tests for _get_backend_for_job helper that resolves backend per job."""

    @patch("jobs.tasks._get_storage_backend")
    def test_returns_current_backend_when_field_empty(self, mock_get_storage_backend):
        """When storage_backend_used is empty, return the current config backend."""
        backend = MagicMock()
        backend.backend_name = "nfs"
        mock_get_storage_backend.return_value = backend

        job = Job.objects.create(
            source_file="/test.pdf",
            storage_backend_used="",
        )

        result = _get_backend_for_job(job)

        assert result.backend_name == "nfs"
        mock_get_storage_backend.assert_called_once()

    @patch("jobs.tasks._get_storage_backend")
    def test_returns_current_backend_when_matching(self, mock_get_storage_backend):
        """When storage_backend_used matches current config, return current backend."""
        backend = MagicMock()
        backend.backend_name = "nfs"
        mock_get_storage_backend.return_value = backend

        job = Job.objects.create(
            source_file="/test.pdf",
            storage_backend_used="nfs",
        )

        result = _get_backend_for_job(job)

        assert result.backend_name == "nfs"
        mock_get_storage_backend.assert_called_once()

    @patch("jobs.tasks.create_storage_backend")
    @patch("jobs.tasks._get_storage_backend")
    def test_returns_ingested_backend_on_mismatch(
        self, mock_get_storage_backend, mock_create_backend
    ):
        """When storage_backend_used differs from current, return ingested backend."""
        current_backend = MagicMock()
        current_backend.backend_name = "s3"
        mock_get_storage_backend.return_value = current_backend

        nfs_backend = MagicMock()
        nfs_backend.backend_name = "nfs"
        mock_create_backend.return_value = nfs_backend

        job = Job.objects.create(
            source_file="/test.pdf",
            storage_backend_used="nfs",
        )

        result = _get_backend_for_job(job)

        assert result.backend_name == "nfs"
        mock_create_backend.assert_called_once()
        # Verify it was called with the job's original backend name
        call_kwargs = mock_create_backend.call_args[1]
        assert call_kwargs["backend_name"] == "nfs"

    @patch("jobs.tasks.create_storage_backend")
    @patch("jobs.tasks._get_storage_backend")
    def test_falls_back_to_current_on_creation_error(
        self, mock_get_storage_backend, mock_create_backend
    ):
        """When backend creation fails for mismatched name, fall back to current."""
        current_backend = MagicMock()
        current_backend.backend_name = "s3"
        mock_get_storage_backend.return_value = current_backend

        mock_create_backend.side_effect = ValueError("Unsupported storage backend 'invalid'")

        job = Job.objects.create(
            source_file="/test.pdf",
            storage_backend_used="invalid",
        )

        result = _get_backend_for_job(job)

        # Should fall back to the current backend
        assert result.backend_name == "s3"
        # _get_storage_backend called twice: once initial, once for fallback
        assert mock_get_storage_backend.call_count == 2

