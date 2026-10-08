"""Tests for Celery task definitions.

Tests the coordinator, OCR, and periodic tasks with all external
dependencies (PaddleOCR, Tesseract, filesystem, Celery broker) mocked.

For bind=True tasks, we use task.run(...) which auto-injects `self`.
For non-bind tasks, we call them directly as functions.

Run with: cd coordinator && python -m pytest jobs/tests/test_tasks.py -v
"""

import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import types
import uuid
from contextlib import nullcontext
from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.db import models, transaction
from django.test import TestCase, TransactionTestCase, override_settings
from django.utils import timezone

from jobs.models import CustodyEvent, Job, PageResult, TaskDispatchOutbox, Worker
from jobs.storage import CachedS3Backend, NFSBackend
from jobs.tasks import (
    FANOUT_THRESHOLD,
    _claim_ingest_lease,
    _compression_execution_lock,
    _compute_file_hash,
    _create_storage_backend,
    _discard_if_postprocessing_stale,
    _dispatch_processing,
    _durable_publish_nfs_custody,
    _engine_policy_from_job_settings,
    _ensure_job_dirs,
    _finalize_custody_chain_s3,
    _get_backend_for_job,
    _get_job_processing_timeout_minutes,
    _get_storage_backend,
    _job_storage_key,
    _nfs_job_path,
    _OcrFallbackRequired,
    _process_single_page,
    _reconcile_fanout_page,
    _record_custody_event,
    _recover_fair_page_admission,
    _renew_or_discard_published_artifacts,
    _reset_storage_backend,
    assemble_document,
    check_worker_heartbeats,
    chord_error_handler,
    cleanup_completed_jobs,
    cleanup_stale_jobs,
    compress_pdf,
    extract_entities,
    extract_pages,
    extract_structured_data,
    finalize_job,
    ingest_document,
    process_document,
    process_page,
    register_worker,
    unregister_worker,
)
from ocr_local.engines.governance import (
    BLOCKED_DENYLIST,
    BLOCKED_NOT_IN_ALLOWLIST,
    EnginePolicy,
    compute_policy_hash,
)
from ocr_local.engines.provenance import (
    ENGINE_PROVENANCE,
    LANE_BOUNDARY_ENFORCED,
    LICENSE_BLOCKED,
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


class TestDurableNfsCustodyPublish(TestCase):
    def test_same_directory_temp_is_fsynced_then_atomically_published(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = os.path.join(tmpdir, "doc.custody.jsonl")
            captured_temp_paths = []
            real_mkstemp = tempfile.mkstemp

            def capture_mkstemp(*args, **kwargs):
                assert os.path.abspath(kwargs["dir"]) == os.path.abspath(tmpdir)
                fd, temp_path = real_mkstemp(*args, **kwargs)
                captured_temp_paths.append(temp_path)
                return fd, temp_path

            records = [
                {
                    "document_id": "doc",
                    "event_type": "file_ingested",
                    "timestamp": "2026-07-12T00:00:00.000+00:00",
                    "data": {},
                    "prev_hash": None,
                    "hash": "a" * 64,
                }
            ]
            with (
                patch(
                    "jobs.tasks.tempfile.mkstemp",
                    side_effect=capture_mkstemp,
                ),
                patch("jobs.tasks.os.fsync", wraps=os.fsync) as mock_fsync,
            ):
                _durable_publish_nfs_custody(records, destination)

            # One fsync persists the file bytes; _durable_replace then fsyncs
            # the parent directory so the rename itself survives a crash.
            assert mock_fsync.call_count == 2
            assert os.path.isfile(destination)
            assert all(not os.path.exists(path) for path in captured_temp_paths)
            with open(destination, encoding="utf-8") as custody_file:
                assert json.loads(custody_file.read()) == records[0]

    def test_pre_replace_fsync_failure_preserves_existing_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = os.path.join(tmpdir, "doc.custody.jsonl")
            with open(destination, "wb") as custody_file:
                custody_file.write(b"old-valid-chain\n")

            with patch("jobs.tasks.os.fsync", side_effect=OSError("disk failure")):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    _durable_publish_nfs_custody([], destination)

            with open(destination, "rb") as custody_file:
                assert custody_file.read() == b"old-valid-chain\n"
            assert not [name for name in os.listdir(tmpdir) if name.endswith(".tmp")]


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

    def test_ingest_claim_is_single_delivery(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.SUBMITTED,
        )

        assert _claim_ingest_lease(job, "ingest-a") == "ingest-a"
        assert _claim_ingest_lease(job, "ingest-b") is None
        job.refresh_from_db()
        assert job.status == Job.Status.INGESTING
        assert job.celery_task_id == "ingest-a"


class TestFinalizeJob(TransactionTestCase):
    """Tests for finalize_job task."""

    def _stage_final_pdf(
        self,
        job,
        *,
        event_type="compression_complete",
        disposition="APPLIED",
        pdf_bytes=b"%PDF-1.4 final bytes",
    ):
        job_path = tempfile.mkdtemp(prefix="finalize_job_")
        self.addCleanup(shutil.rmtree, job_path, True)
        _ensure_job_dirs(job_path)
        job.nfs_job_path = job_path
        job.storage_backend_used = "nfs"
        job.save(update_fields=["nfs_job_path", "storage_backend_used"])

        base_name = os.path.splitext(os.path.basename(job.source_file))[0]
        pdf_path = os.path.join(
            job_path,
            "output",
            "EXPORT",
            "PDF",
            f"{base_name}.pdf",
        )
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(pdf_bytes)
        pdf_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
        document_id = job.source_hash[:16] if job.source_hash else str(job.job_id)[:16]
        assembly_event = _record_custody_event(
            job,
            document_id,
            "assembly_complete",
            data={
                "output_pdf": pdf_path,
                "pages_assembled": job.total_pages,
            },
        )
        _record_custody_event(
            job,
            document_id,
            event_type,
            data={
                "causal_event_id": assembly_event.id,
                "pdf_path": pdf_path,
                "disposition": disposition,
                "pdf_sha256": pdf_sha256,
            },
        )
        return job_path, pdf_path, pdf_sha256, document_id

    def _header_results(self, job):
        compression_event = (
            CustodyEvent.objects.filter(
                job=job,
                event_type__in=("compression_complete", "compression_skipped"),
            )
            .order_by("-id")
            .first()
        )
        if compression_event is None:
            event_id = 0
            disposition = "APPLIED"
            pdf_sha256 = "0" * 64
            status = "compressed"
        else:
            event_id = compression_event.id
            disposition = compression_event.data["disposition"]
            pdf_sha256 = compression_event.data["pdf_sha256"]
            status = (
                {
                    "APPLIED": "compressed",
                    "SKIPPED_VISUAL_BUDGET": "skipped_visual_budget",
                    "SKIPPED_CANDIDATE_REJECTED": "skipped_candidate_rejected",
                }[disposition]
            )
        job_id = str(job.job_id)
        return [
            {
                "stage": "compression",
                "status": status,
                "job_id": job_id,
                "disposition": disposition,
                "pdf_sha256": pdf_sha256,
                "custody_event_id": event_id,
            },
            {
                "stage": "entities",
                "status": "extracted",
                "job_id": job_id,
                "entity_count": 0,
            },
            {
                "stage": "structured_data",
                "status": "skipped",
                "reason": "not_requested",
                "job_id": job_id,
            },
        ]

    def test_missing_job_returns_error(self):
        fake_id = str(uuid.uuid4())
        result = finalize_job.run([], fake_id)
        assert result["status"] == "error"

    def test_cancelled_job_returns_cancelled(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.CANCELLED,
        )
        result = finalize_job.run([], str(job.job_id))
        assert result["status"] == "cancelled"

    def test_computes_result_summary(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=4,
        )
        job_path, _, _, document_id = self._stage_final_pdf(job)
        # Create page results
        PageResult.objects.create(
            job=job,
            document_id="d1",
            page_num=1,
            status="ok",
            ocr_confidence=0.95,
            ocr_method="PaddleOCR",
        )
        PageResult.objects.create(
            job=job,
            document_id="d1",
            page_num=2,
            status="ok",
            ocr_confidence=0.90,
            ocr_method="PaddleOCR",
        )
        PageResult.objects.create(
            job=job,
            document_id="d1",
            page_num=3,
            status="fallback",
            ocr_confidence=0.70,
            ocr_method="Tesseract",
        )
        PageResult.objects.create(
            job=job,
            document_id="d1",
            page_num=4,
            status="image_only",
            ocr_confidence=0.0,
            ocr_method="ImageOnly",
        )

        result = finalize_job.run(self._header_results(job), str(job.job_id))

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
        custody_path = os.path.join(
            job_path,
            "output",
            "EXPORT",
            "CUSTODY",
            f"{document_id}.custody.jsonl",
        )
        with open(custody_path, encoding="utf-8") as custody_file:
            exported = [json.loads(line) for line in custody_file]
        assert exported[0]["prev_hash"] is None
        for index, row in enumerate(exported):
            expected_prev = None if index == 0 else exported[index - 1]["hash"]
            assert row["prev_hash"] == expected_prev
            payload = {key: value for key, value in row.items() if key != "hash"}
            assert (
                row["hash"]
                == hashlib.sha256(
                    json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
            )
        compression_row = next(
            row for row in exported if row["event_type"] == "compression_complete"
        )
        assert compression_row["data"]["pdf_sha256"] == _compute_file_hash(
            os.path.join(job_path, "output", "EXPORT", "PDF", "test.pdf")
        )
        assert compression_row["hash"]

    def test_finalize_with_no_page_results(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=0,
        )
        self._stage_final_pdf(job)
        result = finalize_job.run(self._header_results(job), str(job.job_id))
        assert result["status"] == "completed"
        assert result["summary"]["average_confidence"] == 0.0

    def test_causal_event_ids_override_untrusted_worker_timestamps(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
        )
        self._stage_final_pdf(job)
        PageResult.objects.create(
            job=job,
            document_id=str(job.job_id)[:16],
            page_num=1,
            status="ok",
        )
        assembly_event = CustodyEvent.objects.get(
            job=job,
            event_type="assembly_complete",
        )
        compression_event = CustodyEvent.objects.get(
            job=job,
            event_type="compression_complete",
        )
        CustodyEvent.objects.filter(pk=assembly_event.pk).update(
            timestamp=timezone.now(),
        )
        CustodyEvent.objects.filter(pk=compression_event.pk).update(
            timestamp=timezone.now() - timedelta(days=1),
        )

        # This test deliberately exercises causal ordering with legacy mutable
        # timestamps.  Production keeps PA-A10 re-verification enabled; the
        # explicit opt-in here models emergency development tooling only.
        with patch.dict(os.environ, {"CUSTODY_EXPORT_REVERIFY": "false"}):
            result = finalize_job.run(self._header_results(job), str(job.job_id))

        assert result["status"] == "completed"

    def test_success_custody_is_not_published_until_outer_commit(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
        )
        job_path, _, _, document_id = self._stage_final_pdf(job)
        PageResult.objects.create(
            job=job,
            document_id=document_id,
            page_num=1,
            status="ok",
        )
        custody_path = os.path.join(
            job_path,
            "output",
            "EXPORT",
            "CUSTODY",
            f"{document_id}.custody.jsonl",
        )

        with transaction.atomic():
            result = finalize_job.run(
                self._header_results(job),
                str(job.job_id),
            )
            assert result["status"] == "completed"
            assert not os.path.exists(custody_path)

        assert os.path.isfile(custody_path)
        job.refresh_from_db()
        assert job.custody_published_at is not None

    def test_republisher_filters_published_jobs_before_limit(self):
        from jobs.tasks import _republish_pending_custody_control_plane

        old = Job.objects.create(
            source_file="/old.pdf",
            status=Job.Status.COMPLETED,
            completed_at=timezone.now() - timedelta(days=2),
            custody_published_at=timezone.now() - timedelta(days=2),
        )
        missing = Job.objects.create(
            source_file="/missing.pdf",
            status=Job.Status.COMPLETED,
            completed_at=timezone.now() - timedelta(days=1),
        )
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = False

        with (
            patch(
                "jobs.tasks._get_backend_for_job", return_value=backend
            ) as get_backend,
            patch("jobs.tasks._prepare_custody_chain", return_value=[{"hash": "h"}]),
            patch("jobs.tasks._durable_publish_s3_custody") as publish,
        ):
            result = _republish_pending_custody_control_plane(limit=1)

        assert result == {"scanned": 1, "published": 1, "failures": []}
        get_backend.assert_called_once()
        assert get_backend.call_args.args[0].pk == missing.pk
        publish.assert_called_once()
        missing.refresh_from_db()
        old.refresh_from_db()
        assert missing.custody_published_at is not None
        assert old.custody_published_at is not None

    def test_rolled_back_outer_commit_never_publishes_success_custody(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
        )
        job_path, _, _, document_id = self._stage_final_pdf(job)
        PageResult.objects.create(
            job=job,
            document_id=document_id,
            page_num=1,
            status="ok",
        )
        custody_path = os.path.join(
            job_path,
            "output",
            "EXPORT",
            "CUSTODY",
            f"{document_id}.custody.jsonl",
        )

        with transaction.atomic():
            finalize_job.run(self._header_results(job), str(job.job_id))
            transaction.set_rollback(True)

        assert not os.path.exists(custody_path)
        job.refresh_from_db()
        assert job.status == Job.Status.ASSEMBLING

    def test_finalize_marks_job_failed_when_any_page_failed(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=2,
        )
        job_path, _, _, document_id = self._stage_final_pdf(job)
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

        result = finalize_job.run(self._header_results(job), str(job.job_id))
        assert result["status"] == "failed"
        assert result["summary"]["pages_failed"] == 1

        job.refresh_from_db()
        assert job.status == Job.Status.FAILED
        custody_path = os.path.join(
            job_path,
            "output",
            "EXPORT",
            "CUSTODY",
            f"{document_id}.custody.jsonl",
        )
        with open(custody_path, encoding="utf-8") as custody_file:
            exported = [json.loads(line) for line in custody_file]
        assert exported[-1]["event_type"] == "processing_failed"
        assert "1 page(s) failed" in exported[-1]["data"]["error"]

    def test_missing_compression_outcome_fails_closed(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
        )
        document_id = str(job.job_id)[:16]
        _record_custody_event(
            job,
            document_id,
            "assembly_complete",
            data={"pages_assembled": job.total_pages},
        )

        with self.assertRaises(RuntimeError):
            finalize_job.run(self._header_results(job), str(job.job_id))

        job.refresh_from_db()
        assert job.status == Job.Status.FAILED
        assert "compression custody outcome" in job.error_message

    def test_final_pdf_hash_mismatch_publishes_failure_custody(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
        )
        job_path, pdf_path, _, document_id = self._stage_final_pdf(job)
        with open(pdf_path, "ab") as pdf_file:
            pdf_file.write(b"tamper")

        with self.assertRaises(RuntimeError):
            finalize_job.run(self._header_results(job), str(job.job_id))

        job.refresh_from_db()
        assert job.status == Job.Status.FAILED
        assert "hash no longer matches" in job.error_message
        custody_path = os.path.join(
            job_path,
            "output",
            "EXPORT",
            "CUSTODY",
            f"{document_id}.custody.jsonl",
        )
        with open(custody_path, encoding="utf-8") as custody_file:
            exported = [json.loads(line) for line in custody_file]
        assert exported[-1]["event_type"] == "processing_failed"
        assert "hash no longer matches" in exported[-1]["data"]["error"]

    def test_late_callback_does_not_resurrect_failed_job(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.FAILED,
            error_message="compression header failed",
        )

        result = finalize_job.run([], str(job.job_id))

        job.refresh_from_db()
        assert result["status"] == "failed"
        assert job.status == Job.Status.FAILED
        assert job.completed_at is None

    def test_non_assembling_callback_publishes_failure_custody(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=1,
        )
        job_path, _, _, document_id = self._stage_final_pdf(job)
        PageResult.objects.create(
            job=job,
            document_id=document_id,
            page_num=1,
            status="ok",
        )

        with self.assertRaisesRegex(RuntimeError, "requires the job to be assembling"):
            finalize_job.run(self._header_results(job), str(job.job_id))

        job.refresh_from_db()
        assert job.status == Job.Status.FAILED
        custody_path = os.path.join(
            job_path,
            "output",
            "EXPORT",
            "CUSTODY",
            f"{document_id}.custody.jsonl",
        )
        with open(custody_path, encoding="utf-8") as custody_file:
            exported = [json.loads(line) for line in custody_file]
        assert exported[-1]["event_type"] == "processing_failed"

    def test_invalid_header_result_publishes_failure_custody(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
        )
        job_path, _, _, document_id = self._stage_final_pdf(job)
        PageResult.objects.create(
            job=job,
            document_id=document_id,
            page_num=1,
            status="ok",
        )
        header_results = self._header_results(job)
        header_results[1] = {
            **header_results[1],
            "status": "error",
            "message": "NER silently failed",
        }

        with self.assertRaisesRegex(RuntimeError, "entity extraction did not succeed"):
            finalize_job.run(header_results, str(job.job_id))

        job.refresh_from_db()
        assert job.status == Job.Status.FAILED
        custody_path = os.path.join(
            job_path,
            "output",
            "EXPORT",
            "CUSTODY",
            f"{document_id}.custody.jsonl",
        )
        with open(custody_path, encoding="utf-8") as custody_file:
            exported = [json.loads(line) for line in custody_file]
        assert exported[-1]["event_type"] == "processing_failed"
        assert "entity extraction did not succeed" in exported[-1]["data"]["error"]

    def test_late_export_error_republishes_failure_shaped_chain(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
        )
        job_path, _, _, document_id = self._stage_final_pdf(job)
        PageResult.objects.create(
            job=job,
            document_id=document_id,
            page_num=1,
            status="ok",
        )
        publish_calls = 0

        def publish_then_report_late_failure(records, jsonl_path):
            nonlocal publish_calls
            publish_calls += 1
            _durable_publish_nfs_custody(records, jsonl_path)
            if publish_calls == 1:
                raise OSError("late directory durability failure")

        with (
            patch(
                "jobs.tasks._durable_publish_nfs_custody",
                side_effect=publish_then_report_late_failure,
            ),
            self.assertRaisesRegex(RuntimeError, "late directory durability failure"),
        ):
            finalize_job.run(self._header_results(job), str(job.job_id))

        assert publish_calls == 2
        job.refresh_from_db()
        assert job.status == Job.Status.FAILED
        custody_path = os.path.join(
            job_path,
            "output",
            "EXPORT",
            "CUSTODY",
            f"{document_id}.custody.jsonl",
        )
        with open(custody_path, encoding="utf-8") as custody_file:
            exported = [json.loads(line) for line in custody_file]
        assert exported[-1]["event_type"] == "processing_failed"
        assert "late directory durability failure" in exported[-1]["data"]["error"]

    def test_s3_finalization_rehashes_stored_pdf_before_export(self):
        pdf_bytes = b"%PDF-1.4 remote final bytes"
        pdf_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
        job = Job.objects.create(
            source_file="test.pdf",
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
            storage_backend_used="s3",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
        )
        PageResult.objects.create(
            job=job,
            document_id=job.source_hash[:16],
            page_num=1,
            status="ok",
            ocr_method="PaddleOCR",
        )
        pdf_key = _job_storage_key(
            job.job_id,
            "output/EXPORT/PDF/test.pdf",
        )
        assembly_event = _record_custody_event(
            job,
            job.source_hash[:16],
            "assembly_complete",
            data={"pages_assembled": 1},
        )
        _record_custody_event(
            job,
            job.source_hash[:16],
            "compression_complete",
            data={
                "causal_event_id": assembly_event.id,
                "pdf_path": pdf_key,
                "disposition": "APPLIED",
                "pdf_sha256": pdf_sha256,
            },
        )
        inner = MagicMock(backend_name="s3")
        inner.endpoint = "https://s3.example.test"
        inner.bucket = "test-bucket"
        inner.region = "us-east-1"
        inner.exists.return_value = True
        inner.object_revision.return_value = ("final-etag", "v2", len(pdf_bytes))
        cache_dir = tempfile.mkdtemp(prefix="stale_s3_cache_")
        self.addCleanup(shutil.rmtree, cache_dir, True)
        backend = CachedS3Backend(inner=inner, cache_dir=cache_dir)
        stale_bytes = b"%PDF-1.4 stale pre-compression bytes"
        with open(
            backend._cache_path(pdf_key, ("stale-etag", "v1", len(stale_bytes))),
            "wb",
        ) as cached_pdf:
            cached_pdf.write(stale_bytes)
        downloaded = {}

        def download_remote(_key, local_path):
            downloaded["path"] = local_path
            with open(local_path, "wb") as pdf_file:
                pdf_file.write(pdf_bytes)

        inner.download_file.side_effect = download_remote
        with (
            patch("jobs.tasks._get_backend_for_job", return_value=backend),
            patch("jobs.tasks._schedule_s3_custody_publish") as mock_schedule,
        ):
            result = finalize_job.run(self._header_results(job), str(job.job_id))

        assert result["status"] == "completed"
        inner.download_file.assert_called_once_with(pdf_key, downloaded["path"])
        mock_schedule.assert_called_once()
        scheduled_job, scheduled_document_id, scheduled_backend = (
            mock_schedule.call_args.args
        )
        assert scheduled_job.pk == job.pk
        assert scheduled_document_id == job.source_hash[:16]
        assert scheduled_backend is backend
        assert not os.path.exists(downloaded["path"])

    def test_cancellation_during_s3_verification_suppresses_success(self):
        pdf_bytes = b"%PDF-1.4 remote final bytes"
        pdf_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
        job = Job.objects.create(
            source_file="test.pdf",
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
            storage_backend_used="s3",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
        )
        PageResult.objects.create(
            job=job,
            document_id=job.source_hash[:16],
            page_num=1,
            status="ok",
        )
        pdf_key = _job_storage_key(
            job.job_id,
            "output/EXPORT/PDF/test.pdf",
        )
        assembly_event = _record_custody_event(
            job,
            job.source_hash[:16],
            "assembly_complete",
            data={"pages_assembled": 1},
        )
        _record_custody_event(
            job,
            job.source_hash[:16],
            "compression_complete",
            data={
                "causal_event_id": assembly_event.id,
                "pdf_path": pdf_key,
                "disposition": "APPLIED",
                "pdf_sha256": pdf_sha256,
            },
        )
        backend = MagicMock(backend_name="s3")
        backend.exists.return_value = True

        def cancel_while_downloading(_key, local_path):
            Job.objects.filter(pk=job.pk).update(status=Job.Status.CANCELLED)
            with open(local_path, "wb") as pdf_file:
                pdf_file.write(pdf_bytes)

        backend.download_file.side_effect = cancel_while_downloading
        with (
            patch("jobs.tasks._get_backend_for_job", return_value=backend),
            patch("jobs.tasks._schedule_s3_custody_publish") as mock_schedule,
        ):
            result = finalize_job.run(self._header_results(job), str(job.job_id))

        job.refresh_from_db()
        assert result["status"] == "cancelled"
        assert job.status == Job.Status.CANCELLED
        mock_schedule.assert_not_called()
        assert not CustodyEvent.objects.filter(
            job=job,
            event_type="processing_failed",
        ).exists()

    def test_s3_hash_mismatch_publishes_failure_custody(self):
        expected = hashlib.sha256(b"expected").hexdigest()
        job = Job.objects.create(
            source_file="test.pdf",
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
            storage_backend_used="s3",
            status=Job.Status.ASSEMBLING,
            total_pages=1,
        )
        pdf_key = _job_storage_key(
            job.job_id,
            "output/EXPORT/PDF/test.pdf",
        )
        assembly_event = _record_custody_event(
            job,
            job.source_hash[:16],
            "assembly_complete",
            data={"pages_assembled": 1},
        )
        _record_custody_event(
            job,
            job.source_hash[:16],
            "compression_complete",
            data={
                "causal_event_id": assembly_event.id,
                "pdf_path": pdf_key,
                "disposition": "APPLIED",
                "pdf_sha256": expected,
            },
        )
        backend = MagicMock(backend_name="s3")
        backend.exists.return_value = True
        downloaded = {}
        uploaded = {}

        def download_tampered(_key, local_path):
            downloaded["path"] = local_path
            with open(local_path, "wb") as pdf_file:
                pdf_file.write(b"tampered")

        def capture_custody(local_path, storage_key):
            uploaded["key"] = storage_key
            with open(local_path, encoding="utf-8") as custody_file:
                uploaded["rows"] = [json.loads(line) for line in custody_file]

        backend.download_file.side_effect = download_tampered
        backend.upload_file.side_effect = capture_custody
        with (
            patch(
                "jobs.tasks._get_backend_for_job",
                return_value=backend,
            ),
            self.assertRaises(RuntimeError),
        ):
            finalize_job.run(self._header_results(job), str(job.job_id))

        job.refresh_from_db()
        assert job.status == Job.Status.FAILED
        assert "object hash no longer matches" in job.error_message
        assert uploaded["key"].endswith(f"/{job.source_hash[:16]}.custody.jsonl")
        assert uploaded["rows"][-1]["event_type"] == "processing_failed"
        assert (
            "object hash no longer matches" in (uploaded["rows"][-1]["data"]["error"])
        )
        assert not os.path.exists(downloaded["path"])

    def test_s3_custody_is_id_ordered_fsynced_and_genesis_is_null(self):
        job = Job.objects.create(source_file="test.pdf", storage_backend_used="s3")
        document_id = str(job.job_id)[:16]
        first = CustodyEvent.objects.create(
            job=job,
            document_id=document_id,
            event_type="file_ingested",
            timestamp=timezone.now(),
            data={"sequence": 1},
        )
        second = CustodyEvent.objects.create(
            job=job,
            document_id=document_id,
            event_type="compression_complete",
            timestamp=timezone.now() - timedelta(days=1),
            data={"sequence": 2},
        )
        captured = {}
        backend = MagicMock(backend_name="s3")

        def capture_upload(local_path, storage_key):
            captured["path"] = local_path
            captured["key"] = storage_key
            with open(local_path, encoding="utf-8") as custody_file:
                captured["rows"] = [json.loads(line) for line in custody_file]

        backend.upload_file.side_effect = capture_upload
        with patch("jobs.tasks.os.fsync", wraps=os.fsync) as mock_fsync:
            with patch.dict(
                os.environ,
                {
                    "DEPLOYMENT_ENV": "development",
                    "CUSTODY_ALLOW_LEGACY_BLANK_RESEAL": "true",
                },
            ):
                with transaction.atomic():
                    _finalize_custody_chain_s3(job, document_id, backend)

        assert mock_fsync.call_count == 1
        assert not os.path.exists(captured["path"])
        rows = captured["rows"]
        assert [row["data"]["sequence"] for row in rows] == [1, 2]
        assert rows[0]["prev_hash"] is None
        assert rows[1]["prev_hash"] == rows[0]["hash"]
        first.refresh_from_db()
        second.refresh_from_db()
        assert first.prev_hash == ""
        assert second.prev_hash == first.event_hash


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

    def test_does_not_overwrite_terminal_transition_after_scan(self):
        stale_time = timezone.now() - timedelta(minutes=45)
        job = Job.objects.create(
            source_file="/raced.pdf",
            status=Job.Status.PROCESSING,
            started_at=stale_time,
        )
        original_filter = Job.objects.filter
        filter_calls = 0

        def transition_before_compare_and_set(*args, **kwargs):
            nonlocal filter_calls
            filter_calls += 1
            if filter_calls == 2:
                original_filter(pk=job.pk).update(status=Job.Status.COMPLETED)
            return original_filter(*args, **kwargs)

        with patch.object(
            Job.objects,
            "filter",
            side_effect=transition_before_compare_and_set,
        ):
            result = cleanup_stale_jobs()

        assert result["stale_jobs"] == 0
        job.refresh_from_db()
        assert job.status == Job.Status.COMPLETED

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

    @override_settings(JOB_PROCESSING_TIMEOUT_MINUTES=30)
    def test_caps_extreme_per_job_timeout_override(self):
        job = Job.objects.create(
            source_file="/poisoned-timeout.pdf",
            status=Job.Status.PROCESSING,
            started_at=timezone.now() - timedelta(minutes=45),
            settings_json={"processing_timeout_minutes": 10**18},
        )

        assert _get_job_processing_timeout_minutes(job) == 10_080
        result = cleanup_stale_jobs()

        assert result["stale_jobs"] == 0
        job.refresh_from_db()
        assert job.status == Job.Status.PROCESSING


class TestCompressPdf(TestCase):
    """Tests for compress_pdf task."""

    def test_missing_job_returns_error(self):
        fake_id = str(uuid.uuid4())
        result = compress_pdf.run(fake_id)
        assert result["status"] == "error"

    def test_cancelled_job_discards_postprocessing_publish(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job = Job.objects.create(
                source_file="/cancelled.pdf",
                status=Job.Status.CANCELLED,
                nfs_job_path=tmpdir,
            )
            artifact = os.path.join(tmpdir, "output", "EXPORT", "PDF", "cancelled.pdf")
            os.makedirs(os.path.dirname(artifact), exist_ok=True)
            with open(artifact, "wb") as handle:
                handle.write(b"late publish")
            backend = NFSBackend(tmpdir)

            assert _discard_if_postprocessing_stale(job, backend, artifact) is True
            assert not os.path.exists(artifact)

    def test_live_job_keeps_postprocessing_publish(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job = Job.objects.create(
                source_file="/live.pdf",
                status=Job.Status.ASSEMBLING,
                nfs_job_path=tmpdir,
            )
            artifact = os.path.join(tmpdir, "output", "EXPORT", "PDF", "live.pdf")
            os.makedirs(os.path.dirname(artifact), exist_ok=True)
            with open(artifact, "wb") as handle:
                handle.write(b"current publish")
            backend = NFSBackend(tmpdir)

            assert _discard_if_postprocessing_stale(job, backend, artifact) is False
            assert os.path.exists(artifact)

    def test_lost_publish_fence_removes_every_page_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job = Job.objects.create(
                source_file="/cancelled.pdf",
                status=Job.Status.CANCELLED,
                nfs_job_path=tmpdir,
            )
            artifacts = []
            for suffix in ("pdf", "txt"):
                artifact = os.path.join(tmpdir, "temp", "doc", f"1.{suffix}")
                os.makedirs(os.path.dirname(artifact), exist_ok=True)
                with open(artifact, "wb") as handle:
                    handle.write(b"late page publish")
                artifacts.append((artifact, None))
            backend = NFSBackend(tmpdir)

            def _lost_fence():
                raise RuntimeError("tenant purge fence committed")

            with self.assertRaisesRegex(RuntimeError, "tenant purge fence"):
                _renew_or_discard_published_artifacts(
                    _lost_fence,
                    job,
                    backend,
                    tuple(artifacts),
                )

            assert all(not os.path.exists(path) for path, _ in artifacts)

    def test_no_pdf_file_raises_for_chord_errback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job = Job.objects.create(
                source_file="/test.pdf",
                status=Job.Status.ASSEMBLING,
                nfs_job_path=tmpdir,
                source_hash="a" * 64,
            )
            _ensure_job_dirs(tmpdir)
            document_id = job.source_hash[:16]
            CustodyEvent.objects.create(
                job=job,
                document_id=document_id,
                event_type="assembly_complete",
                data={"pages_assembled": 1},
            )
            with self.assertRaises(FileNotFoundError):
                compress_pdf.run(str(job.job_id))


class TestExtractEntities(TestCase):
    """Tests for extract_entities task."""

    def test_missing_job_raises_for_chord_errback(self):
        fake_id = str(uuid.uuid4())
        with self.assertRaisesRegex(RuntimeError, "not found for entity extraction"):
            extract_entities.run(fake_id)

    def test_no_text_file_raises_for_chord_errback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job = Job.objects.create(
                source_file="/test.pdf",
                nfs_job_path=tmpdir,
            )
            _ensure_job_dirs(tmpdir)
            with self.assertRaisesRegex(FileNotFoundError, "NER text file is missing"):
                extract_entities.run(str(job.job_id))

    def test_ner_module_unavailable_raises_for_chord_errback(self):
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
                with self.assertRaisesRegex(RuntimeError, "NER module is unavailable"):
                    extract_entities.run(str(job.job_id))


class TestExtractStructuredData(TestCase):
    """Requested extraction failures must fail the post-processing chord."""

    def test_not_requested_is_the_only_explicit_skip(self):
        job = Job.objects.create(
            source_file="/test.pdf",
            settings_json={"enable_docintel": False},
        )

        result = extract_structured_data.run(str(job.job_id))

        assert result == {
            "stage": "structured_data",
            "status": "skipped",
            "reason": "not_requested",
            "job_id": str(job.job_id),
        }

    def test_requested_missing_text_raises_for_chord_errback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job = Job.objects.create(
                source_file="/test.pdf",
                nfs_job_path=tmpdir,
                settings_json={"enable_docintel": True},
            )
            _ensure_job_dirs(tmpdir)

            with self.assertRaisesRegex(
                FileNotFoundError,
                "structured-extraction text file is missing",
            ):
                extract_structured_data.run(str(job.job_id))

    def test_requested_unavailable_module_raises_for_chord_errback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job = Job.objects.create(
                source_file="/document.pdf",
                nfs_job_path=tmpdir,
                settings_json={"enable_docintel": True},
            )
            _ensure_job_dirs(tmpdir)
            text_path = os.path.join(
                tmpdir,
                "output",
                "EXPORT",
                "TEXT",
                "document.txt",
            )
            with open(text_path, "w", encoding="utf-8") as text_file:
                text_file.write("Extract these fields")
            import builtins

            real_import = builtins.__import__

            def reject_extraction(name, *args, **kwargs):
                if name == "extraction":
                    raise ImportError("module unavailable")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=reject_extraction):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "structured-extraction module is unavailable",
                ):
                    extract_structured_data.run(str(job.job_id))


# ---------------------------------------------------------------------------
# Step 5a: extract_pages tests
# ---------------------------------------------------------------------------


class TestExtractPages(TestCase):
    """Tests for extract_pages task (chord fan-out orchestration)."""

    def _make_job(self, total_pages=25, status=Job.Status.PROCESSING, **kwargs):
        return Job.objects.create(
            source_file="/docs/big_report.pdf",
            source_type="pdf",
            status=status,
            total_pages=total_pages,
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
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

    def test_persists_page_outbox_with_correct_page_count(self):
        job = self._make_job(total_pages=30)
        extract_pages.run(str(job.job_id))
        rows = TaskDispatchOutbox.objects.filter(
            job=job,
            stage__startswith="process_page:",
        )
        assert rows.count() == 30
        assert all(row.task_name == "jobs.tasks.process_page" for row in rows)

    def test_fair_window_admits_only_capacity_and_replenishes_once(self):
        job = self._make_job(total_pages=6)
        with patch.dict(
            os.environ,
            {
                "OCR_FAIR_FANOUT_ENABLED": "true",
                "OCR_MAX_INFLIGHT_PAGES_PER_DOCUMENT": "2",
            },
        ):
            result = extract_pages.run(str(job.job_id))
            assert result["page_tasks"] == 2
            assert result["pages_pending_admission"] == 4
            assert list(
                TaskDispatchOutbox.objects.filter(job=job, stage__startswith="process_page:")
                .order_by("stage")
                .values_list("stage", flat=True)
            ) == ["process_page:1", "process_page:2"]
            PageResult.objects.filter(job=job, page_num=1).update(status="processing")
            assert _recover_fair_page_admission() == 0  # worker still owns the slot
            PageResult.objects.filter(job=job, page_num=1).update(status="ok")
            assert len(_reconcile_fanout_page(job.job_id, 1)) == 1
            assert len(_reconcile_fanout_page(job.job_id, 1)) == 0
            assert TaskDispatchOutbox.objects.filter(
                job=job, stage__startswith="process_page:"
            ).count() == 3

    def test_fair_recovery_admits_one_page_per_job_after_crash_gap(self):
        jobs = [self._make_job(total_pages=4) for _ in range(2)]
        with patch.dict(
            os.environ,
            {
                "OCR_FAIR_FANOUT_ENABLED": "true",
                "OCR_MAX_INFLIGHT_PAGES_PER_DOCUMENT": "2",
            },
        ):
            for job in jobs:
                extract_pages.run(str(job.job_id))
                PageResult.objects.filter(job=job, page_num=1).update(status="ok")
            assert _recover_fair_page_admission(limit=2) == 2
            assert [
                TaskDispatchOutbox.objects.filter(
                    job=job, stage__startswith="process_page:"
                ).count()
                for job in jobs
            ] == [3, 3]

    def test_fair_recovery_skips_full_jobs_before_limiting_scan(self):
        with patch.dict(
            os.environ,
            {
                "OCR_FAIR_FANOUT_ENABLED": "true",
                "OCR_MAX_INFLIGHT_PAGES_PER_DOCUMENT": "1",
            },
        ):
            for _ in range(11):
                full = self._make_job(total_pages=2)
                full.fanout_dispatched = True
                full.save(update_fields=["fanout_dispatched"])
                PageResult.objects.create(
                    job=full,
                    document_id=full.source_hash[:16],
                    page_num=1,
                    status="processing",
                    admitted_generation=full.dispatch_generation,
                )
                PageResult.objects.create(
                    job=full,
                    document_id=full.source_hash[:16],
                    page_num=2,
                    status="pending",
                )
            eligible = self._make_job(total_pages=1)
            eligible.fanout_dispatched = True
            eligible.save(update_fields=["fanout_dispatched"])
            PageResult.objects.create(
                job=eligible,
                document_id=eligible.source_hash[:16],
                page_num=1,
                status="pending",
            )
            assert _recover_fair_page_admission(limit=10) == 1
            assert TaskDispatchOutbox.objects.filter(
                job=eligible, stage="process_page:1"
            ).exists()

    def test_page_outbox_contract_carries_queue_and_generation(self):
        job = self._make_job(total_pages=3)
        extract_pages.run(str(job.job_id))
        rows = list(TaskDispatchOutbox.objects.filter(job=job).order_by("stage"))
        assert len(rows) == 3
        assert all(row.queue in {"ocr_gpu", "ocr_cpu"} for row in rows)
        assert all(row.generation == str(job.dispatch_generation) for row in rows)

    def test_separated_fanout_routes_unsupported_language_to_tesseract(self):
        job = self._make_job(total_pages=2, detected_language="el")
        with patch.dict(os.environ, {"OCR_SEPARATE_TESSERACT_LANE": "true"}):
            extract_pages.run(str(job.job_id))
        rows = list(TaskDispatchOutbox.objects.filter(job=job).order_by("stage"))
        assert len(rows) == 2
        assert all(row.queue == "ocr_tesseract" for row in rows)
        assert all(row.kwargs.get("ocr_engine_lane") == "tesseract" for row in rows)

    def test_separated_fanout_routes_unknown_language_to_paddle_bootstrap(self):
        job = self._make_job(total_pages=2, detected_language="und")
        with patch.dict(
            os.environ,
            {"OCR_SEPARATE_TESSERACT_LANE": "true", "OCR_TASK_ROUTING": "gpu"},
        ):
            extract_pages.run(str(job.job_id))
        rows = list(TaskDispatchOutbox.objects.filter(job=job).order_by("stage"))
        assert len(rows) == 2
        assert all(row.queue == "ocr_gpu" for row in rows)
        assert all(row.kwargs.get("ocr_engine_lane") == "paddle" for row in rows)

    def test_separated_fanout_routes_supported_language_to_paddle(self):
        job = self._make_job(total_pages=2, detected_language="en")
        with patch.dict(
            os.environ,
            {"OCR_SEPARATE_TESSERACT_LANE": "true", "OCR_TASK_ROUTING": "cpu"},
        ):
            extract_pages.run(str(job.job_id))
        rows = list(TaskDispatchOutbox.objects.filter(job=job).order_by("stage"))
        assert all(row.queue == "ocr_cpu" for row in rows)
        assert all(row.kwargs.get("ocr_engine_lane") == "paddle" for row in rows)

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
        PageResult.objects.create(
            job=job, page_num=1, document_id=doc_id, status="failed"
        )
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
            source_type="pdf",
            status=Job.Status.PROCESSING,
            total_pages=1,
            source_hash="",
        )
        extract_pages.run(str(job.job_id))
        pr = PageResult.objects.get(job=job, page_num=1)
        assert pr.document_id == str(job.job_id)[:16]

    @patch("jobs.tasks.chord")
    def test_records_requested_vlm_denial_once_before_fanout(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job = self._make_job(
            total_pages=3,
            settings_json={"ocr_engine": "paddleocr-vl-0.9b"},
        )
        extract_pages.run(str(job.job_id))
        boundaries = CustodyEvent.objects.filter(
            job=job,
            event_type=LANE_BOUNDARY_ENFORCED,
        )
        assert boundaries.count() == 1
        boundary = boundaries.get()
        assert boundary.data["page_id"] is None
        assert boundary.data["engine_id"] == "paddleocr-vl-0.9b"
        assert boundary.data["is_generative"] is True


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
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
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
            "page_num": 1,
            "method": "PaddleOCR",
            "confidence": 0.95,
            "text_length": 100,
            "status": "ok",
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
    def test_paddle_handoff_retries_owned_queue_without_failed_terminal(
        self, mock_process
    ):
        mock_process.side_effect = _OcrFallbackRequired("no Paddle text")
        job, tmpdir = self._make_job_with_nfs()
        job.tenant_id = "tenant-lane-test"
        job.save(update_fields=["tenant_id"])
        try:
            with patch.object(
                process_page,
                "retry",
                side_effect=RuntimeError("retry scheduled"),
            ) as retry:
                with self.assertRaisesRegex(RuntimeError, "retry scheduled"):
                    process_page.run(
                        str(job.job_id),
                        1,
                        ocr_engine_lane="paddle",
                        tenant_id=job.tenant_id,
                    )

            retry.assert_called_once()
            retry_kwargs = retry.call_args.kwargs
            assert retry_kwargs["queue"] == "ocr_tesseract"
            assert retry_kwargs["countdown"] == 0
            assert retry_kwargs["kwargs"]["ocr_engine_lane"] == "tesseract"
            page = PageResult.objects.get(job=job, page_num=1)
            assert page.status == "processing"
            assert not CustodyEvent.objects.filter(
                job=job,
                event_type="processing_failed",
            ).exists()
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
            "page_num": 1,
            "status": "ok",
            "method": "PaddleOCR",
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
    def test_progress_reconciles_from_durable_page_rows(self, mock_process):
        """A stale counter is replaced by the exact durable success count."""
        mock_process.return_value = {"page_num": 1, "status": "ok"}
        job, tmpdir = self._make_job_with_nfs()
        job.pages_completed = 5
        job.save(update_fields=["pages_completed"])
        try:
            process_page.run(str(job.job_id), 1)
            job.refresh_from_db()
            assert job.pages_completed == 1
        finally:
            self._cleanup(tmpdir)


# ---------------------------------------------------------------------------
# Step 5c: process_document tests
# ---------------------------------------------------------------------------


class TestProcessDocument(TestCase):
    """Old broker contracts redirect into page-fenced dispatch."""

    @patch("jobs.tasks._process_single_page")
    @patch("jobs.tasks.extract_pages.run")
    def test_pdf_redirects_without_unleased_ocr(self, extract, legacy_ocr):
        job = Job.objects.create(
            source_file="/test.pdf", source_type="pdf",
            status=Job.Status.PROCESSING, dispatch_generation=1,
        )
        extract.return_value = {"status": "fanout_dispatched"}
        result = process_document.run(str(job.job_id), dispatch_generation=1)
        assert result["status"] == "fanout_dispatched"
        extract.assert_called_once_with(
            str(job.job_id), tenant_id=None, dispatch_generation=1,
        )
        legacy_ocr.assert_not_called()

    @patch("jobs.tasks.extract_pages.run")
    def test_image_redirects_and_stale_generation_does_not(self, extract):
        job = Job.objects.create(
            source_file="/test.png", source_type="image",
            status=Job.Status.PROCESSING, dispatch_generation=2,
        )
        extract.return_value = {"status": "fanout_dispatched"}
        assert process_document.run(str(job.job_id), dispatch_generation=2)["status"] == "fanout_dispatched"
        assert process_document.run(str(job.job_id), dispatch_generation=1)["status"] != "fanout_dispatched"
        assert extract.call_count == 1

    @patch("jobs.tasks.extract_pages.run")
    def test_unclassified_root_task_fails_closed(self, extract):
        job = Job.objects.create(
            source_file="/test.bin", source_type="",
            status=Job.Status.PROCESSING, dispatch_generation=1,
        )
        result = process_document.run(str(job.job_id), dispatch_generation=1)
        job.refresh_from_db()
        assert result["status"] == "error"
        assert job.status == Job.Status.FAILED
        extract.assert_not_called()

    def test_missing_and_cancelled_jobs_do_not_dispatch(self):
        assert process_document.run(str(uuid.uuid4()))["status"] == "error"
        job = Job.objects.create(source_file="/test.pdf", source_type="pdf", status=Job.Status.CANCELLED)
        assert process_document.run(str(job.job_id))["status"] == "cancelled"


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
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
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
                job=job,
                document_id=document_id,
                page_num=pn,
                status="ok",
                admitted_generation=job.dispatch_generation,
                ocr_confidence=0.9,
                ocr_provenance={
                    "text_sha256": hashlib.sha256(
                        f"Text for page {pn}".encode("utf-8")
                    ).hexdigest(),
                },
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
        with open(
            os.path.join(temp_dir, f"{page_num}.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(text)

    def _cleanup(self, tmpdir):
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    def _active_output_paths(self, job):
        """Resolve generation-pointer PDF/TEXT locators after assembly."""
        from jobs.artifact_generations import (
            ACTIVE_OUTPUT_POINTER_KEY,
            resolve_active_artifacts,
        )

        job.refresh_from_db()
        pointer = (job.settings_json or {}).get(ACTIVE_OUTPUT_POINTER_KEY)
        assert pointer is not None, "assembly must publish an active output pointer"
        artifacts = resolve_active_artifacts(
            job_id=job.job_id,
            source_file=job.source_file,
            settings_json=job.settings_json,
            backend="nfs",
            job_root=job.nfs_job_path,
        )
        return artifacts.pdf.locator, artifacts.text.locator

    @patch("jobs.tasks.chord")
    def test_merges_all_page_pdfs(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, _ = self._make_job_with_pages(total_pages=3)
        try:
            result = assemble_document.run(str(job.job_id))
            assert result["status"] == "assembled"
            assert result["pages_assembled"] == 3
            output_pdf, _ = self._active_output_paths(job)
            assert os.path.isfile(output_pdf)
            assert "/output/generations/" in output_pdf.replace("\\", "/")
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_assembly_losslessly_deflates_raster_without_moving_search_text(self, mock_chord):
        import io

        import fitz
        import numpy as np
        from PIL import Image, ImageDraw

        mock_chord.return_value = MagicMock()
        job, tmpdir, document_id = self._make_job_with_pages(total_pages=1)
        try:
            image = Image.new("RGB", (1200, 1600), "white")
            ImageDraw.Draw(image).rectangle((200, 200, 700, 230), fill="black")
            encoded = io.BytesIO()
            image.save(encoded, format="PNG")
            page_path = os.path.join(tmpdir, "temp", document_id, "1.pdf")
            with fitz.open() as page_doc:
                page = page_doc.new_page(width=600, height=800)
                page.insert_image(page.rect, stream=encoded.getvalue())
                page.insert_text((100, 100), "Evidence.", render_mode=3)
                source_pixels = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).samples
                source_box, = page.search_for("Evidence.")
                page_doc.save(page_path)
            assert os.path.getsize(page_path) > 5_000_000
            assert assemble_document.run(str(job.job_id))["status"] == "assembled"
            output_pdf, _ = self._active_output_paths(job)
            assert os.path.getsize(output_pdf) < 100_000
            with fitz.open(output_pdf) as merged:
                page = merged[0]
                assert page.get_text().strip() == "Evidence."
                box, = page.search_for("Evidence.")
                assert abs(box.x0 - source_box.x0) < 0.01
                assert abs(box.y0 - source_box.y0) < 0.01
                output_pixels = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).samples
                assert np.array_equal(np.frombuffer(output_pixels, dtype=np.uint8),
                                      np.frombuffer(source_pixels, dtype=np.uint8))
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_assembly_reads_winning_page_lease_paths(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, document_id = self._make_job_with_pages(total_pages=1)
        try:
            winner_dir = os.path.join(
                tmpdir, "temp", document_id, "g0", "p1", "lwinner"
            )
            os.makedirs(winner_dir)
            self._create_page_pdf(winner_dir, 1)
            self._create_page_text(winner_dir, 1, "Winning lease text")
            PageResult.objects.filter(job=job, page_num=1).update(
                temp_pdf_path=os.path.join(winner_dir, "1.pdf"),
                ocr_provenance={
                    "temp_text_path": os.path.join(winner_dir, "1.txt"),
                    "text_sha256": hashlib.sha256(b"Winning lease text").hexdigest(),
                },
            )
            result = assemble_document.run(str(job.pk))
            assert result["status"] == "assembled"
            _, output_text = self._active_output_paths(job)
            with open(output_text, encoding="utf-8") as output:
                assert output.read() == "Winning lease text"
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_writes_merged_text(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, _ = self._make_job_with_pages(total_pages=2)
        try:
            assemble_document.run(str(job.job_id))
            _, output_text = self._active_output_paths(job)
            assert os.path.isfile(output_text)
            with open(output_text, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Text for page 1" in content
            assert "Text for page 2" in content
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_changed_accepted_page_text_fails_before_publication(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, document_id = self._make_job_with_pages(total_pages=1)
        try:
            text_path = os.path.join(tmpdir, "temp", document_id, "1.txt")
            with open(text_path, "w", encoding="utf-8") as output:
                output.write("Fext for page 1")  # Same byte count, different evidence.
            result = assemble_document.run(str(job.job_id))
            assert result["status"] == "failed"
            failure = CustodyEvent.objects.get(job=job, event_type="assembly_failed")
            assert failure.data["failure_kind"] == "page_text_digest_mismatch"
            assert not CustodyEvent.objects.filter(job=job, event_type="assembly_complete").exists()
            job.refresh_from_db()
            assert not (job.settings_json or {}).get("active_output_pointer")
            with open(text_path, encoding="utf-8") as source:
                assert source.read() == "Fext for page 1"
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_unsealed_accepted_page_text_fails_closed(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, _ = self._make_job_with_pages(total_pages=1)
        try:
            PageResult.objects.filter(job=job, page_num=1).update(ocr_provenance={})
            result = assemble_document.run(str(job.job_id))
            assert result["status"] == "failed"
            failure = CustodyEvent.objects.get(job=job, event_type="assembly_failed")
            assert failure.data["failure_kind"] == "page_text_digest_mismatch"
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_unadmitted_or_other_source_page_cannot_be_assembled(self, mock_chord):
        mock_chord.return_value = MagicMock()
        for changes in (
            {"admitted_generation": None},
            {"admitted_generation": 9},
            {"document_id": "other-source"},
        ):
            with self.subTest(changes=changes):
                job, tmpdir, _ = self._make_job_with_pages(total_pages=1)
                try:
                    PageResult.objects.filter(job=job, page_num=1).update(**changes)
                    result = assemble_document.run(str(job.job_id))
                    assert result["status"] == "failed"
                    failure = CustodyEvent.objects.get(job=job, event_type="assembly_failed")
                    assert failure.data["failure_kind"] == "stale_page_identity"
                    assert not CustodyEvent.objects.filter(
                        job=job, event_type="assembly_complete"
                    ).exists()
                    job.refresh_from_db()
                    assert not (job.settings_json or {}).get("active_output_pointer")
                finally:
                    self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_large_assembly_uses_periodic_checkpointing(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, document_id = self._make_job_with_pages(total_pages=2)
        checkpoint_path = os.path.join(
            tmpdir, "temp", document_id, "_merged_checkpoint.pdf"
        )
        import jobs.tasks as tasks_module

        try:
            with (
                patch("jobs.tasks.ASSEMBLY_STREAMING_THRESHOLD_PAGES", 1),
                patch("jobs.tasks.ASSEMBLY_CHECKPOINT_INTERVAL_PAGES", 1),
                patch(
                    "jobs.tasks._checkpoint_merged_pdf",
                    wraps=tasks_module._checkpoint_merged_pdf,
                ) as mock_checkpoint,
            ):
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

    def test_persists_post_processing_handoff(self):
        job, tmpdir, _ = self._make_job_with_pages(total_pages=1)
        try:
            assemble_document.run(str(job.job_id))
            row = TaskDispatchOutbox.objects.get(job=job, stage="postprocessing")
            assert row.task_name == "jobs.tasks.dispatch_postprocessing"
            assert row.queue == "coordinator"
        finally:
            self._cleanup(tmpdir)

    def test_postprocessing_handoff_is_idempotent(self):
        job, tmpdir, _ = self._make_job_with_pages(total_pages=1)
        try:
            assemble_document.run(str(job.job_id))
            assemble_document.run(str(job.job_id))
            assert (
                TaskDispatchOutbox.objects.filter(
                    job=job,
                    stage="postprocessing",
                ).count()
                == 1
            )
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
    def test_missing_page_pdf_fails_without_partial_success(self, mock_chord):
        mock_chord.return_value = MagicMock()
        # Page 2 has neither an accepted result nor an artifact.
        job, tmpdir, _ = self._make_job_with_pages(total_pages=3, pages_ok=[1, 3])
        try:
            with self.captureOnCommitCallbacks(execute=True):
                result = assemble_document.run(str(job.job_id))

            job.refresh_from_db()
            assert result["status"] == "failed"
            assert job.status == Job.Status.FAILED
            assert not CustodyEvent.objects.filter(
                job=job,
                event_type="assembly_complete",
            ).exists()
            failure = CustodyEvent.objects.get(job=job, event_type="assembly_failed")
            assert failure.data["failure_kind"] == "missing_accepted_page_result"
            assert failure.data["failure_page"] == 2
            assert not os.path.exists(
                os.path.join(tmpdir, "output", "EXPORT", "PDF", "document.pdf")
            )
            assert not os.path.exists(
                os.path.join(tmpdir, "output", "EXPORT", "TEXT", "document.txt")
            )
            mock_chord.assert_not_called()
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_corrupt_page_pdf_fails_without_postprocessing(self, mock_chord):
        job, tmpdir, document_id = self._make_job_with_pages(total_pages=1)
        page_pdf = os.path.join(tmpdir, "temp", document_id, "1.pdf")
        with open(page_pdf, "wb") as corrupt_pdf:
            corrupt_pdf.write(b"not a PDF")

        try:
            with self.captureOnCommitCallbacks(execute=True):
                result = assemble_document.run(str(job.job_id))

            job.refresh_from_db()
            assert result["status"] == "failed"
            assert job.status == Job.Status.FAILED
            failure = CustodyEvent.objects.get(job=job, event_type="assembly_failed")
            assert failure.data["failure_kind"] == "corrupt_page_pdf"
            assert failure.data["failure_page"] == 1
            assert not CustodyEvent.objects.filter(
                job=job,
                event_type="assembly_complete",
            ).exists()
            mock_chord.assert_not_called()
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
    def test_terminal_jobs_are_never_resurrected(self, mock_chord):
        for status in (
            Job.Status.COMPLETED,
            Job.Status.FAILED,
            Job.Status.CANCELLED,
        ):
            with self.subTest(status=status):
                job = Job.objects.create(source_file="/terminal.pdf", status=status)
                result = assemble_document.run(str(job.job_id))
                job.refresh_from_db()
                assert result["status"] == status
                assert job.status == status
                assert not CustodyEvent.objects.filter(job=job).exists()
        mock_chord.assert_not_called()

    @patch("jobs.tasks.chord")
    def test_invalid_entry_state_fails_closed(self, mock_chord):
        job, tmpdir, _ = self._make_job_with_pages(total_pages=1)
        job.status = Job.Status.SUBMITTED
        job.save(update_fields=["status"])
        try:
            with self.captureOnCommitCallbacks(execute=True):
                result = assemble_document.run(str(job.job_id))
            job.refresh_from_db()
            assert result["status"] == "failed"
            assert job.status == Job.Status.FAILED
            assert "requires processing/assembling state" in job.error_message
            assert (
                CustodyEvent.objects.filter(
                    job=job,
                    event_type="assembly_failed",
                ).count()
                == 1
            )
            mock_chord.assert_not_called()
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_cancellation_during_publication_suppresses_success(self, mock_chord):
        import jobs.tasks as tasks_module

        job, tmpdir, _ = self._make_job_with_pages(total_pages=1)
        real_commit = tasks_module._durably_commit_assembly_file
        calls = 0

        def cancel_after_first_publish(staged_path, final_path):
            nonlocal calls
            real_commit(staged_path, final_path)
            calls += 1
            if calls == 1:
                Job.objects.filter(job_id=job.job_id).update(
                    status=Job.Status.CANCELLED
                )

        try:
            with patch(
                "jobs.tasks._durably_commit_assembly_file",
                side_effect=cancel_after_first_publish,
            ):
                result = assemble_document.run(str(job.job_id))

            job.refresh_from_db()
            assert result["status"] == "cancelled"
            assert job.status == Job.Status.CANCELLED
            assert not CustodyEvent.objects.filter(
                job=job,
                event_type="assembly_complete",
            ).exists()
            mock_chord.assert_not_called()
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_supersession_during_publication_suppresses_success(self, mock_chord):
        import jobs.tasks as tasks_module

        job, tmpdir, _ = self._make_job_with_pages(total_pages=1)
        real_commit = tasks_module._durably_commit_assembly_file
        calls = 0

        def supersede_after_first_publish(staged_path, final_path):
            nonlocal calls
            real_commit(staged_path, final_path)
            calls += 1
            if calls == 1:
                Job.objects.filter(job_id=job.job_id).update(
                    settings_json={"_assembly_claim_token": "new-owner"}
                )

        try:
            with patch(
                "jobs.tasks._durably_commit_assembly_file",
                side_effect=supersede_after_first_publish,
            ):
                result = assemble_document.run(str(job.job_id))

            job.refresh_from_db()
            assert result["status"] == "superseded"
            assert job.status == Job.Status.ASSEMBLING
            assert not CustodyEvent.objects.filter(
                job=job,
                event_type="assembly_complete",
            ).exists()
            mock_chord.assert_not_called()
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_defers_custody_chain_until_post_processing_finishes(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, doc_id = self._make_job_with_pages(total_pages=1)
        # Add a pre-existing custody event
        _record_custody_event(job, doc_id, "file_ingested", data={"hash": "abc"})
        try:
            assemble_document.run(str(job.job_id))
            # Compression/other post-processing events do not exist yet, so
            # assembly must not freeze or export an incomplete hash chain.
            finalized = CustodyEvent.objects.filter(job=job, chain_finalized=True)
            assert finalized.count() == 0
        finally:
            self._cleanup(tmpdir)

    @patch("jobs.tasks.chord")
    def test_does_not_export_partial_custody_jsonl(self, mock_chord):
        mock_chord.return_value = MagicMock()
        job, tmpdir, doc_id = self._make_job_with_pages(total_pages=1)
        try:
            assemble_document.run(str(job.job_id))
            jsonl_path = os.path.join(
                tmpdir, "output", "EXPORT", "CUSTODY", f"{doc_id}.custody.jsonl"
            )
            assert not os.path.isfile(jsonl_path)
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


class TestEnginePolicyFromJobSettings:
    def test_parses_explicit_policy_fields_without_dataclass_introspection(self):
        policy = _engine_policy_from_job_settings(
            {
                "engine_policy": {
                    "forensic_mode": "false",
                    "allow_generative": "yes",
                    "allow_cloud_egress": "0",
                    "allow_nc_licensed": "on",
                    "allow_use_restrictions": "true",
                    "allow_gated_licensed": "true",
                    "gated_license_ack": "1",
                    "enterprise_license_key": "ent-123",
                    "allowed_engine_ids": "paddle-ctc",
                    "blocked_engine_ids": ["dots-ocr"],
                    "preferred_engine_ids": ("tesseract",),
                    "max_monthly_mau_attested": "1000",
                    "max_monthly_revenue_usd": "",
                    "unknown_field": "ignored",
                },
            }
        )

        assert policy.forensic_mode is False
        assert policy.allow_generative is True
        assert policy.allow_cloud_egress is False
        assert policy.allow_nc_licensed is True
        assert policy.allow_use_restrictions is True
        assert policy.allow_gated_licensed is True
        assert policy.gated_license_ack is True
        assert policy.enterprise_license_key == "ent-123"
        assert policy.allowed_engine_ids == ("paddle-ctc",)
        assert policy.blocked_engine_ids == ("dots-ocr",)
        assert policy.preferred_engine_ids == ("tesseract",)
        assert policy.max_monthly_mau_attested == 1000
        assert policy.max_monthly_revenue_usd is None


class TestProcessSinglePage(TestCase):
    """Tests for _process_single_page helper function.

    _process_single_page does local imports from ocr_distributed.ocr_utils
    and fitz, which may not be on sys.path or may segfault on repeated import
    in test environments. We inject mock modules via sys.modules patches.
    """

    def _make_job_with_nfs(self, settings_json=None):
        tmpdir = tempfile.mkdtemp()
        _ensure_job_dirs(tmpdir)
        job = Job.objects.create(
            source_file="/test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=5,
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
            nfs_job_path=tmpdir,
            detected_language="en",
            source_type="pdf",
            settings_json=settings_json or {},
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
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__getitem__.return_value = mock_page

        def fake_save(path, **kwargs):
            # Write a minimal valid PDF
            with open(path, "wb") as f:
                f.write(b"%PDF-1.0\n1 0 obj<</Type/Catalog>>endobj\n%%EOF\n")

        mock_doc.save = fake_save
        mock_fitz.open.return_value = mock_doc
        return mock_fitz

    def _ocr_mods(self, mock_utils, mock_fitz):
        """Return a sys.modules dict patch for ocr_distributed and fitz."""
        # Legacy unit tests use a synthetic PDF writer. Reflect only the
        # mocked insertions for readback; real PyMuPDF parity is covered by
        # test_ocr_rescue_integration.py.
        mock_fitz.open.return_value.__getitem__.return_value.get_text.side_effect = (
            lambda **_kwargs: " ".join(
                str(call.args[1])
                for call in mock_utils.insert_text_line.call_args_list
            )
        )
        mock_pkg = MagicMock()
        mock_pkg.ocr_utils = mock_utils
        mock_pytesseract = types.SimpleNamespace(
            image_to_string=MagicMock(return_value=""),
            Output=types.SimpleNamespace(DICT="dict"),
        )

        def image_to_data(image, *, lang, output_type, timeout):
            # Existing test cases patch image_to_string.  Adapt their sample
            # words to the production image_to_data contract with positions.
            assert output_type == "dict"
            words = str(
                mock_pytesseract.image_to_string(
                    image, lang=lang, timeout=timeout
                )
            ).split()
            return {
                "text": words,
                "conf": [90] * len(words),
                "left": [5 + index * 20 for index in range(len(words))],
                "top": [10] * len(words),
                "width": [18] * len(words),
                "height": [10] * len(words),
            }

        mock_pytesseract.image_to_data = image_to_data
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
        tess_ocr = MagicMock(return_value="Hello Tess")
        try:
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("jobs.tasks._gpu_available", return_value=False),
                patch("pytesseract.image_to_string", tess_ocr),
            ):
                result = _process_single_page(
                    job,
                    "/test.pdf",
                    1,
                    job.source_hash[:16],
                    "worker-01",
                )
            assert result["method"] == "Tesseract"
            assert result["status"] == "fallback"
            tess_ocr.assert_called_once_with(img, lang="eng", timeout=120)
            pr = PageResult.objects.get(job=job, page_num=1)
            assert pr.ocr_method == "Tesseract"
            assert pr.status == "fallback"
        finally:
            self._cleanup(tmpdir)

    def test_paddle_lane_hands_off_without_running_tesseract(self):
        """Separated Paddle workers never execute the fallback binary."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                return_value=MagicMock(ocr=MagicMock(return_value=None))
            ),
            extract_paddle_lines=MagicMock(return_value=[]),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        tess_ocr = MagicMock(side_effect=AssertionError("wrong capability lane"))
        try:
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("jobs.tasks._gpu_available", return_value=False),
                patch("pytesseract.image_to_string", tess_ocr),
            ):
                with self.assertRaises(_OcrFallbackRequired):
                    _process_single_page(
                        job,
                        "/test.pdf",
                        1,
                        job.source_hash[:16],
                        "paddle-worker-01",
                        ocr_engine_lane="paddle",
                    )
            tess_ocr.assert_not_called()
            assert not PageResult.objects.filter(job=job, page_num=1).exists()
        finally:
            self._cleanup(tmpdir)

    def test_tesseract_lane_never_initializes_paddle(self):
        """Fallback workers execute only their explicitly owned engine."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="white")
        create_paddle = MagicMock(side_effect=AssertionError("wrong capability lane"))
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=create_paddle,
            extract_paddle_lines=MagicMock(
                side_effect=AssertionError("wrong capability lane")
            ),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        tess_ocr = MagicMock(return_value="lane-owned fallback")
        try:
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("pytesseract.image_to_string", tess_ocr),
            ):
                result = _process_single_page(
                    job,
                    "/test.pdf",
                    1,
                    job.source_hash[:16],
                    "tesseract-worker-01",
                    ocr_engine_lane="tesseract",
                )
            assert result["method"] == "Tesseract"
            assert result["status"] == "fallback"
            create_paddle.assert_not_called()
            tess_ocr.assert_called_once_with(img, lang="eng", timeout=120)
        finally:
            self._cleanup(tmpdir)

    def test_tesseract_lane_failure_is_not_silent_image_only_success(self):
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                side_effect=AssertionError("wrong capability lane")
            ),
            extract_paddle_lines=MagicMock(
                side_effect=AssertionError("wrong capability lane")
            ),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        try:
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch(
                    "pytesseract.image_to_string",
                    side_effect=RuntimeError("fallback engine unavailable"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Tesseract lane failed"):
                    _process_single_page(
                        job,
                        "/test.pdf",
                        1,
                        job.source_hash[:16],
                        "tesseract-worker-01",
                        ocr_engine_lane="tesseract",
                    )
            assert not PageResult.objects.filter(job=job, page_num=1).exists()
        finally:
            self._cleanup(tmpdir)

    def test_unsupported_paddle_script_routes_directly_to_tesseract(self):
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="white")
        create_paddle = MagicMock(side_effect=AssertionError("must not use Paddle"))
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=create_paddle,
            extract_paddle_lines=MagicMock(return_value=[]),
        )
        job, tmpdir = self._make_job_with_nfs()
        job.detected_language = "el"
        job.save(update_fields=["detected_language"])
        mock_fitz = self._make_fitz_mock(tmpdir)
        tess_ocr = MagicMock(return_value="Greek text")
        try:
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("pytesseract.image_to_string", tess_ocr),
            ):
                result = _process_single_page(
                    job,
                    "/test.pdf",
                    1,
                    job.source_hash[:16],
                    "worker-01",
                )
            assert result["method"] == "Tesseract"
            create_paddle.assert_not_called()
            tess_ocr.assert_called_once_with(img, lang="ell", timeout=120)
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
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("jobs.tasks._gpu_available", return_value=False),
                patch(
                    "pytesseract.image_to_string",
                    side_effect=RuntimeError("No tesseract"),
                ),
            ):
                result = _process_single_page(
                    job,
                    "/test.pdf",
                    1,
                    job.source_hash[:16],
                    "worker-01",
                )
            assert result["method"] == "ImageOnly"
            assert result["status"] == "failed"
            pr = PageResult.objects.get(job=job, page_num=1)
            assert pr.ocr_method == "ImageOnly"
            assert pr.status == "failed"
            assert pr.needs_review
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
            extract_paddle_lines=MagicMock(
                return_value=[
                    ("Hello world", [[0, 0], [100, 0], [100, 20], [0, 20]], 0.95),
                ]
            ),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        try:
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("jobs.tasks._gpu_available", return_value=False),
            ):
                result = _process_single_page(
                    job,
                    "/test.pdf",
                    1,
                    job.source_hash[:16],
                    "worker-01",
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
                    job,
                    "/test.pdf",
                    1,
                    job.source_hash[:16],
                    "worker-01",
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

        owned_temp_dir = os.path.join(
            tempfile.gettempdir(), f"ocr_page_{uuid.uuid4().hex}"
        )
        os.makedirs(owned_temp_dir, exist_ok=True)
        try:
            with (
                patch("tempfile.mkdtemp", return_value=owned_temp_dir),
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
            ):
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
            extract_paddle_lines=MagicMock(
                return_value=[
                    ("text", [[0, 0], [10, 0], [10, 10], [0, 10]], 0.8),
                ]
            ),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        doc_id = job.source_hash[:16]
        try:
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("jobs.tasks._gpu_available", return_value=False),
            ):
                _process_single_page(job, "/test.pdf", 1, doc_id, "w1")
            events = CustodyEvent.objects.filter(job=job, event_type="ocr_primary")
            assert events.count() == 1
        finally:
            self._cleanup(tmpdir)

    def test_records_engine_provenance_for_paddle_success(self):
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                return_value=MagicMock(ocr=MagicMock(return_value="r"))
            ),
            extract_paddle_lines=MagicMock(
                return_value=[
                    ("text", [[0, 0], [10, 0], [10, 10], [0, 10]], 0.8),
                ]
            ),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        doc_id = job.source_hash[:16]
        try:
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("jobs.tasks._gpu_available", return_value=False),
            ):
                _process_single_page(job, "/test.pdf", 1, doc_id, "w1")
            event = CustodyEvent.objects.get(job=job, event_type=ENGINE_PROVENANCE)
            assert event.data["page_id"] == f"{doc_id}:1"
            assert event.data["engine_id"] == "paddle-ctc"
            assert event.data["policy_hash"] == compute_policy_hash(EnginePolicy())
            assert event.data["decision"] == "ADMITTED"
        finally:
            self._cleanup(tmpdir)

    def test_requested_vlm_forensic_mode_falls_back_without_page_level_boundary(self):
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                return_value=MagicMock(ocr=MagicMock(return_value=None))
            ),
            extract_paddle_lines=MagicMock(return_value=[]),
        )
        job, tmpdir = self._make_job_with_nfs(
            settings_json={"ocr_engine": "paddleocr-vl-0.9b"},
        )
        mock_fitz = self._make_fitz_mock(tmpdir)
        doc_id = job.source_hash[:16]
        try:
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("jobs.tasks._gpu_available", return_value=False),
                patch(
                    "pytesseract.image_to_string",
                    side_effect=RuntimeError("force image-only fallback"),
                ),
            ):
                result = _process_single_page(job, "/test.pdf", 1, doc_id, "w1")
            assert result["method"] == "ImageOnly"
            assert result["status"] == "failed"
            assert not CustodyEvent.objects.filter(
                job=job,
                event_type=LANE_BOUNDARY_ENFORCED,
            ).exists()
            assert not CustodyEvent.objects.filter(
                job=job,
                event_type=ENGINE_PROVENANCE,
            ).exists()
        finally:
            self._cleanup(tmpdir)

    def test_requested_vlm_denial_is_not_recorded_per_page(self):
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                return_value=MagicMock(ocr=MagicMock(return_value=None))
            ),
            extract_paddle_lines=MagicMock(return_value=[]),
        )
        job, tmpdir = self._make_job_with_nfs(
            settings_json={"ocr_engine": "paddleocr-vl-0.9b"},
        )
        mock_fitz = self._make_fitz_mock(tmpdir)
        doc_id = job.source_hash[:16]
        try:
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("jobs.tasks._gpu_available", return_value=False),
                patch("pytesseract.image_to_string", return_value="fallback"),
            ):
                _process_single_page(job, "/test.pdf", 1, doc_id, "w1")
                _process_single_page(job, "/test.pdf", 2, doc_id, "w1")

            boundaries = CustodyEvent.objects.filter(
                job=job,
                event_type=LANE_BOUNDARY_ENFORCED,
            )
            assert boundaries.count() == 0
        finally:
            self._cleanup(tmpdir)

    def test_policy_denied_paddle_uses_tesseract_and_records_denial(self):
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                side_effect=AssertionError("paddle should not run")
            ),
            extract_paddle_lines=MagicMock(return_value=[]),
        )
        policy = {"blocked_engine_ids": ["paddle-ctc"]}
        job, tmpdir = self._make_job_with_nfs(settings_json={"engine_policy": policy})
        mock_fitz = self._make_fitz_mock(tmpdir)
        doc_id = job.source_hash[:16]
        try:
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("pytesseract.image_to_string", return_value="fallback"),
            ):
                result = _process_single_page(job, "/test.pdf", 1, doc_id, "w1")
            assert result["method"] == "Tesseract"
            assert result["status"] == "fallback"
            denial = CustodyEvent.objects.get(job=job, event_type=LICENSE_BLOCKED)
            assert denial.data["engine_id"] == "paddle-ctc"
            assert denial.data["reason"] == BLOCKED_DENYLIST
            provenance = CustodyEvent.objects.get(job=job, event_type=ENGINE_PROVENANCE)
            assert provenance.data["engine_id"] == "tesseract"
        finally:
            self._cleanup(tmpdir)

    def test_policy_denied_all_ocr_preserves_image_only_without_engine_provenance(self):
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="white")
        mock_utils = _make_ocr_utils_mock(
            iter_source_images=MagicMock(return_value=[img]),
            create_paddle_engine=MagicMock(
                side_effect=AssertionError("paddle should not run")
            ),
            extract_paddle_lines=MagicMock(return_value=[]),
        )
        policy = {"allowed_engine_ids": ["not-paddle-or-tesseract"]}
        job, tmpdir = self._make_job_with_nfs(settings_json={"engine_policy": policy})
        mock_fitz = self._make_fitz_mock(tmpdir)
        doc_id = job.source_hash[:16]
        try:
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch(
                    "pytesseract.image_to_string",
                    side_effect=AssertionError("tesseract should not run"),
                ),
            ):
                result = _process_single_page(job, "/test.pdf", 1, doc_id, "w1")
            assert result["method"] == "ImageOnly"
            assert result["status"] == "image_only"
            denials = CustodyEvent.objects.filter(job=job, event_type=LICENSE_BLOCKED)
            assert denials.count() == 2
            assert {
                event.data["engine_id"]: event.data["reason"] for event in denials
            } == {
                "paddle-ctc": BLOCKED_NOT_IN_ALLOWLIST,
                "tesseract": BLOCKED_NOT_IN_ALLOWLIST,
            }
            assert not CustodyEvent.objects.filter(
                job=job,
                event_type=ENGINE_PROVENANCE,
            ).exists()
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
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("jobs.tasks._gpu_available", return_value=False),
                patch("pytesseract.image_to_string", return_value="test text"),
            ):
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
            extract_paddle_lines=MagicMock(
                return_value=[
                    ("text", [[0, 0], [10, 0], [10, 10], [0, 10]], 0.9),
                ]
            ),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        try:
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("jobs.tasks._gpu_available", return_value=False),
            ):
                result = _process_single_page(
                    job,
                    "/test.pdf",
                    1,
                    job.source_hash[:16],
                    "w1",
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
            extract_paddle_lines=MagicMock(
                return_value=[
                    ("line1", [[0, 0], [100, 0], [100, 10], [0, 10]], 0.8),
                    ("line2", [[0, 20], [100, 20], [100, 30], [0, 30]], 0.6),
                ]
            ),
        )
        job, tmpdir = self._make_job_with_nfs()
        mock_fitz = self._make_fitz_mock(tmpdir)
        try:
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("jobs.tasks._gpu_available", return_value=False),
            ):
                result = _process_single_page(
                    job,
                    "/test.pdf",
                    1,
                    job.source_hash[:16],
                    "w1",
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
            with (
                patch.dict("sys.modules", self._ocr_mods(mock_utils, mock_fitz)),
                patch("jobs.tasks._gpu_available", return_value=False),
                patch("pytesseract.image_to_string", return_value="fallback"),
            ):
                _process_single_page(
                    job,
                    "/test.pdf",
                    1,
                    job.source_hash[:16],
                    "w1",
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
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
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

    def test_recovers_job_id_from_direct_celery_errback_shape(self):
        """Celery may call a multi-arg errback without the bound suffix arg."""
        job = self._make_job()
        request = MagicMock()
        request.id = "task-direct-errback"
        request.args = [str(job.job_id), 1]
        request.kwargs = {}

        result = chord_error_handler.run(request, RuntimeError("direct failure"))

        job.refresh_from_db()
        assert result["status"] == "failed"
        assert job.status == Job.Status.FAILED
        assert "direct failure" in job.error_message

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

    def test_late_errback_does_not_overwrite_cancelled_job(self):
        job = self._make_job()
        job.status = Job.Status.CANCELLED
        job.save(update_fields=["status"])
        request = MagicMock(id="late-error")

        result = chord_error_handler.run(
            request,
            RuntimeError("late header error"),
            None,
            str(job.job_id),
        )

        job.refresh_from_db()
        assert result["status"] == "cancelled"
        assert job.status == Job.Status.CANCELLED
        assert not CustodyEvent.objects.filter(
            job=job, event_type="chord_failed"
        ).exists()

    def test_late_errback_does_not_regress_completed_job(self):
        job = self._make_job()
        job.status = Job.Status.COMPLETED
        job.save(update_fields=["status"])
        request = MagicMock(id="late-error")

        result = chord_error_handler.run(
            request,
            RuntimeError("late header error"),
            None,
            str(job.job_id),
        )

        job.refresh_from_db()
        assert result["status"] == "completed"
        assert job.status == Job.Status.COMPLETED
        assert not CustodyEvent.objects.filter(
            job=job, event_type="chord_failed"
        ).exists()


# ---------------------------------------------------------------------------
# Step 5g: Worker task counter tests
# ---------------------------------------------------------------------------


class TestWorkerTaskCounters(TestCase):
    """Tests for M3 worker task counter increments."""

    def test_process_page_increments_completed(self):
        """process_page increments tasks_completed on success."""
        worker = Worker.objects.create(
            hostname=socket.gethostname(),
            status=Worker.Status.ONLINE,
            tasks_completed=0,
        )
        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
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
            status=Job.Status.PROCESSING,
            source_hash="abc123",
            total_pages=5,
            nfs_job_path=tempfile.mkdtemp(),
        )
        _ensure_job_dirs(job.nfs_job_path)
        # Create PageResult row for the page
        PageResult.objects.create(
            job=job,
            page_num=1,
            document_id="abc123abc123abc1",
        )

        with patch("jobs.tasks._process_single_page", side_effect=RuntimeError("fail")):
            with self.assertRaises(RuntimeError):
                process_page.run(str(job.job_id), 1)

        worker.refresh_from_db()
        assert worker.tasks_failed == 1
        self._cleanup(job.nfs_job_path)

    def test_compress_pdf_increments_completed(self):
        """compress_pdf increments tasks_completed on success."""
        from optimize_pdfs import OptimizationDisposition, OptimizeResult

        worker = Worker.objects.create(
            hostname=socket.gethostname(),
            status=Worker.Status.ONLINE,
            tasks_completed=0,
        )
        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.ASSEMBLING,
            source_hash="abc123",
            nfs_job_path=tempfile.mkdtemp(),
        )
        _ensure_job_dirs(job.nfs_job_path)
        # Create a dummy PDF file
        pdf_path = os.path.join(job.nfs_job_path, "output", "EXPORT", "PDF", "test.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.0 test content")
        assembly_event = CustodyEvent.objects.create(
            job=job,
            document_id=job.source_hash[:16],
            event_type="assembly_complete",
            data={"pages_assembled": job.total_pages},
        )

        mock_optimize_mod = types.ModuleType("optimize_pdfs")
        mock_optimize_mod.OptimizationDisposition = OptimizationDisposition
        mock_optimize_mod.optimize_pdf_detailed = MagicMock(
            return_value=OptimizeResult(OptimizationDisposition.APPLIED)
        )
        with patch.dict("sys.modules", {"optimize_pdfs": mock_optimize_mod}):
            result = compress_pdf.run(str(job.job_id))

        worker.refresh_from_db()
        assert worker.tasks_completed == 1
        assert result["status"] == "compressed"
        assert result["disposition"] == "APPLIED"
        assert result["pdf_sha256"] == _compute_file_hash(pdf_path)
        event = CustodyEvent.objects.get(
            job=job,
            event_type="compression_complete",
        )
        assert event.data["causal_event_id"] == assembly_event.id
        assert event.data["pdf_sha256"] == result["pdf_sha256"]
        self._cleanup(job.nfs_job_path)

    def test_compress_pdf_failed_optimizer_does_not_emit_complete(self):
        """Failed optimizer results must fail the Celery task (chord errback).

        No compression_complete custody event and no S3 upload after failure.
        """
        from optimize_pdfs import OptimizationDisposition, OptimizeResult

        worker = Worker.objects.create(
            hostname=socket.gethostname(),
            status=Worker.Status.ONLINE,
            tasks_completed=0,
            tasks_failed=0,
        )
        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.ASSEMBLING,
            source_hash="abc123def4567890",  # pragma: allowlist secret
            nfs_job_path=tempfile.mkdtemp(),
        )
        _ensure_job_dirs(job.nfs_job_path)
        pdf_path = os.path.join(job.nfs_job_path, "output", "EXPORT", "PDF", "test.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.0 test content")
        CustodyEvent.objects.create(
            job=job,
            document_id=job.source_hash[:16],
            event_type="assembly_complete",
            data={"pages_assembled": job.total_pages},
        )

        mock_optimize_mod = types.ModuleType("optimize_pdfs")
        mock_optimize_mod.OptimizationDisposition = OptimizationDisposition
        mock_optimize_mod.optimize_pdf_detailed = MagicMock(
            return_value=OptimizeResult(
                OptimizationDisposition.FAILED_PRESERVED,
                reason="visual integrity failed",
            )
        )
        with patch.dict("sys.modules", {"optimize_pdfs": mock_optimize_mod}):
            with patch("jobs.tasks._record_custody_event") as mock_custody:
                with patch("jobs.presigned_io.upload_presigned") as mock_upload:
                    with self.assertRaises(RuntimeError) as raised:
                        compress_pdf.run(str(job.job_id))

        worker.refresh_from_db()
        assert worker.tasks_failed == 1
        assert worker.tasks_completed == 0
        assert "FAILED_PRESERVED" in str(raised.exception)
        assert "visual integrity failed" in str(raised.exception)
        mock_custody.assert_not_called()
        mock_upload.assert_not_called()
        self._cleanup(job.nfs_job_path)

    def test_compress_pdf_visual_budget_skip_emits_skipped(self):
        """SKIPPED_VISUAL_BUDGET emits compression_skipped, not complete."""
        from optimize_pdfs import OptimizationDisposition, OptimizeResult

        worker = Worker.objects.create(
            hostname=socket.gethostname(),
            status=Worker.Status.ONLINE,
            tasks_completed=0,
        )
        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.ASSEMBLING,
            source_hash="abc123def4567890",  # pragma: allowlist secret
            nfs_job_path=tempfile.mkdtemp(),
        )
        _ensure_job_dirs(job.nfs_job_path)
        pdf_path = os.path.join(job.nfs_job_path, "output", "EXPORT", "PDF", "test.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.0 test content")
        assembly_event = CustodyEvent.objects.create(
            job=job,
            document_id=job.source_hash[:16],
            event_type="assembly_complete",
            data={"pages_assembled": job.total_pages},
        )

        mock_optimize_mod = types.ModuleType("optimize_pdfs")
        mock_optimize_mod.OptimizationDisposition = OptimizationDisposition
        mock_optimize_mod.optimize_pdf_detailed = MagicMock(
            return_value=OptimizeResult(
                OptimizationDisposition.SKIPPED_VISUAL_BUDGET,
                reason="page 1 exceeds compression visual-validation budget",
            )
        )
        with patch.dict("sys.modules", {"optimize_pdfs": mock_optimize_mod}):
            with patch("jobs.tasks._record_custody_event") as mock_custody:
                result = compress_pdf.run(str(job.job_id))

        worker.refresh_from_db()
        assert worker.tasks_completed == 1
        assert result["status"] == "skipped_visual_budget"
        assert result["disposition"] == "SKIPPED_VISUAL_BUDGET"
        mock_custody.assert_called_once()
        assert mock_custody.call_args[0][2] == "compression_skipped"
        assert mock_custody.call_args.kwargs["data"]["causal_event_id"] == (
            assembly_event.id
        )
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
        text_path = os.path.join(
            job.nfs_job_path, "output", "EXPORT", "TEXT", "test.txt"
        )
        with open(text_path, "w") as f:
            f.write("Sample text for NER testing")

        mock_ner_mod = MagicMock()
        mock_ner_mod.extract_entities.return_value = [
            {"entity": "test", "type": "PERSON"}
        ]
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
        Job.objects.filter(job_id__in=[j1.job_id, j2.job_id]).update(
            created_at=old_date, completed_at=old_date
        )

        result = cleanup_completed_jobs()

        assert result["deleted"] == 2
        assert Job.objects.filter(job_id=j1.job_id).count() == 0
        assert Job.objects.filter(job_id=j2.job_id).count() == 0

    def test_preserves_recent_jobs(self):
        """Recent completed jobs within retention period are not deleted."""
        job = Job.objects.create(source_file="/recent.pdf", status=Job.Status.COMPLETED)
        # created_at is auto_now_add (set to now), within 30-day retention

        result = cleanup_completed_jobs()

        assert result["deleted"] == 0
        assert Job.objects.filter(job_id=job.job_id).exists()

    def test_returns_count(self):
        """Task returns a dict with 'deleted' and 'nfs_cleaned' counts."""
        j1 = Job.objects.create(source_file="/c1.pdf", status=Job.Status.COMPLETED)
        j2 = Job.objects.create(source_file="/c2.pdf", status=Job.Status.CANCELLED)
        old_date = timezone.now() - timezone.timedelta(days=45)
        Job.objects.filter(job_id__in=[j1.job_id, j2.job_id]).update(
            created_at=old_date, completed_at=old_date
        )

        result = cleanup_completed_jobs()

        assert "deleted" in result
        assert "nfs_cleaned" in result
        assert result["deleted"] == 2
        assert result["nfs_cleaned"] == 0  # no nfs_job_path set

    @patch("jobs.tasks._get_backend_for_job")
    def test_retention_fence_blocks_failed_job_retry_before_storage(self, backend_for):
        backend = MagicMock(backend_name="nfs")
        attempted = []
        job = Job.objects.create(
            source_file="/retry-race.pdf", status=Job.Status.FAILED
        )
        old_date = timezone.now() - timezone.timedelta(days=45)
        Job.objects.filter(pk=job.pk).update(completed_at=old_date)

        def observe_fence(locked_job):
            attempted.append(
                Job.objects.filter(pk=locked_job.pk, status=Job.Status.FAILED).update(
                    status=Job.Status.SUBMITTED,
                    dispatch_generation=models.F("dispatch_generation") + 1,
                )
            )
            return backend

        backend_for.side_effect = observe_fence
        result = cleanup_completed_jobs()

        assert attempted == [0]
        assert result["deleted"] == 1
        assert not Job.objects.filter(pk=job.pk).exists()


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
        object_keys = [
            "jobs/key-1/source/file.pdf",
            "jobs/key-1/output/EXPORT/PDF/result.pdf",
        ]
        # Each locked job is listed before and after deletion; the second
        # listing must prove the backend is empty before DB rows are removed.
        backend.list_objects.side_effect = [object_keys, [], object_keys, []]
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
        Job.objects.filter(job_id__in=[j1.job_id, j2.job_id]).update(
            created_at=old_date, completed_at=old_date
        )

        result = cleanup_completed_jobs()

        assert result["deleted"] == 2
        assert result["nfs_cleaned"] == 0
        assert result["s3_jobs_cleaned"] == 2
        assert result["s3_objects_cleaned"] == 4
        assert backend.list_objects.call_count == 4
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
        Job.objects.filter(job_id=job.job_id).update(
            created_at=old_date, completed_at=old_date
        )

        # Storage failure must leave the row retryable rather than claiming
        # successful deletion while its objects may still contain PII.
        result = cleanup_completed_jobs()
        assert result["deleted"] == 0
        assert result["cleanup_failures"]
        assert Job.objects.filter(job_id=job.job_id).exists()


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

    def test_strict_per_gpu_contract_persists_stable_logical_queue(self):
        with patch.dict(
            os.environ,
            {
                "OCR_TASK_ROUTING": "gpu",
                "ENABLE_PER_GPU_QUEUES": "true",
                "OCR_PER_GPU_STRICT": "true",
                "GPU_COUNT": "2",
            },
        ):
            from jobs.tasks import _get_ocr_queue

            assert _get_ocr_queue() == "ocr_gpu"

    def test_per_gpu_routing_falls_back_without_strict_contract(self):
        with patch.dict(
            os.environ,
            {
                "OCR_TASK_ROUTING": "gpu",
                "ENABLE_PER_GPU_QUEUES": "true",
                "OCR_PER_GPU_STRICT": "false",
                "GPU_COUNT": "2",
            },
        ):
            from jobs.tasks import _get_ocr_queue

            assert _get_ocr_queue() == "ocr_gpu"

    @patch("jobs.tasks.process_document.apply_async")
    @patch("jobs.tasks.extract_pages.apply_async")
    def test_separated_lane_forces_page_fanout_for_small_document(
        self,
        mock_extract_pages_apply,
        mock_process_document_apply,
    ):
        """Page leases, not whole-document retries, fence engine handoff."""
        job = Job.objects.create(
            source_file="/test.pdf",
            source_type="pdf",
            status=Job.Status.INGESTING,
            total_pages=1,
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
        )

        with patch.dict(os.environ, {"OCR_SEPARATE_TESSERACT_LANE": "true"}):
            result = _dispatch_processing(job, job.job_id, 1)

        assert result["mode"] == "capability_fanout"
        mock_extract_pages_apply.assert_called_once_with(
            args=[str(job.job_id)],
            kwargs={"dispatch_generation": 1},
            queue="coordinator",
            priority=5,
        )
        mock_process_document_apply.assert_not_called()


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
        """S3 mode uploads source and dispatches page-fenced OCR for a small PDF."""
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
                # One page still receives its own leased PageResult.
                mock_extract_pages_apply.assert_called_once_with(
                    args=[str(job.job_id)],
                    kwargs={"dispatch_generation": 1},
                    queue="coordinator",
                    priority=5,
                )
                mock_process_document_apply.assert_not_called()
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
                    args=[str(job.job_id)],
                    kwargs={"dispatch_generation": 1},
                    queue="coordinator",
                    priority=5,
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

        with (
            tempfile.TemporaryDirectory() as allowed_root,
            tempfile.TemporaryDirectory() as other_root,
        ):
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
    def test_s3_mode_leases_shared_source_before_processing(
        self, mock_process, mock_get_storage_backend
    ):
        """S3 page fan-out uses a revision-bound shared source lease."""
        class FencedPath(str):
            def check(self):
                return "source-intact"

        source_path = FencedPath("/tmp/cached-source.pdf")
        backend = MagicMock(spec=CachedS3Backend)
        backend.backend_name = "s3"
        backend.source_revision_info.return_value = ("revision-1", 1024)
        backend.lease_source.return_value = nullcontext(source_path)
        mock_get_storage_backend.return_value = backend
        mock_process.return_value = {"page_num": 1, "status": "ok"}

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=5,
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
            detected_language="en",
        )

        process_page.run(str(job.job_id), 1)

        backend.lease_source.assert_called_once()
        identity = backend.lease_source.call_args.args[0]
        assert identity.job_id == str(job.pk)
        assert identity.object_key == f"jobs/{job.job_id}/source/test.pdf"
        assert identity.sha256 == job.source_hash
        assert identity.revision == "revision-1"
        assert mock_process.call_args.args[1] == "/tmp/cached-source.pdf"
        assert mock_process.call_args.kwargs["source_fence"]() == "source-intact"
        mock_process.assert_called_once()

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_download_failure_marks_page_failed(self, mock_get_storage_backend):
        backend = MagicMock(spec=CachedS3Backend)
        backend.backend_name = "s3"
        backend.source_revision_info.return_value = ("revision-1", 1024)
        backend.lease_source.side_effect = RuntimeError("download failed")
        mock_get_storage_backend.return_value = backend

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
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
    @patch("jobs.tasks.chord")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_assembly_downloads_winning_attempt_keys(self, mock_backend, mock_chord):
        import fitz

        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = True
        mock_backend.return_value = backend
        mock_chord.return_value = MagicMock()

        def download_page(key, local_path):
            if key.endswith(".pdf"):
                with fitz.open() as page:
                    page.new_page(width=100, height=100)
                    page.save(local_path)
            else:
                with open(local_path, "w", encoding="utf-8") as output:
                    output.write("winner")

        backend.download_file.side_effect = download_page
        job = Job.objects.create(
            source_file="test.pdf", status=Job.Status.PROCESSING,
            total_pages=1, source_hash="abcdef1234567890" * 4,
        )
        prefix = f"jobs/{job.pk}/temp/{job.source_hash[:16]}/g0/p1/lwinner"
        pdf_key = f"{prefix}/1.pdf"
        text_key = f"{prefix}/1.txt"
        PageResult.objects.create(
            job=job, page_num=1, document_id=job.source_hash[:16],
            status="ok", admitted_generation=job.dispatch_generation,
            temp_pdf_path=pdf_key,
            ocr_provenance={
                "temp_text_path": text_key,
                "text_sha256": hashlib.sha256(b"winner").hexdigest(),
            },
        )
        result = assemble_document.run(str(job.pk))
        assert result["status"] == "assembled"
        assert [call.args[0] for call in backend.download_file.call_args_list] == [
            pdf_key, text_key,
        ]

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks.chord")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_downloaded_text_tamper_fails_before_publication(self, mock_backend, mock_chord):
        import fitz

        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = True
        mock_backend.return_value = backend
        mock_chord.return_value = MagicMock()

        def download_page(key, local_path):
            if key.endswith(".pdf"):
                with fitz.open() as page:
                    page.new_page(width=100, height=100)
                    page.save(local_path)
            else:
                with open(local_path, "wb") as output:
                    output.write(b"mutated")

        backend.download_file.side_effect = download_page
        job = Job.objects.create(
            source_file="test.pdf", status=Job.Status.PROCESSING,
            total_pages=1, source_hash="abcdef1234567890" * 4,
        )
        PageResult.objects.create(
            job=job, page_num=1, document_id=job.source_hash[:16],
            status="ok", admitted_generation=job.dispatch_generation,
            ocr_provenance={
                "text_sha256": hashlib.sha256(b"original").hexdigest(),
            },
        )
        result = assemble_document.run(str(job.job_id))
        assert result["status"] == "failed"
        failure = CustodyEvent.objects.get(job=job, event_type="assembly_failed")
        assert failure.data["failure_kind"] == "page_text_digest_mismatch"
        assert not CustodyEvent.objects.filter(job=job, event_type="assembly_complete").exists()
        assert backend.upload_file.call_count == 0

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks.publish_task_dispatch_outbox")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_mode_uploads_final_artifacts(
        self, mock_get_storage_backend, mock_publish_outbox
    ):
        """S3 mode uploads artifacts and durably enqueues post-processing."""
        import fitz

        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = True
        mock_get_storage_backend.return_value = backend

        def download_page(key, local_path):
            if key.endswith(".pdf"):
                page = fitz.open()
                page.new_page(width=100, height=100)
                page.save(local_path)
                page.close()
            else:
                with open(local_path, "w", encoding="utf-8") as text_file:
                    text_file.write("page text")

        backend.download_file.side_effect = download_page

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=1,
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
        )
        PageResult.objects.create(
            job=job, page_num=1, document_id=job.source_hash[:16],
            status="ok", admitted_generation=job.dispatch_generation,
            ocr_provenance={
                "text_sha256": hashlib.sha256(b"page text").hexdigest(),
            },
        )

        with self.captureOnCommitCallbacks(execute=True):
            result = assemble_document.run(str(job.job_id))

        assert result["status"] == "assembled"
        assert backend.download_file.call_count == 2
        assert backend.upload_file.call_count == 2
        row = TaskDispatchOutbox.objects.get(job=job, stage="postprocessing")
        assert row.task_name == "jobs.tasks.dispatch_postprocessing"
        assert row.status == TaskDispatchOutbox.Status.PENDING
        mock_publish_outbox.assert_called_once_with(dispatch_id=row.dispatch_id)

    @override_settings(STORAGE_BACKEND="s3")
    @patch("tempfile.mkdtemp")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_mode_cleans_temp_dir_when_upload_fails(
        self, mock_get_storage_backend, mock_mkdtemp
    ):
        import fitz

        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = True
        backend.upload_file.side_effect = RuntimeError("upload failed")
        mock_get_storage_backend.return_value = backend

        def download_page(key, local_path):
            if key.endswith(".pdf"):
                page = fitz.open()
                page.new_page(width=100, height=100)
                page.save(local_path)
                page.close()
            else:
                with open(local_path, "w", encoding="utf-8") as text_file:
                    text_file.write("page text")

        backend.download_file.side_effect = download_page

        work_dir = os.path.join(
            tempfile.gettempdir(), f"assemble_fail_{uuid.uuid4().hex}"
        )
        os.makedirs(work_dir, exist_ok=True)
        mock_mkdtemp.return_value = work_dir

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
            total_pages=1,
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
        )
        PageResult.objects.create(
            job=job, page_num=1, document_id=job.source_hash[:16],
            status="ok", admitted_generation=job.dispatch_generation,
            ocr_provenance={
                "text_sha256": hashlib.sha256(b"page text").hexdigest(),
            },
        )

        try:
            with patch.object(
                assemble_document,
                "retry",
                side_effect=RuntimeError("retry scheduled"),
            ):
                with self.assertRaisesRegex(RuntimeError, "retry scheduled"):
                    assemble_document.run(str(job.job_id))
            assert not os.path.isdir(work_dir)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


class TestProcessDocumentS3(TestCase):
    """A stale whole-document broker delivery cannot download or publish."""

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._get_storage_backend")
    @patch("jobs.tasks.extract_pages.run")
    def test_redirect_uses_control_plane_page_admission(self, extract, backend):
        job = Job.objects.create(
            source_file="test.pdf", source_type="pdf", status=Job.Status.PROCESSING,
            dispatch_generation=1, source_hash="abcdef1234567890" * 4,
        )
        extract.return_value = {"status": "fanout_dispatched"}
        assert process_document.run(str(job.job_id), dispatch_generation=1)["status"] == "fanout_dispatched"
        extract.assert_called_once()
        backend.assert_not_called()


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
            status=Job.Status.ASSEMBLING,
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
        )
        CustodyEvent.objects.create(
            job=job,
            document_id=job.source_hash[:16],
            event_type="assembly_complete",
            data={"pages_assembled": job.total_pages},
        )

        from optimize_pdfs import OptimizationDisposition, OptimizeResult

        mock_optimize_mod = types.ModuleType("optimize_pdfs")
        mock_optimize_mod.OptimizationDisposition = OptimizationDisposition
        mock_optimize_mod.optimize_pdf_detailed = MagicMock(
            return_value=OptimizeResult(OptimizationDisposition.APPLIED)
        )
        with patch.dict(sys.modules, {"optimize_pdfs": mock_optimize_mod}):
            result = compress_pdf.run(str(job.job_id))

        assert result["status"] == "compressed"
        # Verify exists check
        backend.exists.assert_called_once()
        # Verify download
        backend.download_file.assert_called_once()
        # Verify optimization
        mock_optimize_mod.optimize_pdf_detailed.assert_called_once()
        # Verify upload
        backend.upload_file.assert_called_once()
        pdf_path = backend.download_file.call_args[0][1]
        assert not os.path.isdir(os.path.dirname(pdf_path))

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._get_storage_backend")
    def test_compression_invalidates_precompression_s3_cache(
        self, mock_get_storage_backend
    ):
        from optimize_pdfs import OptimizationDisposition, OptimizeResult

        inner = MagicMock(backend_name="s3")
        inner.endpoint = "https://s3.example.test"
        inner.bucket = "test-bucket"
        inner.region = "us-east-1"
        inner.exists.return_value = True
        with tempfile.TemporaryDirectory(prefix="compression_cache_") as cache_dir:
            backend = CachedS3Backend(inner=inner, cache_dir=cache_dir)
            mock_get_storage_backend.return_value = backend
            job = Job.objects.create(
                source_file="test.pdf",
                status=Job.Status.ASSEMBLING,
                source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
            )
            CustodyEvent.objects.create(
                job=job,
                document_id=job.source_hash[:16],
                event_type="assembly_complete",
                data={"pages_assembled": job.total_pages},
            )
            pdf_key = _job_storage_key(
                job.job_id,
                "output/EXPORT/PDF/test.pdf",
            )
            cached_bytes = b"%PDF-1.4 assembled pre-compression bytes"
            revision = ("assembled-etag", "v1", len(cached_bytes))
            inner.object_revision.return_value = revision
            cached_path = backend._cache_path(pdf_key, revision)
            with open(cached_path, "wb") as cached_pdf:
                cached_pdf.write(cached_bytes)

            mock_optimize_mod = types.ModuleType("optimize_pdfs")
            mock_optimize_mod.OptimizationDisposition = OptimizationDisposition
            mock_optimize_mod.optimize_pdf_detailed = MagicMock(
                return_value=OptimizeResult(OptimizationDisposition.APPLIED)
            )
            with patch.dict(sys.modules, {"optimize_pdfs": mock_optimize_mod}):
                result = compress_pdf.run(str(job.job_id))

            assert result["status"] == "compressed"
            inner.upload_file.assert_called_once()
            assert not os.path.exists(cached_path)

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_mode_no_pdf_raises_for_chord_errback_and_cleans_up(
        self, mock_get_storage_backend
    ):
        """Missing assembled PDF is a task failure, never a normal skip."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = False
        mock_get_storage_backend.return_value = backend

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.ASSEMBLING,
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
        )
        CustodyEvent.objects.create(
            job=job,
            document_id=job.source_hash[:16],
            event_type="assembly_complete",
            data={"pages_assembled": job.total_pages},
        )

        with self.assertRaises(FileNotFoundError):
            compress_pdf.run(str(job.job_id))

        backend.exists.assert_called_once()
        backend.download_file.assert_not_called()

    @override_settings(STORAGE_BACKEND="s3")
    @patch("jobs.tasks._get_storage_backend")
    def test_s3_download_failure_raises_for_chord_errback(
        self, mock_get_storage_backend
    ):
        backend = MagicMock(backend_name="s3")
        backend.exists.return_value = True
        backend.download_file.side_effect = RuntimeError("download failed")
        mock_get_storage_backend.return_value = backend
        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.ASSEMBLING,
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
        )
        CustodyEvent.objects.create(
            job=job,
            document_id=job.source_hash[:16],
            event_type="assembly_complete",
            data={"pages_assembled": job.total_pages},
        )

        with self.assertRaisesRegex(RuntimeError, "download failed"):
            compress_pdf.run(str(job.job_id))

        backend.upload_file.assert_not_called()


class TestCompressionIdempotency(TestCase):
    """Compression redelivery and execution-lock regressions."""

    def test_terminal_jobs_skip_optimizer_and_custody(self):
        for status in (
            Job.Status.COMPLETED,
            Job.Status.FAILED,
            Job.Status.CANCELLED,
        ):
            with self.subTest(status=status):
                job = Job.objects.create(
                    source_file="terminal.pdf",
                    status=status,
                )
                with patch("jobs.tasks._compress_pdf_once") as compress_once:
                    result = compress_pdf.run(str(job.job_id))

                assert result["status"] == "already_finalized"
                assert result["terminal_status"] == status
                compress_once.assert_not_called()
                assert not CustodyEvent.objects.filter(job=job).exists()

    def test_existing_compression_event_is_an_idempotent_redelivery(self):
        job = Job.objects.create(
            source_file="document.pdf",
            status=Job.Status.ASSEMBLING,
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
        )
        assembly = CustodyEvent.objects.create(
            job=job,
            document_id=job.source_hash[:16],
            event_type="assembly_complete",
            data={"pages_assembled": 1},
        )
        existing = CustodyEvent.objects.create(
            job=job,
            document_id=job.source_hash[:16],
            event_type="compression_complete",
            data={
                "causal_event_id": assembly.id,
                "disposition": "APPLIED",
                "pdf_sha256": "a" * 64,
            },
        )

        with patch("jobs.tasks._compress_pdf_once") as compress_once:
            result = compress_pdf.run(str(job.job_id))

        assert result["status"] == "compressed"
        assert result["custody_event_id"] == existing.id
        assert result["idempotent_redelivery"] is True
        compress_once.assert_not_called()
        assert (
            CustodyEvent.objects.filter(
                job=job,
                event_type="compression_complete",
            ).count()
            == 1
        )

    def test_process_local_fallback_serializes_same_job(self):
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()
        errors = []

        def first_worker():
            try:
                with _compression_execution_lock("same-job"):
                    first_entered.set()
                    if not release_first.wait(timeout=3):
                        raise TimeoutError("test did not release first worker")
            except BaseException as exc:  # pragma: no cover - diagnostic path
                errors.append(exc)

        def second_worker():
            try:
                if not first_entered.wait(timeout=3):
                    raise TimeoutError("first worker did not acquire lock")
                with _compression_execution_lock("same-job"):
                    second_entered.set()
            except BaseException as exc:  # pragma: no cover - diagnostic path
                errors.append(exc)

        fake_connection = MagicMock(vendor="sqlite")
        with patch("jobs.tasks.connection", fake_connection):
            first_thread = threading.Thread(target=first_worker)
            second_thread = threading.Thread(target=second_worker)
            first_thread.start()
            assert first_entered.wait(timeout=3)
            second_thread.start()
            assert not second_entered.wait(timeout=0.1)
            release_first.set()
            first_thread.join(timeout=3)
            second_thread.join(timeout=3)

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert not errors
        assert second_entered.is_set()

    def test_postgres_lock_sql_order_and_exception_unlock(self):
        calls = []

        class FakeCursor:
            def __init__(self, owner):
                self.owner = owner

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def execute(self, sql, params):
                calls.append((sql, params))

            def fetchone(self):
                return (self.owner.unlock_result,)

        class FakeConnection:
            vendor = "postgresql"
            unlock_result = True
            closed = False

            def cursor(self):
                return FakeCursor(self)

            def close(self):
                self.closed = True

        fake_connection = FakeConnection()
        with patch("jobs.tasks.connection", fake_connection):
            with self.assertRaisesRegex(ValueError, "body failed"):
                with _compression_execution_lock("job-42"):
                    calls.append("body")
                    raise ValueError("body failed")

        import jobs.tasks as tasks_module

        expected_key = tasks_module._compression_advisory_key("job-42")
        assert calls == [
            ("SELECT pg_advisory_lock(%s)", [expected_key]),
            "body",
            ("SELECT pg_advisory_unlock(%s)", [expected_key]),
        ]
        assert fake_connection.closed is False

    def test_postgres_unlock_failure_closes_connection(self):
        class FakeCursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def execute(self, sql, params):
                return None

            def fetchone(self):
                return (False,)

        class FakeConnection:
            vendor = "postgresql"
            closed = False

            def cursor(self):
                return FakeCursor()

            def close(self):
                self.closed = True

        fake_connection = FakeConnection()
        with patch("jobs.tasks.connection", fake_connection):
            with self.assertRaisesRegex(RuntimeError, "advisory unlock failed"):
                with _compression_execution_lock("job-42"):
                    pass

        assert fake_connection.closed is True


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
        fake_ner.extract_entities = MagicMock(
            return_value=[
                {"text": "John Doe", "label": "PERSON"},
                {"text": "New York", "label": "LOCATION"},
            ]
        )

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
            source_hash="abcdef1234567890" * 4,  # pragma: allowlist secret
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
    def test_s3_mode_no_text_raises_for_chord_errback_and_cleans_up(
        self, mock_get_storage_backend
    ):
        """S3 mode with no text file must fail the requested NER header."""
        backend = MagicMock()
        backend.backend_name = "s3"
        backend.exists.return_value = False
        mock_get_storage_backend.return_value = backend

        job = Job.objects.create(
            source_file="test.pdf",
            status=Job.Status.PROCESSING,
        )

        with self.assertRaisesRegex(RuntimeError, "NER text object is missing"):
            extract_entities.run(str(job.job_id))
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
    def test_fails_closed_on_persisted_backend_creation_error(
        self, mock_get_storage_backend, mock_create_backend
    ):
        """A persisted backend mismatch must not fail open to another backend."""
        current_backend = MagicMock()
        current_backend.backend_name = "s3"
        mock_get_storage_backend.return_value = current_backend

        mock_create_backend.side_effect = ValueError(
            "Unsupported storage backend 'invalid'"
        )

        job = Job.objects.create(
            source_file="/test.pdf",
            storage_backend_used="invalid",
        )

        with self.assertRaisesRegex(RuntimeError, "cannot be created"):
            _get_backend_for_job(job)
        assert mock_get_storage_backend.call_count == 1
