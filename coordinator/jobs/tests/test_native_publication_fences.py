"""Native-text publication uses a unique document execution owner."""

import hashlib
import os
import socket
import tempfile
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest
from celery.exceptions import Retry
from django.test import TestCase, override_settings
from django.utils import timezone

from jobs import tasks
from jobs.models import (
    CustodyEvent,
    Job,
    PageResult,
    PresignedTaskCapability,
    TaskDispatchOutbox,
    Worker,
)
from jobs.presigned import text_only_attempt_identity
from jobs.storage import NFSBackend
from jobs.tasks import (
    _claim_processing_lease,
    _commit_native_page_if_owned,
    _enqueue_assembly_dispatch,
    _fail_processing_job_if_owned,
    _native_text_attempt_suffix,
    _ProcessingLeaseLost,
    _renew_or_discard_published_artifacts,
    process_text_only,
)


class NativePublicationFenceTests(TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.job_path = self.root / "job"
        (self.job_path / "source").mkdir(parents=True)
        self.source = self.job_path / "source" / "input.txt"
        self.source.write_text("NATIVE TEXT EVIDENCE", encoding="utf-8")
        self.job = Job.objects.create(
            source_file="input.txt", source_type="text",
            source_hash=hashlib.sha256(self.source.read_bytes()).hexdigest(),
            status=Job.Status.PROCESSING, total_pages=1,
            dispatch_generation=1, nfs_job_path=str(self.job_path),
        )

    def _defaults(self, locator, document_id=None):
        return {
            "document_id": document_id or self.job.source_hash[:16], "status": "ok",
            "ocr_method": "NativeText", "ocr_language": "en",
            "ocr_confidence": 1.0, "text_length": 6,
            "ocr_provenance": {"text_sha256": hashlib.sha256(b"sample").hexdigest()},
            "worker_hostname": "native-test", "processing_time_ms": 1,
            "temp_pdf_path": locator,
        }

    def _attempt_paths(self, token, page_num=1):
        base = (
            self.job_path / "temp" / self.job.source_hash[:16]
            / _native_text_attempt_suffix(self.job, token)
        )
        return base / f"{page_num}.pdf", base / f"{page_num}.txt"

    def test_coordinator_pool_uses_bounded_native_watchdog(self):
        with (
            patch.dict(os.environ, {"OCR_NATIVE_CALL_TIMEOUT_SECONDS": "300"}),
            patch.object(tasks, "_OCR_ENGINE_POOL", None),
            patch.object(tasks, "_OCR_ENGINE_POOL_PID", None),
            patch.object(tasks, "EngineRuntimePool") as factory,
            patch.object(tasks.atexit, "register"),
        ):
            tasks._ocr_runtime_pool()
        assert factory.call_args.kwargs["native_call_timeout_seconds"] == 300

    def test_same_broker_id_gets_distinct_execution_namespace_after_expiry(self):
        first = _claim_processing_lease(self.job, "same-celery-id")
        assert first and len(first) == 32
        assert _claim_processing_lease(self.job, "same-celery-id") is None
        Job.objects.filter(pk=self.job.pk).update(
            processing_lease_expires_at=timezone.now() - timezone.timedelta(seconds=1)
        )
        second = _claim_processing_lease(self.job, "same-celery-id")
        assert second and second != first
        assert _native_text_attempt_suffix(self.job, first) != _native_text_attempt_suffix(self.job, second)
        self.job.refresh_from_db()
        assert self.job.celery_task_id == "same-celery-id"
        with pytest.raises(_ProcessingLeaseLost):
            _commit_native_page_if_owned(
                self.job, first, 1, self.job.source_hash[:16],
                self._defaults("old-attempt.pdf"), text_locator="old-attempt.txt",
            )
        assert _commit_native_page_if_owned(
            self.job, second, 1, self.job.source_hash[:16],
            self._defaults("new-attempt.pdf"), text_locator="new-attempt.txt",
        )
        page = PageResult.objects.get(job=self.job, page_num=1)
        self.job.refresh_from_db()
        assert page.temp_pdf_path == "new-attempt.pdf"
        assert page.ocr_provenance["temp_text_path"] == "new-attempt.txt"
        assert page.admitted_generation == 1 and page.celery_task_id == second
        assert self.job.pages_completed == 1

    def test_duplicate_live_delivery_retries_until_nonce_can_be_reclaimed(self):
        assert _claim_processing_lease(self.job, "same-celery-id")
        process_text_only.push_request(id="same-celery-id", retries=0)
        try:
            with patch.object(process_text_only, "retry", side_effect=Retry()) as retry:
                with pytest.raises(Retry):
                    process_text_only.run(str(self.job.job_id), dispatch_generation=1)
            assert 1 <= retry.call_args.kwargs["countdown"] <= 60
        finally:
            process_text_only.pop_request()
        self.job.refresh_from_db()
        assert self.job.status == Job.Status.PROCESSING
        assert self.job.processing_attempt_token

    def test_native_lease_is_shorter_than_custom_job_timeout(self):
        self.job.settings_json = {"processing_timeout_minutes": 1}
        self.job.save(update_fields=["settings_json"])
        with patch.dict(os.environ, {"OCR_NATIVE_PROCESSING_LEASE_SECONDS": "1800"}):
            assert _claim_processing_lease(self.job, "delivery")
        self.job.refresh_from_db()
        remaining = (self.job.processing_lease_expires_at - timezone.now()).total_seconds()
        assert 0 < remaining <= 30

    def test_budget_failure_after_parent_replacement_cannot_fail_new_job(self):
        def replace_then_reject(_job):
            Job.objects.filter(pk=self.job.pk).update(dispatch_generation=2)
            raise ValueError("too many pages")

        with patch("jobs.tasks._validate_job_page_budget", side_effect=replace_then_reject):
            result = process_text_only.run(str(self.job.job_id), dispatch_generation=1)
        assert result["status"] == "stale"
        self.job.refresh_from_db()
        assert self.job.status == Job.Status.PROCESSING
        assert self.job.dispatch_generation == 2

    def test_foreign_or_noncanonical_terminal_page_cannot_be_reused(self):
        first = _claim_processing_lease(self.job, "same-task-id")
        first_pdf, first_txt = self._attempt_paths(first)
        assert _commit_native_page_if_owned(
            self.job, first, 1, self.job.source_hash[:16],
            self._defaults(str(first_pdf)), text_locator=str(first_txt),
        )
        Job.objects.filter(pk=self.job.pk).update(
            processing_lease_expires_at=timezone.now() - timezone.timedelta(seconds=1)
        )
        second = _claim_processing_lease(self.job, "same-task-id")
        self.job.refresh_from_db()
        second_pdf, second_txt = self._attempt_paths(second)
        page = PageResult.objects.get(job=self.job, page_num=1)
        original_method = page.ocr_method
        page.ocr_method = "Tesseract"
        page.save(update_fields=["ocr_method"])
        with pytest.raises(_ProcessingLeaseLost, match="foreign terminal winner"):
            _commit_native_page_if_owned(
                self.job, second, 1, self.job.source_hash[:16],
                self._defaults(str(second_pdf)), text_locator=str(second_txt),
            )
        page.ocr_method = original_method
        page.temp_pdf_path = "other-owner.pdf"
        page.save(update_fields=["ocr_method", "temp_pdf_path"])
        with pytest.raises(_ProcessingLeaseLost, match="not canonical"):
            _commit_native_page_if_owned(
                self.job, second, 1, self.job.source_hash[:16],
                self._defaults(str(second_pdf)), text_locator=str(second_txt),
            )
        assert Job.objects.get(pk=self.job.pk).pages_completed == 1

    def test_cancel_after_preflight_rejects_terminal_page_and_custody(self):
        token = _claim_processing_lease(self.job, "delivery")
        from jobs import tasks
        original = tasks._renew_processing_lease

        def revoke_after_renew(*args, **kwargs):
            original(*args, **kwargs)
            Job.objects.filter(pk=self.job.pk).update(status=Job.Status.CANCELLED)

        with patch("jobs.tasks._renew_processing_lease", side_effect=revoke_after_renew):
            with pytest.raises(_ProcessingLeaseLost):
                _commit_native_page_if_owned(
                    self.job, token, 1, self.job.source_hash[:16],
                    self._defaults("cancelled.pdf"), text_locator="cancelled.txt",
                    custody_data={"page_num": 1},
                )
        self.job.refresh_from_db()
        assert self.job.pages_completed == 0
        assert not PageResult.objects.filter(job=self.job).exists()
        assert not CustodyEvent.objects.filter(job=self.job, event_type="text_extracted_natively").exists()

    def test_generation_and_source_change_deny_success_and_failure(self):
        token = _claim_processing_lease(self.job, "delivery")
        old_job = Job.objects.get(pk=self.job.pk)
        Job.objects.filter(pk=self.job.pk).update(
            dispatch_generation=2, source_hash="b" * 64,
        )
        with pytest.raises(_ProcessingLeaseLost):
            _commit_native_page_if_owned(
                old_job, token, 1, old_job.source_hash[:16],
                self._defaults("stale.pdf"), text_locator="stale.txt",
            )
        assert _fail_processing_job_if_owned(
            self.job.pk, token, "stale failure", expected_job=old_job,
        ) == 0
        self.job.refresh_from_db()
        assert self.job.status == Job.Status.PROCESSING
        assert self.job.pages_completed == 0
        assert not PageResult.objects.filter(job=self.job).exists()

    def test_identity_replacement_allows_immediate_new_attempt_without_old_grant(self):
        old = _claim_processing_lease(self.job, "same-celery-id")
        Job.objects.filter(pk=self.job.pk).update(
            dispatch_generation=2, source_hash="b" * 64,
        )
        fresh = Job.objects.get(pk=self.job.pk)
        replacement = _claim_processing_lease(fresh, "same-celery-id")
        assert replacement and replacement != old
        fresh.refresh_from_db()
        assert fresh.processing_attempt_identity == text_only_attempt_identity(fresh)
        assert fresh.processing_attempt_token == replacement
        assert fresh.processing_lease_expires_at > timezone.now()
        with pytest.raises(_ProcessingLeaseLost):
            _commit_native_page_if_owned(
                self.job, old, 1, self.job.source_hash[:16],
                self._defaults("old.pdf"), text_locator="old.txt",
            )

    def test_broker_replacement_allows_immediate_new_attempt(self):
        old = _claim_processing_lease(self.job, "broker-A")
        Job.objects.filter(pk=self.job.pk).update(celery_task_id="broker-B")
        fresh = Job.objects.get(pk=self.job.pk)
        new = _claim_processing_lease(fresh, "broker-B")
        assert new and new != old
        assert _fail_processing_job_if_owned(
            self.job.pk, old, "old worker exception", expected_job=self.job,
        ) == 0

    def test_same_generation_source_replacement_cannot_reuse_terminal_winner(self):
        first = _claim_processing_lease(self.job, "broker-A")
        assert _commit_native_page_if_owned(
            self.job, first, 1, self.job.source_hash[:16],
            self._defaults("source-A.pdf"), text_locator="source-A.txt",
        )
        Job.objects.filter(pk=self.job.pk).update(source_hash="b" * 64)
        fresh = Job.objects.get(pk=self.job.pk)
        second = _claim_processing_lease(fresh, "broker-A")
        assert second and second != first
        with pytest.raises(_ProcessingLeaseLost, match="new dispatch generation"):
            _commit_native_page_if_owned(
                fresh, second, 1, fresh.source_hash[:16],
                self._defaults("source-B-same-gen.pdf", fresh.source_hash[:16]), text_locator="source-B.txt",
            )
        page = PageResult.objects.get(job=self.job, page_num=1)
        assert page.temp_pdf_path == "source-A.pdf"
        Job.objects.filter(pk=self.job.pk).update(dispatch_generation=2)
        newer = Job.objects.get(pk=self.job.pk)
        third = _claim_processing_lease(newer, "broker-A")
        assert third and third != second
        assert _commit_native_page_if_owned(
            newer, third, 1, newer.source_hash[:16],
            self._defaults("source-B-gen2.pdf", newer.source_hash[:16]), text_locator="source-B-gen2.txt",
        )
        page.refresh_from_db()
        assert page.temp_pdf_path == "source-B-gen2.pdf"
        assert page.admitted_generation == 2
        assert page.ocr_provenance["source_sha256"] == "b" * 64

    def test_dead_owner_page_one_winner_is_reused_by_new_nonce_for_page_two(self):
        self.job.total_pages = 2
        self.job.save(update_fields=["total_pages"])
        first = _claim_processing_lease(self.job, "same-task-id")
        a_pdf, a_txt = self._attempt_paths(first)
        a_pdf.parent.mkdir(parents=True)
        a_pdf.write_bytes(b"page one winner")
        assert _commit_native_page_if_owned(
            self.job, first, 1, self.job.source_hash[:16],
            self._defaults(str(a_pdf)), text_locator=str(a_txt),
            custody_data={"page_num": 1},
        )
        Job.objects.filter(pk=self.job.pk).update(
            processing_lease_expires_at=timezone.now() - timezone.timedelta(seconds=1)
        )
        second = _claim_processing_lease(self.job, "same-task-id")
        assert second and second != first
        b_pdf, b_txt = self._attempt_paths(second)
        b_pdf.parent.mkdir(parents=True)
        b_pdf.write_bytes(b"duplicate candidate")
        b_txt.write_text("duplicate candidate", encoding="utf-8")
        self.job.refresh_from_db()
        assert _renew_or_discard_published_artifacts(
            lambda: _commit_native_page_if_owned(
                self.job, second, 1, self.job.source_hash[:16],
                self._defaults(str(b_pdf)), text_locator=str(b_txt),
                custody_data={"page_num": 1},
            ),
            self.job, NFSBackend(root=str(self.root)),
            ((str(b_pdf), None), (str(b_txt), None)),
        ) is False
        assert not b_pdf.exists() and not b_txt.exists()
        assert a_pdf.read_bytes() == b"page one winner"
        second_pdf, second_txt = self._attempt_paths(second, 2)
        assert _commit_native_page_if_owned(
            self.job, second, 2, self.job.source_hash[:16],
            self._defaults(str(second_pdf)), text_locator=str(second_txt),
            custody_data={"page_num": 2},
        )
        self.job.refresh_from_db()
        assert self.job.pages_completed == 2
        assert PageResult.objects.get(job=self.job, page_num=1).temp_pdf_path == str(a_pdf)
        assert PageResult.objects.get(job=self.job, page_num=2).temp_pdf_path == str(second_pdf)
        assert CustodyEvent.objects.filter(job=self.job, event_type="text_extracted_natively").count() == 2

    def test_raw_text_committed_before_handoff_can_reenter_and_assemble(self):
        first = _claim_processing_lease(self.job, "same-task-id")
        first_pdf, first_txt = self._attempt_paths(first)
        assert _commit_native_page_if_owned(
            self.job, first, 1, self.job.source_hash[:16],
            self._defaults(str(first_pdf)), text_locator=str(first_txt),
        )
        Job.objects.filter(pk=self.job.pk).update(
            processing_lease_expires_at=timezone.now() - timezone.timedelta(seconds=1)
        )
        second = _claim_processing_lease(self.job, "same-task-id")
        self.job.refresh_from_db()
        second_pdf, second_txt = self._attempt_paths(second)
        assert _commit_native_page_if_owned(
            self.job, second, 1, self.job.source_hash[:16],
            self._defaults(str(second_pdf)), text_locator=str(second_txt),
        ) is False
        handoff = _enqueue_assembly_dispatch(
            self.job.pk, expected_processing_token=second, expected_job=self.job,
        )
        assert handoff["status"] != "stale"
        self.job.refresh_from_db()
        assert self.job.status == Job.Status.PROCESSING
        assert TaskDispatchOutbox.objects.filter(
            job=self.job, stage="assemble_document",
            task_name="jobs.tasks.assemble_document",
        ).count() == 1
        assert PageResult.objects.get(job=self.job, page_num=1).temp_pdf_path == str(first_pdf)

    def test_invalid_utf8_native_source_fails_without_erasing_source(self):
        invalid = b"SOURCE-\xff-EVIDENCE"
        self.source.write_bytes(invalid)
        self.job.source_hash = hashlib.sha256(invalid).hexdigest()
        self.job.save(update_fields=["source_hash"])
        with (override_settings(NFS_ROOT=str(self.root)),
              patch("jobs.tasks._get_backend_for_job", return_value=NFSBackend(root=str(self.root)))):
            result = process_text_only.run(str(self.job.pk), dispatch_generation=1)
        assert result["status"] == "error"
        self.job.refresh_from_db()
        assert self.job.status == Job.Status.FAILED
        assert self.source.read_bytes() == invalid
        assert not PageResult.objects.filter(job=self.job).exists()
        failure = CustodyEvent.objects.get(job=self.job, event_type="processing_failed")
        assert len(failure.event_hash) == 64

    def test_native_failure_custody_and_grant_revoke_are_atomic(self):
        token = _claim_processing_lease(self.job, "delivery")
        grant = PresignedTaskCapability.objects.create(
            job=self.job, operation="process_text_only", dispatch_generation=1,
            payload="sealed", expires_at=timezone.now() + timezone.timedelta(hours=1),
        )
        with patch("jobs.tasks._record_custody_event", side_effect=RuntimeError("custody unavailable")):
            with pytest.raises(RuntimeError, match="custody unavailable"):
                _fail_processing_job_if_owned(
                    self.job.pk, token, "source rejected", expected_job=self.job,
                )
        self.job.refresh_from_db()
        assert self.job.status == Job.Status.PROCESSING
        assert self.job.processing_attempt_token == token
        assert PresignedTaskCapability.objects.filter(pk=grant.pk).exists()
        assert not CustodyEvent.objects.filter(job=self.job, event_type="processing_failed").exists()
        assert _fail_processing_job_if_owned(
            self.job.pk, token, "source rejected", expected_job=self.job,
        ) == 1
        self.job.refresh_from_db()
        assert self.job.status == Job.Status.FAILED
        assert not PresignedTaskCapability.objects.filter(pk=grant.pk).exists()
        assert CustodyEvent.objects.filter(job=self.job, event_type="processing_failed").count() == 1

    def test_changed_nfs_source_is_rejected_with_both_hashes_and_no_worker_marker_loss(self):
        admitted = self.job.source_hash
        self.source.write_text("NATIVE TEXT EVIDENCf", encoding="utf-8")
        observed = hashlib.sha256(self.source.read_bytes()).hexdigest()
        assert len(self.source.read_bytes()) == len(b"NATIVE TEXT EVIDENCE")
        hostname = socket.gethostname()
        Worker.objects.create(hostname=hostname, status=Worker.Status.BUSY, current_task_id="other")
        with (
            override_settings(NFS_ROOT=str(self.root)),
            patch("jobs.tasks._get_backend_for_job", return_value=NFSBackend(root=str(self.root))),
        ):
            result = process_text_only.run(str(self.job.pk), dispatch_generation=1)
        assert result["status"] == "error"
        self.job.refresh_from_db()
        assert self.job.status == Job.Status.FAILED
        assert not PageResult.objects.filter(job=self.job).exists()
        event = CustodyEvent.objects.get(job=self.job, event_type="processing_failed")
        assert event.data["expected_source_sha256"] == admitted
        assert event.data["observed_source_sha256"] == observed
        assert Worker.objects.get(hostname=hostname).current_task_id == "other"

    def test_source_failure_custody_fault_rolls_back_and_removes_private_snapshot(self):
        self.source.write_text("NATIVE TEXT EVIDENCf", encoding="utf-8")
        private_dir = self.root / "private-source"

        def make_private(**_kwargs):
            private_dir.mkdir(mode=0o700)
            return str(private_dir)

        with (
            override_settings(NFS_ROOT=str(self.root)),
            patch("jobs.tasks._get_backend_for_job", return_value=NFSBackend(root=str(self.root))),
            patch("jobs.tasks.tempfile.mkdtemp", side_effect=make_private),
            patch("jobs.tasks._record_custody_event", side_effect=RuntimeError("custody unavailable")),
        ):
            with pytest.raises(RuntimeError, match="custody unavailable"):
                process_text_only.run(str(self.job.pk), dispatch_generation=1)
        self.job.refresh_from_db()
        assert self.job.status == Job.Status.PROCESSING
        assert self.job.processing_attempt_token
        assert not private_dir.exists()
        assert self.source.read_text(encoding="utf-8") == "NATIVE TEXT EVIDENCf"

    def test_symlinked_nfs_source_fails_without_reading_target(self):
        target = self.root / "outside.txt"
        target.write_bytes(self.source.read_bytes())
        self.source.unlink()
        self.source.symlink_to(target)
        with (
            override_settings(NFS_ROOT=str(self.root)),
            patch("jobs.tasks._get_backend_for_job", return_value=NFSBackend(root=str(self.root))),
        ):
            result = process_text_only.run(str(self.job.pk), dispatch_generation=1)
        assert result["status"] == "error"
        assert target.read_bytes() == b"NATIVE TEXT EVIDENCE"
        assert not PageResult.objects.filter(job=self.job).exists()
        assert CustodyEvent.objects.filter(job=self.job, event_type="processing_failed").count() == 1

    def test_changed_s3_download_is_rejected_before_page_publication(self):
        changed = b"NATIVE TEXT EVIDENCf"

        class ChangedSourceBackend:
            backend_name = "s3"

            def download_file(self, _key, path, *, max_bytes=None):
                assert max_bytes is not None and max_bytes >= len(changed)
                Path(path).write_bytes(changed)
                return path

        with (
            patch("jobs.tasks._get_backend_for_job", return_value=ChangedSourceBackend()),
            patch("jobs.tasks.is_presigned_mode", return_value=False),
        ):
            result = process_text_only.run(str(self.job.pk), dispatch_generation=1)
        assert result["status"] == "error"
        assert not PageResult.objects.filter(job=self.job).exists()
        event = CustodyEvent.objects.get(job=self.job, event_type="processing_failed")
        assert event.data["expected_source_sha256"] == self.job.source_hash
        assert event.data["observed_source_sha256"] == hashlib.sha256(changed).hexdigest()

    def test_native_pdf_page_count_must_match_admission(self):
        pdf_path = self.source.with_suffix(".pdf")
        with fitz.open() as document:
            for text in ("FIRST PAGE", "SECOND PAGE"):
                page = document.new_page()
                page.insert_text((72, 72), text)
            document.save(pdf_path)
        self.job.source_file = pdf_path.name
        self.job.source_type = "pdf"
        self.job.source_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        self.job.save(update_fields=["source_file", "source_type", "source_hash"])
        with (
            override_settings(NFS_ROOT=str(self.root)),
            patch("jobs.tasks._get_backend_for_job", return_value=NFSBackend(root=str(self.root))),
        ):
            result = process_text_only.run(str(self.job.pk), dispatch_generation=1)
        assert result["status"] == "error"
        assert "page count changed" in result["message"]
        assert not PageResult.objects.filter(job=self.job).exists()
        assert CustodyEvent.objects.filter(job=self.job, event_type="processing_failed").count() == 1

    def test_native_pdf_commits_exact_two_page_text_hashes(self):
        pdf_path = self.source.with_suffix(".pdf")
        with fitz.open() as document:
            for text in ("FIRST PAGE", "SECOND PAGE"):
                page = document.new_page()
                page.insert_text((72, 72), text)
            document.save(pdf_path)
        self.job.source_file = pdf_path.name
        self.job.source_type = "pdf"
        self.job.source_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        self.job.total_pages = 2
        self.job.save(update_fields=["source_file", "source_type", "source_hash", "total_pages"])
        with (
            override_settings(NFS_ROOT=str(self.root)),
            patch("jobs.tasks._get_backend_for_job", return_value=NFSBackend(root=str(self.root))),
        ):
            result = process_text_only.run(str(self.job.pk), dispatch_generation=1)
        assert result["status"] == "processed"
        for page in PageResult.objects.filter(job=self.job).order_by("page_num"):
            text_path = Path(page.ocr_provenance["temp_text_path"])
            assert page.ocr_method == "NativePDF"
            assert page.ocr_language == "und"
            assert page.ocr_provenance["text_sha256"] == hashlib.sha256(text_path.read_bytes()).hexdigest()
        assert CustodyEvent.objects.filter(job=self.job, event_type="text_extracted_natively").count() == 2

    def test_raw_text_commit_records_custody_and_unknown_language(self):
        with (
            override_settings(NFS_ROOT=str(self.root)),
            patch("jobs.tasks._get_backend_for_job", return_value=NFSBackend(root=str(self.root))),
        ):
            result = process_text_only.run(str(self.job.pk), dispatch_generation=1)
        assert result["status"] == "processed"
        page = PageResult.objects.get(job=self.job, page_num=1)
        assert page.ocr_method == "NativeText"
        assert page.ocr_language == "und"
        assert page.ocr_provenance["source_sha256"] == self.job.source_hash
        assert page.ocr_provenance["text_sha256"] == hashlib.sha256(b"NATIVE TEXT EVIDENCE").hexdigest()
        custody = CustodyEvent.objects.get(job=self.job, event_type="text_extracted_natively")
        assert custody.data["method"] == "NativeText"
        assert custody.data["text_length"] == len("NATIVE TEXT EVIDENCE")

    def test_backend_selection_failure_uses_atomic_owned_failure(self):
        with patch("jobs.tasks._get_backend_for_job", side_effect=RuntimeError("backend unavailable")):
            result = process_text_only.run(str(self.job.pk), dispatch_generation=1)
        assert result["status"] == "error"
        self.job.refresh_from_db()
        assert self.job.status == Job.Status.FAILED
        assert self.job.processing_attempt_token == ""
        assert not PageResult.objects.filter(job=self.job).exists()
        event = CustodyEvent.objects.get(job=self.job, event_type="processing_failed")
        assert event.data["error"] == "native processing setup failed (RuntimeError)"

    def test_nfs_setup_mkdir_failure_cleans_private_snapshot_and_fails_once(self):
        created = []
        real_mkdtemp = tempfile.mkdtemp

        def private_dir(*args, **kwargs):
            path = real_mkdtemp(*args, **kwargs)
            created.append(path)
            return path

        with (
            override_settings(NFS_ROOT=str(self.root)),
            patch("jobs.tasks._get_backend_for_job", return_value=NFSBackend(root=str(self.root))),
            patch("jobs.tasks.tempfile.mkdtemp", side_effect=private_dir),
            patch("jobs.tasks._mkdir_tree_no_follow", side_effect=OSError("mkdir failed")),
        ):
            result = process_text_only.run(str(self.job.pk), dispatch_generation=1)
        assert result["status"] == "error"
        assert created and all(not os.path.exists(path) for path in created)
        assert self.source.read_bytes() == b"NATIVE TEXT EVIDENCE"
        assert not PageResult.objects.filter(job=self.job).exists()
        assert CustodyEvent.objects.filter(job=self.job, event_type="processing_failed").count() == 1

    def test_input_capability_failure_uses_atomic_owned_failure(self):
        class S3Backend:
            backend_name = "s3"

        with (
            patch("jobs.tasks._get_backend_for_job", return_value=S3Backend()),
            patch("jobs.tasks.is_presigned_mode", return_value=True),
        ):
            result = process_text_only.run(str(self.job.pk), dispatch_generation=1)
        assert result["status"] == "error"
        assert not PageResult.objects.filter(job=self.job).exists()
        assert CustodyEvent.objects.filter(job=self.job, event_type="processing_failed").count() == 1
