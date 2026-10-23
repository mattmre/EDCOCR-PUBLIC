"""Coordinator regressions for the sixth expanded ARCH-AEP overlay."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from django.test import TestCase, override_settings


class TestRoundSixCoordinatorBoundaries(TestCase):
    def test_document_processing_lease_fences_redelivery(self):
        import uuid

        from django.utils import timezone

        from jobs.models import Job
        from jobs.tasks import (
            _claim_processing_lease,
            _renew_processing_lease,
        )

        job = Job.objects.create(
            source_file="lease.pdf",
            status=Job.Status.PROCESSING,
        )
        first = _claim_processing_lease(job, "task-a")
        assert uuid.UUID(first).hex == first
        assert _claim_processing_lease(job, "task-a") is None
        assert _claim_processing_lease(job, "task-b") is None
        job.refresh_from_db()
        assert job.celery_task_id == "task-a"
        _renew_processing_lease(job.job_id, first)
        Job.objects.filter(pk=job.pk).update(
            processing_lease_expires_at=timezone.now() - timezone.timedelta(seconds=1)
        )
        replacement = _claim_processing_lease(job, "task-a")
        assert replacement and replacement != first

    def test_late_failure_cannot_overwrite_recovered_processing_owner(self):
        import hashlib

        from django.utils import timezone

        from jobs.models import Job
        from jobs.presigned import text_only_attempt_identity
        from jobs.tasks import _fail_processing_job_if_owned

        job = Job.objects.create(
            source_file="late-failure.pdf",
            source_hash=hashlib.sha256(b"admitted source").hexdigest(),
            status=Job.Status.PROCESSING,
            celery_task_id="task-new",
            processing_attempt_token="b" * 32,
            processing_lease_expires_at=timezone.now() + timezone.timedelta(minutes=10),
        )
        job.processing_attempt_identity = text_only_attempt_identity(job)
        job.save(update_fields=["processing_attempt_identity"])
        assert _fail_processing_job_if_owned(job.job_id, "task-old", "late") == 0
        job.refresh_from_db()
        assert job.status == Job.Status.PROCESSING
        assert job.celery_task_id == "task-new"
        assert _fail_processing_job_if_owned(job.job_id, "b" * 32, "owned") == 1
        job.refresh_from_db()
        assert job.status == Job.Status.FAILED
        assert job.error_message == "owned"

    def test_fanout_page_leases_allow_distinct_pages_but_fence_duplicate(self):
        import uuid

        from django.utils import timezone

        from jobs.models import Job, PageResult
        from jobs.tasks import (
            _claim_page_processing_lease,
            _ProcessingLeaseLost,
            _renew_page_processing_lease,
        )

        job = Job.objects.create(
            source_file="fanout.pdf",
            status=Job.Status.PROCESSING,
        )
        page_one = _claim_page_processing_lease(
            job, 1, "page-task-a", "document-1", "worker-a"
        )
        page_two = _claim_page_processing_lease(
            job, 2, "page-task-b", "document-1", "worker-b"
        )
        assert uuid.UUID(page_one).hex == page_one
        assert uuid.UUID(page_two).hex == page_two
        assert page_one != page_two
        assert PageResult.objects.get(job=job, page_num=1).celery_task_id == page_one
        assert PageResult.objects.get(job=job, page_num=2).celery_task_id == page_two
        assert (
            _claim_page_processing_lease(
                job, 1, "page-task-c", "document-1", "worker-c"
            )
            is None
        )
        PageResult.objects.filter(job=job, page_num=1).update(
            lease_expires_at=timezone.now() - timezone.timedelta(seconds=1)
        )
        replacement = _claim_page_processing_lease(
            job, 1, "page-task-c", "document-1", "worker-c"
        )
        assert replacement and replacement != page_one
        with pytest.raises(_ProcessingLeaseLost):
            _renew_page_processing_lease(job.job_id, 1, page_one)

    def test_control_plane_archive_record_does_not_invent_job_identity(self):
        from jobs.management.commands.rotate_audit_logs import _serialize_event

        event = SimpleNamespace(
            id=1,
            job_id=None,
            document_id="system-event",
            event_type="audit_logs_rotated",
            timestamp=SimpleNamespace(isoformat=lambda: "2026-08-04T00:00:00+00:00"),
            worker_hostname="",
            data={},
            prev_hash="",
            event_hash="a" * 64,
            chain_finalized=True,
        )
        assert _serialize_event(event)["job_id"] == ""

    def test_purge_temp_custody_event_is_unscoped(self):
        from jobs.management.commands.purge_temp_files import (
            _record_purge_custody_event,
        )
        from jobs.models import CustodyEvent, Job

        Job.objects.create(source_file="/surviving.pdf")
        _record_purge_custody_event("/shared/jobs/orphan", 12)
        event = CustodyEvent.objects.get(event_type="temp_files_purged")
        assert event.job_id is None

    @override_settings(
        DEBUG=False, DEPLOYMENT_ENV="production", NFS_ROOT="/srv/ocr-nfs"
    )
    def test_production_worker_rejects_unbound_persisted_nfs_path(self):
        from jobs.tasks import _nfs_job_path_for_job

        job = SimpleNamespace(job_id="job-bound", nfs_job_path="/tmp/foreign")
        with pytest.raises(RuntimeError, match="not canonical"):
            _nfs_job_path_for_job(job)

    @override_settings(DEBUG=True, NFS_ROOT="/srv/ocr-nfs")
    def test_debug_fixture_compatibility_is_explicit(self):
        from jobs.tasks import _nfs_job_path_for_job

        job = SimpleNamespace(job_id="job-bound", nfs_job_path="/tmp/fixture")
        assert _nfs_job_path_for_job(job) == "/tmp/fixture"


class _SizedResponse:
    status = 200

    def __init__(self, body: bytes):
        self._body = io.BytesIO(body)

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_federation_probe_rejects_oversized_peer_body():
    from federation.reconciler import _probe_management_api

    healthy, payload = _probe_management_api(
        "https://rabbit.example:15671",
        opener=lambda *_args, **_kwargs: _SizedResponse(b"x" * (256 * 1024 + 1)),
    )
    assert healthy is False
    assert payload == {}


def test_federation_registry_file_read_is_bounded(tmp_path):
    from federation.reconciler import load_registry_from_path

    path = tmp_path / "clusters.json"
    path.write_text("{" + (" " * (256 * 1024 + 1)) + "}", encoding="utf-8")
    with pytest.raises(ValueError, match="safety limit"):
        load_registry_from_path(str(path))


def test_atomic_audit_archive_has_no_partial_temp_file(tmp_path):
    from jobs.management.commands.rotate_audit_logs import _write_archive

    archive_path = tmp_path / "archive.jsonl"
    written = _write_archive([], str(archive_path))
    assert written == 0
    assert archive_path.read_bytes() == b""
    assert list(tmp_path.glob("*.tmp")) == []


def test_nfs_backend_binds_each_operation_to_open_directory_descriptors(tmp_path):
    from jobs.storage import NFSBackend

    backend = NFSBackend(root=str(tmp_path))
    source = tmp_path / "source.pdf"
    source.write_bytes(b"forensic-bytes")
    locator = backend.upload_file(str(source), "jobs/job-1/source/source.pdf")
    assert locator.endswith("jobs/job-1/source/source.pdf")

    downloaded = tmp_path / "download.pdf"
    backend.download_file("jobs/job-1/source/source.pdf", str(downloaded))
    assert downloaded.read_bytes() == b"forensic-bytes"

    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"must-survive")
    link = tmp_path / "jobs" / "job-1" / "source" / "link.pdf"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    with pytest.raises(RuntimeError, match="symlink"):
        backend.download_file("jobs/job-1/source/link.pdf", str(tmp_path / "x.pdf"))
    assert outside.read_bytes() == b"must-survive"


def test_job_storage_binding_fingerprint_refuses_changed_root(tmp_path):
    from jobs.tasks import (
        _get_backend_for_job,
        _reset_storage_backend,
        _storage_binding_fingerprint,
    )

    with override_settings(STORAGE_BACKEND="nfs", NFS_ROOT=str(tmp_path)):
        _reset_storage_backend()
        fingerprint = _storage_binding_fingerprint("nfs")
        job = SimpleNamespace(
            job_id="job-bound",
            storage_backend_used="nfs",
            settings_json={"storage_binding_fingerprint": fingerprint},
        )
        assert _get_backend_for_job(job).backend_name == "nfs"

    with override_settings(STORAGE_BACKEND="nfs", NFS_ROOT=str(tmp_path / "changed")):
        with pytest.raises(RuntimeError, match="Storage binding changed"):
            _get_backend_for_job(job)


def test_presigned_artifact_binding_rejects_same_job_wrong_key(monkeypatch):
    from jobs.presigned_io import _validate_presigned_url

    monkeypatch.setenv("S3_PRESIGNED_ALLOWED_HOSTS", "s3.example.com,bucket.s3.example.com")
    monkeypatch.setenv("S3_BUCKET", "bucket")
    monkeypatch.setenv("S3_ENDPOINT", "https://s3.example.com")
    url = "https://s3.example.com/bucket/jobs/job-1/output/a.pdf"
    _validate_presigned_url(
        url,
        job_id="job-1",
        expected_key="jobs/job-1/output/a.pdf",
    )
    _validate_presigned_url(
        "https://bucket.s3.example.com/jobs/job-1/output/a.pdf",
        job_id="job-1",
        expected_key="jobs/job-1/output/a.pdf",
    )
    with pytest.raises(ValueError, match="expected artifact"):
        _validate_presigned_url(
            url,
            job_id="job-1",
            expected_key="jobs/job-1/output/other.pdf",
        )
    with pytest.raises(ValueError, match="expected artifact"):
        _validate_presigned_url(
            "https://s3.example.com/other-bucket/jobs/job-1/output/a.pdf",
            job_id="job-1",
            expected_key="jobs/job-1/output/a.pdf",
        )
    with pytest.raises(ValueError, match="expected artifact"):
        _validate_presigned_url(
            "https://s3.example.com/jobs/job-1/output/a.pdf",
            job_id="job-1",
            expected_key="jobs/job-1/output/a.pdf",
        )
