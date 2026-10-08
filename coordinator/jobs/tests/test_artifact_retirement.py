"""Owned delayed-publication fixtures retain cleanup after Job deletion."""

from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.db import transaction
from django.test import TestCase, override_settings
from django.utils import timezone

from jobs.artifact_retirement import (
    retain_cleanup_authority,
    sweep_retired_job_artifacts,
)
from jobs.models import Job, JobArtifactRetirement, PageOCRByteBudget, PageOCRStage
from jobs.tasks import _storage_binding_fingerprint, cleanup_completed_jobs


class OwnedObjectStore:
    backend_name = "s3"

    def __init__(self):
        self.objects = {}
        self.fail_delete = False

    def list_objects(self, prefix):
        return [key for key in self.objects if key.startswith(prefix)]

    def delete_many(self, keys):
        if self.fail_delete:
            raise OSError("owned fixture unavailable")
        for key in keys:
            del self.objects[key]
        return len(keys)


@override_settings(S3_ENDPOINT="https://owned.invalid", S3_BUCKET="owned", S3_REGION="us-east-1")
class TestArtifactRetirement(TestCase):
    def setUp(self):
        self.backend = OwnedObjectStore()
        self.backend_patch = patch("jobs.tasks._get_storage_backend", return_value=self.backend)
        self.backend_patch.start()
        self.addCleanup(self.backend_patch.stop)
        self.hold_patch = patch("jobs.tasks.is_litigation_hold_active", return_value=False)
        self.hold_patch.start()
        self.addCleanup(self.hold_patch.stop)
        self.job = Job.objects.create(
            tenant_id="owned-tenant", source_file="owned.pdf",
            status=Job.Status.COMPLETED, storage_backend_used="s3",
            completed_at=timezone.now() - timezone.timedelta(days=45),
            settings_json={"storage_binding_fingerprint": _storage_binding_fingerprint("s3")},
        )
        self.job_id = self.job.pk
        self.prefix = f"jobs/{self.job_id}/"

    def due(self):
        JobArtifactRetirement.objects.filter(pk=self.job_id).update(next_sweep_at=timezone.now())

    def test_purge_then_delayed_upload_and_repeated_empty_sweep(self):
        self.backend.objects[self.prefix + "source.pdf"] = b"source"
        neighbor = f"jobs/{self.job_id}-neighbor/output.pdf"
        self.backend.objects[neighbor] = b"unrelated"
        output = StringIO()
        call_command("purge_tenant", tenant_id="owned-tenant", confirm=True, stdout=output)
        assert not Job.objects.filter(pk=self.job_id).exists()
        assert "periodic cleanup for late uploads" in output.getvalue()
        assert JobArtifactRetirement.objects.filter(pk=self.job_id).exists()
        assert sweep_retired_job_artifacts() == {"swept": 1, "failed": 0}
        # An empty sweep cannot retire cleanup authority: PUT finishes later.
        self.backend.objects[self.prefix + "output/final.pdf"] = b"late final PDF"
        self.due()
        assert sweep_retired_job_artifacts() == {"swept": 1, "failed": 0}
        assert self.backend.objects == {neighbor: b"unrelated"}
        assert JobArtifactRetirement.objects.get(pk=self.job_id).sweep_count == 2

    def test_retention_keeps_authority_after_delete_and_storage_failure(self):
        self.backend.objects[self.prefix + "source.pdf"] = b"source"
        neighbor = f"jobs/{self.job_id}-neighbor/output.pdf"
        self.backend.objects[neighbor] = b"unrelated"
        assert cleanup_completed_jobs()["deleted"] == 1
        assert self.backend.objects[neighbor] == b"unrelated"
        self.backend.objects[self.prefix + "output/late.txt"] = b"late"
        self.backend.fail_delete = True
        assert sweep_retired_job_artifacts() == {"swept": 0, "failed": 1}
        assert JobArtifactRetirement.objects.get(pk=self.job_id).last_error == "OSError"
        self.backend.fail_delete = False
        self.due()
        assert sweep_retired_job_artifacts() == {"swept": 1, "failed": 0}
        assert self.backend.objects == {neighbor: b"unrelated"}

    def test_database_rollback_cannot_separate_deletion_and_retirement(self):
        try:
            with transaction.atomic():
                retain_cleanup_authority(self.job, self.backend)
                self.job.delete()
                raise RuntimeError("owned rollback")
        except RuntimeError:
            pass
        assert Job.objects.filter(pk=self.job_id).exists()
        assert not JobArtifactRetirement.objects.filter(pk=self.job_id).exists()

    def test_retention_waits_for_staged_capacity_cleanup(self):
        stage = PageOCRStage.objects.create(
            job=self.job, tenant_id=self.job.tenant_id, page_num=1, generation=1,
            source_sha256="a" * 64, status="cleanup", reserved_bytes=4096,
        )
        PageOCRByteBudget.objects.update_or_create(pk=1, defaults={"reserved_bytes": 4096})
        self.backend.objects[self.prefix + "source.pdf"] = b"source"
        result = cleanup_completed_jobs()
        assert result["deleted"] == 0
        assert "not drained" in result["cleanup_failures"][0]
        assert Job.objects.filter(pk=self.job_id).exists()
        assert PageOCRStage.objects.get(pk=stage.pk).reserved_bytes == 4096
        assert PageOCRByteBudget.objects.get(pk=1).reserved_bytes == 4096
        assert self.backend.objects[self.prefix + "source.pdf"] == b"source"
        assert not JobArtifactRetirement.objects.filter(pk=self.job_id).exists()
        stage.status, stage.reserved_bytes = "done", 0
        stage.save()
        PageOCRByteBudget.objects.filter(pk=1).update(reserved_bytes=0)
        assert cleanup_completed_jobs()["deleted"] == 1

    def test_live_job_and_changed_bucket_are_never_swept(self):
        retain_cleanup_authority(self.job, self.backend)
        self.backend.objects[self.prefix + "source.pdf"] = b"source"
        assert sweep_retired_job_artifacts()["failed"] == 1
        self.job.delete()
        self.due()
        with override_settings(S3_BUCKET="different"):
            assert sweep_retired_job_artifacts()["failed"] == 1
        assert self.backend.objects[self.prefix + "source.pdf"] == b"source"

    def test_litigation_hold_preserves_late_bytes(self):
        retain_cleanup_authority(self.job, self.backend)
        self.job.delete()
        self.backend.objects[self.prefix + "output.txt"] = b"held"
        with patch("jobs.tasks.is_litigation_hold_active", return_value=True):
            assert sweep_retired_job_artifacts()["litigation_hold"] is True
        assert self.backend.objects[self.prefix + "output.txt"] == b"held"
