"""Owned delayed-publication fixtures retain cleanup after Job deletion."""

import uuid
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.db import IntegrityError, OperationalError, transaction
from django.test import TestCase, override_settings
from django.utils import timezone

from jobs.artifact_retirement import (
    retain_cleanup_authority,
    sweep_retired_job_artifacts,
)
from jobs.models import (
    Job,
    JobArtifactRetirement,
    PageOCRByteBudget,
    PageOCRRasterRetirement,
    PageOCRStage,
)
from jobs.tasks import (
    _stage_cleanup_artifact,
    _storage_binding_fingerprint,
    _sweep_retired_stage_rasters,
    cleanup_completed_jobs,
    recover_page_stages,
)


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

    def delete(self, key):
        self.objects.pop(key, None)


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

    def test_bookkeeping_failure_keeps_authority_and_sweeps_other_prefix(self):
        retain_cleanup_authority(self.job, self.backend)
        self.job.delete()
        other_id = uuid.uuid4()
        other_key = f"jobs/{other_id}/output/late-final.pdf"
        self.backend.objects[other_key] = b"late"
        JobArtifactRetirement.objects.create(
            job_uuid=other_id,
            tenant_id=self.job.tenant_id,
            backend_binding=_storage_binding_fingerprint("s3"),
            next_sweep_at=timezone.now(),
        )
        JobArtifactRetirement.objects.filter(pk=self.job_id).update(
            next_sweep_at=timezone.now() - timezone.timedelta(minutes=5),
        )
        real_filter = JobArtifactRetirement.objects.filter

        def filter_with_poison(*args, **kwargs):
            qs = real_filter(*args, **kwargs)
            if kwargs.get("pk") == self.job_id:
                real_update = qs.update

                def update(**fields):
                    if "sweep_count" in fields:
                        raise IntegrityError("sweep_count_bookkeeping_failed")
                    return real_update(**fields)

                qs.update = update
            return qs

        with patch(
            "jobs.artifact_retirement.JobArtifactRetirement.objects.filter",
            side_effect=filter_with_poison,
        ):
            result = sweep_retired_job_artifacts()
        assert result == {"swept": 1, "failed": 1}
        assert other_key not in self.backend.objects
        poisoned = JobArtifactRetirement.objects.get(pk=self.job_id)
        assert poisoned.last_error == "IntegrityError"
        assert poisoned.sweep_count == 0

    def test_sweep_relation_error_retains_authority_and_late_bytes(self):
        retain_cleanup_authority(self.job, self.backend)
        self.job.delete()
        late_key = self.prefix + "output/late-final.pdf"
        self.backend.objects[late_key] = b"late"
        with patch(
            "jobs.artifact_retirement.JobArtifactRetirement.objects.filter",
            side_effect=OperationalError("owned relation unavailable"),
        ):
            result = sweep_retired_job_artifacts()
        assert result["error"] == "OperationalError"
        assert result["swept"] == 0
        assert JobArtifactRetirement.objects.filter(pk=self.job_id).exists()
        assert self.backend.objects[late_key] == b"late"
        self.due()
        assert sweep_retired_job_artifacts() == {"swept": 1, "failed": 0}
        assert late_key not in self.backend.objects

    def test_sweep_exception_cannot_skip_stage_recovery(self):
        with patch(
            "jobs.artifact_retirement.sweep_retired_job_artifacts",
            side_effect=RuntimeError("owned sweep failure"),
        ):
            with patch("jobs.tasks._recover_staged_pages", return_value=1) as recovered:
                result = recover_page_stages()
        assert recovered.called
        assert result["recovered"] == 1
        assert result["retired_jobs"]["error"] == "RuntimeError"

    def test_litigation_hold_preserves_raster_retirement_and_stage_bytes(self):
        raster = self.prefix + "raster/old.png"
        self.backend.objects[raster] = b"held-raster"
        PageOCRRasterRetirement.objects.create(
            job_uuid=self.job_id,
            tenant_id=self.job.tenant_id,
            generation=1,
            page_num=1,
            locator=raster,
            backend_binding=_storage_binding_fingerprint("s3"),
            reserved_bytes=0,
            safe_to_release_after=timezone.now() - timezone.timedelta(minutes=1),
            next_sweep_at=timezone.now() - timezone.timedelta(minutes=1),
        )
        stage = PageOCRStage.objects.create(
            job=self.job, tenant_id=self.job.tenant_id, page_num=1, generation=1,
            source_sha256="a" * 64, status="cleanup", raster_locator=raster,
        )
        with patch("jobs.tasks.is_litigation_hold_active", return_value=True):
            assert _sweep_retired_stage_rasters() == 0
            _stage_cleanup_artifact(stage.pk, self.backend)
            assert recover_page_stages()["retired_jobs"]["litigation_hold"] is True
        assert self.backend.objects[raster] == b"held-raster"
        assert PageOCRStage.objects.get(pk=stage.pk).status == "cleanup"
        assert PageOCRRasterRetirement.objects.filter(locator=raster).exists()
