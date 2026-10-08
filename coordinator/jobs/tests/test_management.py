"""Tests for management commands (Phase M5 production hardening).

Tests cleanup_old_jobs, purge_temp_files, and fleet_status commands
with filesystem operations, dry-run modes, and edge cases.

Run with: cd coordinator && python -m pytest jobs/tests/test_management.py -v
"""

import json
import os
import shutil
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from cryptography.fernet import Fernet
from django.apps import apps
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.test import TestCase, override_settings
from django.utils import timezone

from jobs.encrypted_field import EncryptedTextField
from jobs.extraction_models import ExtractedFormValue
from jobs.management.commands.rotate_audit_logs import _verify_archive
from jobs.models import (
    CustodyEvent,
    Job,
    PageOCRByteBudget,
    PageOCRStage,
    PresignedTaskCapability,
    TaskDispatchOutbox,
    Worker,
)
from jobs.presigned import create_presigned_capability


class TestCleanupOldJobs(TestCase):
    """Tests for the cleanup_old_jobs management command."""

    def test_deletes_old_completed_jobs(self):
        """Old completed/failed/cancelled jobs are deleted by default (30 days)."""
        j1 = Job.objects.create(source_file="/old1.pdf", status=Job.Status.COMPLETED)
        j2 = Job.objects.create(source_file="/old2.pdf", status=Job.Status.FAILED)
        j3 = Job.objects.create(source_file="/old3.pdf", status=Job.Status.CANCELLED)
        old_date = timezone.now() - timezone.timedelta(days=45)
        Job.objects.filter(job_id__in=[j1.job_id, j2.job_id, j3.job_id]).update(
            created_at=old_date, completed_at=old_date
        )

        out = StringIO()
        call_command("cleanup_old_jobs", "--confirm", stdout=out)

        assert Job.objects.filter(job_id=j1.job_id).count() == 0
        assert Job.objects.filter(job_id=j2.job_id).count() == 0
        assert Job.objects.filter(job_id=j3.job_id).count() == 0
        # deleted_count from Django delete() includes cascade-deleted related
        # objects (e.g. CustodyEvent created by _record_deletion_custody_event),
        # so we check for the message pattern rather than an exact count.
        output = out.getvalue()
        assert "Deleted" in output
        assert "jobs older than" in output

    def test_subject_match_in_form_value_is_not_stranded_without_pii_entity(self):
        job = Job.objects.create(
            source_file="/derivative-only.pdf",
            status=Job.Status.COMPLETED,
            completed_at=timezone.now(),
            tenant_id="tenant-a",
        )
        form = ExtractedFormValue.objects.create(
            job=job,
            page_number=1,
            field_key="patient",
            field_value="Jane Doe",
            source_module="form",
        )
        other_subject = ExtractedFormValue.objects.create(
            job=job,
            page_number=1,
            field_key="physician",
            field_value="Bob Smith",
            source_module="form",
        )
        with (
            patch(
                "jobs.management.commands.purge_pii._delete_pii_sidecars_for_job_checked",
                return_value=(0, []),
            ),
            patch(
                "jobs.management.commands.purge_pii._clear_redis_cache_for_jobs",
                return_value=(0, []),
            ),
        ):
            call_command("purge_pii", "--subject", "Jane Doe", "--confirm")

        assert not ExtractedFormValue.objects.filter(pk=form.pk).exists()
        # Subject matching selects the job; indivisible job-scoped sidecars
        # require the database erasure plan to use the same whole-job scope.
        assert not ExtractedFormValue.objects.filter(pk=other_subject.pk).exists()
        job.refresh_from_db()
        assert job.status == Job.Status.COMPLETED
        assert job.last_failure_reason == ""
        assert "_pii_erasure_fence" not in job.settings_json
        assert list(
            CustodyEvent.objects.filter(document_id=f"pii-purge:{job.job_id}")
            .order_by("id")
            .values_list("event_type", flat=True)
        ) == ["pii_deletion_started", "pii_purged"]

    def test_active_job_is_rejected_before_any_storage_mutation(self):
        job = Job.objects.create(
            source_file="/active.pdf",
            status=Job.Status.PROCESSING,
            tenant_id="tenant-a",
        )
        with patch(
            "jobs.management.commands.purge_pii._delete_pii_sidecars_for_job_checked"
        ) as delete_sidecars:
            with self.assertRaisesRegex(CommandError, "must be terminal"):
                call_command("purge_pii", "--job-id", str(job.job_id), "--confirm")
        delete_sidecars.assert_not_called()

    def test_manual_retention_erases_derivative_only_pii_with_shared_saga(self):
        job = Job.objects.create(
            source_file="/retention-derivative-only.pdf",
            status=Job.Status.COMPLETED,
            completed_at=timezone.now(),
            tenant_id="tenant-retention",
        )
        form = ExtractedFormValue.objects.create(
            job=job,
            page_number=1,
            field_key="patient",
            field_value="Jane Doe",
            source_module="form",
        )
        ExtractedFormValue.objects.filter(pk=form.pk).update(
            created_at=timezone.now() - timezone.timedelta(days=100)
        )
        with (
            patch(
                "jobs.management.commands.purge_pii."
                "_delete_pii_sidecars_for_job_checked",
                return_value=(2, []),
            ) as delete_sidecars,
            patch(
                "jobs.management.commands.purge_pii._clear_redis_cache_for_jobs",
                return_value=(1, []),
            ),
        ):
            call_command(
                "cleanup_old_jobs",
                "--days",
                "30",
                "--include-pii",
                "--confirm",
            )

        assert not ExtractedFormValue.objects.filter(pk=form.pk).exists()
        delete_sidecars.assert_called_once()
        assert list(
            CustodyEvent.objects.filter(document_id=f"pii-purge:{job.job_id}")
            .order_by("id")
            .values_list("event_type", flat=True)
        ) == ["pii_deletion_started", "pii_purged"]

    def test_s3_sidecar_cleanup_covers_all_pii_export_planes(self):
        from jobs.management.commands.purge_pii import (
            _delete_pii_sidecars_for_job_checked,
        )

        job = Job.objects.create(
            source_file="/s3.pdf",
            status=Job.Status.COMPLETED,
            storage_backend_used="s3",
        )
        backend = MagicMock(backend_name="s3")
        deleted_prefixes = set()

        def list_objects(prefix):
            return [] if prefix in deleted_prefixes else [prefix + "record.json"]

        def delete_many(keys):
            deleted_prefixes.add(keys[0].removesuffix("record.json"))
            return len(keys)

        backend.list_objects.side_effect = list_objects
        backend.delete_many.side_effect = delete_many
        with patch("jobs.tasks._get_backend_for_job", return_value=backend):
            removed, failures = _delete_pii_sidecars_for_job_checked(job)

        assert failures == []
        assert removed == 6
        prefixes = [call.args[0] for call in backend.list_objects.call_args_list]
        for sidecar in (
            "NER",
            "ENTITIES",
            "EXTRACTION",
            "FORM",
            "HANDWRITING",
            "SIGNATURE",
        ):
            assert any(f"/EXPORT/{sidecar}/" in prefix for prefix in prefixes)

    def test_preserves_recent_jobs(self):
        """Jobs created recently (within retention period) are not deleted."""
        j_recent = Job.objects.create(
            source_file="/recent.pdf", status=Job.Status.COMPLETED
        )
        # created_at is auto_now_add, so it's set to now by default

        out = StringIO()
        call_command("cleanup_old_jobs", "--confirm", stdout=out)

        assert Job.objects.filter(job_id=j_recent.job_id).exists()
        assert "No jobs older than" in out.getvalue()

    def test_preserves_active_jobs(self):
        """Old jobs that are still processing/ingesting/assembling are not deleted."""
        j_processing = Job.objects.create(
            source_file="/active.pdf", status=Job.Status.PROCESSING
        )
        j_ingesting = Job.objects.create(
            source_file="/ingest.pdf", status=Job.Status.INGESTING
        )
        j_assembling = Job.objects.create(
            source_file="/assemble.pdf", status=Job.Status.ASSEMBLING
        )
        old_date = timezone.now() - timezone.timedelta(days=60)
        Job.objects.filter(
            job_id__in=[j_processing.job_id, j_ingesting.job_id, j_assembling.job_id]
        ).update(created_at=old_date)

        out = StringIO()
        call_command("cleanup_old_jobs", "--confirm", stdout=out)

        assert Job.objects.filter(job_id=j_processing.job_id).exists()
        assert Job.objects.filter(job_id=j_ingesting.job_id).exists()
        assert Job.objects.filter(job_id=j_assembling.job_id).exists()

    def test_dry_run_no_deletion(self):
        """--dry-run previews deletions without actually deleting jobs."""
        job = Job.objects.create(source_file="/dryrun.pdf", status=Job.Status.COMPLETED)
        old_date = timezone.now() - timezone.timedelta(days=45)
        Job.objects.filter(job_id=job.job_id).update(
            created_at=old_date, completed_at=old_date
        )

        out = StringIO()
        call_command("cleanup_old_jobs", "--dry-run", stdout=out)

        assert Job.objects.filter(job_id=job.job_id).exists()
        assert "DRY RUN" in out.getvalue()
        assert "Would delete 1 jobs" in out.getvalue()

    def test_custom_days_retention(self):
        """--days N overrides the default 30-day retention period."""
        j_7days = Job.objects.create(
            source_file="/seven.pdf", status=Job.Status.COMPLETED
        )
        j_3days = Job.objects.create(
            source_file="/three.pdf", status=Job.Status.COMPLETED
        )
        # Make j_7days 10 days old, j_3days 5 days old
        Job.objects.filter(job_id=j_7days.job_id).update(
            created_at=timezone.now() - timezone.timedelta(days=10),
            completed_at=timezone.now() - timezone.timedelta(days=10),
        )
        Job.objects.filter(job_id=j_3days.job_id).update(
            created_at=timezone.now() - timezone.timedelta(days=5),
            completed_at=timezone.now() - timezone.timedelta(days=5),
        )

        out = StringIO()
        call_command("cleanup_old_jobs", "--days", "7", "--confirm", stdout=out)

        # j_7days (10 days old) should be deleted, j_3days (5 days old) preserved
        assert Job.objects.filter(job_id=j_7days.job_id).count() == 0
        assert Job.objects.filter(job_id=j_3days.job_id).exists()
        assert "Deleted 1 jobs" in out.getvalue()

    def test_nfs_cleanup(self):
        """NFS directories for old jobs with nfs_job_path are removed."""
        tmpdir = tempfile.mkdtemp()
        try:
            with override_settings(NFS_ROOT=tmpdir):
                job = Job.objects.create(
                    source_file="/nfs.pdf",
                    status=Job.Status.COMPLETED,
                )
                job_path = os.path.join(tmpdir, str(job.job_id))
                os.makedirs(job_path)
                job.nfs_job_path = job_path
                job.save(update_fields=["nfs_job_path"])
                old_date = timezone.now() - timezone.timedelta(days=45)
                Job.objects.filter(job_id=job.job_id).update(
                    created_at=old_date, completed_at=old_date
                )

                assert os.path.isdir(job_path)

                out = StringIO()
                call_command("cleanup_old_jobs", "--confirm", stdout=out)

                assert not os.path.isdir(job_path)
                assert "1 NFS dirs removed" in out.getvalue()
        finally:
            if os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)

    def test_partial_storage_failure_commits_per_job_custody(self):
        tmpdir = tempfile.mkdtemp()
        try:
            old_date = timezone.now() - timezone.timedelta(days=45)
            jobs = []
            for name in ("first", "second"):
                job = Job.objects.create(
                    source_file=f"/{name}.pdf",
                    status=Job.Status.COMPLETED,
                    tenant_id="tenant-retention",
                )
                job_path = os.path.join(tmpdir, "jobs", str(job.job_id))
                os.makedirs(job_path)
                with open(os.path.join(job_path, "artifact.pdf"), "wb") as handle:
                    handle.write(name.encode("ascii"))
                job.nfs_job_path = job_path
                job.save(update_fields=["nfs_job_path"])
                Job.objects.filter(pk=job.pk).update(
                    created_at=old_date,
                    completed_at=old_date,
                )
                jobs.append(job)

            real_remove = __import__(
                "jobs.management.commands.cleanup_old_jobs",
                fromlist=["remove_path_no_follow"],
            ).remove_path_no_follow

            def fail_second(path):
                if str(jobs[1].job_id) in str(path):
                    return False
                return real_remove(path)

            with (
                override_settings(NFS_ROOT=tmpdir),
                patch(
                    "jobs.management.commands.cleanup_old_jobs.remove_path_no_follow",
                    side_effect=fail_second,
                ),
                self.assertRaisesRegex(CommandError, "artifact removal was incomplete"),
            ):
                call_command("cleanup_old_jobs", "--confirm", stdout=StringIO())

            # Per-job sagas commit independently: the first fully successful
            # candidate is removed even though the second remains retryable.
            assert not Job.objects.filter(pk=jobs[0].pk).exists()
            assert Job.objects.filter(pk=jobs[1].pk).exists()
            events = CustodyEvent.objects.filter(
                tenant_id="tenant-retention",
                document_id__startswith="retention-",
            )
            assert events.filter(event_type="data_deletion_started").count() == 2
            assert events.filter(event_type="data_deleted").count() == 1
            assert events.filter(event_type="data_deletion_failed").count() == 1
            assert all(len(event.event_hash) == 64 for event in events)
            assert all(event.chain_finalized for event in events)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_backend_client_exception_records_failure_and_continues(self):
        class BackendClientError(Exception):
            pass

        old_date = timezone.now() - timezone.timedelta(days=45)
        failed = Job.objects.create(
            source_file="/backend-error.pdf",
            status=Job.Status.COMPLETED,
            tenant_id="tenant-backend-error",
            completed_at=old_date,
            storage_backend_used="s3",
        )
        later = Job.objects.create(
            source_file="/later-job.pdf",
            status=Job.Status.COMPLETED,
            tenant_id="tenant-backend-error",
            completed_at=old_date,
            storage_backend_used="nfs",
        )
        broken_backend = MagicMock(backend_name="s3")
        broken_backend.list_objects.side_effect = BackendClientError("access denied")
        nfs_backend = MagicMock(backend_name="nfs")

        def backend_for(job):
            return broken_backend if job.pk == failed.pk else nfs_backend

        with (
            patch("jobs.tasks._get_backend_for_job", side_effect=backend_for),
            override_settings(NFS_ROOT=None),
            self.assertRaisesRegex(CommandError, "artifact removal was incomplete"),
        ):
            call_command("cleanup_old_jobs", "--confirm", stdout=StringIO())

        assert Job.objects.filter(pk=failed.pk).exists()
        assert not Job.objects.filter(pk=later.pk).exists()
        events = CustodyEvent.objects.filter(
            tenant_id="tenant-backend-error",
            document_id=f"retention-{failed.job_id}",
        )
        assert events.filter(event_type="data_deletion_started").count() == 1
        assert events.filter(event_type="data_deletion_failed").count() == 1

    def test_post_storage_custody_failure_cannot_rollback_start_or_bytes(self):
        tmpdir = tempfile.mkdtemp()
        try:
            old_date = timezone.now() - timezone.timedelta(days=45)
            job = Job.objects.create(
                source_file="/custody-failure.pdf",
                status=Job.Status.COMPLETED,
                tenant_id="tenant-retention-custody",
                completed_at=old_date,
            )
            job_path = os.path.join(tmpdir, "jobs", str(job.job_id))
            os.makedirs(job_path)
            with open(os.path.join(job_path, "artifact.pdf"), "wb") as handle:
                handle.write(b"irreversible bytes")
            job.nfs_job_path = job_path
            job.save(update_fields=["nfs_job_path"])

            module = __import__(
                "jobs.management.commands.cleanup_old_jobs",
                fromlist=["_record_deletion_custody_event"],
            )
            real_record = module._record_deletion_custody_event

            def fail_completion(event_type, data, job=None):
                if event_type == "data_deleted":
                    return False
                return real_record(event_type, data, job)

            with (
                override_settings(NFS_ROOT=tmpdir),
                patch(
                    "jobs.management.commands.cleanup_old_jobs._record_deletion_custody_event",
                    side_effect=fail_completion,
                ),
                self.assertRaisesRegex(CommandError, "artifact removal was incomplete"),
            ):
                call_command("cleanup_old_jobs", "--confirm", stdout=StringIO())

            assert not os.path.exists(job_path)
            assert Job.objects.filter(pk=job.pk).exists()
            events = CustodyEvent.objects.filter(
                tenant_id="tenant-retention-custody",
                document_id=f"retention-{job.job_id}",
            )
            assert events.filter(event_type="data_deletion_started").count() == 1
            assert events.filter(event_type="data_deletion_failed").count() == 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_database_delete_failure_preserves_storage_success_custody(self):
        tmpdir = tempfile.mkdtemp()
        try:
            old_date = timezone.now() - timezone.timedelta(days=45)
            job = Job.objects.create(
                source_file="/database-failure.pdf",
                status=Job.Status.COMPLETED,
                tenant_id="tenant-retention-db",
                completed_at=old_date,
            )
            job_path = os.path.join(tmpdir, "jobs", str(job.job_id))
            os.makedirs(job_path)
            with open(os.path.join(job_path, "artifact.pdf"), "wb") as handle:
                handle.write(b"irreversible bytes")
            job.nfs_job_path = job_path
            job.save(update_fields=["nfs_job_path"])

            with (
                override_settings(NFS_ROOT=tmpdir),
                patch.object(Job, "delete", side_effect=RuntimeError("db unavailable")),
                self.assertRaisesRegex(CommandError, "artifact removal was incomplete"),
            ):
                call_command("cleanup_old_jobs", "--confirm", stdout=StringIO())

            assert not os.path.exists(job_path)
            assert Job.objects.filter(pk=job.pk).exists()
            events = CustodyEvent.objects.filter(
                tenant_id="tenant-retention-db",
                document_id=f"retention-{job.job_id}",
            )
            assert events.filter(event_type="data_deletion_started").count() == 1
            # SUCCESS is atomic with row deletion, so a failed database phase
            # cannot leave a false completion boundary.
            assert events.filter(event_type="data_deleted").count() == 0
            assert events.filter(event_type="data_deletion_failed").count() == 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_manual_retention_commits_generation_fence_before_storage(self):
        old_date = timezone.now() - timezone.timedelta(days=45)
        job = Job.objects.create(
            source_file="/fenced-retention.pdf",
            status=Job.Status.FAILED,
            tenant_id="tenant-retention-fence",
            completed_at=old_date,
            dispatch_generation=3,
        )
        capability = PresignedTaskCapability.objects.create(
            job=job,
            operation="process_page",
            dispatch_generation=3,
            payload="",
            expires_at=timezone.now() + timezone.timedelta(days=1),
        )
        observed = {}

        def inspect_fence(_job):
            current = Job.objects.get(pk=job.pk)
            observed.update(
                status=current.status,
                generation=current.dispatch_generation,
                capability_exists=PresignedTaskCapability.objects.filter(
                    pk=capability.pk
                ).exists(),
            )
            return MagicMock(backend_name="nfs")

        with (
            patch("jobs.tasks._get_backend_for_job", side_effect=inspect_fence),
            override_settings(NFS_ROOT=None),
        ):
            call_command("cleanup_old_jobs", "--confirm", stdout=StringIO())

        assert observed == {
            "status": Job.Status.CANCELLED,
            "generation": 4,
            "capability_exists": False,
        }
        assert not Job.objects.filter(pk=job.pk).exists()

    def test_tenant_purge_include_custody_deletes_jobless_tenant_rows(self):
        old_event = CustodyEvent.objects.create(
            job=None,
            tenant_id="tenant-jobless-custody",
            document_id="deleted-job-chain",
            event_type="legacy_event",
        )

        call_command(
            "purge_tenant",
            "--tenant-id",
            "tenant-jobless-custody",
            "--confirm",
            "--include-custody",
            "--skip-output",
            stdout=StringIO(),
        )

        assert not CustodyEvent.objects.filter(pk=old_event.pk).exists()
        assert list(
            CustodyEvent.objects.filter(tenant_id="tenant-jobless-custody")
            .order_by("id")
            .values_list("event_type", flat=True)
        ) == ["tenant_purge_started", "tenant_purged"]

    def test_output_partial_failure_commits_per_job_custody(self):
        tmpdir = tempfile.mkdtemp()
        try:
            old_date = timezone.now() - timezone.timedelta(days=120)
            jobs = []
            for name in ("first-output", "second-output"):
                job = Job.objects.create(
                    source_file=f"/{name}.pdf",
                    status=Job.Status.COMPLETED,
                    tenant_id="tenant-output-retention",
                    completed_at=old_date,
                )
                job_path = os.path.join(tmpdir, "jobs", str(job.job_id))
                pdf_dir = os.path.join(job_path, "output", "EXPORT", "PDF")
                os.makedirs(pdf_dir)
                with open(os.path.join(pdf_dir, "artifact.pdf"), "wb") as handle:
                    handle.write(name.encode("ascii"))
                job.nfs_job_path = job_path
                job.save(update_fields=["nfs_job_path"])
                Job.objects.filter(pk=job.pk).update(completed_at=old_date)
                jobs.append(job)

            real_remove = __import__(
                "jobs.management.commands.cleanup_output",
                fromlist=["remove_path_no_follow"],
            ).remove_path_no_follow

            def fail_second(path):
                if str(jobs[1].job_id) in str(path):
                    return False
                return real_remove(path)

            with (
                override_settings(NFS_ROOT=tmpdir),
                patch(
                    "jobs.management.commands.cleanup_output.remove_path_no_follow",
                    side_effect=fail_second,
                ),
                self.assertRaisesRegex(CommandError, "directories remain"),
            ):
                call_command(
                    "cleanup_output",
                    "--retention-days=90",
                    "--confirm",
                    stdout=StringIO(),
                )

            events = CustodyEvent.objects.filter(
                tenant_id="tenant-output-retention",
                document_id__startswith="output-retention-",
            )
            assert events.filter(event_type="output_deletion_started").count() == 2
            assert events.filter(event_type="output_deleted").count() == 1
            assert events.filter(event_type="output_deletion_failed").count() == 1
            assert all(len(event.event_hash) == 64 for event in events)
            assert all(event.chain_finalized for event in events)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_output_discovery_failure_isolated_from_later_healthy_job(self):
        tmpdir = tempfile.mkdtemp()
        try:
            now = timezone.now()
            poison = Job.objects.create(
                source_file="/poison-output.pdf",
                status=Job.Status.COMPLETED,
                tenant_id="tenant-output-discovery",
                completed_at=now - timezone.timedelta(days=121),
            )
            healthy = Job.objects.create(
                source_file="/healthy-output.pdf",
                status=Job.Status.COMPLETED,
                tenant_id="tenant-output-discovery",
                completed_at=now - timezone.timedelta(days=120),
            )
            healthy_job_path = os.path.join(tmpdir, "jobs", str(healthy.job_id))
            healthy_pdf_dir = os.path.join(healthy_job_path, "output", "EXPORT", "PDF")
            os.makedirs(healthy_pdf_dir)
            healthy_artifact = os.path.join(healthy_pdf_dir, "artifact.pdf")
            with open(healthy_artifact, "wb") as handle:
                handle.write(b"healthy")
            healthy.nfs_job_path = healthy_job_path
            healthy.save(update_fields=["nfs_job_path"])

            visited = []

            def backend_for_job(job):
                visited.append(job.job_id)
                if job.job_id == poison.job_id:
                    raise RuntimeError("binding unavailable")
                return MagicMock(backend_name="nfs")

            with (
                override_settings(NFS_ROOT=tmpdir),
                patch("jobs.tasks._get_backend_for_job", side_effect=backend_for_job),
                self.assertRaisesRegex(CommandError, "output discovery failed"),
            ):
                call_command(
                    "cleanup_output",
                    "--retention-days=90",
                    "--confirm",
                    stdout=StringIO(),
                )

            assert visited == [poison.job_id, healthy.job_id]
            assert not os.path.exists(healthy_artifact)
            failure = CustodyEvent.objects.get(
                tenant_id="tenant-output-discovery",
                document_id=f"output-retention-{poison.job_id}",
                event_type="output_deletion_discovery_failed",
            )
            assert failure.data["phase"] == "discovery"
            assert "binding unavailable" in failure.data["error"]
            assert CustodyEvent.objects.filter(
                tenant_id="tenant-output-discovery",
                document_id=f"output-retention-{healthy.job_id}",
                event_type="output_deleted",
            ).exists()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_output_cleanup_reconciles_committed_start_after_crash_gap(self):
        tmpdir = tempfile.mkdtemp()
        try:
            old_date = timezone.now() - timezone.timedelta(days=120)
            job = Job.objects.create(
                source_file="/output-reconcile.pdf",
                status=Job.Status.FAILED,
                tenant_id="tenant-output-reconcile",
                completed_at=old_date,
            )
            job_path = os.path.join(tmpdir, "jobs", str(job.job_id))
            pdf_dir = os.path.join(job_path, "output", "EXPORT", "PDF")
            os.makedirs(pdf_dir)
            with open(os.path.join(pdf_dir, "artifact.pdf"), "wb") as handle:
                handle.write(b"artifact")
            job.nfs_job_path = job_path
            job.save(update_fields=["nfs_job_path"])
            Job.objects.filter(pk=job.pk).update(completed_at=old_date)

            real_record = __import__(
                "jobs.management.commands.cleanup_output",
                fromlist=["_record_output_job_custody_event"],
            )._record_output_job_custody_event

            def lose_first_success(target_job, event_type, data):
                if event_type == "output_deleted":
                    return False
                return real_record(target_job, event_type, data)

            with (
                override_settings(NFS_ROOT=tmpdir),
                patch(
                    "jobs.management.commands.cleanup_output._record_output_job_custody_event",
                    side_effect=lose_first_success,
                ),
                self.assertRaisesRegex(CommandError, "completion custody failed"),
            ):
                call_command(
                    "cleanup_output",
                    "--retention-days=90",
                    "--confirm",
                    stdout=StringIO(),
                )

            assert not os.path.exists(os.path.join(pdf_dir, "artifact.pdf"))
            job.refresh_from_db()
            assert job.status == Job.Status.CANCELLED
            assert job.settings_json["_output_deletion_fence"] == {
                "original_status": Job.Status.FAILED,
                "fenced_generation": 1,
            }

            with (
                patch(
                    "jobs.tasks._get_backend_for_job",
                    side_effect=RuntimeError("temporary discovery outage"),
                ),
                self.assertRaisesRegex(CommandError, "output discovery failed"),
            ):
                call_command(
                    "cleanup_output",
                    "--retention-days=90",
                    "--confirm",
                    stdout=StringIO(),
                )

            assert list(
                CustodyEvent.objects.filter(
                    tenant_id="tenant-output-reconcile",
                    document_id=f"output-retention-{job.job_id}",
                )
                .order_by("id")
                .values_list("event_type", flat=True)
            ) == ["output_deletion_started", "output_deletion_discovery_failed"]

            call_command(
                "cleanup_output",
                "--retention-days=90",
                "--confirm",
                stdout=StringIO(),
            )
            events = CustodyEvent.objects.filter(
                tenant_id="tenant-output-reconcile",
                document_id=f"output-retention-{job.job_id}",
            )
            assert list(events.order_by("id").values_list("event_type", flat=True)) == [
                "output_deletion_started",
                "output_deletion_discovery_failed",
                "output_deleted",
            ]
            assert (
                events.get(event_type="output_deleted").data[
                    "reconciled_after_interruption"
                ]
                is True
            )
            job.refresh_from_db()
            assert job.status == Job.Status.FAILED
            assert "_output_deletion_fence" not in job.settings_json
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_output_cleanup_fences_retry_and_removes_table_output_planes(self):
        tmpdir = tempfile.mkdtemp()
        try:
            old_date = timezone.now() - timezone.timedelta(days=120)
            job = Job.objects.create(
                source_file="/output-tables.pdf",
                status=Job.Status.FAILED,
                tenant_id="tenant-output-fence",
                completed_at=old_date,
                dispatch_generation=9,
            )
            job_path = os.path.join(tmpdir, "jobs", str(job.job_id))
            output_dir = os.path.join(job_path, "output")
            for subdir in ("TABLES", "TABLE_FALLBACK", "ROUTING"):
                path = os.path.join(output_dir, "EXPORT", subdir)
                os.makedirs(path)
                with open(os.path.join(path, "patient.txt"), "w") as handle:
                    handle.write("Jane Doe")
            job.nfs_job_path = job_path
            job.save(update_fields=["nfs_job_path"])
            outbox = TaskDispatchOutbox.objects.create(
                job=job,
                stage="root",
                generation="9",
                task_name="jobs.tasks.process_document",
            )
            capability = PresignedTaskCapability.objects.create(
                job=job,
                operation="process_document",
                dispatch_generation=9,
                payload="",
                expires_at=timezone.now() + timezone.timedelta(days=1),
            )

            with override_settings(NFS_ROOT=tmpdir):
                call_command(
                    "cleanup_output",
                    "--retention-days=90",
                    "--confirm",
                    stdout=StringIO(),
                )

            job.refresh_from_db()
            outbox.refresh_from_db()
            assert job.status == Job.Status.FAILED
            assert job.dispatch_generation == 10
            assert "_output_deletion_fence" not in job.settings_json
            assert outbox.status == TaskDispatchOutbox.Status.DEAD
            assert not PresignedTaskCapability.objects.filter(pk=capability.pk).exists()
            assert not os.path.exists(os.path.join(output_dir, "EXPORT", "TABLES"))
            assert not os.path.exists(
                os.path.join(output_dir, "EXPORT", "TABLE_FALLBACK")
            )
            assert not os.path.exists(os.path.join(output_dir, "EXPORT", "ROUTING"))
            success = CustodyEvent.objects.get(
                tenant_id="tenant-output-fence",
                document_id=f"output-retention-{job.job_id}",
                event_type="output_deleted",
            )
            assert success.data["fenced_generation"] == 10
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestRotatePiiKey(TestCase):
    def test_target_registry_covers_every_encrypted_text_field(self):
        from jobs.management.commands.rotate_pii_key import _ENCRYPTED_TARGETS

        discovered = {
            (model, field.name)
            for model in apps.get_models()
            for field in model._meta.get_fields()
            if isinstance(field, EncryptedTextField)
        }
        assert set(_ENCRYPTED_TARGETS) == discovered

    def test_failure_rolls_back_prior_model_updates(self):
        from jobs.management.commands.rotate_pii_key import Command as RotateCommand
        from jobs.pii_encryption import reset_encryptor

        old_key = Fernet.generate_key().decode("ascii")
        new_key = Fernet.generate_key().decode("ascii")
        with patch.dict(os.environ, {"PII_ENCRYPTION_KEY": old_key}):
            reset_encryptor()
            job = Job.objects.create(
                source_file="/key-rotation.pdf",
                status=Job.Status.COMPLETED,
                tenant_id="tenant-key-rotation",
            )
            value = ExtractedFormValue.objects.create(
                job=job,
                page_number=1,
                field_key="patient",
                field_value="Jane Doe",
                source_module="form",
            )
            table = connection.ops.quote_name(ExtractedFormValue._meta.db_table)
            column = connection.ops.quote_name("field_value")
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {column} FROM {table} WHERE id = %s", [value.pk]
                )
                before = cursor.fetchone()[0]

            os.environ["PII_ENCRYPTION_KEY"] = new_key
            reset_encryptor()

            original = RotateCommand._rotate_model

            def fail_on_capability(command, **kwargs):
                if kwargs["model"] is PresignedTaskCapability:
                    raise RuntimeError("injected capability rotation failure")
                return original(command, **kwargs)

            with (
                patch.object(RotateCommand, "_rotate_model", fail_on_capability),
                self.assertRaisesRegex(CommandError, "rolled back"),
            ):
                command = RotateCommand()
                command.stdout = StringIO()
                command.handle(
                    old_key=old_key,
                    new_key=new_key,
                    confirm_writers_drained=True,
                    batch_size=500,
                    dry_run=False,
                )

            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {column} FROM {table} WHERE id = %s", [value.pk]
                )
                after = cursor.fetchone()[0]
            assert after == before
            reset_encryptor()

    def test_success_writes_single_layer_ciphertext_readable_by_new_key(self):
        from jobs.management.commands.rotate_pii_key import Command as RotateCommand
        from jobs.pii_encryption import PiiEncryptor, reset_encryptor

        old_key = Fernet.generate_key().decode("ascii")
        new_key = Fernet.generate_key().decode("ascii")
        with patch.dict(os.environ, {"PII_ENCRYPTION_KEY": old_key}):
            reset_encryptor()
            job = Job.objects.create(
                source_file="/successful-key-rotation.pdf",
                status=Job.Status.COMPLETED,
                tenant_id="tenant-key-rotation-success",
            )
            value = ExtractedFormValue.objects.create(
                job=job,
                page_number=1,
                field_key="patient",
                field_value="Jane Doe",
                source_module="form",
            )

            os.environ["PII_ENCRYPTION_KEY"] = new_key
            reset_encryptor()
            command = RotateCommand()
            old_encryptor = PiiEncryptor(old_key)
            new_encryptor = PiiEncryptor(new_key)
            rotated, failed, skipped = command._rotate_model(
                model=ExtractedFormValue,
                field_name="field_value",
                old_encryptor=old_encryptor,
                new_encryptor=new_encryptor,
                old_key=old_key,
                new_key=new_key,
                batch_size=10,
            )

            assert (rotated, failed, skipped) == (1, 0, 0)
            value.refresh_from_db()
            assert value.field_value == "Jane Doe"

            table = connection.ops.quote_name(ExtractedFormValue._meta.db_table)
            column = connection.ops.quote_name("field_value")
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {column} FROM {table} WHERE id = %s", [value.pk]
                )
                raw_value = cursor.fetchone()[0]
            assert new_encryptor.decrypt(raw_value) == "Jane Doe"
            reset_encryptor()


class TestTenantPurgeFence(TestCase):
    def test_unsettled_stage_preserves_source_and_capacity_until_drain(self):
        with tempfile.TemporaryDirectory() as root, override_settings(NFS_ROOT=root):
            job = Job.objects.create(
                tenant_id="purge-stage-drain", source_file="input.pdf",
                status=Job.Status.PROCESSING, dispatch_generation=1,
            )
            job_path = Path(root) / "jobs" / str(job.pk)
            job_path.mkdir(parents=True)
            source = job_path / "input.pdf"
            source.write_bytes(b"owned source evidence")
            job.nfs_job_path = str(job_path)
            job.save(update_fields=["nfs_job_path"])
            stage = PageOCRStage.objects.create(
                job=job, tenant_id=job.tenant_id, generation=1, page_num=1,
                source_sha256="a" * 64, status="rendering", reserved_bytes=4096,
                lease_token="b" * 32,
                lease_expires_at=timezone.now() + timezone.timedelta(minutes=5),
                publish_safe_after=timezone.now() + timezone.timedelta(hours=1),
            )
            PageOCRByteBudget.objects.update_or_create(pk=1, defaults={"reserved_bytes": 4096})
            with self.assertRaisesMessage(CommandError, "not drained"):
                call_command("purge_tenant", tenant_id=job.tenant_id, confirm=True, stdout=StringIO())
            job.refresh_from_db()
            stage.refresh_from_db()
            assert job.status == Job.Status.PROCESSING
            assert job.dispatch_generation == 1
            assert stage.reserved_bytes == 4096
            assert PageOCRByteBudget.objects.get(pk=1).reserved_bytes == 4096
            assert source.read_bytes() == b"owned source evidence"
            assert not CustodyEvent.objects.filter(event_type="tenant_purge_started").exists()

            # Once ordinary worker cleanup has drained, retry can remove the
            # same tenant. No manual deletion of stage or reservation authority.
            stage.status = "done"
            stage.reserved_bytes = 0
            stage.lease_token = ""
            stage.lease_expires_at = None
            stage.publish_safe_after = None
            stage.save()
            PageOCRByteBudget.objects.filter(pk=1).update(reserved_bytes=0)
            call_command("purge_tenant", tenant_id=job.tenant_id, confirm=True, stdout=StringIO())
            assert not Job.objects.filter(pk=job.pk).exists()
            assert not source.exists()
            assert PageOCRByteBudget.objects.get(pk=1).reserved_bytes == 0

    def test_terminal_stage_does_not_hide_outstanding_writer_authority(self):
        for remaining in (
            {"status": "pending"},
            {"reserved_bytes": 1},
            {"lease_token": "c" * 32},
            {"lease_expires_at": timezone.now() - timezone.timedelta(minutes=1)},
            {"publish_safe_after": timezone.now() - timezone.timedelta(minutes=1)},
        ):
            with self.subTest(remaining=remaining):
                job = Job.objects.create(tenant_id="purge-terminal-drain", source_file="input.pdf")
                stage = PageOCRStage.objects.create(
                    job=job, tenant_id=job.tenant_id, generation=1, page_num=1,
                    source_sha256="a" * 64, **{"status": "done", **remaining},
                )
                with self.assertRaisesMessage(CommandError, "not drained"):
                    call_command("purge_tenant", tenant_id=job.tenant_id, confirm=True,
                                 skip_output=True, stdout=StringIO())
                assert Job.objects.filter(pk=job.pk).exists()
                stage.delete()
                job.delete()

    def test_capability_and_job_fence_commit_before_storage_deletion(self):
        tmpdir = tempfile.mkdtemp()
        try:
            key = Fernet.generate_key().decode("ascii")
            job = Job.objects.create(
                source_file="/tenant-purge.pdf",
                status=Job.Status.ASSEMBLING,
                tenant_id="tenant-purge-fence",
            )
            job_path = os.path.join(tmpdir, "jobs", str(job.job_id))
            os.makedirs(job_path)
            job.nfs_job_path = job_path
            job.save(update_fields=["nfs_job_path"])
            with override_settings(S3_PRESIGNED_TASK_KEY=key):
                capability_id = create_presigned_capability(
                    job,
                    "compress_pdf",
                    {
                        "pdf_put": (
                            f"https://s3.example/jobs/{job.job_id}/output/result.pdf"
                        )
                    },
                )

            observed = []

            def storage_delete(**kwargs):
                job.refresh_from_db()
                observed.append(
                    (
                        job.status,
                        PresignedTaskCapability.objects.filter(
                            capability_id=capability_id
                        ).exists(),
                    )
                )
                return 0

            with (
                override_settings(NFS_ROOT=tmpdir),
                patch(
                    "jobs.management.commands.purge_tenant.purge_fs_job_artifacts",
                    side_effect=storage_delete,
                ),
            ):
                call_command(
                    "purge_tenant",
                    "--tenant-id=tenant-purge-fence",
                    "--confirm",
                    stdout=StringIO(),
                )

            assert observed == [(Job.Status.CANCELLED, False)]
            assert not Job.objects.filter(pk=job.pk).exists()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestPurgeTempFiles(TestCase):
    """Tests for the purge_temp_files management command."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.jobs_dir = os.path.join(self.tmpdir, "jobs")
        os.makedirs(self.jobs_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @override_settings()
    def test_removes_orphaned_dirs(self):
        """Directories without matching Job records are removed."""
        from django.conf import settings

        settings.NFS_ROOT = self.tmpdir

        # Create orphan dirs (UUIDs that don't match any Job)
        orphan1 = os.path.join(self.jobs_dir, "orphan-dir-1")
        orphan2 = os.path.join(self.jobs_dir, "orphan-dir-2")
        os.makedirs(orphan1)
        os.makedirs(orphan2)
        # Add a file so we can verify size reporting
        with open(os.path.join(orphan1, "data.bin"), "wb") as f:
            f.write(b"x" * 1024)

        out = StringIO()
        call_command("purge_temp_files", stdout=out)

        assert not os.path.isdir(orphan1)
        assert not os.path.isdir(orphan2)
        assert "Removed 2 orphaned directories" in out.getvalue()

    @override_settings()
    def test_preserves_active_job_dirs(self):
        """Directories matching existing job_ids are not removed."""
        from django.conf import settings

        settings.NFS_ROOT = self.tmpdir

        job = Job.objects.create(source_file="/keep.pdf")
        job_dir = os.path.join(self.jobs_dir, str(job.job_id))
        os.makedirs(job_dir)

        out = StringIO()
        call_command("purge_temp_files", stdout=out)

        assert os.path.isdir(job_dir)
        assert "No orphaned directories" in out.getvalue()

    @override_settings()
    def test_dry_run_no_removal(self):
        """--dry-run reports orphans without deleting them."""
        from django.conf import settings

        settings.NFS_ROOT = self.tmpdir

        orphan = os.path.join(self.jobs_dir, "orphan-dry-run")
        os.makedirs(orphan)

        out = StringIO()
        call_command("purge_temp_files", "--dry-run", stdout=out)

        assert os.path.isdir(orphan)
        assert "DRY RUN" in out.getvalue()

    @override_settings()
    def test_missing_nfs_dir(self):
        """Command handles missing NFS directory gracefully."""
        from django.conf import settings

        settings.NFS_ROOT = os.path.join(self.tmpdir, "nonexistent")

        out = StringIO()
        call_command("purge_temp_files", stdout=out)

        assert "does not exist" in out.getvalue()


class TestFleetStatus(TestCase):
    """Tests for the fleet_status management command."""

    def test_shows_worker_counts(self):
        """Output includes correct counts for workers of different statuses."""
        Worker.objects.create(hostname="w-online", status=Worker.Status.ONLINE)
        Worker.objects.create(
            hostname="w-busy",
            status=Worker.Status.BUSY,
            gpu_available=True,
            gpu_model="RTX 4090",
        )
        Worker.objects.create(hostname="w-offline", status=Worker.Status.OFFLINE)
        Worker.objects.create(hostname="w-drain", status=Worker.Status.DRAINING)

        out = StringIO()
        call_command("fleet_status", stdout=out)
        output = out.getvalue()

        assert "Workers: 4 total" in output
        assert "Online:   1" in output
        assert "Busy:     1" in output
        assert "Draining: 1" in output
        assert "Offline:  1" in output

    def test_shows_active_jobs(self):
        """Active (processing) jobs appear in the active jobs section."""
        job = Job.objects.create(
            source_file="/active.pdf",
            status=Job.Status.PROCESSING,
            total_pages=10,
            pages_completed=3,
        )

        out = StringIO()
        call_command("fleet_status", stdout=out)
        output = out.getvalue()

        assert "Active Jobs:" in output
        assert str(job.job_id)[:8] in output
        assert "3/10" in output

    def test_empty_fleet(self):
        """Command runs without error when there are no workers or jobs."""
        out = StringIO()
        call_command("fleet_status", stdout=out)
        output = out.getvalue()

        assert "Workers: 0 total" in output
        assert "Total:" in output


class TestRotateAuditLogs(TestCase):
    """Tests for deterministic custody archive ordering."""

    @staticmethod
    def _archive_record(record_id, *, prev_hash=None, event_type="event"):
        import hashlib

        timestamp = timezone.now()
        event_dict = {
            "document_id": "archive-doc",
            "event_type": event_type,
            "timestamp": timestamp.isoformat(timespec="milliseconds"),
            "data": {},
            "prev_hash": prev_hash,
        }
        event_hash = hashlib.sha256(
            json.dumps(event_dict, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return {
            "id": record_id,
            "document_id": "archive-doc",
            "job_id": "archive-job",
            "tenant_id": "archive-tenant",
            "chain_id": hashlib.sha256(b"archive-chain").hexdigest(),
            "hash_version": 1,
            "event_type": event_type,
            "timestamp": timestamp.isoformat(),
            "worker_hostname": "worker",
            "data": {},
            "prev_hash": prev_hash or "",
            "event_hash": event_hash,
            "chain_finalized": True,
        }

    def test_archive_uses_id_order_when_timestamps_regress(self):
        job = Job.objects.create(source_file="/clock-skew.pdf")
        first = CustodyEvent.objects.create(
            job=job,
            document_id="clock-skew-doc",
            event_type="first",
            timestamp=timezone.now() - timezone.timedelta(days=3),
        )
        second = CustodyEvent.objects.create(
            job=job,
            document_id="clock-skew-doc",
            event_type="second",
            timestamp=timezone.now() - timezone.timedelta(days=4),
        )

        # Populate the same sealed fields produced by the coordinator so the
        # rotation test exercises ordering rather than an intentionally
        # malformed event.
        import hashlib

        previous = None
        for event in (first, second):
            event_dict = {
                "document_id": event.document_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(timespec="milliseconds"),
                "data": event.data,
                "prev_hash": previous,
            }
            event.prev_hash = previous or ""
            event.event_hash = hashlib.sha256(
                json.dumps(event_dict, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            event.chain_finalized = True
            event.save(update_fields=["prev_hash", "event_hash", "chain_finalized"])
            previous = event.event_hash
        first_id = first.id
        second_id = second.id

        assert first_id < second_id
        assert first.timestamp > second.timestamp

        with tempfile.TemporaryDirectory() as archive_dir:
            call_command(
                "rotate_audit_logs",
                "--retention-days",
                "1",
                "--archive-dir",
                archive_dir,
                "--confirm",
            )

            archives = [
                name for name in os.listdir(archive_dir) if name.endswith(".jsonl")
            ]
            assert len(archives) == 1
            with open(
                os.path.join(archive_dir, archives[0]),
                "r",
                encoding="utf-8",
            ) as archive_file:
                records = [json.loads(line) for line in archive_file if line.strip()]

        assert [record["id"] for record in records] == [first_id, second_id]

    def test_archive_verifier_rejects_every_chain_break_shape(self):
        genesis = self._archive_record(1, event_type="genesis")
        child = self._archive_record(
            2,
            prev_hash=genesis["event_hash"],
            event_type="child",
        )
        cases = {}

        broken_genesis = [dict(genesis), dict(child)]
        broken_genesis[0]["prev_hash"] = "f" * 64
        cases["genesis break"] = (broken_genesis, "genesis break")

        broken_prev = [dict(genesis), dict(child)]
        broken_prev[1]["prev_hash"] = "e" * 64
        cases["prev_hash break"] = (broken_prev, "prev_hash break")

        broken_hash = [dict(genesis), dict(child)]
        broken_hash[1]["event_hash"] = "d" * 64
        cases["event_hash mismatch"] = (broken_hash, "event_hash mismatch")

        high_id_genesis = self._archive_record(2, event_type="high-id-genesis")
        cases["ID order break"] = (
            [high_id_genesis, dict(genesis)],
            "ID order break",
        )

        for case_name, (records, expected_message) in cases.items():
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as tmpdir:
                archive_path = os.path.join(tmpdir, "crafted.jsonl")
                with open(archive_path, "w", encoding="utf-8") as archive_file:
                    for record in records:
                        archive_file.write(json.dumps(record) + "\n")

                is_valid, message = _verify_archive(archive_path)

                assert is_valid is False
                assert expected_message in message

    def test_broken_archive_never_deletes_source_rows(self):
        import hashlib

        job = Job.objects.create(source_file="/broken-chain.pdf")
        document_id = "broken-chain-doc"
        first_timestamp = timezone.now() - timezone.timedelta(days=3)
        first_dict = {
            "document_id": document_id,
            "event_type": "genesis",
            "timestamp": first_timestamp.isoformat(timespec="milliseconds"),
            "data": {},
            "prev_hash": None,
        }
        first_hash = hashlib.sha256(
            json.dumps(first_dict, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        first = CustodyEvent.objects.create(
            job=job,
            document_id=document_id,
            event_type="genesis",
            timestamp=first_timestamp,
            prev_hash="",
            event_hash=first_hash,
            chain_finalized=True,
        )
        second = CustodyEvent.objects.create(
            job=job,
            document_id=document_id,
            event_type="child",
            timestamp=timezone.now() - timezone.timedelta(days=2),
            prev_hash=first_hash,
            event_hash="0" * 64,
            chain_finalized=True,
        )

        with tempfile.TemporaryDirectory() as archive_dir:
            out = StringIO()
            with self.assertRaisesRegex(CommandError, "Archive verification failed"):
                call_command(
                    "rotate_audit_logs",
                    "--retention-days",
                    "1",
                    "--archive-dir",
                    archive_dir,
                    "--confirm",
                    stdout=out,
                )

        assert CustodyEvent.objects.filter(pk=first.pk).exists()
        assert CustodyEvent.objects.filter(pk=second.pk).exists()
        assert not CustodyEvent.objects.filter(
            job=job,
            event_type="audit_logs_rotated",
        ).exists()

    def test_sequential_rotations_continue_from_durable_archive_head(self):
        import hashlib

        job = Job.objects.create(source_file="/sequential.pdf")
        document_id = "sequential-doc"
        chain_id = hashlib.sha256(
            f"job\0{job.job_id}\0{document_id}".encode()
        ).hexdigest()

        def create_event(event_type, timestamp, previous=""):
            envelope = {
                "document_id": document_id,
                "event_type": event_type,
                "timestamp": timestamp.isoformat(timespec="milliseconds"),
                "data": {},
                "prev_hash": previous or None,
            }
            return CustodyEvent.objects.create(
                job=job,
                document_id=document_id,
                chain_id=chain_id,
                hash_version=1,
                event_type=event_type,
                timestamp=timestamp,
                prev_hash=previous,
                event_hash=hashlib.sha256(
                    json.dumps(envelope, sort_keys=True).encode()
                ).hexdigest(),
                chain_finalized=True,
            )

        first = create_event("first", timezone.now() - timezone.timedelta(days=4))
        second = create_event("second", timezone.now(), first.event_hash)

        with tempfile.TemporaryDirectory() as archive_dir:
            call_command(
                "rotate_audit_logs",
                "--retention-days",
                "2",
                "--archive-dir",
                archive_dir,
                "--confirm",
            )
            assert not CustodyEvent.objects.filter(pk=first.pk).exists()
            assert CustodyEvent.objects.filter(pk=second.pk).exists()

            second.timestamp = timezone.now() - timezone.timedelta(days=3)
            envelope = {
                "document_id": document_id,
                "event_type": second.event_type,
                "timestamp": second.timestamp.isoformat(timespec="milliseconds"),
                "data": second.data,
                "prev_hash": first.event_hash,
            }
            second.event_hash = hashlib.sha256(
                json.dumps(envelope, sort_keys=True).encode()
            ).hexdigest()
            second.save(update_fields=["timestamp", "event_hash"])

            call_command(
                "rotate_audit_logs",
                "--retention-days",
                "2",
                "--archive-dir",
                archive_dir,
                "--confirm",
            )
            archives = sorted(
                os.path.join(archive_dir, name)
                for name in os.listdir(archive_dir)
                if name.endswith(".jsonl")
            )
            assert len(archives) == 2
            with open(archives[-1], encoding="utf-8") as archive_file:
                continuation = json.loads(next(archive_file))

        assert continuation["chain_id"] == chain_id
        assert continuation["chain_anchor"] == first.event_hash
        assert len(continuation["previous_archive_checksum"]) == 64
        assert not CustodyEvent.objects.filter(pk=second.pk).exists()

    def test_retry_reuses_deterministic_archive_after_database_abort(self):
        import hashlib

        job = Job.objects.create(source_file="/rotation-retry.pdf")
        timestamp = timezone.now() - timezone.timedelta(days=3)
        envelope = {
            "document_id": "rotation-retry",
            "event_type": "genesis",
            "timestamp": timestamp.isoformat(timespec="milliseconds"),
            "data": {},
            "prev_hash": None,
        }
        event = CustodyEvent.objects.create(
            job=job,
            document_id="rotation-retry",
            event_type="genesis",
            timestamp=timestamp,
            event_hash=hashlib.sha256(
                json.dumps(envelope, sort_keys=True).encode()
            ).hexdigest(),
            chain_finalized=True,
        )

        with tempfile.TemporaryDirectory() as archive_dir:
            with (
                patch(
                    "jobs.management.commands.rotate_audit_logs."
                    "_record_rotation_custody_event",
                    return_value=False,
                ),
                self.assertRaisesRegex(CommandError, "custody event"),
            ):
                call_command(
                    "rotate_audit_logs",
                    "--retention-days",
                    "1",
                    "--archive-dir",
                    archive_dir,
                    "--confirm",
                )
            staged = sorted(Path(archive_dir).glob("*.jsonl"))
            assert len(staged) == 1
            assert CustodyEvent.objects.filter(pk=event.pk).exists()

            call_command(
                "rotate_audit_logs",
                "--retention-days",
                "1",
                "--archive-dir",
                archive_dir,
                "--confirm",
            )
            assert sorted(Path(archive_dir).glob("*.jsonl")) == staged
            assert not CustodyEvent.objects.filter(pk=event.pk).exists()

    def test_deleted_jobs_with_same_document_keep_distinct_chain_identity(self):
        import hashlib

        document_id = "shared-document"
        events = []
        for source in ("/one.pdf", "/two.pdf"):
            job = Job.objects.create(source_file=source)
            timestamp = timezone.now() - timezone.timedelta(days=3)
            envelope = {
                "document_id": document_id,
                "event_type": "genesis",
                "timestamp": timestamp.isoformat(timespec="milliseconds"),
                "data": {},
                "prev_hash": None,
            }
            event = CustodyEvent.objects.create(
                job=job,
                document_id=document_id,
                event_type="genesis",
                timestamp=timestamp,
                event_hash=hashlib.sha256(
                    json.dumps(envelope, sort_keys=True).encode()
                ).hexdigest(),
                chain_finalized=True,
            )
            events.append(event)
            job.delete()

        assert events[0].chain_id != events[1].chain_id
        with tempfile.TemporaryDirectory() as archive_dir:
            call_command(
                "rotate_audit_logs",
                "--retention-days",
                "1",
                "--archive-dir",
                archive_dir,
                "--confirm",
            )
            archive = next(
                os.path.join(archive_dir, name)
                for name in os.listdir(archive_dir)
                if name.endswith(".jsonl")
            )
            valid, message = _verify_archive(archive)

        assert valid, message
        assert not CustodyEvent.objects.filter(pk__in=[event.pk for event in events])


class TestSeedPlaywrightAdmin(TestCase):
    """Tests for the seed_playwright_admin management command."""

    def test_creates_superuser_and_seed_records(self):
        from django.contrib.auth import get_user_model

        out = StringIO()
        call_command(
            "seed_playwright_admin",
            "--username",
            "pw-admin",
            "--password",
            "pw-secret",
            stdout=out,
        )

        user = get_user_model().objects.get(username="pw-admin")
        assert user.is_staff is True
        assert user.is_superuser is True
        assert user.check_password("pw-secret") is True

        assert Job.objects.filter(source_file="/seed/playwright-failed.pdf").exists()
        assert Job.objects.filter(
            source_file="/seed/playwright-processing.pdf"
        ).exists()
        completed = Job.objects.get(source_file="/seed/playwright-completed.pdf")
        assert completed.page_results.count() == 2
        assert completed.custody_events.count() == 2
        assert Worker.objects.filter(hostname="playwright-worker-online").exists()
        assert Worker.objects.filter(hostname="playwright-worker-busy").exists()
        assert "Seeded Playwright admin data" in out.getvalue()

    def test_reset_removes_old_seed_records(self):
        Job.objects.create(
            source_file="/seed/playwright-old.pdf", status=Job.Status.FAILED
        )
        Worker.objects.create(
            hostname="playwright-worker-old", status=Worker.Status.ONLINE
        )

        call_command("seed_playwright_admin", "--reset", "--password", "pw-secret")

        assert Job.objects.filter(source_file="/seed/playwright-old.pdf").count() == 0
        assert Worker.objects.filter(hostname="playwright-worker-old").count() == 0
        assert Job.objects.filter(source_file="/seed/playwright-failed.pdf").exists()

    def test_json_output(self):
        out = StringIO()
        call_command(
            "seed_playwright_admin", "--json", "--password", "pw-secret", stdout=out
        )
        payload = json.loads(out.getvalue())

        assert payload["seeded"] is True
        assert "password" not in payload
        assert payload["jobs"]["failed_job_id"]
        assert payload["workers"]["online_worker"] == "playwright-worker-online"

    def test_requires_explicit_password(self):
        with self.assertRaisesMessage(
            CommandError,
            "seed_playwright_admin requires an explicit --password value.",
        ):
            call_command("seed_playwright_admin")

    @override_settings(DEPLOYMENT_ENV="production", DEBUG=False)
    def test_rejects_production_environment(self):
        with self.assertRaisesMessage(
            CommandError,
            "seed_playwright_admin cannot run when DEPLOYMENT_ENV=production.",
        ):
            call_command("seed_playwright_admin", "--password", "pw-secret")
