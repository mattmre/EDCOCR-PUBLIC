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

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, override_settings
from django.utils import timezone

from jobs.models import Job, Worker


class TestCleanupOldJobs(TestCase):
    """Tests for the cleanup_old_jobs management command."""

    def test_deletes_old_completed_jobs(self):
        """Old completed/failed/cancelled jobs are deleted by default (30 days)."""
        j1 = Job.objects.create(source_file="/old1.pdf", status=Job.Status.COMPLETED)
        j2 = Job.objects.create(source_file="/old2.pdf", status=Job.Status.FAILED)
        j3 = Job.objects.create(source_file="/old3.pdf", status=Job.Status.CANCELLED)
        old_date = timezone.now() - timezone.timedelta(days=45)
        Job.objects.filter(
            job_id__in=[j1.job_id, j2.job_id, j3.job_id]
        ).update(created_at=old_date)

        out = StringIO()
        call_command("cleanup_old_jobs", stdout=out)

        assert Job.objects.filter(job_id=j1.job_id).count() == 0
        assert Job.objects.filter(job_id=j2.job_id).count() == 0
        assert Job.objects.filter(job_id=j3.job_id).count() == 0
        # deleted_count from Django delete() includes cascade-deleted related
        # objects (e.g. CustodyEvent created by _record_deletion_custody_event),
        # so we check for the message pattern rather than an exact count.
        output = out.getvalue()
        assert "Deleted" in output
        assert "jobs older than" in output

    def test_preserves_recent_jobs(self):
        """Jobs created recently (within retention period) are not deleted."""
        j_recent = Job.objects.create(
            source_file="/recent.pdf", status=Job.Status.COMPLETED
        )
        # created_at is auto_now_add, so it's set to now by default

        out = StringIO()
        call_command("cleanup_old_jobs", stdout=out)

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
        call_command("cleanup_old_jobs", stdout=out)

        assert Job.objects.filter(job_id=j_processing.job_id).exists()
        assert Job.objects.filter(job_id=j_ingesting.job_id).exists()
        assert Job.objects.filter(job_id=j_assembling.job_id).exists()

    def test_dry_run_no_deletion(self):
        """--dry-run previews deletions without actually deleting jobs."""
        job = Job.objects.create(
            source_file="/dryrun.pdf", status=Job.Status.COMPLETED
        )
        old_date = timezone.now() - timezone.timedelta(days=45)
        Job.objects.filter(job_id=job.job_id).update(created_at=old_date)

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
            created_at=timezone.now() - timezone.timedelta(days=10)
        )
        Job.objects.filter(job_id=j_3days.job_id).update(
            created_at=timezone.now() - timezone.timedelta(days=5)
        )

        out = StringIO()
        call_command("cleanup_old_jobs", "--days", "7", stdout=out)

        # j_7days (10 days old) should be deleted, j_3days (5 days old) preserved
        assert Job.objects.filter(job_id=j_7days.job_id).count() == 0
        assert Job.objects.filter(job_id=j_3days.job_id).exists()
        assert "Deleted 1 jobs" in out.getvalue()

    def test_nfs_cleanup(self):
        """NFS directories for old jobs with nfs_job_path are removed."""
        tmpdir = tempfile.mkdtemp()
        try:
            job = Job.objects.create(
                source_file="/nfs.pdf",
                status=Job.Status.COMPLETED,
                nfs_job_path=tmpdir,
            )
            old_date = timezone.now() - timezone.timedelta(days=45)
            Job.objects.filter(job_id=job.job_id).update(created_at=old_date)

            assert os.path.isdir(tmpdir)

            out = StringIO()
            call_command("cleanup_old_jobs", stdout=out)

            assert not os.path.isdir(tmpdir)
            assert "1 NFS dirs removed" in out.getvalue()
        finally:
            if os.path.isdir(tmpdir):
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
            hostname="w-busy", status=Worker.Status.BUSY,
            gpu_available=True, gpu_model="RTX 4090",
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
        assert Job.objects.filter(source_file="/seed/playwright-processing.pdf").exists()
        completed = Job.objects.get(source_file="/seed/playwright-completed.pdf")
        assert completed.page_results.count() == 2
        assert completed.custody_events.count() == 2
        assert Worker.objects.filter(hostname="playwright-worker-online").exists()
        assert Worker.objects.filter(hostname="playwright-worker-busy").exists()
        assert "Seeded Playwright admin data" in out.getvalue()

    def test_reset_removes_old_seed_records(self):
        Job.objects.create(source_file="/seed/playwright-old.pdf", status=Job.Status.FAILED)
        Worker.objects.create(hostname="playwright-worker-old", status=Worker.Status.ONLINE)

        call_command("seed_playwright_admin", "--reset", "--password", "pw-secret")

        assert Job.objects.filter(source_file="/seed/playwright-old.pdf").count() == 0
        assert Worker.objects.filter(hostname="playwright-worker-old").count() == 0
        assert Job.objects.filter(source_file="/seed/playwright-failed.pdf").exists()

    def test_json_output(self):
        out = StringIO()
        call_command("seed_playwright_admin", "--json", "--password", "pw-secret", stdout=out)
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
