"""Tests for C-02: Data retention env var enforcement.

Covers:
- cleanup_completed_jobs emits CustodyEvents before deletion
- cleanup_completed_jobs respects LITIGATION_HOLD
- New Celery Beat schedule entries (6 total)
- purge_temp_files respects LITIGATION_HOLD
- cleanup_pii_entities task filters correctly
- cleanup_output_files task wraps management command
- rotate_audit_logs_task task wraps management command
- DOCUMENT_RETENTION_DAYS dead code removed from cleanup_old_jobs
- Helm configmap has all retention env vars
- values.yaml has retention defaults
"""

import importlib
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

import pytest
import yaml

# Determine project root (the repo root, not the tests/ directory)
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helper: skip when Django is not configured (avoids import errors in root
# test suite that cannot configure Django settings).
# ---------------------------------------------------------------------------
def _django_available():
    """Return True if the full coordinator Django environment can be configured.

    This checks for Django *and* coordinator-specific dependencies like
    ``django_otp`` that are needed for ``INSTALLED_APPS`` to populate.
    Without these, ``django.setup()`` raises ``ModuleNotFoundError``.

    The tests use the repository's SQLite ``coordinator.settings_test``
    module, so they do not require production credentials or a live database.
    """
    for mod in ("django", "django_otp", "django_celery_results", "django_celery_beat"):
        if importlib.util.find_spec(mod) is None:
            return False
    return True


def _configure_django_if_needed():
    """Configure Django settings for test isolation.

    Only called when Django is importable and settings are not yet
    configured.
    """
    os.environ["DJANGO_SETTINGS_MODULE"] = "coordinator.settings_test"
    os.environ["DJANGO_DEBUG"] = "True"
    os.environ["DJANGO_SECRET_KEY"] = "test-secret-key-for-unit-tests"
    os.environ.setdefault("CELERY_BROKER_URL", "memory://")

    import django
    from django.apps import apps

    # Add coordinator to sys.path so 'coordinator.settings_test' resolves.
    # pytest-django may configure settings before it populates the app registry,
    # so settings.configured alone is not a sufficient readiness check.
    coordinator_dir = str(_REPO_ROOT / "coordinator")
    if coordinator_dir not in sys.path:
        sys.path.insert(0, coordinator_dir)
    if not apps.ready:
        django.setup()


# ============================================================================
# Pure-Python tests (no Django required)
# ============================================================================


class TestHelmConfigmap(unittest.TestCase):
    """Verify Helm configmap exposes all retention env vars."""

    def setUp(self):
        self.configmap_path = (
            _REPO_ROOT / "helm" / "ocr-local" / "templates" / "configmap.yaml"
        )

    def test_configmap_has_job_retention_days(self):
        content = self.configmap_path.read_text(encoding="utf-8")
        assert "JOB_RETENTION_DAYS" in content

    def test_configmap_has_pii_entity_retention_days(self):
        content = self.configmap_path.read_text(encoding="utf-8")
        assert "PII_ENTITY_RETENTION_DAYS" in content

    def test_configmap_has_audit_log_retention_days(self):
        content = self.configmap_path.read_text(encoding="utf-8")
        assert "AUDIT_LOG_RETENTION_DAYS" in content

    def test_configmap_has_output_retention_days(self):
        content = self.configmap_path.read_text(encoding="utf-8")
        assert "OUTPUT_RETENTION_DAYS" in content


class TestValuesYaml(unittest.TestCase):
    """Verify values.yaml has retention defaults under cleanup section."""

    def setUp(self):
        self.values_path = _REPO_ROOT / "helm" / "ocr-local" / "values.yaml"
        self.values = yaml.safe_load(self.values_path.read_text(encoding="utf-8"))

    def test_cleanup_section_exists(self):
        assert "cleanup" in self.values

    def test_retention_days_default(self):
        assert self.values["cleanup"]["retentionDays"] == 30

    def test_pii_retention_days_default(self):
        assert self.values["cleanup"]["piiRetentionDays"] == 90

    def test_audit_log_retention_days_default(self):
        assert self.values["cleanup"]["auditLogRetentionDays"] == 2555

    def test_output_retention_days_default(self):
        assert self.values["cleanup"]["outputRetentionDays"] == 90


class TestDocumentRetentionDaysRemoved(unittest.TestCase):
    """Verify _DEFAULT_DOCUMENT_RETENTION_DAYS is removed from cleanup_old_jobs.py."""

    def test_no_document_retention_days_constant(self):
        cmd_path = (
            _REPO_ROOT
            / "coordinator"
            / "jobs"
            / "management"
            / "commands"
            / "cleanup_old_jobs.py"
        )
        content = cmd_path.read_text(encoding="utf-8")
        assert "_DEFAULT_DOCUMENT_RETENTION_DAYS" not in content

    def test_docstring_updated(self):
        cmd_path = (
            _REPO_ROOT
            / "coordinator"
            / "jobs"
            / "management"
            / "commands"
            / "cleanup_old_jobs.py"
        )
        content = cmd_path.read_text(encoding="utf-8")
        # The docstring should reference OUTPUT_RETENTION_DAYS instead
        assert "OUTPUT_RETENTION_DAYS" in content
        # And should NOT reference DOCUMENT_RETENTION_DAYS in the docstring
        assert "DOCUMENT_RETENTION_DAYS" not in content


class TestCeleryBeatSchedule(unittest.TestCase):
    """Verify settings.py CELERY_BEAT_SCHEDULE has all expected entries."""

    def setUp(self):
        self.settings_path = _REPO_ROOT / "coordinator" / "coordinator" / "settings.py"
        self.content = self.settings_path.read_text(encoding="utf-8")

    def test_has_check_worker_heartbeats(self):
        assert "check-worker-heartbeats" in self.content

    def test_has_cleanup_stale_jobs(self):
        assert "cleanup-stale-jobs" in self.content

    def test_has_cleanup_completed_jobs(self):
        assert "cleanup-completed-jobs" in self.content

    def test_has_cleanup_pii_entities(self):
        assert "cleanup-pii-entities" in self.content

    def test_has_cleanup_output_files(self):
        assert "cleanup-output-files" in self.content

    def test_has_rotate_audit_logs(self):
        assert "rotate-audit-logs" in self.content

    def test_schedule_entries_include_custody_repair(self):
        # Count occurrences of 'task': pattern.  Custody publication repair is
        # an intentional lifecycle task. Presigned capability expiry cleanup
        # is a separate entry so bearer material is bounded independently of
        # the artifact-retention jobs. The DB-to-broker outbox relay is also a
        # dedicated lifecycle entry and must remain on the schedule. Durable
        # staged OCR recovery is another required lifecycle task.
        matches = re.findall(r"['\"]task['\"]\s*:", self.content)
        assert len(matches) == 11, f"Expected 11 Beat entries, found {len(matches)}"
        assert '"recover-page-stages"' in self.content
        assert '"task": "jobs.tasks.recover_page_stages"' in self.content


class TestPurgeTempFilesLitigationHold(unittest.TestCase):
    """Verify purge_temp_files.py checks LITIGATION_HOLD."""

    def test_litigation_hold_check_in_source(self):
        cmd_path = (
            _REPO_ROOT
            / "coordinator"
            / "jobs"
            / "management"
            / "commands"
            / "purge_temp_files.py"
        )
        content = cmd_path.read_text(encoding="utf-8")
        # Command imports from the centralized litigation_hold module
        assert "litigation_hold" in content


# ============================================================================
# Django-dependent tests (skipped when Django is not importable)
# ============================================================================


@unittest.skipUnless(_django_available(), "Django not installed")
@pytest.mark.django_db(transaction=True)
class TestCleanupCompletedJobsCustodyEvent(unittest.TestCase):
    """Verify cleanup_completed_jobs emits a CustodyEvent and respects holds."""

    @classmethod
    def setUpClass(cls):
        _configure_django_if_needed()
        # Import after Django is configured
        from jobs.litigation_hold import is_litigation_hold_active
        from jobs.tasks import cleanup_completed_jobs

        cls.cleanup_completed_jobs = staticmethod(cleanup_completed_jobs)
        cls._is_litigation_hold_active = staticmethod(is_litigation_hold_active)

    def test_litigation_hold_skips_cleanup(self):
        """When LITIGATION_HOLD=true, cleanup returns early."""
        with mock.patch.dict(os.environ, {"LITIGATION_HOLD": "true"}):
            result = self.cleanup_completed_jobs.run()
        assert result["deleted"] == 0
        assert result["litigation_hold"] is True

    def test_litigation_hold_values(self):
        """LITIGATION_HOLD accepts 1, true, yes (case-insensitive)."""
        for val in ("1", "true", "True", "YES", "yes"):
            with mock.patch.dict(os.environ, {"LITIGATION_HOLD": val}):
                assert self._is_litigation_hold_active() is True

    def test_litigation_hold_disabled(self):
        """LITIGATION_HOLD=false, 0, empty means not active."""
        for val in ("false", "0", "no", ""):
            with mock.patch.dict(os.environ, {"LITIGATION_HOLD": val}):
                assert self._is_litigation_hold_active() is False

    @mock.patch("jobs.tasks.ExtractedEntity")
    @mock.patch("jobs.tasks.ExtractedFormValue")
    @mock.patch("jobs.tasks._record_control_plane_custody_event")
    @mock.patch("jobs.tasks.Job")
    def test_creates_custody_event_before_deletion(
        self,
        mock_job_cls,
        mock_record_custody,
        mock_form_value_cls,
        mock_entity_cls,
    ):
        """cleanup_completed_jobs should create a CustodyEvent before deleting."""
        # Setup: Job.objects.filter returns queryset with items
        mock_qs = mock.MagicMock()
        mock_qs.count.return_value = 1
        mock_qs.delete.return_value = (1, {"jobs.Job": 1})
        job = mock.MagicMock(
            job_id="job-1",
            tenant_id="",
            nfs_job_path=None,
            storage_backend_used="nfs",
        )
        mock_qs.__iter__ = mock.Mock(return_value=iter([job]))
        mock_qs.select_for_update.return_value = mock_qs
        mock_qs.filter.return_value = mock_qs
        mock_qs.order_by.return_value = mock_qs
        mock_qs.first.return_value = job
        # The cleanup task materializes the locked identity set before
        # touching storage; model that persisted candidate list explicitly.
        mock_qs.values_list.return_value = [
            "job-1",
        ]
        mock_job_cls.objects.filter.return_value = mock_qs
        mock_job_cls.objects.select_for_update.return_value = mock_qs
        mock_job_cls.Status.COMPLETED = "completed"
        mock_job_cls.Status.FAILED = "failed"
        mock_job_cls.Status.CANCELLED = "cancelled"
        mock_form_value_cls.objects.filter.return_value.count.return_value = 0
        mock_entity_cls.objects.filter.return_value.count.return_value = 0

        with mock.patch.dict(
            os.environ, {"LITIGATION_HOLD": "false", "JOB_RETENTION_DAYS": "30"}
        ):
            with (
                mock.patch("jobs.tasks._get_backend_for_job") as mock_backend,
                mock.patch("jobs.tasks.TaskDispatchOutbox"),
                mock.patch("jobs.tasks.PresignedTaskCapability"),
                mock.patch("jobs.tasks.PageOCRStage") as mock_stage,
            ):
                mock_backend.return_value.backend_name = "nfs"
                mock_stage.objects.filter.return_value.filter.return_value.exists.return_value = False
                result = self.cleanup_completed_jobs.run()

        assert result["deleted"] == 1
        event_types = [
            call.kwargs["event_type"] for call in mock_record_custody.call_args_list
        ]
        assert event_types == ["data_deletion_started", "data_deleted"]


@unittest.skipUnless(_django_available(), "Django not installed")
class TestCleanupPiiEntities(unittest.TestCase):
    """Verify cleanup_pii_entities task filters correctly."""

    @classmethod
    def setUpClass(cls):
        _configure_django_if_needed()
        from jobs.tasks import cleanup_pii_entities

        cls.cleanup_pii_entities = staticmethod(cleanup_pii_entities)

    def test_litigation_hold_skips(self):
        with mock.patch.dict(os.environ, {"LITIGATION_HOLD": "true"}):
            result = self.cleanup_pii_entities.run()
        assert result["deleted"] == 0
        assert result["litigation_hold"] is True

    @mock.patch("jobs.tasks.ExtractedEntity")
    @mock.patch("jobs.tasks.DocumentChunk")
    @mock.patch("jobs.tasks.ExtractedFormValue")
    @mock.patch("jobs.tasks.Job")
    @mock.patch("jobs.tasks.PiiEntity")
    def test_deletes_old_pii_entities(
        self,
        mock_pii_cls,
        mock_job_cls,
        mock_form_cls,
        mock_chunk_cls,
        mock_extracted_cls,
    ):
        """Beat retention delegates every derivative job to the shared saga."""
        mock_qs = mock.MagicMock()
        mock_qs.count.return_value = 10
        mock_qs.values_list.return_value = ["job-1"]
        mock_pii_cls.objects.filter.return_value = mock_qs
        mock_job_cls.Status.COMPLETED = "completed"
        mock_job_cls.Status.FAILED = "failed"
        mock_job_cls.Status.CANCELLED = "cancelled"
        job = mock.MagicMock(job_id="job-1")
        job_qs = mock.MagicMock()
        job_qs.order_by.return_value = job_qs
        job_qs.__iter__ = mock.Mock(return_value=iter([job]))
        mock_job_cls.objects.filter.return_value = job_qs
        for model in (mock_extracted_cls, mock_form_cls, mock_chunk_cls):
            model.objects.filter.return_value.values_list.return_value = []

        shared_result = {
            "ok": True,
            "failures": [],
            "deleted": {
                "pii_entities": 10,
                "extracted_entities": 0,
                "form_values": 0,
                "pii_chunks": 0,
            },
            "sidecars_deleted": 2,
            "redis_deleted": 1,
        }
        with (
            mock.patch.dict(
                os.environ,
                {"LITIGATION_HOLD": "false", "PII_ENTITY_RETENTION_DAYS": "90"},
            ),
            mock.patch(
                "jobs.management.commands.purge_pii.erase_pii_for_job",
                return_value=shared_result,
            ) as erase,
        ):
            result = self.cleanup_pii_entities.run()

        assert result["deleted"] == 10
        assert result["retention_days"] == 90
        erase.assert_called_once()
        assert erase.call_args.kwargs["source"] == "celery-beat"

    @mock.patch("jobs.tasks.DocumentChunk")
    @mock.patch("jobs.tasks.ExtractedFormValue")
    @mock.patch("jobs.tasks.ExtractedEntity")
    @mock.patch("jobs.tasks.CustodyEvent")
    @mock.patch("jobs.tasks.Job")
    @mock.patch("jobs.tasks.PiiEntity")
    def test_no_entities_returns_early(
        self,
        mock_pii_cls,
        mock_job_cls,
        mock_ce_cls,
        mock_extracted_cls,
        mock_form_cls,
        mock_chunk_cls,
    ):
        """When no old PII entities exist, return early without deletion."""
        mock_qs = mock.MagicMock()
        mock_qs.count.return_value = 0
        mock_pii_cls.objects.filter.return_value = mock_qs
        mock_extracted_cls.objects.filter.return_value.count.return_value = 0
        mock_form_cls.objects.filter.return_value.count.return_value = 0
        mock_chunk_cls.objects.filter.return_value.count.return_value = 0
        mock_job_cls.Status.COMPLETED = "completed"
        mock_job_cls.Status.FAILED = "failed"
        mock_job_cls.Status.CANCELLED = "cancelled"

        with mock.patch.dict(os.environ, {"LITIGATION_HOLD": "false"}):
            result = self.cleanup_pii_entities.run()

        assert result["deleted"] == 0
        mock_ce_cls.objects.create.assert_not_called()


@unittest.skipUnless(_django_available(), "Django not installed")
class TestCleanupOutputFiles(unittest.TestCase):
    """Verify cleanup_output_files wraps the management command."""

    @classmethod
    def setUpClass(cls):
        _configure_django_if_needed()
        from jobs.tasks import cleanup_output_files

        cls.cleanup_output_files = staticmethod(cleanup_output_files)

    def test_litigation_hold_skips(self):
        with mock.patch.dict(os.environ, {"LITIGATION_HOLD": "yes"}):
            result = self.cleanup_output_files.run()
        assert result["status"] == "skipped"
        assert result["litigation_hold"] is True

    @mock.patch("jobs.tasks.call_command", create=True)
    def test_calls_management_command(self, mock_call_command):
        """cleanup_output_files should call cleanup_output management command."""
        # We need to mock at the point where call_command is imported inside the function
        with mock.patch.dict(
            os.environ, {"LITIGATION_HOLD": "false", "OUTPUT_RETENTION_DAYS": "60"}
        ):
            with mock.patch("django.core.management.call_command") as mock_cmd:
                result = self.cleanup_output_files.run()
        mock_cmd.assert_called_once_with(
            "cleanup_output", "--confirm", "--retention-days=60"
        )
        assert result["status"] == "ok"
        assert result["retention_days"] == 60


@unittest.skipUnless(_django_available(), "Django not installed")
class TestRotateAuditLogsTask(unittest.TestCase):
    """Verify rotate_audit_logs_task wraps the management command."""

    @classmethod
    def setUpClass(cls):
        _configure_django_if_needed()
        from jobs.tasks import rotate_audit_logs_task

        cls.rotate_audit_logs_task = staticmethod(rotate_audit_logs_task)

    def test_litigation_hold_skips(self):
        with mock.patch.dict(os.environ, {"LITIGATION_HOLD": "1"}):
            result = self.rotate_audit_logs_task.run()
        assert result["status"] == "skipped"
        assert result["litigation_hold"] is True

    def test_calls_management_command(self):
        with mock.patch.dict(
            os.environ, {"LITIGATION_HOLD": "false", "AUDIT_LOG_RETENTION_DAYS": "3650"}
        ):
            with mock.patch("django.core.management.call_command") as mock_cmd:
                result = self.rotate_audit_logs_task.run()
        mock_cmd.assert_called_once_with(
            "rotate_audit_logs", "--confirm", "--retention-days=3650"
        )
        assert result["status"] == "ok"
        assert result["retention_days"] == 3650


@unittest.skipUnless(_django_available(), "Django not installed")
class TestTasksHaveLitigationHoldHelper(unittest.TestCase):
    """Verify is_litigation_hold_active is available in tasks module."""

    @classmethod
    def setUpClass(cls):
        _configure_django_if_needed()

    def test_helper_exists_in_tasks_module(self):
        from jobs import tasks

        assert hasattr(tasks, "is_litigation_hold_active")
        assert callable(tasks.is_litigation_hold_active)


if __name__ == "__main__":
    unittest.main()
