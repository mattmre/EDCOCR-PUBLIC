"""Tests for compliance data-retention controls.

Validates LITIGATION_HOLD, tiered retention env vars, and deletion
custody-event recording without requiring a running Django/PostgreSQL
instance.  Django models are mocked at the module boundary.
"""

import importlib
import os
import sys
import types
from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap helpers — build a minimal mock Django environment so the
# management command module can be imported without a real Django project.
# ---------------------------------------------------------------------------

def _build_mock_django():
    """Return a dict of mock modules that satisfy Django imports."""

    # django.utils.timezone
    tz_mod = types.ModuleType("django.utils.timezone")
    tz_mod.now = lambda: datetime.now(timezone.utc)
    tz_mod.timedelta = timedelta

    # django.core.management.base
    base_mod = types.ModuleType("django.core.management.base")

    class _FakeStyle:
        def SUCCESS(self, msg):
            return msg

        def WARNING(self, msg):
            return msg

    class FakeBaseCommand:
        help = ""
        stdout = StringIO()
        style = _FakeStyle()

        def add_arguments(self, parser):
            pass

    base_mod.BaseCommand = FakeBaseCommand

    # django module stubs
    django_mod = types.ModuleType("django")
    django_utils = types.ModuleType("django.utils")
    django_core = types.ModuleType("django.core")
    django_core_management = types.ModuleType("django.core.management")

    return {
        "django": django_mod,
        "django.utils": django_utils,
        "django.utils.timezone": tz_mod,
        "django.core": django_core,
        "django.core.management": django_core_management,
        "django.core.management.base": base_mod,
    }


def _build_mock_job_models():
    """Return a mock ``jobs.models`` module with Job, CustodyEvent, PiiEntity."""

    models_mod = types.ModuleType("jobs.models")

    # -- Job -----------------------------------------------------------------
    class _JobStatus:
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"

    class _FakeQuerySet:
        """Minimal queryset stand-in supporting filter/count/delete/slice."""

        def __init__(self, items):
            self._items = list(items)

        def filter(self, **kwargs):
            return _FakeQuerySet(self._items)

        def exclude(self, **kwargs):
            return _FakeQuerySet(self._items)

        def count(self):
            return len(self._items)

        def delete(self):
            n = len(self._items)
            self._items.clear()
            return (n, {})

        def first(self):
            return self._items[0] if self._items else None

        def __getitem__(self, key):
            return self._items[key]

        def __iter__(self):
            return iter(self._items)

        def __len__(self):
            return len(self._items)

    class FakeJob:
        Status = _JobStatus

        class objects:
            _store = []

            @classmethod
            def filter(cls, **kwargs):
                return _FakeQuerySet(cls._store)

            @classmethod
            def count(cls):
                return len(cls._store)

            @classmethod
            def exclude(cls, **kwargs):
                return _FakeQuerySet(cls._store)

            @classmethod
            def delete(cls):
                n = len(cls._store)
                cls._store.clear()
                return (n, {})

            @classmethod
            def first(cls):
                return cls._store[0] if cls._store else None

    # -- CustodyEvent --------------------------------------------------------
    class FakeCustodyEvent:
        _created = []

        class objects:
            @classmethod
            def create(cls, **kwargs):
                FakeCustodyEvent._created.append(kwargs)
                return MagicMock()

    # -- PiiEntity -----------------------------------------------------------
    class FakePiiEntity:
        class objects:
            _store = []

            @classmethod
            def filter(cls, **kwargs):
                return _FakeQuerySet(cls._store)

            @classmethod
            def count(cls):
                return len(cls._store)

            @classmethod
            def delete(cls):
                n = len(cls._store)
                cls._store.clear()
                return (n, {})

    models_mod.Job = FakeJob
    models_mod.CustodyEvent = FakeCustodyEvent
    models_mod.PiiEntity = FakePiiEntity

    jobs_mod = types.ModuleType("jobs")

    return {
        "jobs": jobs_mod,
        "jobs.models": models_mod,
    }


# ---------------------------------------------------------------------------
# Fixture — import the command module with mocks in place
# ---------------------------------------------------------------------------

@pytest.fixture()
def cleanup_module(monkeypatch):
    """Import cleanup_old_jobs under a mocked Django/jobs environment.

    Returns a namespace dict with ``cmd_module``, ``FakeJob``,
    ``FakeCustodyEvent``, ``FakePiiEntity``.
    """
    django_mocks = _build_mock_django()
    model_mocks = _build_mock_job_models()

    all_mocks = {**django_mocks, **model_mocks}

    saved = {}
    for name, mod in all_mocks.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    # Inject the real litigation_hold module so the command can import it
    lh_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "coordinator", "jobs", "litigation_hold.py",
    )
    lh_spec = importlib.util.spec_from_file_location("jobs.litigation_hold", lh_path)
    lh_mod = importlib.util.module_from_spec(lh_spec)
    lh_spec.loader.exec_module(lh_mod)
    saved["jobs.litigation_hold"] = sys.modules.get("jobs.litigation_hold")
    sys.modules["jobs.litigation_hold"] = lh_mod

    # Remove any cached import of the command module itself
    cmd_key = "coordinator.jobs.management.commands.cleanup_old_jobs"
    saved[cmd_key] = sys.modules.pop(cmd_key, None)
    # Also clear the shorter key that importlib may use
    short_key = "cleanup_old_jobs"
    saved[short_key] = sys.modules.pop(short_key, None)

    cmd_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "coordinator",
        "jobs",
        "management",
        "commands",
        "cleanup_old_jobs.py",
    )
    spec = importlib.util.spec_from_file_location("cleanup_old_jobs", cmd_path)
    cmd_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmd_module)

    FakeJob = all_mocks["jobs.models"].Job
    FakeCustodyEvent = all_mocks["jobs.models"].CustodyEvent
    FakePiiEntity = all_mocks["jobs.models"].PiiEntity

    # Reset state between tests
    FakeJob.objects._store = []
    FakeCustodyEvent._created = []
    FakePiiEntity.objects._store = []

    yield {
        "cmd_module": cmd_module,
        "FakeJob": FakeJob,
        "FakeCustodyEvent": FakeCustodyEvent,
        "FakePiiEntity": FakePiiEntity,
    }

    # Restore original sys.modules entries
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


# ===========================================================================
# Tests
# ===========================================================================


class TestLitigationHold:
    """LITIGATION_HOLD should block all automated deletions."""

    @staticmethod
    def _get_shared_check():
        return sys.modules["jobs.litigation_hold"].is_litigation_hold_active

    def test_litigation_hold_true_blocks_cleanup(self, cleanup_module, monkeypatch):
        monkeypatch.setenv("LITIGATION_HOLD", "true")
        assert self._get_shared_check()() is True

    def test_litigation_hold_yes_blocks_cleanup(self, cleanup_module, monkeypatch):
        monkeypatch.setenv("LITIGATION_HOLD", "yes")
        assert self._get_shared_check()() is True

    def test_litigation_hold_1_blocks_cleanup(self, cleanup_module, monkeypatch):
        monkeypatch.setenv("LITIGATION_HOLD", "1")
        assert self._get_shared_check()() is True

    def test_litigation_hold_TRUE_case_insensitive(self, cleanup_module, monkeypatch):
        monkeypatch.setenv("LITIGATION_HOLD", "TRUE")
        assert self._get_shared_check()() is True

    def test_litigation_hold_false_allows_cleanup(self, cleanup_module, monkeypatch):
        monkeypatch.setenv("LITIGATION_HOLD", "false")
        assert self._get_shared_check()() is False

    def test_litigation_hold_unset_allows_cleanup(self, cleanup_module, monkeypatch):
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        assert self._get_shared_check()() is False

    def test_command_handle_exits_on_litigation_hold(self, cleanup_module, monkeypatch):
        """Command.handle() should return immediately when litigation hold is active."""
        mod = cleanup_module["cmd_module"]
        FakeJob = cleanup_module["FakeJob"]
        FakeCustodyEvent = cleanup_module["FakeCustodyEvent"]

        monkeypatch.setenv("LITIGATION_HOLD", "true")

        # Put something in the store to prove nothing gets deleted
        FakeJob.objects._store = [MagicMock()]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.stderr = StringIO()
        cmd.handle(days=1, dry_run=False, include_pii=False)

        output = cmd.stderr.getvalue()
        assert "LITIGATION_HOLD" in output
        assert "blocked" in output

        # Nothing should have been deleted
        assert len(FakeJob.objects._store) == 1
        assert len(FakeCustodyEvent._created) == 0


class TestRetentionEnvVars:
    """Env vars for tiered retention should be read correctly."""

    def test_pii_entity_retention_default(self, cleanup_module, monkeypatch):
        mod = cleanup_module["cmd_module"]
        monkeypatch.delenv("PII_ENTITY_RETENTION_DAYS", raising=False)

        result = mod._get_retention_days(
            "PII_ENTITY_RETENTION_DAYS", mod._DEFAULT_PII_ENTITY_RETENTION_DAYS
        )
        assert result == 90

    def test_pii_entity_retention_custom(self, cleanup_module, monkeypatch):
        mod = cleanup_module["cmd_module"]
        monkeypatch.setenv("PII_ENTITY_RETENTION_DAYS", "30")

        result = mod._get_retention_days(
            "PII_ENTITY_RETENTION_DAYS", mod._DEFAULT_PII_ENTITY_RETENTION_DAYS
        )
        assert result == 30

    def test_audit_log_retention_default(self, cleanup_module, monkeypatch):
        mod = cleanup_module["cmd_module"]
        monkeypatch.delenv("AUDIT_LOG_RETENTION_DAYS", raising=False)

        result = mod._get_retention_days(
            "AUDIT_LOG_RETENTION_DAYS", mod._DEFAULT_AUDIT_LOG_RETENTION_DAYS
        )
        assert result == 2555

    def test_document_retention_default(self, cleanup_module, monkeypatch):
        mod = cleanup_module["cmd_module"]
        monkeypatch.delenv("DOCUMENT_RETENTION_DAYS", raising=False)

        # DOCUMENT_RETENTION_DAYS uses OUTPUT_RETENTION_DAYS (90 days) via
        # the cleanup_output command; cleanup_old_jobs does not define a
        # separate _DEFAULT_DOCUMENT_RETENTION_DAYS constant.
        result = mod._get_retention_days("DOCUMENT_RETENTION_DAYS", 365)
        assert result == 365

    def test_invalid_retention_value_falls_back_to_default(
        self, cleanup_module, monkeypatch
    ):
        mod = cleanup_module["cmd_module"]
        monkeypatch.setenv("PII_ENTITY_RETENTION_DAYS", "not_a_number")

        result = mod._get_retention_days(
            "PII_ENTITY_RETENTION_DAYS", mod._DEFAULT_PII_ENTITY_RETENTION_DAYS
        )
        assert result == 90

    def test_job_retention_days_default(self, cleanup_module, monkeypatch):
        mod = cleanup_module["cmd_module"]
        monkeypatch.delenv("JOB_RETENTION_DAYS", raising=False)

        result = mod._get_retention_days(
            "JOB_RETENTION_DAYS", mod._DEFAULT_JOB_RETENTION_DAYS
        )
        assert result == 30


class TestDeletionCustodyEvents:
    """Deletion operations must create custody audit events."""

    def test_job_cleanup_creates_custody_event(self, cleanup_module, monkeypatch):
        """cleanup_old_jobs should record a data_deleted custody event."""
        mod = cleanup_module["cmd_module"]
        FakeJob = cleanup_module["FakeJob"]
        FakeCustodyEvent = cleanup_module["FakeCustodyEvent"]

        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        # Populate fake store with jobs to delete
        fake_job = MagicMock()
        fake_job.job_id = "test-job-1"
        fake_job.status = "completed"
        fake_job.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        fake_job.nfs_job_path = ""
        FakeJob.objects._store = [fake_job]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd._cleanup_jobs(days=1, dry_run=False)

        # Verify a custody event was created
        assert len(FakeCustodyEvent._created) >= 1
        event = FakeCustodyEvent._created[0]
        assert event["event_type"] == "data_deleted"
        assert "retention_policy" in event["data"]
        assert "JOB_RETENTION_DAYS=1" in event["data"]["retention_policy"]
        assert event["data"]["reason"] == "retention_policy"

    def test_pii_cleanup_creates_custody_event(self, cleanup_module, monkeypatch):
        """cleanup_pii_entities should record a pii_deleted custody event."""
        mod = cleanup_module["cmd_module"]
        FakeJob = cleanup_module["FakeJob"]
        FakePiiEntity = cleanup_module["FakePiiEntity"]
        FakeCustodyEvent = cleanup_module["FakeCustodyEvent"]

        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        # Need at least one job for the custody event FK
        fake_job = MagicMock()
        FakeJob.objects._store = [fake_job]

        # Populate PII entities
        FakePiiEntity.objects._store = [MagicMock(), MagicMock()]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd._cleanup_pii_entities(days=90, dry_run=False)

        assert len(FakeCustodyEvent._created) >= 1
        event = FakeCustodyEvent._created[0]
        assert event["event_type"] == "pii_deleted"
        assert "PII_ENTITY_RETENTION_DAYS=90" in event["data"]["retention_policy"]

    def test_dry_run_does_not_create_custody_event(self, cleanup_module, monkeypatch):
        """Dry-run mode should not create custody events or delete data."""
        mod = cleanup_module["cmd_module"]
        FakeJob = cleanup_module["FakeJob"]
        FakeCustodyEvent = cleanup_module["FakeCustodyEvent"]

        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        fake_job = MagicMock()
        fake_job.job_id = "test-job-dry"
        fake_job.status = "completed"
        fake_job.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        FakeJob.objects._store = [fake_job]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd._cleanup_jobs(days=1, dry_run=True)

        # No custody events in dry-run mode
        assert len(FakeCustodyEvent._created) == 0
        # Data should still be present
        assert len(FakeJob.objects._store) == 1

    def test_pii_dry_run_no_deletion(self, cleanup_module, monkeypatch):
        """PII dry-run should neither delete nor record events."""
        mod = cleanup_module["cmd_module"]
        FakePiiEntity = cleanup_module["FakePiiEntity"]
        FakeCustodyEvent = cleanup_module["FakeCustodyEvent"]

        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        FakePiiEntity.objects._store = [MagicMock()]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd._cleanup_pii_entities(days=90, dry_run=True)

        assert len(FakeCustodyEvent._created) == 0
        assert len(FakePiiEntity.objects._store) == 1


class TestCustodyEventRecording:
    """Test the _record_deletion_custody_event helper."""

    def test_record_with_job(self, cleanup_module):
        mod = cleanup_module["cmd_module"]
        FakeCustodyEvent = cleanup_module["FakeCustodyEvent"]

        fake_job = MagicMock()
        mod._record_deletion_custody_event(
            event_type="data_deleted",
            data={"count": 5},
            job=fake_job,
        )

        assert len(FakeCustodyEvent._created) == 1
        assert FakeCustodyEvent._created[0]["event_type"] == "data_deleted"
        assert FakeCustodyEvent._created[0]["job"] == fake_job

    def test_record_without_job_uses_first(self, cleanup_module):
        mod = cleanup_module["cmd_module"]
        FakeJob = cleanup_module["FakeJob"]
        FakeCustodyEvent = cleanup_module["FakeCustodyEvent"]

        sentinel_job = MagicMock()
        FakeJob.objects._store = [sentinel_job]

        mod._record_deletion_custody_event(
            event_type="pii_deleted",
            data={"count": 3},
        )

        assert len(FakeCustodyEvent._created) == 1
        assert FakeCustodyEvent._created[0]["job"] == sentinel_job

    def test_record_without_any_job_logs_instead(self, cleanup_module):
        """When no Job exists, the helper should log rather than raise."""
        mod = cleanup_module["cmd_module"]
        FakeJob = cleanup_module["FakeJob"]
        FakeCustodyEvent = cleanup_module["FakeCustodyEvent"]

        FakeJob.objects._store = []

        # Should not raise
        mod._record_deletion_custody_event(
            event_type="data_deleted",
            data={"count": 0},
        )

        # No CustodyEvent created (no job to attach to)
        assert len(FakeCustodyEvent._created) == 0


class TestIncludePiiFlag:
    """The --include-pii flag triggers PII entity cleanup alongside jobs."""

    def test_include_pii_triggers_pii_cleanup(self, cleanup_module, monkeypatch):
        mod = cleanup_module["cmd_module"]
        FakeJob = cleanup_module["FakeJob"]
        FakePiiEntity = cleanup_module["FakePiiEntity"]
        FakeCustodyEvent = cleanup_module["FakeCustodyEvent"]

        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        monkeypatch.setenv("PII_ENTITY_RETENTION_DAYS", "60")

        fake_job = MagicMock()
        FakeJob.objects._store = [fake_job]
        FakePiiEntity.objects._store = [MagicMock()]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(days=30, dry_run=False, include_pii=True)

        # Should have custody events for both job and PII cleanup
        event_types = [e["event_type"] for e in FakeCustodyEvent._created]
        assert "data_deleted" in event_types
        assert "pii_deleted" in event_types


class TestModuleConstants:
    """Verify the retention default constants are correctly defined."""

    def test_default_constants(self, cleanup_module):
        mod = cleanup_module["cmd_module"]
        assert mod._DEFAULT_JOB_RETENTION_DAYS == 30
        assert mod._DEFAULT_PII_ENTITY_RETENTION_DAYS == 90
        assert mod._DEFAULT_AUDIT_LOG_RETENTION_DAYS == 2555
