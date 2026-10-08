"""Tests for C-03: audit deletion events via CustodyEvent.

Validates that:
- purge_temp_files emits CustodyEvent records after each directory purge
- cleanup_old_jobs includes operator identity in custody event data
- dry-run mode does NOT emit CustodyEvent records
- event data payloads contain required fields

Django models are mocked at the module boundary (same approach as
test_compliance_retention.py).
"""

import getpass
import importlib
import os
import sys
import types
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap helpers -- minimal mock Django environment
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

    class FakeCommandError(Exception):
        pass

    base_mod.CommandError = FakeCommandError

    # django.conf -- needed by purge_temp_files
    conf_mod = types.ModuleType("django.conf")

    class _FakeSettings:
        NFS_ROOT = "/tmp/test-nfs"

    conf_mod.settings = _FakeSettings()

    transaction_mod = types.ModuleType("django.db.transaction")

    def atomic(func=None):
        if callable(func):

            def wrapped(*args, **kwargs):
                with nullcontext():
                    return func(*args, **kwargs)

            return wrapped
        return nullcontext()

    transaction_mod.atomic = atomic
    db_mod = types.ModuleType("django.db")
    db_mod.transaction = transaction_mod

    # django module stubs
    django_mod = types.ModuleType("django")
    django_utils = types.ModuleType("django.utils")
    django_core = types.ModuleType("django.core")
    django_core_management = types.ModuleType("django.core.management")

    return {
        "django": django_mod,
        "django.conf": conf_mod,
        "django.db": db_mod,
        "django.db.transaction": transaction_mod,
        "django.utils": django_utils,
        "django.utils.timezone": tz_mod,
        "django.core": django_core,
        "django.core.management": django_core_management,
        "django.core.management.base": base_mod,
    }


class _FakeQuerySet:
    """Minimal queryset stand-in supporting filter/count/delete/slice/values_list."""

    def __init__(self, items):
        self._items = list(items)

    def filter(self, **kwargs):
        return _FakeQuerySet(self._items)

    def exclude(self, **kwargs):
        return _FakeQuerySet(self._items)

    def values_list(self, *fields, flat=False):
        return _FakeQuerySet(self._items)

    def count(self):
        return len(self._items)

    def delete(self):
        n = len(self._items)
        self._items.clear()
        return (n, {})

    def update(self, **kwargs):
        for item in self._items:
            for key, value in kwargs.items():
                setattr(item, key, value)
        return len(self._items)

    def first(self):
        return self._items[0] if self._items else None

    def select_for_update(self):
        return self

    def __getitem__(self, key):
        return self._items[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


def _build_mock_job_models():
    """Return a mock ``jobs.models`` module with Job, CustodyEvent, PiiEntity."""

    models_mod = types.ModuleType("jobs.models")

    class _JobStatus:
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"

    class FakeJob:
        Status = _JobStatus

        class objects:
            _store = []

            @classmethod
            def filter(cls, **kwargs):
                return _FakeQuerySet(cls._store)

            @classmethod
            def select_for_update(cls):
                return _FakeQuerySet(cls._store)

            @classmethod
            def count(cls):
                return len(cls._store)

            @classmethod
            def exclude(cls, **kwargs):
                return _FakeQuerySet(cls._store)

            @classmethod
            def values_list(cls, *fields, flat=False):
                return _FakeQuerySet([])

            @classmethod
            def delete(cls):
                n = len(cls._store)
                cls._store.clear()
                return (n, {})

            @classmethod
            def first(cls):
                return cls._store[0] if cls._store else None

    class FakeCustodyEvent:
        _created = []

        class objects:
            @classmethod
            def create(cls, **kwargs):
                FakeCustodyEvent._created.append(kwargs)
                return MagicMock()

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

    class FakeTaskDispatchOutbox:
        class Status:
            PENDING = "pending"
            FAILED = "failed"
            DEAD = "dead"

        class objects:
            @classmethod
            def filter(cls, **kwargs):
                return _FakeQuerySet([])

    class FakePresignedTaskCapability:
        class objects:
            @classmethod
            def filter(cls, **kwargs):
                return _FakeQuerySet([])

    models_mod.TaskDispatchOutbox = FakeTaskDispatchOutbox
    models_mod.PresignedTaskCapability = FakePresignedTaskCapability

    jobs_mod = types.ModuleType("jobs")
    jobs_mod.__path__ = []

    return {
        "jobs": jobs_mod,
        "jobs.models": models_mod,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _import_command(monkeypatch, command_name, django_mocks, model_mocks):
    """Import a management command module with mocks in place."""
    all_mocks = {**django_mocks, **model_mocks}

    # Mock jobs.litigation_hold module used by cleanup commands
    litigation_mod = types.ModuleType("jobs.litigation_hold")
    litigation_mod.check_litigation_hold = MagicMock(return_value=False)
    litigation_mod.is_litigation_hold_active = MagicMock(return_value=False)
    all_mocks["jobs.litigation_hold"] = litigation_mod

    # cleanup_old_jobs resolves the backend captured at ingest.  Keep the
    # command's import boundary honest without importing the real Django task
    # module into this deliberately isolated fake-Django test.
    tasks_mod = types.ModuleType("jobs.tasks")

    class _FakeNfsBackend:
        backend_name = "nfs"

    tasks_mod._get_backend_for_job = lambda _job: _FakeNfsBackend()
    tasks_mod._record_control_plane_custody_event = lambda **kwargs: all_mocks[
        "jobs.models"
    ].CustodyEvent.objects.create(job=None, **kwargs)
    all_mocks["jobs.tasks"] = tasks_mod

    rls_mod = types.ModuleType("jobs.rls")
    rls_mod.control_plane_rls_command = lambda func: func
    all_mocks["jobs.rls"] = rls_mod

    saved = {}
    for name, mod in all_mocks.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    # Remove cached command module imports
    cmd_key = f"coordinator.jobs.management.commands.{command_name}"
    saved[cmd_key] = sys.modules.pop(cmd_key, None)
    saved[command_name] = sys.modules.pop(command_name, None)

    cmd_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "coordinator",
        "jobs",
        "management",
        "commands",
        f"{command_name}.py",
    )
    spec = importlib.util.spec_from_file_location(command_name, cmd_path)
    cmd_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(cmd_module)
    except Exception:
        _restore_modules(saved)
        raise

    return cmd_module, all_mocks, saved


def _restore_modules(saved):
    """Restore original sys.modules entries."""
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture()
def purge_module(monkeypatch):
    """Import purge_temp_files under a mocked Django/jobs environment."""
    django_mocks = _build_mock_django()
    model_mocks = _build_mock_job_models()

    cmd_module, all_mocks, saved = _import_command(
        monkeypatch, "purge_temp_files", django_mocks, model_mocks
    )

    FakeJob = all_mocks["jobs.models"].Job
    FakeCustodyEvent = all_mocks["jobs.models"].CustodyEvent

    # Reset state
    FakeJob.objects._store = []
    FakeCustodyEvent._created = []

    yield {
        "cmd_module": cmd_module,
        "FakeJob": FakeJob,
        "FakeCustodyEvent": FakeCustodyEvent,
        "django_mocks": django_mocks,
    }

    _restore_modules(saved)


@pytest.fixture()
def cleanup_module(monkeypatch):
    """Import cleanup_old_jobs under a mocked Django/jobs environment."""
    django_mocks = _build_mock_django()
    model_mocks = _build_mock_job_models()

    cmd_module, all_mocks, saved = _import_command(
        monkeypatch, "cleanup_old_jobs", django_mocks, model_mocks
    )

    FakeJob = all_mocks["jobs.models"].Job
    FakeCustodyEvent = all_mocks["jobs.models"].CustodyEvent
    FakePiiEntity = all_mocks["jobs.models"].PiiEntity

    # Reset state
    FakeJob.objects._store = []
    FakeCustodyEvent._created = []
    FakePiiEntity.objects._store = []

    yield {
        "cmd_module": cmd_module,
        "FakeJob": FakeJob,
        "FakeCustodyEvent": FakeCustodyEvent,
        "FakePiiEntity": FakePiiEntity,
    }

    _restore_modules(saved)


# ===========================================================================
# Tests: purge_temp_files emits CustodyEvent
# ===========================================================================


class TestPurgeTempFilesCustodyEvent:
    """purge_temp_files should emit CustodyEvent after each directory purge."""

    def test_purge_emits_custody_event(self, purge_module, tmp_path):
        """Each purged orphan directory should produce a custody event."""
        mod = purge_module["cmd_module"]
        FakeJob = purge_module["FakeJob"]
        FakeCustodyEvent = purge_module["FakeCustodyEvent"]

        # Create a fake NFS jobs dir with an orphan directory
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        orphan = jobs_dir / "orphan-dir-1"
        orphan.mkdir()
        # Create a file inside so size > 0
        (orphan / "page.pdf").write_bytes(b"x" * 1024)

        # Provide an anchor job for the FK
        anchor_job = MagicMock()
        FakeJob.objects._store = [anchor_job]

        # Override NFS_ROOT via the django.conf mock
        django_conf = purge_module["django_mocks"]["django.conf"]
        django_conf.settings.NFS_ROOT = str(tmp_path)

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(dry_run=False)

        # Verify custody event was created
        assert [event["event_type"] for event in FakeCustodyEvent._created] == [
            "temp_files_deletion_started",
            "temp_files_purged",
        ]
        event = FakeCustodyEvent._created[-1]
        assert event["event_type"] == "temp_files_purged"
        assert event["data"]["action"] == "temp_purge"
        assert event["data"]["reason"] == "orphaned_directory"
        assert "operator" in event["data"]
        assert event["data"]["size_bytes"] == 1024
        assert str(orphan) in event["data"]["directory"]

    def test_purge_multiple_orphans_emits_multiple_events(self, purge_module, tmp_path):
        """Each orphan directory should get its own custody event."""
        mod = purge_module["cmd_module"]
        FakeJob = purge_module["FakeJob"]
        FakeCustodyEvent = purge_module["FakeCustodyEvent"]

        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        (jobs_dir / "orphan-a").mkdir()
        (jobs_dir / "orphan-b").mkdir()

        anchor_job = MagicMock()
        FakeJob.objects._store = [anchor_job]

        django_conf = purge_module["django_mocks"]["django.conf"]
        django_conf.settings.NFS_ROOT = str(tmp_path)

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(dry_run=False)

        assert len(FakeCustodyEvent._created) == 4
        assert [event["event_type"] for event in FakeCustodyEvent._created] == [
            "temp_files_deletion_started",
            "temp_files_purged",
            "temp_files_deletion_started",
            "temp_files_purged",
        ]

    def test_purge_event_data_payload_structure(self, purge_module, tmp_path):
        """Custody event data must contain all required fields."""
        mod = purge_module["cmd_module"]
        FakeJob = purge_module["FakeJob"]
        FakeCustodyEvent = purge_module["FakeCustodyEvent"]

        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        orphan = jobs_dir / "test-orphan"
        orphan.mkdir()
        (orphan / "data.bin").write_bytes(b"y" * 512)

        FakeJob.objects._store = [MagicMock()]

        django_conf = purge_module["django_mocks"]["django.conf"]
        django_conf.settings.NFS_ROOT = str(tmp_path)

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(dry_run=False)

        assert len(FakeCustodyEvent._created) == 2
        data = FakeCustodyEvent._created[-1]["data"]

        required_keys = {"action", "directory", "size_bytes", "reason", "operator"}
        assert required_keys.issubset(set(data.keys()))
        assert data["action"] == "temp_purge"
        assert data["reason"] == "orphaned_directory"
        assert isinstance(data["size_bytes"], int)
        assert isinstance(data["operator"], str)
        assert len(data["operator"]) > 0

    def test_purge_no_job_available_does_not_raise(self, purge_module, tmp_path):
        """When no Job exists, custody event should be skipped gracefully."""
        mod = purge_module["cmd_module"]
        FakeJob = purge_module["FakeJob"]
        FakeCustodyEvent = purge_module["FakeCustodyEvent"]

        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        (jobs_dir / "orphan-no-job").mkdir()

        FakeJob.objects._store = []  # No jobs available

        django_conf = purge_module["django_mocks"]["django.conf"]
        django_conf.settings.NFS_ROOT = str(tmp_path)

        cmd = mod.Command()
        cmd.stdout = StringIO()
        # Should not raise
        cmd.handle(dry_run=False)

        # Control-plane purge events are standalone and remain auditable even
        # when there is no OCR Job FK to anchor them to.
        assert len(FakeCustodyEvent._created) == 2
        assert all(event["job"] is None for event in FakeCustodyEvent._created)


# ===========================================================================
# Tests: dry-run does NOT emit CustodyEvent
# ===========================================================================


class TestPurgeDryRunNoCustodyEvent:
    """Dry-run mode should not produce any CustodyEvent records."""

    def test_dry_run_does_not_emit_custody_event(self, purge_module, tmp_path):
        """purge_temp_files --dry-run should not create any custody events."""
        mod = purge_module["cmd_module"]
        FakeJob = purge_module["FakeJob"]
        FakeCustodyEvent = purge_module["FakeCustodyEvent"]

        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        orphan = jobs_dir / "orphan-dry"
        orphan.mkdir()
        (orphan / "file.txt").write_bytes(b"data")

        FakeJob.objects._store = [MagicMock()]

        django_conf = purge_module["django_mocks"]["django.conf"]
        django_conf.settings.NFS_ROOT = str(tmp_path)

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(dry_run=True)

        # No custody events in dry-run mode
        assert len(FakeCustodyEvent._created) == 0
        # Directory should still exist
        assert orphan.exists()

    def test_dry_run_preserves_directories(self, purge_module, tmp_path):
        """Dry-run should not remove any directories."""
        mod = purge_module["cmd_module"]
        FakeJob = purge_module["FakeJob"]

        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        orphan = jobs_dir / "keep-me"
        orphan.mkdir()

        FakeJob.objects._store = [MagicMock()]

        django_conf = purge_module["django_mocks"]["django.conf"]
        django_conf.settings.NFS_ROOT = str(tmp_path)

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(dry_run=True)

        assert orphan.exists()


# ===========================================================================
# Tests: cleanup_old_jobs includes operator identity
# ===========================================================================


class TestCleanupOldJobsOperator:
    """cleanup_old_jobs custody events must include operator identity."""

    def test_job_cleanup_includes_operator(self, cleanup_module, monkeypatch):
        """data_deleted custody event should contain 'operator' field."""
        mod = cleanup_module["cmd_module"]
        FakeJob = cleanup_module["FakeJob"]
        FakeCustodyEvent = cleanup_module["FakeCustodyEvent"]

        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        fake_job = MagicMock()
        fake_job.job_id = "test-job-op"
        fake_job.status = "completed"
        fake_job.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        fake_job.nfs_job_path = ""
        FakeJob.objects._store = [fake_job]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd._cleanup_jobs(days=1, dry_run=False)

        assert len(FakeCustodyEvent._created) >= 1
        event = next(
            item
            for item in FakeCustodyEvent._created
            if item["event_type"] == "data_deleted" and "operator" in item["data"]
        )
        assert event["event_type"] == "data_deleted"
        assert "operator" in event["data"]
        assert isinstance(event["data"]["operator"], str)
        assert len(event["data"]["operator"]) > 0

    def test_pii_cleanup_uses_shared_operator_audited_saga(
        self, cleanup_module, monkeypatch
    ):
        """Retention delegates to the fenced purge primitive with operator audit."""
        mod = cleanup_module["cmd_module"]
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        import inspect
        from pathlib import Path

        retention_source = inspect.getsource(mod.Command._cleanup_pii_entities)
        purge_source = (
            Path(__file__).resolve().parents[1]
            / "coordinator"
            / "jobs"
            / "management"
            / "commands"
            / "purge_pii.py"
        ).read_text(encoding="utf-8")
        assert "erase_pii_for_job" in retention_source
        assert '"operator": getpass.getuser()' in purge_source
        assert 'event_type="pii_deletion_started"' in purge_source
        assert 'event_type="pii_purged"' in purge_source

    def test_operator_matches_current_user(self, cleanup_module, monkeypatch):
        """Operator field should match getpass.getuser() output."""
        mod = cleanup_module["cmd_module"]
        FakeJob = cleanup_module["FakeJob"]
        FakeCustodyEvent = cleanup_module["FakeCustodyEvent"]

        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        fake_job = MagicMock()
        fake_job.job_id = "test-job-user"
        fake_job.status = "completed"
        fake_job.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        fake_job.nfs_job_path = ""
        FakeJob.objects._store = [fake_job]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd._cleanup_jobs(days=1, dry_run=False)

        event = next(
            item for item in FakeCustodyEvent._created if "operator" in item["data"]
        )
        assert event["data"]["operator"] == getpass.getuser()

    def test_dry_run_no_operator_event(self, cleanup_module, monkeypatch):
        """Dry-run should not create custody events (and thus no operator)."""
        mod = cleanup_module["cmd_module"]
        FakeJob = cleanup_module["FakeJob"]
        FakeCustodyEvent = cleanup_module["FakeCustodyEvent"]

        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        fake_job = MagicMock()
        fake_job.job_id = "test-dry-op"
        fake_job.status = "completed"
        fake_job.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        FakeJob.objects._store = [fake_job]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd._cleanup_jobs(days=1, dry_run=True)

        assert len(FakeCustodyEvent._created) == 0


# ===========================================================================
# Tests: custody.py EVENT_TYPES
# ===========================================================================


class TestCustodyEventTypes:
    """Verify temp_files_purged is registered in custody.py EVENT_TYPES."""

    def test_temp_files_purged_in_event_types(self):
        """The temp_files_purged event type should be defined in custody.py."""
        from custody import EVENT_TYPES

        assert "temp_files_purged" in EVENT_TYPES
        assert isinstance(EVENT_TYPES["temp_files_purged"], str)
        assert len(EVENT_TYPES["temp_files_purged"]) > 0
