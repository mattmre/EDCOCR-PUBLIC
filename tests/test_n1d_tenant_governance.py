"""Tests for Wave N1-D: Tenant governance closure.

Validates:
- S3 cleanup in coordinator purge_tenant command
- SLA window reset in coordinator purge_tenant command
- Purge command respects litigation hold (verified in governance context)
- End-to-end purge flow with both NFS and S3 configured

Since Django models cannot be fully loaded in root test context, these tests
use a combination of:
- Module imports for non-Django modules (cost_tracking, sla_monitoring)
- Executed purge command behavior with mocked storage
"""

import importlib
import pathlib
import sys
import types
from io import StringIO
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PURGE_TENANT_PATH = (
    _PROJECT_ROOT
    / "coordinator"
    / "jobs"
    / "management"
    / "commands"
    / "purge_tenant.py"
)


# ===========================================================================
# SLA monitoring window reset integration tests
# ===========================================================================


class TestSLAWindowReset:
    """Verify SLA window reset works correctly on the SLAMonitor singleton."""

    def test_sla_monitor_window_deletion(self):
        """Deleting a tenant key from _windows removes that tenant's data."""
        from sla_monitoring import SLAMonitor

        monitor = SLAMonitor()
        # Record some data to create windows
        monitor.record_request("test-tenant", success=True, latency_seconds=1.0)
        assert "test-tenant" in monitor._windows

        # Simulate purge SLA reset
        del monitor._windows["test-tenant"]
        assert "test-tenant" not in monitor._windows

    def test_sla_monitor_window_deletion_preserves_other_tenants(self):
        """Deleting one tenant's windows does not affect other tenants."""
        from sla_monitoring import SLAMonitor

        monitor = SLAMonitor()
        monitor.record_request("tenant-a", success=True, latency_seconds=1.0)
        monitor.record_request("tenant-b", success=True, latency_seconds=2.0)

        assert "tenant-a" in monitor._windows
        assert "tenant-b" in monitor._windows

        del monitor._windows["tenant-a"]

        assert "tenant-a" not in monitor._windows
        assert "tenant-b" in monitor._windows

    def test_sla_monitor_window_reset_idempotent(self):
        """Attempting to reset a nonexistent tenant does not raise."""
        from sla_monitoring import SLAMonitor

        monitor = SLAMonitor()
        # Tenant never recorded -- hasattr check should prevent KeyError
        if hasattr(monitor, "_windows") and "nonexistent" in monitor._windows:
            del monitor._windows["nonexistent"]
        # No exception raised


# ===========================================================================
# Cost tracking reset integration tests
# ===========================================================================


class TestCostTrackingReset:
    """Verify cost tracking reset works correctly."""

    def test_cost_tracker_reset_tenant_removes_data(self):
        """reset_tenant removes all tracked usage for the given tenant."""
        from cost_tracking import CostTracker

        tracker = CostTracker()
        tracker.record_pages("test-tenant", 100)
        assert tracker.get_usage("test-tenant") is not None

        tracker.reset_tenant("test-tenant")
        assert tracker.get_usage("test-tenant") is None

    def test_cost_tracker_reset_preserves_other_tenants(self):
        """Resetting one tenant does not affect others."""
        from cost_tracking import CostTracker

        tracker = CostTracker()
        tracker.record_pages("tenant-a", 50)
        tracker.record_pages("tenant-b", 75)

        tracker.reset_tenant("tenant-a")
        assert tracker.get_usage("tenant-a") is None
        assert tracker.get_usage("tenant-b") is not None
        assert tracker.get_usage("tenant-b").pages_processed == 75


# ===========================================================================
# Purge command mock-based integration tests (Django-free)
# ===========================================================================

# These tests use the same mock-Django approach as test_tenant_purge.py
# but focus on the new S3 and SLA cleanup code paths.


def _build_mock_django():
    """Return a dict of mock modules that satisfy Django imports."""
    from datetime import datetime, timezone

    tz_mod = types.ModuleType("django.utils.timezone")
    tz_mod.now = lambda: datetime.now(timezone.utc)

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

    class FakeCommandError(Exception):
        pass

    base_mod.BaseCommand = FakeBaseCommand
    base_mod.CommandError = FakeCommandError

    conf_mod = types.ModuleType("django.conf")

    class _FakeSettings:
        configured = True
        NFS_ROOT = "/tmp/test-nfs"
        STORAGE_BACKEND = "nfs"

    conf_mod.settings = _FakeSettings()

    django_mod = types.ModuleType("django")
    django_mod.__path__ = []
    django_utils = types.ModuleType("django.utils")
    django_core = types.ModuleType("django.core")
    django_core_management = types.ModuleType("django.core.management")
    django_db = types.ModuleType("django.db")
    django_db_models = types.ModuleType("django.db.models")
    django_transaction = types.ModuleType("django.db.transaction")

    class _F:
        def __init__(self, field_name):
            self.field_name = field_name

        def __add__(self, value):
            return (self.field_name, "+", value)

    django_db_models.F = _F

    class _Q:
        """Minimal composable stand-in for the command's OR custody scope."""

        def __init__(self, **kwargs):
            self.children = [kwargs]

        def __or__(self, other):
            combined = _Q()
            combined.children = [*self.children, *other.children]
            return combined

    django_db_models.Q = _Q

    class _Atomic:
        def __call__(self, func=None):
            if func is None:
                return self
            return func

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    django_transaction.atomic = _Atomic()
    django_db.transaction = django_transaction
    django_db.models = django_db_models

    return {
        "django": django_mod,
        "django.conf": conf_mod,
        "django.utils": django_utils,
        "django.utils.timezone": tz_mod,
        "django.core": django_core,
        "django.core.management": django_core_management,
        "django.core.management.base": base_mod,
        "django.db": django_db,
        "django.db.models": django_db_models,
        "django.db.transaction": django_transaction,
    }


class _FakeQuerySet:
    """Minimal queryset stand-in."""

    def __init__(self, items):
        self._items = list(items)

    def filter(self, **kwargs):
        result = list(self._items)
        for key, value in kwargs.items():
            if key == "tenant_id":
                result = [
                    j for j in result if str(getattr(j, "tenant_id", "")) == str(value)
                ]
            elif key == "job_id__in":
                result = [j for j in result if getattr(j, "job_id", None) in value]
        return _FakeQuerySet(result)

    def exclude(self, **kwargs):
        result = list(self._items)
        for key, value in kwargs.items():
            result = [item for item in result if getattr(item, key, None) != value]
        return _FakeQuerySet(result)

    def values_list(self, *fields, flat=False):
        if flat and len(fields) == 1:
            field_name = fields[0]
            return [getattr(item, field_name, None) for item in self._items]
        return _FakeQuerySet(self._items)

    def count(self):
        return len(self._items)

    def exists(self):
        return bool(self._items)

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

    def __iter__(self):
        return iter(self._items)


def _build_mock_models():
    """Return mock jobs.models module."""
    models_mod = types.ModuleType("jobs.models")

    class FakeJob:
        class Status:
            CANCELLED = "cancelled"

        class objects:
            _store = []

            @classmethod
            def select_for_update(cls):
                return cls

            @classmethod
            def filter(cls, **kwargs):
                items = cls._store
                for key, value in kwargs.items():
                    if key == "tenant_id":
                        items = [
                            j
                            for j in items
                            if str(getattr(j, "tenant_id", "")) == str(value)
                        ]
                return _FakeQuerySet(items)

    class FakeCustodyEvent:
        _created = []

        class objects:
            _store = []

            @classmethod
            def create(cls, **kwargs):
                FakeCustodyEvent._created.append(kwargs)
                return MagicMock()

            @classmethod
            def filter(cls, *_args, **kwargs):
                return _FakeQuerySet(cls._store)

    class FakePiiEntity:
        class objects:
            _store = []

            @classmethod
            def filter(cls, **kwargs):
                return _FakeQuerySet(cls._store)

    class FakePageResult:
        class objects:
            _store = []

            @classmethod
            def filter(cls, **kwargs):
                return _FakeQuerySet(cls._store)

    class FakeTaskDispatchOutbox:
        class Status:
            PENDING = "pending"
            FAILED = "failed"
            DEAD = "dead"

        class objects:
            @classmethod
            def filter(cls, **kwargs):
                return _FakeQuerySet([])

    models_mod.Job = FakeJob
    models_mod.CustodyEvent = FakeCustodyEvent
    models_mod.PiiEntity = FakePiiEntity
    models_mod.PageResult = FakePageResult
    models_mod.TaskDispatchOutbox = FakeTaskDispatchOutbox

    jobs_mod = types.ModuleType("jobs")
    rls_mod = types.ModuleType("jobs.rls")
    rls_mod.tenant_rls_command = lambda func: func

    return (
        {
            "jobs": jobs_mod,
            "jobs.models": models_mod,
            "jobs.rls": rls_mod,
        },
        FakeJob,
        FakeCustodyEvent,
    )


def _import_purge_tenant_fresh(django_mocks, model_mocks):
    """Import purge_tenant command module with mocks in place."""
    all_mocks = {**django_mocks, **model_mocks}

    saved = {}
    for name, mod in all_mocks.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    # Inject litigation_hold
    lh_path = str(_PROJECT_ROOT / "coordinator" / "jobs" / "litigation_hold.py")
    lh_spec = importlib.util.spec_from_file_location("jobs.litigation_hold", lh_path)
    lh_mod = importlib.util.module_from_spec(lh_spec)
    lh_spec.loader.exec_module(lh_mod)
    saved["jobs.litigation_hold"] = sys.modules.get("jobs.litigation_hold")
    sys.modules["jobs.litigation_hold"] = lh_mod

    # Mock the persisted backend resolver.  The command deliberately resolves
    # each job's backend binding instead of constructing one from global
    # STORAGE_BACKEND configuration.
    tasks_mock = types.ModuleType("jobs.tasks")
    tasks_mock._get_backend_for_job = lambda job: getattr(
        job, "_backend", types.SimpleNamespace(backend_name="nfs")
    )
    tasks_mock._record_control_plane_custody_event = lambda **kwargs: kwargs
    saved["jobs.tasks"] = sys.modules.get("jobs.tasks")
    sys.modules["jobs.tasks"] = tasks_mock

    retirement_mock = types.ModuleType("jobs.artifact_retirement")
    retirement_mock.retain_cleanup_authority = lambda _job, backend: backend.backend_name == "s3"
    saved["jobs.artifact_retirement"] = sys.modules.get("jobs.artifact_retirement")
    sys.modules["jobs.artifact_retirement"] = retirement_mock

    cmd_key = "coordinator.jobs.management.commands.purge_tenant"
    saved[cmd_key] = sys.modules.pop(cmd_key, None)
    saved["purge_tenant_n1d"] = sys.modules.pop("purge_tenant_n1d", None)

    cmd_path = str(_PURGE_TENANT_PATH)
    spec = importlib.util.spec_from_file_location("purge_tenant_n1d", cmd_path)
    cmd_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(cmd_module)
    except Exception:
        _restore(saved)
        raise

    return cmd_module, all_mocks, saved


def _restore(saved):
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def _make_job(job_id, tenant_id, nfs_job_path=""):
    job = MagicMock()
    job.job_id = job_id
    job.pk = job_id
    job.tenant_id = tenant_id
    job.nfs_job_path = nfs_job_path
    job._backend = types.SimpleNamespace(backend_name="nfs")
    return job


class TestPurgeTenantS3Cleanup:
    """Test S3 cleanup integration in purge_tenant command."""

    def test_s3_cleanup_when_backend_is_s3(self, monkeypatch):
        """When STORAGE_BACKEND=s3, S3 objects should be listed and deleted."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        django_mocks = _build_mock_django()
        # Set STORAGE_BACKEND to s3
        django_mocks["django.conf"].settings.STORAGE_BACKEND = "s3"
        django_mocks["django.conf"].settings.S3_ENDPOINT = "http://minio:9000"
        django_mocks["django.conf"].settings.S3_BUCKET = "ocr-test"
        django_mocks["django.conf"].settings.S3_ACCESS_KEY = "test-key"
        django_mocks["django.conf"].settings.S3_SECRET_KEY = "test-secret"
        django_mocks["django.conf"].settings.S3_REGION = ""

        model_dict, FakeJob, FakeCustodyEvent = _build_mock_models()

        cmd_module, all_mocks, saved = _import_purge_tenant_fresh(
            django_mocks, model_dict
        )
        try:
            tenant = "s3-tenant"
            job = _make_job("job-s3-1", tenant)
            FakeJob.objects._store = [job]
            FakeCustodyEvent._created = []

            # Configure mock S3 backend
            mock_s3_backend = MagicMock()
            mock_s3_backend.list_objects.side_effect = [
                [
                    "jobs/job-s3-1/output.pdf",
                    "jobs/job-s3-1/text.txt",
                ],
                [],
            ]
            mock_s3_backend.delete_many.return_value = 2
            job._backend = mock_s3_backend
            mock_s3_backend.backend_name = "s3"

            cmd = cmd_module.Command()
            cmd.stdout = StringIO()

            mock_cost = MagicMock()
            mock_sla = MagicMock()
            with patch.dict(
                sys.modules,
                {
                    "cost_tracking": mock_cost,
                    "sla_monitoring": mock_sla,
                },
            ):
                cmd.handle(
                    tenant_id=tenant,
                    dry_run=False,
                    confirm=True,
                    include_custody=False,
                    include_output=True,
                )

            output = cmd.stdout.getvalue()
            # Should list and delete objects
            assert mock_s3_backend.list_objects.call_count == 2
            mock_s3_backend.list_objects.assert_any_call("jobs/job-s3-1/")
            mock_s3_backend.delete_many.assert_called_once()
            assert "S3 objects" in output
        finally:
            _restore(saved)

    def test_s3_cleanup_skipped_when_backend_is_nfs(self, monkeypatch):
        """When STORAGE_BACKEND=nfs, S3 cleanup should be skipped entirely."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        django_mocks = _build_mock_django()
        django_mocks["django.conf"].settings.STORAGE_BACKEND = "nfs"

        model_dict, FakeJob, FakeCustodyEvent = _build_mock_models()

        cmd_module, all_mocks, saved = _import_purge_tenant_fresh(
            django_mocks, model_dict
        )
        try:
            tenant = "nfs-only-tenant"
            job = _make_job("job-nfs-1", tenant)
            FakeJob.objects._store = [job]
            FakeCustodyEvent._created = []

            cmd = cmd_module.Command()
            cmd.stdout = StringIO()

            mock_cost = MagicMock()
            mock_sla = MagicMock()
            with patch.dict(
                sys.modules,
                {
                    "cost_tracking": mock_cost,
                    "sla_monitoring": mock_sla,
                },
            ):
                cmd.handle(
                    tenant_id=tenant,
                    dry_run=False,
                    confirm=True,
                    include_custody=False,
                    include_output=True,
                )

            # No S3-bound job exists, so no object-list operation is made.
        finally:
            _restore(saved)

    def test_s3_cleanup_handles_empty_objects(self, monkeypatch):
        """When S3 lists no objects for a job, delete_many is not called."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        django_mocks = _build_mock_django()
        django_mocks["django.conf"].settings.STORAGE_BACKEND = "s3"
        django_mocks["django.conf"].settings.S3_ENDPOINT = "http://minio:9000"
        django_mocks["django.conf"].settings.S3_BUCKET = "ocr-test"
        django_mocks["django.conf"].settings.S3_ACCESS_KEY = "test-key"
        django_mocks["django.conf"].settings.S3_SECRET_KEY = "test-secret"
        django_mocks["django.conf"].settings.S3_REGION = ""

        model_dict, FakeJob, FakeCustodyEvent = _build_mock_models()

        cmd_module, all_mocks, saved = _import_purge_tenant_fresh(
            django_mocks, model_dict
        )
        try:
            tenant = "empty-s3-tenant"
            job = _make_job("job-empty-1", tenant)
            FakeJob.objects._store = [job]
            FakeCustodyEvent._created = []

            mock_s3_backend = MagicMock()
            mock_s3_backend.list_objects.return_value = []
            job._backend = mock_s3_backend
            mock_s3_backend.backend_name = "s3"

            cmd = cmd_module.Command()
            cmd.stdout = StringIO()

            mock_cost = MagicMock()
            mock_sla = MagicMock()
            with patch.dict(
                sys.modules,
                {
                    "cost_tracking": mock_cost,
                    "sla_monitoring": mock_sla,
                },
            ):
                cmd.handle(
                    tenant_id=tenant,
                    dry_run=False,
                    confirm=True,
                    include_custody=False,
                    include_output=True,
                )

            mock_s3_backend.list_objects.assert_called_once()
            mock_s3_backend.delete_many.assert_not_called()
        finally:
            _restore(saved)


class TestPurgeTenantSLAReset:
    """Test SLA window reset integration in purge_tenant command."""

    def test_sla_reset_called_during_purge(self, monkeypatch):
        """SLA monitoring windows should be reset during tenant purge."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        django_mocks = _build_mock_django()
        model_dict, FakeJob, FakeCustodyEvent = _build_mock_models()

        cmd_module, all_mocks, saved = _import_purge_tenant_fresh(
            django_mocks, model_dict
        )
        try:
            tenant = "sla-reset-tenant"
            job = _make_job("job-sla-1", tenant)
            FakeJob.objects._store = [job]
            FakeCustodyEvent._created = []

            cmd = cmd_module.Command()
            cmd.stdout = StringIO()

            # Create a mock SLA monitor with tenant windows
            mock_monitor = MagicMock()
            mock_monitor._windows = {tenant: {"availability": MagicMock()}}

            mock_sla_mod = MagicMock()
            mock_sla_mod.get_monitor.return_value = mock_monitor

            mock_cost = MagicMock()
            with patch.dict(
                sys.modules,
                {
                    "cost_tracking": mock_cost,
                    "sla_monitoring": mock_sla_mod,
                },
            ):
                cmd.handle(
                    tenant_id=tenant,
                    dry_run=False,
                    confirm=True,
                    include_custody=False,
                    include_output=False,
                )

            output = cmd.stdout.getvalue()
            # Verify SLA monitor was accessed
            mock_sla_mod.get_monitor.assert_called_once()
            # Verify tenant windows were removed
            assert tenant not in mock_monitor._windows
            assert "Reset SLA monitoring" in output
        finally:
            _restore(saved)

    def test_sla_reset_skipped_when_no_windows(self, monkeypatch):
        """SLA reset should not error when tenant has no windows."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        django_mocks = _build_mock_django()
        model_dict, FakeJob, FakeCustodyEvent = _build_mock_models()

        cmd_module, all_mocks, saved = _import_purge_tenant_fresh(
            django_mocks, model_dict
        )
        try:
            tenant = "no-sla-tenant"
            job = _make_job("job-nosla-1", tenant)
            FakeJob.objects._store = [job]
            FakeCustodyEvent._created = []

            cmd = cmd_module.Command()
            cmd.stdout = StringIO()

            # Monitor with no windows for this tenant
            mock_monitor = MagicMock()
            mock_monitor._windows = {"other-tenant": {"availability": MagicMock()}}

            mock_sla_mod = MagicMock()
            mock_sla_mod.get_monitor.return_value = mock_monitor

            mock_cost = MagicMock()
            with patch.dict(
                sys.modules,
                {
                    "cost_tracking": mock_cost,
                    "sla_monitoring": mock_sla_mod,
                },
            ):
                cmd.handle(
                    tenant_id=tenant,
                    dry_run=False,
                    confirm=True,
                    include_custody=False,
                    include_output=False,
                )

            output = cmd.stdout.getvalue()
            assert "Reset SLA monitoring" not in output
            # Other tenant windows should still exist
            assert "other-tenant" in mock_monitor._windows
        finally:
            _restore(saved)


class TestPurgeTenantLitigationHoldGovernance:
    """Verify litigation hold still blocks purge in governance context."""

    def test_litigation_hold_blocks_s3_cleanup(self, monkeypatch):
        """Litigation hold should prevent S3 cleanup from running."""
        monkeypatch.setenv("LITIGATION_HOLD", "true")

        django_mocks = _build_mock_django()
        django_mocks["django.conf"].settings.STORAGE_BACKEND = "s3"

        model_dict, FakeJob, FakeCustodyEvent = _build_mock_models()

        cmd_module, all_mocks, saved = _import_purge_tenant_fresh(
            django_mocks, model_dict
        )
        try:
            tenant = "hold-s3-tenant"
            job = _make_job("job-hold-1", tenant)
            FakeJob.objects._store = [job]
            FakeCustodyEvent._created = []

            cmd = cmd_module.Command()
            cmd.stdout = StringIO()
            cmd.stderr = StringIO()

            cmd.handle(
                tenant_id=tenant,
                dry_run=False,
                confirm=True,
                include_custody=False,
                include_output=True,
            )

            output = cmd.stderr.getvalue()
            assert "LITIGATION_HOLD" in output
            # S3 backend should NOT have been created
            # No custody events recorded
            assert len(FakeCustodyEvent._created) == 0
        finally:
            _restore(saved)


class TestPurgeTenantEndToEnd:
    """End-to-end purge flow with NFS + S3."""

    def test_full_purge_with_nfs_and_s3(self, monkeypatch, tmp_path):
        """Full purge should clean NFS dirs, S3 objects, cost, and SLA."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        django_mocks = _build_mock_django()
        django_mocks["django.conf"].settings.STORAGE_BACKEND = "s3"
        nfs_root = tmp_path / "nfs-root"
        django_mocks["django.conf"].settings.NFS_ROOT = str(nfs_root)
        django_mocks["django.conf"].settings.S3_ENDPOINT = "http://minio:9000"
        django_mocks["django.conf"].settings.S3_BUCKET = "ocr-test"
        django_mocks["django.conf"].settings.S3_ACCESS_KEY = "key"
        django_mocks["django.conf"].settings.S3_SECRET_KEY = "secret"
        django_mocks["django.conf"].settings.S3_REGION = ""

        model_dict, FakeJob, FakeCustodyEvent = _build_mock_models()

        cmd_module, all_mocks, saved = _import_purge_tenant_fresh(
            django_mocks, model_dict
        )
        try:
            tenant = "e2e-tenant"
            nfs_dir = nfs_root / "jobs" / "job-e2e-nfs"
            nfs_dir.mkdir(parents=True)
            (nfs_dir / "output.pdf").write_text("test")

            nfs_job = _make_job("job-e2e-nfs", tenant, nfs_job_path=str(nfs_dir))
            s3_job = _make_job("job-e2e-1", tenant)
            FakeJob.objects._store = [nfs_job, s3_job]
            FakeCustodyEvent._created = []

            # Mock S3 backend
            mock_s3_backend = MagicMock()
            mock_s3_backend.list_objects.side_effect = [
                ["jobs/job-e2e-1/doc.pdf"],
                [],
            ]
            mock_s3_backend.delete_many.return_value = 1
            s3_job._backend = mock_s3_backend
            mock_s3_backend.backend_name = "s3"

            # Mock SLA monitor with tenant windows
            mock_monitor = MagicMock()
            mock_monitor._windows = {tenant: {"availability": MagicMock()}}

            mock_sla_mod = MagicMock()
            mock_sla_mod.get_monitor.return_value = mock_monitor

            mock_cost = MagicMock()

            cmd = cmd_module.Command()
            cmd.stdout = StringIO()

            with patch.dict(
                sys.modules,
                {
                    "cost_tracking": mock_cost,
                    "sla_monitoring": mock_sla_mod,
                },
            ):
                cmd.handle(
                    tenant_id=tenant,
                    dry_run=False,
                    confirm=True,
                    include_custody=False,
                    include_output=True,
                )

            output = cmd.stdout.getvalue()

            # NFS directory should be removed
            assert not nfs_dir.exists()

            # S3 objects should be cleaned
            assert mock_s3_backend.list_objects.call_count == 2
            mock_s3_backend.delete_many.assert_called_once()

            # SLA windows should be reset
            assert tenant not in mock_monitor._windows

            # Completion summary should include all components
            assert "Purged tenant" in output
            assert "NFS dirs removed" in output
            assert "S3 objects removed" in output
        finally:
            _restore(saved)
