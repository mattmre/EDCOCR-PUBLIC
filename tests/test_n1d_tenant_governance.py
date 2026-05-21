"""Tests for Wave N1-D: Tenant governance closure.

Validates:
- S3 cleanup in coordinator purge_tenant command
- SLA window reset in coordinator purge_tenant command
- Source-of-truth docstring markers in CostTracker and SLAMonitor
- Purge command respects litigation hold (verified in governance context)
- End-to-end purge flow with both NFS and S3 configured
- Architecture doc exists and contains required sections

Since Django models cannot be fully loaded in root test context, these tests
use a combination of:
- Source inspection (verify docstrings and code content)
- Module imports for non-Django modules (cost_tracking, sla_monitoring)
- File content assertions for coordinator command (Django-dependent)
"""

import importlib
import inspect
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
_ARCH_DOC_PATH = (
    _PROJECT_ROOT / "docs" / "architecture" / "n1d-tenant-governance.md"
)


# ===========================================================================
# Source-of-truth marker tests
# ===========================================================================


class TestCostTrackerSourceOfTruth:
    """CostTracker class must have source-of-truth documentation."""

    def test_cost_tracker_docstring_contains_source_of_truth_note(self):
        """CostTracker class docstring includes authoritative cost source note."""
        import cost_tracking

        src = inspect.getsource(cost_tracking.CostTracker)
        src_lower = src.lower()
        assert "authoritative cost" in src_lower or "source of truth" in src_lower

    def test_cost_tracker_docstring_references_api_database(self):
        """CostTracker docstring references api/database.py as the authoritative source."""
        import cost_tracking

        src = inspect.getsource(cost_tracking.CostTracker)
        assert "api/database.py" in src

    def test_cost_tracker_docstring_notes_process_restart_risk(self):
        """CostTracker docstring warns about data loss on process restart."""
        import cost_tracking

        src = inspect.getsource(cost_tracking.CostTracker)
        assert "restart" in src.lower()

    def test_cost_tracker_is_cache_not_truth(self):
        """CostTracker docstring identifies itself as a cache, not the truth source."""
        import cost_tracking

        src = inspect.getsource(cost_tracking.CostTracker)
        assert "cache" in src.lower()


class TestSLAMonitorSourceOfTruth:
    """SLAMonitor class must have source-of-truth documentation."""

    def test_sla_monitor_docstring_contains_source_of_truth_note(self):
        """SLAMonitor class docstring includes authoritative SLA source note."""
        import sla_monitoring

        src = inspect.getsource(sla_monitoring.SLAMonitor)
        src_lower = src.lower()
        assert "authoritative" in src_lower or "source of truth" in src_lower

    def test_sla_monitor_docstring_references_api_slo(self):
        """SLAMonitor docstring references api/slo.py as the authoritative source."""
        import sla_monitoring

        src = inspect.getsource(sla_monitoring.SLAMonitor)
        assert "api/slo.py" in src

    def test_sla_monitor_docstring_notes_process_restart_risk(self):
        """SLAMonitor docstring warns about data loss on process restart."""
        import sla_monitoring

        src = inspect.getsource(sla_monitoring.SLAMonitor)
        assert "restart" in src.lower()

    def test_sla_monitor_is_cache_not_truth(self):
        """SLAMonitor docstring identifies itself as a cache, not the truth source."""
        import sla_monitoring

        src = inspect.getsource(sla_monitoring.SLAMonitor)
        assert "cache" in src.lower()


# ===========================================================================
# Purge tenant command source-level tests
# ===========================================================================


class TestPurgeTenantCommandSource:
    """Verify purge_tenant management command includes S3 and SLA cleanup code."""

    def test_purge_tenant_file_exists(self):
        """The purge_tenant management command file exists."""
        assert _PURGE_TENANT_PATH.exists(), f"Missing: {_PURGE_TENANT_PATH}"

    def test_purge_tenant_includes_s3_cleanup_code(self):
        """purge_tenant management command includes S3 cleanup logic."""
        src = _PURGE_TENANT_PATH.read_text(encoding="utf-8")
        # Must reference S3 storage backend
        assert "s3" in src.lower()
        assert "create_storage_backend" in src

    def test_purge_tenant_includes_s3_list_and_delete(self):
        """purge_tenant S3 cleanup uses list_objects and delete_many."""
        src = _PURGE_TENANT_PATH.read_text(encoding="utf-8")
        assert "list_objects" in src
        assert "delete_many" in src

    def test_purge_tenant_includes_sla_reset(self):
        """purge_tenant management command includes SLA monitoring reset."""
        src = _PURGE_TENANT_PATH.read_text(encoding="utf-8")
        assert "sla_monitoring" in src

    def test_purge_tenant_includes_sla_window_deletion(self):
        """purge_tenant SLA reset deletes tenant windows from _windows dict."""
        src = _PURGE_TENANT_PATH.read_text(encoding="utf-8")
        assert "_windows" in src
        assert "get_monitor" in src

    def test_purge_tenant_s3_cleanup_is_conditional(self):
        """S3 cleanup only runs when STORAGE_BACKEND is s3."""
        src = _PURGE_TENANT_PATH.read_text(encoding="utf-8")
        assert "STORAGE_BACKEND" in src

    def test_purge_tenant_s3_cleanup_handles_errors_gracefully(self):
        """S3 cleanup catches exceptions so purge is not blocked."""
        src = _PURGE_TENANT_PATH.read_text(encoding="utf-8")
        # Should have try/except around S3 cleanup
        assert "S3 cleanup skipped" in src or "S3 backend not available" in src

    def test_purge_tenant_summary_includes_s3_count(self):
        """Purge completion summary reports S3 objects removed."""
        src = _PURGE_TENANT_PATH.read_text(encoding="utf-8")
        assert "S3 objects removed" in src


# ===========================================================================
# Architecture document tests
# ===========================================================================


class TestArchitectureDocument:
    """Verify the tenant governance architecture document exists and is complete."""

    def test_architecture_doc_exists(self):
        """The n1d-tenant-governance.md architecture doc exists."""
        assert _ARCH_DOC_PATH.exists(), f"Missing: {_ARCH_DOC_PATH}"

    def test_doc_designates_cost_source_of_truth(self):
        """Architecture doc designates SQLite UsageRecord as cost source of truth."""
        content = _ARCH_DOC_PATH.read_text(encoding="utf-8")
        assert "UsageRecord" in content
        assert "api/database.py" in content

    def test_doc_designates_sla_source_of_truth(self):
        """Architecture doc designates api/slo.py Job-table as SLA source of truth."""
        content = _ARCH_DOC_PATH.read_text(encoding="utf-8")
        assert "api/slo.py" in content

    def test_doc_documents_dual_path_purge(self):
        """Architecture doc documents the dual-path purge design."""
        content = _ARCH_DOC_PATH.read_text(encoding="utf-8")
        assert "Dual-Path Purge" in content or "dual-path purge" in content.lower()

    def test_doc_mentions_coordinator_purge(self):
        """Architecture doc covers coordinator purge scope."""
        content = _ARCH_DOC_PATH.read_text(encoding="utf-8")
        assert "PostgreSQL" in content
        assert "NFS" in content
        assert "S3" in content

    def test_doc_mentions_api_purge(self):
        """Architecture doc covers API purge scope."""
        content = _ARCH_DOC_PATH.read_text(encoding="utf-8")
        assert "SQLite" in content

    def test_doc_has_known_limitations(self):
        """Architecture doc includes a Known Limitations section."""
        content = _ARCH_DOC_PATH.read_text(encoding="utf-8")
        assert "Known Limitations" in content

    def test_doc_mentions_prometheus_registry_isolation(self):
        """Architecture doc notes isolated Prometheus registries."""
        content = _ARCH_DOC_PATH.read_text(encoding="utf-8")
        assert "Prometheus" in content
        assert "registr" in content.lower()

    def test_doc_mentions_rate_limit_cache(self):
        """Architecture doc notes process-local rate limit cache."""
        content = _ARCH_DOC_PATH.read_text(encoding="utf-8")
        assert "rate limit" in content.lower()


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
        if hasattr(monitor, '_windows') and "nonexistent" in monitor._windows:
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
        NFS_ROOT = "/tmp/test-nfs"
        STORAGE_BACKEND = "nfs"

    conf_mod.settings = _FakeSettings()

    django_mod = types.ModuleType("django")
    django_utils = types.ModuleType("django.utils")
    django_core = types.ModuleType("django.core")
    django_core_management = types.ModuleType("django.core.management")

    return {
        "django": django_mod,
        "django.conf": conf_mod,
        "django.utils": django_utils,
        "django.utils.timezone": tz_mod,
        "django.core": django_core,
        "django.core.management": django_core_management,
        "django.core.management.base": base_mod,
    }


class _FakeQuerySet:
    """Minimal queryset stand-in."""

    def __init__(self, items):
        self._items = list(items)

    def filter(self, **kwargs):
        result = list(self._items)
        for key, value in kwargs.items():
            if key == "tenant_id":
                result = [j for j in result if str(getattr(j, "tenant_id", "")) == str(value)]
            elif key == "job_id__in":
                result = [j for j in result if getattr(j, "job_id", None) in value]
        return _FakeQuerySet(result)

    def values_list(self, *fields, flat=False):
        if flat and len(fields) == 1:
            field_name = fields[0]
            return [getattr(item, field_name, None) for item in self._items]
        return _FakeQuerySet(self._items)

    def count(self):
        return len(self._items)

    def delete(self):
        n = len(self._items)
        self._items.clear()
        return (n, {})

    def first(self):
        return self._items[0] if self._items else None

    def __iter__(self):
        return iter(self._items)


def _build_mock_models():
    """Return mock jobs.models module."""
    models_mod = types.ModuleType("jobs.models")

    class FakeJob:
        class objects:
            _store = []

            @classmethod
            def filter(cls, **kwargs):
                items = cls._store
                for key, value in kwargs.items():
                    if key == "tenant_id":
                        items = [j for j in items if str(getattr(j, "tenant_id", "")) == str(value)]
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
            def filter(cls, **kwargs):
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

    models_mod.Job = FakeJob
    models_mod.CustodyEvent = FakeCustodyEvent
    models_mod.PiiEntity = FakePiiEntity
    models_mod.PageResult = FakePageResult

    jobs_mod = types.ModuleType("jobs")

    return {
        "jobs": jobs_mod,
        "jobs.models": models_mod,
    }, FakeJob, FakeCustodyEvent


def _import_purge_tenant_fresh(django_mocks, model_mocks):
    """Import purge_tenant command module with mocks in place."""
    all_mocks = {**django_mocks, **model_mocks}

    saved = {}
    for name, mod in all_mocks.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    # Inject litigation_hold
    lh_path = str(
        _PROJECT_ROOT / "coordinator" / "jobs" / "litigation_hold.py"
    )
    lh_spec = importlib.util.spec_from_file_location("jobs.litigation_hold", lh_path)
    lh_mod = importlib.util.module_from_spec(lh_spec)
    lh_spec.loader.exec_module(lh_mod)
    saved["jobs.litigation_hold"] = sys.modules.get("jobs.litigation_hold")
    sys.modules["jobs.litigation_hold"] = lh_mod

    # Also mock jobs.storage to avoid real boto3 import
    storage_mock = types.ModuleType("jobs.storage")
    storage_mock.create_storage_backend = MagicMock()
    saved["jobs.storage"] = sys.modules.get("jobs.storage")
    sys.modules["jobs.storage"] = storage_mock

    cmd_key = "coordinator.jobs.management.commands.purge_tenant"
    saved[cmd_key] = sys.modules.pop(cmd_key, None)
    saved["purge_tenant_n1d"] = sys.modules.pop("purge_tenant_n1d", None)

    cmd_path = str(_PURGE_TENANT_PATH)
    spec = importlib.util.spec_from_file_location("purge_tenant_n1d", cmd_path)
    cmd_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmd_module)

    return cmd_module, all_mocks, saved, storage_mock


def _restore(saved):
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def _make_job(job_id, tenant_id, nfs_job_path=""):
    job = MagicMock()
    job.job_id = job_id
    job.tenant_id = tenant_id
    job.nfs_job_path = nfs_job_path
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

        cmd_module, all_mocks, saved, storage_mock = _import_purge_tenant_fresh(
            django_mocks, model_dict
        )
        try:
            tenant = "s3-tenant"
            job = _make_job("job-s3-1", tenant)
            FakeJob.objects._store = [job]
            FakeCustodyEvent._created = []

            # Configure mock S3 backend
            mock_s3_backend = MagicMock()
            mock_s3_backend.list_objects.return_value = [
                "jobs/job-s3-1/output.pdf",
                "jobs/job-s3-1/text.txt",
            ]
            mock_s3_backend.delete_many.return_value = 2
            storage_mock.create_storage_backend.return_value = mock_s3_backend

            cmd = cmd_module.Command()
            cmd.stdout = StringIO()

            mock_cost = MagicMock()
            mock_sla = MagicMock()
            with patch.dict(sys.modules, {
                "cost_tracking": mock_cost,
                "sla_monitoring": mock_sla,
            }):
                cmd.handle(
                    tenant_id=tenant,
                    dry_run=False,
                    confirm=True,
                    include_custody=False,
                    include_output=True,
                )

            output = cmd.stdout.getvalue()
            # S3 backend should have been created
            storage_mock.create_storage_backend.assert_called_once()
            create_kwargs = storage_mock.create_storage_backend.call_args
            assert create_kwargs[1]["backend_name"] == "s3"
            # Should list and delete objects
            mock_s3_backend.list_objects.assert_called_once_with("jobs/job-s3-1/")
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

        cmd_module, all_mocks, saved, storage_mock = _import_purge_tenant_fresh(
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
            with patch.dict(sys.modules, {
                "cost_tracking": mock_cost,
                "sla_monitoring": mock_sla,
            }):
                cmd.handle(
                    tenant_id=tenant,
                    dry_run=False,
                    confirm=True,
                    include_custody=False,
                    include_output=True,
                )

            # S3 backend should NOT be created
            storage_mock.create_storage_backend.assert_not_called()
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

        cmd_module, all_mocks, saved, storage_mock = _import_purge_tenant_fresh(
            django_mocks, model_dict
        )
        try:
            tenant = "empty-s3-tenant"
            job = _make_job("job-empty-1", tenant)
            FakeJob.objects._store = [job]
            FakeCustodyEvent._created = []

            mock_s3_backend = MagicMock()
            mock_s3_backend.list_objects.return_value = []
            storage_mock.create_storage_backend.return_value = mock_s3_backend

            cmd = cmd_module.Command()
            cmd.stdout = StringIO()

            mock_cost = MagicMock()
            mock_sla = MagicMock()
            with patch.dict(sys.modules, {
                "cost_tracking": mock_cost,
                "sla_monitoring": mock_sla,
            }):
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

        cmd_module, all_mocks, saved, storage_mock = _import_purge_tenant_fresh(
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
            with patch.dict(sys.modules, {
                "cost_tracking": mock_cost,
                "sla_monitoring": mock_sla_mod,
            }):
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

        cmd_module, all_mocks, saved, storage_mock = _import_purge_tenant_fresh(
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
            with patch.dict(sys.modules, {
                "cost_tracking": mock_cost,
                "sla_monitoring": mock_sla_mod,
            }):
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

        cmd_module, all_mocks, saved, storage_mock = _import_purge_tenant_fresh(
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
            storage_mock.create_storage_backend.assert_not_called()
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
        django_mocks["django.conf"].settings.S3_ENDPOINT = "http://minio:9000"
        django_mocks["django.conf"].settings.S3_BUCKET = "ocr-test"
        django_mocks["django.conf"].settings.S3_ACCESS_KEY = "key"
        django_mocks["django.conf"].settings.S3_SECRET_KEY = "secret"
        django_mocks["django.conf"].settings.S3_REGION = ""

        model_dict, FakeJob, FakeCustodyEvent = _build_mock_models()

        cmd_module, all_mocks, saved, storage_mock = _import_purge_tenant_fresh(
            django_mocks, model_dict
        )
        try:
            tenant = "e2e-tenant"
            nfs_dir = tmp_path / "nfs_e2e"
            nfs_dir.mkdir()
            (nfs_dir / "output.pdf").write_text("test")

            job = _make_job("job-e2e-1", tenant, nfs_job_path=str(nfs_dir))
            FakeJob.objects._store = [job]
            FakeCustodyEvent._created = []

            # Mock S3 backend
            mock_s3_backend = MagicMock()
            mock_s3_backend.list_objects.return_value = ["jobs/job-e2e-1/doc.pdf"]
            mock_s3_backend.delete_many.return_value = 1
            storage_mock.create_storage_backend.return_value = mock_s3_backend

            # Mock SLA monitor with tenant windows
            mock_monitor = MagicMock()
            mock_monitor._windows = {tenant: {"availability": MagicMock()}}

            mock_sla_mod = MagicMock()
            mock_sla_mod.get_monitor.return_value = mock_monitor

            mock_cost = MagicMock()

            cmd = cmd_module.Command()
            cmd.stdout = StringIO()

            with patch.dict(sys.modules, {
                "cost_tracking": mock_cost,
                "sla_monitoring": mock_sla_mod,
            }):
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
            mock_s3_backend.list_objects.assert_called_once()
            mock_s3_backend.delete_many.assert_called_once()

            # SLA windows should be reset
            assert tenant not in mock_monitor._windows

            # Completion summary should include all components
            assert "Purged tenant" in output
            assert "NFS dirs removed" in output
            assert "S3 objects removed" in output
        finally:
            _restore(saved)
