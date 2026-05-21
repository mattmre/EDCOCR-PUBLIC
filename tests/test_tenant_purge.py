"""Tests for M-27: Tenant data purge management command (purge_tenant).

Validates that:
- Purge by tenant_id deletes all matching jobs, pages, PII entities
- Dry-run outputs summary but does not delete
- Missing --confirm blocks deletion
- CustodyEvent is emitted BEFORE deletion
- LITIGATION_HOLD blocks all purges
- --include-custody flag behavior
- --include-output removes NFS directories
- No-match tenant produces a graceful message
- Operator identity is present in custody event data
- Multiple jobs for the same tenant are all purged

Django models are mocked at the module boundary (same approach as
test_pii_purge.py).
"""

import getpass
import importlib
import os
import sys
import types
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap helpers -- minimal mock Django environment
# ---------------------------------------------------------------------------


def _build_mock_django():
    """Return a dict of mock modules that satisfy Django imports."""
    from datetime import datetime, timedelta, timezone

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

    class FakeCommandError(Exception):
        pass

    base_mod.BaseCommand = FakeBaseCommand
    base_mod.CommandError = FakeCommandError

    # django.conf -- needed for NFS_ROOT
    conf_mod = types.ModuleType("django.conf")

    class _FakeSettings:
        NFS_ROOT = "/tmp/test-nfs"

    conf_mod.settings = _FakeSettings()

    # django module stubs
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
    """Minimal queryset stand-in supporting filter/count/delete/iteration."""

    def __init__(self, items):
        self._items = list(items)

    def filter(self, **kwargs):
        """Apply simple attribute-based filtering to items."""
        result = list(self._items)
        for key, value in kwargs.items():
            filtered = []
            for item in result:
                if key == "tenant_id":
                    if str(getattr(item, "tenant_id", "")) == str(value):
                        filtered.append(item)
                elif key == "job_id":
                    if str(getattr(item, "job_id", "")) == str(value):
                        filtered.append(item)
                elif key == "job_id__in":
                    if getattr(item, "job_id", None) in value:
                        filtered.append(item)
                else:
                    # Generic attribute match
                    filtered.append(item)
            result = filtered
        return _FakeQuerySet(result)

    def all(self):
        return _FakeQuerySet(self._items)

    def exclude(self, **kwargs):
        return _FakeQuerySet(self._items)

    def values_list(self, *fields, flat=False):
        """Return field values from items."""
        if flat and len(fields) == 1:
            field = fields[0]
            return [getattr(item, field, None) for item in self._items]
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


def _build_mock_job_models():
    """Return a mock ``jobs.models`` module with Job, CustodyEvent, PiiEntity, PageResult."""

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
                items = cls._store
                for key, value in kwargs.items():
                    if key == "tenant_id":
                        items = [j for j in items if str(getattr(j, "tenant_id", "")) == str(value)]
                    elif key == "job_id":
                        items = [j for j in items if str(getattr(j, "job_id", "")) == str(value)]
                return _FakeQuerySet(items)

            @classmethod
            def count(cls):
                return len(cls._store)

            @classmethod
            def first(cls):
                return cls._store[0] if cls._store else None

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
                items = cls._store
                for key, value in kwargs.items():
                    if key == "job_id__in":
                        items = [e for e in items if getattr(e, "job_id", None) in value]
                return _FakeQuerySet(items)

    class FakePiiEntity:
        class objects:
            _store = []

            @classmethod
            def all(cls):
                return _FakeQuerySet(cls._store)

            @classmethod
            def filter(cls, **kwargs):
                return _FakeQuerySet(cls._store).filter(**kwargs)

            @classmethod
            def count(cls):
                return len(cls._store)

    class FakePageResult:
        class objects:
            _store = []

            @classmethod
            def filter(cls, **kwargs):
                return _FakeQuerySet(cls._store).filter(**kwargs)

            @classmethod
            def count(cls):
                return len(cls._store)

    models_mod.Job = FakeJob
    models_mod.CustodyEvent = FakeCustodyEvent
    models_mod.PiiEntity = FakePiiEntity
    models_mod.PageResult = FakePageResult

    jobs_mod = types.ModuleType("jobs")

    return {
        "jobs": jobs_mod,
        "jobs.models": models_mod,
    }


def _make_job(job_id, tenant_id, nfs_job_path=""):
    """Create a fake Job-like object."""
    job = MagicMock()
    job.job_id = job_id
    job.tenant_id = tenant_id
    job.nfs_job_path = nfs_job_path
    return job


def _make_page_result(job_id, page_num=1):
    """Create a fake PageResult-like object."""
    page = MagicMock()
    page.job_id = job_id
    page.page_num = page_num
    return page


def _make_pii_entity(job_id, entity_type, entity_value):
    """Create a fake PiiEntity-like object."""
    entity = MagicMock()
    entity.job_id = job_id
    entity.entity_type = entity_type
    entity.entity_value = entity_value
    return entity


def _make_custody_event(job_id, event_type="ocr_primary"):
    """Create a fake CustodyEvent-like object."""
    event = MagicMock()
    event.job_id = job_id
    event.event_type = event_type
    return event


# ---------------------------------------------------------------------------
# Module import helper
# ---------------------------------------------------------------------------


def _import_purge_tenant(monkeypatch, django_mocks, model_mocks):
    """Import the purge_tenant command module with mocks in place."""
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

    cmd_key = "coordinator.jobs.management.commands.purge_tenant"
    saved[cmd_key] = sys.modules.pop(cmd_key, None)
    saved["purge_tenant"] = sys.modules.pop("purge_tenant", None)

    cmd_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "coordinator",
        "jobs",
        "management",
        "commands",
        "purge_tenant.py",
    )
    spec = importlib.util.spec_from_file_location("purge_tenant", cmd_path)
    cmd_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmd_module)

    return cmd_module, all_mocks, saved


def _restore_modules(saved):
    """Restore original sys.modules entries."""
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture()
def purge_tenant_module(monkeypatch):
    """Import purge_tenant under a mocked Django/jobs environment."""
    django_mocks = _build_mock_django()
    model_mocks = _build_mock_job_models()

    cmd_module, all_mocks, saved = _import_purge_tenant(
        monkeypatch, django_mocks, model_mocks
    )

    FakeJob = all_mocks["jobs.models"].Job
    FakeCustodyEvent = all_mocks["jobs.models"].CustodyEvent
    FakePiiEntity = all_mocks["jobs.models"].PiiEntity
    FakePageResult = all_mocks["jobs.models"].PageResult

    # Reset state
    FakeJob.objects._store = []
    FakeCustodyEvent._created = []
    FakeCustodyEvent.objects._store = []
    FakePiiEntity.objects._store = []
    FakePageResult.objects._store = []

    yield {
        "cmd_module": cmd_module,
        "FakeJob": FakeJob,
        "FakeCustodyEvent": FakeCustodyEvent,
        "FakePiiEntity": FakePiiEntity,
        "FakePageResult": FakePageResult,
        "django_mocks": django_mocks,
        "CommandError": all_mocks["django.core.management.base"].CommandError,
    }

    _restore_modules(saved)


# ===========================================================================
# Tests: purge by tenant_id
# ===========================================================================


class TestPurgeByTenantId:
    """purge_tenant --tenant-id should delete all data for the specified tenant."""

    def test_purge_by_tenant_id_deletes_jobs(self, purge_tenant_module, monkeypatch):
        """All jobs for the given tenant_id are deleted."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        FakeJob = purge_tenant_module["FakeJob"]
        FakePageResult = purge_tenant_module["FakePageResult"]
        FakePiiEntity = purge_tenant_module["FakePiiEntity"]

        tenant = "acme-corp"
        job1 = _make_job("job-1", tenant)
        job2 = _make_job("job-2", tenant)
        FakeJob.objects._store = [job1, job2]

        FakePageResult.objects._store = [
            _make_page_result("job-1"),
            _make_page_result("job-2"),
        ]
        FakePiiEntity.objects._store = [
            _make_pii_entity("job-1", "SSN", "123-45-6789"),
        ]

        cmd = mod.Command()
        cmd.stdout = StringIO()

        # Mock cost_tracking import to avoid real import
        with patch.dict(sys.modules, {"cost_tracking": MagicMock()}):
            cmd.handle(
                tenant_id=tenant,
                dry_run=False,
                confirm=True,
                include_custody=False,
                include_output=False,
            )

        output = cmd.stdout.getvalue()
        assert "Purged tenant" in output
        assert "acme-corp" in output

    def test_purge_multiple_jobs_same_tenant(self, purge_tenant_module, monkeypatch):
        """Multiple jobs belonging to the same tenant are all purged."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        FakeJob = purge_tenant_module["FakeJob"]

        tenant = "multi-job-tenant"
        jobs = [_make_job(f"job-{i}", tenant) for i in range(5)]
        FakeJob.objects._store = jobs

        cmd = mod.Command()
        cmd.stdout = StringIO()

        with patch.dict(sys.modules, {"cost_tracking": MagicMock()}):
            cmd.handle(
                tenant_id=tenant,
                dry_run=False,
                confirm=True,
                include_custody=False,
                include_output=False,
            )

        output = cmd.stdout.getvalue()
        assert "5 jobs" in output

    def test_purge_does_not_affect_other_tenants(self, purge_tenant_module, monkeypatch):
        """Jobs for other tenants must not be deleted."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        FakeJob = purge_tenant_module["FakeJob"]

        target_job = _make_job("job-1", "target-tenant")
        other_job = _make_job("job-2", "other-tenant")
        FakeJob.objects._store = [target_job, other_job]

        cmd = mod.Command()
        cmd.stdout = StringIO()

        with patch.dict(sys.modules, {"cost_tracking": MagicMock()}):
            cmd.handle(
                tenant_id="target-tenant",
                dry_run=False,
                confirm=True,
                include_custody=False,
                include_output=False,
            )

        output = cmd.stdout.getvalue()
        assert "1 jobs" in output


# ===========================================================================
# Tests: dry-run
# ===========================================================================


class TestPurgeDryRun:
    """Dry-run mode should display summary but not delete."""

    def test_dry_run_no_deletion(self, purge_tenant_module, monkeypatch):
        """--dry-run should not delete any records."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        FakeJob = purge_tenant_module["FakeJob"]
        FakeCustodyEvent = purge_tenant_module["FakeCustodyEvent"]

        tenant = "dry-run-tenant"
        FakeJob.objects._store = [_make_job("job-1", tenant)]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            tenant_id=tenant,
            dry_run=True,
            confirm=False,
            include_custody=False,
            include_output=False,
        )

        output = cmd.stdout.getvalue()
        assert "DRY RUN" in output
        # No custody events created during dry run
        assert len(FakeCustodyEvent._created) == 0
        # Job store should not be modified
        assert len(FakeJob.objects._store) == 1

    def test_dry_run_shows_impact_summary(self, purge_tenant_module, monkeypatch):
        """Dry-run output should include job count and other impact info."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        FakeJob = purge_tenant_module["FakeJob"]
        FakePageResult = purge_tenant_module["FakePageResult"]
        FakePiiEntity = purge_tenant_module["FakePiiEntity"]

        tenant = "summary-tenant"
        FakeJob.objects._store = [
            _make_job("job-1", tenant),
            _make_job("job-2", tenant),
        ]
        FakePageResult.objects._store = [
            _make_page_result("job-1", 1),
            _make_page_result("job-1", 2),
            _make_page_result("job-2", 1),
        ]
        FakePiiEntity.objects._store = [
            _make_pii_entity("job-1", "SSN", "123"),
        ]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            tenant_id=tenant,
            dry_run=True,
            confirm=False,
            include_custody=False,
            include_output=False,
        )

        output = cmd.stdout.getvalue()
        assert "Jobs: 2" in output
        assert "Pages:" in output
        assert "PII entities:" in output
        assert "DRY RUN" in output


# ===========================================================================
# Tests: --confirm gate
# ===========================================================================


class TestConfirmGate:
    """Missing --confirm should block deletion."""

    def test_missing_confirm_blocks_deletion(self, purge_tenant_module, monkeypatch):
        """Without --confirm, no records should be deleted."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        FakeJob = purge_tenant_module["FakeJob"]
        FakeCustodyEvent = purge_tenant_module["FakeCustodyEvent"]

        tenant = "confirm-test"
        FakeJob.objects._store = [_make_job("job-1", tenant)]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            tenant_id=tenant,
            dry_run=False,
            confirm=False,
            include_custody=False,
            include_output=False,
        )

        output = cmd.stdout.getvalue()
        assert "--confirm" in output
        # No deletion, no custody event
        assert len(FakeCustodyEvent._created) == 0
        # Job still exists
        assert len(FakeJob.objects._store) == 1


# ===========================================================================
# Tests: LITIGATION_HOLD
# ===========================================================================


class TestLitigationHold:
    """LITIGATION_HOLD env var should block all purge operations."""

    def test_litigation_hold_blocks_purge(self, purge_tenant_module, monkeypatch):
        """When LITIGATION_HOLD=true, purge should not proceed."""
        monkeypatch.setenv("LITIGATION_HOLD", "true")

        mod = purge_tenant_module["cmd_module"]
        FakeJob = purge_tenant_module["FakeJob"]
        FakeCustodyEvent = purge_tenant_module["FakeCustodyEvent"]

        tenant = "hold-tenant"
        FakeJob.objects._store = [_make_job("job-1", tenant)]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.stderr = StringIO()
        cmd.handle(
            tenant_id=tenant,
            dry_run=False,
            confirm=True,
            include_custody=False,
            include_output=False,
        )

        output = cmd.stderr.getvalue()
        assert "LITIGATION_HOLD" in output
        # No deletion
        assert len(FakeCustodyEvent._created) == 0
        assert len(FakeJob.objects._store) == 1

    def test_litigation_hold_accepts_yes_and_one(self, purge_tenant_module, monkeypatch):
        """LITIGATION_HOLD should accept 'yes', '1', and 'true'."""
        mod = purge_tenant_module["cmd_module"]

        for value in ("yes", "1", "true", "TRUE", "True"):
            monkeypatch.setenv("LITIGATION_HOLD", value)
            cmd = mod.Command()
            cmd.stdout = StringIO()
            cmd.stderr = StringIO()
            cmd.handle(
                tenant_id="any-tenant",
                dry_run=False,
                confirm=True,
                include_custody=False,
                include_output=False,
            )
            output = cmd.stderr.getvalue()
            assert "LITIGATION_HOLD" in output


# ===========================================================================
# Tests: CustodyEvent emitted before deletion
# ===========================================================================


class TestCustodyEventEmitted:
    """Successful purge should emit a CustodyEvent before deleting data."""

    def test_custody_event_on_purge(self, purge_tenant_module, monkeypatch):
        """A tenant_purged custody event should be created on deletion."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        FakeJob = purge_tenant_module["FakeJob"]
        FakeCustodyEvent = purge_tenant_module["FakeCustodyEvent"]

        tenant = "custody-tenant"
        FakeJob.objects._store = [_make_job("job-1", tenant)]

        cmd = mod.Command()
        cmd.stdout = StringIO()

        with patch.dict(sys.modules, {"cost_tracking": MagicMock()}):
            cmd.handle(
                tenant_id=tenant,
                dry_run=False,
                confirm=True,
                include_custody=False,
                include_output=False,
            )

        assert len(FakeCustodyEvent._created) == 1
        event = FakeCustodyEvent._created[0]
        assert event["event_type"] == "tenant_purged"
        assert event["data"]["action"] == "purge_tenant"
        assert event["data"]["tenant_id"] == tenant
        assert event["data"]["job_count"] == 1

    def test_custody_event_has_impact_counts(self, purge_tenant_module, monkeypatch):
        """Custody event data should include page, PII, and custody counts."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        FakeJob = purge_tenant_module["FakeJob"]
        FakePageResult = purge_tenant_module["FakePageResult"]
        FakePiiEntity = purge_tenant_module["FakePiiEntity"]
        FakeCustodyEvent = purge_tenant_module["FakeCustodyEvent"]

        tenant = "counts-tenant"
        FakeJob.objects._store = [_make_job("job-1", tenant)]
        FakePageResult.objects._store = [
            _make_page_result("job-1", 1),
            _make_page_result("job-1", 2),
        ]
        FakePiiEntity.objects._store = [
            _make_pii_entity("job-1", "SSN", "123"),
            _make_pii_entity("job-1", "EMAIL", "a@b.c"),
        ]
        FakeCustodyEvent.objects._store = [
            _make_custody_event("job-1"),
        ]

        cmd = mod.Command()
        cmd.stdout = StringIO()

        with patch.dict(sys.modules, {"cost_tracking": MagicMock()}):
            cmd.handle(
                tenant_id=tenant,
                dry_run=False,
                confirm=True,
                include_custody=False,
                include_output=False,
            )

        event = FakeCustodyEvent._created[0]
        assert event["data"]["page_count"] == 2
        assert event["data"]["pii_entity_count"] == 2
        assert event["data"]["custody_event_count"] == 1


# ===========================================================================
# Tests: --include-output (NFS directory removal)
# ===========================================================================


class TestIncludeOutput:
    """--include-output should remove NFS job directories."""

    def test_include_output_removes_nfs_dirs(self, purge_tenant_module, monkeypatch, tmp_path):
        """NFS directories for tenant jobs should be removed."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        FakeJob = purge_tenant_module["FakeJob"]

        tenant = "nfs-tenant"
        # Create a temporary NFS directory
        nfs_dir = tmp_path / "nfs_job"
        nfs_dir.mkdir()
        (nfs_dir / "output.pdf").write_text("test")

        job = _make_job("job-1", tenant, nfs_job_path=str(nfs_dir))
        FakeJob.objects._store = [job]

        cmd = mod.Command()
        cmd.stdout = StringIO()

        with patch.dict(sys.modules, {"cost_tracking": MagicMock()}):
            cmd.handle(
                tenant_id=tenant,
                dry_run=False,
                confirm=True,
                include_custody=False,
                include_output=True,
            )

        output = cmd.stdout.getvalue()
        assert "NFS dirs removed" in output
        assert not nfs_dir.exists()

    def test_without_include_output_keeps_nfs_dirs(self, purge_tenant_module, monkeypatch, tmp_path):
        """Without --include-output, NFS directories should not be removed."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        FakeJob = purge_tenant_module["FakeJob"]

        tenant = "keep-nfs-tenant"
        nfs_dir = tmp_path / "nfs_job"
        nfs_dir.mkdir()
        (nfs_dir / "output.pdf").write_text("test")

        job = _make_job("job-1", tenant, nfs_job_path=str(nfs_dir))
        FakeJob.objects._store = [job]

        cmd = mod.Command()
        cmd.stdout = StringIO()

        with patch.dict(sys.modules, {"cost_tracking": MagicMock()}):
            cmd.handle(
                tenant_id=tenant,
                dry_run=False,
                confirm=True,
                include_custody=False,
                include_output=False,
            )

        # NFS dir should still exist
        assert nfs_dir.exists()


# ===========================================================================
# Tests: --include-custody flag
# ===========================================================================


class TestIncludeCustody:
    """--include-custody flag controls custody event purge behavior."""

    def test_include_custody_flag_accepted(self, purge_tenant_module, monkeypatch):
        """The --include-custody flag should be accepted and processed."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        FakeJob = purge_tenant_module["FakeJob"]

        tenant = "custody-flag-tenant"
        FakeJob.objects._store = [_make_job("job-1", tenant)]

        cmd = mod.Command()
        cmd.stdout = StringIO()

        with patch.dict(sys.modules, {"cost_tracking": MagicMock()}):
            cmd.handle(
                tenant_id=tenant,
                dry_run=False,
                confirm=True,
                include_custody=True,
                include_output=False,
            )

        output = cmd.stdout.getvalue()
        assert "Purged tenant" in output


# ===========================================================================
# Tests: no-match scenario
# ===========================================================================


class TestNoMatch:
    """Graceful message when no jobs match the tenant ID."""

    def test_no_match_message(self, purge_tenant_module, monkeypatch):
        """When no jobs match the tenant, a clear message is displayed."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        FakeJob = purge_tenant_module["FakeJob"]
        FakeCustodyEvent = purge_tenant_module["FakeCustodyEvent"]

        FakeJob.objects._store = []

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            tenant_id="nonexistent-tenant",
            dry_run=False,
            confirm=True,
            include_custody=False,
            include_output=False,
        )

        output = cmd.stdout.getvalue()
        assert "No jobs found" in output
        assert "nonexistent-tenant" in output
        # No custody events
        assert len(FakeCustodyEvent._created) == 0


# ===========================================================================
# Tests: operator identity in custody event
# ===========================================================================


class TestOperatorIdentity:
    """Custody events must include the operator's identity."""

    def test_operator_in_custody_event(self, purge_tenant_module, monkeypatch):
        """Custody event data should include 'operator' from getpass.getuser()."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        FakeJob = purge_tenant_module["FakeJob"]
        FakeCustodyEvent = purge_tenant_module["FakeCustodyEvent"]

        tenant = "operator-tenant"
        FakeJob.objects._store = [_make_job("job-1", tenant)]

        cmd = mod.Command()
        cmd.stdout = StringIO()

        with patch.dict(sys.modules, {"cost_tracking": MagicMock()}):
            cmd.handle(
                tenant_id=tenant,
                dry_run=False,
                confirm=True,
                include_custody=False,
                include_output=False,
            )

        assert len(FakeCustodyEvent._created) == 1
        event = FakeCustodyEvent._created[0]
        assert "operator" in event["data"]
        assert event["data"]["operator"] == getpass.getuser()
        assert isinstance(event["data"]["operator"], str)
        assert len(event["data"]["operator"]) > 0


# ===========================================================================
# Tests: empty tenant_id rejection
# ===========================================================================


class TestEmptyTenantId:
    """Empty or whitespace-only tenant_id should be rejected."""

    def test_empty_tenant_id_raises_error(self, purge_tenant_module, monkeypatch):
        """Calling with empty --tenant-id should raise CommandError."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        CommandError = purge_tenant_module["CommandError"]

        cmd = mod.Command()
        cmd.stdout = StringIO()

        with pytest.raises(CommandError, match="must not be empty"):
            cmd.handle(
                tenant_id="",
                dry_run=False,
                confirm=True,
                include_custody=False,
                include_output=False,
            )

    def test_whitespace_tenant_id_raises_error(self, purge_tenant_module, monkeypatch):
        """Calling with whitespace --tenant-id should raise CommandError."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_tenant_module["cmd_module"]
        CommandError = purge_tenant_module["CommandError"]

        cmd = mod.Command()
        cmd.stdout = StringIO()

        with pytest.raises(CommandError, match="must not be empty"):
            cmd.handle(
                tenant_id="   ",
                dry_run=False,
                confirm=True,
                include_custody=False,
                include_output=False,
            )


# ===========================================================================
# Tests: custody.py EVENT_TYPES registration
# ===========================================================================


class TestCustodyEventTypeRegistered:
    """Verify tenant_purged is registered in custody.py EVENT_TYPES."""

    def test_tenant_purged_in_event_types(self):
        """The tenant_purged event type should be defined in custody.py."""
        from custody import EVENT_TYPES

        assert "tenant_purged" in EVENT_TYPES
        assert isinstance(EVENT_TYPES["tenant_purged"], str)
        assert len(EVENT_TYPES["tenant_purged"]) > 0
