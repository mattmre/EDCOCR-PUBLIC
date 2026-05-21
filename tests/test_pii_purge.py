"""Tests for C-04: PII/PHI purge management command (purge_pii).

Validates that:
- Purge by job_id deletes matching PiiEntity records
- Purge by subject name matches case-insensitively
- Entity-type filter narrows the purge scope
- Dry-run outputs summary but does not delete
- Missing --confirm blocks deletion
- CustodyEvent is emitted on successful purge
- LITIGATION_HOLD blocks all purges
- No-match scenario produces a graceful message
- Operator identity is present in custody event data

Django models are mocked at the module boundary (same approach as
test_audit_deletion_events.py).
"""

import getpass
import importlib
import os
import sys
import types
from io import StringIO
from unittest.mock import MagicMock

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
                if key == "job_id":
                    if str(getattr(item, "job_id", "")) == str(value):
                        filtered.append(item)
                elif key == "entity_value__icontains":
                    val = getattr(item, "entity_value", "")
                    if value.lower() in val.lower():
                        filtered.append(item)
                elif key == "entity_type":
                    if getattr(item, "entity_type", "") == value:
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

    def __or__(self, other):
        """Support queryset union (qs1 | qs2)."""
        combined = list(self._items)
        if hasattr(other, '_items'):
            combined.extend(other._items)
        return _FakeQuerySet(combined)

    def exists(self):
        return len(self._items) > 0


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
                items = cls._store
                for key, value in kwargs.items():
                    if key == "job_id":
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
            @classmethod
            def create(cls, **kwargs):
                FakeCustodyEvent._created.append(kwargs)
                return MagicMock()

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

    models_mod.Job = FakeJob
    models_mod.CustodyEvent = FakeCustodyEvent
    models_mod.PiiEntity = FakePiiEntity

    jobs_mod = types.ModuleType("jobs")

    # Also mock jobs.redis_streams to avoid real Redis connections
    redis_streams_mod = types.ModuleType("jobs.redis_streams")

    class FakeRedisStreamClient:
        def __init__(self, redis_url=None):
            pass

        def delete_chunk_cache(self, chunk_id):
            return 0

    redis_streams_mod.RedisStreamClient = FakeRedisStreamClient

    # Mock jobs.extraction_models for ExtractedEntity import
    extraction_mod = types.ModuleType("jobs.extraction_models")

    class FakeExtractedEntity:
        class objects:
            @classmethod
            def none(cls):
                return _FakeQuerySet([])

            @classmethod
            def filter(cls, **kwargs):
                return _FakeQuerySet([])

    extraction_mod.ExtractedEntity = FakeExtractedEntity

    return {
        "jobs": jobs_mod,
        "jobs.models": models_mod,
        "jobs.redis_streams": redis_streams_mod,
        "jobs.extraction_models": extraction_mod,
    }


def _make_pii_entity(job_id, entity_type, entity_value):
    """Create a fake PiiEntity-like object."""
    entity = MagicMock()
    entity.job_id = job_id
    entity.entity_type = entity_type
    entity.entity_value = entity_value
    return entity


# ---------------------------------------------------------------------------
# Module import helper
# ---------------------------------------------------------------------------


def _import_purge_pii(monkeypatch, django_mocks, model_mocks):
    """Import the purge_pii command module with mocks in place."""
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

    cmd_key = "coordinator.jobs.management.commands.purge_pii"
    saved[cmd_key] = sys.modules.pop(cmd_key, None)
    saved["purge_pii"] = sys.modules.pop("purge_pii", None)

    cmd_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "coordinator",
        "jobs",
        "management",
        "commands",
        "purge_pii.py",
    )
    spec = importlib.util.spec_from_file_location("purge_pii", cmd_path)
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
def purge_pii_module(monkeypatch):
    """Import purge_pii under a mocked Django/jobs environment."""
    django_mocks = _build_mock_django()
    model_mocks = _build_mock_job_models()

    cmd_module, all_mocks, saved = _import_purge_pii(
        monkeypatch, django_mocks, model_mocks
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
        "django_mocks": django_mocks,
        "CommandError": all_mocks["django.core.management.base"].CommandError,
    }

    _restore_modules(saved)


# ===========================================================================
# Tests: purge by job_id
# ===========================================================================


class TestPurgeByJobId:
    """purge_pii --job-id should delete PII for the specified job."""

    def test_purge_by_job_id_deletes_matching(self, purge_pii_module, monkeypatch):
        """PII records for the given job_id are deleted."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_pii_module["cmd_module"]
        FakeJob = purge_pii_module["FakeJob"]
        FakePiiEntity = purge_pii_module["FakePiiEntity"]

        job_id = "aaaa-bbbb-cccc-dddd"
        anchor_job = MagicMock()
        anchor_job.job_id = job_id
        FakeJob.objects._store = [anchor_job]

        FakePiiEntity.objects._store = [
            _make_pii_entity(job_id, "SSN", "123-45-6789"),
            _make_pii_entity(job_id, "EMAIL", "test@example.com"),
        ]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            job_id=job_id,
            subject=None,
            entity_type=None,
            dry_run=False,
            confirm=True,
        )

        output = cmd.stdout.getvalue()
        assert "Purged 2 PII entities" in output

    def test_purge_by_job_id_no_match(self, purge_pii_module, monkeypatch):
        """No entities found for a non-existent job_id is handled gracefully."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_pii_module["cmd_module"]
        FakePiiEntity = purge_pii_module["FakePiiEntity"]

        FakePiiEntity.objects._store = []

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            job_id="nonexistent-id",
            subject=None,
            entity_type=None,
            dry_run=False,
            confirm=True,
        )

        output = cmd.stdout.getvalue()
        assert "No matching PII entities found" in output


# ===========================================================================
# Tests: purge by subject name (case-insensitive)
# ===========================================================================


class TestPurgeBySubject:
    """purge_pii --subject should match entity_value case-insensitively."""

    def test_purge_by_subject_case_insensitive(self, purge_pii_module, monkeypatch):
        """Subject matching is case-insensitive."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_pii_module["cmd_module"]
        FakeJob = purge_pii_module["FakeJob"]
        FakePiiEntity = purge_pii_module["FakePiiEntity"]

        anchor_job = MagicMock()
        anchor_job.job_id = "job-1"
        FakeJob.objects._store = [anchor_job]

        FakePiiEntity.objects._store = [
            _make_pii_entity("job-1", "NAME", "Jane Doe"),
            _make_pii_entity("job-1", "NAME", "JANE DOE"),
            _make_pii_entity("job-1", "NAME", "John Smith"),
        ]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            job_id=None,
            subject="jane doe",
            entity_type=None,
            dry_run=False,
            confirm=True,
        )

        output = cmd.stdout.getvalue()
        assert "Purged 2 PII entities" in output


# ===========================================================================
# Tests: entity-type filter
# ===========================================================================


class TestPurgeWithEntityTypeFilter:
    """purge_pii --entity-type should narrow the purge scope."""

    def test_entity_type_filter(self, purge_pii_module, monkeypatch):
        """Only entities matching the specified type are purged."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_pii_module["cmd_module"]
        FakeJob = purge_pii_module["FakeJob"]
        FakePiiEntity = purge_pii_module["FakePiiEntity"]

        anchor_job = MagicMock()
        anchor_job.job_id = "job-1"
        FakeJob.objects._store = [anchor_job]

        FakePiiEntity.objects._store = [
            _make_pii_entity("job-1", "SSN", "123-45-6789"),
            _make_pii_entity("job-1", "EMAIL", "test@example.com"),
            _make_pii_entity("job-1", "SSN", "987-65-4321"),
        ]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            job_id="job-1",
            subject=None,
            entity_type="SSN",
            dry_run=False,
            confirm=True,
        )

        output = cmd.stdout.getvalue()
        assert "Purged 2 PII entities" in output


# ===========================================================================
# Tests: dry-run
# ===========================================================================


class TestPurgeDryRun:
    """Dry-run mode should display summary but not delete."""

    def test_dry_run_no_deletion(self, purge_pii_module, monkeypatch):
        """--dry-run should not delete any records."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_pii_module["cmd_module"]
        FakeJob = purge_pii_module["FakeJob"]
        FakePiiEntity = purge_pii_module["FakePiiEntity"]
        FakeCustodyEvent = purge_pii_module["FakeCustodyEvent"]

        anchor_job = MagicMock()
        anchor_job.job_id = "job-1"
        FakeJob.objects._store = [anchor_job]

        FakePiiEntity.objects._store = [
            _make_pii_entity("job-1", "SSN", "123-45-6789"),
        ]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            job_id="job-1",
            subject=None,
            entity_type=None,
            dry_run=True,
            confirm=False,
        )

        output = cmd.stdout.getvalue()
        assert "DRY RUN" in output
        # Records should still exist (not deleted)
        assert FakePiiEntity.objects._store  # store still has items from original
        # No custody events in dry-run
        assert len(FakeCustodyEvent._created) == 0

    def test_dry_run_shows_summary(self, purge_pii_module, monkeypatch):
        """Dry-run output should include entity count and types."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_pii_module["cmd_module"]
        FakeJob = purge_pii_module["FakeJob"]
        FakePiiEntity = purge_pii_module["FakePiiEntity"]

        anchor_job = MagicMock()
        anchor_job.job_id = "job-1"
        FakeJob.objects._store = [anchor_job]

        FakePiiEntity.objects._store = [
            _make_pii_entity("job-1", "SSN", "123-45-6789"),
            _make_pii_entity("job-1", "EMAIL", "test@test.com"),
        ]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            job_id="job-1",
            subject=None,
            entity_type=None,
            dry_run=True,
            confirm=False,
        )

        output = cmd.stdout.getvalue()
        assert "Found 2 PII entities" in output
        assert "Entity types:" in output


# ===========================================================================
# Tests: --confirm gate
# ===========================================================================


class TestConfirmGate:
    """Missing --confirm should block deletion."""

    def test_missing_confirm_blocks_deletion(self, purge_pii_module, monkeypatch):
        """Without --confirm, no records should be deleted."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_pii_module["cmd_module"]
        FakeJob = purge_pii_module["FakeJob"]
        FakePiiEntity = purge_pii_module["FakePiiEntity"]
        FakeCustodyEvent = purge_pii_module["FakeCustodyEvent"]

        anchor_job = MagicMock()
        anchor_job.job_id = "job-1"
        FakeJob.objects._store = [anchor_job]

        FakePiiEntity.objects._store = [
            _make_pii_entity("job-1", "SSN", "123-45-6789"),
        ]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            job_id="job-1",
            subject=None,
            entity_type=None,
            dry_run=False,
            confirm=False,
        )

        output = cmd.stdout.getvalue()
        assert "--confirm" in output
        # No deletion, no custody event
        assert len(FakeCustodyEvent._created) == 0


# ===========================================================================
# Tests: CustodyEvent emitted on purge
# ===========================================================================


class TestCustodyEventEmitted:
    """Successful purge should emit a CustodyEvent."""

    def test_custody_event_on_purge(self, purge_pii_module, monkeypatch):
        """A pii_purged custody event should be created on deletion."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_pii_module["cmd_module"]
        FakeJob = purge_pii_module["FakeJob"]
        FakePiiEntity = purge_pii_module["FakePiiEntity"]
        FakeCustodyEvent = purge_pii_module["FakeCustodyEvent"]

        anchor_job = MagicMock()
        anchor_job.job_id = "job-1"
        FakeJob.objects._store = [anchor_job]

        FakePiiEntity.objects._store = [
            _make_pii_entity("job-1", "SSN", "123-45-6789"),
        ]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            job_id="job-1",
            subject=None,
            entity_type=None,
            dry_run=False,
            confirm=True,
        )

        assert len(FakeCustodyEvent._created) == 1
        event = FakeCustodyEvent._created[0]
        assert event["event_type"] == "pii_purged"
        assert event["data"]["action"] == "purge_pii"
        assert event["data"]["entity_count"] == 1
        assert "SSN" in event["data"]["entity_types"]
        assert "job-1" in event["data"]["affected_jobs"]

    def test_custody_event_has_purge_scope(self, purge_pii_module, monkeypatch):
        """Custody event data should include purge_scope describing the filter."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_pii_module["cmd_module"]
        FakeJob = purge_pii_module["FakeJob"]
        FakePiiEntity = purge_pii_module["FakePiiEntity"]
        FakeCustodyEvent = purge_pii_module["FakeCustodyEvent"]

        anchor_job = MagicMock()
        anchor_job.job_id = "job-1"
        FakeJob.objects._store = [anchor_job]

        FakePiiEntity.objects._store = [
            _make_pii_entity("job-1", "EMAIL", "me@test.com"),
        ]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            job_id="job-1",
            subject=None,
            entity_type="EMAIL",
            dry_run=False,
            confirm=True,
        )

        event = FakeCustodyEvent._created[0]
        assert "job_id=job-1" in event["data"]["purge_scope"]
        assert "entity_type=EMAIL" in event["data"]["purge_scope"]


# ===========================================================================
# Tests: LITIGATION_HOLD
# ===========================================================================


class TestLitigationHold:
    """LITIGATION_HOLD env var should block all purge operations."""

    def test_litigation_hold_blocks_purge(self, purge_pii_module, monkeypatch):
        """When LITIGATION_HOLD=true, purge should not proceed."""
        monkeypatch.setenv("LITIGATION_HOLD", "true")

        mod = purge_pii_module["cmd_module"]
        FakePiiEntity = purge_pii_module["FakePiiEntity"]
        FakeCustodyEvent = purge_pii_module["FakeCustodyEvent"]

        FakePiiEntity.objects._store = [
            _make_pii_entity("job-1", "SSN", "123-45-6789"),
        ]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.stderr = StringIO()
        cmd.handle(
            job_id="job-1",
            subject=None,
            entity_type=None,
            dry_run=False,
            confirm=True,
        )

        output = cmd.stderr.getvalue()
        assert "LITIGATION_HOLD" in output
        # No deletion
        assert len(FakeCustodyEvent._created) == 0
        # Store not touched
        assert len(FakePiiEntity.objects._store) == 1

    def test_litigation_hold_accepts_yes_and_one(self, purge_pii_module, monkeypatch):
        """LITIGATION_HOLD should accept 'yes', '1', and 'true'."""
        mod = purge_pii_module["cmd_module"]

        for value in ("yes", "1", "true", "TRUE", "True"):
            monkeypatch.setenv("LITIGATION_HOLD", value)
            cmd = mod.Command()
            cmd.stdout = StringIO()
            cmd.stderr = StringIO()
            cmd.handle(
                job_id="job-1",
                subject=None,
                entity_type=None,
                dry_run=False,
                confirm=True,
            )
            output = cmd.stderr.getvalue()
            assert "LITIGATION_HOLD" in output


# ===========================================================================
# Tests: no-match scenario
# ===========================================================================


class TestNoMatch:
    """Graceful message when no entities match the query."""

    def test_no_match_message(self, purge_pii_module, monkeypatch):
        """When no entities match, a clear message is displayed."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_pii_module["cmd_module"]
        FakePiiEntity = purge_pii_module["FakePiiEntity"]

        FakePiiEntity.objects._store = []

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            job_id=None,
            subject="Nobody",
            entity_type=None,
            dry_run=False,
            confirm=True,
        )

        output = cmd.stdout.getvalue()
        assert "No matching PII entities found" in output


# ===========================================================================
# Tests: operator identity in custody event
# ===========================================================================


class TestOperatorIdentity:
    """Custody events must include the operator's identity."""

    def test_operator_in_custody_event(self, purge_pii_module, monkeypatch):
        """Custody event data should include 'operator' from getpass.getuser()."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_pii_module["cmd_module"]
        FakeJob = purge_pii_module["FakeJob"]
        FakePiiEntity = purge_pii_module["FakePiiEntity"]
        FakeCustodyEvent = purge_pii_module["FakeCustodyEvent"]

        anchor_job = MagicMock()
        anchor_job.job_id = "job-1"
        FakeJob.objects._store = [anchor_job]

        FakePiiEntity.objects._store = [
            _make_pii_entity("job-1", "SSN", "123-45-6789"),
        ]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            job_id="job-1",
            subject=None,
            entity_type=None,
            dry_run=False,
            confirm=True,
        )

        assert len(FakeCustodyEvent._created) == 1
        event = FakeCustodyEvent._created[0]
        assert "operator" in event["data"]
        assert event["data"]["operator"] == getpass.getuser()
        assert isinstance(event["data"]["operator"], str)
        assert len(event["data"]["operator"]) > 0


# ===========================================================================
# Tests: requires at least one filter
# ===========================================================================


class TestRequiresFilter:
    """Command must require at least --job-id or --subject."""

    def test_no_filter_raises_error(self, purge_pii_module, monkeypatch):
        """Calling without --job-id or --subject should raise CommandError."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = purge_pii_module["cmd_module"]
        CommandError = purge_pii_module["CommandError"]

        cmd = mod.Command()
        cmd.stdout = StringIO()

        with pytest.raises(CommandError, match="--job-id or --subject"):
            cmd.handle(
                job_id=None,
                subject=None,
                entity_type=None,
                dry_run=False,
                confirm=True,
            )


# ===========================================================================
# Tests: custody.py EVENT_TYPES registration
# ===========================================================================


class TestCustodyEventTypeRegistered:
    """Verify pii_purged is registered in custody.py EVENT_TYPES."""

    def test_pii_purged_in_event_types(self):
        """The pii_purged event type should be defined in custody.py."""
        from custody import EVENT_TYPES

        assert "pii_purged" in EVENT_TYPES
        assert isinstance(EVENT_TYPES["pii_purged"], str)
        assert len(EVENT_TYPES["pii_purged"]) > 0
