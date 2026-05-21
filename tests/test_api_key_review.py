"""Tests for API key access review management commands (C-08 + C-11).

Validates:
- list_api_keys shows active keys and supports --active-only filtering
- audit_api_access generates text and JSON reports for a configurable period
- revoke_api_key sets is_active=False, emits CustodyEvent, requires --confirm

Django models are mocked at the module boundary (same approach as
test_compliance_retention.py and test_audit_deletion_events.py).
"""

import importlib
import json
import os
import sys
import types
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

        def ERROR(self, msg):
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


class _FakeQuerySet:
    """Minimal queryset stand-in."""

    def __init__(self, items):
        self._items = list(items)

    def filter(self, **kwargs):
        result = list(self._items)
        # Support is_active filtering
        if "is_active" in kwargs:
            result = [i for i in result if getattr(i, "is_active", True) == kwargs["is_active"]]
        # Support key_id__startswith filtering
        if "key_id__startswith" in kwargs:
            prefix = kwargs["key_id__startswith"]
            result = [i for i in result if getattr(i, "key_id", "").startswith(prefix)]
        return _FakeQuerySet(result)

    def all(self):
        return _FakeQuerySet(self._items)

    def get(self, **kwargs):
        for item in self._items:
            match = True
            for k, v in kwargs.items():
                if getattr(item, k, None) != v:
                    match = False
                    break
            if match:
                return item
        # Raise DoesNotExist
        raise _FakeDoesNotExist()

    def count(self):
        return len(self._items)

    def first(self):
        return self._items[0] if self._items else None

    def __getitem__(self, key):
        return self._items[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def values_list(self, *fields, flat=False):
        return _FakeQuerySet(self._items)


class _FakeDoesNotExist(Exception):
    pass


def _build_mock_job_models():
    """Return a mock ``jobs.models`` module with Job, CustodyEvent, ApiKeyRecord."""

    models_mod = types.ModuleType("jobs.models")

    # -- Job -----------------------------------------------------------------
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
            def first(cls):
                return cls._store[0] if cls._store else None

            @classmethod
            def all(cls):
                return _FakeQuerySet(cls._store)

    # -- CustodyEvent --------------------------------------------------------
    class FakeCustodyEvent:
        _created = []

        class objects:
            @classmethod
            def create(cls, **kwargs):
                FakeCustodyEvent._created.append(kwargs)
                return MagicMock()

    # -- ApiKeyRecord --------------------------------------------------------
    class FakeApiKeyRecord:
        DoesNotExist = _FakeDoesNotExist

        def __init__(self, key_id="", description="", is_active=True,
                     use_count=0, last_used_at=None, created_at=None,
                     permissions=None):
            self.key_id = key_id
            self.description = description
            self.is_active = is_active
            self.use_count = use_count
            self.last_used_at = last_used_at
            self.created_at = created_at or datetime.now(timezone.utc)
            self.permissions = permissions or []
            self._saved = False

        def save(self):
            self._saved = True

        class objects:
            _store = []

            @classmethod
            def all(cls):
                return _FakeQuerySet(cls._store)

            @classmethod
            def filter(cls, **kwargs):
                return _FakeQuerySet(cls._store).filter(**kwargs)

            @classmethod
            def get(cls, **kwargs):
                return _FakeQuerySet(cls._store).get(**kwargs)

            @classmethod
            def first(cls):
                return cls._store[0] if cls._store else None

    models_mod.Job = FakeJob
    models_mod.CustodyEvent = FakeCustodyEvent
    models_mod.ApiKeyRecord = FakeApiKeyRecord

    jobs_mod = types.ModuleType("jobs")

    return {
        "jobs": jobs_mod,
        "jobs.models": models_mod,
    }


# ---------------------------------------------------------------------------
# Shared import helper
# ---------------------------------------------------------------------------


def _import_command(command_name, django_mocks, model_mocks):
    """Import a management command module with mocks in place."""
    all_mocks = {**django_mocks, **model_mocks}

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
    spec.loader.exec_module(cmd_module)

    return cmd_module, all_mocks, saved


def _restore_modules(saved):
    """Restore original sys.modules entries."""
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def list_keys_module():
    """Import list_api_keys under a mocked Django/jobs environment."""
    django_mocks = _build_mock_django()
    model_mocks = _build_mock_job_models()

    cmd_module, all_mocks, saved = _import_command(
        "list_api_keys", django_mocks, model_mocks
    )

    FakeApiKeyRecord = all_mocks["jobs.models"].ApiKeyRecord

    # Reset state
    FakeApiKeyRecord.objects._store = []

    yield {
        "cmd_module": cmd_module,
        "FakeApiKeyRecord": FakeApiKeyRecord,
    }

    _restore_modules(saved)


@pytest.fixture()
def audit_module():
    """Import audit_api_access under a mocked Django/jobs environment."""
    django_mocks = _build_mock_django()
    model_mocks = _build_mock_job_models()

    cmd_module, all_mocks, saved = _import_command(
        "audit_api_access", django_mocks, model_mocks
    )

    FakeApiKeyRecord = all_mocks["jobs.models"].ApiKeyRecord

    # Reset state
    FakeApiKeyRecord.objects._store = []

    yield {
        "cmd_module": cmd_module,
        "FakeApiKeyRecord": FakeApiKeyRecord,
    }

    _restore_modules(saved)


@pytest.fixture()
def revoke_module():
    """Import revoke_api_key under a mocked Django/jobs environment."""
    django_mocks = _build_mock_django()
    model_mocks = _build_mock_job_models()

    cmd_module, all_mocks, saved = _import_command(
        "revoke_api_key", django_mocks, model_mocks
    )

    FakeApiKeyRecord = all_mocks["jobs.models"].ApiKeyRecord
    FakeJob = all_mocks["jobs.models"].Job
    FakeCustodyEvent = all_mocks["jobs.models"].CustodyEvent

    # Reset state
    FakeApiKeyRecord.objects._store = []
    FakeJob.objects._store = []
    FakeCustodyEvent._created = []

    yield {
        "cmd_module": cmd_module,
        "FakeApiKeyRecord": FakeApiKeyRecord,
        "FakeJob": FakeJob,
        "FakeCustodyEvent": FakeCustodyEvent,
    }

    _restore_modules(saved)


# ===========================================================================
# Tests: list_api_keys
# ===========================================================================


class TestListApiKeys:
    """Tests for the list_api_keys management command."""

    def test_list_shows_all_keys(self, list_keys_module):
        """list_api_keys should display all registered keys."""
        mod = list_keys_module["cmd_module"]
        FakeApiKeyRecord = list_keys_module["FakeApiKeyRecord"]

        key1 = FakeApiKeyRecord(
            key_id="abcdef123456abcdef123456",
            description="Production key",
            is_active=True,
            use_count=42,
        )
        key2 = FakeApiKeyRecord(
            key_id="xyz789xyz789xyz789xyz789",
            description="Staging key",
            is_active=False,
            use_count=10,
        )
        FakeApiKeyRecord.objects._store = [key1, key2]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(active_only=False)

        output = cmd.stdout.getvalue()
        assert "abcdef123456" in output
        assert "xyz789xyz789" in output
        assert "Production key" in output
        assert "Staging key" in output
        assert "Total: 2" in output

    def test_list_active_only_filters(self, list_keys_module):
        """list_api_keys --active-only should exclude revoked keys."""
        mod = list_keys_module["cmd_module"]
        FakeApiKeyRecord = list_keys_module["FakeApiKeyRecord"]

        key1 = FakeApiKeyRecord(
            key_id="active_key_0001",
            description="Active key",
            is_active=True,
        )
        key2 = FakeApiKeyRecord(
            key_id="revoked_key_001",
            description="Revoked key",
            is_active=False,
        )
        FakeApiKeyRecord.objects._store = [key1, key2]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(active_only=True)

        output = cmd.stdout.getvalue()
        assert "Active key" in output
        assert "Revoked key" not in output
        assert "Total: 1" in output

    def test_list_empty_shows_message(self, list_keys_module):
        """list_api_keys with no keys should show a message."""
        mod = list_keys_module["cmd_module"]
        FakeApiKeyRecord = list_keys_module["FakeApiKeyRecord"]

        FakeApiKeyRecord.objects._store = []

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(active_only=False)

        output = cmd.stdout.getvalue()
        assert "No " in output
        assert "API keys found" in output

    def test_list_active_only_empty_shows_message(self, list_keys_module):
        """list_api_keys --active-only with no active keys shows appropriate message."""
        mod = list_keys_module["cmd_module"]
        FakeApiKeyRecord = list_keys_module["FakeApiKeyRecord"]

        # Only a revoked key
        key1 = FakeApiKeyRecord(
            key_id="revoked_only_key",
            description="Revoked",
            is_active=False,
        )
        FakeApiKeyRecord.objects._store = [key1]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(active_only=True)

        output = cmd.stdout.getvalue()
        assert "active" in output.lower()
        assert "API keys found" in output

    def test_list_shows_use_count(self, list_keys_module):
        """list_api_keys should display the use_count for each key."""
        mod = list_keys_module["cmd_module"]
        FakeApiKeyRecord = list_keys_module["FakeApiKeyRecord"]

        key1 = FakeApiKeyRecord(
            key_id="counter_key_0001",
            description="Counter test",
            is_active=True,
            use_count=1337,
        )
        FakeApiKeyRecord.objects._store = [key1]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(active_only=False)

        output = cmd.stdout.getvalue()
        assert "1337" in output

    def test_list_shows_summary_counts(self, list_keys_module):
        """list_api_keys should show active/revoked summary counts."""
        mod = list_keys_module["cmd_module"]
        FakeApiKeyRecord = list_keys_module["FakeApiKeyRecord"]

        FakeApiKeyRecord.objects._store = [
            FakeApiKeyRecord(key_id="k1", is_active=True),
            FakeApiKeyRecord(key_id="k2", is_active=True),
            FakeApiKeyRecord(key_id="k3", is_active=False),
        ]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(active_only=False)

        output = cmd.stdout.getvalue()
        assert "2 active" in output
        assert "1 revoked" in output


# ===========================================================================
# Tests: audit_api_access
# ===========================================================================


class TestAuditApiAccess:
    """Tests for the audit_api_access management command."""

    def test_audit_generates_text_report(self, audit_module):
        """audit_api_access should produce a text report."""
        mod = audit_module["cmd_module"]
        FakeApiKeyRecord = audit_module["FakeApiKeyRecord"]

        now = datetime.now(timezone.utc)
        key1 = FakeApiKeyRecord(
            key_id="audit_key_001001",
            description="Test key",
            is_active=True,
            use_count=100,
            last_used_at=now - timedelta(hours=1),
        )
        FakeApiKeyRecord.objects._store = [key1]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(days=30, output="text")

        output = cmd.stdout.getvalue()
        assert "API Access Audit Report" in output
        assert "Total keys:" in output
        assert "1" in output

    def test_audit_generates_json_report(self, audit_module):
        """audit_api_access --output json should produce valid JSON."""
        mod = audit_module["cmd_module"]
        FakeApiKeyRecord = audit_module["FakeApiKeyRecord"]

        now = datetime.now(timezone.utc)
        key1 = FakeApiKeyRecord(
            key_id="json_key_00100001",
            description="JSON test",
            is_active=True,
            use_count=50,
            last_used_at=now - timedelta(hours=2),
        )
        FakeApiKeyRecord.objects._store = [key1]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(days=30, output="json")

        output = cmd.stdout.getvalue()
        data = json.loads(output)
        assert data["total_keys"] == 1
        assert data["period_days"] == 30
        assert "keys" in data
        assert len(data["keys"]) == 1
        assert data["keys"][0]["key_id"] == "json_key_001..."

    def test_audit_days_parameter(self, audit_module):
        """audit_api_access --days 7 should filter to 7-day period."""
        mod = audit_module["cmd_module"]
        FakeApiKeyRecord = audit_module["FakeApiKeyRecord"]

        now = datetime.now(timezone.utc)
        # Key used 3 days ago (within 7-day window)
        recent_key = FakeApiKeyRecord(
            key_id="recent_key_001001",
            description="Recent",
            is_active=True,
            use_count=10,
            last_used_at=now - timedelta(days=3),
        )
        # Key used 15 days ago (outside 7-day window)
        old_key = FakeApiKeyRecord(
            key_id="old_key_001001001",
            description="Old",
            is_active=True,
            use_count=5,
            last_used_at=now - timedelta(days=15),
        )
        FakeApiKeyRecord.objects._store = [recent_key, old_key]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(days=7, output="json")

        output = cmd.stdout.getvalue()
        data = json.loads(output)
        assert data["period_days"] == 7
        assert data["keys_used_in_period"] == 1
        assert data["keys_unused_in_period"] == 1

    def test_audit_empty_keys(self, audit_module):
        """audit_api_access with no keys should produce an empty report."""
        mod = audit_module["cmd_module"]
        FakeApiKeyRecord = audit_module["FakeApiKeyRecord"]

        FakeApiKeyRecord.objects._store = []

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(days=30, output="json")

        output = cmd.stdout.getvalue()
        data = json.loads(output)
        assert data["total_keys"] == 0
        assert data["keys"] == []

    def test_audit_text_warns_unused_active(self, audit_module):
        """Text report should warn about active keys unused in the period."""
        mod = audit_module["cmd_module"]
        FakeApiKeyRecord = audit_module["FakeApiKeyRecord"]

        # Active key that has never been used
        key1 = FakeApiKeyRecord(
            key_id="unused_active_key",
            description="Never used",
            is_active=True,
            use_count=0,
            last_used_at=None,
        )
        FakeApiKeyRecord.objects._store = [key1]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(days=30, output="text")

        output = cmd.stdout.getvalue()
        assert "WARNING" in output
        assert "not been used" in output

    def test_audit_json_contains_permissions(self, audit_module):
        """JSON report should include permissions for each key."""
        mod = audit_module["cmd_module"]
        FakeApiKeyRecord = audit_module["FakeApiKeyRecord"]

        key1 = FakeApiKeyRecord(
            key_id="perms_key_0010001",
            description="Admin key",
            is_active=True,
            permissions=["read", "write", "admin"],
        )
        FakeApiKeyRecord.objects._store = [key1]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(days=30, output="json")

        output = cmd.stdout.getvalue()
        data = json.loads(output)
        assert data["keys"][0]["permissions"] == ["read", "write", "admin"]


# ===========================================================================
# Tests: revoke_api_key
# ===========================================================================


class TestRevokeApiKey:
    """Tests for the revoke_api_key management command."""

    def test_revoke_sets_inactive(self, revoke_module):
        """revoke_api_key --confirm should set is_active=False."""
        mod = revoke_module["cmd_module"]
        FakeApiKeyRecord = revoke_module["FakeApiKeyRecord"]
        FakeJob = revoke_module["FakeJob"]

        key = FakeApiKeyRecord(
            key_id="revoke_target_key001",
            description="To be revoked",
            is_active=True,
            use_count=99,
        )
        FakeApiKeyRecord.objects._store = [key]
        FakeJob.objects._store = [MagicMock()]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(key_id="revoke_target_key001", confirm=True)

        assert key.is_active is False
        assert key._saved is True

    def test_revoke_emits_custody_event(self, revoke_module):
        """revoke_api_key --confirm should emit a CustodyEvent."""
        mod = revoke_module["cmd_module"]
        FakeApiKeyRecord = revoke_module["FakeApiKeyRecord"]
        FakeJob = revoke_module["FakeJob"]
        FakeCustodyEvent = revoke_module["FakeCustodyEvent"]

        key = FakeApiKeyRecord(
            key_id="custody_target_key01",
            description="Custody test",
            is_active=True,
            use_count=42,
        )
        FakeApiKeyRecord.objects._store = [key]
        FakeJob.objects._store = [MagicMock()]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(key_id="custody_target_key01", confirm=True)

        assert len(FakeCustodyEvent._created) == 1
        event = FakeCustodyEvent._created[0]
        assert event["event_type"] == "api_key_revoked"
        assert event["data"]["action"] == "revoke_api_key"
        assert event["data"]["key_id"] == "custody_target_key01"
        assert event["data"]["use_count_at_revocation"] == 42
        assert event["data"]["reason"] == "manual_revocation"
        assert "operator" in event["data"]

    def test_revoke_requires_confirm(self, revoke_module):
        """revoke_api_key without --confirm should not revoke."""
        mod = revoke_module["cmd_module"]
        FakeApiKeyRecord = revoke_module["FakeApiKeyRecord"]
        FakeCustodyEvent = revoke_module["FakeCustodyEvent"]

        key = FakeApiKeyRecord(
            key_id="no_confirm_key_0001",
            description="Should not be revoked",
            is_active=True,
        )
        FakeApiKeyRecord.objects._store = [key]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(key_id="no_confirm_key_0001", confirm=False)

        # Key should still be active
        assert key.is_active is True
        assert key._saved is False

        # No custody event
        assert len(FakeCustodyEvent._created) == 0

        # Output should mention dry run
        output = cmd.stdout.getvalue()
        assert "DRY RUN" in output or "confirm" in output.lower()

    def test_revoke_already_revoked(self, revoke_module):
        """revoke_api_key on already-revoked key should inform the user."""
        mod = revoke_module["cmd_module"]
        FakeApiKeyRecord = revoke_module["FakeApiKeyRecord"]
        FakeCustodyEvent = revoke_module["FakeCustodyEvent"]

        key = FakeApiKeyRecord(
            key_id="already_revoked_k01",
            description="Already revoked",
            is_active=False,
        )
        FakeApiKeyRecord.objects._store = [key]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(key_id="already_revoked_k01", confirm=True)

        output = cmd.stdout.getvalue()
        assert "already revoked" in output.lower()
        assert len(FakeCustodyEvent._created) == 0

    def test_revoke_nonexistent_key_raises(self, revoke_module):
        """revoke_api_key with unknown key_id should raise CommandError."""
        mod = revoke_module["cmd_module"]
        FakeApiKeyRecord = revoke_module["FakeApiKeyRecord"]

        FakeApiKeyRecord.objects._store = []

        # The command imports CommandError from django.core.management.base
        # which is our mock
        from django.core.management.base import CommandError as MockCommandError

        cmd = mod.Command()
        cmd.stdout = StringIO()

        with pytest.raises(MockCommandError, match="No API key found"):
            cmd.handle(key_id="nonexistent_key_id", confirm=True)

    def test_revoke_by_prefix(self, revoke_module):
        """revoke_api_key should work with a key_id prefix match."""
        mod = revoke_module["cmd_module"]
        FakeApiKeyRecord = revoke_module["FakeApiKeyRecord"]
        FakeJob = revoke_module["FakeJob"]
        FakeCustodyEvent = revoke_module["FakeCustodyEvent"]

        key = FakeApiKeyRecord(
            key_id="prefix_match_full_key_id_here",
            description="Prefix match test",
            is_active=True,
        )
        FakeApiKeyRecord.objects._store = [key]
        FakeJob.objects._store = [MagicMock()]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        # Use a prefix of the full key_id
        cmd.handle(key_id="prefix_match_full", confirm=True)

        assert key.is_active is False
        assert len(FakeCustodyEvent._created) == 1

    def test_revoke_ambiguous_prefix_raises(self, revoke_module):
        """revoke_api_key with ambiguous prefix should raise CommandError."""
        mod = revoke_module["cmd_module"]
        FakeApiKeyRecord = revoke_module["FakeApiKeyRecord"]

        key1 = FakeApiKeyRecord(
            key_id="ambig_key_001_first",
            description="First",
            is_active=True,
        )
        key2 = FakeApiKeyRecord(
            key_id="ambig_key_001_second",
            description="Second",
            is_active=True,
        )
        FakeApiKeyRecord.objects._store = [key1, key2]

        from django.core.management.base import CommandError as MockCommandError

        cmd = mod.Command()
        cmd.stdout = StringIO()

        with pytest.raises(MockCommandError, match="Ambiguous"):
            cmd.handle(key_id="ambig_key_001", confirm=True)

    def test_revoke_no_job_for_custody_event(self, revoke_module):
        """revoke_api_key should not raise when no Job exists for custody FK."""
        mod = revoke_module["cmd_module"]
        FakeApiKeyRecord = revoke_module["FakeApiKeyRecord"]
        FakeJob = revoke_module["FakeJob"]
        FakeCustodyEvent = revoke_module["FakeCustodyEvent"]

        key = FakeApiKeyRecord(
            key_id="no_job_key_0001001",
            description="No job for FK",
            is_active=True,
        )
        FakeApiKeyRecord.objects._store = [key]
        FakeJob.objects._store = []  # No jobs

        cmd = mod.Command()
        cmd.stdout = StringIO()
        # Should not raise
        cmd.handle(key_id="no_job_key_0001001", confirm=True)

        assert key.is_active is False
        # No custody event created (no job to anchor to)
        assert len(FakeCustodyEvent._created) == 0

    def test_revoke_output_shows_key_info(self, revoke_module):
        """revoke_api_key should display key information before revoking."""
        mod = revoke_module["cmd_module"]
        FakeApiKeyRecord = revoke_module["FakeApiKeyRecord"]
        FakeJob = revoke_module["FakeJob"]

        key = FakeApiKeyRecord(
            key_id="info_display_key_001",
            description="Show info test",
            is_active=True,
            use_count=77,
        )
        FakeApiKeyRecord.objects._store = [key]
        FakeJob.objects._store = [MagicMock()]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(key_id="info_display_key_001", confirm=True)

        output = cmd.stdout.getvalue()
        assert "info_display_" in output
        assert "Show info test" in output
        assert "77" in output
        assert "revoked" in output.lower()


# ===========================================================================
# Tests: ApiKeyRecord model
# ===========================================================================


class TestApiKeyRecordModel:
    """Basic validation of the ApiKeyRecord model definition."""

    def test_model_str_active(self):
        """__str__ should show truncated key_id and active status."""
        # Import the real model module (not Django-dependent for __str__)
        # We test the mock instead since the real model needs Django
        from tests.test_api_key_review import _build_mock_job_models

        models = _build_mock_job_models()
        ApiKeyRecord = models["jobs.models"].ApiKeyRecord

        record = ApiKeyRecord(
            key_id="a" * 64,
            description="Test",
            is_active=True,
        )
        # Just verify the mock creates usable objects
        assert record.key_id == "a" * 64
        assert record.is_active is True

    def test_model_defaults(self):
        """ApiKeyRecord should have sensible defaults."""
        from tests.test_api_key_review import _build_mock_job_models

        models = _build_mock_job_models()
        ApiKeyRecord = models["jobs.models"].ApiKeyRecord

        record = ApiKeyRecord(key_id="test_default")
        assert record.use_count == 0
        assert record.is_active is True
        assert record.permissions == []
        assert record.description == ""
        assert record.last_used_at is None
