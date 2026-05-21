"""Tests for C-12+C-13: Output cleanup and audit log rotation commands.

Validates that:
- cleanup_output removes output files for old completed jobs
- cleanup_output dry-run doesn't delete
- cleanup_output emits CustodyEvent
- cleanup_output respects LITIGATION_HOLD
- cleanup_output respects --retention-days
- cleanup_output respects --confirm gate
- rotate_audit_logs archives records to JSONL
- rotate_audit_logs dry-run doesn't delete
- rotate_audit_logs emits CustodyEvent
- rotate_audit_logs verifies archive integrity
- rotate_audit_logs respects LITIGATION_HOLD
- rotate_audit_logs respects --confirm gate
- Both new event types are registered in custody.py

Django models are mocked at the module boundary (same approach as
test_pii_purge.py and test_tenant_purge.py).
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
        result = list(self._items)
        for key, value in kwargs.items():
            filtered = []
            for item in result:
                if key == "status__in":
                    if getattr(item, "status", "") in value:
                        filtered.append(item)
                elif key == "completed_at__lt":
                    completed = getattr(item, "completed_at", None)
                    if completed is not None and completed < value:
                        filtered.append(item)
                elif key == "timestamp__lt":
                    ts = getattr(item, "timestamp", None)
                    if ts is not None and ts < value:
                        filtered.append(item)
                else:
                    filtered.append(item)
            result = filtered
        return _FakeQuerySet(result)

    def all(self):
        return _FakeQuerySet(self._items)

    def exclude(self, **kwargs):
        return _FakeQuerySet(self._items)

    def order_by(self, *fields):
        return _FakeQuerySet(self._items)

    def values_list(self, *fields, flat=False):
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
    """Return a mock ``jobs.models`` module with Job, CustodyEvent."""

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
                return _FakeQuerySet(cls._store).filter(**kwargs)

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
                return _FakeQuerySet(cls._store).filter(**kwargs)

    models_mod.Job = FakeJob
    models_mod.CustodyEvent = FakeCustodyEvent

    jobs_mod = types.ModuleType("jobs")

    return {
        "jobs": jobs_mod,
        "jobs.models": models_mod,
    }


def _make_job(job_id, status="completed", completed_at=None, nfs_job_path=""):
    """Create a fake Job-like object."""
    job = MagicMock()
    job.job_id = job_id
    job.status = status
    job.completed_at = completed_at
    job.nfs_job_path = nfs_job_path
    return job


def _make_custody_event(event_id, document_id, event_type, timestamp,
                        data=None, prev_hash="", event_hash="",
                        job_id="job-1", worker_hostname=""):
    """Create a fake CustodyEvent-like object."""
    event = MagicMock()
    event.id = event_id
    event.document_id = document_id
    event.event_type = event_type
    event.timestamp = timestamp
    event.data = data or {}
    event.prev_hash = prev_hash
    event.event_hash = event_hash
    event.job_id = job_id
    event.worker_hostname = worker_hostname
    event.chain_finalized = False
    return event


# ---------------------------------------------------------------------------
# Module import helpers
# ---------------------------------------------------------------------------


def _import_command(monkeypatch, django_mocks, model_mocks, command_name):
    """Import a management command module with mocks in place."""
    all_mocks = {**django_mocks, **model_mocks}

    saved = {}
    for name, mod in all_mocks.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    # Inject the real litigation_hold module so commands can import it
    lh_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "coordinator", "jobs", "litigation_hold.py",
    )
    lh_spec = importlib.util.spec_from_file_location("jobs.litigation_hold", lh_path)
    lh_mod = importlib.util.module_from_spec(lh_spec)
    lh_spec.loader.exec_module(lh_mod)
    saved["jobs.litigation_hold"] = sys.modules.get("jobs.litigation_hold")
    sys.modules["jobs.litigation_hold"] = lh_mod

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
def cleanup_output_module(monkeypatch):
    """Import cleanup_output under a mocked Django/jobs environment."""
    django_mocks = _build_mock_django()
    model_mocks = _build_mock_job_models()

    cmd_module, all_mocks, saved = _import_command(
        monkeypatch, django_mocks, model_mocks, "cleanup_output"
    )

    FakeJob = all_mocks["jobs.models"].Job
    FakeCustodyEvent = all_mocks["jobs.models"].CustodyEvent

    # Reset state
    FakeJob.objects._store = []
    FakeCustodyEvent._created = []
    FakeCustodyEvent.objects._store = []

    yield {
        "cmd_module": cmd_module,
        "FakeJob": FakeJob,
        "FakeCustodyEvent": FakeCustodyEvent,
        "django_mocks": django_mocks,
    }

    _restore_modules(saved)


@pytest.fixture()
def rotate_audit_module(monkeypatch):
    """Import rotate_audit_logs under a mocked Django/jobs environment."""
    django_mocks = _build_mock_django()
    model_mocks = _build_mock_job_models()

    cmd_module, all_mocks, saved = _import_command(
        monkeypatch, django_mocks, model_mocks, "rotate_audit_logs"
    )

    FakeJob = all_mocks["jobs.models"].Job
    FakeCustodyEvent = all_mocks["jobs.models"].CustodyEvent

    # Reset state
    FakeJob.objects._store = []
    FakeCustodyEvent._created = []
    FakeCustodyEvent.objects._store = []

    yield {
        "cmd_module": cmd_module,
        "FakeJob": FakeJob,
        "FakeCustodyEvent": FakeCustodyEvent,
        "django_mocks": django_mocks,
    }

    _restore_modules(saved)


# ===========================================================================
# Tests: cleanup_output -- removes old output files
# ===========================================================================


class TestCleanupOutputRemovesFiles:
    """cleanup_output --confirm should remove output files for old completed jobs."""

    def test_cleanup_removes_old_output_files(self, cleanup_output_module, monkeypatch, tmp_path):
        """Output EXPORT subdirs are removed for old completed jobs."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = cleanup_output_module["cmd_module"]
        FakeJob = cleanup_output_module["FakeJob"]

        # Create a fake NFS output directory with EXPORT subdirs
        job_dir = tmp_path / "job-1"
        output_dir = job_dir / "output"
        pdf_dir = output_dir / "EXPORT" / "PDF"
        text_dir = output_dir / "EXPORT" / "TEXT"
        pdf_dir.mkdir(parents=True)
        text_dir.mkdir(parents=True)
        (pdf_dir / "doc.pdf").write_bytes(b"fake pdf content")
        (text_dir / "doc.txt").write_text("fake text content")

        old_date = datetime.now(timezone.utc) - timedelta(days=120)
        job = _make_job("job-1", status="completed", completed_at=old_date,
                        nfs_job_path=str(job_dir))
        FakeJob.objects._store = [job]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            retention_days=90,
            dry_run=False,
            confirm=True,
        )

        output = cmd.stdout.getvalue()
        assert "Cleaned output" in output
        assert "2 files removed" in output
        # EXPORT subdirs should be removed
        assert not pdf_dir.exists()
        assert not text_dir.exists()

    def test_cleanup_preserves_job_record(self, cleanup_output_module, monkeypatch, tmp_path):
        """Job records should remain after output cleanup (only files removed)."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = cleanup_output_module["cmd_module"]
        FakeJob = cleanup_output_module["FakeJob"]

        job_dir = tmp_path / "job-2"
        output_dir = job_dir / "output"
        pdf_dir = output_dir / "EXPORT" / "PDF"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "doc.pdf").write_bytes(b"content")

        old_date = datetime.now(timezone.utc) - timedelta(days=100)
        job = _make_job("job-2", status="completed", completed_at=old_date,
                        nfs_job_path=str(job_dir))
        FakeJob.objects._store = [job]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            retention_days=90,
            dry_run=False,
            confirm=True,
        )

        # Job should still be in the store (not deleted)
        assert len(FakeJob.objects._store) == 1


# ===========================================================================
# Tests: cleanup_output -- dry-run
# ===========================================================================


class TestCleanupOutputDryRun:
    """cleanup_output --dry-run should preview without deleting."""

    def test_dry_run_no_deletion(self, cleanup_output_module, monkeypatch, tmp_path):
        """--dry-run should not remove any files."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = cleanup_output_module["cmd_module"]
        FakeJob = cleanup_output_module["FakeJob"]
        FakeCustodyEvent = cleanup_output_module["FakeCustodyEvent"]

        job_dir = tmp_path / "job-dry"
        output_dir = job_dir / "output"
        pdf_dir = output_dir / "EXPORT" / "PDF"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "doc.pdf").write_bytes(b"content")

        old_date = datetime.now(timezone.utc) - timedelta(days=100)
        job = _make_job("job-dry", status="completed", completed_at=old_date,
                        nfs_job_path=str(job_dir))
        FakeJob.objects._store = [job]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            retention_days=90,
            dry_run=True,
            confirm=False,
        )

        output = cmd.stdout.getvalue()
        assert "DRY RUN" in output
        # Files should still exist
        assert pdf_dir.exists()
        assert (pdf_dir / "doc.pdf").exists()
        # No custody events in dry-run
        assert len(FakeCustodyEvent._created) == 0


# ===========================================================================
# Tests: cleanup_output -- CustodyEvent emitted
# ===========================================================================


class TestCleanupOutputCustodyEvent:
    """cleanup_output should emit a CustodyEvent on successful cleanup."""

    def test_custody_event_emitted(self, cleanup_output_module, monkeypatch, tmp_path):
        """An output_cleaned custody event should be created."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = cleanup_output_module["cmd_module"]
        FakeJob = cleanup_output_module["FakeJob"]
        FakeCustodyEvent = cleanup_output_module["FakeCustodyEvent"]

        job_dir = tmp_path / "job-ce"
        output_dir = job_dir / "output"
        pdf_dir = output_dir / "EXPORT" / "PDF"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "doc.pdf").write_bytes(b"content")

        old_date = datetime.now(timezone.utc) - timedelta(days=100)
        job = _make_job("job-ce", status="completed", completed_at=old_date,
                        nfs_job_path=str(job_dir))
        FakeJob.objects._store = [job]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            retention_days=90,
            dry_run=False,
            confirm=True,
        )

        assert len(FakeCustodyEvent._created) == 1
        event = FakeCustodyEvent._created[0]
        assert event["event_type"] == "output_cleaned"
        assert event["data"]["action"] == "cleanup_output"
        assert event["data"]["jobs_cleaned"] == 1
        assert event["data"]["files_removed"] >= 1
        assert event["data"]["bytes_freed"] >= 0
        assert "operator" in event["data"]
        assert "retention_policy" in event["data"]


# ===========================================================================
# Tests: cleanup_output -- LITIGATION_HOLD
# ===========================================================================


class TestCleanupOutputLitigationHold:
    """LITIGATION_HOLD should block cleanup_output."""

    def test_litigation_hold_blocks_cleanup(self, cleanup_output_module, monkeypatch, tmp_path):
        """When LITIGATION_HOLD=true, cleanup should not proceed."""
        monkeypatch.setenv("LITIGATION_HOLD", "true")

        mod = cleanup_output_module["cmd_module"]
        FakeJob = cleanup_output_module["FakeJob"]
        FakeCustodyEvent = cleanup_output_module["FakeCustodyEvent"]

        job_dir = tmp_path / "job-hold"
        output_dir = job_dir / "output"
        pdf_dir = output_dir / "EXPORT" / "PDF"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "doc.pdf").write_bytes(b"content")

        old_date = datetime.now(timezone.utc) - timedelta(days=100)
        job = _make_job("job-hold", status="completed", completed_at=old_date,
                        nfs_job_path=str(job_dir))
        FakeJob.objects._store = [job]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.stderr = StringIO()
        cmd.handle(
            retention_days=90,
            dry_run=False,
            confirm=True,
        )

        output = cmd.stderr.getvalue()
        assert "LITIGATION_HOLD" in output
        # Files should still exist
        assert pdf_dir.exists()
        assert len(FakeCustodyEvent._created) == 0

    def test_litigation_hold_accepts_yes_and_one(self, cleanup_output_module, monkeypatch):
        """LITIGATION_HOLD should accept 'yes', '1', and 'true'."""
        mod = cleanup_output_module["cmd_module"]

        for value in ("yes", "1", "true", "TRUE", "True"):
            monkeypatch.setenv("LITIGATION_HOLD", value)
            cmd = mod.Command()
            cmd.stdout = StringIO()
            cmd.stderr = StringIO()
            cmd.handle(
                retention_days=90,
                dry_run=False,
                confirm=True,
            )
            output = cmd.stderr.getvalue()
            assert "LITIGATION_HOLD" in output


# ===========================================================================
# Tests: cleanup_output -- retention-days respected
# ===========================================================================


class TestCleanupOutputRetentionDays:
    """cleanup_output --retention-days should control what gets cleaned."""

    def test_recent_jobs_not_cleaned(self, cleanup_output_module, monkeypatch, tmp_path):
        """Jobs newer than retention period should not be cleaned."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = cleanup_output_module["cmd_module"]
        FakeJob = cleanup_output_module["FakeJob"]

        job_dir = tmp_path / "job-recent"
        output_dir = job_dir / "output"
        pdf_dir = output_dir / "EXPORT" / "PDF"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "doc.pdf").write_bytes(b"content")

        # Job completed only 10 days ago
        recent_date = datetime.now(timezone.utc) - timedelta(days=10)
        job = _make_job("job-recent", status="completed", completed_at=recent_date,
                        nfs_job_path=str(job_dir))
        FakeJob.objects._store = [job]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            retention_days=90,
            dry_run=False,
            confirm=True,
        )

        output = cmd.stdout.getvalue()
        assert "No completed/failed jobs" in output
        # Files should still exist
        assert (pdf_dir / "doc.pdf").exists()


# ===========================================================================
# Tests: cleanup_output -- confirm gate
# ===========================================================================


class TestCleanupOutputConfirmGate:
    """cleanup_output without --confirm should not delete."""

    def test_missing_confirm_blocks_deletion(self, cleanup_output_module, monkeypatch, tmp_path):
        """Without --confirm, no files should be removed."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = cleanup_output_module["cmd_module"]
        FakeJob = cleanup_output_module["FakeJob"]
        FakeCustodyEvent = cleanup_output_module["FakeCustodyEvent"]

        job_dir = tmp_path / "job-no-confirm"
        output_dir = job_dir / "output"
        pdf_dir = output_dir / "EXPORT" / "PDF"
        pdf_dir.mkdir(parents=True)
        (pdf_dir / "doc.pdf").write_bytes(b"content")

        old_date = datetime.now(timezone.utc) - timedelta(days=100)
        job = _make_job("job-no-confirm", status="completed", completed_at=old_date,
                        nfs_job_path=str(job_dir))
        FakeJob.objects._store = [job]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            retention_days=90,
            dry_run=False,
            confirm=False,
        )

        output = cmd.stdout.getvalue()
        assert "--confirm" in output
        # Files should still exist
        assert (pdf_dir / "doc.pdf").exists()
        assert len(FakeCustodyEvent._created) == 0


# ===========================================================================
# Tests: rotate_audit_logs -- archives records
# ===========================================================================


class TestRotateAuditLogsArchives:
    """rotate_audit_logs --confirm should archive and delete old records."""

    def test_archives_old_records(self, rotate_audit_module, monkeypatch, tmp_path):
        """Old CustodyEvent records are archived to JSONL and deleted."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = rotate_audit_module["cmd_module"]
        FakeJob = rotate_audit_module["FakeJob"]
        FakeCustodyEvent = rotate_audit_module["FakeCustodyEvent"]

        anchor_job = MagicMock()
        anchor_job.job_id = "job-1"
        FakeJob.objects._store = [anchor_job]

        old_date = datetime.now(timezone.utc) - timedelta(days=3000)
        event1 = _make_custody_event(
            1, "doc-1", "ocr_primary", old_date,
            data={"page": 1}, prev_hash="", event_hash="aaa111",
        )
        event2 = _make_custody_event(
            2, "doc-1", "assembly_complete", old_date + timedelta(seconds=30),
            data={"pages": 5}, prev_hash="aaa111", event_hash="bbb222",
        )
        FakeCustodyEvent.objects._store = [event1, event2]

        archive_dir = str(tmp_path / "archive")

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            retention_days=2555,
            archive_dir=archive_dir,
            dry_run=False,
            confirm=True,
        )

        output = cmd.stdout.getvalue()
        assert "Archived 2 records" in output
        assert "Deleted 2 records" in output

        # Archive file should exist
        archive_files = list((tmp_path / "archive").glob("*.jsonl"))
        assert len(archive_files) == 1

        # Verify archive contents
        with open(archive_files[0], "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["document_id"] == "doc-1"
        assert record["event_type"] == "ocr_primary"
        assert record["event_hash"] == "aaa111"


# ===========================================================================
# Tests: rotate_audit_logs -- dry-run
# ===========================================================================


class TestRotateAuditLogsDryRun:
    """rotate_audit_logs --dry-run should preview without archiving."""

    def test_dry_run_no_archival(self, rotate_audit_module, monkeypatch, tmp_path):
        """--dry-run should not create archives or delete records."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = rotate_audit_module["cmd_module"]
        FakeCustodyEvent = rotate_audit_module["FakeCustodyEvent"]

        old_date = datetime.now(timezone.utc) - timedelta(days=3000)
        event = _make_custody_event(1, "doc-1", "ocr_primary", old_date)
        FakeCustodyEvent.objects._store = [event]

        archive_dir = str(tmp_path / "archive")

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            retention_days=2555,
            archive_dir=archive_dir,
            dry_run=True,
            confirm=False,
        )

        output = cmd.stdout.getvalue()
        assert "DRY RUN" in output
        # No archive files created
        archive_path = tmp_path / "archive"
        if archive_path.exists():
            assert list(archive_path.glob("*.jsonl")) == []
        # Records still in store
        assert len(FakeCustodyEvent.objects._store) == 1
        # No custody events created
        assert len(FakeCustodyEvent._created) == 0


# ===========================================================================
# Tests: rotate_audit_logs -- CustodyEvent emitted
# ===========================================================================


class TestRotateAuditLogsCustodyEvent:
    """rotate_audit_logs should emit an audit_logs_rotated CustodyEvent."""

    def test_custody_event_emitted(self, rotate_audit_module, monkeypatch, tmp_path):
        """An audit_logs_rotated custody event should be created."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = rotate_audit_module["cmd_module"]
        FakeJob = rotate_audit_module["FakeJob"]
        FakeCustodyEvent = rotate_audit_module["FakeCustodyEvent"]

        anchor_job = MagicMock()
        anchor_job.job_id = "job-1"
        FakeJob.objects._store = [anchor_job]

        old_date = datetime.now(timezone.utc) - timedelta(days=3000)
        event = _make_custody_event(1, "doc-1", "ocr_primary", old_date)
        FakeCustodyEvent.objects._store = [event]

        archive_dir = str(tmp_path / "archive")

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            retention_days=2555,
            archive_dir=archive_dir,
            dry_run=False,
            confirm=True,
        )

        assert len(FakeCustodyEvent._created) == 1
        ce = FakeCustodyEvent._created[0]
        assert ce["event_type"] == "audit_logs_rotated"
        assert ce["data"]["action"] == "rotate_audit_logs"
        assert ce["data"]["records_archived"] == 1
        assert "archive_path" in ce["data"]
        assert "archive_checksum" in ce["data"]
        assert "operator" in ce["data"]
        assert "retention_policy" in ce["data"]


# ===========================================================================
# Tests: rotate_audit_logs -- archive integrity verification
# ===========================================================================


class TestRotateAuditLogsVerification:
    """rotate_audit_logs should verify archive integrity before deletion."""

    def test_valid_archive_passes_verification(self, rotate_audit_module, monkeypatch, tmp_path):
        """A properly written archive should pass verification."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = rotate_audit_module["cmd_module"]
        FakeJob = rotate_audit_module["FakeJob"]
        FakeCustodyEvent = rotate_audit_module["FakeCustodyEvent"]

        anchor_job = MagicMock()
        anchor_job.job_id = "job-1"
        FakeJob.objects._store = [anchor_job]

        old_date = datetime.now(timezone.utc) - timedelta(days=3000)
        event = _make_custody_event(
            1, "doc-1", "ocr_primary", old_date,
            data={"page": 1}, event_hash="abc123",
        )
        FakeCustodyEvent.objects._store = [event]

        archive_dir = str(tmp_path / "archive")

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            retention_days=2555,
            archive_dir=archive_dir,
            dry_run=False,
            confirm=True,
        )

        output = cmd.stdout.getvalue()
        # Should succeed (not show verification failure)
        assert "verification failed" not in output.lower()
        assert "Archived" in output

    def test_verify_archive_function(self, rotate_audit_module, tmp_path):
        """The _verify_archive function should validate JSONL format."""
        mod = rotate_audit_module["cmd_module"]

        # Write a valid JSONL file
        archive_path = str(tmp_path / "test.jsonl")
        with open(archive_path, "w") as f:
            f.write(json.dumps({"id": 1, "event_hash": "abc"}) + "\n")
            f.write(json.dumps({"id": 2, "event_hash": "def", "prev_hash": "abc"}) + "\n")

        is_valid, msg = mod._verify_archive(archive_path)
        assert is_valid
        assert "2 records" in msg

    def test_verify_archive_invalid_json(self, rotate_audit_module, tmp_path):
        """The _verify_archive function should reject invalid JSON."""
        mod = rotate_audit_module["cmd_module"]

        archive_path = str(tmp_path / "bad.jsonl")
        with open(archive_path, "w") as f:
            f.write("this is not json\n")

        is_valid, msg = mod._verify_archive(archive_path)
        assert not is_valid
        assert "Invalid JSON" in msg


# ===========================================================================
# Tests: rotate_audit_logs -- LITIGATION_HOLD
# ===========================================================================


class TestRotateAuditLogsLitigationHold:
    """LITIGATION_HOLD should block rotate_audit_logs."""

    def test_litigation_hold_blocks_rotation(self, rotate_audit_module, monkeypatch, tmp_path):
        """When LITIGATION_HOLD=true, rotation should not proceed."""
        monkeypatch.setenv("LITIGATION_HOLD", "true")

        mod = rotate_audit_module["cmd_module"]
        FakeCustodyEvent = rotate_audit_module["FakeCustodyEvent"]

        old_date = datetime.now(timezone.utc) - timedelta(days=3000)
        event = _make_custody_event(1, "doc-1", "ocr_primary", old_date)
        FakeCustodyEvent.objects._store = [event]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.stderr = StringIO()
        cmd.handle(
            retention_days=2555,
            archive_dir=str(tmp_path / "archive"),
            dry_run=False,
            confirm=True,
        )

        output = cmd.stderr.getvalue()
        assert "LITIGATION_HOLD" in output
        # Records should still exist
        assert len(FakeCustodyEvent.objects._store) == 1
        assert len(FakeCustodyEvent._created) == 0

    def test_litigation_hold_accepts_yes_and_one(self, rotate_audit_module, monkeypatch, tmp_path):
        """LITIGATION_HOLD should accept 'yes', '1', and 'true'."""
        mod = rotate_audit_module["cmd_module"]

        for value in ("yes", "1", "true", "TRUE", "True"):
            monkeypatch.setenv("LITIGATION_HOLD", value)
            cmd = mod.Command()
            cmd.stdout = StringIO()
            cmd.stderr = StringIO()
            cmd.handle(
                retention_days=2555,
                archive_dir=str(tmp_path / "archive"),
                dry_run=False,
                confirm=True,
            )
            output = cmd.stderr.getvalue()
            assert "LITIGATION_HOLD" in output


# ===========================================================================
# Tests: rotate_audit_logs -- confirm gate
# ===========================================================================


class TestRotateAuditLogsConfirmGate:
    """rotate_audit_logs without --confirm should not archive or delete."""

    def test_missing_confirm_blocks_rotation(self, rotate_audit_module, monkeypatch, tmp_path):
        """Without --confirm, no records should be archived or deleted."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = rotate_audit_module["cmd_module"]
        FakeCustodyEvent = rotate_audit_module["FakeCustodyEvent"]

        old_date = datetime.now(timezone.utc) - timedelta(days=3000)
        event = _make_custody_event(1, "doc-1", "ocr_primary", old_date)
        FakeCustodyEvent.objects._store = [event]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            retention_days=2555,
            archive_dir=str(tmp_path / "archive"),
            dry_run=False,
            confirm=False,
        )

        output = cmd.stdout.getvalue()
        assert "--confirm" in output
        # Records still in store
        assert len(FakeCustodyEvent.objects._store) == 1
        assert len(FakeCustodyEvent._created) == 0


# ===========================================================================
# Tests: rotate_audit_logs -- no old records
# ===========================================================================


class TestRotateAuditLogsNoRecords:
    """rotate_audit_logs with no old records should report gracefully."""

    def test_no_old_records(self, rotate_audit_module, monkeypatch, tmp_path):
        """When no records are old enough, a clear message is displayed."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = rotate_audit_module["cmd_module"]
        FakeCustodyEvent = rotate_audit_module["FakeCustodyEvent"]

        # Only recent events
        recent_date = datetime.now(timezone.utc) - timedelta(days=10)
        event = _make_custody_event(1, "doc-1", "ocr_primary", recent_date)
        FakeCustodyEvent.objects._store = [event]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            retention_days=2555,
            archive_dir=str(tmp_path / "archive"),
            dry_run=False,
            confirm=True,
        )

        output = cmd.stdout.getvalue()
        assert "No CustodyEvent records older than" in output


# ===========================================================================
# Tests: custody.py EVENT_TYPES registration
# ===========================================================================


class TestCustodyEventTypesRegistered:
    """Verify new event types are registered in custody.py EVENT_TYPES."""

    def test_output_cleaned_in_event_types(self):
        """The output_cleaned event type should be defined in custody.py."""
        from custody import EVENT_TYPES

        assert "output_cleaned" in EVENT_TYPES
        assert isinstance(EVENT_TYPES["output_cleaned"], str)
        assert len(EVENT_TYPES["output_cleaned"]) > 0

    def test_audit_logs_rotated_in_event_types(self):
        """The audit_logs_rotated event type should be defined in custody.py."""
        from custody import EVENT_TYPES

        assert "audit_logs_rotated" in EVENT_TYPES
        assert isinstance(EVENT_TYPES["audit_logs_rotated"], str)
        assert len(EVENT_TYPES["audit_logs_rotated"]) > 0


# ===========================================================================
# Tests: cleanup_output -- no output on disk
# ===========================================================================


class TestCleanupOutputNoFiles:
    """cleanup_output should handle jobs with no output files gracefully."""

    def test_no_output_on_disk(self, cleanup_output_module, monkeypatch, tmp_path):
        """Jobs with no output directory should produce a clear message."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = cleanup_output_module["cmd_module"]
        FakeJob = cleanup_output_module["FakeJob"]

        old_date = datetime.now(timezone.utc) - timedelta(days=100)
        job = _make_job("job-nofiles", status="completed", completed_at=old_date,
                        nfs_job_path=str(tmp_path / "nonexistent"))
        FakeJob.objects._store = [job]

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            retention_days=90,
            dry_run=False,
            confirm=True,
        )

        output = cmd.stdout.getvalue()
        assert "none have output files" in output


# ===========================================================================
# Tests: cleanup_output -- env var default
# ===========================================================================


class TestCleanupOutputEnvVar:
    """OUTPUT_RETENTION_DAYS env var should control default retention."""

    def test_env_var_override(self, cleanup_output_module, monkeypatch):
        """OUTPUT_RETENTION_DAYS env var should be readable by the command."""
        monkeypatch.setenv("OUTPUT_RETENTION_DAYS", "45")
        mod = cleanup_output_module["cmd_module"]
        # Re-import to pick up env var (function reads at call time)
        days = mod._get_retention_days()
        assert days == 45

    def test_env_var_invalid_fallback(self, cleanup_output_module, monkeypatch):
        """Invalid OUTPUT_RETENTION_DAYS should fall back to default."""
        monkeypatch.setenv("OUTPUT_RETENTION_DAYS", "not-a-number")
        mod = cleanup_output_module["cmd_module"]
        days = mod._get_retention_days()
        assert days == 90  # default


# ===========================================================================
# Tests: rotate_audit_logs -- env var default
# ===========================================================================


class TestRotateAuditLogsEnvVar:
    """AUDIT_LOG_RETENTION_DAYS env var should control default retention."""

    def test_env_var_override(self, rotate_audit_module, monkeypatch):
        """AUDIT_LOG_RETENTION_DAYS env var should be readable by the command."""
        monkeypatch.setenv("AUDIT_LOG_RETENTION_DAYS", "1825")
        mod = rotate_audit_module["cmd_module"]
        days = mod._get_retention_days()
        assert days == 1825

    def test_env_var_invalid_fallback(self, rotate_audit_module, monkeypatch):
        """Invalid AUDIT_LOG_RETENTION_DAYS should fall back to default."""
        monkeypatch.setenv("AUDIT_LOG_RETENTION_DAYS", "abc")
        mod = rotate_audit_module["cmd_module"]
        days = mod._get_retention_days()
        assert days == 2555  # default


# ===========================================================================
# Tests: rotate_audit_logs -- archive checksum
# ===========================================================================


class TestRotateAuditLogsChecksum:
    """rotate_audit_logs should compute and record archive checksum."""

    def test_checksum_in_custody_event(self, rotate_audit_module, monkeypatch, tmp_path):
        """Archive checksum should be present in the custody event data."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)

        mod = rotate_audit_module["cmd_module"]
        FakeJob = rotate_audit_module["FakeJob"]
        FakeCustodyEvent = rotate_audit_module["FakeCustodyEvent"]

        anchor_job = MagicMock()
        anchor_job.job_id = "job-1"
        FakeJob.objects._store = [anchor_job]

        old_date = datetime.now(timezone.utc) - timedelta(days=3000)
        event = _make_custody_event(1, "doc-1", "ocr_primary", old_date)
        FakeCustodyEvent.objects._store = [event]

        archive_dir = str(tmp_path / "archive")

        cmd = mod.Command()
        cmd.stdout = StringIO()
        cmd.handle(
            retention_days=2555,
            archive_dir=archive_dir,
            dry_run=False,
            confirm=True,
        )

        assert len(FakeCustodyEvent._created) == 1
        ce = FakeCustodyEvent._created[0]
        checksum = ce["data"]["archive_checksum"]
        # SHA-256 produces 64 hex characters
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)
