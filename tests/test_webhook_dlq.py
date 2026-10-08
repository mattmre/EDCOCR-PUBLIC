"""Tests for api/webhook_dlq.py — Webhook dead-letter queue operations."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from api import webhook_dlq


@pytest.fixture()
def dlq_file(tmp_path):
    """Return a temporary DLQ file path."""
    return tmp_path / "dlq.jsonl"


# ---------------------------------------------------------------------------
# add_to_dlq
# ---------------------------------------------------------------------------


class TestAddToDlq:
    """Tests for add_to_dlq()."""

    def test_adds_entry_to_file(self, dlq_file):
        """Adding an entry creates a JSONL line in the DLQ file."""
        entry_id = webhook_dlq.add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://example.com/hook",
            event_type="job.completed",
            payload={"event": "job.completed", "job_id": "job_aaaaaaaaaaaa"},
            last_error="HTTP 500",
            attempts=4,
            dlq_file=dlq_file,
        )

        assert entry_id.startswith("dlq_")
        assert dlq_file.exists()

        lines = dlq_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["id"] == entry_id
        assert entry["job_id"] == "job_aaaaaaaaaaaa"
        assert entry["webhook_url"] == "https://example.com/hook"
        assert entry["event_type"] == "job.completed"
        assert entry["last_error"] == "HTTP 500"
        assert entry["attempts"] == 4
        assert entry["tenant_id"] is None
        assert entry["retried_at"] is None

    def test_stamps_tenant_id_when_provided(self, dlq_file):
        """tenant_id is written onto the JSONL entry when known (AEP BE003)."""
        entry_id = webhook_dlq.add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://example.com/hook",
            event_type="job.completed",
            payload={},
            last_error="HTTP 500",
            attempts=4,
            tenant_id="tenant_aaaaaaaaaaaa",
            dlq_file=dlq_file,
        )

        entry = json.loads(dlq_file.read_text(encoding="utf-8").strip())
        assert entry["id"] == entry_id
        assert entry["tenant_id"] == "tenant_aaaaaaaaaaaa"

    def test_empty_tenant_id_stamped_as_null(self, dlq_file):
        """Blank tenant_id is normalized to null on the entry."""
        webhook_dlq.add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://example.com/hook",
            event_type="job.completed",
            payload={},
            last_error="error",
            attempts=1,
            tenant_id="",
            dlq_file=dlq_file,
        )
        entry = json.loads(dlq_file.read_text(encoding="utf-8").strip())
        assert entry["tenant_id"] is None

    def test_appends_multiple_entries(self, dlq_file):
        """Multiple entries are appended to the same file."""
        webhook_dlq.add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://a.example.com",
            event_type="job.completed",
            payload={},
            last_error="error1",
            attempts=1,
            dlq_file=dlq_file,
        )
        webhook_dlq.add_to_dlq(
            job_id="job_bbbbbbbbbbbb",
            webhook_url="https://b.example.com",
            event_type="job.failed",
            payload={},
            last_error="error2",
            attempts=2,
            dlq_file=dlq_file,
        )

        lines = dlq_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_disabled_returns_empty_string(self, dlq_file):
        """When DLQ is disabled, add_to_dlq returns empty string."""
        with patch.object(webhook_dlq, "_config_value", return_value=False):
            entry_id = webhook_dlq.add_to_dlq(
                job_id="job_aaaaaaaaaaaa",
                webhook_url="https://example.com",
                event_type="job.completed",
                payload={},
                last_error="error",
                attempts=1,
                dlq_file=dlq_file,
            )

        assert entry_id == ""
        assert not dlq_file.exists()


# ---------------------------------------------------------------------------
# list_dlq
# ---------------------------------------------------------------------------


class TestListDlq:
    """Tests for list_dlq()."""

    def test_empty_when_no_file(self, dlq_file):
        """Returns empty list when DLQ file does not exist."""
        entries = webhook_dlq.list_dlq(dlq_file=dlq_file)
        assert entries == []

    def test_returns_entries_reverse_chronological(self, dlq_file):
        """Returns entries most recent first."""
        webhook_dlq.add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://a.example.com",
            event_type="job.completed",
            payload={"seq": 1},
            last_error="e1",
            attempts=1,
            dlq_file=dlq_file,
        )
        webhook_dlq.add_to_dlq(
            job_id="job_bbbbbbbbbbbb",
            webhook_url="https://b.example.com",
            event_type="job.failed",
            payload={"seq": 2},
            last_error="e2",
            attempts=2,
            dlq_file=dlq_file,
        )

        entries = webhook_dlq.list_dlq(dlq_file=dlq_file)
        assert len(entries) == 2
        # Most recent first
        assert entries[0]["job_id"] == "job_bbbbbbbbbbbb"
        assert entries[1]["job_id"] == "job_aaaaaaaaaaaa"

    def test_respects_limit(self, dlq_file):
        """Limit parameter caps the number of returned entries."""
        for i in range(5):
            webhook_dlq.add_to_dlq(
                job_id=f"job_{i:012x}",
                webhook_url="https://example.com",
                event_type="job.completed",
                payload={},
                last_error="error",
                attempts=1,
                dlq_file=dlq_file,
            )

        entries = webhook_dlq.list_dlq(limit=2, dlq_file=dlq_file)
        assert len(entries) == 2

    def test_skips_malformed_lines(self, dlq_file):
        """Malformed JSONL lines are skipped without error."""
        dlq_file.write_text("not valid json\n", encoding="utf-8")
        webhook_dlq.add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://example.com",
            event_type="job.completed",
            payload={},
            last_error="error",
            attempts=1,
            dlq_file=dlq_file,
        )

        entries = webhook_dlq.list_dlq(dlq_file=dlq_file)
        assert len(entries) == 1

    def test_skips_oversized_line_and_keeps_following_entry(
        self, dlq_file, monkeypatch
    ):
        """A hostile giant JSONL record cannot hide later bounded entries."""
        monkeypatch.setattr(webhook_dlq, "_DLQ_MAX_LINE_BYTES", 128)
        dlq_file.write_bytes(b"x" * 256 + b"\n{" + b'"id":"keep"' + b"}\n")

        entries = webhook_dlq.list_dlq(dlq_file=dlq_file)

        assert [entry["id"] for entry in entries] == ["keep"]

    def test_filters_by_stamped_tenant(self, dlq_file):
        """list_dlq(tenant_id=...) returns only that tenant's stamped entries."""
        webhook_dlq.add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://a.example.com",
            event_type="job.completed",
            payload={},
            last_error="e1",
            attempts=1,
            tenant_id="tenant_aaaaaaaaaaaa",
            dlq_file=dlq_file,
        )
        webhook_dlq.add_to_dlq(
            job_id="job_bbbbbbbbbbbb",
            webhook_url="https://b.example.com",
            event_type="job.failed",
            payload={},
            last_error="e2",
            attempts=1,
            tenant_id="tenant_bbbbbbbbbbbb",
            dlq_file=dlq_file,
        )

        a_entries = webhook_dlq.list_dlq(
            tenant_id="tenant_aaaaaaaaaaaa", dlq_file=dlq_file
        )
        assert len(a_entries) == 1
        assert a_entries[0]["job_id"] == "job_aaaaaaaaaaaa"

        b_entries = webhook_dlq.list_dlq(
            tenant_id="tenant_bbbbbbbbbbbb", dlq_file=dlq_file
        )
        assert len(b_entries) == 1
        assert b_entries[0]["job_id"] == "job_bbbbbbbbbbbb"

    def test_legacy_unstamped_resolved_via_lookup(self, dlq_file):
        """Unstamped legacy entries are owned by the resolved job tenant."""
        # Write a legacy line without tenant_id key
        legacy = {
            "id": "dlq_abc0000000000001",
            "job_id": "job_aaaaaaaaaaaa",
            "webhook_url": "https://example.com",
            "event_type": "job.completed",
            "payload": {},
            "last_error": "old",
            "attempts": 3,
            "created_at": "2026-01-01T00:00:00+00:00",
            "retried_at": None,
        }
        dlq_file.write_text(
            json.dumps(legacy, separators=(",", ":")) + "\n", encoding="utf-8"
        )

        def _lookup(resource_id: str):
            return {
                "job_aaaaaaaaaaaa": "tenant_aaaaaaaaaaaa",
            }.get(resource_id)

        a = webhook_dlq.list_dlq(
            tenant_id="tenant_aaaaaaaaaaaa",
            dlq_file=dlq_file,
            lookup=_lookup,
        )
        assert len(a) == 1
        assert a[0]["id"] == "dlq_abc0000000000001"

        b = webhook_dlq.list_dlq(
            tenant_id="tenant_bbbbbbbbbbbb",
            dlq_file=dlq_file,
            lookup=_lookup,
        )
        assert b == []

    def test_legacy_unresolvable_hidden_from_tenant(self, dlq_file):
        """Unstamped unresolvable entries are admin-only (hidden from tenant keys)."""
        legacy = {
            "id": "dlq_abc0000000000002",
            "job_id": "job_deadbeefdead",
            "webhook_url": "https://example.com",
            "event_type": "job.completed",
            "payload": {"secret": "payload"},
            "last_error": "gone",
            "attempts": 4,
            "created_at": "2026-01-01T00:00:00+00:00",
            "retried_at": None,
        }
        dlq_file.write_text(
            json.dumps(legacy, separators=(",", ":")) + "\n", encoding="utf-8"
        )

        tenant_view = webhook_dlq.list_dlq(
            tenant_id="tenant_aaaaaaaaaaaa",
            dlq_file=dlq_file,
            lookup=lambda _rid: None,
        )
        assert tenant_view == []

        # Unscoped (single-tenant / platform admin) still sees the entry
        admin_view = webhook_dlq.list_dlq(dlq_file=dlq_file)
        assert len(admin_view) == 1


# ---------------------------------------------------------------------------
# ownership helpers
# ---------------------------------------------------------------------------


class TestEntryOwnership:
    """Unit tests for resolve_entry_tenant_id / entry_owned_by_tenant."""

    def test_stamped_entry_owned(self):
        entry = {"tenant_id": "tenant_aaaaaaaaaaaa", "job_id": "job_aaaaaaaaaaaa"}
        assert webhook_dlq.entry_owned_by_tenant(entry, "tenant_aaaaaaaaaaaa")
        assert not webhook_dlq.entry_owned_by_tenant(entry, "tenant_bbbbbbbbbbbb")

    def test_legacy_resolved_owned(self):
        entry = {"job_id": "job_aaaaaaaaaaaa"}  # no tenant_id stamp
        assert webhook_dlq.entry_owned_by_tenant(
            entry,
            "tenant_aaaaaaaaaaaa",
            lookup=lambda rid: "tenant_aaaaaaaaaaaa"
            if rid == "job_aaaaaaaaaaaa"
            else None,
        )
        assert not webhook_dlq.entry_owned_by_tenant(
            entry,
            "tenant_bbbbbbbbbbbb",
            lookup=lambda rid: "tenant_aaaaaaaaaaaa"
            if rid == "job_aaaaaaaaaaaa"
            else None,
        )

    def test_unresolvable_not_owned(self):
        entry = {"job_id": "job_missing00000"}
        assert not webhook_dlq.entry_owned_by_tenant(
            entry, "tenant_aaaaaaaaaaaa", lookup=lambda _rid: None
        )

    def test_resolve_prefers_stamp_over_lookup(self):
        entry = {
            "tenant_id": "tenant_aaaaaaaaaaaa",
            "job_id": "job_bbbbbbbbbbbb",
        }
        # lookup would claim B — stamp wins
        resolved = webhook_dlq.resolve_entry_tenant_id(
            entry, lookup=lambda _rid: "tenant_bbbbbbbbbbbb"
        )
        assert resolved == "tenant_aaaaaaaaaaaa"


# ---------------------------------------------------------------------------
# get_dlq_entry
# ---------------------------------------------------------------------------


class TestGetDlqEntry:
    """Tests for get_dlq_entry()."""

    def test_returns_entry_by_id(self, dlq_file):
        """get_dlq_entry finds entry by ID."""
        entry_id = webhook_dlq.add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://example.com",
            event_type="job.completed",
            payload={"key": "value"},
            last_error="error",
            attempts=3,
            dlq_file=dlq_file,
        )

        entry = webhook_dlq.get_dlq_entry(entry_id, dlq_file=dlq_file)
        assert entry is not None
        assert entry["id"] == entry_id
        assert entry["payload"]["key"] == "value"

    def test_returns_none_for_nonexistent(self, dlq_file):
        """Returns None when entry ID is not found."""
        entry = webhook_dlq.get_dlq_entry("dlq_doesnotexist1234", dlq_file=dlq_file)
        assert entry is None

    def test_returns_none_when_no_file(self, dlq_file):
        """Returns None when DLQ file does not exist."""
        entry = webhook_dlq.get_dlq_entry("dlq_doesnotexist1234", dlq_file=dlq_file)
        assert entry is None


# ---------------------------------------------------------------------------
# mark_dlq_retried
# ---------------------------------------------------------------------------


class TestMarkDlqRetried:
    """Tests for mark_dlq_retried()."""

    def test_marks_entry_retried(self, dlq_file):
        """mark_dlq_retried sets retried_at on the matching entry."""
        entry_id = webhook_dlq.add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://example.com",
            event_type="job.completed",
            payload={},
            last_error="error",
            attempts=3,
            dlq_file=dlq_file,
        )

        result = webhook_dlq.mark_dlq_retried(entry_id, dlq_file=dlq_file)
        assert result is True

        # Verify the entry was updated
        entry = webhook_dlq.get_dlq_entry(entry_id, dlq_file=dlq_file)
        assert entry["retried_at"] is not None

    def test_returns_false_for_nonexistent(self, dlq_file):
        """Returns False when entry ID is not found."""
        result = webhook_dlq.mark_dlq_retried("dlq_doesnotexist1234", dlq_file=dlq_file)
        assert result is False

    def test_returns_false_when_no_file(self, dlq_file):
        """Returns False when DLQ file does not exist."""
        result = webhook_dlq.mark_dlq_retried("dlq_doesnotexist1234", dlq_file=dlq_file)
        assert result is False

    def test_preserves_other_entries(self, dlq_file):
        """Marking one entry does not modify other entries."""
        id1 = webhook_dlq.add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://a.example.com",
            event_type="job.completed",
            payload={"seq": 1},
            last_error="e1",
            attempts=1,
            dlq_file=dlq_file,
        )
        id2 = webhook_dlq.add_to_dlq(
            job_id="job_bbbbbbbbbbbb",
            webhook_url="https://b.example.com",
            event_type="job.failed",
            payload={"seq": 2},
            last_error="e2",
            attempts=2,
            dlq_file=dlq_file,
        )

        webhook_dlq.mark_dlq_retried(id1, dlq_file=dlq_file)

        entry1 = webhook_dlq.get_dlq_entry(id1, dlq_file=dlq_file)
        entry2 = webhook_dlq.get_dlq_entry(id2, dlq_file=dlq_file)

        assert entry1["retried_at"] is not None
        assert entry2["retried_at"] is None


class TestDlqRetryClaims:
    def test_claim_is_exclusive_and_release_is_reusable(self, dlq_file):
        entry_id = webhook_dlq.add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://example.com",
            event_type="job.completed",
            payload={"ok": True},
            last_error="error",
            attempts=3,
            dlq_file=dlq_file,
        )

        first = webhook_dlq.claim_dlq_retry(entry_id, dlq_file=dlq_file)
        assert first
        assert webhook_dlq.claim_dlq_retry(entry_id, dlq_file=dlq_file) is None
        assert webhook_dlq.release_dlq_retry(entry_id, first, dlq_file=dlq_file) is True
        assert webhook_dlq.claim_dlq_retry(entry_id, dlq_file=dlq_file)

    def test_atomic_mark_preserves_readable_entry(self, dlq_file):
        entry_id = webhook_dlq.add_to_dlq(
            job_id="job_aaaaaaaaaaaa",
            webhook_url="https://example.com",
            event_type="job.completed",
            payload={"ok": True},
            last_error="error",
            attempts=3,
            dlq_file=dlq_file,
        )

        assert webhook_dlq.mark_dlq_retried(entry_id, dlq_file=dlq_file) is True
        entry = webhook_dlq.get_dlq_entry(entry_id, dlq_file=dlq_file)
        assert entry is not None
        assert entry["retried_at"]
