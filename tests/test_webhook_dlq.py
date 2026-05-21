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
        assert entry["retried_at"] is None

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
