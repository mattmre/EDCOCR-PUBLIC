"""Tests for api/event_store.py — SQLite-backed durable event store."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from api.event_store import EventStore, get_event_store, reset_event_store


@pytest.fixture(autouse=True)
def _reset_store():
    """Ensure module singleton is clean between tests."""
    reset_event_store()
    yield
    reset_event_store()


@pytest.fixture()
def store(tmp_path):
    """Create an EventStore backed by a temporary database."""
    db_path = str(tmp_path / "events.db")
    return EventStore(db_path=db_path)


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


class TestStoreEvent:
    """Tests for store_event()."""

    def test_store_returns_event_id(self, store):
        """store_event returns a unique event ID string."""
        event_id = store.store_event("job.submitted", "job_aaaaaaaaaaaa", {"status": "submitted"})
        assert event_id.startswith("evt_")
        assert len(event_id) == 20  # "evt_" + 16 hex chars

    def test_store_multiple_events(self, store):
        """Multiple events can be stored for the same job."""
        id1 = store.store_event("job.submitted", "job_aaaaaaaaaaaa", {"status": "submitted"})
        id2 = store.store_event("job.processing", "job_aaaaaaaaaaaa", {"status": "processing"})
        id3 = store.store_event("job.completed", "job_aaaaaaaaaaaa", {"status": "completed"})

        assert id1 != id2 != id3

    def test_stored_event_is_retrievable(self, store):
        """A stored event appears in get_events_since results."""
        store.store_event("job.submitted", "job_aaaaaaaaaaaa", {"key": "value"})
        events = store.get_events_since("job_aaaaaaaaaaaa")

        assert len(events) == 1
        assert events[0]["event_type"] == "job.submitted"
        assert events[0]["job_id"] == "job_aaaaaaaaaaaa"
        assert events[0]["payload"] == {"key": "value"}
        assert events[0]["created_at"] is not None
        assert events[0]["delivered_at"] is None


class TestGetEventsSince:
    """Tests for get_events_since()."""

    def test_returns_all_events_for_job(self, store):
        """All events for a job are returned when no since_id."""
        for i in range(5):
            store.store_event(f"job.event_{i}", "job_aaaaaaaaaaaa", {"seq": i})

        events = store.get_events_since("job_aaaaaaaaaaaa")
        assert len(events) == 5

    def test_returns_events_after_since_id(self, store):
        """Only events after since_id are returned."""
        ids = []
        for i in range(5):
            ids.append(store.store_event(f"job.event_{i}", "job_aaaaaaaaaaaa", {"seq": i}))

        events = store.get_events_since("job_aaaaaaaaaaaa", since_id=ids[2])
        assert len(events) == 2
        assert events[0]["payload"]["seq"] == 3
        assert events[1]["payload"]["seq"] == 4

    def test_returns_empty_for_nonexistent_job(self, store):
        """No events for a job that does not exist."""
        events = store.get_events_since("job_doesnotexst")
        assert events == []

    def test_since_id_nonexistent_returns_all(self, store):
        """If since_id event does not exist, return all events (seq > 0)."""
        store.store_event("job.submitted", "job_aaaaaaaaaaaa", {"status": "submitted"})
        events = store.get_events_since("job_aaaaaaaaaaaa", since_id="evt_doesnotexist1234")
        # since_id not found -> since_seq=0, so all events with seq > 0 are returned
        assert len(events) == 1

    def test_respects_limit(self, store):
        """Limit parameter caps the number of returned events."""
        for i in range(10):
            store.store_event(f"event_{i}", "job_aaaaaaaaaaaa", {"seq": i})

        events = store.get_events_since("job_aaaaaaaaaaaa", limit=3)
        assert len(events) == 3
        # Should be the first 3 (ordered by seq ASC)
        assert events[0]["payload"]["seq"] == 0

    def test_events_from_different_jobs_isolated(self, store):
        """Events from different jobs are not mixed."""
        store.store_event("job.submitted", "job_aaaaaaaaaaaa", {"job": "A"})
        store.store_event("job.submitted", "job_bbbbbbbbbbbb", {"job": "B"})
        store.store_event("job.completed", "job_aaaaaaaaaaaa", {"job": "A"})

        events_a = store.get_events_since("job_aaaaaaaaaaaa")
        events_b = store.get_events_since("job_bbbbbbbbbbbb")

        assert len(events_a) == 2
        assert len(events_b) == 1

    def test_events_ordered_chronologically(self, store):
        """Events are returned in insertion order (by seq)."""
        for i in range(5):
            store.store_event(f"step_{i}", "job_aaaaaaaaaaaa", {"seq": i})

        events = store.get_events_since("job_aaaaaaaaaaaa")
        seqs = [e["payload"]["seq"] for e in events]
        assert seqs == [0, 1, 2, 3, 4]


class TestMarkDelivered:
    """Tests for mark_delivered()."""

    def test_marks_event_as_delivered(self, store):
        """mark_delivered sets the delivered_at timestamp."""
        event_id = store.store_event("job.submitted", "job_aaaaaaaaaaaa", {})

        assert store.mark_delivered(event_id) is True

        events = store.get_events_since("job_aaaaaaaaaaaa")
        assert events[0]["delivered_at"] is not None

    def test_returns_false_for_nonexistent(self, store):
        """mark_delivered returns False for nonexistent event ID."""
        assert store.mark_delivered("evt_doesnotexist1234") is False

    def test_idempotent_mark(self, store):
        """Marking an already-delivered event returns False (no update)."""
        event_id = store.store_event("job.submitted", "job_aaaaaaaaaaaa", {})
        store.mark_delivered(event_id)

        # Second call returns False (already delivered)
        assert store.mark_delivered(event_id) is False


class TestGetUndelivered:
    """Tests for get_undelivered()."""

    def test_returns_undelivered_events(self, store):
        """Undelivered events within the age window are returned."""
        store.store_event("job.submitted", "job_aaaaaaaaaaaa", {"status": "submitted"})
        store.store_event("job.completed", "job_aaaaaaaaaaaa", {"status": "completed"})

        undelivered = store.get_undelivered(max_age_hours=1)
        assert len(undelivered) == 2

    def test_excludes_delivered_events(self, store):
        """Delivered events are not included."""
        id1 = store.store_event("job.submitted", "job_aaaaaaaaaaaa", {})
        store.store_event("job.completed", "job_aaaaaaaaaaaa", {})

        store.mark_delivered(id1)

        undelivered = store.get_undelivered(max_age_hours=1)
        assert len(undelivered) == 1
        assert undelivered[0]["event_type"] == "job.completed"


class TestCleanup:
    """Tests for cleanup()."""

    def test_cleanup_removes_old_events(self, store):
        """Events older than the threshold are deleted."""
        store.store_event("old", "job_aaaaaaaaaaaa", {})

        # Cleanup with a 0-hour window should delete everything
        deleted = store.cleanup(older_than_hours=0)
        assert deleted == 1

        events = store.get_events_since("job_aaaaaaaaaaaa")
        assert len(events) == 0

    def test_cleanup_preserves_recent_events(self, store):
        """Recent events within the threshold are preserved."""
        store.store_event("recent", "job_aaaaaaaaaaaa", {})

        # Cleanup with a large window should delete nothing
        deleted = store.cleanup(older_than_hours=999)
        assert deleted == 0

        events = store.get_events_since("job_aaaaaaaaaaaa")
        assert len(events) == 1


class TestCloseAndReopen:
    """Tests for close() and connection reopen."""

    def test_close_and_reopen(self, tmp_path):
        """Store can be closed and reopened."""
        db_path = str(tmp_path / "events.db")
        store = EventStore(db_path=db_path)
        store.store_event("job.submitted", "job_aaaaaaaaaaaa", {"v": 1})
        store.close()

        # Reopen with same path
        store2 = EventStore(db_path=db_path)
        events = store2.get_events_since("job_aaaaaaaaaaaa")
        assert len(events) == 1
        assert events[0]["payload"]["v"] == 1
        store2.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    """Tests for get_event_store() and reset_event_store()."""

    def test_get_returns_same_instance(self, tmp_path):
        """get_event_store() returns the same instance on repeated calls."""
        db_path = str(tmp_path / "singleton.db")
        with patch("api.event_store._DEFAULT_DB_PATH", db_path):
            s1 = get_event_store(db_path=db_path)
            s2 = get_event_store(db_path=db_path)
            assert s1 is s2

    def test_reset_clears_singleton(self, tmp_path):
        """reset_event_store() allows creating a new instance."""
        db_path = str(tmp_path / "singleton.db")
        s1 = get_event_store(db_path=db_path)
        reset_event_store()
        s2 = get_event_store(db_path=db_path)
        assert s1 is not s2
