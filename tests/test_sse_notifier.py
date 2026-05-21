"""Tests for api.sse_notifier.

Verifies the in-process asyncio.Event notification hub used by SSE streams.
"""

import asyncio
import threading
import time

import pytest

from api.sse_notifier import (
    _events,
    _lock,
    create_job_event,
    notify_job_update,
    remove_job_event,
)


@pytest.fixture(autouse=True)
def _clean_events():
    """Ensure the global event registry is empty before/after each test."""
    with _lock:
        _events.clear()
    yield
    with _lock:
        _events.clear()


def _registered_events(job_id: str):
    """Return the set of asyncio.Event objects registered for *job_id*."""
    with _lock:
        return {pair[0] for pair in _events.get(job_id, set())}


# ------------------------------------------------------------------
# Async unit tests (need running event loop for create_job_event)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_job_event_registers():
    """create_job_event adds an asyncio.Event to the registry."""
    evt = create_job_event("job-1")
    assert isinstance(evt, asyncio.Event)
    assert evt in _registered_events("job-1")


@pytest.mark.asyncio
async def test_remove_event_cleans_up():
    """remove_job_event discards the event and deletes the key when empty."""
    evt = create_job_event("job-2")
    remove_job_event("job-2", evt)
    with _lock:
        assert "job-2" not in _events


@pytest.mark.asyncio
async def test_remove_event_keeps_others():
    """Removing one event leaves other events for the same job intact."""
    evt1 = create_job_event("job-3")
    evt2 = create_job_event("job-3")
    remove_job_event("job-3", evt1)
    registered = _registered_events("job-3")
    assert evt2 in registered
    assert evt1 not in registered


@pytest.mark.asyncio
async def test_remove_event_idempotent():
    """Removing a non-existent event or job is a no-op."""
    evt = create_job_event("job-4")
    remove_job_event("job-4", evt)
    # Second remove should be harmless
    remove_job_event("job-4", evt)
    # Remove for unknown job
    remove_job_event("nonexistent", evt)


@pytest.mark.asyncio
async def test_notify_sets_event():
    """notify_job_update sets the asyncio.Event so waiters wake up."""
    evt = create_job_event("job-5")
    assert not evt.is_set()
    notify_job_update("job-5")
    # call_soon_threadsafe schedules on the current loop; yield control
    await asyncio.sleep(0)
    assert evt.is_set()


@pytest.mark.asyncio
async def test_multiple_streams_all_notified():
    """All events registered for a job are set on a single notify call."""
    evts = [create_job_event("job-6") for _ in range(5)]
    assert all(not e.is_set() for e in evts)
    notify_job_update("job-6")
    await asyncio.sleep(0)
    assert all(e.is_set() for e in evts)


@pytest.mark.asyncio
async def test_notify_unknown_job_is_noop():
    """Notifying a job with no registered events does nothing."""
    # Should not raise
    notify_job_update("nonexistent-job")


# ------------------------------------------------------------------
# Async integration tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notify_wakes_waiting_event():
    """An awaiting coroutine wakes immediately when notify fires from a thread."""
    evt = create_job_event("job-async-1")

    async def _waiter():
        await asyncio.wait_for(evt.wait(), timeout=5.0)

    # Schedule the notify after a short delay from a background thread
    # (simulating the sync job_manager path).
    def _delayed_notify():
        time.sleep(0.05)
        notify_job_update("job-async-1")

    t = threading.Thread(target=_delayed_notify, daemon=True)
    t.start()

    start = time.monotonic()
    await _waiter()
    elapsed = time.monotonic() - start

    # Should complete well under 1 second (the notify fires at ~50ms).
    assert elapsed < 1.0
    t.join(timeout=2)


@pytest.mark.asyncio
async def test_fallback_timeout_fires():
    """When no notification arrives, asyncio.wait_for times out gracefully."""
    evt = create_job_event("job-timeout-1")
    start = time.monotonic()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            asyncio.shield(evt.wait()), timeout=0.1,
        )

    elapsed = time.monotonic() - start
    # Should have waited approximately 0.1s (not much longer).
    assert 0.08 < elapsed < 2.0


@pytest.mark.asyncio
async def test_event_clear_allows_reuse():
    """After being set and cleared, an event can wait again."""
    evt = create_job_event("job-reuse-1")
    notify_job_update("job-reuse-1")
    await asyncio.sleep(0)
    assert evt.is_set()
    evt.clear()
    assert not evt.is_set()

    # Notify again
    notify_job_update("job-reuse-1")
    await asyncio.sleep(0)
    assert evt.is_set()


# ------------------------------------------------------------------
# Thread-safety smoke test
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_create_remove():
    """Concurrent create/remove from multiple threads does not corrupt state."""
    job_id = "job-concurrent"
    errors = []
    loop = asyncio.get_event_loop()

    def _worker(idx: int):
        try:
            # create_job_event needs asyncio.get_event_loop(); set it for
            # this thread so it returns the main loop.
            asyncio.set_event_loop(loop)
            evt = create_job_event(job_id)
            notify_job_update(job_id)
            remove_job_event(job_id, evt)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors
    # After all threads finish, the registry should be empty for this job
    with _lock:
        assert job_id not in _events
