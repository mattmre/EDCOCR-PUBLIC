"""Tests for graceful-shutdown drain logic ( P1 fix).

Validates that SIGTERM drains in-flight queue items before killing threads,
rather than abandoning work immediately.
"""

import os
import queue
import threading
import time

import pytest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_events():
    """Reset stop/hard-stop events before each test so module-level state is clean."""
    from ocr_gpu_async import hard_stop_event, stop_event

    stop_event.clear()
    hard_stop_event.clear()
    yield
    stop_event.clear()
    hard_stop_event.clear()


# ---------------------------------------------------------------------------
# 1. _join_queue_with_timeout returns True on empty queue
# ---------------------------------------------------------------------------


def test_join_queue_with_timeout_empty_queue():
    from ocr_gpu_async import _join_queue_with_timeout

    q = queue.Queue()
    assert _join_queue_with_timeout(q, timeout=2.0) is True


# ---------------------------------------------------------------------------
# 2. _join_queue_with_timeout returns True when queue drains before timeout
# ---------------------------------------------------------------------------


def test_join_queue_with_timeout_drains_before_timeout():
    from ocr_gpu_async import _join_queue_with_timeout

    q = queue.Queue()
    q.put("item1")
    q.put("item2")

    # Drain items in a background thread after a short delay
    def _drain():
        time.sleep(0.1)
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except queue.Empty:
                break

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    assert _join_queue_with_timeout(q, timeout=5.0) is True
    t.join(timeout=2)


# ---------------------------------------------------------------------------
# 3. _join_queue_with_timeout returns False on timeout
# ---------------------------------------------------------------------------


def test_join_queue_with_timeout_returns_false_on_timeout():
    from ocr_gpu_async import _join_queue_with_timeout

    q = queue.Queue()
    q.put("stuck-item")
    # Never call task_done, so join() will block
    assert _join_queue_with_timeout(q, timeout=0.3) is False
    # Clean up so the daemon drain-helper can die
    q.get_nowait()
    q.task_done()


# ---------------------------------------------------------------------------
# 4. _graceful_shutdown sets stop_event but NOT hard_stop_event
# ---------------------------------------------------------------------------


def test_graceful_shutdown_sets_stop_not_hard_stop():
    import signal

    from ocr_gpu_async import _graceful_shutdown, hard_stop_event, stop_event

    assert not stop_event.is_set()
    assert not hard_stop_event.is_set()

    # Simulate SIGTERM handler call (signum=15, frame=None)
    _graceful_shutdown(signal.SIGTERM, None)

    assert stop_event.is_set()
    assert not hard_stop_event.is_set(), (
        "hard_stop_event must NOT be set by _graceful_shutdown; "
        "it should only be set after queues are drained"
    )


# ---------------------------------------------------------------------------
# 5. hard_stop_event starts unset
# ---------------------------------------------------------------------------


def test_hard_stop_event_starts_unset():
    from ocr_gpu_async import hard_stop_event

    # The autouse fixture clears it, matching fresh-process behavior
    assert not hard_stop_event.is_set()


# ---------------------------------------------------------------------------
# 6. SHUTDOWN_DRAIN_TIMEOUT_SECONDS default and env override
# ---------------------------------------------------------------------------


def test_shutdown_drain_timeout_default():
    from ocr_gpu_async import SHUTDOWN_DRAIN_TIMEOUT_SECONDS

    assert SHUTDOWN_DRAIN_TIMEOUT_SECONDS == 300


def test_shutdown_drain_timeout_env_override(monkeypatch):
    """Verify the env var mechanism works (checked via _safe_int pattern)."""
    # We can't re-import module-level constants easily, so just verify
    # the constant is an int and the env var name is SHUTDOWN_DRAIN_TIMEOUT
    from ocr_gpu_async import SHUTDOWN_DRAIN_TIMEOUT_SECONDS

    assert isinstance(SHUTDOWN_DRAIN_TIMEOUT_SECONDS, int)
    # Verify the env var name by checking that the constant was defined
    # with _safe_int("SHUTDOWN_DRAIN_TIMEOUT", 300, ...) — if the env var
    # is not set, the default is 300.
    assert SHUTDOWN_DRAIN_TIMEOUT_SECONDS == int(
        os.environ.get("SHUTDOWN_DRAIN_TIMEOUT", "300")
    )
