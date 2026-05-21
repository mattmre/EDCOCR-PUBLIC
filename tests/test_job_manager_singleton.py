"""Tests for the application-wide JobManager singleton.

Verifies that ``get_manager()`` returns the same ``JobManager`` instance
across sequential and concurrent callers, preventing per-request thread
and queue leaks.
"""

from __future__ import annotations

import threading

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_get_manager_returns_job_manager_instance():
    """get_manager() should return a JobManager instance."""
    from api.job_manager import JobManager, get_manager

    mgr = get_manager()
    assert isinstance(mgr, JobManager)


def test_get_manager_returns_same_instance():
    """Consecutive calls to get_manager() must return the same object."""
    from api.job_manager import get_manager

    mgr1 = get_manager()
    mgr2 = get_manager()
    assert mgr1 is mgr2


def test_get_manager_thread_safety():
    """Concurrent calls from multiple threads must all receive the same instance."""
    from api.job_manager import get_manager

    results: list = [None] * 8
    barrier = threading.Barrier(8)

    def _worker(index: int) -> None:
        barrier.wait()  # synchronize so all threads call get_manager() concurrently
        results[index] = get_manager()

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    # All threads must have received the exact same instance.
    assert all(r is results[0] for r in results)


def test_singleton_reset_allows_new_instance():
    """After clearing _manager_instance, get_manager() creates a fresh one."""
    import api.job_manager as mod
    from api.job_manager import get_manager

    first = get_manager()
    mod._manager_instance = None
    second = get_manager()

    assert first is not second
