"""Track API sidecar/progress writers across responses and orderly shutdown."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

_lock = threading.Lock()
_pending: set[threading.Event] = set()


def active_count() -> int:
    with _lock:
        return len(_pending)


def register_writer() -> Callable[[], None]:
    """Reserve before scheduling; the returned callback releases exactly once."""
    done = threading.Event()
    with _lock:
        _pending.add(done)

    def finish():
        with _lock:
            _pending.discard(done)
            done.set()

    return finish


def start_writer(target: Callable, *, args=(), thread_factory=threading.Thread, name=None):
    finish = register_writer()

    def run():
        try:
            target(*args)
        finally:
            finish()

    try:
        kwargs = {"target": run, "daemon": True}
        if name is not None:
            kwargs["name"] = name
        worker = thread_factory(**kwargs)
        worker.start()
        return worker
    except BaseException:
        finish()
        raise


def wait_for_writers(timeout: float) -> bool:
    """Called after request admission and JobManager producers have stopped."""
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        with _lock:
            pending = tuple(_pending)
        if not pending:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not pending[0].wait(remaining):
            return False
