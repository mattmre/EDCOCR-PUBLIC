"""Bounded, process-local runtime for stateful OCR predictors.

Paddle predictors can be sensitive to thread migration.  Each replica is
constructed, invoked, and closed by one dedicated owner thread.  Callers may
submit from any thread; the default of one replica per key serializes access
to a predictor.  The pool does not construct an engine until the first call.

Create this pool lazily in the worker process.  A forked child discards the
parent's pool bookkeeping and predictors without touching inherited CUDA
state; its first call creates a fresh predictor in the child.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from .native_watchdog import NativeCallWatchdog

T = TypeVar("T")
_STOP = object()
_owner_context = threading.local()
logger = logging.getLogger(__name__)


def production_native_call_timeout() -> int:
    """Require a bounded deadline for every production predictor owner."""
    raw = os.environ.get("OCR_NATIVE_CALL_TIMEOUT_SECONDS", "300")
    try:
        seconds = int(raw)
    except ValueError as exc:
        raise ValueError("OCR_NATIVE_CALL_TIMEOUT_SECONDS must be an integer") from exc
    if not 10 <= seconds <= 3600:
        raise ValueError("OCR_NATIVE_CALL_TIMEOUT_SECONDS must be between 10 and 3600")
    return seconds


def _clear_finished_traceback_frames(exc: BaseException) -> None:
    """Drop predictor refs from unwound native frames before guarded disposal."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        traceback = current.__traceback__
        while traceback is not None:
            try:
                traceback.tb_frame.clear()
            except RuntimeError:
                # The current owner frame remains live; it drops `engine`
                # explicitly before leaving the watchdog guard.
                pass
            traceback = traceback.tb_next
        current = current.__cause__ or current.__context__


def _record_construction(key: EngineKey) -> None:
    """Append one actual predictor construction to an optional lab event file.

    The path is configured only in an isolated runtime lab. A single append
    under flock keeps records intact across prefork Celery children.
    """
    path = os.environ.get("OCR_ENGINE_DIAGNOSTICS_PATH", "").strip()
    if not path:
        return
    import fcntl

    record = {
        "event": "constructed", "pid": os.getpid(), "owner_thread": threading.get_ident(),
        "device": key.device, "language": key.language, "model_family": key.model_family,
        "backend": key.backend,
        "runtime_role": os.environ.get("OCR_ENGINE_DIAGNOSTICS_ROLE", "unspecified"),
        "config_sha256": hashlib.sha256(repr(key.config_identity).encode()).hexdigest(),
        "monotonic_ns": time.monotonic_ns(),
    }
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, payload)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class EngineKey:
    """Identity of the *actual* predictor configuration, not just language.

    ``config_identity`` should be a stable digest or immutable tuple covering
    model paths/versions and every constructor option that changes inference.
    Device must identify the actual device (for example ``gpu:0``), not just
    whether a GPU is available.
    """

    device: str
    language: str
    model_family: str
    backend: str
    config_identity: Hashable

    def __post_init__(self) -> None:
        for name in ("device", "language", "model_family", "backend"):
            if not getattr(self, name):
                raise ValueError(f"EngineKey.{name} must be nonempty")
        hash(self.config_identity)


@dataclass
class _Work:
    factory: Callable[[], Any]
    operation: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    result: Future[Any]


class _Replica:
    def __init__(self, pool: EngineRuntimePool, key: EngineKey) -> None:
        self.pool = pool
        self.key = key
        self.pending = 0  # Read/write only while pool._condition is held.
        self.work: queue.SimpleQueue[_Work | object] = queue.SimpleQueue()
        self.thread = threading.Thread(
            target=self._serve,
            name=f"ocr-engine-{key.language}-{key.device}",
            daemon=True,
        )
        self.thread.start()

    def _serve(self) -> None:
        _owner_context.pool = self.pool
        engine: Any = None
        try:
            while True:
                item = self.work.get()
                if item is _STOP:
                    break
                assert isinstance(item, _Work)
                try:
                    if engine is None:
                        with self.pool._native_guard():
                            engine = item.factory()
                        if engine is None:
                            raise RuntimeError("engine factory returned None")
                        try:
                            _record_construction(self.key)
                        except Exception:
                            logger.exception("Could not record OCR engine construction")
                    with self.pool._native_guard():
                        value = item.operation(engine, *item.args, **item.kwargs)
                    item.result.set_result(value)
                except BaseException as exc:
                    # An inference failure may leave native predictor state
                    # unusable. Retire it on its owner before the next call.
                    if engine is not None:
                        try:
                            with self.pool._native_guard():
                                _clear_finished_traceback_frames(exc)
                                try:
                                    self.pool._close_engine(engine)
                                except BaseException as close_exc:
                                    _clear_finished_traceback_frames(close_exc)
                                    raise
                                finally:
                                    engine = None
                        except Exception:
                            logger.exception("Could not close failed OCR engine")
                    item.result.set_exception(exc)
                finally:
                    with self.pool._condition:
                        self.pending -= 1
                        self.pool._condition.notify_all()
        finally:
            try:
                if engine is not None:
                    with self.pool._native_guard():
                        try:
                            self.pool._close_engine(engine)
                        except BaseException as close_exc:
                            _clear_finished_traceback_frames(close_exc)
                            raise
                        finally:
                            engine = None
            finally:
                _owner_context.pool = None

    def stop(self) -> None:
        self.work.put(_STOP)

    def join(self) -> None:
        self.thread.join()


def _default_close(engine: Any) -> None:
    close = getattr(engine, "close", None)
    if callable(close):
        close()


class EngineRuntimePool:
    """Reuse a bounded number of predictors without migrating their threads.

    ``max_entries`` bounds distinct keys.  ``max_replicas_per_key`` bounds
    predictors per key; ``max_replicas_per_device`` bounds all predictor keys
    on one actual device (including GPU index or MIG assignment). Leave it at
    one for CPU Paddle unless the installed
    engine is known to support separate concurrent instances.  An LRU entry
    can be retired only after all of its submitted calls have completed.
    Closing waits for accepted calls and invokes each engine's ``close`` on
    its owner thread when such a method exists.
    """

    def __init__(
        self,
        *,
        max_entries: int = 4,
        max_replicas_per_key: int = 1,
        max_replicas_per_device: int = 1,
        max_pending_per_replica: int = 8,
        close_engine: Callable[[Any], None] | None = None,
        native_call_timeout_seconds: float | None = None,
    ) -> None:
        if min(max_entries, max_replicas_per_key, max_replicas_per_device, max_pending_per_replica) < 1:
            raise ValueError("pool limits must be positive")
        self.max_entries = max_entries
        self.max_replicas_per_key = max_replicas_per_key
        self.max_replicas_per_device = max_replicas_per_device
        self.max_pending_per_replica = max_pending_per_replica
        self._close_engine = close_engine or _default_close
        self.native_call_timeout_seconds = native_call_timeout_seconds
        self._watchdog = (
            NativeCallWatchdog(native_call_timeout_seconds)
            if native_call_timeout_seconds is not None else None
        )
        self._pid = os.getpid()
        self._condition = threading.Condition()
        self._entries: OrderedDict[EngineKey, list[_Replica]] = OrderedDict()
        self._retiring = 0
        self._retiring_devices: dict[str, int] = {}
        self._miss_waiters: deque[object] = deque()
        self._miss_waiter_devices: dict[object, str] = {}
        self._draining_key: EngineKey | None = None
        self._closed = False
        self._close_complete = False
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._reset_after_fork)

    def _reset_after_fork(self) -> None:
        # Never acquire an inherited lock or close an inherited CUDA object.
        # The only live thread in the child is the one that called fork().
        self._pid = os.getpid()
        if self._watchdog is not None:
            self._watchdog.detach_after_fork()
            self._watchdog = NativeCallWatchdog(self.native_call_timeout_seconds)
        self._condition = threading.Condition()
        self._entries = OrderedDict()
        self._retiring = 0
        self._retiring_devices = {}
        self._miss_waiters = deque()
        self._miss_waiter_devices = {}
        self._draining_key = None
        self._closed = False
        self._close_complete = False

    def _check_pid(self) -> None:
        # Also covers platforms without register_at_fork and unusual forks.
        if self._pid != os.getpid():
            self._reset_after_fork()

    @contextmanager
    def _native_guard(self):
        watchdog = self._watchdog
        if watchdog is None:
            yield
            return
        token = watchdog.arm()
        try:
            yield
        finally:
            watchdog.disarm(token)

    def run(
        self,
        key: EngineKey,
        factory: Callable[[], Any],
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run ``operation(engine, *args, **kwargs)`` on the engine owner thread.

        Accepted calls complete even if another thread closes the pool.
        Construction and invocation errors reach the caller.  A failed
        construction is retried by a later call, so it cannot poison the key.
        """
        if not isinstance(key, EngineKey):
            raise TypeError("key must be an EngineKey")
        if getattr(_owner_context, "pool", None) is self:
            raise RuntimeError("recursive submission from an engine owner thread")
        self._check_pid()
        ticket: object | None = None

        def wait_for_capacity() -> None:
            # Called with _condition held. An interrupted waiter must not
            # strand its FIFO ticket or the drain reservation.
            try:
                self._condition.wait()
            except BaseException:
                if ticket is not None:
                    self._miss_waiters.remove(ticket)
                    del self._miss_waiter_devices[ticket]
                    if not self._miss_waiters:
                        self._draining_key = None
                    self._condition.notify_all()
                raise

        while True:
            retired: list[_Replica] | None = None
            retired_device: str | None = None
            with self._condition:
                if self._closed:
                    if ticket is not None:
                        self._miss_waiters.remove(ticket)
                        del self._miss_waiter_devices[ticket]
                    raise RuntimeError("engine pool is closed")
                replicas = self._entries.get(key)
                if replicas is not None and ticket is not None:
                    self._miss_waiters.remove(ticket)
                    del self._miss_waiter_devices[ticket]
                    ticket = None
                    self._condition.notify_all()
                if key == self._draining_key:
                    wait_for_capacity()
                    continue
                if replicas is None:
                    device_count = self._retiring_devices.get(key.device, 0) + sum(
                        len(group) for old_key, group in self._entries.items() if old_key.device == key.device
                    )
                    device_full = device_count >= self.max_replicas_per_device
                    entries_full = len(self._entries) + self._retiring >= self.max_entries
                    same_device_waiter = any(waiter_device == key.device for waiter_device in self._miss_waiter_devices.values())
                    if ticket is None and (entries_full or device_full or same_device_waiter):
                        ticket = object()
                        self._miss_waiters.append(ticket)
                        self._miss_waiter_devices[ticket] = key.device
                    if ticket is not None and self._miss_waiters[0] is not ticket:
                        wait_for_capacity()
                        continue
                    if entries_full or device_full:
                        for old_key, old_replicas in self._entries.items():
                            if device_full and old_key.device != key.device:
                                continue
                            if all(replica.pending == 0 for replica in old_replicas):
                                retired = self._entries.pop(old_key)
                                self._retiring += 1
                                retired_device = old_key.device
                                self._retiring_devices[old_key.device] = self._retiring_devices.get(old_key.device, 0) + len(retired)
                                # The queued miss now owns a replacement slot,
                                # even if another key became idle first.  A
                                # stale drain would block that key forever.
                                self._draining_key = None
                                break
                        if retired is None:
                            if self._entries and self._draining_key is None:
                                self._draining_key = next((old_key for old_key in self._entries if not device_full or old_key.device == key.device), None)
                            wait_for_capacity()
                            continue
                    else:
                        replicas = []
                        self._entries[key] = replicas
                        if ticket is not None:
                            self._miss_waiters.popleft()
                            del self._miss_waiter_devices[ticket]
                            ticket = None
                            self._condition.notify_all()
                if retired is None:
                    assert replicas is not None
                    self._entries.move_to_end(key)
                    device_count = self._retiring_devices.get(key.device, 0) + sum(
                        len(group) for old_key, group in self._entries.items() if old_key.device == key.device
                    )
                    if (
                        replicas
                        and min(replica.pending for replica in replicas) == 0
                    ) or len(replicas) >= self.max_replicas_per_key or device_count >= self.max_replicas_per_device:
                        replica = min(replicas, key=lambda candidate: candidate.pending)
                    else:
                        replica = _Replica(self, key)
                        replicas.append(replica)
                    if replica.pending >= self.max_pending_per_replica:
                        wait_for_capacity()
                        continue
                    result: Future[T] = Future()
                    replica.pending += 1
                    replica.work.put(_Work(factory, operation, args, kwargs, result))
                    break
            # A replacement cannot start until retired predictors have closed.
            assert retired is not None
            try:
                for replica in retired:
                    replica.stop()
                for replica in retired:
                    replica.join()
            finally:
                with self._condition:
                    self._retiring -= 1
                    assert retired_device is not None
                    remaining = self._retiring_devices[retired_device] - len(retired)
                    if remaining:
                        self._retiring_devices[retired_device] = remaining
                    else:
                        del self._retiring_devices[retired_device]
                    self._condition.notify_all()
        if self.native_call_timeout_seconds is None:
            return result.result()
        # This is a caller diagnostic, not a thread-based native interrupter.
        # The independent pidfd watchdog kills only on an ACTIVE native call.
        # Each accepted item ahead of us may consume factory, invocation and
        # failed-engine disposal budgets. A fresh ARM acknowledgement has its
        # own bounded startup wait. This does not time admission to the pool.
        queue_envelope = self.max_pending_per_replica * (
            3 * self.native_call_timeout_seconds + 9
        ) + 5
        return result.result(timeout=queue_envelope)

    def close(self) -> None:
        """Stop accepting work, drain accepted calls, and release predictors."""
        self._check_pid()
        if getattr(_owner_context, "pool", None) is self:
            raise RuntimeError("cannot close an engine pool from its owner thread")
        with self._condition:
            if self._closed:
                while not self._close_complete:
                    self._condition.wait()
                return
            self._closed = True
            replicas = [replica for group in self._entries.values() for replica in group]
            self._entries.clear()
            self._condition.notify_all()
            for replica in replicas:
                replica.stop()
            while self._retiring:
                self._condition.wait()
        try:
            for replica in replicas:
                replica.join()
        finally:
            with self._condition:
                self._close_complete = True
                self._condition.notify_all()
            if self._watchdog is not None:
                self._watchdog.close()

    def __enter__(self) -> EngineRuntimePool:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
