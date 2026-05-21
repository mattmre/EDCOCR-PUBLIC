"""Thread-safe TTL cache for Prometheus metric collection results."""

import threading
import time
from typing import Any, Callable, Optional


class MetricsCache:
    """Thread-safe TTL cache for Prometheus metric collection results.

    Each cache slot is identified by a string key. Values are stored with
    a timestamp and returned on subsequent lookups until the TTL expires.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, tuple[Any, float]] = {}
        self._inflight: dict[str, threading.Event] = {}

    def _get_unlocked(self, key: str) -> Optional[Any]:
        """Return cached value if present and not expired.

        Caller must hold ``self._lock``.
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if time.monotonic() > expiry:
            del self._store[key]
            return None
        return value

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if present and not expired, else None."""
        with self._lock:
            return self._get_unlocked(key)

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        """Store a value with the given TTL."""
        with self._lock:
            self._store[key] = (value, time.monotonic() + ttl_seconds)

    def get_or_compute(
        self, key: str, ttl_seconds: float, compute_fn: Callable[[], Any]
    ) -> Any:
        """Return cached value or compute, cache, and return."""
        while True:
            with self._lock:
                cached = self._get_unlocked(key)
                if cached is not None:
                    return cached

                waiter = self._inflight.get(key)
                if waiter is None:
                    waiter = threading.Event()
                    self._inflight[key] = waiter
                    should_compute = True
                else:
                    should_compute = False

            if should_compute:
                try:
                    value = compute_fn()
                except Exception:
                    with self._lock:
                        current = self._inflight.get(key)
                        if current is waiter:
                            self._inflight.pop(key, None)
                            waiter.set()
                    raise

                with self._lock:
                    self._store[key] = (value, time.monotonic() + ttl_seconds)
                    current = self._inflight.get(key)
                    if current is waiter:
                        self._inflight.pop(key, None)
                        waiter.set()
                return value

            waiter.wait()

    def invalidate(self, key: Optional[str] = None) -> None:
        """Clear a specific key or the entire cache."""
        with self._lock:
            if key is None:
                self._store.clear()
            else:
                self._store.pop(key, None)


_cache = MetricsCache()
