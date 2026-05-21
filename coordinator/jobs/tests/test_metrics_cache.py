"""Tests for the MetricsCache thread-safe TTL cache.

Run with: cd coordinator && python -m pytest jobs/tests/test_metrics_cache.py -v
"""

import threading
import time
from unittest.mock import patch

from django.test import TestCase

from jobs.metrics_cache import MetricsCache


class TestMetricsCache(TestCase):
    """Tests for MetricsCache get/set/get_or_compute/invalidate."""

    def setUp(self):
        self.cache = MetricsCache()

    def test_get_returns_none_for_missing_key(self):
        """get() returns None when the key has never been set."""
        assert self.cache.get("nonexistent") is None

    def test_set_and_get_within_ttl(self):
        """A value set with a TTL is retrievable before expiry."""
        self.cache.set("k", 42, ttl_seconds=60)
        assert self.cache.get("k") == 42

    @patch("jobs.metrics_cache.time")
    def test_get_returns_none_after_ttl_expires(self, mock_time):
        """get() returns None after the TTL has elapsed."""
        mock_time.monotonic.return_value = 100.0
        self.cache.set("k", "val", ttl_seconds=10)
        # Advance past expiry
        mock_time.monotonic.return_value = 111.0
        assert self.cache.get("k") is None

    def test_get_or_compute_calls_fn_on_miss(self):
        """get_or_compute() calls compute_fn when key is not cached."""
        calls = []

        def compute():
            calls.append(1)
            return "computed"

        result = self.cache.get_or_compute("k", 60, compute)
        assert result == "computed"
        assert len(calls) == 1

    def test_get_or_compute_returns_cached_on_hit(self):
        """get_or_compute() returns cached value without calling compute_fn."""
        self.cache.set("k", "cached_val", ttl_seconds=60)
        calls = []

        def compute():
            calls.append(1)
            return "fresh"

        result = self.cache.get_or_compute("k", 60, compute)
        assert result == "cached_val"
        assert len(calls) == 0

    @patch("jobs.metrics_cache.time")
    def test_get_or_compute_recomputes_after_expiry(self, mock_time):
        """get_or_compute() calls compute_fn again after TTL expires."""
        mock_time.monotonic.return_value = 100.0
        self.cache.set("k", "old", ttl_seconds=10)

        # Advance past expiry
        mock_time.monotonic.return_value = 111.0

        calls = []

        def compute():
            calls.append(1)
            return "new"

        result = self.cache.get_or_compute("k", 10, compute)
        assert result == "new"
        assert len(calls) == 1

    def test_invalidate_specific_key(self):
        """invalidate(key) removes only that key."""
        self.cache.set("a", 1, ttl_seconds=60)
        self.cache.set("b", 2, ttl_seconds=60)
        self.cache.invalidate("a")
        assert self.cache.get("a") is None
        assert self.cache.get("b") == 2

    def test_invalidate_all(self):
        """invalidate() with no args clears all entries."""
        self.cache.set("a", 1, ttl_seconds=60)
        self.cache.set("b", 2, ttl_seconds=60)
        self.cache.invalidate()
        assert self.cache.get("a") is None
        assert self.cache.get("b") is None

    def test_thread_safety_concurrent_writes(self):
        """Concurrent set() calls from multiple threads do not corrupt state."""
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    self.cache.set(f"key-{n}-{i}", i, ttl_seconds=60)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Verify at least some values are accessible
        assert self.cache.get("key-0-99") == 99

    def test_thread_safety_concurrent_get_or_compute(self):
        """Concurrent get_or_compute() calls share a single computation."""
        errors = []
        results = []
        compute_calls = []
        start = threading.Barrier(8)

        def compute_once():
            compute_calls.append(1)
            time.sleep(0.05)
            return "computed"

        def reader():
            try:
                start.wait()
                val = self.cache.get_or_compute("shared", 60, compute_once)
                results.append(val)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All threads should get the same value
        assert all(r == "computed" for r in results)
        assert len(compute_calls) == 1

    def test_set_overwrites_existing(self):
        """set() with the same key replaces the previous value."""
        self.cache.set("k", "first", ttl_seconds=60)
        self.cache.set("k", "second", ttl_seconds=60)
        assert self.cache.get("k") == "second"

    def test_get_or_compute_does_not_hold_lock_during_compute(self):
        """compute_fn runs outside the lock so other threads are not blocked.

        Verifies that a concurrent get() on a different key can proceed
        while compute_fn is executing.
        """
        barrier = threading.Barrier(2, timeout=5)
        other_result = [None]

        def slow_compute():
            # Signal that compute has started
            barrier.wait()
            # Give the other thread time to complete its get()
            barrier.wait()
            return "slow_result"

        def concurrent_getter():
            # Wait until compute has started
            barrier.wait()
            # This should not be blocked by the compute_fn
            other_result[0] = self.cache.get("other_key")
            barrier.wait()

        self.cache.set("other_key", "other_val", ttl_seconds=60)

        t = threading.Thread(target=concurrent_getter)
        t.start()

        result = self.cache.get_or_compute("slow_key", 60, slow_compute)
        t.join(timeout=5)

        assert result == "slow_result"
        assert other_result[0] == "other_val"
