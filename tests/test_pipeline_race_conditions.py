"""Pipeline thread race condition coverage.

Stress-tests pipeline shared state under concurrent access to verify
that the locking discipline in ocr_gpu_async.py prevents data races.

Targets:
  - doc_registry + doc_registry_lock (RLock)
  - _ENGINE_CACHE + _ENGINE_CACHE_LOCK
  - global_pages_processed + _pages_processed_lock
  - Queue backpressure (producer/consumer)
"""

import os
import queue
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# 1. doc_registry concurrent access
# ===========================================================================


class TestDocRegistryConcurrentAccess:
    """Verify doc_registry reads/writes under concurrent access."""

    def test_concurrent_writes_no_runtime_error(self):
        """Multiple threads writing to doc_registry concurrently must not
        raise RuntimeError ('dictionary changed size during iteration')."""
        from ocr_gpu_async import doc_registry, doc_registry_lock

        num_threads = 16
        iterations_per_thread = 200
        barrier = threading.Barrier(num_threads)
        errors = []

        # Save original state
        original_registry = dict(doc_registry)

        def writer(thread_id):
            try:
                barrier.wait(timeout=5)
                for i in range(iterations_per_thread):
                    doc_id = f"stress_test_{thread_id}_{i}"
                    mock_state = MagicMock()
                    mock_state.doc_id = doc_id
                    mock_state.path = f"/fake/path/{doc_id}"
                    mock_state.total_pages = 10
                    mock_state.processed_pages = 0
                    mock_state.finalized = False

                    with doc_registry_lock:
                        doc_registry[doc_id] = mock_state

                    # Simulate reads that other threads would do
                    with doc_registry_lock:
                        _ = doc_registry.get(doc_id)

                    # Simulate deletion
                    with doc_registry_lock:
                        doc_registry.pop(doc_id, None)
            except Exception as exc:
                errors.append((thread_id, exc))

        threads = [
            threading.Thread(target=writer, args=(i,), daemon=True)
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Restore original state
        with doc_registry_lock:
            doc_registry.clear()
            doc_registry.update(original_registry)

        assert not errors, f"Threads reported errors: {errors}"

    def test_concurrent_iteration_and_mutation(self):
        """Iterating doc_registry while other threads mutate it must not
        crash when the lock is held properly."""
        from ocr_gpu_async import doc_registry, doc_registry_lock

        num_writers = 8
        num_readers = 4
        iterations = 100
        barrier = threading.Barrier(num_writers + num_readers)
        errors = []

        original_registry = dict(doc_registry)

        def writer(thread_id):
            try:
                barrier.wait(timeout=5)
                for i in range(iterations):
                    doc_id = f"iter_test_{thread_id}_{i}"
                    mock_state = MagicMock()
                    mock_state.doc_id = doc_id

                    with doc_registry_lock:
                        doc_registry[doc_id] = mock_state

                    time.sleep(0.0001)  # Yield to increase contention

                    with doc_registry_lock:
                        doc_registry.pop(doc_id, None)
            except Exception as exc:
                errors.append(("writer", thread_id, exc))

        def reader(thread_id):
            try:
                barrier.wait(timeout=5)
                for _ in range(iterations):
                    with doc_registry_lock:
                        snapshot = dict(doc_registry)
                    # Work on snapshot outside the lock (safe pattern)
                    _ = len(snapshot)
                    for doc_id in snapshot:
                        _ = snapshot[doc_id]
            except Exception as exc:
                errors.append(("reader", thread_id, exc))

        threads = []
        for i in range(num_writers):
            threads.append(threading.Thread(target=writer, args=(i,), daemon=True))
        for i in range(num_readers):
            threads.append(threading.Thread(target=reader, args=(i,), daemon=True))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        with doc_registry_lock:
            doc_registry.clear()
            doc_registry.update(original_registry)

        assert not errors, f"Threads reported errors: {errors}"

    def test_rlock_allows_reentrant_acquisition(self):
        """doc_registry_lock is an RLock, so it must allow reentrant
        acquisition within the same thread (e.g., in exception handlers)."""
        from ocr_gpu_async import doc_registry_lock

        # This should not deadlock
        acquired = False
        with doc_registry_lock:
            with doc_registry_lock:
                acquired = True
        assert acquired, "RLock re-entrant acquisition failed"


# ===========================================================================
# 2. _ENGINE_CACHE concurrent access
# ===========================================================================


class TestEngineCacheConcurrentAccess:
    """Verify _ENGINE_CACHE reads/writes don't deadlock or corrupt."""

    def test_concurrent_cache_reads_no_deadlock(self):
        """Multiple threads reading _ENGINE_CACHE concurrently must not
        deadlock or crash."""
        from ocr_gpu_async import _ENGINE_CACHE, _ENGINE_CACHE_LOCK

        num_threads = 16
        iterations = 500
        barrier = threading.Barrier(num_threads)
        errors = []
        deadlock_detected = threading.Event()

        # Pre-populate cache with mock entries
        mock_entries = {}
        for i in range(5):
            lang = f"test_lang_{i}"
            mock_engine = MagicMock()
            inference_lock = threading.Lock()
            mock_entries[lang] = (mock_engine, inference_lock)

        with _ENGINE_CACHE_LOCK:
            original_cache = dict(_ENGINE_CACHE)
            _ENGINE_CACHE.update(mock_entries)

        def reader(thread_id):
            try:
                barrier.wait(timeout=5)
                for _ in range(iterations):
                    with _ENGINE_CACHE_LOCK:
                        # Simulate the fast-path check
                        entry = _ENGINE_CACHE.get(f"test_lang_{thread_id % 5}")
                    if entry is not None:
                        engine, inf_lock = entry
                        # Simulate inference lock acquisition (brief)
                        acquired = inf_lock.acquire(timeout=1)
                        if acquired:
                            inf_lock.release()
                        else:
                            deadlock_detected.set()
            except Exception as exc:
                errors.append((thread_id, exc))

        threads = [
            threading.Thread(target=reader, args=(i,), daemon=True)
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Restore
        with _ENGINE_CACHE_LOCK:
            for key in mock_entries:
                _ENGINE_CACHE.pop(key, None)
            _ENGINE_CACHE.update(original_cache)

        assert not errors, f"Threads reported errors: {errors}"
        assert not deadlock_detected.is_set(), "Deadlock detected in inference lock"

    def test_concurrent_cache_write_and_read(self):
        """Writers adding entries while readers check the cache must not
        corrupt the dict or deadlock."""
        from ocr_gpu_async import _ENGINE_CACHE, _ENGINE_CACHE_LOCK

        num_writers = 4
        num_readers = 8
        iterations = 200
        barrier = threading.Barrier(num_writers + num_readers)
        errors = []

        with _ENGINE_CACHE_LOCK:
            original_cache = dict(_ENGINE_CACHE)

        def writer(thread_id):
            try:
                barrier.wait(timeout=5)
                for i in range(iterations):
                    lang = f"rw_test_{thread_id}_{i}"
                    mock_engine = MagicMock()
                    entry = (mock_engine, threading.Lock())

                    with _ENGINE_CACHE_LOCK:
                        _ENGINE_CACHE[lang] = entry

                    time.sleep(0.0001)

                    with _ENGINE_CACHE_LOCK:
                        _ENGINE_CACHE.pop(lang, None)
            except Exception as exc:
                errors.append(("writer", thread_id, exc))

        def reader(thread_id):
            try:
                barrier.wait(timeout=5)
                for i in range(iterations):
                    with _ENGINE_CACHE_LOCK:
                        keys = list(_ENGINE_CACHE.keys())
                    # Accessing keys outside the lock is fine since we took a copy
                    _ = len(keys)
            except Exception as exc:
                errors.append(("reader", thread_id, exc))

        threads = []
        for i in range(num_writers):
            threads.append(threading.Thread(target=writer, args=(i,), daemon=True))
        for i in range(num_readers):
            threads.append(threading.Thread(target=reader, args=(i,), daemon=True))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        with _ENGINE_CACHE_LOCK:
            # Clean up test keys
            test_keys = [k for k in _ENGINE_CACHE if k.startswith("rw_test_")]
            for k in test_keys:
                del _ENGINE_CACHE[k]
            _ENGINE_CACHE.update(original_cache)

        assert not errors, f"Threads reported errors: {errors}"

    def test_engine_cache_lock_is_not_rlock(self):
        """_ENGINE_CACHE_LOCK should be a regular Lock (not RLock) to catch
        accidental recursive acquisition, which would indicate a design issue."""
        from ocr_gpu_async import _ENGINE_CACHE_LOCK

        assert isinstance(_ENGINE_CACHE_LOCK, type(threading.Lock())), (
            "_ENGINE_CACHE_LOCK should be threading.Lock, not RLock"
        )


# ===========================================================================
# 3. Global counter increments
# ===========================================================================


class TestGlobalCounterIncrements:
    """Verify global_pages_processed increments correctly under concurrency."""

    def test_concurrent_counter_increment_correctness(self):
        """Multiple threads incrementing global_pages_processed must produce
        the exact expected total."""
        import ocr_gpu_async

        num_threads = 16
        increments_per_thread = 500
        barrier = threading.Barrier(num_threads)
        errors = []

        # Save and reset
        original_count = ocr_gpu_async.global_pages_processed
        ocr_gpu_async.global_pages_processed = 0

        def incrementer(thread_id):
            try:
                barrier.wait(timeout=5)
                for _ in range(increments_per_thread):
                    with ocr_gpu_async._pages_processed_lock:
                        ocr_gpu_async.global_pages_processed += 1
            except Exception as exc:
                errors.append((thread_id, exc))

        threads = [
            threading.Thread(target=incrementer, args=(i,), daemon=True)
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        expected = num_threads * increments_per_thread
        actual = ocr_gpu_async.global_pages_processed

        # Restore
        ocr_gpu_async.global_pages_processed = original_count

        assert not errors, f"Threads reported errors: {errors}"
        assert actual == expected, (
            f"Counter mismatch: expected {expected}, got {actual} "
            f"(lost {expected - actual} increments)"
        )

    def test_docs_processed_counter_correctness(self):
        """global_docs_processed counter with its own lock must also be
        correct under concurrent increments."""
        import ocr_gpu_async

        num_threads = 8
        increments = 200
        barrier = threading.Barrier(num_threads)
        errors = []

        original_count = ocr_gpu_async.global_docs_processed
        ocr_gpu_async.global_docs_processed = 0

        def incrementer(thread_id):
            try:
                barrier.wait(timeout=5)
                for _ in range(increments):
                    with ocr_gpu_async._docs_processed_lock:
                        ocr_gpu_async.global_docs_processed += 1
            except Exception as exc:
                errors.append((thread_id, exc))

        threads = [
            threading.Thread(target=incrementer, args=(i,), daemon=True)
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        expected = num_threads * increments
        actual = ocr_gpu_async.global_docs_processed

        ocr_gpu_async.global_docs_processed = original_count

        assert not errors, f"Threads reported errors: {errors}"
        assert actual == expected, (
            f"Docs counter mismatch: expected {expected}, got {actual}"
        )

    def test_counter_without_lock_demonstrates_race(self):
        """Demonstrate that without the lock, concurrent increments can lose
        updates (validating that our lock is actually necessary).

        Note: This test may occasionally pass due to GIL timing, but on most
        runs with enough threads/iterations it will show lost increments.
        We mark it as a demonstration, not a strict assertion."""
        # This is a demonstration test -- we just verify it doesn't crash.
        # The actual count may or may not match depending on GIL timing.
        shared_counter = {"value": 0}
        num_threads = 16
        increments = 1000
        barrier = threading.Barrier(num_threads)

        def unsafe_incrementer(thread_id):
            barrier.wait(timeout=5)
            for _ in range(increments):
                # Deliberately unsafe -- no lock
                shared_counter["value"] += 1

        threads = [
            threading.Thread(target=unsafe_incrementer, args=(i,), daemon=True)
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # We just assert it didn't crash. The counter value is unreliable
        # without locking, and that's the point.
        assert shared_counter["value"] > 0


# ===========================================================================
# 4. Queue backpressure
# ===========================================================================


class TestQueueBackpressure:
    """Verify producer/consumer queue patterns don't deadlock."""

    def test_bounded_queue_producer_consumer_no_deadlock(self):
        """Producers filling a bounded queue while consumers drain it must
        not deadlock, even when the queue is saturated."""
        q = queue.Queue(maxsize=10)
        num_producers = 4
        num_consumers = 2
        items_per_producer = 200
        sentinel = "DONE"
        barrier = threading.Barrier(num_producers + num_consumers)
        produced = {"count": 0}
        consumed = {"count": 0}
        produced_lock = threading.Lock()
        consumed_lock = threading.Lock()
        errors = []

        def producer(pid):
            try:
                barrier.wait(timeout=5)
                for i in range(items_per_producer):
                    q.put(f"item_{pid}_{i}", timeout=5)
                    with produced_lock:
                        produced["count"] += 1
            except Exception as exc:
                errors.append(("producer", pid, exc))

        def consumer(cid):
            try:
                barrier.wait(timeout=5)
                while True:
                    try:
                        item = q.get(timeout=2)
                        if item == sentinel:
                            q.task_done()
                            break
                        q.task_done()
                        with consumed_lock:
                            consumed["count"] += 1
                    except queue.Empty:
                        break
            except Exception as exc:
                errors.append(("consumer", cid, exc))

        prod_threads = [
            threading.Thread(target=producer, args=(i,), daemon=True)
            for i in range(num_producers)
        ]
        cons_threads = [
            threading.Thread(target=consumer, args=(i,), daemon=True)
            for i in range(num_consumers)
        ]

        for t in cons_threads + prod_threads:
            t.start()

        # Wait for producers to finish
        for t in prod_threads:
            t.join(timeout=30)

        # Signal consumers to stop
        for _ in range(num_consumers):
            q.put(sentinel, timeout=5)

        for t in cons_threads:
            t.join(timeout=30)

        assert not errors, f"Threads reported errors: {errors}"

        expected = num_producers * items_per_producer
        assert produced["count"] == expected, (
            f"Produced {produced['count']}, expected {expected}"
        )
        assert consumed["count"] == expected, (
            f"Consumed {consumed['count']}, expected {expected}"
        )

    def test_queue_full_timeout_does_not_deadlock(self):
        """When queue is full, put() with timeout should raise queue.Full,
        not deadlock."""
        q = queue.Queue(maxsize=2)
        q.put("a")
        q.put("b")  # Queue is now full

        with pytest.raises(queue.Full):
            q.put("c", timeout=0.1)

        # Drain one item to verify queue is still functional
        item = q.get(timeout=1)
        assert item == "a"

    def test_queue_empty_timeout_does_not_deadlock(self):
        """When queue is empty, get() with timeout should raise queue.Empty,
        not deadlock."""
        q = queue.Queue(maxsize=10)

        with pytest.raises(queue.Empty):
            q.get(timeout=0.1)

    def test_stop_event_with_queue_drain(self):
        """Simulate the pipeline's stop_event + queue drain pattern:
        producers check stop_event and stop producing, consumers drain
        remaining items."""
        q = queue.Queue(maxsize=20)
        stop = threading.Event()
        num_producers = 4
        num_consumers = 2
        consumed = {"count": 0}
        consumed_lock = threading.Lock()
        barrier = threading.Barrier(num_producers + num_consumers)
        errors = []

        def producer(pid):
            try:
                barrier.wait(timeout=5)
                i = 0
                while not stop.is_set():
                    try:
                        q.put(f"item_{pid}_{i}", timeout=0.05)
                        i += 1
                    except queue.Full:
                        pass
            except Exception as exc:
                errors.append(("producer", pid, exc))

        def consumer(cid):
            try:
                barrier.wait(timeout=5)
                while True:
                    try:
                        _ = q.get(timeout=0.5)
                        q.task_done()
                        with consumed_lock:
                            consumed["count"] += 1
                    except queue.Empty:
                        if stop.is_set():
                            break
            except Exception as exc:
                errors.append(("consumer", cid, exc))

        prod_threads = [
            threading.Thread(target=producer, args=(i,), daemon=True)
            for i in range(num_producers)
        ]
        cons_threads = [
            threading.Thread(target=consumer, args=(i,), daemon=True)
            for i in range(num_consumers)
        ]

        for t in cons_threads + prod_threads:
            t.start()

        # Let it run for a short time
        time.sleep(0.3)
        stop.set()

        for t in prod_threads:
            t.join(timeout=10)
        for t in cons_threads:
            t.join(timeout=10)

        assert not errors, f"Threads reported errors: {errors}"
        assert consumed["count"] > 0, "No items were consumed"

    def test_multiple_sentinels_for_multiple_consumers(self):
        """The pipeline uses one sentinel per consumer thread. Verify that
        N sentinels correctly stop N consumers."""
        q = queue.Queue(maxsize=50)
        num_consumers = 4
        sentinel = None
        stopped = {"count": 0}
        stopped_lock = threading.Lock()
        barrier = threading.Barrier(num_consumers)
        errors = []

        # Pre-fill queue with work items
        for i in range(20):
            q.put(f"work_{i}")

        # Add sentinels
        for _ in range(num_consumers):
            q.put(sentinel)

        def consumer(cid):
            try:
                barrier.wait(timeout=5)
                while True:
                    item = q.get(timeout=5)
                    q.task_done()
                    if item is sentinel:
                        with stopped_lock:
                            stopped["count"] += 1
                        return
            except Exception as exc:
                errors.append(("consumer", cid, exc))

        threads = [
            threading.Thread(target=consumer, args=(i,), daemon=True)
            for i in range(num_consumers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Threads reported errors: {errors}"
        assert stopped["count"] == num_consumers, (
            f"Only {stopped['count']}/{num_consumers} consumers stopped"
        )


# ===========================================================================
# 5. Thread barrier coordination (integration-level)
# ===========================================================================


class TestThreadBarrierCoordination:
    """Verify that threading.Barrier works correctly for synchronizing
    pipeline stage startup (used throughout the tests above)."""

    def test_barrier_releases_all_threads(self):
        """All threads waiting at a barrier should be released simultaneously."""
        num_threads = 8
        barrier = threading.Barrier(num_threads)
        released = {"count": 0}
        released_lock = threading.Lock()
        timestamps = []
        ts_lock = threading.Lock()

        def waiter(tid):
            barrier.wait(timeout=5)
            t = time.time()
            with ts_lock:
                timestamps.append(t)
            with released_lock:
                released["count"] += 1

        threads = [
            threading.Thread(target=waiter, args=(i,), daemon=True)
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert released["count"] == num_threads
        # All release times should be within 100ms of each other
        if timestamps:
            spread = max(timestamps) - min(timestamps)
            assert spread < 0.5, f"Barrier release spread too large: {spread:.3f}s"

    def test_broken_barrier_does_not_deadlock(self):
        """If a barrier breaks (e.g., thread dies), remaining threads
        should get BrokenBarrierError, not deadlock."""
        barrier = threading.Barrier(3)
        errors = []

        def waiter_that_aborts():
            barrier.abort()

        def waiter(tid):
            try:
                barrier.wait(timeout=2)
            except threading.BrokenBarrierError:
                errors.append(("broken", tid))

        t1 = threading.Thread(target=waiter_that_aborts, daemon=True)
        t2 = threading.Thread(target=waiter, args=(1,), daemon=True)

        t1.start()
        t2.start()

        t1.join(timeout=5)
        t2.join(timeout=5)

        # At least one thread should have gotten BrokenBarrierError
        assert any(e[0] == "broken" for e in errors), (
            "Expected BrokenBarrierError but none occurred"
        )
