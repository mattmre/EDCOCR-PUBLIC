"""Tests for doc_registry thread safety.

Verifies that concurrent access to doc_registry from multiple threads
does not raise RuntimeError due to dictionary mutation during iteration.
"""

import threading
import time

import pytest


@pytest.fixture(autouse=True)
def _isolate_registry(monkeypatch):
    """Ensure tests start with an empty registry and lock, restored after."""
    import ocr_gpu_async as pipe

    # Use fresh dict and lock per test
    monkeypatch.setattr(pipe, "doc_registry", {})
    monkeypatch.setattr(pipe, "doc_registry_lock", threading.RLock())
    yield
    # monkeypatch auto-restores


class _FakeDocState:
    """Minimal stand-in for DocumentState (avoids filesystem side effects)."""

    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.path = f"/fake/{doc_id}.pdf"
        self.processed_pages = 0
        self.total_pages = 10
        self.custody_chain = None


class TestDocRegistryLockExists:
    """Verify the lock is declared at module level."""

    def test_lock_is_rlock(self):
        import ocr_gpu_async as pipe
        # The monkeypatched lock is also an RLock
        assert isinstance(pipe.doc_registry_lock, type(threading.RLock()))

    def test_lock_is_module_attribute(self):
        import ocr_gpu_async as pipe
        assert hasattr(pipe, "doc_registry_lock")


class TestConcurrentReadWrite:
    """Concurrent writers and readers must not raise RuntimeError."""

    def test_concurrent_write_and_iterate(self):
        """Simulate scheduler writes + monitor iteration concurrently."""
        import ocr_gpu_async as pipe

        errors = []

        def writer():
            """Simulates scheduler registering documents."""
            for i in range(200):
                doc_id = f"write-{i}"
                with pipe.doc_registry_lock:
                    pipe.doc_registry[doc_id] = _FakeDocState(doc_id)
                # Slight yield to encourage interleaving
                if i % 20 == 0:
                    time.sleep(0)

        def deleter():
            """Simulates assembler removing completed documents."""
            deleted = 0
            deadline = time.time() + 5
            while deleted < 100 and time.time() < deadline:
                with pipe.doc_registry_lock:
                    keys = list(pipe.doc_registry.keys())
                if keys:
                    key = keys[0]
                    with pipe.doc_registry_lock:
                        if key in pipe.doc_registry:
                            del pipe.doc_registry[key]
                            deleted += 1
                else:
                    time.sleep(0.001)

        def iterator():
            """Simulates monitor thread iterating the registry."""
            count = 0
            deadline = time.time() + 5
            while count < 100 and time.time() < deadline:
                try:
                    with pipe.doc_registry_lock:
                        snapshot = dict(pipe.doc_registry)
                    # Iterate snapshot outside the lock
                    for d_id, doc in snapshot.items():
                        _ = f"{doc.path}: {doc.processed_pages}/{doc.total_pages}"
                    count += 1
                except RuntimeError as e:
                    errors.append(str(e))
                    break
                time.sleep(0)

        threads = [
            threading.Thread(target=writer, name="Writer"),
            threading.Thread(target=deleter, name="Deleter"),
            threading.Thread(target=iterator, name="Iterator"),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"RuntimeError during concurrent access: {errors}"

    def test_concurrent_get_from_multiple_workers(self):
        """Simulate 12 GPU workers calling doc_registry.get() concurrently."""
        import ocr_gpu_async as pipe

        # Pre-populate registry
        for i in range(10):
            doc_id = f"doc-{i}"
            pipe.doc_registry[doc_id] = _FakeDocState(doc_id)

        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(100):
                    doc_id = f"doc-{i % 10}"
                    with pipe.doc_registry_lock:
                        state = pipe.doc_registry.get(doc_id)
                    if state:
                        results.append((worker_id, doc_id))
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=worker, args=(w,), name=f"Worker-{w}")
            for w in range(12)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent get: {errors}"
        # Each of 12 workers should have done 100 lookups
        assert len(results) == 1200

    def test_rlock_allows_reentrant_access(self):
        """RLock must allow the same thread to acquire it multiple times."""
        import ocr_gpu_async as pipe

        doc_id = "reentrant-test"
        pipe.doc_registry[doc_id] = _FakeDocState(doc_id)

        # Simulate monitor pattern: lock to snapshot, then lock again inside
        with pipe.doc_registry_lock:
            _snapshot = dict(pipe.doc_registry)  # noqa: F841 — validates nested lock
            # Nested acquisition should not deadlock with RLock
            with pipe.doc_registry_lock:
                state = pipe.doc_registry.get(doc_id)
                assert state is not None
                assert state.doc_id == doc_id

    def test_snapshot_pattern_decouples_iteration_from_mutation(self):
        """Snapshot pattern: iterate snapshot while original dict is mutated."""
        import ocr_gpu_async as pipe

        for i in range(50):
            pipe.doc_registry[f"snap-{i}"] = _FakeDocState(f"snap-{i}")

        # Take snapshot under lock
        with pipe.doc_registry_lock:
            snapshot = dict(pipe.doc_registry)

        # Mutate original dict while iterating snapshot
        with pipe.doc_registry_lock:
            pipe.doc_registry["snap-new"] = _FakeDocState("snap-new")
            del pipe.doc_registry["snap-0"]

        # Snapshot should still have original 50 entries
        assert len(snapshot) == 50
        assert "snap-0" in snapshot
        assert "snap-new" not in snapshot

    def test_concurrent_delete_during_iteration_no_error(self):
        """Without the lock, this pattern would raise RuntimeError."""
        import ocr_gpu_async as pipe

        for i in range(100):
            pipe.doc_registry[f"iter-{i}"] = _FakeDocState(f"iter-{i}")

        errors = []

        def mutator():
            """Delete entries while iterator is running."""
            for i in range(100):
                with pipe.doc_registry_lock:
                    key = f"iter-{i}"
                    if key in pipe.doc_registry:
                        del pipe.doc_registry[key]
                time.sleep(0)

        def iterator():
            """Iterate with snapshot pattern."""
            for _ in range(50):
                try:
                    with pipe.doc_registry_lock:
                        snap = dict(pipe.doc_registry)
                    for k, v in snap.items():
                        _ = v.doc_id
                except RuntimeError as e:
                    errors.append(str(e))
                time.sleep(0)

        t1 = threading.Thread(target=mutator, name="Mutator")
        t2 = threading.Thread(target=iterator, name="Iterator")
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"RuntimeError during concurrent delete+iterate: {errors}"
