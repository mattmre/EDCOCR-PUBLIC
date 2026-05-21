"""Thread-safety tests for SQLite-backed stores.

Verifies that concurrent writes from multiple threads do not corrupt
data in EventStore, EntityIndex, and ReviewQueue.
"""

from __future__ import annotations

import os
import threading

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _output_dir(tmp_path, monkeypatch):
    """Ensure OUTPUT_FOLDER / OCR_OUTPUT_DIR point at a writable temp dir."""
    out = str(tmp_path / "ocr_output")
    os.makedirs(out, exist_ok=True)
    monkeypatch.setenv("OUTPUT_FOLDER", out)
    monkeypatch.setenv("OCR_OUTPUT_DIR", out)
    return out


# ---------------------------------------------------------------------------
# EventStore
# ---------------------------------------------------------------------------

class TestEventStoreThreadSafety:
    """Concurrent writes to EventStore must not lose data."""

    def test_concurrent_writes_no_data_loss(self, tmp_path, _output_dir):
        """10 threads each write 20 events; all 200 must be present."""
        from api.event_store import EventStore

        db_path = str(tmp_path / "event_test.db")
        store = EventStore(db_path=db_path)

        num_threads = 10
        events_per_thread = 20
        barrier = threading.Barrier(num_threads)
        errors: list[str] = []

        def writer(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(events_per_thread):
                    store.store_event(
                        event_type="job.completed",
                        job_id=f"job_{thread_id}",
                        payload={"thread": thread_id, "seq": i},
                    )
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")

        threads = [
            threading.Thread(target=writer, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Writer threads raised errors: {errors}"

        # Verify all events are present
        total = 0
        for tid in range(num_threads):
            events = store.get_events_since(job_id=f"job_{tid}")
            total += len(events)
            assert len(events) == events_per_thread, (
                f"job_{tid}: expected {events_per_thread}, got {len(events)}"
            )

        assert total == num_threads * events_per_thread
        store.close()

    def test_concurrent_write_and_read(self, tmp_path, _output_dir):
        """Writers and readers operating concurrently do not crash."""
        from api.event_store import EventStore

        db_path = str(tmp_path / "event_rw.db")
        store = EventStore(db_path=db_path)
        barrier = threading.Barrier(6)
        errors: list[str] = []

        def writer(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(30):
                    store.store_event(
                        event_type="job.progress",
                        job_id="shared_job",
                        payload={"t": thread_id, "i": i},
                    )
            except Exception as exc:
                errors.append(f"Writer {thread_id}: {exc}")

        def reader(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(30):
                    store.get_events_since(job_id="shared_job")
                    store.get_undelivered(max_age_hours=1)
            except Exception as exc:
                errors.append(f"Reader {thread_id}: {exc}")

        threads = (
            [threading.Thread(target=writer, args=(i,)) for i in range(3)]
            + [threading.Thread(target=reader, args=(i,)) for i in range(3)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Threads raised errors: {errors}"
        store.close()

    def test_concurrent_mark_delivered(self, tmp_path, _output_dir):
        """Multiple threads marking events delivered do not conflict."""
        from api.event_store import EventStore

        db_path = str(tmp_path / "event_deliver.db")
        store = EventStore(db_path=db_path)

        event_ids = []
        for i in range(50):
            eid = store.store_event("job.done", f"j{i}", {"i": i})
            event_ids.append(eid)

        barrier = threading.Barrier(5)
        errors: list[str] = []

        def marker(chunk: list[str]) -> None:
            try:
                barrier.wait(timeout=5)
                for eid in chunk:
                    store.mark_delivered(eid)
            except Exception as exc:
                errors.append(str(exc))

        chunks = [event_ids[i::5] for i in range(5)]
        threads = [threading.Thread(target=marker, args=(c,)) for c in chunks]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors
        undelivered = store.get_undelivered(max_age_hours=1)
        assert len(undelivered) == 0
        store.close()

    def test_rlock_used(self, tmp_path, _output_dir):
        """EventStore uses RLock (not Lock) for re-entrancy safety."""
        from api.event_store import EventStore

        store = EventStore(db_path=str(tmp_path / "rlock.db"))
        assert isinstance(store._lock, type(threading.RLock()))
        store.close()


# ---------------------------------------------------------------------------
# EntityIndex
# ---------------------------------------------------------------------------

class TestEntityIndexThreadSafety:
    """Concurrent writes to EntityIndex must not lose data."""

    def test_concurrent_index_entities(self, tmp_path, _output_dir):
        """10 threads each index 15 entities; all 150 must be present."""
        from api.entity_index import EntityIndex

        db_path = str(tmp_path / "entity_test.db")
        idx = EntityIndex(db_path=db_path)

        num_threads = 10
        entities_per_thread = 15
        barrier = threading.Barrier(num_threads)
        errors: list[str] = []

        def indexer(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                entities = [
                    {
                        "type": "PERSON",
                        "text": f"Person_{thread_id}_{i}",
                        "confidence": 0.9,
                        "source": "ner",
                        "page": i,
                    }
                    for i in range(entities_per_thread)
                ]
                idx.index_entities(
                    job_id=f"job_{thread_id}",
                    document_name=f"doc_{thread_id}.pdf",
                    entities=entities,
                )
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")

        threads = [
            threading.Thread(target=indexer, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Indexer threads raised errors: {errors}"

        stats = idx.stats()
        assert stats["total_entities"] == num_threads * entities_per_thread

    def test_concurrent_index_extractions(self, tmp_path, _output_dir):
        """10 threads each index 15 extractions; all 150 must be present."""
        from api.entity_index import EntityIndex

        db_path = str(tmp_path / "extraction_test.db")
        idx = EntityIndex(db_path=db_path)

        num_threads = 10
        extractions_per_thread = 15
        barrier = threading.Barrier(num_threads)
        errors: list[str] = []

        def indexer(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                extractions = [
                    {
                        "key": f"field_{i}",
                        "value": f"value_{thread_id}_{i}",
                        "confidence": 0.85,
                        "page": i,
                    }
                    for i in range(extractions_per_thread)
                ]
                idx.index_extractions(
                    job_id=f"job_{thread_id}",
                    document_name=f"doc_{thread_id}.pdf",
                    extractions=extractions,
                )
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")

        threads = [
            threading.Thread(target=indexer, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Indexer threads raised errors: {errors}"

        stats = idx.stats()
        assert stats["total_extractions"] == num_threads * extractions_per_thread

    def test_concurrent_search_and_index(self, tmp_path, _output_dir):
        """Concurrent indexing and searching do not crash."""
        from api.entity_index import EntityIndex

        db_path = str(tmp_path / "entity_rw.db")
        idx = EntityIndex(db_path=db_path)
        barrier = threading.Barrier(6)
        errors: list[str] = []

        def indexer(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(20):
                    idx.index_entities(
                        job_id=f"job_{thread_id}",
                        document_name="doc.pdf",
                        entities=[{"type": "ORG", "text": f"Org_{i}", "confidence": 0.8}],
                    )
            except Exception as exc:
                errors.append(f"Indexer {thread_id}: {exc}")

        def searcher(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(20):
                    idx.search_entities(entity_type="ORG")
                    idx.stats()
            except Exception as exc:
                errors.append(f"Searcher {thread_id}: {exc}")

        threads = (
            [threading.Thread(target=indexer, args=(i,)) for i in range(3)]
            + [threading.Thread(target=searcher, args=(i,)) for i in range(3)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Threads raised errors: {errors}"

    def test_rlock_used(self, tmp_path, _output_dir):
        """EntityIndex uses RLock (not Lock) for re-entrancy safety."""
        from api.entity_index import EntityIndex

        idx = EntityIndex(db_path=str(tmp_path / "rlock.db"))
        assert isinstance(idx._lock, type(threading.RLock()))


# ---------------------------------------------------------------------------
# ReviewQueue
# ---------------------------------------------------------------------------

class TestReviewQueueThreadSafety:
    """Concurrent writes to ReviewQueue must not lose data."""

    def test_concurrent_add(self, tmp_path, _output_dir):
        """10 threads each add 15 items; all 150 must be present."""
        from api.review_queue import ReviewQueue

        db_path = str(tmp_path / "review_test.db")
        queue = ReviewQueue(db_path=db_path)

        num_threads = 10
        items_per_thread = 15
        barrier = threading.Barrier(num_threads)
        errors: list[str] = []

        def adder(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(items_per_thread):
                    queue.add(
                        job_id=f"job_{thread_id}_{i}",
                        reason="low_confidence",
                        confidence=0.3,
                        quality_classification="degraded",
                    )
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")

        threads = [
            threading.Thread(target=adder, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Adder threads raised errors: {errors}"

        stats = queue.stats()
        assert stats["total"] == num_threads * items_per_thread

    def test_concurrent_decide(self, tmp_path, _output_dir):
        """Concurrent decisions on different items do not conflict."""
        from api.review_queue import ReviewQueue

        db_path = str(tmp_path / "review_decide.db")
        queue = ReviewQueue(db_path=db_path)

        # Pre-populate items
        review_ids = []
        for i in range(50):
            item = queue.add(
                job_id=f"job_{i}",
                reason="low_confidence",
                confidence=0.4,
            )
            review_ids.append(item.review_id)

        barrier = threading.Barrier(5)
        errors: list[str] = []

        def decider(chunk: list[str]) -> None:
            try:
                barrier.wait(timeout=5)
                for rid in chunk:
                    queue.decide(rid, status="approved", reviewer="auto")
            except Exception as exc:
                errors.append(str(exc))

        chunks = [review_ids[i::5] for i in range(5)]
        threads = [threading.Thread(target=decider, args=(c,)) for c in chunks]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Decider threads raised errors: {errors}"

        stats = queue.stats()
        assert stats["approved"] == 50
        assert stats["pending"] == 0

    def test_concurrent_add_and_list(self, tmp_path, _output_dir):
        """Concurrent adds and list_all queries do not crash."""
        from api.review_queue import ReviewQueue

        db_path = str(tmp_path / "review_rw.db")
        queue = ReviewQueue(db_path=db_path)
        barrier = threading.Barrier(6)
        errors: list[str] = []

        def adder(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(20):
                    queue.add(
                        job_id=f"job_{thread_id}_{i}",
                        reason="manual_flag",
                        confidence=0.5,
                    )
            except Exception as exc:
                errors.append(f"Adder {thread_id}: {exc}")

        def lister(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(20):
                    queue.list_all()
                    queue.stats()
                    queue.count()
            except Exception as exc:
                errors.append(f"Lister {thread_id}: {exc}")

        threads = (
            [threading.Thread(target=adder, args=(i,)) for i in range(3)]
            + [threading.Thread(target=lister, args=(i,)) for i in range(3)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Threads raised errors: {errors}"

    def test_rlock_used(self, tmp_path, _output_dir):
        """ReviewQueue uses RLock (not Lock) for re-entrancy safety."""
        from api.review_queue import ReviewQueue

        queue = ReviewQueue(db_path=str(tmp_path / "rlock.db"))
        assert isinstance(queue._lock, type(threading.RLock()))
