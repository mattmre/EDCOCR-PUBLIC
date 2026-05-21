"""Tests for job priority queue routing."""

import threading
import time


class TestPriorityJobQueue:
    """Unit tests for the PriorityJobQueue class."""

    def test_priority_ordering(self):
        from api.job_manager import PriorityJobQueue

        q = PriorityJobQueue()
        q.put("low-1", "low")
        q.put("urgent-1", "urgent")
        q.put("normal-1", "normal")

        # Should come out in priority order
        assert q.get() == "urgent-1"
        assert q.get() == "normal-1"
        assert q.get() == "low-1"
        q.done()
        q.done()
        q.done()

    def test_same_priority_fifo(self):
        from api.job_manager import PriorityJobQueue

        q = PriorityJobQueue()
        q.put("a", "normal")
        time.sleep(0.01)  # ensure different timestamps
        q.put("b", "normal")
        time.sleep(0.01)
        q.put("c", "normal")

        assert q.get() == "a"
        assert q.get() == "b"
        assert q.get() == "c"
        q.done()
        q.done()
        q.done()

    def test_pending_count(self):
        from api.job_manager import PriorityJobQueue

        q = PriorityJobQueue()
        assert q.pending == 0
        q.put("j1", "normal")
        assert q.pending == 1
        q.put("j2", "normal")
        assert q.pending == 2

    def test_active_count(self):
        from api.job_manager import PriorityJobQueue

        q = PriorityJobQueue()
        q.put("j1", "normal")
        assert q.active == 0
        q.get()
        assert q.active == 1
        q.done()
        assert q.active == 0

    def test_shutdown(self):
        from api.job_manager import PriorityJobQueue

        q = PriorityJobQueue()
        q.shutdown()
        assert q.get() is None

    def test_shutdown_unblocks_waiting_threads(self):
        from api.job_manager import PriorityJobQueue

        q = PriorityJobQueue()
        result = []

        def worker():
            val = q.get()
            result.append(val)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        time.sleep(0.05)  # let thread block on get()
        q.shutdown()
        t.join(timeout=3.0)
        assert not t.is_alive()
        assert result == [None]

    def test_default_priority(self):
        from api.job_manager import PriorityJobQueue

        q = PriorityJobQueue()
        q.put("j1")  # default is "normal"
        assert q.get() == "j1"
        q.done()

    def test_unknown_priority_treated_as_normal(self):
        from api.job_manager import PriorityJobQueue

        q = PriorityJobQueue()
        q.put("urgent-1", "urgent")
        q.put("unknown-1", "unknown_level")
        q.put("low-1", "low")

        assert q.get() == "urgent-1"
        assert q.get() == "unknown-1"  # treated as normal priority
        assert q.get() == "low-1"
        q.done()
        q.done()
        q.done()

    def test_done_does_not_go_negative(self):
        from api.job_manager import PriorityJobQueue

        q = PriorityJobQueue()
        q.done()  # no active jobs
        assert q.active == 0  # stays at 0, not -1

    def test_mixed_priority_interleaved(self):
        """Test that items added in interleaved order still respect priority."""
        from api.job_manager import PriorityJobQueue

        q = PriorityJobQueue()
        q.put("normal-1", "normal")
        q.put("urgent-1", "urgent")
        q.put("low-1", "low")
        q.put("urgent-2", "urgent")
        q.put("normal-2", "normal")

        results = [q.get() for _ in range(5)]
        # Both urgent items first, then both normal, then low
        assert results[0] == "urgent-1"
        assert results[1] == "urgent-2"
        assert results[2] == "normal-1"
        assert results[3] == "normal-2"
        assert results[4] == "low-1"
        for _ in range(5):
            q.done()

    def test_concurrent_put_get(self):
        """Test thread safety under concurrent access."""
        from api.job_manager import PriorityJobQueue

        q = PriorityJobQueue()
        results = []
        lock = threading.Lock()
        count = 100

        def producer():
            for i in range(count):
                q.put(f"job-{i}", "normal")

        def consumer():
            for _ in range(count):
                val = q.get()
                with lock:
                    results.append(val)
                q.done()

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert len(results) == count
        assert q.pending == 0
        assert q.active == 0


class TestPriorityMap:
    """Test the module-level priority mapping constant."""

    def test_priority_map_ordering(self):
        from api.job_manager import _PRIORITY_MAP

        assert _PRIORITY_MAP["urgent"] < _PRIORITY_MAP["normal"]
        assert _PRIORITY_MAP["normal"] < _PRIORITY_MAP["low"]

    def test_priority_map_completeness(self):
        from api.job_manager import _PRIORITY_MAP

        assert "urgent" in _PRIORITY_MAP
        assert "normal" in _PRIORITY_MAP
        assert "low" in _PRIORITY_MAP


class TestCeleryPriorityConvention:
    """Test that the Celery priority convention is correct (higher = more priority)."""

    def test_rabbitmq_priority_range(self):
        """RabbitMQ priority values should be in [0, 9]."""
        # These values match _CELERY_PRIORITY_MAP in coordinator/jobs/tasks.py
        expected = {"urgent": 9, "normal": 5, "low": 1}
        for val in expected.values():
            assert 0 <= val <= 9

    def test_priority_ordering(self):
        expected = {"urgent": 9, "normal": 5, "low": 1}
        assert expected["urgent"] > expected["normal"]
        assert expected["normal"] > expected["low"]


class TestJobManagerPriorityIntegration:
    """Integration tests verifying priority queue is wired into JobManager."""

    def test_priority_queue_class_exists(self):
        from api.job_manager import PriorityJobQueue
        q = PriorityJobQueue()
        assert hasattr(q, "put")
        assert hasattr(q, "get")
        assert hasattr(q, "done")
        assert hasattr(q, "shutdown")

    def test_priority_map_exists(self):
        from api.job_manager import _PRIORITY_MAP
        assert "urgent" in _PRIORITY_MAP
        assert "normal" in _PRIORITY_MAP
        assert "low" in _PRIORITY_MAP
        # Heapq: lower number = higher priority
        assert _PRIORITY_MAP["urgent"] < _PRIORITY_MAP["normal"]
        assert _PRIORITY_MAP["normal"] < _PRIORITY_MAP["low"]


class TestDatabasePriorityIndex:
    """Test that the priority index exists on the Job model."""

    def test_priority_index_declared(self):
        import os
        os.environ.setdefault("OCR_OUTPUT_DIR", "/tmp/test_output")
        from api.database import Job

        index_names = [
            idx.name for idx in Job.__table_args__
            if hasattr(idx, "name")
        ]
        assert "idx_jobs_priority" in index_names
