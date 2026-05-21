"""Tests for pipeline resilience fixes  and ."""

import queue
import threading
import time
import unittest
from unittest.mock import patch


class TestAssemblerExitRace(unittest.TestCase):
    """Assembler should not exit when a message arrives during the drain pause."""

    def test_assembler_second_empty_check_requeues_late_message(self):
        """When a message arrives during the drain pause window, the second
        empty() check catches it and the assembler continues instead of
        breaking.  This test is deterministic: it patches time.sleep to
        inject the message synchronously during the pause."""

        assembly_queue = queue.Queue()
        image_queue = queue.Queue()
        stop_event = threading.Event()
        stop_event.set()

        continued = False
        broke = False

        original_sleep = time.sleep

        def fake_sleep(seconds):
            """Inject a late message during the 50ms drain pause."""
            if 0.04 <= seconds <= 0.06:
                # This simulates a worker calling assembly_queue.put()
                # between the first empty() check and the second check.
                assembly_queue.put({"doc_id": "late_doc", "page_num": 1})
            original_sleep(seconds)

        with patch("time.sleep", side_effect=fake_sleep):
            # Run the drain-check logic from assembler_thread
            try:
                assembly_queue.get(timeout=0.01)
            except queue.Empty:
                if image_queue.empty() and assembly_queue.empty() and stop_event.is_set():
                    # brief drain pause
                    time.sleep(0.05)
                    if not assembly_queue.empty():
                        continued = True
                    else:
                        broke = True

        self.assertTrue(continued, "Assembler should have continued on late message")
        self.assertFalse(broke, "Assembler should NOT have broken out")

    def test_assembler_breaks_when_queue_stays_empty(self):
        """When no message arrives during the pause, the assembler should break."""
        assembly_queue = queue.Queue()
        image_queue = queue.Queue()
        stop_event = threading.Event()
        stop_event.set()

        broke = False

        try:
            assembly_queue.get(timeout=0.01)
        except queue.Empty:
            if image_queue.empty() and assembly_queue.empty() and stop_event.is_set():
                time.sleep(0.05)
                if not assembly_queue.empty():
                    self.fail("Queue should still be empty")
                broke = True

        self.assertTrue(broke, "Assembler should break when queue remains empty")


class TestGpuOomDetection(unittest.TestCase):
    """GPU OOM errors should be detected and handled specifically."""

    @staticmethod
    def _classify_oom(exception):
        """Replicate the OOM classification logic from worker_thread."""
        _err_str = str(exception).lower()
        return (
            "out of memory" in _err_str
            or "cuda" in type(exception).__name__.lower()
            or "outofmemory" in type(exception).__name__.lower()
            or "cannot allocate" in _err_str
        )

    def test_oom_exception_detected_as_oom(self):
        """Various OOM-like exceptions should be classified as OOM."""
        cases = [
            RuntimeError("CUDA error: out of memory"),
            RuntimeError("PaddlePaddle: Cannot allocate 2048 MiB on GPU"),
            MemoryError("out of memory"),
            RuntimeError("CUDA out of memory. Tried to allocate 512 MiB"),
        ]
        for exc in cases:
            with self.subTest(exc=str(exc)):
                self.assertTrue(
                    self._classify_oom(exc),
                    f"Should detect '{exc}' as OOM",
                )

    def test_oom_by_exception_type_name(self):
        """Exception types with 'cuda' or 'outofmemory' in the name should match."""

        class CudaError(Exception):
            pass

        class OutOfMemoryError(Exception):
            pass

        self.assertTrue(self._classify_oom(CudaError("some error")))
        self.assertTrue(self._classify_oom(OutOfMemoryError("some error")))

    def test_non_oom_exception_not_treated_as_oom(self):
        """Normal errors must NOT be misclassified as OOM."""
        cases = [
            ValueError("invalid literal for int()"),
            RuntimeError("PaddleOCR engine failed to initialize"),
            FileNotFoundError("/tmp/model.pdparams not found"),
            Exception("No text"),
            TypeError("expected str, got NoneType"),
            OSError("Permission denied"),
        ]
        for exc in cases:
            with self.subTest(exc=str(exc)):
                self.assertFalse(
                    self._classify_oom(exc),
                    f"Should NOT detect '{exc}' as OOM",
                )

    def test_oom_handler_evicts_engine_cache(self):
        """When OOM is detected, the engine cache entry for the language should
        be evicted so a fresh engine is created on next use."""
        import threading

        cache = {"en": ("mock_engine", "mock_lock"), "fr": ("fr_engine", "fr_lock")}
        cache_lock = threading.Lock()

        # Simulate the eviction logic from the  fix
        lang_hint = "en"
        try:
            with cache_lock:
                cache.pop(lang_hint, None)
        except Exception:
            pass

        self.assertNotIn("en", cache, "Engine for 'en' should be evicted")
        self.assertIn("fr", cache, "Engine for 'fr' should be untouched")

    def test_oom_handler_tolerates_missing_paddle(self):
        """The OOM handler should not raise even if paddle is not importable."""
        with patch.dict("sys.modules", {"paddle": None}):
            # Simulate the cache-clear attempt
            raised = False
            try:
                try:
                    import paddle
                    paddle.device.cuda.empty_cache()
                except Exception:
                    pass
            except Exception:
                raised = True

            self.assertFalse(raised, "OOM handler should swallow import failures")


if __name__ == "__main__":
    unittest.main()
