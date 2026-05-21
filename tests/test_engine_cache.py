"""Tests for the shared PaddleOCR engine cache.

The shared cache ensures only ONE engine per language exists across all
GPU worker threads, reducing VRAM from O(threads x languages) to O(languages).
"""

import sys
import threading
import types
import unittest
from unittest.mock import MagicMock, patch


def _stub_paddleocr():
    """Create a stub paddleocr module so ocr_gpu_async can import."""
    mod = types.ModuleType("paddleocr")
    mod.PaddleOCR = MagicMock
    return mod


def _import_ocr_module():
    """Import ocr_gpu_async with heavy dependencies stubbed out."""
    # Ensure paddleocr is importable (mocked)
    if "paddleocr" not in sys.modules:
        sys.modules["paddleocr"] = _stub_paddleocr()
    import ocr_gpu_async
    return ocr_gpu_async


class TestSharedEngineCache(unittest.TestCase):
    """Verify the module-level _ENGINE_CACHE and get_or_create_engine()."""

    def setUp(self):
        self.mod = _import_ocr_module()
        # Clear cache before each test
        with self.mod._ENGINE_CACHE_LOCK:
            self.mod._ENGINE_CACHE.clear()

    def tearDown(self):
        # Clean up cache after each test
        with self.mod._ENGINE_CACHE_LOCK:
            self.mod._ENGINE_CACHE.clear()

    @patch("ocr_gpu_async.create_paddle_engine")
    def test_shared_cache_returns_same_instance(self, mock_create):
        """Two calls for the same language should return the same engine tuple."""
        sentinel_engine = MagicMock(name="en_engine")
        mock_create.return_value = sentinel_engine

        result_1 = self.mod.get_or_create_engine("en", device="gpu")
        result_2 = self.mod.get_or_create_engine("en", device="gpu")

        # Should be the exact same tuple object
        assert result_1 is result_2
        # create_paddle_engine should only have been called once
        mock_create.assert_called_once_with("en", device="gpu")
        # Result is a (engine, Lock) tuple
        engine, lock = result_1
        assert engine is sentinel_engine
        assert isinstance(lock, type(threading.Lock()))

    @patch("ocr_gpu_async.create_paddle_engine")
    def test_different_langs_return_different_instances(self, mock_create):
        """Different language codes should produce separate cache entries."""
        en_engine = MagicMock(name="en_engine")
        fr_engine = MagicMock(name="fr_engine")
        mock_create.side_effect = lambda lang, device="gpu": (
            en_engine if lang == "en" else fr_engine
        )

        result_en = self.mod.get_or_create_engine("en", device="gpu")
        result_fr = self.mod.get_or_create_engine("fr", device="gpu")

        assert result_en is not result_fr
        assert result_en[0] is en_engine
        assert result_fr[0] is fr_engine
        assert mock_create.call_count == 2

    @patch("ocr_gpu_async.create_paddle_engine")
    def test_cache_thread_safety(self, mock_create):
        """Concurrent get_or_create from N threads returns the same object."""
        sentinel_engine = MagicMock(name="shared_engine")
        # Simulate slow model loading to increase contention
        call_count = 0
        call_lock = threading.Lock()

        def slow_create(lang, device="gpu"):
            nonlocal call_count
            with call_lock:
                call_count += 1
            return sentinel_engine

        mock_create.side_effect = slow_create

        results = [None] * 20
        barrier = threading.Barrier(20)

        def worker(idx):
            barrier.wait()  # Synchronize all threads to start at once
            results[idx] = self.mod.get_or_create_engine("en", device="gpu")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # All threads should get the exact same tuple
        first = results[0]
        for i, r in enumerate(results):
            assert r is first, f"Thread {i} got a different instance"

        # Engine creation should have happened exactly once
        assert call_count == 1, f"Engine was created {call_count} times instead of 1"

    @patch("ocr_gpu_async.create_paddle_engine")
    def test_fallback_to_english_on_failure(self, mock_create):
        """If primary language fails, cache should store English fallback."""
        en_fallback = MagicMock(name="en_fallback")

        def fail_then_fallback(lang, device="gpu"):
            if lang == "xx_nonexistent":
                raise RuntimeError("Model not found")
            return en_fallback

        mock_create.side_effect = fail_then_fallback

        result = self.mod.get_or_create_engine("xx_nonexistent", device="gpu")
        assert result is not None
        engine, lock = result
        assert engine is en_fallback

    @patch("ocr_gpu_async.create_paddle_engine")
    def test_total_failure_cached_as_none(self, mock_create):
        """If both primary and English fallback fail, None is cached."""
        mock_create.side_effect = RuntimeError("All models broken")

        result = self.mod.get_or_create_engine("xx_broken", device="gpu")
        assert result is None

        # Second call should return cached None without calling create again
        mock_create.reset_mock()
        result_2 = self.mod.get_or_create_engine("xx_broken", device="gpu")
        assert result_2 is None
        mock_create.assert_not_called()

    @patch("ocr_gpu_async.create_paddle_engine")
    def test_inference_lock_is_per_engine(self, mock_create):
        """Each language engine should get its own inference lock."""
        mock_create.side_effect = lambda lang, device="gpu": MagicMock(name=f"{lang}_engine")

        result_en = self.mod.get_or_create_engine("en", device="gpu")
        result_fr = self.mod.get_or_create_engine("fr", device="gpu")

        _, lock_en = result_en
        _, lock_fr = result_fr
        assert lock_en is not lock_fr, "Each language should have its own inference lock"


if __name__ == "__main__":
    unittest.main()
