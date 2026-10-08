"""Tests for the shared PaddleOCR engine cache (maintenance review).

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
        self.mod._shutdown_cpu_paddle_executor()
        self.mod._start_cpu_paddle_executor_lifecycle()
        # Clear cache before each test
        with self.mod._ENGINE_CACHE_LOCK:
            self.mod._ENGINE_CACHE.clear()

    def tearDown(self):
        # Clean up cache after each test
        with self.mod._ENGINE_CACHE_LOCK:
            self.mod._ENGINE_CACHE.clear()
        self.mod._shutdown_cpu_paddle_executor()
        self.mod._start_cpu_paddle_executor_lifecycle()

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
    def test_different_devices_return_different_instances(self, mock_create):
        """CPU and GPU requests for one language must not share a cached engine."""
        gpu_engine = MagicMock(name="gpu_engine")
        cpu_engine = MagicMock(name="cpu_engine")
        mock_create.side_effect = lambda lang, device="gpu": (
            cpu_engine if device == "cpu" else gpu_engine
        )

        result_gpu = self.mod.get_or_create_engine("en", device="gpu")
        result_cpu = self.mod.get_or_create_engine("en", device="cpu")

        assert result_gpu is not result_cpu
        assert result_gpu[0] is gpu_engine
        assert result_cpu[0] is cpu_engine
        mock_create.assert_any_call("en", device="gpu")
        mock_create.assert_any_call("en", device="cpu")
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
        assert result.actual_language == "en"

    @patch("ocr_gpu_async.create_paddle_engine")
    def test_und_fallback_reports_english_model_language(self, mock_create):
        """Requested und that loads English weights is model language en."""
        en_weights = MagicMock(name="en_weights")

        def fail_und(lang, device="gpu"):
            if lang == "und":
                raise RuntimeError("no und model")
            return en_weights

        mock_create.side_effect = fail_und
        result = self.mod.get_or_create_engine("und", device="gpu")
        assert result is not None
        assert result[0] is en_weights
        assert result.actual_language == "en"

    def test_stale_failure_cannot_evict_new_predictor_generation(self):
        cache_key = self.mod._engine_cache_key("en", "gpu")
        stale = self.mod._EngineCacheEntry(
            MagicMock(name="stale"),
            threading.Lock(),
            "en",
        )
        replacement = self.mod._EngineCacheEntry(
            MagicMock(name="replacement"),
            threading.Lock(),
            "en",
        )
        with self.mod._ENGINE_CACHE_LOCK:
            self.mod._ENGINE_CACHE[cache_key] = replacement

        assert self.mod._evict_engine_cache_entry(cache_key, stale) is False
        assert self.mod._ENGINE_CACHE[cache_key] is replacement
        assert self.mod._evict_engine_cache_entry(cache_key, replacement) is True
        assert cache_key not in self.mod._ENGINE_CACHE

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
        mock_create.side_effect = lambda lang, device="gpu": MagicMock(
            name=f"{lang}_engine"
        )

        result_en = self.mod.get_or_create_engine("en", device="gpu")
        result_fr = self.mod.get_or_create_engine("fr", device="gpu")

        _, lock_en = result_en
        _, lock_fr = result_fr
        assert (
            lock_en is not lock_fr
        ), "Each language should have its own inference lock"

    def test_empty_cuda_visibility_selects_cpu(self):
        """The public CPU image must not construct a nominal GPU predictor."""
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}):
            assert self.mod._detect_default_paddle_device() == "cpu"

    @patch("ocr_gpu_async.create_paddle_engine")
    def test_cpu_engine_construction_and_inference_share_owner_thread(
        self, mock_create
    ):
        """CPU predictor construction and use must never migrate threads."""
        construction_threads = []
        inference_threads = []

        class FakeEngine:
            def ocr(self, image):
                inference_threads.append(threading.get_ident())
                return image

        def create(_lang, device):
            construction_threads.append(threading.get_ident())
            return FakeEngine()

        mock_create.side_effect = create
        engine, lock = self.mod.get_or_create_engine("en", device="cpu")
        caller_thread = threading.get_ident()
        result = self.mod._run_paddle_ocr(engine, lock, "page", "cpu")

        assert result == "page"
        assert construction_threads == inference_threads
        assert construction_threads[0] != caller_thread

    @patch("ocr_gpu_async.create_paddle_engine")
    def test_shutdown_evicts_predictors_bound_to_retired_cpu_thread(self, mock_create):
        """A second in-process run must reconstruct CPU predictors."""
        mock_create.side_effect = [MagicMock(name="first"), MagicMock(name="second")]

        first, _ = self.mod.get_or_create_engine("en", device="cpu")
        self.mod._shutdown_cpu_paddle_executor()
        with self.assertRaisesRegex(RuntimeError, "closed for shutdown"):
            self.mod.get_or_create_engine("en", device="cpu")
        self.mod._start_cpu_paddle_executor_lifecycle()
        second, _ = self.mod.get_or_create_engine("en", device="cpu")

        assert first is not second
        assert mock_create.call_count == 2

    def test_closed_lifecycle_rejects_late_cpu_submission(self):
        """A timed-out worker cannot recreate an owner thread during teardown."""
        self.mod._close_cpu_paddle_executor_lifecycle()

        with self.assertRaisesRegex(RuntimeError, "closed for shutdown"):
            self.mod._call_on_cpu_paddle_thread(lambda: "late")

    @patch("ocr_gpu_async.create_paddle_engine")
    def test_cpu_replicas_construct_and_infer_concurrently_on_owners(
        self, mock_create
    ):
        """Independent replicas overlap while each predictor stays on its owner."""
        if self.mod.CPU_PADDLE_REPLICAS < 2:
            self.skipTest("CPU predictor capacity configured to one replica")
        construct_barrier = threading.Barrier(2, timeout=3)
        infer_barrier = threading.Barrier(2, timeout=3)
        owner_ids = {}
        errors = []

        class FakeEngine:
            def __init__(self, owner):
                self.owner = owner

            def ocr(self, image):
                assert threading.get_ident() == self.owner
                infer_barrier.wait()
                return image

        def create(_lang, device):
            assert device == "cpu"
            owner = threading.get_ident()
            construct_barrier.wait()
            return FakeEngine(owner)

        mock_create.side_effect = create

        def run(replica_id):
            try:
                entry = self.mod.get_or_create_engine(
                    "en", device="cpu", replica_id=replica_id
                )
                engine, lock = entry
                owner_ids[replica_id] = engine.owner
                assert self.mod._run_paddle_ocr(
                    engine, lock, replica_id, "cpu", replica_id=replica_id
                ) == replica_id
            except Exception as exc:
                errors.append(exc)

        callers = [threading.Thread(target=run, args=(index,)) for index in (0, 1)]
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join(timeout=5)
        assert not any(caller.is_alive() for caller in callers)
        assert not errors
        assert len(set(owner_ids.values())) == 2
        assert mock_create.call_count == 2


if __name__ == "__main__":
    unittest.main()
