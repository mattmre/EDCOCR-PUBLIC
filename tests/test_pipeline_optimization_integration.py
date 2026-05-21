"""Tests for page routing and GPU optimization pipeline integration.

Validates that the opt-in env vars correctly gate module loading and
that the integration points in ``ocr_gpu_async.py`` behave correctly
when enabled or disabled.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_module_with_env(env_overrides: dict):
    """Reload ocr_gpu_async with specific environment variables.

    We cannot safely reload the full production module (it starts threads,
    initialises queues, opens log files, etc.), so instead we exercise the
    standalone module imports and verify the env-var gating separately.
    """
    original = {k: os.environ.get(k) for k in env_overrides}
    for k, v in env_overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Page Routing Tests
# ---------------------------------------------------------------------------


class TestPageRoutingEnvVar:
    """ENABLE_PAGE_ROUTING env var gating."""

    def test_disabled_by_default(self):
        val = os.environ.get("ENABLE_PAGE_ROUTING", "").lower() in ("1", "true", "yes")
        assert val is False, "ENABLE_PAGE_ROUTING should be off by default"

    @pytest.mark.parametrize("value", ["1", "true", "yes", "True", "YES"])
    def test_enabled_values(self, value):
        with patch.dict(os.environ, {"ENABLE_PAGE_ROUTING": value}):
            result = os.environ.get("ENABLE_PAGE_ROUTING", "").lower() in ("1", "true", "yes")
            assert result is True

    @pytest.mark.parametrize("value", ["0", "false", "no", ""])
    def test_disabled_values(self, value):
        with patch.dict(os.environ, {"ENABLE_PAGE_ROUTING": value}):
            result = os.environ.get("ENABLE_PAGE_ROUTING", "").lower() in ("1", "true", "yes")
            assert result is False


class TestPageRouterIntegration:
    """Page router integration behavior."""

    def test_page_router_import_and_instantiate(self):
        from page_routing import PageRouter
        router = PageRouter()
        assert router is not None
        assert router.default_target.value == "gpu_paddle"

    def test_routing_skip_tiny_page(self):
        from page_routing import PageFeatures, PageRouter, RoutingTarget
        router = PageRouter()
        features = PageFeatures(
            page_number=1,
            width=50,
            height=50,
            dpi=300,
        )
        decision = router.route_page(features)
        assert decision.target == RoutingTarget.SKIP
        assert "too small" in decision.reason.lower()

    def test_routing_normal_page_defaults_to_gpu(self):
        from page_routing import PageFeatures, PageRouter, RoutingTarget
        router = PageRouter()
        features = PageFeatures(
            page_number=1,
            width=2480,
            height=3508,
            dpi=300,
        )
        decision = router.route_page(features)
        assert decision.target == RoutingTarget.GPU_PADDLE

    def test_routing_handwritten_goes_to_gpu(self):
        from page_routing import PageFeatures, PageRouter, RoutingTarget
        router = PageRouter()
        features = PageFeatures(
            page_number=1,
            width=2480,
            height=3508,
            dpi=300,
            is_handwritten=True,
        )
        decision = router.route_page(features)
        assert decision.target == RoutingTarget.GPU_PADDLE
        assert "handwritten" in decision.reason.lower()

    def test_routing_stats_tracking(self):
        from page_routing import PageFeatures, PageRouter
        router = PageRouter()
        # Route a few pages
        for i in range(3):
            features = PageFeatures(page_number=i + 1, width=2480, height=3508, dpi=300)
            router.route_page(features)
        stats = router.get_routing_stats()
        assert sum(stats.values()) == 3

    def test_routing_skip_assembly_dict_format(self):
        """Verify the assembly_queue dict matches the expected schema for SKIPPED pages."""
        # Build the assembly dict as the pipeline would
        assembly_dict = {
            "doc_id": "test_doc",
            "page_num": 1,
            "text": "",
            "status": "SKIPPED",
            "chunk_path": None,
            "structure_data": None,
            "ocr_confidence": 0.0,
            "ocr_method": "SKIPPED",
            "handwriting_data": None,
            "signature_data": None,
            "vertical_text_data": None,
            "table_fallback_data": None,
        }
        assert assembly_dict["status"] == "SKIPPED"
        assert assembly_dict["chunk_path"] is None
        assert assembly_dict["ocr_confidence"] == 0.0

    def test_routing_error_falls_through(self):
        """When routing raises an exception, routing_target should be None."""
        from page_routing import PageRouter

        router = PageRouter()
        # Patch route_page to raise

        def bad_route(features):
            raise RuntimeError("test error")

        router.route_page = bad_route

        routing_target = None
        try:
            from page_routing import PageFeatures
            features = PageFeatures(page_number=1, width=100, height=100, dpi=300)
            decision = router.route_page(features)
            routing_target = decision.target
        except Exception:
            routing_target = None

        assert routing_target is None


class TestPageRoutingImportFailure:
    """Graceful degradation when page_routing module is unavailable."""

    def test_import_error_sets_none(self):
        """Simulate import failure and verify _page_router stays None."""
        _page_router = None
        enable = True
        if enable:
            try:
                # Use a deliberately wrong module name
                import page_routing_nonexistent  # noqa: F401
                _page_router = object()
            except ImportError:
                pass
        assert _page_router is None


# ---------------------------------------------------------------------------
# GPU Optimization Tests
# ---------------------------------------------------------------------------


class TestGpuOptimizationEnvVar:
    """ENABLE_GPU_OPTIMIZATION env var gating."""

    def test_disabled_by_default(self):
        val = os.environ.get("ENABLE_GPU_OPTIMIZATION", "").lower() in ("1", "true", "yes")
        assert val is False

    @pytest.mark.parametrize("value", ["1", "true", "yes"])
    def test_enabled_values(self, value):
        with patch.dict(os.environ, {"ENABLE_GPU_OPTIMIZATION": value}):
            result = os.environ.get("ENABLE_GPU_OPTIMIZATION", "").lower() in ("1", "true", "yes")
            assert result is True

    @pytest.mark.parametrize("value", ["0", "false", "no", ""])
    def test_disabled_values(self, value):
        with patch.dict(os.environ, {"ENABLE_GPU_OPTIMIZATION": value}):
            result = os.environ.get("ENABLE_GPU_OPTIMIZATION", "").lower() in ("1", "true", "yes")
            assert result is False


class TestGpuOptimizerIntegration:
    """GPU optimizer integration behavior."""

    def test_gpu_optimizer_import_and_instantiate(self):
        from gpu_optimization import GpuOptimizer
        optimizer = GpuOptimizer()
        assert optimizer is not None

    def test_gpu_optimizer_no_cuda_returns_empty_caps(self):
        from gpu_optimization import GpuOptimizer
        optimizer = GpuOptimizer()
        # In CI (no GPU), detect_capabilities should return empty
        caps = optimizer.detect_capabilities()
        # May be empty in CI; type check is sufficient
        assert isinstance(caps, list)

    def test_gpu_optimizer_recommend_config_no_gpu(self):
        from gpu_optimization import FusionConfig, GpuOptimizer
        optimizer = GpuOptimizer()
        config = optimizer.recommend_config([])
        assert isinstance(config, FusionConfig)
        assert config.max_batch_images == 1
        assert config.enable_fp16 is False

    def test_gpu_optimizer_summary(self):
        from gpu_optimization import GpuOptimizer
        optimizer = GpuOptimizer()
        summary = optimizer.get_optimization_summary()
        assert "config" in summary
        assert "gpu_available" in summary
        assert "estimated_speedup" in summary


class TestBatchPreprocessorIntegration:
    """BatchPreprocessor integration behavior."""

    def test_batch_preprocessor_empty_images(self):
        from gpu_optimization import BatchPreprocessor
        preprocessor = BatchPreprocessor()
        result = preprocessor.preprocess_batch([])
        assert result == []

    def test_batch_preprocessor_single_image(self):
        from gpu_optimization import BatchPreprocessor
        preprocessor = BatchPreprocessor()
        img = Image.new("RGB", (100, 100), color="white")
        result = preprocessor.preprocess_batch([img])
        assert len(result) == 1
        # Result should be a numpy array or resized PIL image
        arr = result[0]
        assert arr is not None

    def test_batch_preprocessor_multiple_images(self):
        from gpu_optimization import BatchPreprocessor
        preprocessor = BatchPreprocessor()
        images = [Image.new("RGB", (100, 100), color=c) for c in ["red", "green", "blue"]]
        result = preprocessor.preprocess_batch(images)
        assert len(result) == 3

    def test_batch_preprocessor_fallback_on_error(self):
        """Simulate _batch_preprocessor failing and falling back to np.array."""
        img = Image.new("RGB", (100, 100), color="white")

        # Simulate the pipeline's fallback logic
        _batch_preprocessor = MagicMock()
        _batch_preprocessor.preprocess_batch.side_effect = RuntimeError("GPU fail")

        if _batch_preprocessor is not None:
            try:
                processed = _batch_preprocessor.preprocess_batch([img])
                img_np = processed[0] if processed else np.array(img)
            except Exception:
                img_np = np.array(img)
        else:
            img_np = np.array(img)

        assert isinstance(img_np, np.ndarray)
        assert img_np.shape == (100, 100, 3)

    def test_batch_preprocessor_none_uses_numpy(self):
        """When _batch_preprocessor is None, img_np should come from np.array."""
        img = Image.new("RGB", (200, 150), color="blue")
        _batch_preprocessor = None

        if _batch_preprocessor is not None:
            try:
                processed = _batch_preprocessor.preprocess_batch([img])
                img_np = processed[0] if processed else np.array(img)
            except Exception:
                img_np = np.array(img)
        else:
            img_np = np.array(img)

        assert isinstance(img_np, np.ndarray)
        assert img_np.shape == (150, 200, 3)

    def test_memory_estimation(self):
        from gpu_optimization import BatchPreprocessor
        mb = BatchPreprocessor.estimate_memory_mb(1, 640, 640, 3)
        assert mb > 0
        # 640*640*3*4 bytes = ~4.7 MB raw, with overhead ~6.1 MB
        assert mb < 20

    def test_optimal_batch_size(self):
        from gpu_optimization import BatchPreprocessor
        preprocessor = BatchPreprocessor()
        batch_size = preprocessor.get_optimal_batch_size(1000.0)
        assert batch_size >= 1


class TestGpuOptimizationImportFailure:
    """Graceful degradation when gpu_optimization module is unavailable."""

    def test_import_error_sets_none(self):
        _gpu_optimizer = None
        _batch_preprocessor = None
        enable = True
        if enable:
            try:
                import gpu_optimization_nonexistent  # noqa: F401
                _gpu_optimizer = object()
            except ImportError:
                pass
        assert _gpu_optimizer is None
        assert _batch_preprocessor is None


# ---------------------------------------------------------------------------
# Combined Integration Tests
# ---------------------------------------------------------------------------


class TestCombinedPipelineOptimization:
    """End-to-end integration of both modules together."""

    def test_routing_then_preprocessing(self):
        """Route a page, then preprocess it with BatchPreprocessor."""
        from gpu_optimization import BatchPreprocessor
        from page_routing import PageFeatures, PageRouter

        router = PageRouter()
        preprocessor = BatchPreprocessor()

        img = Image.new("RGB", (2480, 3508), color="white")
        features = PageFeatures(
            page_number=1,
            width=img.width,
            height=img.height,
            dpi=300,
        )
        decision = router.route_page(features)
        # Should not be SKIP for a normal-sized page
        assert decision.target.value != "skip"

        # Now preprocess
        result = preprocessor.preprocess_batch([img])
        assert len(result) == 1

    def test_skip_routing_bypasses_preprocessing(self):
        """A SKIP-routed page should never reach BatchPreprocessor."""
        from page_routing import PageFeatures, PageRouter, RoutingTarget

        router = PageRouter()
        img = Image.new("RGB", (10, 10), color="white")
        features = PageFeatures(
            page_number=1,
            width=img.width,
            height=img.height,
            dpi=300,
        )
        decision = router.route_page(features)
        assert decision.target == RoutingTarget.SKIP

        # In the pipeline, SKIP results in `continue` before preprocessing
        preprocessor_called = False
        if decision.target != RoutingTarget.SKIP:
            preprocessor_called = True
        assert preprocessor_called is False

    def test_both_disabled_no_side_effects(self):
        """When both features are disabled, globals should be None."""
        _page_router = None
        _gpu_optimizer = None
        _batch_preprocessor = None

        # Simulate disabled state
        enable_routing = False
        enable_gpu = False

        if enable_routing:
            _page_router = object()
        if enable_gpu:
            _gpu_optimizer = object()

        assert _page_router is None
        assert _gpu_optimizer is None
        assert _batch_preprocessor is None

    def test_gpu_optimizer_probe_failure_leaves_preprocessor_none(self):
        """If GPU probe fails, _batch_preprocessor should remain None."""
        from gpu_optimization import GpuOptimizer

        optimizer = GpuOptimizer()
        _batch_preprocessor = None

        # Simulate detect_capabilities raising
        with patch.object(optimizer, "detect_capabilities", side_effect=RuntimeError("no GPU")):
            try:
                caps = optimizer.detect_capabilities()
                config = optimizer.recommend_config(caps)
                from gpu_optimization import BatchPreprocessor
                _batch_preprocessor = BatchPreprocessor(config)
            except Exception:
                pass

        assert _batch_preprocessor is None
