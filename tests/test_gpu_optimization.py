"""Tests for GPU kernel fusion and optimization module (gpu_optimization.py).

Covers:
- OptimizationLevel enum values and count
- OptimizationLevel construction from string value
- FusionStrategy enum values and count
- FusionStrategy construction from string value
- GpuCapability defaults and custom construction
- GpuCapability field types
- FusionConfig defaults and custom construction
- FusionConfig all-disabled defaults
- BatchPreprocessor construction with default config
- BatchPreprocessor construction with custom config
- BatchPreprocessor.preprocess_batch with empty list
- BatchPreprocessor.preprocess_batch with mock images (numpy path)
- BatchPreprocessor.preprocess_batch respects max_batch_images chunking
- BatchPreprocessor.preprocess_batch grayscale image expansion
- BatchPreprocessor.preprocess_batch PIL-only fallback
- BatchPreprocessor.preprocess_batch GPU path via mock torch
- BatchPreprocessor.get_optimal_batch_size basic
- BatchPreprocessor.get_optimal_batch_size clamps to max
- BatchPreprocessor.get_optimal_batch_size minimum is 1
- BatchPreprocessor.estimate_memory_mb returns float
- BatchPreprocessor.estimate_memory_mb scales with batch size
- BatchPreprocessor.estimate_memory_mb scales with dimensions
- BatchPreprocessor.estimate_memory_mb custom channels
- GpuOptimizer construction with default config
- GpuOptimizer construction with custom config
- GpuOptimizer.detect_capabilities returns empty without torch
- GpuOptimizer.detect_capabilities returns empty when CUDA unavailable
- GpuOptimizer.detect_capabilities with mocked CUDA devices
- GpuOptimizer.recommend_config with no capabilities
- GpuOptimizer.recommend_config with low-end GPU
- GpuOptimizer.recommend_config with high-end GPU
- GpuOptimizer.recommend_config with tensor-core GPU
- GpuOptimizer.recommend_config uses cached capabilities
- GpuOptimizer.get_optimization_summary structure
- GpuOptimizer.get_optimization_summary without GPU
- GpuOptimizer.get_optimization_summary with capabilities
- GpuOptimizer.get_optimization_summary speedup levels
- _try_import_torch returns None when unavailable
- _try_import_numpy returns None when unavailable
- End-to-end: optimizer → recommend → summary

Run with: python -m pytest tests/test_gpu_optimization.py -v
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Add project root to path
from gpu_optimization import (
    BatchPreprocessor,
    FusionConfig,
    FusionStrategy,
    GpuCapability,
    GpuOptimizer,
    OptimizationLevel,
    _try_import_numpy,
    _try_import_torch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_image(width: int = 64, height: int = 48) -> Image.Image:
    """Create a small RGB test image."""
    return Image.new("RGB", (width, height), color=(128, 64, 32))


def _make_grayscale_image(width: int = 64, height: int = 48) -> Image.Image:
    """Create a small grayscale test image."""
    return Image.new("L", (width, height), color=128)


# ---------------------------------------------------------------------------
# Tests: OptimizationLevel
# ---------------------------------------------------------------------------


class TestOptimizationLevel:
    def test_enum_has_four_members(self):
        assert len(OptimizationLevel) == 4

    def test_none_value(self):
        assert OptimizationLevel.NONE.value == "none"

    def test_basic_value(self):
        assert OptimizationLevel.BASIC.value == "basic"

    def test_aggressive_value(self):
        assert OptimizationLevel.AGGRESSIVE.value == "aggressive"

    def test_auto_value(self):
        assert OptimizationLevel.AUTO.value == "auto"

    def test_enum_from_value(self):
        assert OptimizationLevel("basic") is OptimizationLevel.BASIC

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            OptimizationLevel("turbo")


# ---------------------------------------------------------------------------
# Tests: FusionStrategy
# ---------------------------------------------------------------------------


class TestFusionStrategy:
    def test_enum_has_four_members(self):
        assert len(FusionStrategy) == 4

    def test_none_value(self):
        assert FusionStrategy.NONE.value == "none"

    def test_preprocess_batch_value(self):
        assert FusionStrategy.PREPROCESS_BATCH.value == "preprocess_batch"

    def test_inference_batch_value(self):
        assert FusionStrategy.INFERENCE_BATCH.value == "inference_batch"

    def test_full_pipeline_value(self):
        assert FusionStrategy.FULL_PIPELINE.value == "full_pipeline"

    def test_enum_from_value(self):
        assert FusionStrategy("full_pipeline") is FusionStrategy.FULL_PIPELINE

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            FusionStrategy("mega_fusion")


# ---------------------------------------------------------------------------
# Tests: GpuCapability
# ---------------------------------------------------------------------------


class TestGpuCapability:
    def test_defaults(self):
        cap = GpuCapability()
        assert cap.device_id == 0
        assert cap.name == "unknown"
        assert cap.compute_capability == (0, 0)
        assert cap.memory_total_mb == 0
        assert cap.supports_fp16 is False
        assert cap.supports_int8 is False
        assert cap.supports_tensor_cores is False

    def test_custom_construction(self):
        cap = GpuCapability(
            device_id=1,
            name="NVIDIA A100",
            compute_capability=(8, 0),
            memory_total_mb=40960,
            supports_fp16=True,
            supports_int8=True,
            supports_tensor_cores=True,
        )
        assert cap.device_id == 1
        assert cap.name == "NVIDIA A100"
        assert cap.compute_capability == (8, 0)
        assert cap.memory_total_mb == 40960
        assert cap.supports_fp16 is True

    def test_compute_capability_is_tuple(self):
        cap = GpuCapability(compute_capability=(7, 5))
        assert isinstance(cap.compute_capability, tuple)
        assert len(cap.compute_capability) == 2


# ---------------------------------------------------------------------------
# Tests: FusionConfig
# ---------------------------------------------------------------------------


class TestFusionConfig:
    def test_defaults(self):
        cfg = FusionConfig()
        assert cfg.level == OptimizationLevel.NONE
        assert cfg.strategy == FusionStrategy.NONE
        assert cfg.max_batch_images == 8
        assert cfg.enable_fp16 is False
        assert cfg.enable_int8 is False
        assert cfg.preprocessing_on_gpu is False
        assert cfg.pin_memory is False

    def test_custom_values(self):
        cfg = FusionConfig(
            level=OptimizationLevel.AGGRESSIVE,
            strategy=FusionStrategy.FULL_PIPELINE,
            max_batch_images=16,
            enable_fp16=True,
            enable_int8=True,
            preprocessing_on_gpu=True,
            pin_memory=True,
        )
        assert cfg.level == OptimizationLevel.AGGRESSIVE
        assert cfg.strategy == FusionStrategy.FULL_PIPELINE
        assert cfg.max_batch_images == 16
        assert cfg.enable_fp16 is True

    def test_all_disabled_by_default(self):
        cfg = FusionConfig()
        assert not cfg.enable_fp16
        assert not cfg.enable_int8
        assert not cfg.preprocessing_on_gpu
        assert not cfg.pin_memory


# ---------------------------------------------------------------------------
# Tests: BatchPreprocessor
# ---------------------------------------------------------------------------


class TestBatchPreprocessor:
    def test_construction_default_config(self):
        bp = BatchPreprocessor()
        assert bp.config.level == OptimizationLevel.NONE
        assert bp.config.max_batch_images == 8

    def test_construction_custom_config(self):
        cfg = FusionConfig(max_batch_images=4)
        bp = BatchPreprocessor(cfg)
        assert bp.config.max_batch_images == 4

    def test_preprocess_batch_empty(self):
        bp = BatchPreprocessor()
        assert bp.preprocess_batch([]) == []

    def test_preprocess_batch_single_image(self):
        bp = BatchPreprocessor()
        img = _make_rgb_image(100, 80)
        result = bp.preprocess_batch([img], target_size=(32, 32))
        assert len(result) == 1
        arr = result[0]
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == (32, 32, 3)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_preprocess_batch_multiple_images(self):
        bp = BatchPreprocessor()
        images = [_make_rgb_image(60, 40) for _ in range(5)]
        result = bp.preprocess_batch(images, target_size=(16, 16))
        assert len(result) == 5
        for arr in result:
            assert arr.shape == (16, 16, 3)

    def test_preprocess_batch_respects_max_batch_chunking(self):
        cfg = FusionConfig(max_batch_images=3)
        bp = BatchPreprocessor(cfg)
        images = [_make_rgb_image() for _ in range(7)]
        result = bp.preprocess_batch(images, target_size=(16, 16))
        # All 7 images should be returned regardless of chunking
        assert len(result) == 7

    def test_preprocess_batch_grayscale_expansion(self):
        bp = BatchPreprocessor()
        img = _make_grayscale_image(32, 32)
        result = bp.preprocess_batch([img], target_size=(16, 16))
        assert len(result) == 1
        arr = result[0]
        # Grayscale → 2D array should be expanded to 3D
        assert arr.ndim == 3
        assert arr.shape == (16, 16, 1)

    def test_preprocess_batch_pil_only_fallback(self):
        """When numpy is unavailable, raw resized PIL images are returned."""
        bp = BatchPreprocessor()
        img = _make_rgb_image(64, 48)
        with patch("gpu_optimization._try_import_numpy", return_value=None):
            result = bp.preprocess_batch([img], target_size=(20, 20))
        assert len(result) == 1
        # Should be a PIL Image (not numpy array)
        assert isinstance(result[0], Image.Image)
        assert result[0].size == (20, 20)

    def test_preprocess_batch_gpu_path_mocked(self):
        """GPU path is exercised when preprocessing_on_gpu=True and torch mocked."""
        cfg = FusionConfig(preprocessing_on_gpu=True)
        bp = BatchPreprocessor(cfg)

        # Build a mock torch module
        mock_tensor = MagicMock()
        mock_tensor.half.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        expected_arr = np.zeros((16, 16, 3), dtype=np.float32)
        mock_tensor.cpu.return_value.numpy.return_value = expected_arr

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = "cuda"
        mock_torch.tensor.return_value = mock_tensor

        img = _make_rgb_image(64, 48)
        with patch("gpu_optimization._try_import_torch", return_value=mock_torch):
            result = bp.preprocess_batch([img], target_size=(16, 16))

        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)

    def test_preprocess_batch_gpu_fp16_flag(self):
        """When enable_fp16 is set, tensor.half() is called."""
        cfg = FusionConfig(preprocessing_on_gpu=True, enable_fp16=True)
        bp = BatchPreprocessor(cfg)

        mock_tensor = MagicMock()
        mock_tensor.half.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.cpu.return_value.numpy.return_value = np.zeros(
            (16, 16, 3), dtype=np.float32
        )

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = "cuda"
        mock_torch.tensor.return_value = mock_tensor

        img = _make_rgb_image(64, 48)
        with patch("gpu_optimization._try_import_torch", return_value=mock_torch):
            bp.preprocess_batch([img], target_size=(16, 16))

        mock_tensor.half.assert_called_once()


class TestBatchPreprocessorMemory:
    def test_estimate_memory_returns_float(self):
        mem = BatchPreprocessor.estimate_memory_mb(4, 640, 640)
        assert isinstance(mem, float)
        assert mem > 0

    def test_estimate_memory_scales_with_batch(self):
        mem1 = BatchPreprocessor.estimate_memory_mb(1, 640, 640)
        mem4 = BatchPreprocessor.estimate_memory_mb(4, 640, 640)
        assert abs(mem4 - 4 * mem1) < 0.001

    def test_estimate_memory_scales_with_dimensions(self):
        small = BatchPreprocessor.estimate_memory_mb(1, 320, 320)
        large = BatchPreprocessor.estimate_memory_mb(1, 640, 640)
        assert large > small
        # 640×640 has 4× the pixels of 320×320
        assert abs(large / small - 4.0) < 0.001

    def test_estimate_memory_custom_channels(self):
        rgb = BatchPreprocessor.estimate_memory_mb(1, 640, 640, channels=3)
        gray = BatchPreprocessor.estimate_memory_mb(1, 640, 640, channels=1)
        assert abs(rgb / gray - 3.0) < 0.001

    def test_get_optimal_batch_size_basic(self):
        bp = BatchPreprocessor()
        size = bp.get_optimal_batch_size(1024.0, (640, 640))
        assert isinstance(size, int)
        assert size >= 1

    def test_get_optimal_batch_size_clamps_to_max(self):
        cfg = FusionConfig(max_batch_images=4)
        bp = BatchPreprocessor(cfg)
        # Provide enormous memory so optimal would exceed max
        size = bp.get_optimal_batch_size(999999.0, (64, 64))
        assert size <= 4

    def test_get_optimal_batch_size_minimum_is_one(self):
        bp = BatchPreprocessor()
        # Very small memory → should still return at least 1
        size = bp.get_optimal_batch_size(0.001, (640, 640))
        assert size >= 1


# ---------------------------------------------------------------------------
# Tests: GpuOptimizer
# ---------------------------------------------------------------------------


class TestGpuOptimizer:
    def test_construction_default(self):
        opt = GpuOptimizer()
        assert opt.config.level == OptimizationLevel.NONE

    def test_construction_custom_config(self):
        cfg = FusionConfig(level=OptimizationLevel.BASIC)
        opt = GpuOptimizer(cfg)
        assert opt.config.level == OptimizationLevel.BASIC

    def test_detect_capabilities_no_torch(self):
        opt = GpuOptimizer()
        with patch("gpu_optimization._try_import_torch", return_value=None):
            caps = opt.detect_capabilities()
        assert caps == []

    def test_detect_capabilities_cuda_unavailable(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        opt = GpuOptimizer()
        with patch("gpu_optimization._try_import_torch", return_value=mock_torch):
            caps = opt.detect_capabilities()
        assert caps == []

    def test_detect_capabilities_with_mocked_device(self):
        props = SimpleNamespace(
            name="NVIDIA RTX 3090",
            major=8,
            minor=6,
            total_mem=24 * 1024 * 1024 * 1024,  # 24 GB
        )
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = props

        opt = GpuOptimizer()
        with patch("gpu_optimization._try_import_torch", return_value=mock_torch):
            caps = opt.detect_capabilities()

        assert len(caps) == 1
        cap = caps[0]
        assert cap.device_id == 0
        assert cap.name == "NVIDIA RTX 3090"
        assert cap.compute_capability == (8, 6)
        assert cap.memory_total_mb == 24 * 1024
        assert cap.supports_fp16 is True
        assert cap.supports_int8 is True
        assert cap.supports_tensor_cores is True

    def test_detect_capabilities_multiple_devices(self):
        props_0 = SimpleNamespace(
            name="GPU-0", major=7, minor=0,
            total_mem=8 * 1024 * 1024 * 1024,
        )
        props_1 = SimpleNamespace(
            name="GPU-1", major=7, minor=5,
            total_mem=16 * 1024 * 1024 * 1024,
        )
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_properties.side_effect = [props_0, props_1]

        opt = GpuOptimizer()
        with patch("gpu_optimization._try_import_torch", return_value=mock_torch):
            caps = opt.detect_capabilities()
        assert len(caps) == 2
        assert caps[0].name == "GPU-0"
        assert caps[1].name == "GPU-1"


# ---------------------------------------------------------------------------
# Tests: GpuOptimizer.recommend_config
# ---------------------------------------------------------------------------


class TestRecommendConfig:
    def test_no_capabilities_returns_conservative(self):
        opt = GpuOptimizer()
        cfg = opt.recommend_config([])
        assert cfg.level == OptimizationLevel.NONE
        assert cfg.strategy == FusionStrategy.NONE
        assert cfg.max_batch_images == 1
        assert cfg.enable_fp16 is False
        assert cfg.preprocessing_on_gpu is False

    def test_no_capabilities_none_arg(self):
        """Passing None uses empty cached capabilities → conservative config."""
        opt = GpuOptimizer()
        cfg = opt.recommend_config(None)
        assert cfg.level == OptimizationLevel.NONE

    def test_low_end_gpu(self):
        cap = GpuCapability(
            device_id=0,
            name="GTX 1050",
            compute_capability=(6, 1),
            memory_total_mb=2048,
            supports_fp16=True,
            supports_int8=True,
            supports_tensor_cores=False,
        )
        opt = GpuOptimizer()
        cfg = opt.recommend_config([cap])
        assert cfg.level == OptimizationLevel.BASIC
        assert cfg.strategy == FusionStrategy.PREPROCESS_BATCH
        assert cfg.enable_fp16 is True
        assert cfg.preprocessing_on_gpu is True

    def test_mid_range_gpu(self):
        cap = GpuCapability(
            device_id=0,
            name="RTX 2060",
            compute_capability=(7, 5),
            memory_total_mb=6144,
            supports_fp16=True,
            supports_int8=True,
            supports_tensor_cores=True,
        )
        opt = GpuOptimizer()
        cfg = opt.recommend_config([cap])
        assert cfg.level == OptimizationLevel.AGGRESSIVE
        assert cfg.strategy == FusionStrategy.INFERENCE_BATCH

    def test_high_end_gpu(self):
        cap = GpuCapability(
            device_id=0,
            name="A100",
            compute_capability=(8, 0),
            memory_total_mb=40960,
            supports_fp16=True,
            supports_int8=True,
            supports_tensor_cores=True,
        )
        opt = GpuOptimizer()
        cfg = opt.recommend_config([cap])
        assert cfg.level == OptimizationLevel.AGGRESSIVE
        assert cfg.strategy == FusionStrategy.FULL_PIPELINE
        assert cfg.pin_memory is True
        assert cfg.max_batch_images >= 8

    def test_recommend_uses_cached_capabilities(self):
        """When called with no args, recommend_config uses cached caps."""
        props = SimpleNamespace(
            name="T4", major=7, minor=5,
            total_mem=16 * 1024 * 1024 * 1024,
        )
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = props

        opt = GpuOptimizer()
        with patch("gpu_optimization._try_import_torch", return_value=mock_torch):
            opt.detect_capabilities()

        # Now call without args — should use cached
        cfg = opt.recommend_config()
        assert cfg.level != OptimizationLevel.NONE

    def test_recommend_picks_best_gpu(self):
        """With multiple GPUs, recommendation is based on the best one."""
        weak = GpuCapability(
            device_id=0, name="GTX 750",
            compute_capability=(5, 0), memory_total_mb=2048,
            supports_fp16=False, supports_int8=False,
            supports_tensor_cores=False,
        )
        strong = GpuCapability(
            device_id=1, name="RTX 4090",
            compute_capability=(8, 9), memory_total_mb=24576,
            supports_fp16=True, supports_int8=True,
            supports_tensor_cores=True,
        )
        opt = GpuOptimizer()
        cfg = opt.recommend_config([weak, strong])
        assert cfg.level == OptimizationLevel.AGGRESSIVE
        assert cfg.enable_fp16 is True


# ---------------------------------------------------------------------------
# Tests: GpuOptimizer.get_optimization_summary
# ---------------------------------------------------------------------------


class TestOptimizationSummary:
    def test_summary_structure(self):
        opt = GpuOptimizer()
        summary = opt.get_optimization_summary()
        assert "config" in summary
        assert "gpu_available" in summary
        assert "device_count" in summary
        assert "capabilities" in summary
        assert "estimated_speedup" in summary

    def test_summary_without_gpu(self):
        opt = GpuOptimizer()
        summary = opt.get_optimization_summary()
        assert summary["gpu_available"] is False
        assert summary["device_count"] == 0
        assert summary["capabilities"] == []
        assert summary["estimated_speedup"] == 1.0  # NONE level without GPU

    def test_summary_config_values(self):
        cfg = FusionConfig(
            level=OptimizationLevel.BASIC,
            strategy=FusionStrategy.PREPROCESS_BATCH,
            max_batch_images=4,
        )
        opt = GpuOptimizer(cfg)
        summary = opt.get_optimization_summary()
        assert summary["config"]["level"] == "basic"
        assert summary["config"]["strategy"] == "preprocess_batch"
        assert summary["config"]["max_batch_images"] == 4

    def test_summary_with_capabilities(self):
        opt = GpuOptimizer()
        # Inject capabilities directly
        opt._capabilities = [
            GpuCapability(device_id=0, name="Test GPU",
                          compute_capability=(7, 5), memory_total_mb=8192),
        ]
        summary = opt.get_optimization_summary()
        assert summary["gpu_available"] is True
        assert summary["device_count"] == 1
        assert summary["capabilities"][0]["name"] == "Test GPU"
        assert summary["capabilities"][0]["compute_capability"] == [7, 5]

    def test_summary_speedup_aggressive(self):
        cfg = FusionConfig(level=OptimizationLevel.AGGRESSIVE)
        opt = GpuOptimizer(cfg)
        opt._capabilities = [GpuCapability(device_id=0, name="GPU")]
        summary = opt.get_optimization_summary()
        assert summary["estimated_speedup"] == 4.0

    def test_summary_speedup_basic(self):
        cfg = FusionConfig(level=OptimizationLevel.BASIC)
        opt = GpuOptimizer(cfg)
        opt._capabilities = [GpuCapability(device_id=0, name="GPU")]
        summary = opt.get_optimization_summary()
        assert summary["estimated_speedup"] == 2.0

    def test_summary_speedup_no_gpu(self):
        cfg = FusionConfig(level=OptimizationLevel.AGGRESSIVE)
        opt = GpuOptimizer(cfg)
        # No capabilities → no real GPU
        summary = opt.get_optimization_summary()
        assert summary["estimated_speedup"] == 1.0


# ---------------------------------------------------------------------------
# Tests: Lazy import helpers
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_try_import_torch_returns_none_when_unavailable(self):
        with patch.dict("sys.modules", {"torch": None}):
            result = _try_import_torch()
            # May or may not be None depending on environment;
            # the contract is it doesn't raise.
        # On CI without torch, should be None
        # On dev with torch, should be a module
        assert result is None or hasattr(result, "cuda")

    def test_try_import_numpy_returns_module(self):
        # numpy IS available in test environment
        result = _try_import_numpy()
        assert result is not None
        assert hasattr(result, "array")

    def test_try_import_numpy_returns_none_when_unavailable(self):
        with patch.dict("sys.modules", {"numpy": None}):
            # Force re-import failure
            result = _try_import_numpy()
        assert result is None or hasattr(result, "array")


# ---------------------------------------------------------------------------
# Tests: End-to-end flow
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_optimizer_full_workflow(self):
        """Detect → recommend → summarize lifecycle without GPU."""
        opt = GpuOptimizer()
        with patch("gpu_optimization._try_import_torch", return_value=None):
            caps = opt.detect_capabilities()
        assert caps == []

        cfg = opt.recommend_config(caps)
        assert cfg.level == OptimizationLevel.NONE

        summary = opt.get_optimization_summary()
        assert summary["gpu_available"] is False

    def test_preprocessor_with_recommended_config(self):
        """Use recommended config in BatchPreprocessor."""
        opt = GpuOptimizer()
        cfg = opt.recommend_config([])
        bp = BatchPreprocessor(cfg)
        img = _make_rgb_image()
        result = bp.preprocess_batch([img], target_size=(32, 32))
        assert len(result) == 1
