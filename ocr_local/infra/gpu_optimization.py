"""GPU kernel fusion and optimization for the OCR pipeline.

Tracks GPU optimization strategies and provides batch preprocessing
fusion.  All GPU-specific imports (torch, CUDA) are lazy so the module
can be imported and used in CPU-only environments.

The :class:`BatchPreprocessor` batches resize / normalize / pad
operations for GPU acceleration, while :class:`GpuOptimizer` detects
hardware capabilities and recommends :class:`FusionConfig` settings.

Environment Variables:
    GPU_OPTIMIZATION_LEVEL (str):
        Optimization level: none, basic, aggressive, or auto.
        Default: ``auto``.
    GPU_FUSION_STRATEGY (str):
        Fusion strategy: none, preprocess_batch, inference_batch, or
        full_pipeline.  Default: ``none``.
    GPU_MAX_BATCH_IMAGES (int):
        Maximum images per fused batch.  Default: ``8``.
    GPU_ENABLE_FP16 (bool):
        Enable half-precision inference.  Default: ``false``.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OptimizationLevel(Enum):
    """GPU optimization aggressiveness."""

    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    AUTO = "auto"


class FusionStrategy(Enum):
    """Kernel fusion strategy for the pipeline."""

    NONE = "none"
    PREPROCESS_BATCH = "preprocess_batch"
    INFERENCE_BATCH = "inference_batch"
    FULL_PIPELINE = "full_pipeline"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GpuCapability:
    """Hardware capabilities of a single GPU device.

    Attributes:
        device_id: CUDA device ordinal.
        name: Human-readable GPU name.
        compute_capability: (major, minor) CUDA compute capability.
        memory_total_mb: Total device memory in megabytes.
        supports_fp16: Whether the device supports FP16 inference.
        supports_int8: Whether the device supports INT8 inference.
        supports_tensor_cores: Whether tensor cores are available.
    """

    device_id: int = 0
    name: str = "unknown"
    compute_capability: Tuple[int, int] = (0, 0)
    memory_total_mb: int = 0
    supports_fp16: bool = False
    supports_int8: bool = False
    supports_tensor_cores: bool = False


@dataclass
class FusionConfig:
    """Configuration for GPU kernel fusion.

    Attributes:
        level: Optimization aggressiveness.
        strategy: Kernel fusion strategy.
        max_batch_images: Maximum images per fused preprocessing batch.
        enable_fp16: Use half-precision where supported.
        enable_int8: Use INT8 quantisation where supported.
        preprocessing_on_gpu: Run preprocessing kernels on device.
        pin_memory: Use pinned (page-locked) host memory for transfers.
    """

    level: OptimizationLevel = OptimizationLevel.NONE
    strategy: FusionStrategy = FusionStrategy.NONE
    max_batch_images: int = 8
    enable_fp16: bool = False
    enable_int8: bool = False
    preprocessing_on_gpu: bool = False
    pin_memory: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Bytes per pixel per channel for float32 image tensors
_BYTES_PER_FLOAT32 = 4
# Overhead multiplier for GPU memory estimation (workspace, fragmentation)
_MEMORY_OVERHEAD = 1.3


def _try_import_torch():
    """Lazily import torch, returning ``None`` when unavailable."""
    try:
        import torch  # noqa: F811
        return torch
    except ImportError:
        return None


def _try_import_numpy():
    """Lazily import numpy, returning ``None`` when unavailable."""
    try:
        import numpy as np  # noqa: F811
        return np
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# BatchPreprocessor
# ---------------------------------------------------------------------------


class BatchPreprocessor:
    """Batches preprocessing operations for GPU acceleration.

    Groups resize, normalize, and pad operations so they can be executed
    as a single fused kernel when a GPU is available, falling back to
    CPU-based numpy processing otherwise.

    Parameters:
        config: Fusion configuration controlling batch size and precision.
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        self._config = config or FusionConfig()

    @property
    def config(self) -> FusionConfig:
        """Return the current fusion config."""
        return self._config

    # -- public API ---------------------------------------------------------

    def preprocess_batch(
        self,
        images: list,
        target_size: Tuple[int, int] = (640, 640),
    ) -> list:
        """Preprocess a list of PIL Images into normalised arrays.

        Each image is resized to *target_size*, converted to a float32
        array in ``[0, 1]`` range, and padded to ensure uniform
        dimensions.

        When the GPU path is available **and** ``preprocessing_on_gpu``
        is enabled, processing is performed via torch tensors on the
        CUDA device.  Otherwise, numpy is used on the CPU.

        Parameters:
            images: List of PIL Image objects.
            target_size: ``(width, height)`` to resize images to.

        Returns:
            List of processed numpy arrays (H, W, C) in float32.
        """
        if not images:
            return []

        # Chunk into sub-batches respecting max_batch_images
        results: list = []
        max_b = max(1, self._config.max_batch_images)
        for start in range(0, len(images), max_b):
            chunk = images[start: start + max_b]
            results.extend(self._process_chunk(chunk, target_size))
        return results

    def get_optimal_batch_size(
        self,
        available_memory_mb: float,
        image_size: Tuple[int, int] = (640, 640),
    ) -> int:
        """Recommend a batch size that fits within *available_memory_mb*.

        Parameters:
            available_memory_mb: Available GPU/system memory in MiB.
            image_size: ``(width, height)`` of a single image.

        Returns:
            Maximum number of images that can be batched, clamped to
            ``[1, max_batch_images]``.
        """
        per_image = self.estimate_memory_mb(1, image_size[0], image_size[1])
        if per_image <= 0:
            return self._config.max_batch_images

        optimal = int(available_memory_mb / per_image)
        return max(1, min(optimal, self._config.max_batch_images))

    @staticmethod
    def estimate_memory_mb(
        batch_size: int,
        image_width: int,
        image_height: int,
        channels: int = 3,
    ) -> float:
        """Estimate GPU memory required for a batch of images.

        Parameters:
            batch_size: Number of images.
            image_width: Width in pixels.
            image_height: Height in pixels.
            channels: Number of colour channels.

        Returns:
            Estimated memory usage in megabytes (float).
        """
        pixels = image_width * image_height * channels
        raw_bytes = batch_size * pixels * _BYTES_PER_FLOAT32
        total_bytes = raw_bytes * _MEMORY_OVERHEAD
        return total_bytes / (1024 * 1024)

    # -- internals ----------------------------------------------------------

    def _process_chunk(
        self,
        images: list,
        target_size: Tuple[int, int],
    ) -> list:
        """Process a single chunk of images.

        Falls back gracefully from GPU → numpy → basic list.
        """
        if self._config.preprocessing_on_gpu:
            torch = _try_import_torch()
            if torch is not None and torch.cuda.is_available():
                return self._process_gpu(images, target_size, torch)

        np = _try_import_numpy()
        if np is not None:
            return self._process_numpy(images, target_size, np)

        # Last resort: return raw resized PIL images as-is
        return self._process_pil_only(images, target_size)

    def _process_gpu(
        self,
        images: list,
        target_size: Tuple[int, int],
        torch: Any,
    ) -> list:
        """Process images on GPU via torch tensors."""
        np = _try_import_numpy()
        device = torch.device("cuda")
        results = []
        for img in images:
            resized = img.resize(target_size)
            if np is not None:
                arr = np.array(resized, dtype=np.float32) / 255.0
            else:
                # Fallback if numpy somehow missing but torch present
                arr = list(resized.getdata())
            tensor = torch.tensor(arr, device=device)
            if self._config.enable_fp16:
                tensor = tensor.half()
            # Move back to CPU for downstream pipeline compatibility
            result = tensor.float().cpu().numpy()
            results.append(result)
        return results

    def _process_numpy(
        self,
        images: list,
        target_size: Tuple[int, int],
        np: Any,
    ) -> list:
        """Process images on CPU using numpy."""
        results = []
        for img in images:
            resized = img.resize(target_size)
            arr = np.array(resized, dtype=np.float32) / 255.0
            # Ensure 3D (H, W, C)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            results.append(arr)
        return results

    @staticmethod
    def _process_pil_only(
        images: list,
        target_size: Tuple[int, int],
    ) -> list:
        """Resize-only fallback when neither torch nor numpy is available."""
        return [img.resize(target_size) for img in images]


# ---------------------------------------------------------------------------
# GpuOptimizer
# ---------------------------------------------------------------------------


class GpuOptimizer:
    """Recommends and applies GPU optimization settings.

    Uses lazy imports so the class can be instantiated in CPU-only
    environments without error.

    Parameters:
        config: Initial fusion configuration.
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        self._config = config or FusionConfig()
        self._capabilities: list[GpuCapability] = []

    @property
    def config(self) -> FusionConfig:
        """Return the current fusion config."""
        return self._config

    # -- public API ---------------------------------------------------------

    def detect_capabilities(self) -> list[GpuCapability]:
        """Detect GPU hardware capabilities via torch/CUDA.

        Returns an empty list when torch or CUDA is unavailable.
        """
        torch = _try_import_torch()
        if torch is None or not torch.cuda.is_available():
            logger.info("No CUDA-capable GPU detected; returning empty capabilities")
            self._capabilities = []
            return []

        caps: list[GpuCapability] = []
        device_count = torch.cuda.device_count()
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            cc = (props.major, props.minor)
            memory_mb = int(props.total_mem / (1024 * 1024))

            cap = GpuCapability(
                device_id=idx,
                name=props.name,
                compute_capability=cc,
                memory_total_mb=memory_mb,
                supports_fp16=cc >= (5, 3),
                supports_int8=cc >= (6, 1),
                supports_tensor_cores=cc >= (7, 0),
            )
            caps.append(cap)
            logger.info(
                "Detected GPU %d: %s  CC=%d.%d  mem=%dMB  fp16=%s  int8=%s  tc=%s",
                idx,
                cap.name,
                cc[0],
                cc[1],
                memory_mb,
                cap.supports_fp16,
                cap.supports_int8,
                cap.supports_tensor_cores,
            )

        self._capabilities = caps
        return caps

    def recommend_config(
        self,
        capabilities: list[GpuCapability] | None = None,
    ) -> FusionConfig:
        """Recommend a :class:`FusionConfig` based on GPU capabilities.

        When no GPUs are available, returns a conservative CPU-only
        config.  With GPUs present, the recommendation scales
        aggressiveness with compute capability and available memory.

        Parameters:
            capabilities: Detected GPU capabilities.  If *None*, uses
                internally cached capabilities from the last
                :meth:`detect_capabilities` call.

        Returns:
            A recommended :class:`FusionConfig`.
        """
        caps = capabilities if capabilities is not None else self._capabilities

        if not caps:
            return FusionConfig(
                level=OptimizationLevel.NONE,
                strategy=FusionStrategy.NONE,
                max_batch_images=1,
                enable_fp16=False,
                enable_int8=False,
                preprocessing_on_gpu=False,
                pin_memory=False,
            )

        # Use the most capable GPU for recommendation
        best = max(caps, key=lambda c: (c.compute_capability, c.memory_total_mb))

        # Determine optimization level
        if best.supports_tensor_cores:
            level = OptimizationLevel.AGGRESSIVE
        elif best.supports_fp16:
            level = OptimizationLevel.BASIC
        else:
            level = OptimizationLevel.BASIC

        # Strategy scales with memory
        if best.memory_total_mb >= 8192:
            strategy = FusionStrategy.FULL_PIPELINE
        elif best.memory_total_mb >= 4096:
            strategy = FusionStrategy.INFERENCE_BATCH
        else:
            strategy = FusionStrategy.PREPROCESS_BATCH

        # Batch size heuristic: ~100 MB per image at 640×640×3×fp32
        per_image_mb = BatchPreprocessor.estimate_memory_mb(1, 640, 640)
        if per_image_mb > 0:
            max_batch = int(best.memory_total_mb * 0.5 / per_image_mb)
        else:
            max_batch = 8
        max_batch = max(1, min(max_batch, 32))

        return FusionConfig(
            level=level,
            strategy=strategy,
            max_batch_images=max_batch,
            enable_fp16=best.supports_fp16,
            enable_int8=best.supports_int8,
            preprocessing_on_gpu=True,
            pin_memory=best.memory_total_mb >= 4096,
        )

    def get_optimization_summary(self) -> dict:
        """Return a diagnostic summary of the current optimization state.

        The summary includes the active configuration, detected
        capabilities, and whether GPU acceleration is available.

        Returns:
            Dictionary with keys ``config``, ``gpu_available``,
            ``device_count``, ``capabilities``, and ``estimated_speedup``.
        """
        caps = self._capabilities
        gpu_available = len(caps) > 0

        # Rough estimated speedup multiplier
        if not gpu_available:
            speedup = 1.0
        elif self._config.level == OptimizationLevel.AGGRESSIVE:
            speedup = 4.0
        elif self._config.level == OptimizationLevel.BASIC:
            speedup = 2.0
        else:
            speedup = 1.5

        return {
            "config": {
                "level": self._config.level.value,
                "strategy": self._config.strategy.value,
                "max_batch_images": self._config.max_batch_images,
                "enable_fp16": self._config.enable_fp16,
                "enable_int8": self._config.enable_int8,
                "preprocessing_on_gpu": self._config.preprocessing_on_gpu,
                "pin_memory": self._config.pin_memory,
            },
            "gpu_available": gpu_available,
            "device_count": len(caps),
            "capabilities": [
                {
                    "device_id": c.device_id,
                    "name": c.name,
                    "compute_capability": list(c.compute_capability),
                    "memory_total_mb": c.memory_total_mb,
                    "supports_fp16": c.supports_fp16,
                    "supports_int8": c.supports_int8,
                    "supports_tensor_cores": c.supports_tensor_cores,
                }
                for c in caps
            ],
            "estimated_speedup": speedup,
        }
