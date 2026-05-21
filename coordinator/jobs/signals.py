"""Celery signal handlers for worker registration and heartbeat.

These signals auto-register workers in the PostgreSQL database when they
connect and mark them offline when they disconnect.
"""

import logging
import os
import socket

from celery.signals import (
    heartbeat_sent,
    worker_ready,
    worker_shutdown,
)
from django.utils import timezone

logger = logging.getLogger(__name__)


def _detect_gpu():
    """Detect GPU availability and model info.

    Returns (gpu_available, gpu_model, gpu_vram_mb) for the first assigned GPU.
    When CUDA_VISIBLE_DEVICES pins to a specific GPU, reports that GPU's info.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    assigned_gpu = ""
    if cuda_visible and cuda_visible.lower() not in ("all", "none"):
        assigned_gpu = cuda_visible.split(",")[0].strip()

    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            gpu_count = paddle.device.cuda.device_count()
            if gpu_count > 0:
                gpu_name = paddle.device.cuda.get_device_name(0)
                # Rough VRAM detection
                props = paddle.device.cuda.get_device_properties(0)
                vram_mb = getattr(props, "total_memory", 0) // (1024 * 1024)
                if assigned_gpu:
                    gpu_name = f"{gpu_name} (cuda_visible_devices={assigned_gpu})"
                return True, gpu_name, vram_mb
    except Exception as exc:
        # GPU probes should never block worker startup; CUDA/paddle availability
        # and platform-specific introspection failures are treated as non-fatal.
        logger.debug("GPU detection unavailable: %s", exc)
    return False, "", 0


def _extract_gpu_index() -> int | None:
    """Extract the GPU device index from CUDA_VISIBLE_DEVICES.

    Returns the first GPU index, or None if not set/parseable.
    Examples: "2" -> 2, "0,1" -> 0, "" -> None, "all" -> None
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cuda_visible or cuda_visible.lower() in ("all", "none"):
        return None
    first = cuda_visible.split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


def _detect_cpu():
    """Detect CPU core count and RAM."""
    cpu_cores = os.cpu_count() or 0
    ram_mb = 0
    try:
        import psutil
        ram_mb = psutil.virtual_memory().total // (1024 * 1024)
    except ImportError:
        pass
    return cpu_cores, ram_mb


def recommended_concurrency(vram_mb: int) -> int:
    """Calculate recommended Celery worker concurrency based on GPU VRAM.

    Heuristic: ~4GB VRAM per concurrent OCR task (PaddleOCR model + image buffer).
    Returns a safe default if VRAM is unknown or unavailable.

    Reference table:
        8 GB  → 2 concurrent tasks
        16 GB → 4 concurrent tasks
        24 GB → 6 concurrent tasks
        48 GB → 12 concurrent tasks
    """
    if vram_mb <= 0:
        return 4  # safe default when VRAM unknown
    return max(1, min(vram_mb // 4096, 24))


def recommended_cpu_concurrency() -> int:
    """Recommend concurrency for CPU-only OCR workers.

    CPU OCR workers are limited by:
    - Available CPU cores (1 PaddleOCR inference per core)
    - Available RAM (~2GB per concurrent OCR task with models loaded)
    - MKL-DNN thread pool (defaults to number of physical cores)

    Returns conservative estimate: min(cpu_count // 2, available_ram_gb // 2, 8)
    """
    cpu_count = os.cpu_count() or 4
    # Conservative: use half the cores for OCR, leave rest for system
    cpu_based = max(1, cpu_count // 2)

    # Try to factor in available RAM (~2GB per task)
    ram_based = 8  # default cap if psutil unavailable
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        ram_based = max(1, int(ram_gb // 2))
    except ImportError:
        pass

    # Cap at 8 to avoid memory pressure from multiple model instances
    return min(cpu_based, ram_based, 8)


@worker_ready.connect
def on_worker_ready(sender=None, **kwargs):
    """Register worker in database when Celery worker starts."""
    from .tasks import register_worker

    hostname = socket.gethostname()
    gpu_available, gpu_model, gpu_vram_mb = _detect_gpu()
    if gpu_available and gpu_vram_mb > 0:
        rec = recommended_concurrency(gpu_vram_mb)
        logger.info(
            "GPU VRAM: %dMB → recommended concurrency: %d (current: %s)",
            gpu_vram_mb, rec,
            os.environ.get("WORKER_CONCURRENCY", "not set"),
        )
    cpu_cores, ram_mb = _detect_cpu()
    gpu_index = _extract_gpu_index()

    # Get queue names from worker consumer
    queues = []
    try:
        if hasattr(sender, "consumer") and hasattr(sender.consumer, "queues"):
            queues = [q.name for q in sender.consumer.queues]
    except Exception as exc:
        # Keep registration resilient even if Celery internals or transport
        # behavior changes and queue introspection fails at runtime.
        logger.warning("Falling back to WORKER_QUEUES env var due to queue introspection failure: %s", exc)
        queues = os.environ.get("WORKER_QUEUES", "cpu_general").split(",")

    # Log CPU OCR concurrency recommendation when worker subscribes to ocr_cpu
    if "ocr_cpu" in queues and not gpu_available:
        rec_cpu = recommended_cpu_concurrency()
        logger.info(
            "CPU OCR worker: recommended concurrency: %d (current: %s)",
            rec_cpu,
            os.environ.get("WORKER_CONCURRENCY", "not set"),
        )

    # Determine capabilities from queues
    capabilities = []
    if "ocr_gpu" in queues or any(q.startswith("ocr_gpu_") for q in queues):
        capabilities.append("ocr")
    if "ocr_cpu" in queues:
        capabilities.append("ocr")
    if "cpu_general" in queues:
        capabilities.extend(["compress", "ner"])

    try:
        from ocr_local.config.version import __version__
        pipeline_version = __version__
    except ImportError:
        pipeline_version = "unknown"

    register_worker(
        hostname=hostname,
        queues=queues,
        capabilities=capabilities,
        gpu_available=gpu_available,
        gpu_model=gpu_model,
        gpu_vram_mb=gpu_vram_mb,
        gpu_index=gpu_index,
        cpu_cores=cpu_cores,
        ram_mb=ram_mb,
        pipeline_version=pipeline_version,
    )
    logger.info("Worker %s registered (GPU: %s, queues: %s)",
                hostname, gpu_available, queues)


@worker_shutdown.connect
def on_worker_shutdown(sender=None, **kwargs):
    """Mark worker as offline when it shuts down."""
    from .tasks import unregister_worker

    hostname = socket.gethostname()
    unregister_worker(hostname)
    logger.info("Worker %s unregistered", hostname)


@heartbeat_sent.connect
def on_heartbeat(sender=None, **kwargs):
    """Update worker heartbeat timestamp."""
    from .models import Worker

    hostname = socket.gethostname()
    Worker.objects.filter(hostname=hostname).update(
        last_heartbeat=timezone.now(),
    )
