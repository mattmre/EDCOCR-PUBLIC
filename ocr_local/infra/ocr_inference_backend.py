"""OCR inference backend selection.

Supports multiple inference backends for PaddleOCR to optimize
CPU-only deployments. The backend is selected via the
OCR_INFERENCE_BACKEND environment variable.

Supported backends:
- "paddle" (default): Native PaddlePaddle inference engine
- "onnx": ONNX Runtime via PaddleOCR 2.x ``use_onnx=True`` parameter
- "openvino": OpenVINO via ONNX Runtime with OpenVINO execution provider
- "auto": Auto-select best available backend for the current hardware

PaddleOCR 2.x provides ONNX Runtime acceleration through the
``use_onnx=True`` constructor parameter, which converts PaddlePaddle
models to ONNX format and runs inference via ONNX Runtime (or its
OpenVINO execution provider on Intel CPUs).
"""

import logging
import os

logger = logging.getLogger(__name__)

# Backend selection via environment variable
INFERENCE_BACKEND = os.environ.get("OCR_INFERENCE_BACKEND", "paddle").lower().strip()

# Valid backends
VALID_BACKENDS = ("paddle", "onnx", "openvino", "auto")
DEFAULT_DET_LIMIT_SIDE_LEN = 1536
MIN_DET_LIMIT_SIDE_LEN = 960
MAX_DET_LIMIT_SIDE_LEN = 4096
# Keep low-confidence recognized spans available for review and legal
# completeness.  Disposition is decided after OCR, not inside PaddleOCR.
OCR_DROP_SCORE = 0.0


def _explicit_bounded_int(name, minimum, maximum):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}")
    return value


def get_native_threads_per_predictor():
    """Explicit Paddle CPU native threads, separate from worker concurrency."""
    return _explicit_bounded_int("OCR_NATIVE_THREADS_PER_PREDICTOR", 1, 64)


def get_recognizer_batch_size():
    """Paddle 2.x recognition-crop batch size, not page batching."""
    return _explicit_bounded_int("OCR_RECOGNIZER_BATCH_SIZE", 1, 128)


def get_gpu_device_configuration(device):
    """Return use_gpu/gpu_id after validating explicit local-device binding.

    Fleet GPU pods expose one device each. ``gpu:0`` is its process-local
    index; the physical UUID remains in CUDA/NVIDIA_VISIBLE_DEVICES and the
    runtime-pool key. Multiple visible devices need separate workers rather
    than global Paddle device changes from concurrent predictor threads.
    """
    if device == "cpu":
        return False, None
    if device == "gpu":
        return True, None  # legacy single-device configuration
    if device != "gpu:0":
        raise ValueError("explicit Paddle device must be local gpu:0")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    tokens = [item.strip() for item in visible.split(",") if item.strip()]
    if len(tokens) != 1 or tokens[0].lower() in {
        "all", "none", "void", "-1", "nodevfiles"
    }:
        raise RuntimeError("gpu:0 requires exactly one explicitly visible GPU")
    return True, 0


def get_detection_limit_side_len():
    """Return the validated long-side detector resolution.

    PaddleOCR defaults to 960 pixels, which can discard otherwise legible text
    lines when a 300-DPI document page is reduced for detection. The qualified
    default preserves those lines while keeping CPU inference bounded.
    """
    raw = os.environ.get(
        "OCR_DET_LIMIT_SIDE_LEN", str(DEFAULT_DET_LIMIT_SIDE_LEN)
    ).strip()
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid OCR_DET_LIMIT_SIDE_LEN=%r; using %d",
            raw,
            DEFAULT_DET_LIMIT_SIDE_LEN,
        )
        return DEFAULT_DET_LIMIT_SIDE_LEN
    if not MIN_DET_LIMIT_SIDE_LEN <= value <= MAX_DET_LIMIT_SIDE_LEN:
        logger.warning(
            "OCR_DET_LIMIT_SIDE_LEN=%d is outside %d..%d; using %d",
            value,
            MIN_DET_LIMIT_SIDE_LEN,
            MAX_DET_LIMIT_SIDE_LEN,
            DEFAULT_DET_LIMIT_SIDE_LEN,
        )
        return DEFAULT_DET_LIMIT_SIDE_LEN
    return value


# Set OpenVINO device type at module load time (before worker threads start)
# to avoid race conditions in multi-threaded pipelines.
if INFERENCE_BACKEND in ("openvino", "auto"):
    os.environ.setdefault("ORT_OPENVINO_DEVICE_TYPE", "CPU")


def _detect_best_backend():
    """Auto-detect the best available inference backend.

    Checks for OpenVINO first (best for Intel CPUs), then ONNX Runtime
    (best cross-platform CPU option), then falls back to native PaddlePaddle.

    Returns
    -------
    str
        One of "openvino", "onnx", or "paddle".
    """
    # Try OpenVINO first (best for Intel CPUs)
    try:
        import openvino  # noqa: F401

        logger.info("Auto-detected OpenVINO backend (Intel CPU optimization)")
        return "openvino"
    except ImportError:
        pass

    # Try ONNX Runtime (best cross-platform CPU option)
    try:
        import onnxruntime  # noqa: F401

        logger.info(
            "Auto-detected ONNX Runtime backend (cross-platform CPU optimization)"
        )
        return "onnx"
    except ImportError:
        pass

    # Fall back to native PaddlePaddle
    logger.info("Using default PaddlePaddle inference backend")
    return "paddle"


def get_backend():
    """Return the resolved inference backend name.

    Resolves "auto" to the best available backend, validates configured
    backend names, and falls back to "paddle" for unknown values.

    Returns
    -------
    str
        One of "paddle", "onnx", or "openvino".
    """
    if INFERENCE_BACKEND == "auto":
        return _detect_best_backend()
    if INFERENCE_BACKEND not in VALID_BACKENDS:
        logger.warning(
            "Unknown OCR_INFERENCE_BACKEND=%r, falling back to 'paddle'. "
            "Valid options: %s",
            INFERENCE_BACKEND,
            ", ".join(VALID_BACKENDS),
        )
        return "paddle"
    return INFERENCE_BACKEND


def create_ocr_engine(lang_code, device="gpu"):
    """Create a PaddleOCR engine with the configured inference backend.

    When the resolved backend is "onnx" or "openvino" and the target
    device is "cpu", the engine is created with ``use_onnx=True``
    which enables ONNX Runtime acceleration in PaddleOCR 2.x.

    GPU inference always uses the native PaddlePaddle backend regardless
    of the configured setting.

    Parameters
    ----------
    lang_code : str
        PaddleOCR language code (e.g., "en", "ch", "fr").
    device : str
        Device target -- "gpu" or "cpu".  Converted internally to
        ``use_gpu=True/False`` for the PaddleOCR 2.x constructor.

    Returns
    -------
    PaddleOCR
        Configured OCR engine instance.
    """
    from paddleocr import PaddleOCR

    from ocr_local.config.language_config import get_paddle_runtime_code

    backend = get_backend()
    runtime_lang_code = get_paddle_runtime_code(lang_code)

    # Base configuration (shared across all backends).
    # PaddleOCR 2.x uses use_gpu boolean and use_angle_cls for
    # text-line orientation classification.
    use_gpu, gpu_id = get_gpu_device_configuration(device)
    native_threads = get_native_threads_per_predictor()
    recognizer_batch = get_recognizer_batch_size()
    kwargs = {
        "use_angle_cls": True,
        "lang": runtime_lang_code,
        "use_gpu": use_gpu,
        "show_log": False,
        "det_limit_side_len": get_detection_limit_side_len(),
        "drop_score": OCR_DROP_SCORE,
    }
    if gpu_id is not None:
        kwargs["gpu_id"] = gpu_id
    if native_threads is not None:
        kwargs["cpu_threads"] = native_threads
    if recognizer_batch is not None:
        kwargs["rec_batch_num"] = recognizer_batch

    # Backend-specific configuration.
    # PaddleOCR 2.x uses use_onnx=True to enable ONNX Runtime inference.
    if backend in ("onnx", "openvino") and device == "cpu":
        kwargs["use_onnx"] = True
        logger.info(
            "Creating PaddleOCR engine: lang=%s, device=%s, backend=%s (use_onnx=True)",
            runtime_lang_code,
            device,
            backend,
        )
    else:
        logger.info(
            "Creating PaddleOCR engine: lang=%s, device=%s, backend=paddle",
            runtime_lang_code,
            device,
        )

    try:
        return PaddleOCR(**kwargs)
    except TypeError as exc:
        # Safety net for PaddleOCR versions that do not accept all kwargs.
        # Strip to minimal constructor and retry.
        logger.warning(
            "PaddleOCR constructor rejected kwargs (%s), retrying with minimal config",
            exc,
        )
        fallback_kwargs = {
            "use_angle_cls": True,
            "lang": runtime_lang_code,
            "use_gpu": use_gpu,
            "drop_score": OCR_DROP_SCORE,
        }
        for key in ("gpu_id", "cpu_threads", "rec_batch_num"):
            if key in kwargs:
                fallback_kwargs[key] = kwargs[key]
        if kwargs.get("use_onnx"):
            fallback_kwargs["use_onnx"] = True
        try:
            return PaddleOCR(**fallback_kwargs)
        except TypeError as fallback_exc:
            if not fallback_kwargs.pop("use_onnx", None):
                raise
            # PaddleOCR releases that removed the legacy ``use_onnx``
            # constructor flag must still remain usable.  An explicitly
            # requested accelerator is best-effort; native Paddle is the
            # safe compatibility fallback when the installed constructor
            # rejects that flag even after all unrelated kwargs are removed.
            logger.warning(
                "PaddleOCR rejected legacy use_onnx (%s); falling back to "
                "native Paddle inference",
                fallback_exc,
            )
            return PaddleOCR(**fallback_kwargs)


def get_backend_info():
    """Return a dict describing the current inference backend configuration.

    Useful for metrics endpoints, fleet status reporting, and debugging.
    Probes for optional ONNX Runtime and OpenVINO packages and reports
    their availability and versions.

    Returns
    -------
    dict
        Backend configuration with keys: backend, configured,
        onnx_available, openvino_available, and optional version strings.
    """
    backend = get_backend()
    info = {
        "backend": backend,
        "configured": INFERENCE_BACKEND,
        "det_limit_side_len": get_detection_limit_side_len(),
        "drop_score": OCR_DROP_SCORE,
        "native_threads_per_predictor": get_native_threads_per_predictor(),
        "recognizer_batch_size": get_recognizer_batch_size(),
        "onnx_available": False,
        "openvino_available": False,
    }
    try:
        import onnxruntime

        info["onnx_available"] = True
        info["onnx_version"] = onnxruntime.__version__
    except ImportError:
        pass
    try:
        import openvino

        info["openvino_available"] = True
        info["openvino_version"] = openvino.__version__
    except ImportError:
        pass
    return info
