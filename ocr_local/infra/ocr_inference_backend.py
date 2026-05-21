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

    backend = get_backend()

    # Base configuration (shared across all backends).
    # PaddleOCR 2.x uses use_gpu boolean and use_angle_cls for
    # text-line orientation classification.
    kwargs = {
        "use_angle_cls": True,
        "lang": lang_code,
        "use_gpu": (device == "gpu"),
        "show_log": False,
    }

    # Backend-specific configuration.
    # PaddleOCR 2.x uses use_onnx=True to enable ONNX Runtime inference.
    if backend in ("onnx", "openvino") and device == "cpu":
        kwargs["use_onnx"] = True
        logger.info(
            "Creating PaddleOCR engine: lang=%s, device=%s, backend=%s (use_onnx=True)",
            lang_code,
            device,
            backend,
        )
    else:
        logger.info(
            "Creating PaddleOCR engine: lang=%s, device=%s, backend=paddle",
            lang_code,
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
            "lang": lang_code,
            "use_gpu": (device == "gpu"),
        }
        if kwargs.get("use_onnx"):
            fallback_kwargs["use_onnx"] = True
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
