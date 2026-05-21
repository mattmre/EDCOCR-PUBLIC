"""Image preprocessing for enhanced OCR fallback layers (Phase 4D).

Provides optional image preprocessing steps that can improve OCR accuracy
on degraded, skewed, or noisy document scans.  Each function accepts a
PIL Image and returns a PIL Image, preserving the original on any failure.

OpenCV (cv2) is an optional dependency.  If not installed, all functions
gracefully return the original image unchanged and log a warning once.

ONNX Runtime is an optional dependency for AI-powered backends (NAFNet
denoising, U-Net binarization).  When not installed or when models are
not found on disk, the module gracefully degrades to the classical
OpenCV-based implementations.

This module is designed to NEVER crash the pipeline -- every public
function wraps its logic in try/except and returns the original image
on failure.
"""

import logging
import math
import os
import threading

import numpy as np
from PIL import Image

__all__ = [
    "DENOISE_BACKEND",
    "BINARIZE_BACKEND",
    "NAFNET_MODEL_PATH",
    "UNET_BINARIZE_MODEL_PATH",
    "denoise_nafnet",
    "binarize_unet",
    "deskew_image",
    "binarize_adaptive",
    "denoise_bilateral",
    "denoise_image",
    "binarize_image",
    "enhance_contrast",
    "preprocess_for_ocr",
]

logger = logging.getLogger("ocr_pipeline")

# ---------------------------------------------------------------------------
# OpenCV availability guard
# ---------------------------------------------------------------------------
try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("opencv-python-headless not installed; preprocessing disabled")

# ---------------------------------------------------------------------------
# ONNX Runtime availability guard
# ---------------------------------------------------------------------------
_ONNX_AVAILABLE = False
try:
    import onnxruntime  # noqa: F401

    _ONNX_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Backend configuration (via environment variables)
# ---------------------------------------------------------------------------
DENOISE_BACKEND = os.environ.get("DENOISE_BACKEND", "bilateral").lower().strip()
BINARIZE_BACKEND = os.environ.get("BINARIZE_BACKEND", "adaptive").lower().strip()

# ONNX model paths (configurable for Docker /app/models or local dev paths)
NAFNET_MODEL_PATH = os.environ.get(
    "NAFNET_MODEL_PATH", "/app/models/nafnet_denoise.onnx"
)
UNET_BINARIZE_MODEL_PATH = os.environ.get(
    "UNET_BINARIZE_MODEL_PATH", "/app/models/unet_binarize.onnx"
)

# Tile inference defaults
_TILE_SIZE = 512
_TILE_OVERLAP = 32

# Minimum image dimension (pixels) below which preprocessing is skipped
_MIN_DIM = 16
_DESKEW_MAX_DIM = 1600
_DESKEW_MIN_ANGLE = 0.5
_DESKEW_MAX_ANGLE = 12.0
_DESKEW_MIN_SCORE_IMPROVEMENT = 1.02
_DESKEW_MIN_COMPONENT_AREA = 64

# ---------------------------------------------------------------------------
# ONNX model cache (thread-safe, loaded once per model path)
# ---------------------------------------------------------------------------
_onnx_model_cache: dict = {}
_onnx_cache_lock = threading.Lock()
_ALLOWED_ONNX_MODEL_ROOTS = tuple(
    os.path.realpath(path)
    for path in ("/app/models", os.path.join(os.getcwd(), "models"))
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pil_to_gray(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a single-channel uint8 numpy array."""
    return np.array(img.convert("L"))


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to BGR uint8 numpy array (OpenCV convention)."""
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a BGR uint8 numpy array back to a PIL Image."""
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _too_small(img: Image.Image) -> bool:
    """Return True if the image is too small for meaningful preprocessing."""
    w, h = img.size
    return w < _MIN_DIM or h < _MIN_DIM


def _normalize_skew_angle(angle: float) -> float:
    """Normalize an angle into the conservative deskew range."""
    normalized = ((float(angle) + 45.0) % 90.0) - 45.0
    if abs(normalized) > _DESKEW_MAX_ANGLE:
        return 0.0
    return normalized


def _resize_for_deskew(gray: np.ndarray) -> np.ndarray:
    """Downscale large images before angle estimation and candidate scoring."""
    height, width = gray.shape
    max_dim = max(height, width)
    if max_dim <= _DESKEW_MAX_DIM:
        return gray

    scale = _DESKEW_MAX_DIM / float(max_dim)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(
        gray,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def _deskew_binary(gray: np.ndarray) -> np.ndarray:
    """Build a binary foreground mask for skew estimation and scoring."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary


def _estimate_skew_hough(gray: np.ndarray) -> float | None:
    """Estimate skew using Hough line detection on edge features."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=max(gray.shape[1] // 4, 20),
        maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        return None

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        angle = _normalize_skew_angle(angle)
        if angle and abs(angle) < 45.0:
            angles.append(angle)

    if not angles:
        return None
    return float(np.median(angles))


def _normalize_rect_angle(angle: float, width: float, height: float) -> float:
    """Convert OpenCV min-area rectangle angles into deskew correction angles."""
    correction = float(angle)
    if width < height:
        correction -= 90.0
    return _normalize_skew_angle(correction)


def _estimate_skew_min_area(gray: np.ndarray) -> float | None:
    """Estimate skew from the dominant foreground mass using minAreaRect."""
    binary = _deskew_binary(gray)
    kernel_width = max(15, gray.shape[1] // 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    image_area = gray.shape[0] * gray.shape[1]
    max_component_area = image_area * 0.85

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < _DESKEW_MIN_COMPONENT_AREA or area > max_component_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 8 or height < 3:
            continue
        points.append(contour.reshape(-1, 2))

    if not points:
        foreground = cv2.findNonZero(closed)
        if foreground is None:
            return None
        points = [foreground.reshape(-1, 2)]

    merged = np.vstack(points).astype(np.float32)
    if len(merged) < 5:
        return None

    (_, _), (width, height), angle = cv2.minAreaRect(merged)
    return _normalize_rect_angle(angle, width, height)


def _score_projection(binary: np.ndarray, angle: float) -> float:
    """Score a candidate angle by horizontal projection variance."""
    height, width = binary.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    rotated = cv2.warpAffine(
        binary,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    row_sums = np.sum(rotated > 0, axis=1).astype(np.float32)
    return float(np.var(row_sums))


def _select_best_skew_angle(gray: np.ndarray) -> float:
    """Select the most plausible correction angle from multiple estimators."""
    working = _resize_for_deskew(gray)
    if min(working.shape[:2]) < _MIN_DIM:
        return 0.0

    candidates = {0.0}
    estimators = {
        "hough": _estimate_skew_hough,
        "min_area": _estimate_skew_min_area,
    }
    for name, estimator in estimators.items():
        angle = estimator(working)
        if angle is None:
            continue
        normalized = _normalize_skew_angle(angle)
        if abs(normalized) >= _DESKEW_MIN_ANGLE:
            candidates.add(round(normalized, 2))
            # minAreaRect angle normalization is ambiguous around 90-degree wraps;
            # score both directions and let the projection metric decide.
            if name == "min_area":
                candidates.add(round(-normalized, 2))

    binary = _deskew_binary(working)
    scores = {angle: _score_projection(binary, angle) for angle in candidates}
    baseline = scores.get(0.0, 0.0)
    best_angle, best_score = max(scores.items(), key=lambda item: item[1])

    if abs(best_angle) < _DESKEW_MIN_ANGLE:
        return 0.0
    if baseline > 0 and best_score < baseline * _DESKEW_MIN_SCORE_IMPROVEMENT:
        return 0.0
    if baseline == 0.0 and best_score <= 0.0:
        return 0.0
    return float(best_angle)


def _is_within_allowed_root(path: str, root: str) -> bool:
    """Return True when path resolves under the allowed root."""
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _validate_onnx_model_path(model_path: str) -> str | None:
    """Validate ONNX model paths before loading from disk."""
    if not model_path or not os.path.isabs(model_path):
        logger.warning("Rejecting ONNX model path that is not absolute: %s", model_path)
        return None

    resolved = os.path.realpath(model_path)
    if not resolved.lower().endswith(".onnx"):
        logger.warning("Rejecting ONNX model path with invalid extension: %s", resolved)
        return None

    if not any(
        _is_within_allowed_root(resolved, root)
        for root in _ALLOWED_ONNX_MODEL_ROOTS
    ):
        logger.warning("Rejecting ONNX model path outside allowed roots: %s", resolved)
        return None

    return resolved


# ---------------------------------------------------------------------------
# ONNX model loading and tile-based inference
# ---------------------------------------------------------------------------


def _load_onnx_model(model_path: str):
    """Load an ONNX Runtime InferenceSession with thread-safe caching.

    Returns ``None`` if ONNX Runtime is not installed or the model file
    does not exist.

    Parameters
    ----------
    model_path : str
        Absolute path to the ``.onnx`` model file.

    Returns
    -------
    onnxruntime.InferenceSession or None
    """
    if not _ONNX_AVAILABLE:
        return None

    validated_path = _validate_onnx_model_path(model_path)
    if validated_path is None:
        return None

    with _onnx_cache_lock:
        if validated_path in _onnx_model_cache:
            return _onnx_model_cache[validated_path]

    if not os.path.isfile(validated_path):
        logger.debug("ONNX model not found at %s", validated_path)
        return None

    try:
        import onnxruntime

        session = onnxruntime.InferenceSession(
            validated_path,
            providers=["CPUExecutionProvider"],
        )
        with _onnx_cache_lock:
            _onnx_model_cache[validated_path] = session
        logger.info("Loaded ONNX model: %s", validated_path)
        return session
    except Exception as exc:
        logger.warning("Failed to load ONNX model %s: %s", validated_path, exc)
        return None


def _tile_inference(
    img_array: np.ndarray,
    session,
    tile_size: int = _TILE_SIZE,
    overlap: int = _TILE_OVERLAP,
) -> np.ndarray:
    """Run tile-based ONNX inference on an image array.

    Splits the image into overlapping tiles, runs each tile through the
    ONNX session, and reassembles the output.  This allows processing
    images larger than the model's native resolution without running out
    of memory.

    The input array is expected to be HWC uint8 (either 1 or 3 channels).
    The ONNX model is expected to accept NCHW float32 input normalized
    to [0, 1] and produce NCHW float32 output in [0, 1].

    Parameters
    ----------
    img_array : np.ndarray
        Input image array (H, W) or (H, W, C), dtype uint8.
    session : onnxruntime.InferenceSession
        Loaded ONNX model session.
    tile_size : int
        Tile dimension (square tiles).
    overlap : int
        Pixel overlap between adjacent tiles.

    Returns
    -------
    np.ndarray
        Processed image array, same shape and dtype as input.
    """
    if img_array.ndim == 2:
        h, w = img_array.shape
        channels = 1
        img_work = img_array[:, :, np.newaxis]
    else:
        h, w, channels = img_array.shape
        img_work = img_array

    step = max(tile_size - overlap, 1)
    output = np.zeros_like(img_work, dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    input_name = session.get_inputs()[0].name

    for y in range(0, h, step):
        for x in range(0, w, step):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(y_end - tile_size, 0)
            x_start = max(x_end - tile_size, 0)

            tile = img_work[y_start:y_end, x_start:x_end]
            th, tw = tile.shape[:2]

            # Pad to tile_size if needed
            pad_h = tile_size - th
            pad_w = tile_size - tw
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(
                    tile,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="reflect",
                )

            # NCHW float32 in [0, 1]
            blob = tile.transpose(2, 0, 1).astype(np.float32) / 255.0
            blob = blob[np.newaxis, ...]

            result = session.run(None, {input_name: blob})[0]
            # result shape: (1, C, H, W)
            result_tile = result[0].transpose(1, 2, 0)  # HWC
            result_tile = result_tile[:th, :tw]  # un-pad

            output[y_start:y_end, x_start:x_end] += result_tile
            weight[y_start:y_end, x_start:x_end] += 1.0

    # Average overlapping regions
    weight = np.maximum(weight, 1.0)
    for c in range(channels):
        output[:, :, c] /= weight

    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    if img_array.ndim == 2:
        return output[:, :, 0]
    return output


# ---------------------------------------------------------------------------
# AI-powered denoising (NAFNet via ONNX)
# ---------------------------------------------------------------------------


def denoise_nafnet(img: Image.Image) -> Image.Image:
    """Denoise an image using the NAFNet ONNX model.

    Falls back to bilateral filter (``denoise_bilateral``) if ONNX
    Runtime is not available or the model file is not found.

    Parameters
    ----------
    img : PIL.Image.Image
        Source document page image.

    Returns
    -------
    PIL.Image.Image
        Denoised image (or original on failure).
    """
    session = _load_onnx_model(NAFNET_MODEL_PATH)
    if session is None:
        logger.debug("NAFNet model unavailable, falling back to bilateral denoise")
        return denoise_bilateral(img)

    try:
        rgb = np.array(img.convert("RGB"))
        denoised = _tile_inference(rgb, session)
        return Image.fromarray(denoised, mode="RGB")
    except Exception as exc:
        logger.warning("NAFNet denoise failed, falling back to bilateral: %s", exc)
        return denoise_bilateral(img)


# ---------------------------------------------------------------------------
# AI-powered binarization (U-Net via ONNX)
# ---------------------------------------------------------------------------


def binarize_unet(img: Image.Image) -> Image.Image:
    """Binarize a document image using a U-Net ONNX model.

    Falls back to adaptive thresholding (``binarize_adaptive``) if
    ONNX Runtime is not available or the model file is not found.

    Parameters
    ----------
    img : PIL.Image.Image
        Source document page image.

    Returns
    -------
    PIL.Image.Image
        Binarized image in mode "L" (or original on failure).
    """
    session = _load_onnx_model(UNET_BINARIZE_MODEL_PATH)
    if session is None:
        logger.debug("U-Net model unavailable, falling back to adaptive binarization")
        return binarize_adaptive(img)

    try:
        gray = np.array(img.convert("L"))
        binarized = _tile_inference(gray, session)
        # Threshold the continuous output to binary
        binary = np.where(binarized > 127, 255, 0).astype(np.uint8)
        return Image.fromarray(binary, mode="L")
    except Exception as exc:
        logger.warning(
            "U-Net binarize failed, falling back to adaptive: %s", exc
        )
        return binarize_adaptive(img)


# ---------------------------------------------------------------------------
# Public preprocessing functions
# ---------------------------------------------------------------------------

def deskew_image(img: Image.Image) -> Image.Image:
    """Correct skew using a hybrid estimator with projection scoring.

    Returns the original image if:
    - OpenCV is not available
    - Detected skew is less than 0.5 degrees
    - The image is too small
    - Any error occurs during processing
    """
    if not _CV2_AVAILABLE or _too_small(img):
        return img
    try:
        gray = _pil_to_gray(img)
        selected_angle = _select_best_skew_angle(gray)
        if abs(selected_angle) < _DESKEW_MIN_ANGLE:
            return img

        h_img, w_img = gray.shape
        center = (w_img // 2, h_img // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, selected_angle, 1.0)
        bgr = _pil_to_bgr(img)
        rotated = cv2.warpAffine(
            bgr,
            rotation_matrix,
            (w_img, h_img),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        logger.debug("Deskew: corrected %.2f degrees", selected_angle)
        return _bgr_to_pil(rotated)

    except Exception as exc:
        logger.warning("Deskew failed, returning original: %s", exc)
        return img


def binarize_adaptive(img: Image.Image) -> Image.Image:
    """Apply adaptive Gaussian thresholding for degraded scans.

    Converts the output to a grayscale PIL Image (mode "L").
    Returns the original on failure.
    """
    if not _CV2_AVAILABLE or _too_small(img):
        return img
    try:
        gray = _pil_to_gray(img)

        binary = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        return Image.fromarray(binary, mode="L")

    except Exception as exc:
        logger.warning(f"Binarization failed, returning original: {exc}")
        return img


def denoise_bilateral(img: Image.Image) -> Image.Image:
    """Remove noise using bilateral filter (preserves edges).

    This is the classical OpenCV-based denoising implementation.
    Returns the denoised image or the original on failure.
    """
    if not _CV2_AVAILABLE or _too_small(img):
        return img
    try:
        bgr = _pil_to_bgr(img)

        denoised = cv2.bilateralFilter(
            bgr,
            d=9,
            sigmaColor=75,
            sigmaSpace=75,
        )

        return _bgr_to_pil(denoised)

    except Exception as exc:
        logger.warning(f"Denoise failed, returning original: {exc}")
        return img


def denoise_image(img: Image.Image) -> Image.Image:
    """Remove noise using the configured backend.

    Backend selection is controlled by the ``DENOISE_BACKEND`` environment
    variable:

    - ``"bilateral"`` (default): OpenCV bilateral filter
    - ``"nafnet"``: NAFNet ONNX model (falls back to bilateral)
    - ``"auto"``: Try NAFNet first, fall back to bilateral

    Returns the denoised image or the original on failure.
    """
    if _too_small(img):
        return img

    backend = DENOISE_BACKEND
    if backend == "nafnet":
        return denoise_nafnet(img)
    if backend == "auto":
        return denoise_nafnet(img)
    # Default: bilateral
    return denoise_bilateral(img)


def binarize_image(img: Image.Image) -> Image.Image:
    """Binarize using the configured backend.

    Backend selection is controlled by the ``BINARIZE_BACKEND`` environment
    variable:

    - ``"adaptive"`` (default): OpenCV adaptive Gaussian thresholding
    - ``"unet"``: U-Net ONNX model (falls back to adaptive)
    - ``"auto"``: Try U-Net first, fall back to adaptive

    Returns the binarized image (mode "L") or the original on failure.
    """
    if _too_small(img):
        return img

    backend = BINARIZE_BACKEND
    if backend == "unet":
        return binarize_unet(img)
    if backend == "auto":
        return binarize_unet(img)
    # Default: adaptive
    return binarize_adaptive(img)


def enhance_contrast(img: Image.Image) -> Image.Image:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Operates on the L channel in LAB color space to preserve color
    information while enhancing contrast.  Returns original on failure.
    """
    if not _CV2_AVAILABLE or _too_small(img):
        return img
    try:
        bgr = _pil_to_bgr(img)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        merged = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        return _bgr_to_pil(result)

    except Exception as exc:
        logger.warning(f"Contrast enhancement failed, returning original: {exc}")
        return img


# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------

def preprocess_for_ocr(img: Image.Image, level: str = "standard") -> Image.Image:
    """Apply a preprocessing pipeline based on the requested level.

    Levels
    ------
    - ``"none"``       : No preprocessing (returns image unchanged).
    - ``"standard"``   : Deskew only (safe default for most documents).
    - ``"enhanced"``   : Deskew + denoise + contrast enhancement.
    - ``"aggressive"`` : All steps including adaptive binarization.

    Denoise and binarize steps use the backend-aware routing functions
    (``denoise_image`` and ``binarize_image``), which honour the
    ``DENOISE_BACKEND`` and ``BINARIZE_BACKEND`` environment variables.

    On any failure, the original image is returned unchanged.

    Parameters
    ----------
    img : PIL.Image.Image
        The source document page image.
    level : str
        One of ``"none"``, ``"standard"``, ``"enhanced"``, ``"aggressive"``.

    Returns
    -------
    PIL.Image.Image
        The preprocessed (or original) image.
    """
    if not _CV2_AVAILABLE:
        return img

    if _too_small(img):
        return img

    # Normalize unknown levels to "standard"
    valid_levels = {"none", "standard", "enhanced", "aggressive"}
    if level not in valid_levels:
        logger.warning(
            f"Unknown preprocessing level '{level}', defaulting to 'standard'"
        )
        level = "standard"

    if level == "none":
        return img

    # Noise profiling (opt-in): assess image quality for adaptive parameter tuning
    profile = None
    try:
        from noise_profiling import ENABLE_NOISE_PROFILING as _NP_ENABLED
        from noise_profiling import profile_image

        if _NP_ENABLED:
            profile = profile_image(img)
            logger.debug(
                "Noise profile: variance=%.1f level=%s snr=%.1f contrast=%.2f",
                profile.noise_variance,
                profile.noise_level,
                profile.estimated_snr,
                profile.contrast_score,
            )
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("Noise profiling failed (non-fatal): %s", exc)

    try:
        result = img

        # Standard: deskew only
        if level in ("standard", "enhanced", "aggressive"):
            result = deskew_image(result)

        # Enhanced: add denoise + contrast (uses backend-aware routing)
        if level in ("enhanced", "aggressive"):
            # If noise profiling says the image is clean, skip denoising
            if profile is not None and profile.skip_denoise:
                logger.debug("Noise profile: skipping denoise (image is clean)")
            else:
                result = denoise_image(result)
            result = enhance_contrast(result)

        # Aggressive: add binarization (uses backend-aware routing)
        if level == "aggressive":
            result = binarize_image(result)

        # Advanced preprocessing (opt-in via ENABLE_ADVANCED_PREPROCESSING)
        result = _apply_advanced_preprocessing(result)

        return result

    except Exception as exc:
        logger.warning(f"Preprocessing pipeline failed at level '{level}': {exc}")
        return img


def _apply_advanced_preprocessing(img: Image.Image) -> Image.Image:
    """Apply advanced preprocessing if enabled.

    Imports ``advanced_preprocessing`` lazily and runs the full advanced
    pipeline.  On any failure (including import errors), returns the
    original image unchanged.

    The advanced preprocessor works on numpy arrays, so this function
    handles the PIL-to-numpy and numpy-to-PIL conversions.
    """
    try:
        from advanced_preprocessing import (
            ENABLE_ADVANCED_PREPROCESSING,
            AdvancedPreprocessor,
        )
    except ImportError:
        return img

    if not ENABLE_ADVANCED_PREPROCESSING:
        return img

    if not _CV2_AVAILABLE:
        return img

    try:
        # Convert PIL to BGR numpy array for the advanced pipeline
        bgr = _pil_to_bgr(img)
        preprocessor = AdvancedPreprocessor()
        processed, metadata = preprocessor.process(bgr)

        transforms = metadata.get("transforms_applied", [])
        if transforms:
            logger.info(
                "Advanced preprocessing applied: %s", ", ".join(transforms)
            )

        # Convert back to PIL
        return _bgr_to_pil(processed)

    except Exception as exc:
        logger.warning("Advanced preprocessing failed, returning original: %s", exc)
        return img
