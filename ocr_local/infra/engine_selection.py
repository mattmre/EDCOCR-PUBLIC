"""Smart OCR engine selection based on document quality analysis.

Routes documents to PaddleOCR, Tesseract, or EasyOCR based on pre-scan
quality metrics and handwriting detection.  On CPU-only deployments,
this optimizes throughput by using Tesseract for clean documents (faster)
and PaddleOCR for complex ones (more accurate).  Handwritten pages can
optionally be routed to EasyOCR for superior handwriting recognition.

Configuration via OCR_ENGINE_SELECTION environment variable:
- "auto": Analyze document quality and select best engine (default for CPU)
- "paddle": Always use PaddleOCR (default for GPU)
- "tesseract": Always use Tesseract
- "easyocr": Always use EasyOCR (CPU-only, best for handwriting)
"""

import logging
import os

import numpy as np

__all__ = [
    "ENGINE_SELECTION",
    "VARIANCE_THRESHOLD",
    "EDGE_DENSITY_THRESHOLD",
    "SKEW_THRESHOLD",
    "TEXT_DENSITY_THRESHOLD",
    "DocumentQuality",
    "analyze_page_quality",
    "select_engine",
    "select_engine_for_page",
]

logger = logging.getLogger(__name__)

ENGINE_SELECTION = os.environ.get("OCR_ENGINE_SELECTION", "paddle").lower().strip()

# Quality thresholds for engine routing
# Documents scoring above these thresholds are considered "clean" enough for Tesseract
VARIANCE_THRESHOLD = 1500.0  # Image variance (higher = more contrast/detail)
EDGE_DENSITY_THRESHOLD = 0.03  # Edge pixel ratio (lower = cleaner scan)
SKEW_THRESHOLD = 2.0  # Degrees of skew (lower = better aligned)
TEXT_DENSITY_THRESHOLD = 0.3  # Ratio of text region area to page area


class DocumentQuality:
    """Quality assessment result for a document page."""

    __slots__ = (
        "variance",
        "edge_density",
        "skew_angle",
        "is_clean",
        "recommended_engine",
        "reason",
    )

    def __init__(
        self, variance, edge_density, skew_angle, is_clean, recommended_engine, reason
    ):
        self.variance = variance
        self.edge_density = edge_density
        self.skew_angle = skew_angle
        self.is_clean = is_clean
        self.recommended_engine = recommended_engine
        self.reason = reason

    def to_dict(self):
        return {
            "variance": round(self.variance, 2),
            "edge_density": round(self.edge_density, 4),
            "skew_angle": round(self.skew_angle, 2),
            "is_clean": self.is_clean,
            "recommended_engine": self.recommended_engine,
            "reason": self.reason,
        }


def analyze_page_quality(image):
    """Analyze a page image to determine document quality.

    Parameters
    ----------
    image : PIL.Image.Image
        Page image to analyze.

    Returns
    -------
    DocumentQuality
        Quality assessment with engine recommendation.
    """
    img_array = np.array(image.convert("L"))  # Grayscale

    # 1. Image variance -- measures contrast/detail level
    # High variance = clear text against white background
    # Low variance = faded, washed out, or uniform
    variance = float(np.var(img_array))

    # 2. Edge density -- ratio of edge pixels to total pixels
    # Uses simple Sobel-like gradient magnitude
    # High edge density = noisy/complex, Low = clean
    gx = np.diff(img_array.astype(np.float32), axis=1)
    gy = np.diff(img_array.astype(np.float32), axis=0)
    # Trim to same size
    min_h = min(gx.shape[0], gy.shape[0])
    min_w = min(gx.shape[1], gy.shape[1])
    gradient_mag = np.sqrt(gx[:min_h, :min_w] ** 2 + gy[:min_h, :min_w] ** 2)
    edge_density = float(np.mean(gradient_mag > 30))  # Threshold for "edge" pixel

    # 3. Skew estimation -- simple projection profile method
    # For speed, use a rough estimate from horizontal projection variance
    # across several rotation angles
    skew_angle = _estimate_skew(img_array)

    # Decision logic
    is_clean = (
        variance >= VARIANCE_THRESHOLD
        and edge_density <= EDGE_DENSITY_THRESHOLD
        and abs(skew_angle) <= SKEW_THRESHOLD
    )

    reasons = []
    if variance < VARIANCE_THRESHOLD:
        reasons.append(f"low contrast (variance={variance:.0f})")
    if edge_density > EDGE_DENSITY_THRESHOLD:
        reasons.append(f"noisy (edge_density={edge_density:.3f})")
    if abs(skew_angle) > SKEW_THRESHOLD:
        reasons.append(f"skewed ({skew_angle:.1f}deg)")

    if is_clean:
        engine = "tesseract"
        reason = "clean document -- Tesseract is faster"
    else:
        engine = "paddle"
        reason = "complex document -- " + "; ".join(reasons)

    return DocumentQuality(
        variance=variance,
        edge_density=edge_density,
        skew_angle=skew_angle,
        is_clean=is_clean,
        recommended_engine=engine,
        reason=reason,
    )


def _estimate_skew(gray_array):
    """Estimate document skew angle using projection profiles.

    A lightweight alternative to Hough transform that's fast enough
    for per-page pre-scan use.  The image is downsampled to ~500 rows
    before analysis to keep runtime under 20ms even for 300 DPI pages.

    Returns estimated skew in degrees.
    """
    h, w = gray_array.shape

    # Downsample to ~500 rows max for speed (full 3300-row images are too slow)
    target_rows = 500
    if h > target_rows:
        step = h // target_rows
        crop = gray_array[::step, :]
    else:
        crop = gray_array

    # Sample the middle 50% to avoid margins
    ch, cw = crop.shape
    crop = crop[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4]

    # Binarize
    threshold = np.mean(crop)
    binary = (crop < threshold).astype(np.float32)

    rows, cols = binary.shape
    if rows < 10 or cols < 10:
        return 0.0

    # Test a range of small angles using vectorized numpy shifts
    best_angle = 0.0
    best_score = 0.0

    # Row indices for shear calculation (vectorized)
    row_offsets = np.arange(rows) - rows // 2

    for angle_10x in range(-50, 51, 5):  # -5.0 to +5.0 degrees, step 0.5
        angle = angle_10x / 10.0
        shear = np.tan(np.radians(angle))
        shifts = np.round(shear * row_offsets).astype(np.int32)

        # Compute row sums with shifted columns (vectorized projection)
        projection = np.zeros(rows, dtype=np.float32)
        for i in range(rows):
            s = shifts[i]
            if abs(s) < cols:
                if s >= 0:
                    projection[i] = np.sum(binary[i, s:])
                else:
                    projection[i] = np.sum(binary[i, :cols + s])
            # else: projection[i] stays 0

        score = float(np.var(projection))
        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle


def _is_easyocr_available():
    """Check if EasyOCR engine is enabled and importable.

    Lazy import to avoid circular dependency and unnecessary
    module loading when EasyOCR is not in use.

    Returns
    -------
    bool
        True if EasyOCR is enabled and the package is installed.
    """
    try:
        from easyocr_engine import _EASYOCR_AVAILABLE, ENABLE_EASYOCR

        return ENABLE_EASYOCR and _EASYOCR_AVAILABLE
    except ImportError:
        return False


def select_engine(image=None, force=None):
    """Select the OCR engine for a page.

    Parameters
    ----------
    image : PIL.Image.Image, optional
        Page image for quality analysis (only used when selection is "auto").
    force : str, optional
        Override the configured engine selection.

    Returns
    -------
    str
        Engine name: "paddle", "tesseract", or "easyocr".
    """
    mode = force or ENGINE_SELECTION

    if mode == "tesseract":
        return "tesseract"
    if mode == "paddle":
        return "paddle"
    if mode == "easyocr":
        if _is_easyocr_available():
            return "easyocr"
        logger.warning(
            "EasyOCR requested but not available, falling back to 'paddle'"
        )
        return "paddle"
    if mode == "auto":
        if image is None:
            # Can't analyze without image, default to paddle
            return "paddle"
        quality = analyze_page_quality(image)
        logger.debug(
            "Engine selection: %s (variance=%.0f, edge_density=%.3f, skew=%.1f)",
            quality.recommended_engine,
            quality.variance,
            quality.edge_density,
            quality.skew_angle,
        )
        return quality.recommended_engine

    # Unknown mode, fall back to paddle
    logger.warning("Unknown OCR_ENGINE_SELECTION=%r, using 'paddle'", mode)
    return "paddle"


def select_engine_for_page(image=None, force=None, is_handwritten=False):
    """Select the OCR engine for a page, with handwriting-aware routing.

    Extends :func:`select_engine` by adding handwriting-aware routing.
    In ``auto`` mode, pages flagged as handwritten are routed to EasyOCR
    when it is available and enabled.  All other modes behave identically
    to :func:`select_engine`.

    Parameters
    ----------
    image : PIL.Image.Image, optional
        Page image for quality analysis (only used when selection is "auto").
    force : str, optional
        Override the configured engine selection.
    is_handwritten : bool
        Whether the page has been flagged as handwritten by the
        handwriting detection module.

    Returns
    -------
    str
        Engine name: "paddle", "tesseract", or "easyocr".
    """
    mode = force or ENGINE_SELECTION

    # In auto mode, handwritten pages get routed to EasyOCR when available
    if mode == "auto" and is_handwritten and _is_easyocr_available():
        logger.info("Handwritten page detected -- routing to EasyOCR")
        return "easyocr"

    # For easyocr force mode, delegate to select_engine which handles fallback
    # For all other modes, use standard selection
    return select_engine(image=image, force=force)
