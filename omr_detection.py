"""Optical Mark Recognition (OMR) for forensic OCR pipeline.

Detects checkboxes, radio buttons, and selection marks in page images using
OpenCV contour analysis. No external ML models required -- pure geometry-based
detection with fill-ratio classification for checked vs unchecked state.

Output: contributes marks to EXPORT/SYMBOLOGY/<subfolder>/<document_name>.symbology.json

Graceful degradation: if OpenCV (cv2) is not installed, OMR detection is
skipped and an empty result set is returned.

Configuration (env vars):
- ``ENABLE_SYMBOLOGY_EXTRACTION``: master toggle (default: false)
- ``SYMBOLOGY_OMR_ENABLED``: enable OMR detection (default: true)
- ``OMR_MIN_MARK_SIZE``: minimum mark size in pixels (default: 15)
- ``OMR_MAX_MARK_SIZE``: maximum mark size in pixels (default: 50)
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# --- Guarded OpenCV import ---
try:
    import cv2
    import numpy as np

    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    np = None
    _CV2_AVAILABLE = False

# If cv2 is not available but numpy is (used for type checks), try importing
if not _CV2_AVAILABLE:
    try:
        import numpy as np  # noqa: F811
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Canonical env-parsing helper (DRY consolidation)
from ocr_distributed.ocr_utils import get_env_int as _get_env_int

OMR_MIN_MARK_SIZE = _get_env_int("OMR_MIN_MARK_SIZE", 15)
OMR_MAX_MARK_SIZE = _get_env_int("OMR_MAX_MARK_SIZE", 50)

# Fill ratio thresholds for checked/unchecked classification
_FILL_RATIO_CHECKED_THRESHOLD = 0.35  # Above = checked
_FILL_RATIO_EMPTY_THRESHOLD = 0.15  # Below = definitely empty

# Aspect ratio bounds for rectangular marks (checkboxes/radio buttons)
_MIN_ASPECT_RATIO = 0.6
_MAX_ASPECT_RATIO = 1.67  # ~5:3 max aspect ratio

# Solidity threshold: contours that are roughly rectangular or circular
_MIN_SOLIDITY = 0.7

# Confidence scaling based on how "mark-like" the contour is
_CONFIDENCE_BASE = 0.7
_CONFIDENCE_ASPECT_BONUS = 0.15  # For near-square shapes
_CONFIDENCE_SOLIDITY_BONUS = 0.15  # For high solidity shapes


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DetectedMark:
    """A single detected checkbox or selection mark."""

    mark_type: str = "checkbox"  # "checkbox" or "radio"
    checked: bool = False  # Whether the mark is filled/checked
    bbox: list = field(default_factory=list)  # [x1, y1, x2, y2]
    confidence: float = 0.0  # Detection confidence (0.0-1.0)
    fill_ratio: float = 0.0  # Ratio of filled pixels within the mark
    page_num: int = 0


@dataclass
class PageMarks:
    """OMR detection results for a single page."""

    page_num: int
    marks: list = field(default_factory=list)  # List[DetectedMark as dict]
    total_marks: int = 0
    checked_marks: int = 0
    unchecked_marks: int = 0


# ---------------------------------------------------------------------------
# OMR detector
# ---------------------------------------------------------------------------


class OMRDetector:
    """Detects checkboxes and selection marks using OpenCV contour analysis.

    Graceful degradation: returns empty results when OpenCV is not installed.
    """

    def __init__(
        self,
        min_mark_size: int = OMR_MIN_MARK_SIZE,
        max_mark_size: int = OMR_MAX_MARK_SIZE,
    ):
        self._available = _CV2_AVAILABLE
        self._min_size = min_mark_size
        self._max_size = max_mark_size

        if not self._available:
            logger.info(
                "OpenCV not available; OMR detection will be skipped."
            )

    @property
    def is_available(self) -> bool:
        """Whether OpenCV is installed and functional."""
        return self._available

    def detect_marks(self, image, page_num: int = 0) -> list:
        """Detect checkboxes and selection marks in an image.

        Args:
            image: PIL Image or numpy array.
            page_num: Page number for result attribution (1-based).

        Returns:
            List of DetectedMark dataclass instances.
        """
        if not self._available:
            return []

        gray = self._to_grayscale(image)
        if gray is None:
            return []

        try:
            candidates = self._find_rectangular_marks(gray)
            results = []
            for bbox, contour in candidates:
                state = self._classify_mark_state(gray, bbox)
                confidence = self._compute_confidence(contour, bbox)

                mark = DetectedMark(
                    mark_type=self._classify_mark_type(contour),
                    checked=state["checked"],
                    bbox=list(bbox),
                    confidence=round(confidence, 4),
                    fill_ratio=round(state["fill_ratio"], 4),
                    page_num=page_num,
                )
                results.append(mark)

            return results
        except Exception as exc:
            logger.warning(
                "OMR detection failed on page %d: %s", page_num, exc
            )
            return []

    def detect_page(self, image, page_num: int = 0) -> PageMarks:
        """Detect marks on a page image and return structured results.

        Args:
            image: PIL Image or numpy array.
            page_num: Page number for result attribution (1-based).

        Returns:
            PageMarks dataclass with all detected marks.
        """
        marks = self.detect_marks(image, page_num)

        mark_dicts = []
        checked_count = 0
        for m in marks:
            if m.checked:
                checked_count += 1
            mark_dicts.append({
                "mark_type": m.mark_type,
                "checked": m.checked,
                "bbox": m.bbox,
                "confidence": m.confidence,
                "fill_ratio": m.fill_ratio,
                "page_num": m.page_num,
            })

        return PageMarks(
            page_num=page_num,
            marks=mark_dicts,
            total_marks=len(mark_dicts),
            checked_marks=checked_count,
            unchecked_marks=len(mark_dicts) - checked_count,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _to_grayscale(self, image):
        """Convert image to grayscale numpy array.

        Args:
            image: PIL Image or numpy array.

        Returns:
            Grayscale numpy array or None.
        """
        if not self._available:
            return None

        try:
            # Handle PIL Image
            if hasattr(image, "convert"):
                return np.array(image.convert("L"))

            # Handle numpy array
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                return image

            return None
        except Exception:
            return None

    def _find_rectangular_marks(self, gray):
        """Find contours that look like checkboxes or radio buttons.

        Uses adaptive thresholding and contour filtering based on size,
        aspect ratio, and solidity.

        Args:
            gray: Grayscale numpy array.

        Returns:
            List of (bbox_tuple, contour) pairs where bbox is (x1, y1, x2, y2).
        """
        # Adaptive thresholding to handle varying illumination
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for contour in contours:
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Size filter
            if w < self._min_size or h < self._min_size:
                continue
            if w > self._max_size or h > self._max_size:
                continue

            # Aspect ratio filter (should be roughly square)
            aspect = w / h if h > 0 else 0
            if aspect < _MIN_ASPECT_RATIO or aspect > _MAX_ASPECT_RATIO:
                continue

            # Solidity filter (area / convex hull area)
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < _MIN_SOLIDITY:
                continue

            bbox = (x, y, x + w, y + h)
            candidates.append((bbox, contour))

        return candidates

    def _classify_mark_state(self, gray, bbox):
        """Determine if a mark is checked (filled) or unchecked (empty).

        Uses fill ratio analysis: counts dark pixels within the mark's
        bounding box and compares to the total area.

        Args:
            gray: Grayscale numpy array.
            bbox: Bounding box tuple (x1, y1, x2, y2).

        Returns:
            Dict with 'checked' (bool) and 'fill_ratio' (float).
        """
        x1, y1, x2, y2 = bbox
        roi = gray[y1:y2, x1:x2]

        if roi.size == 0:
            return {"checked": False, "fill_ratio": 0.0}

        # Threshold the ROI to find dark pixels (potential fill)
        _, roi_binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Calculate fill ratio (dark pixels / total pixels)
        total_pixels = roi.size
        dark_pixels = cv2.countNonZero(roi_binary)
        fill_ratio = dark_pixels / total_pixels

        checked = fill_ratio >= _FILL_RATIO_CHECKED_THRESHOLD

        return {"checked": checked, "fill_ratio": fill_ratio}

    def _classify_mark_type(self, contour) -> str:
        """Classify contour as checkbox or radio button.

        Uses circularity metric: circles have circularity close to 1.0.

        Args:
            contour: OpenCV contour.

        Returns:
            "radio" for circular marks, "checkbox" for rectangular.
        """
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return "checkbox"

        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        # High circularity indicates a round mark (radio button)
        if circularity > 0.80:
            return "radio"
        return "checkbox"

    def _compute_confidence(self, contour, bbox) -> float:
        """Compute detection confidence based on shape quality.

        Considers aspect ratio nearness to 1.0 and contour solidity.

        Args:
            contour: OpenCV contour.
            bbox: Bounding box tuple (x1, y1, x2, y2).

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        confidence = _CONFIDENCE_BASE

        # Aspect ratio bonus: closer to 1.0 = more likely a mark
        if h > 0:
            aspect = w / h
            aspect_quality = 1.0 - abs(1.0 - aspect)
            confidence += _CONFIDENCE_ASPECT_BONUS * max(0.0, aspect_quality)

        # Solidity bonus
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            confidence += _CONFIDENCE_SOLIDITY_BONUS * solidity

        return min(1.0, confidence)
