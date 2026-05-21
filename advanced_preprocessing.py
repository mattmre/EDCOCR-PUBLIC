"""Advanced image preprocessing for degraded and distorted document scans.

Extends the existing preprocessing module (Phase 4D) with:
- Perspective correction via four-point transform
- Document boundary detection using contour analysis
- Adaptive thresholding with multi-method selection
- Enhanced degraded scan restoration pipeline

All functions accept numpy arrays (uint8) and return numpy arrays.
The ``AdvancedPreprocessor`` class orchestrates the full pipeline and
emits metadata for forensic traceability.

OpenCV (cv2) is the sole image processing dependency.  If not installed,
the module gracefully returns inputs unchanged.

This module is designed to NEVER crash the pipeline -- every public
method wraps its logic in try/except and returns the original image
on failure.
"""

import logging
import os

import numpy as np

logger = logging.getLogger("ocr_pipeline")

# ---------------------------------------------------------------------------
# OpenCV availability guard
# ---------------------------------------------------------------------------
try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("opencv-python-headless not installed; advanced preprocessing disabled")


# ---------------------------------------------------------------------------
# Configuration (environment variables, all opt-in)
# ---------------------------------------------------------------------------
ENABLE_ADVANCED_PREPROCESSING = os.environ.get(
    "ENABLE_ADVANCED_PREPROCESSING", "false"
).lower() in ("1", "true", "yes")

ADVANCED_PREPROCESS_PERSPECTIVE = os.environ.get(
    "ADVANCED_PREPROCESS_PERSPECTIVE", "true"
).lower() in ("1", "true", "yes")

ADVANCED_PREPROCESS_BOUNDARY_DETECT = os.environ.get(
    "ADVANCED_PREPROCESS_BOUNDARY_DETECT", "true"
).lower() in ("1", "true", "yes")

ADVANCED_PREPROCESS_ADAPTIVE_THRESHOLD = os.environ.get(
    "ADVANCED_PREPROCESS_ADAPTIVE_THRESHOLD", "true"
).lower() in ("1", "true", "yes")

ADVANCED_PREPROCESS_DEGRADED_ENHANCE = os.environ.get(
    "ADVANCED_PREPROCESS_DEGRADED_ENHANCE", "false"
).lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_DIM = 16
_BOUNDARY_APPROX_EPSILON_FACTOR = 0.02
_MIN_BOUNDARY_AREA_RATIO = 0.15
_MAX_BOUNDARY_AREA_RATIO = 0.98
_PERSPECTIVE_MIN_DIM = 100
_ADAPTIVE_BLOCK_SIZE = 11
_ADAPTIVE_C = 2
_SAUVOLA_WINDOW_SIZE = 25
_SAUVOLA_K = 0.2
_NIBLACK_WINDOW_SIZE = 25
_NIBLACK_K = -0.2
_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_SIZE = (8, 8)
_BILATERAL_D = 9
_BILATERAL_SIGMA_COLOR = 75
_BILATERAL_SIGMA_SPACE = 75
_MORPH_KERNEL_SIZE = (2, 2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to single-channel grayscale if needed."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an image to 3-channel BGR if needed."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _too_small(image: np.ndarray) -> bool:
    """Return True if the image is too small for meaningful preprocessing."""
    h, w = image.shape[:2]
    return w < _MIN_DIM or h < _MIN_DIM


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as: top-left, top-right, bottom-right, bottom-left.

    Uses the sum and difference of coordinates to determine corner positions.

    Parameters
    ----------
    pts : np.ndarray
        Array of shape (4, 2) with (x, y) coordinates.

    Returns
    -------
    np.ndarray
        Ordered array of shape (4, 2).
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(s)]      # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]      # bottom-right has largest sum
    rect[1] = pts[np.argmin(d)]      # top-right has smallest difference
    rect[3] = pts[np.argmax(d)]      # bottom-left has largest difference

    return rect


def _compute_contrast_score(image: np.ndarray) -> float:
    """Compute a text/background contrast score for a binarized image.

    Higher scores indicate better separation between text and background.
    Uses the ratio of foreground-to-total variance (inter-class variance
    analog).

    Parameters
    ----------
    image : np.ndarray
        Grayscale or binarized image (uint8).

    Returns
    -------
    float
        Contrast score in range [0, 1]. Higher is better.
    """
    gray = _to_grayscale(image) if image.ndim > 2 else image
    if gray.size == 0:
        return 0.0

    total_var = float(np.var(gray.astype(np.float64)))
    if total_var < 1e-6:
        return 0.0

    # Use Otsu threshold to split foreground/background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_mask = binary == 0
    bg_mask = binary == 255

    fg_count = int(np.sum(fg_mask))
    bg_count = int(np.sum(bg_mask))
    total = fg_count + bg_count

    if total == 0 or fg_count == 0 or bg_count == 0:
        return 0.0

    fg_mean = float(np.mean(gray[fg_mask]))
    bg_mean = float(np.mean(gray[bg_mask]))

    w_fg = fg_count / total
    w_bg = bg_count / total

    inter_class_var = w_fg * w_bg * (fg_mean - bg_mean) ** 2
    return float(min(inter_class_var / (total_var + 1e-6), 1.0))


# ---------------------------------------------------------------------------
# AdvancedPreprocessor
# ---------------------------------------------------------------------------


class AdvancedPreprocessor:
    """Advanced image preprocessing for degraded and distorted document scans.

    All methods accept numpy arrays (uint8, BGR or grayscale) and return
    numpy arrays.  Metadata about applied transforms is collected for
    forensic traceability.

    Thread-safe: no mutable shared state between calls.
    """

    def detect_document_boundary(
        self, image: np.ndarray
    ) -> list[tuple[int, int]] | None:
        """Find document edges in a scanned or photographed image.

        Uses Canny edge detection and contour approximation to locate the
        largest quadrilateral region that could be a document.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or grayscale, uint8).

        Returns
        -------
        list[tuple[int, int]] or None
            Four corner points as (x, y) tuples ordered top-left, top-right,
            bottom-right, bottom-left.  Returns ``None`` if no clear
            document boundary is detected.
        """
        if not _CV2_AVAILABLE or _too_small(image):
            return None

        try:
            gray = _to_grayscale(image)
            h, w = gray.shape[:2]
            image_area = h * w

            # Blur to reduce noise before edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # Dilate edges to close small gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return None

            # Sort by area descending and look for quadrilateral
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours[:10]:  # Check top 10 largest
                area = cv2.contourArea(contour)
                area_ratio = area / image_area

                if area_ratio < _MIN_BOUNDARY_AREA_RATIO:
                    continue
                if area_ratio > _MAX_BOUNDARY_AREA_RATIO:
                    continue

                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(
                    contour, _BOUNDARY_APPROX_EPSILON_FACTOR * peri, True
                )

                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    ordered = _order_corners(pts)
                    return [
                        (int(ordered[i][0]), int(ordered[i][1]))
                        for i in range(4)
                    ]

            return None

        except Exception as exc:
            logger.warning("Document boundary detection failed: %s", exc)
            return None

    def perspective_correct(
        self, image: np.ndarray, corners: list[tuple[int, int]] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Correct perspective distortion using a four-point transform.

        If ``corners`` is not provided, attempts automatic document
        boundary detection via ``detect_document_boundary``.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or grayscale, uint8).
        corners : list[tuple[int, int]] or None
            Optional pre-detected corner points.

        Returns
        -------
        tuple[np.ndarray, dict]
            Corrected image and metadata dict with keys:
            ``applied``, ``corners``, ``output_size``.
        """
        meta = {"applied": False, "corners": None, "output_size": None}

        if not _CV2_AVAILABLE or _too_small(image):
            return image, meta

        try:
            h, w = image.shape[:2]
            if w < _PERSPECTIVE_MIN_DIM or h < _PERSPECTIVE_MIN_DIM:
                return image, meta

            if corners is None:
                corners = self.detect_document_boundary(image)

            if corners is None or len(corners) != 4:
                return image, meta

            src = np.array(corners, dtype=np.float32)
            ordered = _order_corners(src)

            # Compute output dimensions from the ordered corners
            width_top = np.linalg.norm(ordered[1] - ordered[0])
            width_bottom = np.linalg.norm(ordered[2] - ordered[3])
            out_w = int(max(width_top, width_bottom))

            height_left = np.linalg.norm(ordered[3] - ordered[0])
            height_right = np.linalg.norm(ordered[2] - ordered[1])
            out_h = int(max(height_left, height_right))

            if out_w < _MIN_DIM or out_h < _MIN_DIM:
                return image, meta

            dst = np.array(
                [
                    [0, 0],
                    [out_w - 1, 0],
                    [out_w - 1, out_h - 1],
                    [0, out_h - 1],
                ],
                dtype=np.float32,
            )

            matrix = cv2.getPerspectiveTransform(ordered, dst)
            corrected = cv2.warpPerspective(
                image, matrix, (out_w, out_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            meta["applied"] = True
            meta["corners"] = [
                (int(ordered[i][0]), int(ordered[i][1]))
                for i in range(4)
            ]
            meta["output_size"] = (out_w, out_h)

            return corrected, meta

        except Exception as exc:
            logger.warning("Perspective correction failed: %s", exc)
            return image, meta

    def adaptive_deskew(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Enhanced deskew using Hough line detection and projection profile.

        Combines Hough line angle estimation with projection profile
        scoring to select the optimal rotation angle.  Falls back to
        the existing deskew estimator logic if the advanced method fails.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or grayscale, uint8).

        Returns
        -------
        tuple[np.ndarray, float]
            Deskewed image and the detected angle in degrees.
            Angle is 0.0 if no correction was applied.
        """
        if not _CV2_AVAILABLE or _too_small(image):
            return image, 0.0

        try:
            gray = _to_grayscale(image)
            h, w = gray.shape[:2]

            # Downscale for faster angle estimation
            max_dim = max(h, w)
            scale = 1.0
            if max_dim > 1600:
                scale = 1600.0 / max_dim
                working = cv2.resize(
                    gray,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                working = gray

            angle = self._estimate_angle(working)

            if abs(angle) < 0.5:
                return image, 0.0

            # Apply rotation to the full-resolution image
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image,
                rotation_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            logger.debug("Advanced deskew: corrected %.2f degrees", angle)
            return rotated, float(angle)

        except Exception as exc:
            logger.warning("Advanced deskew failed: %s", exc)
            return image, 0.0

    def _estimate_angle(self, gray: np.ndarray) -> float:
        """Estimate skew angle using multiple methods and projection scoring.

        Tries Hough line detection and minAreaRect, then picks the
        candidate whose projection profile variance is highest (indicating
        well-aligned text rows).

        Parameters
        ----------
        gray : np.ndarray
            Grayscale image, potentially downscaled.

        Returns
        -------
        float
            Best estimated skew angle in degrees.
        """
        candidates = {0.0}

        # Method 1: Hough line detection
        hough_angle = self._hough_angle(gray)
        if hough_angle is not None and abs(hough_angle) >= 0.5:
            candidates.add(round(hough_angle, 2))

        # Method 2: minAreaRect on connected components
        rect_angle = self._min_area_rect_angle(gray)
        if rect_angle is not None and abs(rect_angle) >= 0.5:
            candidates.add(round(rect_angle, 2))
            candidates.add(round(-rect_angle, 2))

        if len(candidates) == 1:
            return 0.0

        # Score each candidate by projection profile variance
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        scores = {}
        for angle in candidates:
            scores[angle] = self._projection_score(binary, angle)

        baseline = scores.get(0.0, 0.0)
        best_angle, best_score = max(scores.items(), key=lambda x: x[1])

        if abs(best_angle) < 0.5:
            return 0.0
        # Require at least 2% improvement over baseline
        if baseline > 0 and best_score < baseline * 1.02:
            return 0.0

        # Clamp to safe range
        if abs(best_angle) > 12.0:
            return 0.0

        return float(best_angle)

    def _hough_angle(self, gray: np.ndarray) -> float | None:
        """Estimate skew via Hough line detection."""
        import math

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=max(gray.shape[1] // 5, 20),
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
            # Normalize to [-45, 45] range
            normalized = ((angle + 45.0) % 90.0) - 45.0
            if abs(normalized) <= 12.0:
                angles.append(normalized)

        if not angles:
            return None

        return float(np.median(angles))

    def _min_area_rect_angle(self, gray: np.ndarray) -> float | None:
        """Estimate skew from dominant foreground mass via minAreaRect."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Connect text regions horizontally
        kernel_width = max(15, gray.shape[1] // 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        image_area = gray.shape[0] * gray.shape[1]
        points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 64 or area > image_area * 0.85:
                continue
            points.append(contour.reshape(-1, 2))

        if not points:
            fg = cv2.findNonZero(closed)
            if fg is None:
                return None
            points = [fg.reshape(-1, 2)]

        merged = np.vstack(points).astype(np.float32)
        if len(merged) < 5:
            return None

        (_, _), (w_rect, h_rect), angle = cv2.minAreaRect(merged)
        correction = float(angle)
        if w_rect < h_rect:
            correction -= 90.0
        # Normalize
        normalized = ((correction + 45.0) % 90.0) - 45.0
        if abs(normalized) > 12.0:
            return None
        return normalized

    def _projection_score(self, binary: np.ndarray, angle: float) -> float:
        """Score a candidate angle by horizontal projection variance."""
        h, w = binary.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(
            binary, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        row_sums = np.sum(rotated > 0, axis=1).astype(np.float32)
        return float(np.var(row_sums))

    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Multi-method adaptive thresholding for degraded scans.

        Evaluates three thresholding approaches and selects the one
        producing the best text/background contrast ratio:

        1. Otsu global thresholding
        2. Sauvola local thresholding
        3. Niblack local thresholding (inverted)

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or grayscale, uint8).

        Returns
        -------
        np.ndarray
            Binarized grayscale image (uint8, values 0 or 255).
        """
        if not _CV2_AVAILABLE or _too_small(image):
            return image

        try:
            gray = _to_grayscale(image)

            candidates = {}

            # Method 1: Otsu
            _, otsu = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            candidates["otsu"] = otsu

            # Method 2: Sauvola (approximated using local mean and std)
            sauvola = self._sauvola_threshold(gray)
            if sauvola is not None:
                candidates["sauvola"] = sauvola

            # Method 3: Niblack (approximated using local mean and std)
            niblack = self._niblack_threshold(gray)
            if niblack is not None:
                candidates["niblack"] = niblack

            if not candidates:
                return gray

            # Select by contrast score
            best_name = None
            best_score = -1.0
            for name, result in candidates.items():
                score = _compute_contrast_score(result)
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_name is None:
                return gray

            logger.debug(
                "Adaptive threshold: selected %s (score=%.3f)",
                best_name, best_score,
            )
            return candidates[best_name]

        except Exception as exc:
            logger.warning("Adaptive thresholding failed: %s", exc)
            return image

    def _sauvola_threshold(
        self, gray: np.ndarray,
        window_size: int = _SAUVOLA_WINDOW_SIZE,
        k: float = _SAUVOLA_K,
    ) -> np.ndarray | None:
        """Sauvola local thresholding.

        T(x,y) = mean(x,y) * (1 + k * (std(x,y) / R - 1))
        where R = 128 for 8-bit images.
        """
        try:
            # Ensure odd window size
            ws = window_size if window_size % 2 == 1 else window_size + 1

            gray_f = gray.astype(np.float64)
            mean = cv2.blur(gray_f, (ws, ws))
            mean_sq = cv2.blur(gray_f * gray_f, (ws, ws))
            std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

            r = 128.0
            threshold = mean * (1.0 + k * (std / r - 1.0))

            binary = np.where(gray_f > threshold, 255, 0).astype(np.uint8)
            return binary

        except Exception:
            return None

    def _niblack_threshold(
        self, gray: np.ndarray,
        window_size: int = _NIBLACK_WINDOW_SIZE,
        k: float = _NIBLACK_K,
    ) -> np.ndarray | None:
        """Niblack local thresholding.

        T(x,y) = mean(x,y) + k * std(x,y)
        """
        try:
            ws = window_size if window_size % 2 == 1 else window_size + 1

            gray_f = gray.astype(np.float64)
            mean = cv2.blur(gray_f, (ws, ws))
            mean_sq = cv2.blur(gray_f * gray_f, (ws, ws))
            std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

            threshold = mean + k * std

            binary = np.where(gray_f > threshold, 255, 0).astype(np.uint8)
            return binary

        except Exception:
            return None

    def enhance_degraded(self, image: np.ndarray) -> np.ndarray:
        """Restoration pipeline for severely degraded scans.

        Applies four stages sequentially:
        1. CLAHE contrast enhancement
        2. Bilateral filter noise reduction
        3. Adaptive thresholding (best-method selection)
        4. Morphological cleanup (close small gaps)

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or grayscale, uint8).

        Returns
        -------
        np.ndarray
            Enhanced image (grayscale, uint8).
        """
        if not _CV2_AVAILABLE or _too_small(image):
            return image

        try:
            result = image.copy()

            # Stage 1: CLAHE contrast enhancement
            gray = _to_grayscale(result)
            clahe = cv2.createCLAHE(
                clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=_CLAHE_TILE_SIZE
            )
            enhanced = clahe.apply(gray)

            # Stage 2: Bilateral filter (edge-preserving noise reduction)
            denoised = cv2.bilateralFilter(
                enhanced,
                d=_BILATERAL_D,
                sigmaColor=_BILATERAL_SIGMA_COLOR,
                sigmaSpace=_BILATERAL_SIGMA_SPACE,
            )

            # Stage 3: Adaptive thresholding
            binarized = self.adaptive_threshold(denoised)

            # Stage 4: Morphological cleanup (close small gaps in text)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, _MORPH_KERNEL_SIZE
            )
            cleaned = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)

            logger.debug("Degraded scan enhancement complete")
            return cleaned

        except Exception as exc:
            logger.warning("Degraded scan enhancement failed: %s", exc)
            return image

    def process(
        self,
        image: np.ndarray,
        config: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Run the full advanced preprocessing pipeline.

        Steps are controlled by the ``config`` dict or by the module-level
        environment variable defaults.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or grayscale, uint8).
        config : dict or None
            Optional overrides. Recognized keys:
            ``perspective``, ``boundary_detect``,
            ``adaptive_threshold``, ``degraded_enhance``.

        Returns
        -------
        tuple[np.ndarray, dict]
            Processed image and metadata dict describing what transforms
            were applied, including angles, corners, and method names.
        """
        metadata = {
            "advanced_preprocessing": True,
            "transforms_applied": [],
            "perspective": {"applied": False},
            "deskew": {"applied": False, "angle": 0.0},
            "threshold": {"applied": False, "method": None},
            "degraded_enhance": {"applied": False},
        }

        if not _CV2_AVAILABLE or _too_small(image):
            metadata["advanced_preprocessing"] = False
            metadata["skip_reason"] = (
                "cv2_unavailable" if not _CV2_AVAILABLE else "image_too_small"
            )
            return image, metadata

        cfg = config or {}
        do_perspective = cfg.get("perspective", ADVANCED_PREPROCESS_PERSPECTIVE)
        do_adaptive_threshold = cfg.get(
            "adaptive_threshold", ADVANCED_PREPROCESS_ADAPTIVE_THRESHOLD
        )
        do_degraded = cfg.get(
            "degraded_enhance", ADVANCED_PREPROCESS_DEGRADED_ENHANCE
        )

        result = image.copy()

        # Step 1: Perspective correction
        if do_perspective:
            try:
                corrected, persp_meta = self.perspective_correct(result)
                if persp_meta.get("applied"):
                    result = corrected
                    metadata["perspective"] = persp_meta
                    metadata["transforms_applied"].append("perspective")
            except Exception as exc:
                logger.warning("Perspective correction step failed: %s", exc)

        # Step 2: Adaptive deskew
        try:
            deskewed, angle = self.adaptive_deskew(result)
            if abs(angle) >= 0.5:
                result = deskewed
                metadata["deskew"] = {"applied": True, "angle": angle}
                metadata["transforms_applied"].append("deskew")
        except Exception as exc:
            logger.warning("Adaptive deskew step failed: %s", exc)

        # Step 3: Degraded scan enhancement (includes threshold)
        if do_degraded:
            try:
                original_for_compare = result.copy()
                enhanced = self.enhance_degraded(result)
                # Verify enhancement did not make things worse
                orig_score = _compute_contrast_score(original_for_compare)
                new_score = _compute_contrast_score(enhanced)
                if new_score >= orig_score:
                    result = enhanced
                    metadata["degraded_enhance"] = {"applied": True}
                    metadata["transforms_applied"].append("degraded_enhance")
                else:
                    logger.debug(
                        "Degraded enhancement rejected: score %.3f < %.3f",
                        new_score, orig_score,
                    )
            except Exception as exc:
                logger.warning("Degraded enhancement step failed: %s", exc)

        # Step 4: Adaptive threshold (only if not already done via degraded)
        elif do_adaptive_threshold:
            try:
                thresholded = self.adaptive_threshold(result)
                # Verify thresholding improved contrast
                orig_score = _compute_contrast_score(result)
                new_score = _compute_contrast_score(thresholded)
                if new_score >= orig_score:
                    result = thresholded
                    metadata["threshold"] = {
                        "applied": True,
                        "method": "adaptive_multi",
                    }
                    metadata["transforms_applied"].append("adaptive_threshold")
                else:
                    logger.debug(
                        "Adaptive threshold rejected: score %.3f < %.3f",
                        new_score, orig_score,
                    )
            except Exception as exc:
                logger.warning("Adaptive threshold step failed: %s", exc)

        return result, metadata
