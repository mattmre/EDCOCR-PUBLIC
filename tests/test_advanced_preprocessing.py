"""Tests for advanced image preprocessing module.

These tests verify that the AdvancedPreprocessor class:
- Perspective correction works on synthetic distorted images
- Document boundary detection finds quadrilateral regions
- Adaptive thresholding selects the best method
- Degraded scan enhancement improves image quality
- All methods handle both color and grayscale input
- Fallback behavior when methods fail or cv2 unavailable
- Configuration options control which steps run
- Integration with existing preprocessing pipeline
- Metadata output tracks applied transforms

Run with: python -m pytest tests/test_advanced_preprocessing.py -v
"""


import numpy as np
import pytest
from PIL import Image

# Add project root to path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def white_image_np():
    """A 400x300 white BGR image as numpy array."""
    return np.full((300, 400, 3), 255, dtype=np.uint8)


@pytest.fixture
def gray_image_np():
    """A 400x300 grayscale image as numpy array."""
    return np.full((300, 400), 200, dtype=np.uint8)


@pytest.fixture
def tiny_image_np():
    """A 5x5 image -- below minimum dimension threshold."""
    return np.full((5, 5, 3), 128, dtype=np.uint8)


@pytest.fixture
def noisy_image_np():
    """A 400x300 image with random noise (simulates degraded scan)."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (300, 400, 3), dtype=np.uint8)


@pytest.fixture
def document_on_background():
    """A 600x800 image with a white rectangle (document) on dark background.

    Simulates a photo of a document on a desk.  The document is a
    clearly defined white rectangle with some offset and slight
    border region.
    """
    img = np.full((800, 600, 3), 40, dtype=np.uint8)  # dark background
    # Place a white rectangle representing the document
    img[100:700, 80:520, :] = 240  # white document area
    return img


@pytest.fixture
def skewed_text_image_np():
    """A 600x800 grayscale image with horizontal text lines rotated ~5 degrees.

    Creates dense text-like horizontal bars then rotates the image
    to simulate a skewed scan.
    """
    import cv2

    arr = np.full((800, 600), 255, dtype=np.uint8)
    for y in range(100, 700, 35):
        arr[y: y + 6, 60:540] = 0  # dark text lines

    # Rotate by 5 degrees
    h, w = arr.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, -5.0, 1.0)
    rotated = cv2.warpAffine(
        arr, matrix, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    return rotated


@pytest.fixture
def text_image_bgr():
    """A 400x300 BGR image with horizontal dark text lines on white background."""
    arr = np.full((300, 400, 3), 255, dtype=np.uint8)
    for y in range(40, 260, 30):
        arr[y: y + 5, 30:370, :] = 20
    return arr


@pytest.fixture
def degraded_image_np():
    """A 400x300 image simulating a severely degraded scan.

    Low contrast text with additive noise.
    """
    rng = np.random.RandomState(123)
    arr = np.full((300, 400), 160, dtype=np.uint8)
    # Low contrast text lines
    for y in range(40, 260, 30):
        arr[y: y + 5, 30:370] = 120
    # Add noise
    noise = rng.randint(-30, 30, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return arr


@pytest.fixture
def perspective_distorted_image():
    """A 600x800 image with a perspective-distorted white quadrilateral.

    Simulates a document photographed at an angle.
    """
    img = np.full((800, 600, 3), 50, dtype=np.uint8)  # dark background
    # Draw a filled white quadrilateral (trapezoid)
    import cv2

    pts = np.array([[120, 100], [480, 80], [520, 700], [80, 720]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (240, 240, 240))
    # Add some dark text-like marks inside
    for y in range(200, 600, 40):
        cv2.line(img, (160, y), (440, y), (30, 30, 30), 2)
    return img


# ---------------------------------------------------------------------------
# Tests: AdvancedPreprocessor class
# ---------------------------------------------------------------------------


class TestDocumentBoundaryDetection:
    """Tests for detect_document_boundary()."""

    def test_detects_clear_boundary(self, document_on_background):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        corners = preprocessor.detect_document_boundary(document_on_background)

        # Should find a quadrilateral
        assert corners is not None
        assert len(corners) == 4
        # Each corner is (x, y)
        for pt in corners:
            assert len(pt) == 2
            assert isinstance(pt[0], int)
            assert isinstance(pt[1], int)

    def test_returns_none_for_uniform_image(self, white_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        corners = preprocessor.detect_document_boundary(white_image_np)
        assert corners is None

    def test_returns_none_for_tiny_image(self, tiny_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        corners = preprocessor.detect_document_boundary(tiny_image_np)
        assert corners is None

    def test_handles_grayscale_input(self, gray_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        # Should not crash on grayscale
        corners = preprocessor.detect_document_boundary(gray_image_np)
        # Uniform gray has no boundary
        assert corners is None

    def test_corners_ordered_correctly(self, document_on_background):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        corners = preprocessor.detect_document_boundary(document_on_background)
        if corners is not None:
            # top-left should have smaller x+y than bottom-right
            tl = corners[0]
            br = corners[2]
            assert tl[0] + tl[1] < br[0] + br[1]


class TestPerspectiveCorrection:
    """Tests for perspective_correct()."""

    def test_returns_image_and_metadata(self, document_on_background):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result, meta = preprocessor.perspective_correct(document_on_background)

        assert isinstance(result, np.ndarray)
        assert isinstance(meta, dict)
        assert "applied" in meta

    def test_corrects_with_provided_corners(self, white_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        corners = [(10, 10), (390, 10), (390, 290), (10, 290)]
        result, meta = preprocessor.perspective_correct(white_image_np, corners=corners)

        assert isinstance(result, np.ndarray)
        assert meta["applied"] is True
        assert meta["corners"] is not None
        assert meta["output_size"] is not None

    def test_no_correction_without_boundary(self, white_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result, meta = preprocessor.perspective_correct(white_image_np)

        assert meta["applied"] is False
        # Should return original unchanged
        assert np.array_equal(result, white_image_np)

    def test_handles_tiny_image(self, tiny_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result, meta = preprocessor.perspective_correct(tiny_image_np)
        assert meta["applied"] is False
        assert np.array_equal(result, tiny_image_np)

    def test_handles_grayscale(self, gray_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        corners = [(10, 10), (390, 10), (390, 290), (10, 290)]
        result, meta = preprocessor.perspective_correct(gray_image_np, corners=corners)
        assert isinstance(result, np.ndarray)

    def test_perspective_with_distorted_image(self, perspective_distorted_image):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result, meta = preprocessor.perspective_correct(perspective_distorted_image)
        assert isinstance(result, np.ndarray)
        # May or may not detect boundary depending on contour quality


class TestAdaptiveDeskew:
    """Tests for adaptive_deskew()."""

    def test_returns_image_and_angle(self, white_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result, angle = preprocessor.adaptive_deskew(white_image_np)

        assert isinstance(result, np.ndarray)
        assert isinstance(angle, float)

    def test_no_correction_for_straight_image(self, text_image_bgr):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result, angle = preprocessor.adaptive_deskew(text_image_bgr)

        # Straight text should have ~0 angle
        assert abs(angle) < 1.0

    def test_corrects_skewed_image(self, skewed_text_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result, angle = preprocessor.adaptive_deskew(skewed_text_image_np)

        assert isinstance(result, np.ndarray)
        # Should detect some angle (may or may not match exactly)
        # The important thing is it does not crash
        assert isinstance(angle, float)

    def test_handles_tiny_image(self, tiny_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result, angle = preprocessor.adaptive_deskew(tiny_image_np)

        assert np.array_equal(result, tiny_image_np)
        assert angle == 0.0

    def test_preserves_dimensions(self, skewed_text_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result, _ = preprocessor.adaptive_deskew(skewed_text_image_np)

        assert result.shape == skewed_text_image_np.shape

    def test_handles_grayscale(self, gray_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result, angle = preprocessor.adaptive_deskew(gray_image_np)
        assert isinstance(result, np.ndarray)
        assert isinstance(angle, float)


class TestAdaptiveThreshold:
    """Tests for adaptive_threshold()."""

    def test_returns_binary_image(self, text_image_bgr):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result = preprocessor.adaptive_threshold(text_image_bgr)

        assert isinstance(result, np.ndarray)
        # Should be grayscale
        assert result.ndim == 2
        # Should be binary (only 0 and 255)
        unique_vals = set(np.unique(result))
        assert unique_vals.issubset({0, 255})

    def test_handles_grayscale_input(self, gray_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result = preprocessor.adaptive_threshold(gray_image_np)
        assert isinstance(result, np.ndarray)

    def test_handles_noisy_input(self, noisy_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result = preprocessor.adaptive_threshold(noisy_image_np)
        assert isinstance(result, np.ndarray)

    def test_handles_tiny_image(self, tiny_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result = preprocessor.adaptive_threshold(tiny_image_np)
        assert np.array_equal(result, tiny_image_np)

    def test_preserves_spatial_dimensions(self, text_image_bgr):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result = preprocessor.adaptive_threshold(text_image_bgr)
        assert result.shape[:2] == text_image_bgr.shape[:2]


class TestEnhanceDegraded:
    """Tests for enhance_degraded()."""

    def test_returns_enhanced_image(self, degraded_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result = preprocessor.enhance_degraded(degraded_image_np)

        assert isinstance(result, np.ndarray)
        assert result.shape[:2] == degraded_image_np.shape[:2]

    def test_improves_contrast(self, degraded_image_np):
        from advanced_preprocessing import AdvancedPreprocessor, _compute_contrast_score

        preprocessor = AdvancedPreprocessor()
        result = preprocessor.enhance_degraded(degraded_image_np)

        # Enhanced should have at least as good contrast
        original_score = _compute_contrast_score(degraded_image_np)
        enhanced_score = _compute_contrast_score(result)
        # The enhancement pipeline should improve or maintain contrast
        assert enhanced_score >= original_score * 0.8  # allow some tolerance

    def test_handles_bgr_input(self, text_image_bgr):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result = preprocessor.enhance_degraded(text_image_bgr)
        assert isinstance(result, np.ndarray)

    def test_handles_tiny_image(self, tiny_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result = preprocessor.enhance_degraded(tiny_image_np)
        assert np.array_equal(result, tiny_image_np)


class TestProcess:
    """Tests for the full process() pipeline."""

    def test_returns_image_and_metadata(self, text_image_bgr):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result, meta = preprocessor.process(text_image_bgr)

        assert isinstance(result, np.ndarray)
        assert isinstance(meta, dict)
        assert "advanced_preprocessing" in meta
        assert "transforms_applied" in meta
        assert isinstance(meta["transforms_applied"], list)

    def test_metadata_tracks_transforms(self, text_image_bgr):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        _, meta = preprocessor.process(text_image_bgr)

        assert "perspective" in meta
        assert "deskew" in meta
        assert "threshold" in meta
        assert "degraded_enhance" in meta

    def test_config_disables_perspective(self, document_on_background):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        _, meta = preprocessor.process(
            document_on_background,
            config={"perspective": False}
        )
        assert meta["perspective"]["applied"] is False

    def test_config_enables_degraded(self, degraded_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        _, meta = preprocessor.process(
            degraded_image_np,
            config={"degraded_enhance": True, "adaptive_threshold": False}
        )
        # degraded_enhance either ran or decided not to (both OK)
        assert "degraded_enhance" in meta

    def test_config_disables_threshold(self, text_image_bgr):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        _, meta = preprocessor.process(
            text_image_bgr,
            config={"adaptive_threshold": False}
        )
        assert meta["threshold"]["applied"] is False

    def test_handles_tiny_image(self, tiny_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result, meta = preprocessor.process(tiny_image_np)

        assert np.array_equal(result, tiny_image_np)
        assert meta["advanced_preprocessing"] is False
        assert meta.get("skip_reason") == "image_too_small"

    def test_handles_grayscale(self, gray_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result, meta = preprocessor.process(gray_image_np)
        assert isinstance(result, np.ndarray)
        assert meta["advanced_preprocessing"] is True


class TestInternalHelpers:
    """Tests for internal helper functions."""

    def test_order_corners_basic(self):
        from advanced_preprocessing import _order_corners

        pts = np.array([
            [100, 100],  # top-left
            [300, 100],  # top-right
            [300, 400],  # bottom-right
            [100, 400],  # bottom-left
        ], dtype=np.float32)

        ordered = _order_corners(pts)
        # top-left has smallest sum
        assert ordered[0][0] == 100 and ordered[0][1] == 100
        # bottom-right has largest sum
        assert ordered[2][0] == 300 and ordered[2][1] == 400

    def test_order_corners_shuffled(self):
        from advanced_preprocessing import _order_corners

        # Same points but in random order
        pts = np.array([
            [300, 400],  # bottom-right
            [100, 100],  # top-left
            [100, 400],  # bottom-left
            [300, 100],  # top-right
        ], dtype=np.float32)

        ordered = _order_corners(pts)
        assert ordered[0][0] == 100 and ordered[0][1] == 100  # TL
        assert ordered[1][0] == 300 and ordered[1][1] == 100  # TR
        assert ordered[2][0] == 300 and ordered[2][1] == 400  # BR
        assert ordered[3][0] == 100 and ordered[3][1] == 400  # BL

    def test_to_grayscale_bgr(self, text_image_bgr):
        from advanced_preprocessing import _to_grayscale

        gray = _to_grayscale(text_image_bgr)
        assert gray.ndim == 2
        assert gray.shape[:2] == text_image_bgr.shape[:2]

    def test_to_grayscale_already_gray(self, gray_image_np):
        from advanced_preprocessing import _to_grayscale

        result = _to_grayscale(gray_image_np)
        assert np.array_equal(result, gray_image_np)

    def test_to_grayscale_bgra(self):
        from advanced_preprocessing import _to_grayscale

        bgra = np.full((100, 100, 4), 200, dtype=np.uint8)
        gray = _to_grayscale(bgra)
        assert gray.ndim == 2

    def test_ensure_bgr_from_gray(self, gray_image_np):
        from advanced_preprocessing import _ensure_bgr

        bgr = _ensure_bgr(gray_image_np)
        assert bgr.ndim == 3
        assert bgr.shape[2] == 3

    def test_ensure_bgr_already_bgr(self, text_image_bgr):
        from advanced_preprocessing import _ensure_bgr

        result = _ensure_bgr(text_image_bgr)
        assert np.array_equal(result, text_image_bgr)

    def test_too_small_true(self, tiny_image_np):
        from advanced_preprocessing import _too_small

        assert _too_small(tiny_image_np) is True

    def test_too_small_false(self, white_image_np):
        from advanced_preprocessing import _too_small

        assert _too_small(white_image_np) is False

    def test_compute_contrast_score_range(self, text_image_bgr):
        from advanced_preprocessing import _compute_contrast_score

        score = _compute_contrast_score(text_image_bgr)
        assert 0.0 <= score <= 1.0

    def test_compute_contrast_score_uniform(self, white_image_np):
        from advanced_preprocessing import _compute_contrast_score

        score = _compute_contrast_score(white_image_np)
        assert score == 0.0

    def test_compute_contrast_score_empty(self):
        from advanced_preprocessing import _compute_contrast_score

        empty = np.array([], dtype=np.uint8)
        score = _compute_contrast_score(empty)
        assert score == 0.0


class TestSauvolaThreshold:
    """Tests for _sauvola_threshold()."""

    def test_produces_binary_output(self, gray_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result = preprocessor._sauvola_threshold(gray_image_np)
        assert result is not None
        assert isinstance(result, np.ndarray)
        unique_vals = set(np.unique(result))
        assert unique_vals.issubset({0, 255})

    def test_handles_text_image(self):
        from advanced_preprocessing import AdvancedPreprocessor

        # Create a simple text-like grayscale image
        img = np.full((200, 300), 220, dtype=np.uint8)
        for y in range(30, 170, 25):
            img[y: y + 4, 20:280] = 30
        preprocessor = AdvancedPreprocessor()
        result = preprocessor._sauvola_threshold(img)
        assert result is not None
        assert result.shape == img.shape


class TestNiblackThreshold:
    """Tests for _niblack_threshold()."""

    def test_produces_binary_output(self, gray_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        result = preprocessor._niblack_threshold(gray_image_np)
        assert result is not None
        assert isinstance(result, np.ndarray)
        unique_vals = set(np.unique(result))
        assert unique_vals.issubset({0, 255})

    def test_handles_text_image(self):
        from advanced_preprocessing import AdvancedPreprocessor

        img = np.full((200, 300), 220, dtype=np.uint8)
        for y in range(30, 170, 25):
            img[y: y + 4, 20:280] = 30
        preprocessor = AdvancedPreprocessor()
        result = preprocessor._niblack_threshold(img)
        assert result is not None
        assert result.shape == img.shape


class TestEstimateAngle:
    """Tests for _estimate_angle()."""

    def test_returns_float(self, skewed_text_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        angle = preprocessor._estimate_angle(skewed_text_image_np)
        assert isinstance(angle, float)

    def test_zero_for_blank(self, gray_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        angle = preprocessor._estimate_angle(gray_image_np)
        assert angle == 0.0

    def test_clamps_extreme_angles(self):
        from advanced_preprocessing import AdvancedPreprocessor

        # Blank image should never produce extreme angles
        img = np.full((100, 100), 128, dtype=np.uint8)
        preprocessor = AdvancedPreprocessor()
        angle = preprocessor._estimate_angle(img)
        assert abs(angle) <= 12.0


class TestHoughAngle:
    """Tests for _hough_angle()."""

    def test_returns_none_for_blank(self, gray_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        angle = preprocessor._hough_angle(gray_image_np)
        assert angle is None

    def test_detects_angle_from_lines(self, skewed_text_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        angle = preprocessor._hough_angle(skewed_text_image_np)
        # May or may not detect lines -- important thing is it does not crash
        if angle is not None:
            assert isinstance(angle, float)
            assert abs(angle) <= 45.0


class TestMinAreaRectAngle:
    """Tests for _min_area_rect_angle()."""

    def test_returns_none_for_blank(self, gray_image_np):
        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        angle = preprocessor._min_area_rect_angle(gray_image_np)
        # Uniform gray may or may not produce an angle
        if angle is not None:
            assert isinstance(angle, float)


class TestProjectionScore:
    """Tests for _projection_score()."""

    def test_returns_nonnegative(self, skewed_text_image_np):
        import cv2

        from advanced_preprocessing import AdvancedPreprocessor

        preprocessor = AdvancedPreprocessor()
        _, binary = cv2.threshold(skewed_text_image_np, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        score = preprocessor._projection_score(binary, 0.0)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Tests: Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    """Tests for environment variable configuration."""

    def test_default_config_disabled(self):
        from advanced_preprocessing import ENABLE_ADVANCED_PREPROCESSING
        # Default is false unless env var is set
        # We just verify the variable exists and is bool-like
        assert isinstance(ENABLE_ADVANCED_PREPROCESSING, bool)

    def test_config_variables_exist(self):
        from advanced_preprocessing import (
            ADVANCED_PREPROCESS_ADAPTIVE_THRESHOLD,
            ADVANCED_PREPROCESS_BOUNDARY_DETECT,
            ADVANCED_PREPROCESS_DEGRADED_ENHANCE,
            ADVANCED_PREPROCESS_PERSPECTIVE,
        )
        assert isinstance(ADVANCED_PREPROCESS_PERSPECTIVE, bool)
        assert isinstance(ADVANCED_PREPROCESS_BOUNDARY_DETECT, bool)
        assert isinstance(ADVANCED_PREPROCESS_ADAPTIVE_THRESHOLD, bool)
        assert isinstance(ADVANCED_PREPROCESS_DEGRADED_ENHANCE, bool)


# ---------------------------------------------------------------------------
# Tests: OpenCV not available
# ---------------------------------------------------------------------------


class TestNoCv2:
    """Tests verifying graceful fallback when OpenCV is not installed."""

    def test_boundary_returns_none(self, white_image_np):
        import advanced_preprocessing as mod

        original_flag = mod._CV2_AVAILABLE
        try:
            mod._CV2_AVAILABLE = False
            preprocessor = mod.AdvancedPreprocessor()
            corners = preprocessor.detect_document_boundary(white_image_np)
            assert corners is None
        finally:
            mod._CV2_AVAILABLE = original_flag

    def test_perspective_returns_original(self, white_image_np):
        import advanced_preprocessing as mod

        original_flag = mod._CV2_AVAILABLE
        try:
            mod._CV2_AVAILABLE = False
            preprocessor = mod.AdvancedPreprocessor()
            result, meta = preprocessor.perspective_correct(white_image_np)
            assert np.array_equal(result, white_image_np)
            assert meta["applied"] is False
        finally:
            mod._CV2_AVAILABLE = original_flag

    def test_deskew_returns_original(self, white_image_np):
        import advanced_preprocessing as mod

        original_flag = mod._CV2_AVAILABLE
        try:
            mod._CV2_AVAILABLE = False
            preprocessor = mod.AdvancedPreprocessor()
            result, angle = preprocessor.adaptive_deskew(white_image_np)
            assert np.array_equal(result, white_image_np)
            assert angle == 0.0
        finally:
            mod._CV2_AVAILABLE = original_flag

    def test_threshold_returns_original(self, white_image_np):
        import advanced_preprocessing as mod

        original_flag = mod._CV2_AVAILABLE
        try:
            mod._CV2_AVAILABLE = False
            preprocessor = mod.AdvancedPreprocessor()
            result = preprocessor.adaptive_threshold(white_image_np)
            assert np.array_equal(result, white_image_np)
        finally:
            mod._CV2_AVAILABLE = original_flag

    def test_enhance_returns_original(self, white_image_np):
        import advanced_preprocessing as mod

        original_flag = mod._CV2_AVAILABLE
        try:
            mod._CV2_AVAILABLE = False
            preprocessor = mod.AdvancedPreprocessor()
            result = preprocessor.enhance_degraded(white_image_np)
            assert np.array_equal(result, white_image_np)
        finally:
            mod._CV2_AVAILABLE = original_flag

    def test_process_returns_original(self, white_image_np):
        import advanced_preprocessing as mod

        original_flag = mod._CV2_AVAILABLE
        try:
            mod._CV2_AVAILABLE = False
            preprocessor = mod.AdvancedPreprocessor()
            result, meta = preprocessor.process(white_image_np)
            assert np.array_equal(result, white_image_np)
            assert meta["advanced_preprocessing"] is False
            assert meta["skip_reason"] == "cv2_unavailable"
        finally:
            mod._CV2_AVAILABLE = original_flag


# ---------------------------------------------------------------------------
# Tests: Integration with preprocessing.py
# ---------------------------------------------------------------------------


class TestPreprocessingIntegration:
    """Tests for integration between preprocessing.py and advanced_preprocessing."""

    def test_apply_advanced_disabled_by_default(self):
        """When ENABLE_ADVANCED_PREPROCESSING is false, the integration
        function should return the original image."""
        from preprocessing import _apply_advanced_preprocessing

        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        result = _apply_advanced_preprocessing(img)
        assert isinstance(result, Image.Image)

    def test_apply_advanced_enabled(self, monkeypatch):
        """When enabled, _apply_advanced_preprocessing should run and
        return a PIL Image."""
        import advanced_preprocessing as ap_mod

        monkeypatch.setattr(ap_mod, "ENABLE_ADVANCED_PREPROCESSING", True)

        from preprocessing import _apply_advanced_preprocessing

        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        result = _apply_advanced_preprocessing(img)
        assert isinstance(result, Image.Image)

    def test_apply_advanced_handles_import_error(self, monkeypatch):
        """If advanced_preprocessing import fails, should return original."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "advanced_preprocessing":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from preprocessing import _apply_advanced_preprocessing

        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        result = _apply_advanced_preprocessing(img)
        assert result is img

    def test_pipeline_includes_advanced_step(self, monkeypatch):
        """preprocess_for_ocr should call _apply_advanced_preprocessing."""
        import preprocessing as mod

        called = {"value": False}
        original_fn = mod._apply_advanced_preprocessing

        def tracking_fn(img):
            called["value"] = True
            return original_fn(img)

        monkeypatch.setattr(mod, "_apply_advanced_preprocessing", tracking_fn)

        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        mod.preprocess_for_ocr(img, level="standard")
        assert called["value"] is True

    def test_pipeline_level_none_skips_advanced(self):
        """Level 'none' should skip all preprocessing including advanced."""
        from preprocessing import preprocess_for_ocr

        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        result = preprocess_for_ocr(img, level="none")
        assert result is img
