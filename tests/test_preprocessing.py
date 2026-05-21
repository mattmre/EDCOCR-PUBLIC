"""Tests for image preprocessing module (Phase 4D).

These tests verify that preprocessing functions:
- Accept PIL Images and return PIL Images
- Preserve image dimensions (except binarize which changes mode)
- Handle edge cases (tiny images, grayscale, RGBA)
- Gracefully degrade when OpenCV is not available
- Never raise exceptions (return original on failure)

Run with: python -m pytest tests/test_preprocessing.py -v
"""


import numpy as np
import pytest
from PIL import Image

# Add project root to path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def white_image():
    """A 200x300 white RGB image (simple document-like)."""
    return Image.new("RGB", (200, 300), color=(255, 255, 255))


@pytest.fixture
def gray_image():
    """A 200x300 grayscale image."""
    return Image.new("L", (200, 300), color=200)


@pytest.fixture
def rgba_image():
    """A 200x300 RGBA image with alpha channel."""
    return Image.new("RGBA", (200, 300), color=(255, 255, 255, 128))


@pytest.fixture
def tiny_image():
    """A 5x5 image -- below minimum dimension threshold."""
    return Image.new("RGB", (5, 5), color=(128, 128, 128))


@pytest.fixture
def large_image():
    """A 2550x3300 image (letter-size at 300 DPI)."""
    return Image.new("RGB", (2550, 3300), color=(240, 240, 240))


@pytest.fixture
def noisy_image():
    """A 200x300 image with random noise (simulates degraded scan)."""
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 256, (300, 200, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def skewed_text_image():
    """A 400x300 image with a black line drawn at a slight angle.

    This gives the deskew function something to detect, though in a unit
    test context the Hough lines may or may not fire depending on
    parameters.  The important thing is it does not crash.
    """
    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    arr = np.array(img)
    # Draw a dark line from (20, 140) to (380, 160) -- slight upward slope
    for x in range(20, 380):
        y = 140 + int((x - 20) * 20 / 360)
        if 0 <= y < 300:
            arr[y, x] = [0, 0, 0]
            if y + 1 < 300:
                arr[y + 1, x] = [0, 0, 0]
    return Image.fromarray(arr)


@pytest.fixture
def dense_skewed_text_image():
    """A denser synthetic page used for deterministic deskew helper tests."""
    arr = np.full((800, 600), 255, dtype=np.uint8)
    for y in range(120, 700, 40):
        arr[y : y + 6, 80:520] = 0
    base = Image.fromarray(arr, mode="L")
    return base.rotate(
        4.0,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=255,
    ).convert("RGB")


# ---------------------------------------------------------------------------
# Tests: Individual functions
# ---------------------------------------------------------------------------

class TestDeskew:
    """Tests for deskew_image()."""

    def test_deskew_returns_image(self, white_image):
        from preprocessing import deskew_image
        result = deskew_image(white_image)
        assert isinstance(result, Image.Image)

    def test_deskew_handles_already_straight(self, white_image):
        """A blank white image should come back unchanged (no skew detected)."""
        from preprocessing import deskew_image
        result = deskew_image(white_image)
        assert isinstance(result, Image.Image)
        assert result.size == white_image.size

    def test_deskew_preserves_dimensions(self, skewed_text_image):
        from preprocessing import deskew_image
        result = deskew_image(skewed_text_image)
        assert result.size == skewed_text_image.size

    def test_deskew_tiny_image_returns_original(self, tiny_image):
        from preprocessing import deskew_image
        result = deskew_image(tiny_image)
        # Should return original unchanged (too small)
        assert result is tiny_image

    def test_hough_estimator_returns_finite_angle(self, dense_skewed_text_image):
        import preprocessing as mod

        if not mod._CV2_AVAILABLE:
            pytest.skip("OpenCV not available")

        gray = mod._resize_for_deskew(mod._pil_to_gray(dense_skewed_text_image))
        angle = mod._estimate_skew_hough(gray)
        assert angle is not None
        assert np.isfinite(angle)
        assert 0.5 <= abs(angle) <= 12.0

    def test_min_area_estimator_returns_finite_angle(self, dense_skewed_text_image):
        import preprocessing as mod

        if not mod._CV2_AVAILABLE:
            pytest.skip("OpenCV not available")

        gray = mod._resize_for_deskew(mod._pil_to_gray(dense_skewed_text_image))
        angle = mod._estimate_skew_min_area(gray)
        assert angle is not None
        assert np.isfinite(angle)
        assert 0.5 <= abs(angle) <= 12.0

    def test_select_best_skew_angle_prefers_non_zero(self, dense_skewed_text_image):
        import preprocessing as mod

        if not mod._CV2_AVAILABLE:
            pytest.skip("OpenCV not available")

        gray = mod._pil_to_gray(dense_skewed_text_image)
        angle = mod._select_best_skew_angle(gray)
        assert 0.5 <= abs(angle) <= 12.0

    def test_deskew_improves_projection_score(self, dense_skewed_text_image):
        import preprocessing as mod

        if not mod._CV2_AVAILABLE:
            pytest.skip("OpenCV not available")

        before_gray = mod._resize_for_deskew(mod._pil_to_gray(dense_skewed_text_image))
        before_binary = mod._deskew_binary(before_gray)
        before_score = mod._score_projection(before_binary, 0.0)

        result = mod.deskew_image(dense_skewed_text_image)
        after_gray = mod._resize_for_deskew(mod._pil_to_gray(result))
        after_binary = mod._deskew_binary(after_gray)
        after_score = mod._score_projection(after_binary, 0.0)

        assert after_score > before_score


class TestBinarize:
    """Tests for binarize_adaptive()."""

    def test_binarize_returns_image(self, white_image):
        from preprocessing import binarize_adaptive
        result = binarize_adaptive(white_image)
        assert isinstance(result, Image.Image)

    def test_binarize_returns_binary_image(self, noisy_image):
        """Output should be grayscale with only black and white pixels."""
        from preprocessing import binarize_adaptive
        result = binarize_adaptive(noisy_image)
        assert result.mode == "L"
        arr = np.array(result)
        unique = set(np.unique(arr))
        # Adaptive threshold produces only 0 and 255
        assert unique.issubset({0, 255})

    def test_binarize_preserves_size(self, white_image):
        from preprocessing import binarize_adaptive
        result = binarize_adaptive(white_image)
        assert result.size == white_image.size


class TestDenoise:
    """Tests for denoise_image()."""

    def test_denoise_returns_image(self, noisy_image):
        from preprocessing import denoise_image
        result = denoise_image(noisy_image)
        assert isinstance(result, Image.Image)

    def test_denoise_preserves_dimensions(self, noisy_image):
        from preprocessing import denoise_image
        result = denoise_image(noisy_image)
        assert result.size == noisy_image.size

    def test_denoise_reduces_noise(self, noisy_image):
        """Denoised image should have lower pixel variance than noisy input."""
        from preprocessing import denoise_image
        result = denoise_image(noisy_image)
        original_std = np.std(np.array(noisy_image).astype(float))
        result_std = np.std(np.array(result.convert("RGB")).astype(float))
        # Bilateral filter should smooth out noise
        assert result_std < original_std


class TestEnhanceContrast:
    """Tests for enhance_contrast()."""

    def test_contrast_returns_image(self, white_image):
        from preprocessing import enhance_contrast
        result = enhance_contrast(white_image)
        assert isinstance(result, Image.Image)

    def test_contrast_preserves_dimensions(self, white_image):
        from preprocessing import enhance_contrast
        result = enhance_contrast(white_image)
        assert result.size == white_image.size


# ---------------------------------------------------------------------------
# Tests: Pipeline function (preprocess_for_ocr)
# ---------------------------------------------------------------------------

class TestPreprocessForOcr:
    """Tests for the main preprocess_for_ocr() pipeline."""

    def test_level_none_returns_original(self, white_image):
        """Level 'none' should return the exact same object."""
        from preprocessing import preprocess_for_ocr
        result = preprocess_for_ocr(white_image, level="none")
        assert result is white_image

    def test_level_standard_returns_image(self, white_image):
        from preprocessing import preprocess_for_ocr
        result = preprocess_for_ocr(white_image, level="standard")
        assert isinstance(result, Image.Image)

    def test_level_enhanced_returns_image(self, white_image):
        from preprocessing import preprocess_for_ocr
        result = preprocess_for_ocr(white_image, level="enhanced")
        assert isinstance(result, Image.Image)

    def test_level_aggressive_returns_image(self, white_image):
        from preprocessing import preprocess_for_ocr
        result = preprocess_for_ocr(white_image, level="aggressive")
        assert isinstance(result, Image.Image)

    def test_handles_grayscale_input(self, gray_image):
        """Grayscale images should be handled without error at all levels."""
        from preprocessing import preprocess_for_ocr
        for level in ("standard", "enhanced", "aggressive"):
            result = preprocess_for_ocr(gray_image, level=level)
            assert isinstance(result, Image.Image), f"Failed at level={level}"

    def test_handles_rgba_input(self, rgba_image):
        """RGBA images should be handled without error."""
        from preprocessing import preprocess_for_ocr
        result = preprocess_for_ocr(rgba_image, level="enhanced")
        assert isinstance(result, Image.Image)

    def test_tiny_image_returns_original(self, tiny_image):
        """Images below minimum dimension should be returned unchanged."""
        from preprocessing import preprocess_for_ocr
        result = preprocess_for_ocr(tiny_image, level="aggressive")
        assert result is tiny_image

    def test_large_image_returns_image(self, large_image):
        """Normal document-size images should process without error."""
        from preprocessing import preprocess_for_ocr
        result = preprocess_for_ocr(large_image, level="standard")
        assert isinstance(result, Image.Image)
        assert result.size == large_image.size

    def test_invalid_level_defaults_to_standard(self, white_image):
        """Unknown level string should fall back to 'standard' behavior."""
        from preprocessing import preprocess_for_ocr
        # Should not raise; should log a warning and default to standard
        result = preprocess_for_ocr(white_image, level="turbo")
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# Tests: OpenCV not available
# ---------------------------------------------------------------------------

class TestNoCv2:
    """Tests verifying graceful fallback when OpenCV is not installed."""

    def test_preprocess_without_opencv(self, white_image):
        """When cv2 is unavailable, preprocess_for_ocr returns the original."""
        import preprocessing as mod

        original_flag = mod._CV2_AVAILABLE
        try:
            mod._CV2_AVAILABLE = False
            result = mod.preprocess_for_ocr(white_image, level="aggressive")
            assert result is white_image
        finally:
            mod._CV2_AVAILABLE = original_flag

    def test_deskew_without_opencv(self, white_image):
        import preprocessing as mod

        original_flag = mod._CV2_AVAILABLE
        try:
            mod._CV2_AVAILABLE = False
            result = mod.deskew_image(white_image)
            assert result is white_image
        finally:
            mod._CV2_AVAILABLE = original_flag

    def test_binarize_without_opencv(self, white_image):
        import preprocessing as mod

        original_flag = mod._CV2_AVAILABLE
        try:
            mod._CV2_AVAILABLE = False
            result = mod.binarize_adaptive(white_image)
            assert result is white_image
        finally:
            mod._CV2_AVAILABLE = original_flag

    def test_denoise_without_opencv(self, white_image):
        import preprocessing as mod

        original_flag = mod._CV2_AVAILABLE
        try:
            mod._CV2_AVAILABLE = False
            result = mod.denoise_image(white_image)
            assert result is white_image
        finally:
            mod._CV2_AVAILABLE = original_flag

    def test_contrast_without_opencv(self, white_image):
        import preprocessing as mod

        original_flag = mod._CV2_AVAILABLE
        try:
            mod._CV2_AVAILABLE = False
            result = mod.enhance_contrast(white_image)
            assert result is white_image
        finally:
            mod._CV2_AVAILABLE = original_flag
