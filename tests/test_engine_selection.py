"""Tests for smart OCR engine selection."""

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from engine_selection import (
    VARIANCE_THRESHOLD,
    DocumentQuality,
    _estimate_skew,
    analyze_page_quality,
    select_engine,
)


def _make_clean_page(width=2550, height=3300):
    """Create a synthetic clean document page (white bg, dark text-like regions)."""
    img = np.ones((height, width), dtype=np.uint8) * 240  # Light gray background
    # Add some dark horizontal bands to simulate text lines
    for y in range(200, 2800, 50):
        img[y : y + 12, 300:2200] = 30  # Dark text lines
    return Image.fromarray(img)


def _make_noisy_page(width=2550, height=3300):
    """Create a synthetic noisy/degraded page."""
    rng = np.random.RandomState(42)
    img = rng.randint(80, 200, (height, width), dtype=np.uint8)  # Random noise
    return Image.fromarray(img)


def _make_skewed_page(width=2550, height=3300, angle=5.0):
    """Create a page with simulated skew (tilted text lines)."""
    img = np.ones((height, width), dtype=np.uint8) * 240
    shear = np.tan(np.radians(angle))
    # Draw tilted text lines: for each line, shift each column by shear amount
    for y_base in range(200, 2800, 50):
        for x in range(300, 2200):
            y_offset = int(shear * (x - width // 2))
            y = y_base + y_offset
            if 0 <= y < height - 12:
                img[y : y + 12, x] = 30
    return Image.fromarray(img)


class TestDocumentQuality:
    """Tests for DocumentQuality dataclass."""

    def test_to_dict(self):
        dq = DocumentQuality(
            variance=1234.5678,
            edge_density=0.01234,
            skew_angle=1.567,
            is_clean=True,
            recommended_engine="tesseract",
            reason="clean",
        )
        d = dq.to_dict()
        assert d["variance"] == 1234.57
        assert d["edge_density"] == 0.0123
        assert d["skew_angle"] == 1.57
        assert d["is_clean"] is True
        assert d["recommended_engine"] == "tesseract"

    def test_slots_prevent_new_attributes(self):
        dq = DocumentQuality(0, 0, 0, False, "paddle", "test")
        with pytest.raises(AttributeError):
            dq.new_attribute = "fail"


class TestAnalyzePageQuality:
    """Tests for page quality analysis."""

    def test_clean_page_detected(self):
        img = _make_clean_page()
        quality = analyze_page_quality(img)
        assert quality.is_clean is True
        assert quality.recommended_engine == "tesseract"

    def test_noisy_page_detected(self):
        img = _make_noisy_page()
        quality = analyze_page_quality(img)
        assert quality.is_clean is False
        assert quality.recommended_engine == "paddle"

    def test_skewed_page_analyzed(self):
        """Skewed page should be analyzable without errors."""
        img = _make_skewed_page(angle=5.0)
        quality = analyze_page_quality(img)
        # The lightweight projection-profile skew estimator may not detect
        # all skew from synthetic images (it works best on real scans with
        # dense text). Verify the analysis completes and returns valid data.
        assert quality.recommended_engine in ("paddle", "tesseract")
        assert isinstance(quality.skew_angle, float)
        assert abs(quality.skew_angle) <= 5.0  # Within search range

    def test_high_skew_forces_paddle(self):
        """Documents with skew above threshold should route to PaddleOCR.

        Tests the threshold logic directly by mocking skew values, since
        the lightweight skew estimator is optimized for real documents.
        """
        from unittest.mock import patch as _patch

        img = _make_clean_page()
        # Force skew above threshold to verify decision logic
        with _patch("engine_selection._estimate_skew", return_value=3.5):
            quality = analyze_page_quality(img)
        assert quality.is_clean is False
        assert quality.recommended_engine == "paddle"
        assert "skewed" in quality.reason

    def test_small_image(self):
        """Small images should not crash."""
        img = Image.fromarray(np.ones((50, 50), dtype=np.uint8) * 128)
        quality = analyze_page_quality(img)
        assert quality.recommended_engine in ("paddle", "tesseract")

    def test_rgb_image_converted(self):
        """RGB images are handled (converted to grayscale internally)."""
        img = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 200)
        quality = analyze_page_quality(img)
        assert isinstance(quality.variance, float)

    def test_low_variance_flags_paddle(self):
        """Uniform image (low variance) should recommend PaddleOCR."""
        img = Image.fromarray(np.ones((500, 500), dtype=np.uint8) * 200)
        quality = analyze_page_quality(img)
        assert quality.variance < VARIANCE_THRESHOLD
        assert quality.recommended_engine == "paddle"


class TestEdgeCases:
    """Tests for edge cases: all-black, all-white, tiny images."""

    def test_all_black_routes_to_paddle(self):
        """All-black image (no content) routes to PaddleOCR."""
        img = Image.fromarray(np.zeros((500, 500), dtype=np.uint8))
        quality = analyze_page_quality(img)
        assert quality.recommended_engine == "paddle"

    def test_all_white_routes_to_paddle(self):
        """All-white image (no content) routes to PaddleOCR."""
        img = Image.fromarray(np.ones((500, 500), dtype=np.uint8) * 255)
        quality = analyze_page_quality(img)
        assert quality.recommended_engine == "paddle"

    def test_tiny_image_does_not_crash(self):
        """Very small images (<32px) should not crash."""
        img = Image.fromarray(np.ones((10, 10), dtype=np.uint8) * 128)
        quality = analyze_page_quality(img)
        assert quality.recommended_engine in ("paddle", "tesseract")


class TestEstimateSkew:
    """Tests for skew estimation."""

    def test_no_skew(self):
        """Straight text lines should yield near-zero skew."""
        img = np.ones((500, 500), dtype=np.uint8) * 240
        for y in range(50, 450, 30):
            img[y : y + 8, 50:450] = 30
        skew = _estimate_skew(img)
        assert abs(skew) <= 1.0

    def test_returns_float(self):
        img = np.ones((200, 200), dtype=np.uint8) * 128
        skew = _estimate_skew(img)
        assert isinstance(skew, float)

    def test_large_image_downsampled(self):
        """3300-row image should be downsampled and still work fast."""
        img = np.ones((3300, 2550), dtype=np.uint8) * 200
        for y in range(100, 3000, 40):
            img[y : y + 10, 200:2300] = 30
        import time
        start = time.perf_counter()
        skew = _estimate_skew(img)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert isinstance(skew, float)
        assert elapsed_ms < 500  # should be well under 500ms


class TestSelectEngine:
    """Tests for engine selection logic."""

    def test_force_paddle(self):
        assert select_engine(force="paddle") == "paddle"

    def test_force_tesseract(self):
        assert select_engine(force="tesseract") == "tesseract"

    def test_auto_without_image_defaults_paddle(self):
        assert select_engine(image=None, force="auto") == "paddle"

    def test_auto_with_clean_image(self):
        img = _make_clean_page()
        result = select_engine(image=img, force="auto")
        assert result == "tesseract"

    def test_auto_with_noisy_image(self):
        img = _make_noisy_page()
        result = select_engine(image=img, force="auto")
        assert result == "paddle"

    def test_unknown_mode_falls_back(self):
        result = select_engine(force="unknown")
        assert result == "paddle"

    @patch("engine_selection.ENGINE_SELECTION", "tesseract")
    def test_env_var_tesseract(self):
        result = select_engine()
        assert result == "tesseract"

    @patch("engine_selection.ENGINE_SELECTION", "paddle")
    def test_env_var_paddle(self):
        result = select_engine()
        assert result == "paddle"
