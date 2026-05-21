"""Tests for noise profiling module (adaptive preprocessing).

Verifies that noise estimation, illumination uniformity, contrast scoring,
and the overall profiling pipeline produce correct recommendations for
different image quality scenarios.

Run with: python -m pytest tests/test_noise_profiling.py -v
"""


import numpy as np
import pytest
from PIL import Image

# Add project root to path
from noise_profiling import (
    _NOISE_PROFILING_AVAILABLE,
    ILLUMINATION_UNIFORMITY_THRESHOLD,
    NOISE_VARIANCE_CLEAN_THRESHOLD,
    NoiseProfile,
    _default_profile,
    estimate_contrast,
    estimate_illumination_uniformity,
    estimate_noise_variance,
    estimate_snr,
    profile_image,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_gray():
    """A uniform white grayscale array (no noise)."""
    return np.full((200, 300), 200, dtype=np.uint8)


@pytest.fixture
def noisy_gray():
    """A grayscale array with significant random noise."""
    rng = np.random.RandomState(42)
    base = np.full((200, 300), 128, dtype=np.float64)
    noise = rng.normal(0, 50, size=(200, 300))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def gradient_gray():
    """A horizontal gradient image (uneven illumination)."""
    row = np.linspace(0, 255, 300, dtype=np.uint8)
    return np.tile(row, (200, 1))


@pytest.fixture
def bimodal_gray():
    """An image with two distinct intensity bands (high contrast)."""
    img = np.zeros((200, 300), dtype=np.uint8)
    img[:100, :] = 20
    img[100:, :] = 235
    return img


# ---------------------------------------------------------------------------
# estimate_noise_variance
# ---------------------------------------------------------------------------


class TestEstimateNoiseVariance:
    def test_clean_image_low_variance(self, clean_gray):
        """Uniform image should have very low noise variance."""
        var = estimate_noise_variance(clean_gray)
        assert var < NOISE_VARIANCE_CLEAN_THRESHOLD
        assert var >= 0.0

    def test_noisy_image_high_variance(self, noisy_gray):
        """Noisy image should have high noise variance."""
        var = estimate_noise_variance(noisy_gray)
        assert var > NOISE_VARIANCE_CLEAN_THRESHOLD

    def test_returns_float(self, clean_gray):
        result = estimate_noise_variance(clean_gray)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# estimate_illumination_uniformity
# ---------------------------------------------------------------------------


class TestEstimateIlluminationUniformity:
    def test_even_illumination(self, clean_gray):
        """Uniform image should have uniformity close to 1.0."""
        uniformity = estimate_illumination_uniformity(clean_gray)
        assert uniformity > 0.95

    def test_uneven_illumination(self, gradient_gray):
        """Gradient image should have low uniformity."""
        uniformity = estimate_illumination_uniformity(gradient_gray)
        assert uniformity < ILLUMINATION_UNIFORMITY_THRESHOLD

    def test_black_image(self):
        """All-zero image should return 1.0 (division-by-zero guard)."""
        black = np.zeros((100, 100), dtype=np.uint8)
        uniformity = estimate_illumination_uniformity(black)
        assert uniformity == 1.0

    def test_range_bounded(self, noisy_gray):
        uniformity = estimate_illumination_uniformity(noisy_gray)
        assert 0.0 <= uniformity <= 1.0


# ---------------------------------------------------------------------------
# estimate_contrast
# ---------------------------------------------------------------------------


class TestEstimateContrast:
    def test_bimodal_high_contrast(self, bimodal_gray):
        """Two-tone image should have high contrast score."""
        contrast = estimate_contrast(bimodal_gray)
        assert contrast > 0.5

    def test_uniform_low_contrast(self, clean_gray):
        """Uniform image should have zero contrast."""
        contrast = estimate_contrast(clean_gray)
        assert contrast < 0.05

    def test_range_bounded(self, noisy_gray):
        contrast = estimate_contrast(noisy_gray)
        assert 0.0 <= contrast <= 1.0


# ---------------------------------------------------------------------------
# estimate_snr
# ---------------------------------------------------------------------------


class TestEstimateSNR:
    def test_zero_noise_returns_high_snr(self, clean_gray):
        """Zero noise variance should return 100.0 (infinite SNR sentinel)."""
        snr = estimate_snr(clean_gray, 0.0)
        assert snr == 100.0

    def test_noisy_image_lower_snr(self, noisy_gray):
        """Noisy image should have finite SNR proportional to noise."""
        noise_var = estimate_noise_variance(noisy_gray)
        snr = estimate_snr(noisy_gray, noise_var)
        assert snr > 0.0
        assert snr < 100.0


# ---------------------------------------------------------------------------
# profile_image (end-to-end profiling)
# ---------------------------------------------------------------------------


class TestProfileImage:
    def test_clean_image_recommends_skip_denoise(self, clean_gray):
        """Clean image should recommend skip_denoise=True."""
        profile = profile_image(clean_gray)
        assert isinstance(profile, NoiseProfile)
        assert profile.noise_level == "clean"
        assert profile.skip_denoise is True
        assert profile.recommended_denoise_strength == 0.0

    def test_noisy_image_recommends_denoise(self, noisy_gray):
        """Noisy image should recommend denoising."""
        profile = profile_image(noisy_gray)
        assert profile.noise_level in ("moderate", "heavy")
        assert profile.skip_denoise is False
        assert profile.recommended_denoise_strength > 0.0

    def test_uneven_illumination_recommends_sauvola(self, gradient_gray):
        """Gradient image should recommend sauvola binarization."""
        profile = profile_image(gradient_gray)
        assert profile.has_uneven_illumination is True
        assert profile.recommended_binarize_method == "sauvola"

    def test_pil_input(self):
        """PIL Image input should work without errors."""
        pil_img = Image.new("RGB", (200, 300), color=(180, 180, 180))
        profile = profile_image(pil_img)
        assert isinstance(profile, NoiseProfile)
        assert profile.noise_level == "clean"

    def test_numpy_gray_input(self, clean_gray):
        """Numpy grayscale array input should work."""
        profile = profile_image(clean_gray)
        assert isinstance(profile, NoiseProfile)

    def test_numpy_bgr_input(self):
        """Numpy BGR (3-channel) array input should work."""
        bgr = np.full((200, 300, 3), 180, dtype=np.uint8)
        profile = profile_image(bgr)
        assert isinstance(profile, NoiseProfile)

    def test_clean_high_contrast_recommends_otsu(self):
        """Image with good contrast, even illumination, and moderate noise gets otsu."""
        # Build a roughly 50/50 dark/light image with even quadrant distribution.
        # Each quadrant gets half dark (30) and half light (220) rows, producing
        # high IQR contrast and balanced quadrant means.  Edge transitions
        # register as moderate noise in the Laplacian estimator, but the
        # profiler should still recommend otsu for this scenario.
        img = np.full((200, 300), 220, dtype=np.uint8)
        for y in range(0, 200, 20):
            img[y : y + 10, :] = 30
        profile = profile_image(img)
        assert profile.noise_level in ("clean", "moderate")
        assert profile.contrast_score > 0.3
        assert profile.has_uneven_illumination is False
        assert profile.recommended_binarize_method == "otsu"

    def test_low_contrast_recommends_sauvola(self):
        """Low-contrast image should recommend sauvola."""
        # Create a very narrow intensity range image
        low = np.full((200, 300), 128, dtype=np.uint8)
        low[:100, :] = 130  # Only 2 pixel difference
        profile = profile_image(low)
        assert profile.contrast_score < 0.3
        assert profile.recommended_binarize_method == "sauvola"


# ---------------------------------------------------------------------------
# Module availability
# ---------------------------------------------------------------------------


class TestModuleAvailability:
    def test_noise_profiling_available(self):
        """Module should report available when cv2 is importable."""
        assert _NOISE_PROFILING_AVAILABLE is True

    def test_default_profile_safe(self):
        """Default profile should be a safe no-op."""
        profile = _default_profile()
        assert profile.skip_denoise is True
        assert profile.noise_level == "clean"
        assert profile.recommended_binarize_method == "otsu"


# ---------------------------------------------------------------------------
# NoiseProfile dataclass
# ---------------------------------------------------------------------------


class TestNoiseProfileDataclass:
    def test_fields_present(self):
        """All expected fields should be present on the dataclass."""
        expected_fields = {
            "noise_variance",
            "noise_level",
            "illumination_uniformity",
            "has_uneven_illumination",
            "estimated_snr",
            "contrast_score",
            "recommended_denoise_strength",
            "recommended_binarize_method",
            "skip_denoise",
        }
        actual_fields = {f.name for f in NoiseProfile.__dataclass_fields__.values()}
        assert expected_fields == actual_fields

    def test_construct_from_kwargs(self):
        """Dataclass should be constructable from keyword arguments."""
        profile = NoiseProfile(
            noise_variance=10.0,
            noise_level="clean",
            illumination_uniformity=0.95,
            has_uneven_illumination=False,
            estimated_snr=50.0,
            contrast_score=0.8,
            recommended_denoise_strength=0.0,
            recommended_binarize_method="otsu",
            skip_denoise=True,
        )
        assert profile.noise_variance == 10.0
        assert profile.noise_level == "clean"
