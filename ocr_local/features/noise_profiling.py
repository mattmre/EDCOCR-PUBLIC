"""Noise profiling for adaptive OCR preprocessing parameter selection.

Assesses image quality (noise level, illumination uniformity, contrast,
signal-to-noise ratio) and recommends preprocessing parameters before
the main deskew/denoise/binarize pipeline runs.

OpenCV (cv2) and numpy are optional dependencies.  When not installed,
``profile_image`` returns a safe default ``NoiseProfile`` that skips all
adaptive adjustments.

Enabled via ``ENABLE_NOISE_PROFILING=true`` environment variable or
``--enable-noise-profiling`` CLI flag.
"""

import dataclasses
import logging
import os

logger = logging.getLogger("ocr_pipeline")

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
_NOISE_PROFILING_AVAILABLE = False
try:
    import cv2
    import numpy as np

    _NOISE_PROFILING_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    logger.debug("opencv/numpy not installed; noise profiling unavailable")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENABLE_NOISE_PROFILING = os.environ.get(
    "ENABLE_NOISE_PROFILING", "false"
).lower() in ("1", "true", "yes")

NOISE_VARIANCE_CLEAN_THRESHOLD = 50.0   # Below = clean
NOISE_VARIANCE_HEAVY_THRESHOLD = 200.0  # Above = heavy
ILLUMINATION_UNIFORMITY_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class NoiseProfile:
    """Image quality assessment for adaptive preprocessing."""

    noise_variance: float          # Estimated noise level (Laplacian-based)
    noise_level: str               # "clean", "moderate", "heavy"
    illumination_uniformity: float  # 0.0 (very uneven) to 1.0 (perfectly uniform)
    has_uneven_illumination: bool  # True if uniformity < threshold
    estimated_snr: float           # Signal-to-noise ratio estimate
    contrast_score: float          # 0.0 (low contrast) to 1.0 (high contrast)
    recommended_denoise_strength: float  # 0.0 to 1.0
    recommended_binarize_method: str     # "adaptive", "otsu", "sauvola"
    skip_denoise: bool             # True if image is clean enough


# ---------------------------------------------------------------------------
# Core estimation functions
# ---------------------------------------------------------------------------


def estimate_noise_variance(gray: "np.ndarray") -> float:
    """Estimate noise variance using the Laplacian method (Immerkaer 1996).

    Parameters
    ----------
    gray : np.ndarray
        Single-channel uint8 image.

    Returns
    -------
    float
        Estimated noise variance (higher = noisier).
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sigma = (np.pi / 2.0) ** 0.5 * (1.0 / 6.0) * float(np.abs(laplacian).mean())
    return float(sigma ** 2)


def estimate_illumination_uniformity(gray: "np.ndarray") -> float:
    """Estimate illumination uniformity by comparing quadrant means.

    Splits the image into four quadrants and measures how much each
    quadrant's mean intensity deviates from the overall mean.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel uint8 image.

    Returns
    -------
    float
        Uniformity score from 0.0 (very uneven) to 1.0 (perfectly uniform).
    """
    h, w = gray.shape[:2]
    quadrants = [
        gray[:h // 2, :w // 2],
        gray[:h // 2, w // 2:],
        gray[h // 2:, :w // 2],
        gray[h // 2:, w // 2:],
    ]
    means = [float(q.mean()) for q in quadrants]
    overall_mean = float(gray.mean())
    if overall_mean == 0:
        return 1.0
    max_deviation = max(abs(m - overall_mean) for m in means)
    uniformity = 1.0 - min(max_deviation / overall_mean, 1.0)
    return uniformity


def estimate_contrast(gray: "np.ndarray") -> float:
    """Estimate contrast using the inter-quartile range of the intensity histogram.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel uint8 image.

    Returns
    -------
    float
        Contrast score from 0.0 (low contrast) to 1.0 (high contrast).
    """
    q25, q75 = np.percentile(gray, [25, 75])
    iqr = float(q75) - float(q25)
    return min(iqr / 255.0, 1.0)


def estimate_snr(gray: "np.ndarray", noise_var: float) -> float:
    """Estimate signal-to-noise ratio.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel uint8 image.
    noise_var : float
        Estimated noise variance from ``estimate_noise_variance``.

    Returns
    -------
    float
        SNR estimate (signal variance / noise variance).
    """
    signal_var = float(np.var(gray))
    if noise_var <= 0:
        return 100.0  # Effectively infinite SNR
    return signal_var / noise_var


# ---------------------------------------------------------------------------
# Main profiling entry point
# ---------------------------------------------------------------------------


def profile_image(image) -> NoiseProfile:
    """Profile an image (PIL or numpy) for noise characteristics.

    Accepts either a PIL Image or a numpy array (grayscale or BGR).
    Returns a ``NoiseProfile`` with quality metrics and preprocessing
    recommendations.

    Parameters
    ----------
    image : PIL.Image.Image or np.ndarray
        The source document page image.

    Returns
    -------
    NoiseProfile
        Quality assessment with adaptive preprocessing recommendations.
    """
    import numpy as _np  # local import for safety

    if not _NOISE_PROFILING_AVAILABLE:
        return _default_profile()

    # Convert PIL to numpy if needed
    if hasattr(image, "convert"):
        gray = _np.array(image.convert("L"))
    elif len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    noise_var = estimate_noise_variance(gray)
    uniformity = estimate_illumination_uniformity(gray)
    contrast = estimate_contrast(gray)
    snr = estimate_snr(gray, noise_var)

    # Classify noise level
    if noise_var < NOISE_VARIANCE_CLEAN_THRESHOLD:
        noise_level = "clean"
    elif noise_var < NOISE_VARIANCE_HEAVY_THRESHOLD:
        noise_level = "moderate"
    else:
        noise_level = "heavy"

    has_uneven = uniformity < ILLUMINATION_UNIFORMITY_THRESHOLD

    # Recommendations
    skip_denoise = noise_level == "clean"

    if noise_level == "heavy":
        denoise_strength = min(noise_var / 500.0, 1.0)
    elif noise_level == "moderate":
        denoise_strength = min(noise_var / 400.0, 0.5)
    else:
        denoise_strength = 0.0

    if has_uneven:
        binarize_method = "sauvola"
    elif contrast < 0.3:
        binarize_method = "sauvola"
    elif noise_level == "heavy":
        binarize_method = "adaptive"
    else:
        binarize_method = "otsu"

    return NoiseProfile(
        noise_variance=noise_var,
        noise_level=noise_level,
        illumination_uniformity=uniformity,
        has_uneven_illumination=has_uneven,
        estimated_snr=snr,
        contrast_score=contrast,
        recommended_denoise_strength=denoise_strength,
        recommended_binarize_method=binarize_method,
        skip_denoise=skip_denoise,
    )


def _default_profile() -> NoiseProfile:
    """Return a safe default profile when cv2/numpy are not available."""
    return NoiseProfile(
        noise_variance=0.0,
        noise_level="clean",
        illumination_uniformity=1.0,
        has_uneven_illumination=False,
        estimated_snr=100.0,
        contrast_score=1.0,
        recommended_denoise_strength=0.0,
        recommended_binarize_method="otsu",
        skip_denoise=True,
    )
