"""Extended image format loaders for HEIC, RAW, and DICOM.

Provides graceful-degradation loaders for image formats not natively
supported by Pillow.  Each loader is guarded by an availability flag
that is set at import time.  When the backing library is not installed,
the loader logs a warning once and returns ``None`` so the pipeline can
skip or fall back.

This module is designed to NEVER crash the pipeline -- every public
function wraps its logic in try/except and returns ``None`` on failure.

Optional dependencies (all are truly optional):
- pillow-heif >= 0.16  : HEIC / HEIF (iPhone photos)
- rawpy >= 0.19        : Camera RAW (CR2, NEF, ARW, DNG, RAW)
- pydicom >= 2.4       : DICOM medical imaging
"""

import logging
import os

import numpy as np
from PIL import Image

logger = logging.getLogger("ocr_pipeline")

# ---------------------------------------------------------------------------
# Optional dependency availability flags
# ---------------------------------------------------------------------------
_HEIF_AVAILABLE = False
_RAWPY_AVAILABLE = False
_PYDICOM_AVAILABLE = False

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _HEIF_AVAILABLE = True
except ImportError:
    pass

try:
    import rawpy  # noqa: F401

    _RAWPY_AVAILABLE = True
except ImportError:
    pass

try:
    import pydicom  # noqa: F401

    _PYDICOM_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Extension sets
# ---------------------------------------------------------------------------

# Extensions handled by pillow-heif
HEIC_EXTENSIONS = {".heic", ".heif"}

# Extensions handled by rawpy
RAW_EXTENSIONS = {".cr2", ".nef", ".arw", ".dng", ".raw", ".cr3", ".orf", ".rw2"}

# Extensions handled by pydicom
DICOM_EXTENSIONS = {".dcm", ".dicom"}

# All extended extensions this module can attempt to load
ALL_EXTENDED_EXTENSIONS = HEIC_EXTENSIONS | RAW_EXTENSIONS | DICOM_EXTENSIONS

# One-shot warning flags to avoid log spam
_warned_heif = False
_warned_rawpy = False
_warned_pydicom = False


# ---------------------------------------------------------------------------
# Individual format loaders
# ---------------------------------------------------------------------------


def load_heic(path: str) -> "Image.Image | None":
    """Load a HEIC/HEIF image file and return it as an RGB PIL Image.

    Requires the ``pillow-heif`` package.  When the package is not
    installed, logs a warning (once) and returns ``None``.

    Parameters
    ----------
    path : str
        Filesystem path to the HEIC/HEIF file.

    Returns
    -------
    PIL.Image.Image or None
        The decoded RGB image, or ``None`` on failure.
    """
    global _warned_heif

    if not _HEIF_AVAILABLE:
        if not _warned_heif:
            logger.warning(
                "pillow-heif not installed; HEIC/HEIF loading disabled. "
                "Install with: pip install pillow-heif"
            )
            _warned_heif = True
        return None

    try:
        # pillow-heif registers an opener so Pillow can read HEIC natively
        img = Image.open(path)
        img = img.convert("RGB")
        logger.debug("Loaded HEIC image: %s (%dx%d)", path, img.width, img.height)
        return img
    except Exception as exc:
        logger.warning("Failed to load HEIC file %s: %s", path, exc)
        return None


def load_raw(path: str) -> "Image.Image | None":
    """Load a camera RAW image file and return it as an RGB PIL Image.

    Supports CR2 (Canon), NEF (Nikon), ARW (Sony), DNG (Adobe),
    and generic RAW files via the ``rawpy`` package.

    Parameters
    ----------
    path : str
        Filesystem path to the RAW image file.

    Returns
    -------
    PIL.Image.Image or None
        The decoded RGB image, or ``None`` on failure.
    """
    global _warned_rawpy

    if not _RAWPY_AVAILABLE:
        if not _warned_rawpy:
            logger.warning(
                "rawpy not installed; Camera RAW loading disabled. "
                "Install with: pip install rawpy"
            )
            _warned_rawpy = True
        return None

    try:
        import rawpy

        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=8,
                no_auto_bright=False,
            )
        img = Image.fromarray(rgb, mode="RGB")
        logger.debug("Loaded RAW image: %s (%dx%d)", path, img.width, img.height)
        return img
    except Exception as exc:
        logger.warning("Failed to load RAW file %s: %s", path, exc)
        return None


def load_dicom(path: str) -> "Image.Image | None":
    """Load a DICOM medical image and return it as an RGB PIL Image.

    Handles both MONOCHROME1 (inverted) and MONOCHROME2 (standard)
    photometric interpretations.  Multi-frame DICOM files use only
    the first frame.

    Parameters
    ----------
    path : str
        Filesystem path to the DICOM file.

    Returns
    -------
    PIL.Image.Image or None
        The decoded RGB image, or ``None`` on failure.
    """
    global _warned_pydicom

    if not _PYDICOM_AVAILABLE:
        if not _warned_pydicom:
            logger.warning(
                "pydicom not installed; DICOM loading disabled. "
                "Install with: pip install pydicom"
            )
            _warned_pydicom = True
        return None

    try:
        import pydicom

        ds = pydicom.dcmread(path)
        pixel_array = ds.pixel_array

        # Handle multi-frame DICOM (take first frame for grayscale stacks).
        if pixel_array.ndim == 3:
            if getattr(ds, "SamplesPerPixel", 1) > 1:
                # Likely (H, W, channels) -- keep as-is.
                pass
            else:
                # Likely (frames, H, W) -- take first frame.
                pixel_array = pixel_array[0]

        # Normalize to uint8 range
        arr = pixel_array.astype(np.float64)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
        arr = arr.astype(np.uint8)

        # Handle MONOCHROME1 (inverted: 0 = white, max = black)
        photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        if photometric == "MONOCHROME1":
            arr = 255 - arr

        # Convert grayscale to RGB
        if arr.ndim == 2:
            img = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            img = Image.fromarray(arr).convert("RGB")

        logger.debug("Loaded DICOM image: %s (%dx%d)", path, img.width, img.height)
        return img
    except Exception as exc:
        logger.warning("Failed to load DICOM file %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def load_extended_image(path: str) -> "Image.Image | None":
    """Load an image using the appropriate extended format loader.

    Routes the file to the correct loader based on its extension.
    Returns ``None`` if the format is not supported or loading fails.

    Parameters
    ----------
    path : str
        Filesystem path to the image file.

    Returns
    -------
    PIL.Image.Image or None
        The decoded RGB image, or ``None`` if unsupported or on error.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in HEIC_EXTENSIONS:
        return load_heic(path)
    if ext in RAW_EXTENSIONS:
        return load_raw(path)
    if ext in DICOM_EXTENSIONS:
        return load_dicom(path)

    logger.debug("No extended loader for extension %s: %s", ext, path)
    return None


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def get_supported_extensions() -> set:
    """Return the set of file extensions currently loadable.

    Only includes extensions whose backing library is installed.

    Returns
    -------
    set of str
        Lowercase dotted extensions (e.g. ``{".heic", ".heif"}``).
    """
    supported = set()
    if _HEIF_AVAILABLE:
        supported |= HEIC_EXTENSIONS
    if _RAWPY_AVAILABLE:
        supported |= RAW_EXTENSIONS
    if _PYDICOM_AVAILABLE:
        supported |= DICOM_EXTENSIONS
    return supported


def is_format_supported(path: str) -> bool:
    """Check whether the file at *path* can be loaded by an extended loader.

    Returns ``True`` only if the extension is recognized **and** the
    required library is installed.

    Parameters
    ----------
    path : str
        Filesystem path (only the extension is inspected).

    Returns
    -------
    bool
    """
    ext = os.path.splitext(path)[1].lower()
    return ext in get_supported_extensions()
