"""EasyOCR third engine for handwriting-optimized OCR.

Provides an optional EasyOCR engine for the forensic OCR pipeline,
specifically targeting handwritten document regions where EasyOCR
outperforms PaddleOCR and Tesseract.

EasyOCR is always run in CPU-only mode (``gpu=False``) to avoid
VRAM contention with the primary PaddleOCR engine.

Configuration via environment variables:
- ``ENABLE_EASYOCR``: Enable the engine (default: "false")
- ``EASYOCR_LANGUAGES``: Comma-separated language codes (default: "en")

When the ``easyocr`` package is not installed, all public functions
degrade gracefully -- returning ``None`` or empty results without
raising exceptions.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded import
# ---------------------------------------------------------------------------

try:
    import easyocr as _easyocr_mod  # noqa: F401

    _EASYOCR_AVAILABLE = True
except ImportError:
    _easyocr_mod = None
    _EASYOCR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENABLE_EASYOCR = os.environ.get("ENABLE_EASYOCR", "false").lower().strip() in (
    "true",
    "1",
    "yes",
)

EASYOCR_LANGUAGES = [
    lang.strip()
    for lang in os.environ.get("EASYOCR_LANGUAGES", "en").split(",")
    if lang.strip()
]

# ---------------------------------------------------------------------------
# Language mapping: PaddleOCR codes -> EasyOCR codes
# ---------------------------------------------------------------------------

# PaddleOCR uses slightly different language codes than EasyOCR.
# This mapping converts PaddleOCR codes to EasyOCR equivalents.
# Only languages supported by both engines are listed.
try:
    from ocr_local.config.language_config import EASYOCR_MAP as _REGISTRY_EASYOCR
    PADDLE_TO_EASYOCR_LANG = dict(_REGISTRY_EASYOCR)
except ImportError:
    PADDLE_TO_EASYOCR_LANG = {
        "en": "en",
        "ch": "ch_sim",
        "chinese_cht": "ch_tra",
        "fr": "fr",
        "german": "de",
        "es": "es",
        "it": "it",
        "pt": "pt",
        "nl": "nl",
        "ru": "ru",
        "uk": "uk",
        "be": "be",
        "bg": "bg",
        "ar": "ar",
        "hi": "hi",
        "japan": "ja",
        "korean": "ko",
        "vi": "vi",
        "tr": "tr",
        "pl": "pl",
        "cs": "cs",
        "hu": "hu",
        "ro": "ro",
        "da": "da",
        "fi": "fi",
        "sv": "sv",
        "el": "el",
    }


def map_paddle_to_easyocr(paddle_code):
    """Convert a PaddleOCR language code to an EasyOCR language code.

    Parameters
    ----------
    paddle_code : str
        PaddleOCR language code (e.g. "en", "german", "ch").

    Returns
    -------
    str or None
        EasyOCR language code, or ``None`` if no mapping exists.
    """
    if not paddle_code:
        return None
    return PADDLE_TO_EASYOCR_LANG.get(paddle_code.strip().lower())


# ---------------------------------------------------------------------------
# EasyOCR Engine wrapper
# ---------------------------------------------------------------------------


class EasyOCREngine:
    """Thread-safe, lazily-initialized EasyOCR wrapper.

    The underlying ``easyocr.Reader`` is created on the first call to
    :meth:`ocr`, not at construction time.  This avoids model loading
    overhead when the engine is never actually used.

    EasyOCR is always initialized with ``gpu=False`` to avoid VRAM
    contention with PaddleOCR.

    Parameters
    ----------
    languages : list[str] or None
        EasyOCR language codes.  Defaults to ``["en"]``.
    gpu : bool
        **Always forced to False**.  Accepted for API symmetry but
        ignored -- the engine is CPU-only by design.
    """

    def __init__(self, languages=None, gpu=False):
        self._languages = languages or ["en"]
        # CPU-only: never pass gpu=True regardless of caller request
        self._gpu = False
        self._reader = None
        self._lock = threading.Lock()

    @property
    def is_available(self):
        """Return True if the easyocr package is importable."""
        return _EASYOCR_AVAILABLE

    @property
    def languages(self):
        """Return the configured language list."""
        return list(self._languages)

    def _ensure_reader(self):
        """Lazily create the EasyOCR Reader (thread-safe).

        Raises
        ------
        RuntimeError
            If the ``easyocr`` package is not installed.
        """
        if self._reader is not None:
            return
        with self._lock:
            # Double-checked locking
            if self._reader is not None:
                return
            if not _EASYOCR_AVAILABLE:
                raise RuntimeError(
                    "easyocr package is not installed. "
                    "Install with: pip install easyocr>=1.7"
                )
            logger.info(
                "Initializing EasyOCR Reader: languages=%s, gpu=%s",
                self._languages,
                self._gpu,
            )
            self._reader = _easyocr_mod.Reader(
                self._languages, gpu=self._gpu
            )

    def ocr(self, image, languages=None):
        """Run EasyOCR on an image and return normalized results.

        Parameters
        ----------
        image : PIL.Image.Image or numpy.ndarray
            Input image to process.
        languages : list[str] or None
            Override languages for this call.  If provided and different
            from the engine's configured languages, a new Reader is
            **not** created -- EasyOCR uses the Reader's language set.
            This parameter is accepted for API compatibility but logged
            as a warning if it differs from the configured languages.

        Returns
        -------
        list[tuple[str, list, float]]
            List of ``(text, bbox, confidence)`` triples where:
            - ``text`` is the recognized string
            - ``bbox`` is ``[x1, y1, x2, y2]`` bounding box
            - ``confidence`` is a float in ``[0.0, 1.0]``

            Returns an empty list on any error or if easyocr is
            unavailable.
        """
        if not _EASYOCR_AVAILABLE:
            logger.debug("EasyOCR not available, returning empty results")
            return []

        if languages and set(languages) != set(self._languages):
            logger.warning(
                "EasyOCR language override requested (%s) but Reader "
                "was initialized with %s; using Reader languages",
                languages,
                self._languages,
            )

        try:
            self._ensure_reader()
        except RuntimeError:
            return []

        try:
            # Convert PIL Image to numpy array if needed
            import numpy as np

            if hasattr(image, "convert"):
                img_array = np.array(image.convert("RGB"))
            elif isinstance(image, np.ndarray):
                img_array = image
            else:
                logger.warning("EasyOCR: unsupported image type %s", type(image))
                return []

            # EasyOCR returns: list of (bbox, text, confidence)
            # where bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            raw_results = self._reader.readtext(img_array)

            return _normalize_results(raw_results)
        except Exception as exc:
            logger.warning("EasyOCR inference failed: %s", exc)
            return []


def _normalize_results(raw_results):
    """Convert EasyOCR output to pipeline-standard format.

    EasyOCR returns ``(bbox_polygon, text, confidence)`` where
    ``bbox_polygon`` is four corner points.  We convert to
    ``(text, [x1, y1, x2, y2], confidence)`` axis-aligned bounding
    boxes matching PaddleOCR's normalized output format.

    Parameters
    ----------
    raw_results : list
        Raw EasyOCR ``readtext()`` output.

    Returns
    -------
    list[tuple[str, list, float]]
        Normalized ``(text, bbox, confidence)`` triples.
    """
    normalized = []
    for detection in raw_results:
        try:
            bbox_polygon, text, confidence = detection
            # bbox_polygon: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            # Convert to axis-aligned [x_min, y_min, x_max, y_max]
            xs = [pt[0] for pt in bbox_polygon]
            ys = [pt[1] for pt in bbox_polygon]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            normalized.append((str(text), bbox, float(confidence)))
        except (ValueError, TypeError, IndexError) as exc:
            logger.debug("Skipping malformed EasyOCR detection: %s", exc)
            continue
    return normalized


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_engine_instance = None
_engine_lock = threading.Lock()


def get_easyocr_engine(languages=None):
    """Return a singleton EasyOCREngine instance, or None if unavailable.

    When ``ENABLE_EASYOCR`` is ``False`` or the ``easyocr`` package is
    not installed, returns ``None``.

    Parameters
    ----------
    languages : list[str] or None
        EasyOCR language codes.  Only used on first call (to initialize
        the singleton).  Defaults to :data:`EASYOCR_LANGUAGES`.

    Returns
    -------
    EasyOCREngine or None
        Singleton engine instance, or ``None`` if disabled/unavailable.
    """
    global _engine_instance

    if not ENABLE_EASYOCR:
        logger.debug("EasyOCR is disabled (ENABLE_EASYOCR != true)")
        return None

    if not _EASYOCR_AVAILABLE:
        logger.debug("EasyOCR package is not installed")
        return None

    if _engine_instance is not None:
        return _engine_instance

    with _engine_lock:
        # Double-checked locking
        if _engine_instance is not None:
            return _engine_instance
        langs = languages or EASYOCR_LANGUAGES
        _engine_instance = EasyOCREngine(languages=langs, gpu=False)
        logger.info("Created EasyOCR engine singleton: languages=%s", langs)
        return _engine_instance


def reset_engine():
    """Reset the singleton engine (for testing only).

    This is exposed for unit tests that need to clear module-level
    state between test cases.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
