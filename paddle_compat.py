"""PaddleOCR 2.x utility module.

Provides helper functions for PaddleOCR 2.9.1 engine creation, result
normalization, and device configuration.

Note: v3 support was removed in the 2.x stabilization migration
(2026-03-01). This module provides helper functions for PaddleOCR 2.9.1.
"""

import logging

logger = logging.getLogger(__name__)


def get_paddle_version():
    """Return PaddlePaddle major version as int (2 or 3), or None if not installed.

    Parses the ``paddle.__version__`` string and extracts the major
    version number.  Returns ``None`` if PaddlePaddle is not importable
    or the version string is malformed.

    Returns
    -------
    int or None
        Major version number (2, 3, ...) or None.
    """
    try:
        import paddle

        version_str = getattr(paddle, "__version__", "0.0.0")
        major = int(version_str.split(".")[0])
        return major
    except (ImportError, ValueError, IndexError):
        return None


def is_paddle_v3():
    """Check whether PaddlePaddle 3.x is installed.

    Always returns False - project targets PaddleOCR 2.9.1.
    Kept for backward compatibility with existing call sites.

    Returns
    -------
    bool
    """
    return False


def normalize_device(use_gpu=True):
    """Return device string appropriate for PaddleOCR 2.x.

    Always returns None because PaddleOCR 2.x uses the ``use_gpu``
    boolean parameter directly, not a device string.

    Parameters
    ----------
    use_gpu : bool
        Whether GPU should be used (not used in return value).

    Returns
    -------
    None
        Always None for PaddleOCR 2.x.
    """
    # v2 does not use device strings; callers pass use_gpu directly.
    return None


def get_ocr_engine_kwargs(use_gpu=True, lang="en", use_onnx=False, **extra):
    """Build kwargs dict for PaddleOCR() constructor (v2).

    Returns kwargs with ``use_gpu``, ``use_angle_cls``, and ``lang``
    suitable for PaddleOCR 2.9.1.  When *use_onnx* is True, adds
    ``use_onnx=True`` to enable ONNX Runtime inference.

    Parameters
    ----------
    use_gpu : bool
        Whether GPU should be used.
    lang : str
        PaddleOCR language code (e.g., ``"en"``, ``"ch"``, ``"fr"``).
    use_onnx : bool
        Whether to enable ONNX Runtime inference backend.
    **extra
        Additional keyword arguments passed through to the constructor.

    Returns
    -------
    dict
        Keyword arguments suitable for ``PaddleOCR(**kwargs)``.
    """
    kwargs = {
        "lang": lang,
        "show_log": False,
        "use_angle_cls": True,
        "use_gpu": use_gpu,
    }

    if use_onnx:
        kwargs["use_onnx"] = True

    kwargs.update(extra)
    return kwargs


def normalize_ocr_result(result):
    """Normalize PaddleOCR result format.

    PaddleOCR 2.x returns results as nested lists::

        [[[box, (text, confidence)], ...]]

    This function normalizes that format into a flat list of
    ``(box, text, confidence)`` tuples for uniform downstream
    processing.

    Parameters
    ----------
    result : list or None
        Raw PaddleOCR result from ``ocr.ocr(img)``.

    Returns
    -------
    list[tuple]
        List of ``(box, text, confidence)`` tuples.  Returns an empty
        list for ``None`` or empty input.
    """
    if not result:
        return []

    normalized = []

    if isinstance(result, list) and len(result) > 0:
        first = result[0]

        # Defensive: handles potential dict-format results (not expected in 2.x)
        if isinstance(first, dict) and "rec_texts" in first:
            texts = first.get("rec_texts", [])
            scores = first.get("rec_scores", [])
            polys = first.get("dt_polys", [])
            for i in range(len(texts)):
                box = (
                    polys[i].tolist()
                    if i < len(polys) and hasattr(polys[i], "tolist")
                    else (polys[i] if i < len(polys) else [])
                )
                text = texts[i]
                score = float(scores[i]) if i < len(scores) else 0.0
                normalized.append((box, text, score))
            return normalized

        # Standard list-of-list format (PaddleOCR 2.x)
        if isinstance(first, list):
            for line in first:
                if isinstance(line, (list, tuple)) and len(line) == 2:
                    box = line[0]
                    text_conf = line[1]
                    if isinstance(text_conf, (list, tuple)) and len(text_conf) == 2:
                        text, confidence = text_conf
                        normalized.append((box, str(text), float(confidence)))
            return normalized

    return normalized


def get_structure_engine_class():
    """Return the PPStructure class for structure analysis.

    Returns ``PPStructure`` from the ``paddleocr`` package for
    PaddleOCR 2.x layout and table analysis.

    Returns
    -------
    tuple[callable or None, str or None]
        ``(PPStructure, "ppstructure")`` if available, or
        ``(None, None)`` if the module is not installed.
    """
    try:
        from paddleocr import PPStructure

        return PPStructure, "ppstructure"
    except ImportError:
        logger.warning("PPStructure not available")
        return None, None


# Log version on import
_paddle_version = get_paddle_version()
if _paddle_version is not None:
    logger.info(
        "PaddlePaddle version %d.x detected (targeting v2 compatibility mode)",
        _paddle_version,
    )
