"""Unicode normalization and RTL text support for forensic OCR pipeline.

Provides consistent text normalization (NFC), right-to-left script detection,
and bidirectional text reordering for the 27 supported languages.

All functions degrade gracefully if optional dependencies (python-bidi) are
not installed -- the raw text is returned unchanged.
"""

import logging
import statistics
import unicodedata

logger = logging.getLogger(__name__)
get_display = None

# ---------------------------------------------------------------------------
# RTL language codes (PaddleOCR model names used in this project)
# ---------------------------------------------------------------------------

RTL_LANGUAGES = {"ar", "he", "fa", "ur"}
CJK_VERTICAL_LANGUAGES = {"japan", "ch", "chinese_cht"}

# Bidirectional category values that indicate RTL characters
_RTL_BIDI_CATEGORIES = {"R", "AL", "AN"}

# ---------------------------------------------------------------------------
# Guarded python-bidi import
# ---------------------------------------------------------------------------

_BIDI_AVAILABLE = False

try:
    from bidi.algorithm import get_display as _get_display

    get_display = _get_display
    _BIDI_AVAILABLE = True
except ImportError:
    pass


_CJK_RANGES = (
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
)


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------


def normalize_nfc(text: str) -> str:
    """Normalize Unicode text to NFC (Canonical Decomposition, then Composition).

    NFC ensures consistent byte representation for identical characters,
    which is critical for text search, deduplication, and forensic comparison.

    Examples:
        - Precomposed 'e\\u0301' (e + combining accent) -> '\\u00e9' (single char)
        - Korean Jamo sequences -> precomposed Hangul syllables
        - Already-NFC text is returned unchanged (fast path)

    Args:
        text: Input Unicode string.

    Returns:
        NFC-normalized string. Empty string if input is empty or None.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def normalize_nfkc(text: str) -> str:
    """Normalize Unicode text to NFKC (Compatibility Decomposition, then Composition).

    NFKC additionally normalizes compatibility characters (e.g., fullwidth forms,
    ligatures, superscripts) to their standard equivalents. More aggressive than NFC.

    Args:
        text: Input Unicode string.

    Returns:
        NFKC-normalized string.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def is_cjk_vertical_language(lang_code: str) -> bool:
    """Check if the OCR model can reasonably emit vertical CJK layouts."""
    if not lang_code:
        return False
    return lang_code.lower().strip() in CJK_VERTICAL_LANGUAGES


def has_cjk_chars(text: str) -> bool:
    """Return True when text contains Japanese/Chinese ideographic characters."""
    if not text:
        return False
    for char in text:
        codepoint = ord(char)
        for start, end in _CJK_RANGES:
            if start <= codepoint <= end:
                return True
    return False


# ---------------------------------------------------------------------------
# RTL detection
# ---------------------------------------------------------------------------


def is_rtl_language(lang_code: str) -> bool:
    """Check if a language code corresponds to a right-to-left script.

    Supports both FastText codes (e.g., 'ar') and PaddleOCR model names.

    Args:
        lang_code: Language code string.

    Returns:
        True if the language uses an RTL script.
    """
    if not lang_code:
        return False
    return lang_code.lower().strip() in RTL_LANGUAGES


def has_rtl_chars(text: str) -> bool:
    """Check if text contains any right-to-left characters.

    Inspects Unicode bidirectional category of each character.
    Categories R (Right-to-Left), AL (Arabic Letter), and AN (Arabic Number)
    are treated as RTL indicators.

    Args:
        text: Input text to check.

    Returns:
        True if at least one RTL character is found.
    """
    if not text:
        return False
    for char in text:
        if unicodedata.bidirectional(char) in _RTL_BIDI_CATEGORIES:
            return True
    return False


def get_rtl_ratio(text: str) -> float:
    """Compute the ratio of RTL characters to total alphabetic characters.

    Useful for determining if a mixed-script text is predominantly RTL.

    Args:
        text: Input text.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 for empty text.
    """
    if not text:
        return 0.0

    alpha_chars = [char for char in text if char.isalpha()]
    if not alpha_chars:
        return 0.0

    rtl_count = sum(
        1
        for char in alpha_chars
        if unicodedata.bidirectional(char) in _RTL_BIDI_CATEGORIES
    )
    return round(rtl_count / len(alpha_chars), 4)


# ---------------------------------------------------------------------------
# RTL text reordering
# ---------------------------------------------------------------------------


def reorder_rtl_text(text: str) -> str:
    """Apply logical-to-visual reordering for RTL text.

    Uses the Unicode Bidirectional Algorithm (python-bidi) to convert logical
    order (used in storage) to visual order (used for display). This is
    necessary when embedding text in PDFs that do not support bidi natively.

    Graceful degradation: if python-bidi is not installed, returns text unchanged.

    Args:
        text: Input text in logical order.

    Returns:
        Text in visual order (or unchanged if python-bidi is unavailable).
    """
    if not text:
        return ""
    if not _BIDI_AVAILABLE:
        logger.debug(
            "python-bidi not available; RTL text returned in logical order. "
            "Install python-bidi for visual reordering."
        )
        return text
    try:
        return get_display(text)
    except Exception as e:
        logger.warning("RTL reordering failed: %s", e)
        return text


def _line_bbox(line) -> tuple[float, float, float, float] | None:
    _, box, _ = line
    if not box:
        return None
    first = box[0]
    if isinstance(first, (list, tuple)):
        xs = [float(point[0]) for point in box if len(point) >= 2]
        ys = [float(point[1]) for point in box if len(point) >= 2]
        if not xs or not ys:
            return None
        return min(xs), min(ys), max(xs), max(ys)
    if len(box) >= 4:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return None


def reorder_cjk_vertical_lines(paddle_lines: list, lang_code: str) -> list:
    """Reorder clearly vertical Japanese/Chinese columns right-to-left.

    The function is intentionally conservative:
    - only enabled for CJK vertical-capable language models
    - requires at least two vertical-looking CJK lines
    - only reorders when vertical-looking lines are the clear majority
    """
    if not paddle_lines or not is_cjk_vertical_language(lang_code):
        return paddle_lines

    enriched = []
    for idx, line in enumerate(paddle_lines):
        text, _, _ = line
        bbox = _line_bbox(line)
        if bbox is None:
            return paddle_lines
        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        is_vertical = (
            has_cjk_chars(str(text))
            and height / width >= 1.35
            and len(str(text).strip()) <= 6
        )
        enriched.append(
            {
                "idx": idx,
                "line": line,
                "bbox": bbox,
                "x_center": (x1 + x2) / 2.0,
                "y_top": y1,
                "width": width,
                "vertical": is_vertical,
            }
        )

    vertical_lines = [item for item in enriched if item["vertical"]]
    if len(vertical_lines) < 2:
        return paddle_lines
    if len(vertical_lines) / len(enriched) < 0.6:
        return paddle_lines

    width_samples = [item["width"] for item in vertical_lines]
    tolerance = max(statistics.median(width_samples) * 1.5, 10.0)
    columns = []
    for item in sorted(vertical_lines, key=lambda entry: entry["x_center"], reverse=True):
        placed = False
        for column in columns:
            if abs(item["x_center"] - column["x_center"]) <= tolerance:
                column["items"].append(item)
                column["x_values"].append(item["x_center"])
                column["x_center"] = sum(column["x_values"]) / len(column["x_values"])
                placed = True
                break
        if not placed:
            columns.append(
                {
                    "x_center": item["x_center"],
                    "x_values": [item["x_center"]],
                    "items": [item],
                }
            )

    ordered_vertical = []
    for column in sorted(columns, key=lambda entry: entry["x_center"], reverse=True):
        for item in sorted(column["items"], key=lambda entry: entry["y_top"]):
            ordered_vertical.append(item["line"])

    ordered = []
    vertical_idx = 0
    for item in sorted(enriched, key=lambda entry: entry["idx"]):
        if item["vertical"]:
            ordered.append(ordered_vertical[vertical_idx])
            vertical_idx += 1
        else:
            ordered.append(item["line"])

    return ordered


# ---------------------------------------------------------------------------
# Pipeline integration helpers
# ---------------------------------------------------------------------------


def normalize_ocr_text(text: str, lang_code: str = "") -> str:
    """Normalize OCR output text for consistent storage.

    Applies NFC normalization to all text. For RTL languages, additional
    processing may be applied in the future.

    Args:
        text: Raw OCR output text.
        lang_code: Detected language code (optional).

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    # Always apply NFC normalization
    normalized = normalize_nfc(text)

    return normalized


def bidi_available() -> bool:
    """Check if python-bidi is installed and available.

    Returns:
        True if python-bidi can perform RTL reordering.
    """
    return _BIDI_AVAILABLE
