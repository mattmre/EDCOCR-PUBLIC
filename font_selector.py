"""Language-to-font mapping for PDF text layer embedding.

Maps the 34 supported PaddleOCR languages to Google Noto Sans font files
for proper glyph rendering in searchable PDF output. Falls back to the
default NotoSans-Regular.ttf for unmapped languages.

Font files are expected in NOTO_FONT_DIR (default: /app/fonts/noto).

Graceful degradation: if fonts are not installed, functions return None
and the pipeline uses its default font embedding behavior.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FONT_DIR = os.environ.get("NOTO_FONT_DIR", "/app/fonts/noto")

DEFAULT_FONT = "NotoSans-Regular.ttf"

# ---------------------------------------------------------------------------
# Language-to-font mapping for the expanded 34-language baseline
# ---------------------------------------------------------------------------

try:
    from ocr_local.config.language_config import FONT_MAP as _REGISTRY_FONT_MAP
    from ocr_local.config.language_config import LANG_MAPPING as _REGISTRY_LANG_MAPPING
    LANGUAGE_FONT_MAP = dict(_REGISTRY_FONT_MAP)
    _FASTTEXT_ALIASES = {k: v for k, v in _REGISTRY_LANG_MAPPING.items()
                         if k != v}  # only actual aliases
except ImportError:
    LANGUAGE_FONT_MAP = {
        "en": "NotoSans-Regular.ttf",
        "fr": "NotoSans-Regular.ttf",
        "german": "NotoSans-Regular.ttf",
        "es": "NotoSans-Regular.ttf",
        "it": "NotoSans-Regular.ttf",
        "pt": "NotoSans-Regular.ttf",
        "nl": "NotoSans-Regular.ttf",
        "sv": "NotoSans-Regular.ttf",
        "da": "NotoSans-Regular.ttf",
        "fi": "NotoSans-Regular.ttf",
        "ro": "NotoSans-Regular.ttf",
        "pl": "NotoSans-Regular.ttf",
        "cs": "NotoSans-Regular.ttf",
        "hu": "NotoSans-Regular.ttf",
        "tr": "NotoSans-Regular.ttf",
        "ru": "NotoSans-Regular.ttf",
        "uk": "NotoSans-Regular.ttf",
        "be": "NotoSans-Regular.ttf",
        "bg": "NotoSans-Regular.ttf",
        "ch": "NotoSansCJKsc-Regular.otf",
        "chinese_cht": "NotoSansCJKtc-Regular.otf",
        "japan": "NotoSansCJKjp-Regular.otf",
        "korean": "NotoSansCJKkr-Regular.otf",
        "vi": "NotoSans-Regular.ttf",
        "ar": "NotoSansArabic-Regular.ttf",
        "fa": "NotoSansArabic-Regular.ttf",
        "ur": "NotoSansArabic-Regular.ttf",
        "ug": "NotoSansArabic-Regular.ttf",
        "hi": "NotoSansDevanagari-Regular.ttf",
        "ta": "NotoSansTamil-Regular.ttf",
        "te": "NotoSansTelugu-Regular.ttf",
        "kn": "NotoSansKannada-Regular.ttf",
        "ka": "NotoSansGeorgian-Regular.ttf",
        "el": "NotoSans-Regular.ttf",
    }
    _FASTTEXT_ALIASES = {
        "zh": "ch",
        "zh-cn": "ch",
        "zh-tw": "chinese_cht",
        "ja": "japan",
        "ko": "korean",
        "de": "german",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_font_path(lang_code: str) -> Optional[Path]:
    """Get the path to the Noto Sans font file for a given language.

    Resolves FastText language codes via alias mapping, then looks up the
    PaddleOCR model name in the font map.

    Args:
        lang_code: PaddleOCR model name or FastText language code.

    Returns:
        Path to the font file if it exists on disk, None otherwise.
    """
    return _resolve_font_path(get_font_name(lang_code))


def get_font_name(lang_code: str) -> str:
    """Get the font filename for a language (without checking disk availability).

    Args:
        lang_code: PaddleOCR model name or FastText language code.

    Returns:
        Font filename string.
    """
    if not lang_code:
        return DEFAULT_FONT

    normalized = lang_code.lower().strip()
    if normalized in _FASTTEXT_ALIASES:
        normalized = _FASTTEXT_ALIASES[normalized]

    return LANGUAGE_FONT_MAP.get(normalized, DEFAULT_FONT)


def is_font_available(lang_code: str) -> bool:
    """Check if the font for a given language is installed on disk.

    Args:
        lang_code: PaddleOCR model name or FastText language code.

    Returns:
        True if the font file exists at the expected path.
    """
    path = get_font_path(lang_code)
    return path is not None


def get_available_fonts() -> dict:
    """Get a mapping of language codes to font paths for all installed fonts.

    Scans all supported languages and returns only those whose font
    files are present on disk.

    Returns:
        Dict mapping language code -> Path for available fonts.
    """
    available = {}
    font_path_cache = {}

    for lang_code, font_name in LANGUAGE_FONT_MAP.items():
        if font_name not in font_path_cache:
            font_path_cache[font_name] = _resolve_font_path(font_name)
        path = font_path_cache[font_name]
        if path is not None:
            available[lang_code] = path
    return available


def get_font_dir() -> Path:
    """Get the configured font directory path.

    Returns:
        Path object for the font directory.
    """
    return Path(FONT_DIR)


def get_all_unique_fonts() -> list:
    """Get a deduplicated list of all font filenames needed for full coverage.

    Useful for Docker image builders to know which fonts to download.

    Returns:
        Sorted list of unique font filenames.
    """
    fonts = set(LANGUAGE_FONT_MAP.values())
    fonts.add(DEFAULT_FONT)
    return sorted(fonts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_font_path(font_name: str) -> Optional[Path]:
    """Resolve a font filename to its full path, checking disk availability.

    Args:
        font_name: Font filename (e.g., "NotoSans-Regular.ttf").

    Returns:
        Path if the file exists, None otherwise.
    """
    font_path = Path(FONT_DIR) / font_name
    if font_path.is_file():
        return font_path
    return None
