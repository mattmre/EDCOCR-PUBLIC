"""Consolidated language registry for EDCOCR.

Single source of truth for all per-language metadata.
Replaces duplicated LANG_MAPPING definitions across multiple files.

Tier selection is controlled by the ``OCR_LANGUAGE_TIERS`` environment variable:
- ``"core"`` (default): The 34-language production baseline
- ``"core,extended"``: Adds 11 additional languages
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Tier selection: which language tiers to enable
# Options: "core", "extended" (comma-separated)
OCR_LANGUAGE_TIERS = os.environ.get("OCR_LANGUAGE_TIERS", "core").lower().split(",")


@dataclass(frozen=True)
class LanguageEntry:
    """Configuration for a single supported language."""

    paddle_code: str          # PaddleOCR model name (e.g., "ch", "en", "ar")
    name: str                 # Human-readable name
    fasttext_codes: tuple     # FastText lid.176.bin codes that map to this language
    script: str               # Script family: "latin", "cyrillic", "cjk", "arabic", etc.
    tier: str                 # "core" or "extended"
    tesseract_code: str = ""  # Tesseract language code (e.g., "eng", "fra", "ara")
    easyocr_code: str = ""    # EasyOCR language code
    font: str = "NotoSans-Regular.ttf"  # Default font file
    rtl: bool = False         # Right-to-left script
    bcp47: str = ""           # BCP-47 language tag (e.g., "en", "zh-Hans", "sr-Latn")


# --- THE REGISTRY ---------------------------------------------------------------

LANGUAGE_REGISTRY: dict[str, LanguageEntry] = {}


def _reg(entry: LanguageEntry):
    LANGUAGE_REGISTRY[entry.paddle_code] = entry


# -- Core tier (34 languages: current production baseline) -----------------------

# CJK
_reg(LanguageEntry("ch", "Chinese Simplified", ("zh",), "cjk", "core",
                   "chi_sim", "ch_sim", "NotoSansCJKsc-Regular.otf",
                   bcp47="zh-Hans"))
_reg(LanguageEntry("chinese_cht", "Chinese Traditional", ("zh-tw",), "cjk", "core",
                   "chi_tra", "ch_tra", "NotoSansCJKtc-Regular.otf",
                   bcp47="zh-Hant"))
_reg(LanguageEntry("japan", "Japanese", ("ja",), "cjk", "core",
                   "jpn", "ja", "NotoSansCJKjp-Regular.otf",
                   bcp47="ja"))
_reg(LanguageEntry("korean", "Korean", ("ko",), "cjk", "core",
                   "kor", "ko", "NotoSansCJKkr-Regular.otf",
                   bcp47="ko"))

# Latin scripts
_reg(LanguageEntry("en", "English", ("en",), "latin", "core", "eng", "en", bcp47="en"))
_reg(LanguageEntry("fr", "French", ("fr",), "latin", "core", "fra", "fr", bcp47="fr"))
_reg(LanguageEntry("german", "German", ("de",), "latin", "core", "deu", "de", bcp47="de"))
_reg(LanguageEntry("es", "Spanish", ("es",), "latin", "core", "spa", "es", bcp47="es"))
_reg(LanguageEntry("it", "Italian", ("it",), "latin", "core", "ita", "it", bcp47="it"))
_reg(LanguageEntry("pt", "Portuguese", ("pt",), "latin", "core", "por", "pt", bcp47="pt"))
_reg(LanguageEntry("nl", "Dutch", ("nl",), "latin", "core", "nld", "nl", bcp47="nl"))
_reg(LanguageEntry("sv", "Swedish", ("sv",), "latin", "core", "swe", "sv", bcp47="sv"))
_reg(LanguageEntry("da", "Danish", ("da",), "latin", "core", "dan", "da", bcp47="da"))
_reg(LanguageEntry("fi", "Finnish", ("fi",), "latin", "core", "fin", "fi", bcp47="fi"))
_reg(LanguageEntry("ro", "Romanian", ("ro",), "latin", "core", "ron", "ro", bcp47="ro"))
_reg(LanguageEntry("pl", "Polish", ("pl",), "latin", "core", "pol", "pl", bcp47="pl"))
_reg(LanguageEntry("cs", "Czech", ("cs",), "latin", "core", "ces", "cs", bcp47="cs"))
_reg(LanguageEntry("hu", "Hungarian", ("hu",), "latin", "core", "hun", "hu", bcp47="hu"))
_reg(LanguageEntry("tr", "Turkish", ("tr",), "latin", "core", "tur", "tr", bcp47="tr"))
_reg(LanguageEntry("vi", "Vietnamese", ("vi",), "latin", "core", "vie", "vi", bcp47="vi"))

# Cyrillic
_reg(LanguageEntry("ru", "Russian", ("ru",), "cyrillic", "core", "rus", "ru", bcp47="ru"))
_reg(LanguageEntry("uk", "Ukrainian", ("uk",), "cyrillic", "core", "ukr", "uk", bcp47="uk"))
_reg(LanguageEntry("be", "Belarusian", ("be",), "cyrillic", "core", "bel", "be", bcp47="be"))
_reg(LanguageEntry("bg", "Bulgarian", ("bg",), "cyrillic", "core", "bul", "bg", bcp47="bg"))

# Arabic / RTL
_reg(LanguageEntry("ar", "Arabic", ("ar",), "arabic", "core",
                   "ara", "ar", "NotoSansArabic-Regular.ttf", True, bcp47="ar"))
_reg(LanguageEntry("fa", "Persian", ("fa",), "arabic", "core",
                   "fas", "fa", "NotoSansArabic-Regular.ttf", True, bcp47="fa"))
_reg(LanguageEntry("ur", "Urdu", ("ur",), "arabic", "core",
                   "urd", "ur", "NotoSansArabic-Regular.ttf", True, bcp47="ur"))
_reg(LanguageEntry("ug", "Uyghur", ("ug",), "arabic", "core",
                   "uig", "ug", "NotoSansArabic-Regular.ttf", True, bcp47="ug"))

# Indic / Devanagari
_reg(LanguageEntry("hi", "Hindi", ("hi",), "devanagari", "core",
                   "hin", "hi", "NotoSansDevanagari-Regular.ttf", bcp47="hi"))
_reg(LanguageEntry("ta", "Tamil", ("ta",), "tamil", "core",
                   "tam", "ta", "NotoSansTamil-Regular.ttf", bcp47="ta"))
_reg(LanguageEntry("te", "Telugu", ("te",), "telugu", "core",
                   "tel", "te", "NotoSansTelugu-Regular.ttf", bcp47="te"))
_reg(LanguageEntry("kn", "Kannada", ("kn",), "kannada", "core",
                   "kan", "kn", "NotoSansKannada-Regular.ttf", bcp47="kn"))
_reg(LanguageEntry("ka", "Georgian", ("ka",), "georgian", "core",
                   "kat", "ka", "NotoSansGeorgian-Regular.ttf", bcp47="ka"))

# Greek
_reg(LanguageEntry("el", "Greek", ("el",), "greek", "core", "ell", "el", bcp47="el"))

# -- Extended tier (+11 languages) ------------------------------------------------

_reg(LanguageEntry("th", "Thai", ("th",), "thai", "extended",
                   "tha", "th", "NotoSansThai-Regular.ttf", bcp47="th"))
_reg(LanguageEntry("bn", "Bengali", ("bn",), "bengali", "extended",
                   "ben", "bn", "NotoSansBengali-Regular.ttf", bcp47="bn"))
_reg(LanguageEntry("mr", "Marathi", ("mr",), "devanagari", "extended",
                   "mar", "mr", "NotoSansDevanagari-Regular.ttf", bcp47="mr"))
_reg(LanguageEntry("ne", "Nepali", ("ne",), "devanagari", "extended",
                   "nep", "ne", "NotoSansDevanagari-Regular.ttf", bcp47="ne"))
_reg(LanguageEntry("hr", "Croatian", ("hr",), "latin", "extended", "hrv", "hr", bcp47="hr"))
_reg(LanguageEntry("sk", "Slovak", ("sk",), "latin", "extended", "slk", "sk", bcp47="sk"))
_reg(LanguageEntry("no", "Norwegian", ("no",), "latin", "extended", "nor", "no", bcp47="no"))
_reg(LanguageEntry("lt", "Lithuanian", ("lt",), "latin", "extended", "lit", "lt", bcp47="lt"))
_reg(LanguageEntry("lv", "Latvian", ("lv",), "latin", "extended", "lav", "lv", bcp47="lv"))
_reg(LanguageEntry("et", "Estonian", ("et",), "latin", "extended", "est", "et", bcp47="et"))
_reg(LanguageEntry("rs_latin", "Serbian Latin", ("sr",), "latin", "extended",
                   "srp_latn", "rs_latin", bcp47="sr-Latn"))


# --- Derived maps (backward compatible) -----------------------------------------

def get_enabled_languages() -> dict[str, LanguageEntry]:
    """Return language entries for all enabled tiers."""
    tiers = {t.strip() for t in OCR_LANGUAGE_TIERS}
    return {k: v for k, v in LANGUAGE_REGISTRY.items() if v.tier in tiers}


def build_lang_mapping() -> dict[str, str]:
    """Build FastText-code -> PaddleOCR-code mapping from registry.

    Includes the ``zh-cn`` alias for backward compatibility.
    """
    mapping: dict[str, str] = {}
    for entry in get_enabled_languages().values():
        for ft_code in entry.fasttext_codes:
            mapping[ft_code] = entry.paddle_code
    # Backward-compatible aliases
    mapping.setdefault("zh-cn", mapping.get("zh", "ch"))
    mapping.setdefault("zh-tw", mapping.get("zh-tw", "chinese_cht"))
    return mapping


def build_target_langs() -> list[str]:
    """Build list of PaddleOCR model codes to download."""
    return sorted(get_enabled_languages().keys())


def build_font_map() -> dict[str, str]:
    """Build PaddleOCR-code -> font-file mapping from registry."""
    return {k: v.font for k, v in get_enabled_languages().items()}


def build_tesseract_map() -> dict[str, str]:
    """Build PaddleOCR-code -> Tesseract-code mapping from registry."""
    return {k: v.tesseract_code for k, v in get_enabled_languages().items()
            if v.tesseract_code}


def build_easyocr_map() -> dict[str, str]:
    """Build PaddleOCR-code -> EasyOCR-code mapping from registry."""
    return {k: v.easyocr_code for k, v in get_enabled_languages().items()
            if v.easyocr_code}


def build_rtl_languages() -> set[str]:
    """Build set of PaddleOCR codes that are RTL."""
    return {k for k, v in get_enabled_languages().items() if v.rtl}


# Eagerly built for backward compatibility -- these module-level constants
# match the shape consumers already import from ocr_distributed.constants.
LANG_MAPPING = build_lang_mapping()
TARGET_LANGS = build_target_langs()
FONT_MAP = build_font_map()
TESSERACT_MAP = build_tesseract_map()
EASYOCR_MAP = build_easyocr_map()
RTL_LANGUAGES = build_rtl_languages()


# --- Convenience helpers --------------------------------------------------------

def get_paddle_code(fasttext_code: str) -> str:
    """Map a FastText language code to PaddleOCR model code.

    Falls back to ``'en'`` with a warning for unmapped languages.
    """
    code = fasttext_code.lower().strip()
    result = LANG_MAPPING.get(code)
    if result is None:
        logger.warning(
            "Unmapped language '%s' detected by FastText; falling back to 'en'",
            code,
        )
        return "en"
    return result


def get_tesseract_code(paddle_code: str) -> str:
    """Map a PaddleOCR code to Tesseract language code.  Falls back to ``'eng'``."""
    return TESSERACT_MAP.get(paddle_code, "eng")


def get_font(paddle_code: str) -> str:
    """Get the font file for a language.  Falls back to ``NotoSans-Regular.ttf``."""
    return FONT_MAP.get(paddle_code, "NotoSans-Regular.ttf")


def is_rtl(paddle_code: str) -> bool:
    """Check if a language uses right-to-left script."""
    return paddle_code in RTL_LANGUAGES


def get_supported_language_count() -> int:
    """Return the number of currently enabled languages."""
    return len(get_enabled_languages())


def get_tier_summary() -> dict[str, int]:
    """Return counts by tier for the full registry (all tiers, not just enabled)."""
    summary: dict[str, int] = {}
    for entry in LANGUAGE_REGISTRY.values():
        summary[entry.tier] = summary.get(entry.tier, 0) + 1
    return summary
