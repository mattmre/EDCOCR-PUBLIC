"""Shared constants extracted from ocr_gpu_async.py for distributed use.

These constants are used by both the monolithic pipeline (ocr_gpu_async.py)
and distributed Celery tasks.
"""

# Default OCR resolution
DPI_DEFAULT = 300

# --- Source Format Policy (Phase 1 + Planned Phase 2) ---

PDF_EXTENSIONS = {".pdf"}
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".m4v", ".mpg", ".mpeg", ".mkv", ".webm",
}

PHASE1_IMAGE_EXTENSIONS = {
    ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".gif",
    ".webp", ".jp2", ".jpx", ".pnm", ".pbm", ".pgm", ".ppm",
    ".pcx", ".ico", ".svg", ".svgz", ".wmf", ".emf",
}

PHASE2_IMAGE_EXTENSIONS = {".heic", ".heif", ".avif", ".jxl", ".jxr", ".dcx", ".xps"}

# --- Language Mapping (FastText code -> PaddleOCR model name) ---
# Prefer the consolidated registry when available; keep inline fallback
# so distributed workers that don't ship language_config.py still work.

try:
    from ocr_local.config.language_config import LANG_MAPPING
except ImportError:
    LANG_MAPPING = {
        # CJK (Critical)
        'zh': 'ch', 'zh-cn': 'ch', 'zh-tw': 'chinese_cht',
        'ja': 'japan', 'ko': 'korean', 'vi': 'vi',

        # European
        'en': 'en', 'fr': 'fr', 'de': 'german', 'es': 'es', 'it': 'it',
        'pt': 'pt', 'ru': 'ru', 'ar': 'ar', 'hi': 'hi',
        'uk': 'uk', 'be': 'be', 'bg': 'bg', 'cs': 'cs', 'pl': 'pl',
        'tr': 'tr', 'nl': 'nl', 'sv': 'sv', 'da': 'da',
        'fi': 'fi', 'el': 'el', 'hu': 'hu', 'ro': 'ro',
        'fa': 'fa', 'ur': 'ur', 'ug': 'ug',
        'te': 'te', 'ta': 'ta', 'kn': 'kn', 'ka': 'ka',
    }

# --- Privilege Detection (Phase 3C) ---

PRIVILEGE_KEYWORDS = frozenset({
    "attorney-client",
    "attorney client",
    "work product",
    "privileged and confidential",
    "privileged & confidential",
    "attorney work product",
    "do not disclose",
    "legally privileged",
})

PRIVILEGE_HEURISTIC_CONFIDENCE = 0.85
