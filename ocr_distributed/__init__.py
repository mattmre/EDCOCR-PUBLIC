"""ocr_distributed - Shared OCR utilities for monolithic and distributed pipelines.

This package extracts pure utility functions from ocr_gpu_async.py so they can
be reused by both the monolithic pipeline and Celery distributed tasks without
code duplication.

Phase A additions (transform-stamping-support):
- Transform operations contract layer and registry
- Stamp operations contract layer and registry
"""

from .constants import (
    DPI_DEFAULT,
    LANG_MAPPING,
    PDF_EXTENSIONS,
    PHASE1_IMAGE_EXTENSIONS,
    PHASE2_IMAGE_EXTENSIONS,
    PRIVILEGE_HEURISTIC_CONFIDENCE,
    PRIVILEGE_KEYWORDS,
    VIDEO_EXTENSIONS,
)
from .language import LanguageDetector
from .ocr_utils import (
    box_to_rect_and_anchor,
    build_output_rel_stem,
    classify_source_file,
    create_paddle_engine,
    detect_magic_family,
    extract_paddle_lines,
    get_env_float,
    get_env_int,
    get_file_hash,
    get_source_page_count,
    img_to_bytes,
    insert_text_line,
    iter_fitz_image_frames,
    iter_pil_image_frames,
    iter_source_images,
    read_file_header,
    sanitize_path_segment,
    to_plain_list,
)
from .ssrf import (
    LOOPBACK_NAMES,
    NoRedirectHandler,
    is_private_ip,
    safe_opener,
    validate_webhook_url,
)
from .text_metrics import (
    accuracy_summary,
    character_error_rate,
    edit_distance,
    line_accuracy,
    word_error_rate,
)


# Transform and stamp registries (lazy imports to avoid hard dependencies)
def get_transform_registry():
    """Get the global transform registry singleton."""
    from .transforms import get_transform_registry as _get_registry
    return _get_registry()


def get_stamp_registry():
    """Get the global stamp registry singleton."""
    from .stamps import get_stamp_registry as _get_registry
    return _get_registry()

__all__ = [
    # Constants
    "DPI_DEFAULT",
    "LANG_MAPPING",
    "PDF_EXTENSIONS",
    "PHASE1_IMAGE_EXTENSIONS",
    "PHASE2_IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "PRIVILEGE_HEURISTIC_CONFIDENCE",
    "PRIVILEGE_KEYWORDS",
    # Language detection
    "LanguageDetector",
    # SSRF protection
    "LOOPBACK_NAMES",
    "NoRedirectHandler",
    "is_private_ip",
    "safe_opener",
    "validate_webhook_url",
    # Environment-variable parsing
    "get_env_int",
    "get_env_float",
    # OCR utilities
    "box_to_rect_and_anchor",
    "build_output_rel_stem",
    "classify_source_file",
    "create_paddle_engine",
    "detect_magic_family",
    "extract_paddle_lines",
    "get_file_hash",
    "get_source_page_count",
    "img_to_bytes",
    "insert_text_line",
    "iter_fitz_image_frames",
    "iter_pil_image_frames",
    "iter_source_images",
    "read_file_header",
    "sanitize_path_segment",
    "to_plain_list",
    # Transform and stamp registries (Phase A)
    "get_transform_registry",
    "get_stamp_registry",
    # Text comparison metrics
    "accuracy_summary",
    "character_error_rate",
    "edit_distance",
    "line_accuracy",
    "word_error_rate",
]
