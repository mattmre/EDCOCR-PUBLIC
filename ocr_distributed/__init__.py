"""Shared OCR utilities for monolithic and distributed pipelines.

This package extracts pure utility functions from ocr_gpu_async.py so they can
be reused by both the monolithic pipeline and Celery distributed tasks without
code duplication.

Public convenience attributes are loaded lazily.  Importing a narrow helper
such as :mod:`ocr_distributed.tenant_purge_fs` must not pull the complete OCR
runtime (video/OpenCV, language detection, and PDF helpers) into lightweight
coordinator processes during Celery task autodiscovery.

Phase A additions (transform-stamping-support):
- Transform operations contract layer and registry
- Stamp operations contract layer and registry
"""

from importlib import import_module

_LAZY_EXPORTS = {
    # Constants
    "DPI_DEFAULT": (".constants", "DPI_DEFAULT"),
    "LANG_MAPPING": (".constants", "LANG_MAPPING"),
    "PDF_EXTENSIONS": (".constants", "PDF_EXTENSIONS"),
    "PHASE1_IMAGE_EXTENSIONS": (".constants", "PHASE1_IMAGE_EXTENSIONS"),
    "PHASE2_IMAGE_EXTENSIONS": (".constants", "PHASE2_IMAGE_EXTENSIONS"),
    "PRIVILEGE_HEURISTIC_CONFIDENCE": (
        ".constants",
        "PRIVILEGE_HEURISTIC_CONFIDENCE",
    ),
    "PRIVILEGE_KEYWORDS": (".constants", "PRIVILEGE_KEYWORDS"),
    "VIDEO_EXTENSIONS": (".constants", "VIDEO_EXTENSIONS"),
    # Language detection
    "LanguageDetector": (".language", "LanguageDetector"),
    # OCR utilities
    "box_to_rect_and_anchor": (".ocr_utils", "box_to_rect_and_anchor"),
    "build_output_rel_stem": (".ocr_utils", "build_output_rel_stem"),
    "build_sidecar_base_name": (".ocr_utils", "build_sidecar_base_name"),
    "classify_source_file": (".ocr_utils", "classify_source_file"),
    "create_paddle_engine": (".ocr_utils", "create_paddle_engine"),
    "detect_magic_family": (".ocr_utils", "detect_magic_family"),
    "encode_path_segment": (".ocr_utils", "encode_path_segment"),
    "extract_paddle_lines": (".ocr_utils", "extract_paddle_lines"),
    "get_env_float": (".ocr_utils", "get_env_float"),
    "get_env_int": (".ocr_utils", "get_env_int"),
    "get_file_hash": (".ocr_utils", "get_file_hash"),
    "get_source_page_count": (".ocr_utils", "get_source_page_count"),
    "img_to_bytes": (".ocr_utils", "img_to_bytes"),
    "insert_text_line": (".ocr_utils", "insert_text_line"),
    "iter_fitz_image_frames": (".ocr_utils", "iter_fitz_image_frames"),
    "iter_pil_image_frames": (".ocr_utils", "iter_pil_image_frames"),
    "iter_source_images": (".ocr_utils", "iter_source_images"),
    "read_file_header": (".ocr_utils", "read_file_header"),
    "resolve_safe_output_subfolder": (".ocr_utils", "resolve_safe_output_subfolder"),
    "sanitize_path_segment": (".ocr_utils", "sanitize_path_segment"),
    "to_plain_list": (".ocr_utils", "to_plain_list"),
    "write_json_atomic": (".ocr_utils", "write_json_atomic"),
    "write_text_atomic": (".ocr_utils", "write_text_atomic"),
    # SSRF protection
    "LOOPBACK_NAMES": (".ssrf", "LOOPBACK_NAMES"),
    "NoRedirectHandler": (".ssrf", "NoRedirectHandler"),
    "is_private_ip": (".ssrf", "is_private_ip"),
    "safe_opener": (".ssrf", "safe_opener"),
    "validate_webhook_url": (".ssrf", "validate_webhook_url"),
    # Text comparison metrics
    "accuracy_summary": (".text_metrics", "accuracy_summary"),
    "character_error_rate": (".text_metrics", "character_error_rate"),
    "edit_distance": (".text_metrics", "edit_distance"),
    "line_accuracy": (".text_metrics", "line_accuracy"),
    "word_error_rate": (".text_metrics", "word_error_rate"),
}


def __getattr__(name):
    """Resolve legacy package-level exports without eager OCR imports."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


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
    # Environment-variable parsing (maintenance review)
    "get_env_int",
    "get_env_float",
    # OCR utilities
    "box_to_rect_and_anchor",
    "build_output_rel_stem",
    "build_sidecar_base_name",
    "classify_source_file",
    "create_paddle_engine",
    "detect_magic_family",
    "encode_path_segment",
    "extract_paddle_lines",
    "get_file_hash",
    "get_source_page_count",
    "img_to_bytes",
    "insert_text_line",
    "iter_fitz_image_frames",
    "iter_pil_image_frames",
    "iter_source_images",
    "read_file_header",
    "resolve_safe_output_subfolder",
    "sanitize_path_segment",
    "to_plain_list",
    "write_json_atomic",
    "write_text_atomic",
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
