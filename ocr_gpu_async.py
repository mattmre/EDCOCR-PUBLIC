import contextlib
import csv
import datetime
import enum
import hashlib
import io
import json
import logging
import math
import mmap
import os
import queue
import re
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# PageStatus enum — type-safe replacement for scattered status string literals
# (maintenance review).  Inherits from ``str`` so comparisons with plain strings remain
# backward-compatible (``PageStatus.COMPLETE == "COMPLETE"`` is True).
# Dynamic statuses such as ``f"Paddle-{lang}"`` are intentionally left as
# raw strings because they cannot be enumerated ahead of time.
# ---------------------------------------------------------------------------


class PageStatus(str, enum.Enum):
    """Pipeline page-level status values.

    Used in assembly-queue messages, ``DocumentState.terminal_statuses``,
    and the assembler/validation status-mapping logic.
    """

    OK = "OK"
    PADDLE = "Paddle"
    TESSERACT = "Tesseract"
    EASYOCR = "EasyOCR"
    IMAGE_ONLY = "ImageOnly"
    CRITICAL_FAILED = "CRITICAL_FAILED"
    EXTRACT_FAILED = "EXTRACT_FAILED"
    RESUMED = "RESUMED"
    CACHED = "CACHED"
    SKIPPED = "SKIPPED"
    UNKNOWN = "UNKNOWN"


import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image, ImageFilter, ImageStat

from ocr_distributed.ocr_utils import (
    build_output_rel_stem as _canonical_build_output_rel_stem,
)
from ocr_distributed.ocr_utils import (
    build_sidecar_base_name as _canonical_build_sidecar_base_name,
)
from ocr_distributed.ocr_utils import (
    get_env_int as _get_env_int,
)
from ocr_distributed.ocr_utils import (
    resolve_safe_output_subfolder,
    write_json_atomic,
    write_text_atomic,
)
from ocr_local.config.language_config import LANGUAGE_REGISTRY, get_paddle_runtime_code
from ocr_local.document_fence import DocumentCompletionFence
from ocr_local.engines.runtime_pool import (
    EngineKey,
    EngineRuntimePool,
    production_native_call_timeout,
)
from ocr_local.features.fasttext_adapter import load_model as load_fasttext_model
from ocr_local.features.fasttext_adapter import predict_language
from ocr_local.infra.ocr_inference_backend import (
    OCR_DROP_SCORE,
    get_detection_limit_side_len,
    get_gpu_device_configuration,
    get_native_threads_per_predictor,
    get_recognizer_batch_size,
)
from ocr_local.infra.ocr_inference_backend import (
    get_backend as get_ocr_backend,
)
from ocr_local.language_routing import (
    OCRCandidate,
    compare_cross_engine_candidates,
    compare_grounded_word_candidates,
    compare_ocr_candidates,
    probe_image_script_candidates,
    route_language,
    text_supports_script,
)
from ocr_local.ocr_overlay import (
    bounded_source_raster_size,
    compare_source_grounded_text_revision,
    insert_hidden_text_in_box,
    map_half_turn_detector_lines_to_source,
    map_half_turn_words_to_source,
    map_rotated_words_to_source,
    probe_half_turn_direction,
    read_source_tesseract_words,
    tesseract_data_to_lines_with_review,
)
from ocr_local.pdf_geometry import PageGeometry
from ocr_local.raster_budget import RasterByteBudget
from ocr_local.raster_source import white_composite_for_ocr
from ocr_local.visual_coverage import (
    assess_candidate_coverage,
    assess_raster_text_regions,
)
from optimize_pdfs import (  # Direct Import
    OptimizationDisposition,
    optimize_pdf_detailed,
)
from pipeline_config import PipelineConfig, create_pipeline_config  # maintenance review
from video_utils import get_video_page_count, iter_video_frames

_DISABLE_PADDLEOCR = os.environ.get("OCR_LOCAL_DISABLE_PADDLEOCR", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

try:
    if _DISABLE_PADDLEOCR:
        raise ImportError("PaddleOCR disabled by OCR_LOCAL_DISABLE_PADDLEOCR")
    from paddleocr import PaddleOCR

    _PADDLEOCR_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    _PADDLEOCR_AVAILABLE = False

try:
    from pdf2image import convert_from_path

    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    convert_from_path = None
    _PDF2IMAGE_AVAILABLE = False

try:
    from ocr_local.config.version import __version__ as PIPELINE_VERSION
except Exception as exc:
    logging.debug("Failed to import PIPELINE_VERSION, using fallback: %s", exc)
    PIPELINE_VERSION = "0.0.0"

try:
    from ocr_local.features.custody import CustodyChain

    _CUSTODY_AVAILABLE = True
except ImportError:
    _CUSTODY_AVAILABLE = False

try:
    from validation import (
        DocumentValidation,
        finalize_validation,
        write_validation_json,
    )
    from validation import (
        compute_file_hash as compute_output_hash,
    )

    _VALIDATION_AVAILABLE = True
except ImportError:
    _VALIDATION_AVAILABLE = False

try:
    from dpi_escalation import (
        DPI_SCHEDULE,
        re_extract_page_at_dpi,
        should_escalate,
    )

    _DPI_ESCALATION_AVAILABLE = True
except ImportError:
    _DPI_ESCALATION_AVAILABLE = False

try:
    from ner import (
        DocumentNER,
        extract_custom_entities,
        extract_entities,
        finalize_ner,
        write_ner_json,
    )

    _NER_AVAILABLE = True
except ImportError:
    _NER_AVAILABLE = False

try:
    from handwriting import (
        DocumentHandwriting,
        detect_handwriting_by_confidence,
        detect_handwriting_by_geometry,
        finalize_handwriting,
        merge_handwriting_signals,
        write_handwriting_json,
    )

    _HANDWRITING_AVAILABLE = True
except ImportError:
    _HANDWRITING_AVAILABLE = False

try:
    from signature_verification import (
        DocumentSignatureVerification,
        analyze_signature_page,
        finalize_signature_verification,
        write_signature_verification_json,
    )

    _SIGNATURE_VERIFICATION_AVAILABLE = True
except ImportError:
    _SIGNATURE_VERIFICATION_AVAILABLE = False

try:
    from vertical_text import (
        DocumentVerticalText,
        analyze_page_vertical_text,
        finalize_vertical_text,
        write_vertical_analysis_json,
    )

    _VERTICAL_TEXT_AVAILABLE = True
except ImportError:
    _VERTICAL_TEXT_AVAILABLE = False

try:
    from table_fallback import (
        TableRegion,
        analyze_page_tables,
        finalize_table_fallback,
        write_table_fallback_json,
    )

    _TABLE_FALLBACK_AVAILABLE = True
except ImportError:
    _TABLE_FALLBACK_AVAILABLE = False

try:
    from classification import (
        DocumentClassification,
        classify_page_by_layout,
        classify_page_by_text,
        classify_page_ensemble,
        finalize_classification,
        write_classification_json,
    )

    _CLASSIFICATION_AVAILABLE = True
except ImportError:
    _CLASSIFICATION_AVAILABLE = False

try:
    from extraction import (
        DocumentExtraction,
        extract_page_fields,
        finalize_extraction,
        write_extraction_json,
    )

    _EXTRACTION_AVAILABLE = True
except ImportError:
    _EXTRACTION_AVAILABLE = False

try:
    from semantic_extraction import (
        finalize_entity_output,
        write_entities_json,
    )

    _ENTITY_OUTPUT_AVAILABLE = True
except ImportError:
    _ENTITY_OUTPUT_AVAILABLE = False

try:
    from entity_consolidator import (
        consolidate_entities,
        write_consolidated_entities_json,
    )

    _ENTITY_CONSOLIDATOR_AVAILABLE = True
except ImportError:
    _ENTITY_CONSOLIDATOR_AVAILABLE = False

try:
    from relationship_extraction import (
        extract_and_attach_relationships,
    )

    _RELATIONSHIP_EXTRACTION_AVAILABLE = True
except ImportError:
    _RELATIONSHIP_EXTRACTION_AVAILABLE = False

try:
    from routing import (
        derive_document_routing,
        write_routing_json,
    )

    _ROUTING_AVAILABLE = True
except ImportError:
    _ROUTING_AVAILABLE = False

try:
    from output_assembler import (
        assemble_retrieval_output,
        write_retrieval_json,
        write_retrieval_markdown,
    )

    _OUTPUT_ASSEMBLER_AVAILABLE = True
except ImportError:
    _OUTPUT_ASSEMBLER_AVAILABLE = False

try:
    from unicode_utils import normalize_ocr_text, reorder_cjk_vertical_lines

    _UNICODE_UTILS_AVAILABLE = True
except ImportError:
    _UNICODE_UTILS_AVAILABLE = False

try:
    from font_selector import get_font_path as _get_font_path

    _FONT_SELECTOR_AVAILABLE = True
except ImportError:
    _FONT_SELECTOR_AVAILABLE = False

try:
    from exception_router import ExceptionRouter

    _EXCEPTION_ROUTER_AVAILABLE = True
except ImportError:
    _EXCEPTION_ROUTER_AVAILABLE = False

# --- Pluggable engine registry (Phase 1b, opt-in via OCR_ENGINE_REGISTRY) ---
# Default OFF: when disabled the legacy Paddle->Tesseract->Image cascade runs
# unchanged. When enabled, the per-page engine is resolved through the registry
# selector, but the Paddle (forensic) path and any error always fall back to the
# production paddle runner -- see ocr_local.engines.integration.resolve_engine_lines.
try:
    from ocr_local.engines.adapters import (
        register_builtin_adapters as _register_engine_adapters,
    )
    from ocr_local.engines.integration import (
        EngineAdmissionDenied as _EngineAdmissionDenied,
    )
    from ocr_local.engines.integration import (
        engine_provenance as _engine_provenance,
    )
    from ocr_local.engines.integration import (
        resolve_engine_lines as _resolve_engine_lines,
    )
    from ocr_local.engines.registry import DEFAULT_REGISTRY as _ENGINE_REGISTRY

    _ENGINE_REGISTRY_IMPORTABLE = True
except Exception:
    _ENGINE_REGISTRY_IMPORTABLE = False

    class _EngineAdmissionDenied(RuntimeError):
        pass

    def _engine_provenance(engine_name):
        # Fallback when the engines package is unavailable: only the paddle
        # default path runs, so the forensic label is always PaddleOCR/Paddle.
        return ("PaddleOCR", "Paddle")


ENABLE_ENGINE_REGISTRY = (
    os.environ.get("OCR_ENGINE_REGISTRY", "false").lower().strip()
    in ("1", "true", "yes")
    and _ENGINE_REGISTRY_IMPORTABLE
)
if ENABLE_ENGINE_REGISTRY:
    try:
        _register_engine_adapters(_ENGINE_REGISTRY)
    except Exception:
        ENABLE_ENGINE_REGISTRY = False

# --- Per-span language detection (Plan A -- PR A2, opt-in) ---
try:
    from ocr_local.features.language_detection import (
        PageLanguage,
        SpanLanguage,
        aggregate_page_from_spans,
        detect_span_language,
        finalize_document_language,
        write_language_json,
    )

    _LANGUAGE_DETECTION_AVAILABLE = True
except ImportError:
    _LANGUAGE_DETECTION_AVAILABLE = False

# --- Assembly Queue Message Schema (maintenance review) ---


class _AssemblyMessageRequired(TypedDict):
    """Required keys present in every assembly queue message."""

    doc_id: str
    page_num: int
    text: str
    status: str
    chunk_path: str | None
    # Lowercase SHA-256 of durable checkpoint PDF bytes when chunk_path is set;
    # None only for explicit absent-chunk terminal failure paths.
    checkpoint_sha256: str | None


class AssemblyMessage(_AssemblyMessageRequired, total=False):
    """Queue message from worker/extractor threads to the assembler thread.

    Required keys (always present): doc_id, page_num, text, status, chunk_path,
    checkpoint_sha256. Optional keys (OCR-success paths; None/0.0 in failures).
    """

    structure_data: dict | None
    ocr_confidence: float
    ocr_method: str
    ocr_language: str
    # Produced/consumed as domain objects (PageHandwriting, PageSignatureVerification,
    # VerticalTextAnalysis, PageTableFallbackAnalysis), not plain dicts.
    handwriting_data: object | None
    signature_data: object | None
    vertical_text_data: object | None
    table_fallback_data: object | None
    language_data: object | None  # PageLanguage | None (Plan A -- PR A2)
    resume_metadata: dict | None
    cache_hit: bool


# --- Configuration ---


# PipelineConfig is now the startup authority for runtime configuration.
_ACTIVE_PIPELINE_CONFIG = create_pipeline_config()


def _sync_legacy_globals_from_config(cfg: PipelineConfig) -> None:
    """Mirror the active PipelineConfig into legacy module globals.

    The thread functions and several helper paths still read uppercase globals.
    Phase 3 makes the config object authoritative and treats these names as
    compatibility mirrors instead of independent runtime state.
    """
    global IMAGE_QUEUE_SIZE, CHUNK_QUEUE_SIZE, RESULT_QUEUE_SIZE
    global COMPRESSION_QUEUE_SIZE, NUM_EXTRACTORS, NUM_WORKERS
    global NUM_COMPRESSORS, PDF_CONVERSION_THREADS, THREAD_JOIN_TIMEOUT
    global NUM_ASSEMBLER_WORKERS, POPPLER_TIMEOUT, TESSERACT_TIMEOUT
    global SHUTDOWN_DRAIN_TIMEOUT_SECONDS, CHUNK_TARGET_SIZE
    global EXTRACTOR_MODE, EXTRACTOR_PROCESS_WORKERS, DPI, JPEG_QUALITY
    global MONITOR_SLEEP_SECONDS, KEEP_TEMP_FILES
    global VIDEO_FRAME_SAMPLE_SECONDS, VIDEO_MAX_FRAMES, FASTTEXT_MODEL_PATH
    global ENABLE_DOCUMENT_INTELLIGENCE, ENABLE_LAYOUT_ANALYSIS
    global ENABLE_TABLE_EXTRACTION, DOCINTEL_MODE, EXPORT_TABLES
    global ENABLE_FORM_DETECTION, ENABLE_KV_EXTRACTION
    global ENABLE_PRIVILEGE_DETECTION, ENABLE_CUSTODY
    global ENABLE_PREPROCESSING, PREPROCESSING_LEVEL
    global ENABLE_NOISE_PROFILING, ENABLE_VALIDATION
    global ENABLE_DPI_ESCALATION, DPI_CONFIDENCE_THRESHOLD, ENABLE_NER
    global ENABLE_HANDWRITING, ENABLE_SIGNATURE_VERIFICATION
    global ENABLE_VERTICAL_TEXT, ENABLE_TABLE_FALLBACK
    global ENABLE_CLASSIFICATION, ENABLE_EXTRACTION
    global ENABLE_SPECIALIST_ROUTING, ENABLE_ENTITY_CONSOLIDATION
    global ENABLE_RELATIONSHIP_EXTRACTION, ENABLE_RETRIEVAL_OUTPUT
    global ENABLE_EXCEPTION_ROUTING, ENABLE_ADAPTIVE_BATCH
    global ENABLE_PAGE_CACHE, ENABLE_PAGE_ROUTING, ENABLE_GPU_OPTIMIZATION
    global ENABLE_PER_SPAN_LANGUAGE, LANGUAGE_INCLUDE_SPANS
    global LANGUAGE_SHORT_SPAN_THRESHOLD, LANGUAGE_CONFIDENCE_THRESHOLD
    global LANGUAGE_REDACT_SAMPLES
    global SOURCE_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER, LOG_DIR
    global FAILURE_REPORT, HEALTHCHECK_FILE

    IMAGE_QUEUE_SIZE = cfg.image_queue_size
    CHUNK_QUEUE_SIZE = cfg.chunk_queue_size
    RESULT_QUEUE_SIZE = cfg.result_queue_size
    COMPRESSION_QUEUE_SIZE = cfg.compression_queue_size
    NUM_EXTRACTORS = cfg.num_extractors
    NUM_WORKERS = cfg.num_workers
    NUM_COMPRESSORS = cfg.num_compressors
    PDF_CONVERSION_THREADS = cfg.pdf_conversion_threads
    THREAD_JOIN_TIMEOUT = cfg.thread_join_timeout
    NUM_ASSEMBLER_WORKERS = cfg.num_assembler_workers
    POPPLER_TIMEOUT = cfg.poppler_timeout
    TESSERACT_TIMEOUT = cfg.tesseract_timeout
    SHUTDOWN_DRAIN_TIMEOUT_SECONDS = cfg.shutdown_drain_timeout_seconds
    CHUNK_TARGET_SIZE = cfg.chunk_target_size
    EXTRACTOR_MODE = cfg.extractor_mode
    EXTRACTOR_PROCESS_WORKERS = cfg.extractor_process_workers
    DPI = cfg.dpi
    JPEG_QUALITY = cfg.jpeg_quality
    MONITOR_SLEEP_SECONDS = cfg.monitor_sleep_seconds
    KEEP_TEMP_FILES = cfg.keep_temp_files
    VIDEO_FRAME_SAMPLE_SECONDS = cfg.video_frame_sample_seconds
    VIDEO_MAX_FRAMES = cfg.video_max_frames
    FASTTEXT_MODEL_PATH = cfg.fasttext_model_path
    ENABLE_DOCUMENT_INTELLIGENCE = cfg.enable_document_intelligence
    ENABLE_LAYOUT_ANALYSIS = cfg.enable_layout_analysis
    ENABLE_TABLE_EXTRACTION = cfg.enable_table_extraction
    DOCINTEL_MODE = cfg.docintel_mode
    EXPORT_TABLES = cfg.export_tables
    ENABLE_FORM_DETECTION = cfg.enable_form_detection
    ENABLE_KV_EXTRACTION = cfg.enable_kv_extraction
    ENABLE_PRIVILEGE_DETECTION = cfg.enable_privilege_detection
    ENABLE_CUSTODY = cfg.enable_custody
    ENABLE_PREPROCESSING = cfg.enable_preprocessing
    PREPROCESSING_LEVEL = cfg.preprocessing_level
    ENABLE_NOISE_PROFILING = cfg.enable_noise_profiling
    ENABLE_VALIDATION = cfg.enable_validation
    ENABLE_DPI_ESCALATION = cfg.enable_dpi_escalation
    DPI_CONFIDENCE_THRESHOLD = cfg.dpi_confidence_threshold
    ENABLE_NER = cfg.enable_ner
    ENABLE_HANDWRITING = cfg.enable_handwriting
    ENABLE_SIGNATURE_VERIFICATION = cfg.enable_signature_verification
    ENABLE_VERTICAL_TEXT = cfg.enable_vertical_text
    ENABLE_TABLE_FALLBACK = cfg.enable_table_fallback
    ENABLE_CLASSIFICATION = cfg.enable_classification
    ENABLE_EXTRACTION = cfg.enable_extraction
    ENABLE_SPECIALIST_ROUTING = cfg.enable_specialist_routing
    ENABLE_ENTITY_CONSOLIDATION = cfg.enable_entity_consolidation
    ENABLE_RELATIONSHIP_EXTRACTION = cfg.enable_relationship_extraction
    ENABLE_RETRIEVAL_OUTPUT = cfg.enable_retrieval_output
    ENABLE_EXCEPTION_ROUTING = cfg.enable_exception_routing
    ENABLE_ADAPTIVE_BATCH = cfg.enable_adaptive_batch
    ENABLE_PAGE_CACHE = cfg.enable_page_cache
    ENABLE_PAGE_ROUTING = cfg.enable_page_routing
    ENABLE_GPU_OPTIMIZATION = cfg.enable_gpu_optimization
    ENABLE_PER_SPAN_LANGUAGE = cfg.enable_per_span_language
    LANGUAGE_INCLUDE_SPANS = cfg.language_include_spans
    LANGUAGE_SHORT_SPAN_THRESHOLD = cfg.language_short_span_threshold
    LANGUAGE_CONFIDENCE_THRESHOLD = cfg.language_confidence_threshold
    LANGUAGE_REDACT_SAMPLES = cfg.language_redact_samples
    SOURCE_FOLDER = cfg.source_folder
    OUTPUT_FOLDER = cfg.output_folder
    TEMP_FOLDER = cfg.temp_folder
    LOG_DIR = cfg.log_dir
    FAILURE_REPORT = cfg.failure_report
    HEALTHCHECK_FILE = cfg.healthcheck_file


def _activate_pipeline_config(cfg: PipelineConfig) -> None:
    """Set the active PipelineConfig and mirror it into module globals."""
    global _ACTIVE_PIPELINE_CONFIG
    _ACTIVE_PIPELINE_CONFIG = cfg
    _sync_legacy_globals_from_config(cfg)


_activate_pipeline_config(_ACTIVE_PIPELINE_CONFIG)


# Ensure Dirs
def _ensure_dir_with_fallback(target_dir: str, fallback_dir: str) -> str:
    """Create a directory, falling back to a temp path when target is not writable."""
    try:
        os.makedirs(target_dir, exist_ok=True)
        return target_dir
    except OSError:
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir


OUTPUT_FOLDER = _ensure_dir_with_fallback(
    OUTPUT_FOLDER,
    os.path.join(tempfile.gettempdir(), "ocr_output"),
)
LOG_DIR = _ensure_dir_with_fallback(
    LOG_DIR,
    os.path.join(OUTPUT_FOLDER, "logs"),
)
TEMP_FOLDER = _ensure_dir_with_fallback(
    TEMP_FOLDER,
    os.path.join(tempfile.gettempdir(), "ocr_temp"),
)

failure_report_dir = os.path.dirname(FAILURE_REPORT) or OUTPUT_FOLDER
try:
    os.makedirs(failure_report_dir, exist_ok=True)
except OSError:
    failure_report_dir = OUTPUT_FOLDER
    os.makedirs(failure_report_dir, exist_ok=True)
    FAILURE_REPORT = os.path.join(
        failure_report_dir,
        os.path.basename(FAILURE_REPORT) or "failures.csv",
    )

# Keep the active PipelineConfig aligned with the normalized runtime paths.
_ACTIVE_PIPELINE_CONFIG.output_folder = OUTPUT_FOLDER
_ACTIVE_PIPELINE_CONFIG.log_dir = LOG_DIR
_ACTIVE_PIPELINE_CONFIG.temp_folder = TEMP_FOLDER
_ACTIVE_PIPELINE_CONFIG.failure_report = FAILURE_REPORT
_ACTIVE_PIPELINE_CONFIG.healthcheck_file = HEALTHCHECK_FILE
_sync_legacy_globals_from_config(_ACTIVE_PIPELINE_CONFIG)

# Initialize Failure Report
if not os.path.exists(FAILURE_REPORT):
    with open(FAILURE_REPORT, "w", encoding="utf-8") as f:
        f.write("Timestamp,SourcePath,PageNum,Error\n")

# Logging Setup
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"ocr_pipeline_{timestamp}.log")


# Configure stdlib logging to replace custom PrintLogger (TD-009)
# FlushStreamHandler ensures Docker stdout is unbuffered
class FlushStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit for Docker compatibility."""

    def emit(self, record):
        super().emit(record)
        self.flush()


_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")
_file_handler = logging.FileHandler(log_file)
_file_handler.setFormatter(_log_fmt)
_stream_handler = FlushStreamHandler(sys.stdout)
_stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

logger = logging.getLogger("ocr_pipeline")
logger.setLevel(logging.INFO)
if logger.handlers:
    logger.handlers.clear()
logger.addHandler(_file_handler)
logger.addHandler(_stream_handler)


def _resolve_auto_extractor_mode(mode, num_extractors):
    """Resolve EXTRACTOR_MODE='auto' to a concrete execution strategy."""
    if mode != "auto":
        return mode

    if num_extractors > 4:
        logger.info(
            "EXTRACTOR_MODE=auto resolved to 'process' (NUM_EXTRACTORS=%d > 4).",
            num_extractors,
        )
        return "process"

    logger.info(
        "EXTRACTOR_MODE=auto resolved to 'thread' (NUM_EXTRACTORS=%d <= 4).",
        num_extractors,
    )
    return "thread"


# --- Source Format Policy (Phase 1 + Planned Phase 2) ---
PDF_EXTENSIONS = {".pdf"}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".mkv",
    ".webm",
}
PHASE1_IMAGE_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".jp2",
    ".jpx",
    ".pnm",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pcx",
    ".ico",
    ".svg",
    ".svgz",
}
PHASE2_IMAGE_EXTENSIONS = {
    ".heic",
    ".heif",
    ".avif",
    ".jxl",
    ".jxr",
    ".dcx",
    ".xps",
    ".wmf",
    ".emf",
}
STRICT_PDF_SIGNATURE = os.environ.get("OCR_STRICT_PDF_SIGNATURE", "").lower() in (
    "1",
    "true",
    "yes",
)
RESUME_MANIFEST_FILENAME = "resume_manifest.json"
RESUME_MANIFEST_SCHEMA_VERSION = 2
SOURCE_SNAPSHOT_DIRNAME = ".source-snapshot"
# V3 binds the exact UTF-8 text digest into embedded provenance.  V1/V2
# checkpoints lack text_sha256 and must always reprocess on resume/cache.
CHECKPOINT_METADATA_SCHEMA_VERSION = 4
CHECKPOINT_METADATA_SUBJECT_PREFIX = "OCR_LOCAL_CHECKPOINT_V4:"
_EMPTY_TEXT_SHA256 = hashlib.sha256(b"").hexdigest()
_CHECKPOINT_TEXT_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# --- Helper Functions ---
def read_file_header(path, max_bytes=4096):
    """Reads a small leading byte window for signature checks."""
    try:
        with open(path, "rb") as fh:
            return fh.read(max_bytes)
    except Exception as exc:
        logger.debug("Failed to read file header for %s: %s", path, exc)
        return b""


def detect_magic_family(path):
    """Returns 'pdf', 'image', 'video', or None based on magic bytes."""
    header = read_file_header(path)
    return _detect_magic_family_from_header(header)


def _detect_magic_family_from_header(header):
    """Classify only the captured leading bytes of an admitted generation."""
    if not header:
        return None

    stripped = header.lstrip()

    if header.startswith(b"%PDF-"):
        return "pdf"
    # B01 / Sweep B: PE/COFF and ELF are never OCR sources — do not fall through
    # to extension-only admit (PE-as-PDF residual).
    if header.startswith(b"MZ") or header.startswith(b"\x7fELF"):
        return "executable"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if header[:3] == b"\xff\xd8\xff":
        return "image"
    if header[:2] in (b"II", b"MM") and header[2:4] in (b"*\x00", b"\x00*"):
        return "image"
    if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "image"
    if header.startswith(b"BM"):
        return "image"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return "video"
    if header.startswith(b"\x00\x00\x01\x00"):
        return "image"
    if header.startswith((b"P1", b"P2", b"P3", b"P4", b"P5", b"P6")):
        return "image"
    if header.startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n"):
        return "image"
    if header[:4] == b"\xff\x4f\xff\x51":
        return "image"
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return "video"
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "video"
    if b"<svg" in stripped[:2048].lower():
        return "image"

    return None


def _classify_captured_source(path, header, strict_pdf_signature=None):
    """Classify a source using its original extension and captured prefix.

    No path read occurs here: ``header`` must be the prefix accumulated while
    the original source descriptor was copied into the durable snapshot.
    """
    if strict_pdf_signature is None:
        strict_pdf_signature = bool(globals().get("STRICT_PDF_SIGNATURE", False))

    ext = os.path.splitext(path)[1].lower()
    if ext in PHASE2_IMAGE_EXTENSIONS:
        return None, f"Phase-2 extension not yet enabled ({ext})"
    if ext in PDF_EXTENSIONS:
        ext_family = "pdf"
    elif ext in VIDEO_EXTENSIONS:
        ext_family = "video"
    elif ext in PHASE1_IMAGE_EXTENSIONS:
        ext_family = "image"
    else:
        return None, f"Unsupported extension ({ext or '<none>'})"

    magic_family = _detect_magic_family_from_header(header)
    if magic_family == "executable":
        return None, "Executable binary signature refused (PE/ELF; B01)"
    if magic_family and magic_family != ext_family:
        return None, f"Signature mismatch: extension={ext_family} magic={magic_family}"
    if not magic_family:
        if ext_family == "pdf" and strict_pdf_signature:
            return None, "Missing PDF signature (strict mode)"
        return ext_family, "No known signature match; accepted by extension fallback"
    return ext_family, None


def classify_source_file(path, strict_pdf_signature=None):
    """
    Classifies a source file as 'pdf' or 'image' and enforces phase policy.
    Returns: (source_type_or_none, warning_or_reason_or_none)
    """
    if strict_pdf_signature is None:
        strict_pdf_signature = bool(globals().get("STRICT_PDF_SIGNATURE", False))

    ext = os.path.splitext(path)[1].lower()
    if ext in PHASE2_IMAGE_EXTENSIONS:
        return None, f"Phase-2 extension not yet enabled ({ext})"

    if ext in PDF_EXTENSIONS:
        ext_family = "pdf"
    elif ext in VIDEO_EXTENSIONS:
        ext_family = "video"
    elif ext in PHASE1_IMAGE_EXTENSIONS:
        ext_family = "image"
    else:
        return None, f"Unsupported extension ({ext or '<none>'})"

    magic_family = detect_magic_family(path)
    if magic_family == "executable":
        return None, "Executable binary signature refused (PE/ELF; B01)"
    if magic_family and magic_family != ext_family:
        return None, f"Signature mismatch: extension={ext_family} magic={magic_family}"

    if not magic_family:
        if ext_family == "pdf" and strict_pdf_signature:
            return None, "Missing PDF signature (strict mode)"
        return ext_family, "No known signature match; accepted by extension fallback"

    return ext_family, None


def get_source_page_count(path, source_type):
    """Returns total page/frame count for pdf, image, and video inputs.

    R2-RC-02 / R4-RC-09: PDF page count is capped by MAX_PDF_PAGES; video
    frame sampling applies VIDEO_MAX_FRAMES before any range expansion.
    Images enforce MAX_IMAGE_PIXELS (width*height*frames) before admit/OCR.
    """
    max_pdf_pages = _get_env_int("MAX_PDF_PAGES", 5000, min_val=1, max_val=1_000_000)
    max_image_pixels = _get_env_int(
        "MAX_IMAGE_PIXELS", 100_000_000, min_val=0, max_val=2_000_000_000
    )
    if source_type == "pdf":
        with fitz.open(path) as doc:
            count = int(doc.page_count)
            if count > max_pdf_pages:
                raise ValueError(
                    f"PDF has {count} pages which exceeds MAX_PDF_PAGES={max_pdf_pages}"
                )
            return count
    if source_type == "video":
        # Bound frames before any consumer iterates ranges (R4-RC-09).
        return get_video_page_count(
            path,
            sample_seconds=VIDEO_FRAME_SAMPLE_SECONDS,
            max_frames=min(VIDEO_MAX_FRAMES, max_pdf_pages),
        )

    try:
        try:
            from PIL.Image import DecompressionBombError
        except Exception:  # pragma: no cover
            DecompressionBombError = type("DecompressionBombError", (Exception,), {})
        try:
            with Image.open(path) as img:
                w, h = img.size
                frames = max(1, int(getattr(img, "n_frames", 1) or 1))
                total_pixels = int(w) * int(h) * frames
                if max_image_pixels > 0 and total_pixels > max_image_pixels:
                    raise ValueError(
                        f"Image has {total_pixels} pixels "
                        f"({w}x{h} x {frames}) which exceeds "
                        f"MAX_IMAGE_PIXELS={max_image_pixels}"
                    )
                return frames
        except DecompressionBombError as exc:
            raise ValueError(
                f"Image exceeds MAX_IMAGE_PIXELS={max_image_pixels} "
                f"(PIL bomb; R2-RC-02): {exc}"
            ) from exc
    except ValueError:
        raise
    except Exception as exc:
        # R2-RC-02: do not soft-pass image bombs via fitz fallback without a
        # pixel budget. PDF path is handled above; image family must fail closed.
        if str(source_type).lower() in ("image", "img", "picture"):
            raise ValueError(
                f"Image page/pixel probe failed (fail-closed MAX_IMAGE_PIXELS path): "
                f"{path}: {exc}"
            ) from exc
        logger.debug(
            "PIL page count failed for %s, falling back to fitz: %s", path, exc
        )
        with fitz.open(path) as doc:
            return max(1, int(doc.page_count))


def iter_pil_image_frames(path, start, end):
    """Yields PIL RGB frames for 1-based inclusive page range."""
    with Image.open(path) as src:
        frame_count = max(1, int(getattr(src, "n_frames", 1) or 1))
        upper = min(end, frame_count)
        for page_num in range(start, upper + 1):
            src.seek(page_num - 1)
            # maintenance review: explicitly close copied frames after consumer finishes
            # to prevent PIL Image handle accumulation on multi-frame TIFFs.
            frame = white_composite_for_ocr(src).copy()
            try:
                yield frame
            finally:
                try:
                    frame.close()
                except Exception:
                    pass


def iter_fitz_image_frames(path, start, end):
    """Fallback renderer for formats Pillow cannot decode directly."""
    zoom = DPI / 72.0
    with fitz.open(path) as doc:
        upper = min(end, int(doc.page_count))
        for page_num in range(start, upper + 1):
            pix = doc[page_num - 1].get_pixmap(
                matrix=fitz.Matrix(zoom, zoom), alpha=False
            )
            if pix.n == 1:
                yield Image.frombytes(
                    "L", (pix.width, pix.height), pix.samples
                ).convert("RGB")
            else:
                yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def iter_source_images(path, start, end, source_type):
    """Yield ``(source_page_number, PIL image)`` pairs for a page range.

    Page identity must come from the renderer, not from the consumer's
    enumerate index.  Poppler can return no image for an individual page; an
    enumerate-based consumer would then shift every later image onto the
    wrong source page and could seal that corruption into resume checkpoints.
    """
    if source_type == "pdf":
        if not _PDF2IMAGE_AVAILABLE or convert_from_path is None:
            raise RuntimeError("pdf2image dependency not installed")
        # maintenance review: extract one page at a time to avoid O(chunk_pages * 20MB) peak
        for page_num in range(start, end + 1):
            page_images = convert_from_path(
                path,
                first_page=page_num,
                last_page=page_num,
                dpi=DPI,
                thread_count=1,  # single page -- no benefit from multiple threads
                timeout=POPPLER_TIMEOUT,
                use_cropbox=True,
            )
            if page_images:
                yield page_num, page_images[0].convert("RGB")
        return
    if source_type == "video":
        for page_num, frame in enumerate(
            iter_video_frames(
                path,
                start,
                end,
                sample_seconds=VIDEO_FRAME_SAMPLE_SECONDS,
                max_frames=VIDEO_MAX_FRAMES,
            ),
            start=start,
        ):
            if page_num > end:
                break
            yield page_num, frame
        return

    try:
        for page_num, frame in enumerate(
            iter_pil_image_frames(path, start, end), start=start
        ):
            yield page_num, frame
    except Exception as pil_err:
        logger.warning("PIL decode fallback for %s: %s", path, pil_err)
        for page_num, frame in enumerate(
            iter_fitz_image_frames(path, start, end), start=start
        ):
            yield page_num, frame


def _encode_image_to_jpeg_bytes(img):
    """Serialize a lossless PIL image for optional process-based extraction."""
    payload = io.BytesIO()
    white_composite_for_ocr(img).save(payload, format="PNG", optimize=False)
    return payload.getvalue()


def _decode_image_from_jpeg_bytes(payload):
    """Decode serialized lossless extractor payload into a PIL RGB image."""
    with Image.open(io.BytesIO(payload)) as img:
        return img.convert("RGB")


def _capture_source_geometries(runtime_path, start, end, source_type):
    """Capture visible page sizes from the scheduler-authorized runtime source.

    This runs inside extraction, before pages enter the image queue.  The
    identity path is never used as a later geometry authority.  PDF CropBox and
    rotation are read from the same immutable source used by the renderer.
    """
    if source_type == "pdf":
        with fitz.open(runtime_path) as source:
            if end > source.page_count:
                raise ValueError("PDF extraction range exceeds authorized source")
            return {
                page_num: PageGeometry.from_pdf_page(source[page_num - 1])
                for page_num in range(start, end + 1)
            }
    if source_type != "video":
        try:
            with Image.open(runtime_path) as source:
                result = {}
                for page_num in range(start, min(end, int(getattr(source, "n_frames", 1) or 1)) + 1):
                    source.seek(page_num - 1)
                    result[page_num] = PageGeometry.from_image_pixels(
                        *source.size, dpi=source.info.get("dpi"), fallback_dpi=DPI
                    )
                return result
        except (OSError, ValueError):
            # Pillow-unsupported image formats use the bounded fitz renderer.
            with fitz.open(runtime_path) as source:
                return {
                    page_num: PageGeometry.from_pdf_page(source[page_num - 1])
                    for page_num in range(start, min(end, source.page_count) + 1)
                }
    return {}


def _extract_chunk_in_subprocess(
    input_path, start, end, source_type, dpi, pdf_conversion_threads
):
    """Extract pages in a worker process and return lossless PNG payloads."""
    if end < start or end - start + 1 > 4:
        raise ValueError("subprocess extraction result exceeds four-page IPC bound")
    extracted = []

    if source_type == "pdf":
        if not _PDF2IMAGE_AVAILABLE or convert_from_path is None:
            raise RuntimeError("pdf2image dependency not installed")
        # maintenance review: extract one page at a time to avoid O(chunk_pages * 20MB) peak
        for page_num in range(start, end + 1):
            page_images = convert_from_path(
                input_path,
                first_page=page_num,
                last_page=page_num,
                dpi=dpi,
                thread_count=1,  # single page -- no benefit from multiple threads
                timeout=POPPLER_TIMEOUT,
                use_cropbox=True,
            )
            if page_images:
                extracted.append(
                    (page_num, _encode_image_to_jpeg_bytes(page_images[0]))
                )
        return extracted

    if source_type == "video":
        for page_num, frame in enumerate(
            iter_video_frames(
                input_path,
                start,
                end,
                sample_seconds=VIDEO_FRAME_SAMPLE_SECONDS,
                max_frames=VIDEO_MAX_FRAMES,
            ),
            start=start,
        ):
            if page_num > end:
                break
            extracted.append((page_num, _encode_image_to_jpeg_bytes(frame)))
        return extracted

    # Image pathway: preserve parity with iter_source_images()
    try:
        with Image.open(input_path) as src:
            frame_count = max(1, int(getattr(src, "n_frames", 1) or 1))
            upper = min(end, frame_count)
            for page_num in range(start, upper + 1):
                src.seek(page_num - 1)
                extracted.append((page_num, _encode_image_to_jpeg_bytes(src)))
        return extracted
    except Exception as exc:
        logger.debug(
            "PIL extraction failed for %s, falling back to fitz: %s", input_path, exc
        )
        zoom = dpi / 72.0
        with fitz.open(input_path) as doc:
            upper = min(end, int(doc.page_count))
            for page_num in range(start, upper + 1):
                pix = doc[page_num - 1].get_pixmap(
                    matrix=fitz.Matrix(zoom, zoom), alpha=False
                )
                if pix.n == 1:
                    frame = Image.frombytes(
                        "L", (pix.width, pix.height), pix.samples
                    ).convert("RGB")
                else:
                    frame = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                extracted.append((page_num, _encode_image_to_jpeg_bytes(frame)))
        return extracted


# Canonical version: ocr_distributed.ocr_utils.sanitize_path_segment
def _sanitize_path_segment(segment):
    """Remove or replace characters that are problematic on Windows/Linux filesystems."""
    # Replace characters illegal on Windows: < > : " | ? *
    # Also replace control characters and null bytes
    sanitized = ""
    for ch in segment:
        if ch in '<>:"|?*\x00' or (ord(ch) < 32):
            sanitized += "_"
        else:
            sanitized += ch
    return sanitized.strip(". ")


def build_output_rel_stem(path, source_type):
    """Return the canonical collision-safe stem beneath ``SOURCE_FOLDER``."""
    return _canonical_build_output_rel_stem(path, source_type, SOURCE_FOLDER)


def _build_sidecar_base_name(source_file):
    """Build sidecar basename with non-PDF extension token to avoid collisions."""
    return _canonical_build_sidecar_base_name(source_file)


def _safe_source_relative_path(source_file):
    """Return a source-relative path without prefix or traversal ambiguity."""
    try:
        source_root = os.path.realpath(SOURCE_FOLDER)
        source_real = os.path.realpath(source_file)
        common_root = os.path.commonpath([source_root, source_real])
        if os.path.normcase(common_root) == os.path.normcase(source_root):
            return os.path.relpath(source_file, SOURCE_FOLDER)
    except (OSError, ValueError):
        pass
    return os.path.basename(source_file)


def _validation_sidecar_path_for_source(source_file):
    """Resolve existing validation sidecar path for a source file."""
    validation_dir = os.path.join(OUTPUT_FOLDER, "EXPORT", "VALIDATION")
    source_type = (
        "pdf" if os.path.splitext(source_file)[1].lower() == ".pdf" else "image"
    )
    rel_stem = build_output_rel_stem(source_file, source_type)
    return os.path.join(validation_dir, rel_stem + ".validation.json")


def _extract_text_from_pdf_document(doc):
    """Return the stripped concatenated text layer from an open fitz document."""
    parts = []
    for page in doc:
        text = page.get_text("text")
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _extract_text_from_pdf_bytes(pdf_bytes):
    """Extract embedded text from in-memory PDF bytes (fail closed)."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return _extract_text_from_pdf_document(doc)


def _extract_text_from_chunk_pdf(chunk_path, *, strict=False):
    """Extract embedded text from a chunk PDF.

    When *strict* is False (defensive non-production fallback only), missing
    files and read failures degrade to ``""``.  When *strict* is True (live
    post-write authority and resume/cache validation), extraction failures
    raise so the page fails closed instead of silently shipping divergent
    accumulator text.
    """
    if not chunk_path or not os.path.exists(chunk_path):
        if strict:
            raise FileNotFoundError(
                f"checkpoint PDF missing for canonical text extraction: {chunk_path}"
            )
        return ""

    try:
        with fitz.open(chunk_path) as doc:
            return _extract_text_from_pdf_document(doc)
    except Exception as exc:
        if strict:
            raise
        logger.debug("Failed to extract text from chunk PDF %s: %s", chunk_path, exc)
        return ""


def _canonical_text_sha256(text: str) -> str:
    """Return lowercase SHA-256 hex of *text* encoded as UTF-8."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _checkpoint_bytes_sha256(pdf_bytes: bytes) -> str:
    """Return lowercase SHA-256 hex of exact checkpoint PDF *pdf_bytes*."""
    if not isinstance(pdf_bytes, (bytes, bytearray)):
        raise TypeError("checkpoint bytes must be bytes-like")
    return hashlib.sha256(bytes(pdf_bytes)).hexdigest()


def _is_valid_checkpoint_sha256(value) -> bool:
    """True only for a lowercase 64-hex SHA-256 string."""
    return (
        isinstance(value, str)
        and _CHECKPOINT_TEXT_SHA256_RE.fullmatch(value) is not None
    )


def _read_checkpoint_file_bytes(path: str) -> bytes:
    """Read durable checkpoint PDF bytes once (no fitz open)."""
    with open(path, "rb") as handle:
        return handle.read()


def _durable_checkpoint_matching_intended_bytes(path, intended_bytes):
    """Read durable bytes once and require exact digest equality to intended.

    Live and critical image-only fallback writers compute the intended embedded
    PDF digest before atomic write.  A mutation/write divergence before the
    durable snapshot must not be rebound as authoritative.
    """
    if not isinstance(intended_bytes, (bytes, bytearray)):
        raise TypeError("intended checkpoint bytes must be bytes-like")
    intended = bytes(intended_bytes)
    intended_digest = _checkpoint_bytes_sha256(intended)
    durable_bytes = _read_checkpoint_file_bytes(path)
    durable_digest = _checkpoint_bytes_sha256(durable_bytes)
    if durable_digest != intended_digest:
        raise RuntimeError(
            "post-write checkpoint durable digest diverged from intended embedded bytes"
        )
    return durable_bytes, durable_digest


def _validated_checkpoint_text_value(text_value, metadata):
    """Bind an already-extracted canonical text string to provenance metadata.

    Fail closed unless length, presence, and UTF-8 SHA-256 all match.  Same-length
    substitutions (``AAAA`` -> ``BBBB``) are rejected via the digest check.
    """
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata required for text validation")
    text_length = metadata.get("text_length")
    if (
        isinstance(text_length, bool)
        or not isinstance(text_length, int)
        or text_length < 0
    ):
        raise ValueError("checkpoint metadata text_length is invalid")
    if "has_text" not in metadata or not isinstance(metadata["has_text"], bool):
        raise ValueError("checkpoint metadata has_text is invalid")
    expected_has_text = metadata["has_text"]
    expected_digest = metadata.get("text_sha256")
    if not isinstance(expected_digest, str) or not _CHECKPOINT_TEXT_SHA256_RE.fullmatch(
        expected_digest
    ):
        raise ValueError("checkpoint metadata text_sha256 is invalid")
    if not isinstance(text_value, str):
        raise ValueError("checkpoint text must be a string")
    if len(text_value) != text_length:
        raise ValueError(
            "checkpoint text length mismatch: "
            f"extracted={len(text_value)} metadata={text_length}"
        )
    if bool(text_value) is not expected_has_text:
        raise ValueError(
            "checkpoint text presence mismatch: "
            f"extracted={bool(text_value)} metadata={expected_has_text}"
        )
    actual_digest = _canonical_text_sha256(text_value)
    if actual_digest != expected_digest:
        raise ValueError(
            "checkpoint text digest mismatch: "
            f"extracted={actual_digest} metadata={expected_digest}"
        )
    return text_value


def _validated_checkpoint_text(path, metadata):
    """Strictly extract checkpoint PDF text and bind it to provenance metadata.

    Resume and cache paths must queue this exact string rather than re-deriving
    text later in the assembler.  Extraction uses the same canonical rules as
    live post-write authority and finalization revalidation.
    """
    text_value = _extract_text_from_chunk_pdf(path, strict=True)
    return _validated_checkpoint_text_value(text_value, metadata)


def _validated_checkpoint_text_from_document(doc, metadata):
    """Validate text from an already-open fitz document (no path re-open)."""
    text_value = _extract_text_from_pdf_document(doc)
    return _validated_checkpoint_text_value(text_value, metadata)


def _coherent_checkpoint_bytes_snapshot(
    pdf_bytes,
    *,
    document_id,
    page_num,
    source_label="checkpoint bytes",
):
    """Derive digest, metadata, and text from one immutable byte snapshot."""
    if not isinstance(pdf_bytes, (bytes, bytearray)):
        raise TypeError("checkpoint candidate must be bytes-like")
    pdf_bytes = bytes(pdf_bytes)
    digest = _checkpoint_bytes_sha256(pdf_bytes)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as checkpoint:
        if checkpoint.page_count != 1 or checkpoint.needs_pass:
            raise ValueError(
                f"checkpoint is not a readable single-page PDF: {source_label}"
            )
        checkpoint.load_page(0)
        metadata = _load_page_checkpoint_metadata_from_document(
            checkpoint,
            document_id=document_id,
            page_num=page_num,
        )
        if metadata is None:
            raise ValueError("legacy checkpoint has no provenance")
        incompatibility = _checkpoint_resume_incompatibility(metadata)
        if incompatibility:
            raise ValueError(incompatibility)
        text_value = _validated_checkpoint_text_from_document(checkpoint, metadata)
    return pdf_bytes, digest, metadata, text_value


def _coherent_checkpoint_snapshot(path, *, document_id, page_num):
    """Read a checkpoint once and derive digest, metadata, and validated text.

    Returns ``(pdf_bytes, checkpoint_sha256, metadata, text)`` from a single
    durable byte snapshot so resume/cache producers cannot bind mismatched
    path re-opens.
    """
    pdf_bytes = _read_checkpoint_file_bytes(path)
    return _coherent_checkpoint_bytes_snapshot(
        pdf_bytes,
        document_id=document_id,
        page_num=page_num,
        source_label=path,
    )


def _publish_or_select_cached_checkpoint(
    cache_chunk,
    cached_bytes,
    *,
    document_id,
    page_num,
    expected_geometry=None,
):
    """Validate cache bytes before any canonical write, then return authority.

    A valid existing canonical checkpoint wins.  An absent or invalid canonical
    checkpoint may be replaced only after the cache candidate has passed the
    complete in-memory provenance/text/profile validation.
    """
    candidate_bytes, _candidate_digest, _candidate_meta, _candidate_text = (
        _coherent_checkpoint_bytes_snapshot(
            cached_bytes,
            document_id=document_id,
            page_num=page_num,
            source_label=f"page cache candidate for {document_id} p{page_num}",
        )
    )
    expected_record = (
        json.loads(json.dumps(asdict(expected_geometry)))
        if expected_geometry is not None else None
    )
    if expected_record is not None and _candidate_meta["page_geometry"] != expected_record:
        raise ValueError("page cache candidate geometry differs from current source")
    with _get_checkpoint_artifact_lock(document_id):
        if os.path.isfile(cache_chunk):
            try:
                authority = _coherent_checkpoint_snapshot(
                    cache_chunk,
                    document_id=document_id,
                    page_num=page_num,
                )
            except (OSError, RuntimeError, ValueError):
                _atomic_write_bytes(cache_chunk, candidate_bytes)
            else:
                if expected_record is not None and authority[2]["page_geometry"] != expected_record:
                    raise ValueError("canonical checkpoint geometry differs from current source")
                return authority
        else:
            _atomic_write_bytes(cache_chunk, candidate_bytes)
        return _coherent_checkpoint_snapshot(
            cache_chunk,
            document_id=document_id,
            page_num=page_num,
        )


def _canonical_text_after_checkpoint_pdf(pdf_bytes, *, ocr_text):
    """Derive assembly/metadata text from the checkpoint PDF text layer.

    Live, resume, and cache paths must agree on the bytes that will actually be
    merged.  If OCR produced text but the checkpoint layer cannot be read back,
    fail closed rather than emit divergent accumulator text.
    """
    try:
        canonical = _extract_text_from_pdf_bytes(pdf_bytes)
    except Exception as exc:
        raise RuntimeError(
            f"canonical checkpoint text extraction failed: {exc}"
        ) from exc
    if (ocr_text or "").strip() and not canonical:
        raise RuntimeError(
            "checkpoint PDF text layer is empty despite non-empty OCR text"
        )
    expected = "".join(unicodedata.normalize("NFKC", ocr_text or "").split())
    observed = "".join(unicodedata.normalize("NFKC", canonical).split())
    if expected != observed:
        raise RuntimeError(
            "checkpoint PDF text layer differs from recognized OCR text"
        )
    return canonical


def get_path_based_doc_id(path):
    """Returns a deterministic document ID derived from the file path.

    This is NOT a content hash -- use compute_file_hash() from
    ocr_distributed/ocr_utils.py for content integrity verification.
    """
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


# Backward-compatible alias (deprecated)
get_file_hash = get_path_based_doc_id


def compute_source_fingerprint(path):
    """Compute fingerprint data used to validate resume cache integrity.

    SLOW: reads entire file for SHA-256 content hash.  Use
    ``compute_source_fingerprint_fast`` in hot paths (e.g. the scheduler)
    where mtime+size is sufficient for change detection.
    """
    stat_info = os.stat(path)
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return {
        "content_sha256": digest.hexdigest(),
        "size_bytes": int(stat_info.st_size),
        "mtime_ns": int(
            getattr(stat_info, "st_mtime_ns", int(stat_info.st_mtime * 1_000_000_000))
        ),
    }


def _stage_source_snapshot(source_path):
    """Stream *source_path* into a durable scheduler-owned snapshot.

    The snapshot bytes, rather than a repeatedly reopened mutable source path,
    are the authority used by page counting and every extraction attempt.  A
    unique staging file is used because the document id is derived from the
    snapshot digest and therefore is not known until the copy is complete.
    """
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    suffix = os.path.splitext(source_path)[1]
    fd, staged_path = tempfile.mkstemp(
        dir=TEMP_FOLDER,
        prefix=".source-snapshot-",
        suffix=suffix or ".bin",
    )
    try:
        with open(source_path, "rb") as source, os.fdopen(fd, "wb") as snapshot:
            before = os.fstat(source.fileno())
            digest = hashlib.sha256()
            copied = 0
            captured_prefix = bytearray()
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                snapshot.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
                if len(captured_prefix) < 4096:
                    captured_prefix.extend(chunk[: 4096 - len(captured_prefix)])
            snapshot.flush()
            os.fsync(snapshot.fileno())
            after = os.fstat(source.fileno())
            staged_stat = os.fstat(snapshot.fileno())

        def _generation(stat_info):
            return (
                getattr(stat_info, "st_dev", None),
                getattr(stat_info, "st_ino", None),
                int(stat_info.st_size),
                int(
                    getattr(
                        stat_info,
                        "st_mtime_ns",
                        int(stat_info.st_mtime * 1_000_000_000),
                    )
                ),
                int(
                    getattr(
                        stat_info,
                        "st_ctime_ns",
                        int(stat_info.st_ctime * 1_000_000_000),
                    )
                ),
            )

        if _generation(before) != _generation(after):
            raise RuntimeError(
                "source generation changed while immutable snapshot was copied"
            )
        if copied != int(before.st_size) or int(staged_stat.st_size) != copied:
            raise RuntimeError(
                "source snapshot byte count diverged from the admitted generation"
            )

        # Every identity field comes from the same open source-file generation.
        # Never combine copied bytes with a later path-based stat result.
        fingerprint = {
            "content_sha256": digest.hexdigest(),
            "size_bytes": copied,
            "mtime_ns": int(
                getattr(
                    before,
                    "st_mtime_ns",
                    int(before.st_mtime * 1_000_000_000),
                )
            ),
        }
        return staged_path, fingerprint, bytes(captured_prefix)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.remove(staged_path)
        except OSError:
            pass
        raise


def _publish_source_snapshot(staged_path, temp_dir, source_path, fingerprint):
    """Atomically publish and verify a per-document immutable source spool."""
    snapshot_dir = os.path.join(temp_dir, SOURCE_SNAPSHOT_DIRNAME)
    os.makedirs(snapshot_dir, exist_ok=True)
    suffix = os.path.splitext(source_path)[1]
    snapshot_path = os.path.join(snapshot_dir, "source" + (suffix or ".bin"))
    published_this_attempt = False
    try:
        os.replace(staged_path, snapshot_path)
        published_this_attempt = True
        _fsync_directory(snapshot_dir)
        published = {
            "content_sha256": _file_sha256(snapshot_path),
            "size_bytes": os.path.getsize(snapshot_path),
        }
        if published.get("content_sha256") != fingerprint.get(
            "content_sha256"
        ) or published.get("size_bytes") != fingerprint.get("size_bytes"):
            raise RuntimeError(
                "published source snapshot does not match scheduler fingerprint"
            )
        return snapshot_path
    except Exception:
        # Only artifacts owned by this attempt are removed.  If replacement
        # itself failed, ``snapshot_path`` may still be the prior valid spool
        # and must remain available for crash recovery.
        candidates = [staged_path]
        if published_this_attempt:
            candidates.append(snapshot_path)
        for candidate in candidates:
            try:
                os.remove(candidate)
            except OSError:
                pass
        raise


def _create_linux_memfd(name):
    """Create a sealable memfd even when Python omits the os wrapper."""
    allow_sealing = getattr(os, "MFD_ALLOW_SEALING", 0x0002)
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, flags=allow_sealing)

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    memfd_create = getattr(libc, "memfd_create", None)
    if memfd_create is None:
        raise RuntimeError("Linux libc memfd_create is unavailable")
    memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    memfd_create.restype = ctypes.c_int
    fd = memfd_create(name.encode("utf-8"), allow_sealing)
    if fd < 0:
        errno_value = ctypes.get_errno()
        raise OSError(errno_value, os.strerror(errno_value))
    return fd


class _ImmutableSourceAuthority:
    """Opened, immutable byte generation used by every source consumer.

    Windows keeps a read-only mapping alive; the OS consequently refuses both
    replacement and in-place truncating writes to the mapped snapshot.  Linux
    copies the verified durable snapshot into a write/grow/shrink-sealed memfd
    and exposes the opened generation through the parent process's ``/proc``
    descriptor path, which remains child-openable for process extractors.
    """

    def __init__(self, canonical_path, expected_fingerprint):
        self.canonical_path = os.path.abspath(canonical_path)
        self.runtime_path = ""
        self._file = None
        self._fd = None
        self._mapping = None
        self._closed = False
        expected_size = int(expected_fingerprint.get("size_bytes", -1))
        expected_digest = str(expected_fingerprint.get("content_sha256", ""))
        if expected_size <= 0 or not _is_valid_checkpoint_sha256(expected_digest):
            raise ValueError("immutable source authority fingerprint is invalid")

        try:
            if os.name == "nt":
                self._file = open(self.canonical_path, "rb")
                self._mapping = mmap.mmap(
                    self._file.fileno(),
                    0,
                    access=mmap.ACCESS_READ,
                )
                self.runtime_path = self.canonical_path
            elif sys.platform.startswith("linux"):
                import fcntl

                self._fd = _create_linux_memfd("ocr-local-source")
                digest = hashlib.sha256()
                copied = 0
                with open(self.canonical_path, "rb") as source:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        copied += len(chunk)
                        view = memoryview(chunk)
                        try:
                            while view:
                                written = os.write(self._fd, view)
                                if written <= 0:
                                    raise OSError(
                                        "short write to source authority memfd"
                                    )
                                view = view[written:]
                        finally:
                            view.release()
                if copied != expected_size or digest.hexdigest() != expected_digest:
                    raise RuntimeError(
                        "durable source snapshot diverged before sealed admission"
                    )
                os.fsync(self._fd)
                # Some portable Python builds omit Linux-only fcntl names.
                # These values are part of the stable Linux userspace ABI;
                # fcntl still rejects the operation on an unsupported kernel.
                add_seals = getattr(fcntl, "F_ADD_SEALS", 1033)
                get_seals = getattr(fcntl, "F_GET_SEALS", 1034)
                seal_seal = getattr(fcntl, "F_SEAL_SEAL", 0x0001)
                seal_shrink = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
                seal_grow = getattr(fcntl, "F_SEAL_GROW", 0x0004)
                seal_write = getattr(fcntl, "F_SEAL_WRITE", 0x0008)
                required_seals = seal_write | seal_grow | seal_shrink | seal_seal
                fcntl.fcntl(self._fd, add_seals, required_seals)
                observed_seals = fcntl.fcntl(self._fd, get_seals)
                if observed_seals & required_seals != required_seals:
                    raise RuntimeError("Linux source memfd is not fully sealed")
                os.lseek(self._fd, 0, os.SEEK_SET)
                # F_SEAL_WRITE rejects new shared mappings on Linux/WSL even
                # when Python requests ACCESS_READ.  A private, read-only map
                # preserves the immutable opened generation without requiring
                # write-capable shared-map admission.
                self._mapping = mmap.mmap(
                    self._fd,
                    0,
                    flags=mmap.MAP_PRIVATE,
                    prot=mmap.PROT_READ,
                )
                self.runtime_path = f"/proc/{os.getpid()}/fd/{self._fd}"
                with open(self.runtime_path, "rb") as probe:
                    if probe.read(1) != self._mapping[:1]:
                        raise RuntimeError("sealed source authority path is unreadable")
            else:
                raise RuntimeError(
                    "immutable source authority requires Windows mmap or Linux memfd"
                )

            self.verify(expected_fingerprint)
        except Exception:
            self.close()
            raise

    def verify(self, expected_fingerprint):
        if self._closed or self._mapping is None:
            raise RuntimeError("immutable source authority is closed")
        expected_size = int(expected_fingerprint.get("size_bytes", -1))
        expected_digest = str(expected_fingerprint.get("content_sha256", ""))
        observed_size = len(self._mapping)
        observed_digest = hashlib.sha256(self._mapping).hexdigest()
        if observed_size != expected_size or observed_digest != expected_digest:
            raise RuntimeError("immutable source authority bytes diverged")
        return {
            "content_sha256": observed_digest,
            "size_bytes": observed_size,
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._mapping is not None:
            try:
                self._mapping.close()
            except Exception:
                pass
            self._mapping = None
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


def _open_immutable_source_authority(canonical_path, expected_fingerprint):
    """Create or fail closed; there is no mutable-path fallback."""
    return _ImmutableSourceAuthority(canonical_path, expected_fingerprint)


def compute_source_fingerprint_fast(path):
    """Fast source fingerprint using os.stat() only -- no file I/O.

    Uses path + mtime_ns + size_bytes to detect source changes.  This is
    sufficient for crash-resume cache invalidation: any file modification
    updates mtime, and most content changes also alter size.  The
    vanishingly rare edge case of same-size, same-mtime edits is
    acceptable for the resume use case (not forensic custody).

    Returns a dict compatible with ``build_resume_doc_id`` and
    ``prepare_resume_state``.
    """
    stat_info = os.stat(path)
    mtime_ns = int(
        getattr(stat_info, "st_mtime_ns", int(stat_info.st_mtime * 1_000_000_000))
    )
    size_bytes = int(stat_info.st_size)
    # Derive a deterministic content_sha256 substitute from path+stat so
    # build_resume_doc_id produces a stable, collision-resistant ID
    # without reading the file.
    stat_key = f"{path}:{mtime_ns}:{size_bytes}"
    stat_hash = hashlib.sha256(stat_key.encode("utf-8")).hexdigest()
    return {
        "content_sha256": stat_hash,
        "size_bytes": size_bytes,
        "mtime_ns": mtime_ns,
    }


def build_resume_doc_id(path, source_fingerprint):
    """Build a deterministic document id from path + source fingerprint."""
    content_sha256 = str(source_fingerprint.get("content_sha256", ""))
    size_bytes = int(source_fingerprint.get("size_bytes", 0) or 0)
    mtime_ns = int(source_fingerprint.get("mtime_ns", 0) or 0)
    payload = f"{path}|{content_sha256}|{size_bytes}|{mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _resume_manifest_path(temp_dir):
    """Return the manifest path for a document temp directory."""
    return os.path.join(temp_dir, RESUME_MANIFEST_FILENAME)


def _load_resume_manifest(temp_dir):
    """Load resume manifest JSON when present and valid."""
    manifest_path = _resume_manifest_path(temp_dir)
    if not os.path.exists(manifest_path):
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        logger.debug("Failed to load resume manifest %s: %s", manifest_path, exc)
        return None

    if not isinstance(payload, dict):
        return None

    if payload.get("schema_version") != RESUME_MANIFEST_SCHEMA_VERSION:
        return None

    if not isinstance(payload.get("source_fingerprint"), dict):
        return None

    return payload


def _write_manifest_atomic(path, data):
    """Write JSON manifest atomically using tmp->fsync->rename pattern.

    Crash-safety (maintenance review): writes payload to a uniquely-named temp file in
    the target directory, calls ``flush()`` + ``os.fsync()`` to force the
    bytes to disk, then performs an atomic ``os.replace``.  On any failure
    the temp file is best-effort unlinked so no partial state is left
    behind.  This guarantees that after a power loss the resume manifest
    is either the old valid version or the new valid version -- never a
    truncated/partial file.
    """
    dir_path = os.path.dirname(path) or "."
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=dir_path,
        prefix=".manifest-",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _write_resume_manifest(temp_dir, source_path, source_fingerprint):
    """Write resume manifest atomically with fsync crash-safety."""
    os.makedirs(temp_dir, exist_ok=True)
    manifest_path = _resume_manifest_path(temp_dir)
    payload = {
        "schema_version": RESUME_MANIFEST_SCHEMA_VERSION,
        "source_path": source_path,
        "source_fingerprint": source_fingerprint,
        "updated_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat()
        + "Z",
    }
    _write_manifest_atomic(manifest_path, payload)
    return payload


def _reset_resume_temp_dir(temp_dir):
    """Remove stale temp outputs for a document while preserving the temp root."""
    os.makedirs(temp_dir, exist_ok=True)
    removed_entries = 0
    for entry in os.listdir(temp_dir):
        if entry == RESUME_MANIFEST_FILENAME:
            continue
        entry_path = os.path.join(temp_dir, entry)
        try:
            if os.path.isdir(entry_path) and not os.path.islink(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)
            removed_entries += 1
        except FileNotFoundError:
            continue
    return removed_entries


def prepare_resume_state(temp_dir, source_path, source_fingerprint):
    """Validate resume cache state and invalidate stale chunks when source changed."""
    os.makedirs(temp_dir, exist_ok=True)

    existing_entries = [
        name for name in os.listdir(temp_dir) if name != RESUME_MANIFEST_FILENAME
    ]
    manifest = _load_resume_manifest(temp_dir)
    manifest_fingerprint = manifest.get("source_fingerprint") if manifest else None
    manifest_source_path = manifest.get("source_path") if manifest else None
    is_match = (
        manifest is not None
        and manifest_source_path == source_path
        and manifest_fingerprint == source_fingerprint
    )

    if is_match:
        _write_resume_manifest(temp_dir, source_path, source_fingerprint)
        return {"status": "valid", "removed_entries": 0}

    if manifest is None and not existing_entries:
        _write_resume_manifest(temp_dir, source_path, source_fingerprint)
        return {"status": "initialized", "removed_entries": 0}

    removed_entries = _reset_resume_temp_dir(temp_dir)
    _write_resume_manifest(temp_dir, source_path, source_fingerprint)
    return {"status": "invalidated", "removed_entries": removed_entries}


def compute_resume_gap_chunks(
    total_pages: int,
    existing_pages,
    chunk_target_size: int,
):
    """Compute the ``(start, end)`` chunk ranges the scheduler must dispatch
    to fully reconstruct a document given a set of already-processed pages.

    Uses set-difference semantics (``{1..N} - existing``) to identify the
    exact set of missing pages, then groups them into **contiguous** runs,
    and finally splits each run into sub-chunks no larger than
    ``chunk_target_size``. Each returned tuple ``(start, end)`` is inclusive
    and contiguous, so ``end - start + 1`` is the exact number of pages that
    the extractor must materialize for that chunk.

    This is the maintenance review investigation outcome: the prior in-place algorithm
    in ``scheduler_thread`` was already correct (its per-page accumulator is
    always contiguous by construction), but an explicit set-difference
    formulation is clearer, defensively ignores stale out-of-range entries
    in ``existing_pages``, and is directly unit-testable in isolation.

    Args:
        total_pages: Total pages in the source document (1-indexed range
            ``[1, total_pages]``).
        existing_pages: Iterable of page numbers already processed (e.g.,
            from disk scan or resume manifest). Entries outside
            ``[1, total_pages]`` are ignored.
        chunk_target_size: Maximum pages per dispatched chunk. Must be >= 1.

    Returns:
        A list of ``(start, end)`` tuples (both 1-indexed, inclusive),
        sorted by ``start``. Empty list when no pages are missing.
    """
    if total_pages <= 0:
        return []
    if chunk_target_size < 1:
        chunk_target_size = 1

    all_pages = set(range(1, total_pages + 1))
    existing = {p for p in existing_pages if 1 <= p <= total_pages}
    missing = sorted(all_pages - existing)

    chunks: list[tuple[int, int]] = []
    if not missing:
        return chunks

    # Group sorted missing pages into contiguous runs, then split each run
    # into pieces bounded by chunk_target_size.
    run_start = missing[0]
    prev = missing[0]
    run_len = 1

    def _emit_run(start: int, end: int) -> None:
        # Split contiguous [start, end] into chunks of at most chunk_target_size
        cur = start
        while cur <= end:
            chunk_end = min(cur + chunk_target_size - 1, end)
            chunks.append((cur, chunk_end))
            cur = chunk_end + 1

    for p in missing[1:]:
        if p == prev + 1 and run_len < chunk_target_size:
            # Extend current sub-chunk within the same contiguous run
            prev = p
            run_len += 1
        elif p == prev + 1:
            # Contiguous, but current sub-chunk is full — flush and start new
            chunks.append((run_start, prev))
            run_start = p
            prev = p
            run_len = 1
        else:
            # Non-contiguous: flush current run and start a new one
            chunks.append((run_start, prev))
            run_start = p
            prev = p
            run_len = 1

    chunks.append((run_start, prev))
    return chunks


def _is_valid_page_checkpoint(path):
    """Return True only for a complete, readable, single-page PDF checkpoint."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 100:
            return False
        with fitz.open(path) as checkpoint:
            if checkpoint.page_count != 1 or checkpoint.needs_pass:
                return False
            checkpoint.load_page(0)
            return True
    except Exception:
        return False


def _checkpoint_validation_status(status, has_text):
    """Map a pipeline page status to its durable validation status."""
    status_value = status.value if isinstance(status, enum.Enum) else str(status)
    if status_value in {
        PageStatus.OK.value,
        PageStatus.PADDLE.value,
        PageStatus.CACHED.value,
    } or status_value.startswith("Paddle-"):
        return "ok"
    if status_value in {PageStatus.TESSERACT.value, PageStatus.EASYOCR.value}:
        return "fallback"
    if status_value == PageStatus.IMAGE_ONLY.value:
        return "image_only"
    if status_value in {
        PageStatus.EXTRACT_FAILED.value,
        PageStatus.CRITICAL_FAILED.value,
    }:
        return "failed"
    return "ok" if has_text else "unknown"


def _compact_page_language(language_data):
    """Return the recovery-safe PageLanguage summary (spans stay out of PDF info)."""
    if language_data is None:
        return None

    def _value(name, default):
        if isinstance(language_data, dict):
            return language_data.get(name, default)
        return getattr(language_data, name, default)

    language_char_counts = {}
    for span in _value("spans", []) or []:
        if isinstance(span, dict):
            language = str(span.get("language", "und") or "und")
            char_count = int(span.get("char_count", 0) or 0)
        else:
            language = str(getattr(span, "language", "und") or "und")
            char_count = int(getattr(span, "char_count", 0) or 0)
        if char_count > 0:
            language_char_counts[language] = (
                language_char_counts.get(language, 0) + char_count
            )

    if not language_char_counts:
        provided_counts = _value("language_char_counts", {}) or {}
        language_char_counts = {
            str(key): int(value)
            for key, value in provided_counts.items()
            if int(value) > 0
        }

    return {
        "page_num": int(_value("page_num", 0) or 0),
        "primary_language": str(_value("primary_language", "") or ""),
        "primary_confidence": float(_value("primary_confidence", 0.0) or 0.0),
        "languages_detected": [
            str(value) for value in (_value("languages_detected", []) or [])
        ],
        "language_char_shares": {
            str(key): float(value)
            for key, value in (_value("language_char_shares", {}) or {}).items()
        },
        "language_char_counts": language_char_counts,
        "scripts_detected": [
            str(value) for value in (_value("scripts_detected", []) or [])
        ],
        "mixed_script": bool(_value("mixed_script", False)),
        "span_count": int(_value("span_count", 0) or 0),
        "spans_labeled": int(_value("spans_labeled", 0) or 0),
    }


def _language_model_provenance():
    """Return explicit, non-empty FastText/tokenizer identity fields."""
    model_sha = str(getattr(lang_model, "_sha256", "") or "").strip().lower()
    if _is_valid_checkpoint_sha256(model_sha):
        detector_identity = model_sha
        fasttext_identity = model_sha
    else:
        detector_identity = "unavailable:model_digest_not_attached"
        fasttext_identity = "unavailable:model_digest_not_attached"
    return {
        "detector_model_sha256": detector_identity,
        # FastText performs its own character n-gram preprocessing; there is
        # no separately versioned tokenizer artifact to hash.
        "tokenizer_sha256": "not_applicable:fasttext_internal",
        "fasttext_model_sha256": fasttext_identity,
    }


# Page-local profile keys are persisted for forensic reuse and must be present
# and type-valid, but are not re-compared against a hypothetical re-route of the
# same page (the checkpoint already records what actually ran).
_CHECKPOINT_PAGE_LOCAL_PROFILE_KEYS = frozenset(
    {
        "actual_render_dpi",
        "page_routing_target",
        "execution_device",
        "execution_engine",
        "gpu_optimization_applied",
        "preprocessing_applied",
    }
)
_VALID_CHECKPOINT_EXECUTION_DEVICES = frozenset({"cpu", "gpu", "none"})
_VALID_CHECKPOINT_EXECUTION_ENGINES = frozenset(
    {
        "PaddleOCR",
        "Tesseract",
        "EasyOCR",
        "ImageOnly",
    }
)
_VALID_CHECKPOINT_PIPELINE_STATUSES = frozenset(
    {
        PageStatus.PADDLE.value,
        PageStatus.TESSERACT.value,
        PageStatus.EASYOCR.value,
        PageStatus.IMAGE_ONLY.value,
        PageStatus.CRITICAL_FAILED.value,
    }
)


def _task_render_dpi(task):
    """Return the DPI that produced the image currently attached to *task*."""
    try:
        return int(getattr(task, "render_dpi", DPI) or DPI)
    except (TypeError, ValueError):
        return int(DPI)


def _allowed_actual_render_dpis():
    """Return the closed set of DPIs a reusable checkpoint may claim."""
    allowed = {int(DPI)}
    if _DPI_ESCALATION_AVAILABLE:
        try:
            # Independently observed quality/coverage recovery is part of the
            # default bounded path even when legacy escalation is disabled.
            allowed.update(
                int(value) for value in DPI_SCHEDULE
                if ENABLE_DPI_ESCALATION or DPI < int(value) <= 600
            )
        except Exception:
            pass
    return allowed


def _allowed_page_routing_targets():
    """Return the closed set of page-routing labels a checkpoint may claim."""
    allowed = {"default"}
    try:
        from ocr_local.infra.page_routing import RoutingTarget

        allowed.update(str(target.value) for target in RoutingTarget)
    except Exception:
        pass
    return allowed


def _checkpoint_processing_profile(
    *,
    actual_render_dpi=None,
    page_routing_target=None,
    execution_device=None,
    execution_engine=None,
    gpu_optimization_applied=None,
    preprocessing_applied=None,
):
    """Return settings that materially determine a reusable page artifact.

    Global/runtime fields must match the current process for safe resume/cache
    reuse.  Page-local fields record what actually executed for this page and
    must be present on every new checkpoint (missing values fail closed).
    Production writers must pass complete page-local fields; defaults exist only
    so the expected-profile template has a well-formed shape for structural
    validation of stored page-local keys.
    """
    return {
        "pipeline_version": str(PIPELINE_VERSION),
        "dpi": int(DPI),
        "jpeg_quality": int(JPEG_QUALITY),
        "background_format": "PNG",
        "preprocessing_enabled": bool(ENABLE_PREPROCESSING),
        "preprocessing_level": str(PREPROCESSING_LEVEL),
        "dpi_escalation_enabled": bool(ENABLE_DPI_ESCALATION),
        "dpi_confidence_threshold": float(DPI_CONFIDENCE_THRESHOLD),
        "engine_registry_enabled": bool(ENABLE_ENGINE_REGISTRY),
        "inference_backend": os.environ.get("OCR_INFERENCE_BACKEND", "paddle"),
        "paddle_drop_score": OCR_DROP_SCORE,
        "ocr_engine_selection": os.environ.get("OCR_ENGINE_SELECTION", "paddle")
        .lower()
        .strip(),
        "paddle_device": str(_DEFAULT_PADDLE_DEVICE),
        "per_span_language_enabled": bool(
            ENABLE_PER_SPAN_LANGUAGE and _LANGUAGE_DETECTION_AVAILABLE
        ),
        "page_routing_enabled": bool(ENABLE_PAGE_ROUTING and _page_router is not None),
        # Availability means the batch preprocessor is live, not merely that the
        # optimizer object was constructed at import time.
        "gpu_optimization_enabled": bool(
            ENABLE_GPU_OPTIMIZATION and _batch_preprocessor is not None
        ),
        # Both caches can change which pixels / inference path produce a page
        # artifact.  Persist their effective enablement so a restart never
        # silently reuses checkpoints made under a materially different path.
        "page_cache_enabled": bool(ENABLE_PAGE_CACHE),
        "adaptive_batch_enabled": bool(ENABLE_ADAPTIVE_BATCH),
        "actual_render_dpi": int(
            actual_render_dpi if actual_render_dpi is not None else DPI
        ),
        "page_routing_target": str(
            page_routing_target if page_routing_target is not None else "default"
        ),
        "execution_device": str(
            execution_device if execution_device is not None else "cpu"
        ),
        "execution_engine": str(
            execution_engine if execution_engine is not None else "PaddleOCR"
        ),
        "gpu_optimization_applied": bool(
            False if gpu_optimization_applied is None else gpu_optimization_applied
        ),
        "preprocessing_applied": bool(
            False if preprocessing_applied is None else preprocessing_applied
        ),
    }


def _strict_checkpoint_confidence(value):
    """Return a finite non-bool confidence in [0, 1] or raise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Invalid checkpoint ocr_confidence type")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("Checkpoint ocr_confidence is out of range")
    return confidence


def _strict_checkpoint_text_length(value):
    """Return a non-bool int >= 0 or raise."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Invalid checkpoint text_length")
    return value


def _checkpoint_text_fields_issue(has_text, text_length, text_sha256):
    """Return why has_text/text_length/text_sha256 are contradictory."""
    if not isinstance(has_text, bool):
        return "Invalid checkpoint text flag"
    try:
        text_length = _strict_checkpoint_text_length(text_length)
    except ValueError as exc:
        return str(exc)
    if not isinstance(text_sha256, str) or not _CHECKPOINT_TEXT_SHA256_RE.fullmatch(
        text_sha256
    ):
        return "Invalid checkpoint text_sha256"
    if has_text:
        if text_length <= 0:
            return "has_text true requires text_length > 0"
        if text_sha256 == _EMPTY_TEXT_SHA256:
            return "has_text true requires non-empty text_sha256"
    else:
        if text_length != 0:
            return "has_text false requires text_length == 0"
        if text_sha256 != _EMPTY_TEXT_SHA256:
            return "has_text false requires empty-text text_sha256"
    return None


def _checkpoint_status_consistency_issue(payload):
    """Return why status/method/validation/profile fields disagree."""
    status = payload.get("status")
    validation_status = payload.get("validation_status")
    ocr_method = payload.get("ocr_method")
    profile = payload.get("processing_profile")
    if not isinstance(profile, dict):
        return "Checkpoint processing profile is invalid"
    device = profile.get("execution_device")
    engine = profile.get("execution_engine")
    has_text = payload.get("has_text")
    text_length = payload.get("text_length")
    text_sha256 = payload.get("text_sha256")

    text_issue = _checkpoint_text_fields_issue(has_text, text_length, text_sha256)
    if text_issue:
        return text_issue

    ocr_language = payload.get("ocr_language")
    model_language = payload.get("model_language")
    gpu_applied = profile.get("gpu_optimization_applied")
    preproc_applied = profile.get("preprocessing_applied")

    is_paddle_status = isinstance(status, str) and (
        status == PageStatus.PADDLE.value or status.startswith("Paddle-")
    )
    is_ocr_success_status = is_paddle_status or status in (
        PageStatus.TESSERACT.value,
        PageStatus.EASYOCR.value,
    )
    # ocr_language is the existing assembly alias of model_language. Old v3
    # bytes omit model_language; bind Paddle-<lang> to the model when present.
    if model_language is None:
        bind_language = ocr_language
    elif not isinstance(model_language, str):
        return "Invalid checkpoint model language"
    else:
        bind_language = model_language
        if is_ocr_success_status and model_language != ocr_language:
            return "ocr_language must equal model_language"
    if is_paddle_status:
        if validation_status != "ok":
            return "Paddle status requires validation_status ok"
        if ocr_method != "PaddleOCR":
            return "Paddle status requires ocr_method PaddleOCR"
        if engine != "PaddleOCR":
            return "Paddle status requires execution_engine PaddleOCR"
        if device not in ("cpu", "gpu"):
            return "Paddle status requires execution_device cpu or gpu"
        if not isinstance(ocr_language, str) or not ocr_language.strip():
            return "Paddle status requires non-empty ocr_language"
        # Dynamic two-pass status Paddle-<lang> binds the weights actually used.
        if status.startswith("Paddle-"):
            lang_suffix = status[len("Paddle-") :]
            if not lang_suffix or not lang_suffix.strip():
                return "Paddle-<lang> status requires non-empty language suffix"
            if lang_suffix != bind_language:
                return "Paddle-<lang> status suffix must match ocr_language"
        # Blank pages may legitimately carry has_text=False; empty language
        # or non-OCR engine labels above remain hard failures.
        return None

    if status == PageStatus.TESSERACT.value:
        if validation_status != "fallback":
            return "Tesseract status requires validation_status fallback"
        if ocr_method != "Tesseract":
            return "Tesseract status requires ocr_method Tesseract"
        if engine != "Tesseract":
            return "Tesseract status requires execution_engine Tesseract"
        if device != "cpu":
            return "Tesseract status requires execution_device cpu"
        if not isinstance(ocr_language, str) or not ocr_language.strip():
            return "Tesseract status requires non-empty ocr_language"
        return None

    if status == PageStatus.EASYOCR.value:
        if validation_status != "fallback":
            return "EasyOCR status requires validation_status fallback"
        if ocr_method != "EasyOCR":
            return "EasyOCR status requires ocr_method EasyOCR"
        if engine != "EasyOCR":
            return "EasyOCR status requires execution_engine EasyOCR"
        if device != "cpu":
            return "EasyOCR status requires execution_device cpu"
        if not isinstance(ocr_language, str) or not ocr_language.strip():
            return "EasyOCR status requires non-empty ocr_language"
        return None

    if status == PageStatus.IMAGE_ONLY.value:
        if validation_status != "image_only":
            return "ImageOnly status requires validation_status image_only"
        if ocr_method != "ImageOnly":
            return "ImageOnly status requires ocr_method ImageOnly"
        if engine != "ImageOnly":
            return "ImageOnly status requires execution_engine ImageOnly"
        if device != "none":
            return "ImageOnly status requires execution_device none"
        if has_text or text_length != 0 or text_sha256 != _EMPTY_TEXT_SHA256:
            return "ImageOnly status requires no text"
        if gpu_applied is not False:
            return "ImageOnly status requires gpu_optimization_applied false"
        if preproc_applied is not False:
            return "ImageOnly status requires preprocessing_applied false"
        return None

    if status == PageStatus.CRITICAL_FAILED.value:
        if validation_status != "failed":
            return "CRITICAL_FAILED status requires validation_status failed"
        if ocr_method != "ImageOnly":
            return "CRITICAL_FAILED status requires ocr_method ImageOnly"
        if engine != "ImageOnly":
            return "CRITICAL_FAILED status requires execution_engine ImageOnly"
        if device != "none":
            return "CRITICAL_FAILED status requires execution_device none"
        if has_text or text_length != 0 or text_sha256 != _EMPTY_TEXT_SHA256:
            return "CRITICAL_FAILED status requires no text"
        if gpu_applied is not False:
            return "CRITICAL_FAILED status requires gpu_optimization_applied false"
        if preproc_applied is not False:
            return "CRITICAL_FAILED status requires preprocessing_applied false"
        return None

    return f"Unsupported checkpoint status: {status!r}"


def _embed_page_checkpoint_metadata(
    pdf_bytes,
    *,
    document_id,
    page_num,
    status,
    ocr_method,
    ocr_language,
    ocr_confidence,
    text_length,
    has_text,
    language_data=None,
    processing_profile=None,
    canonical_text=None,
    detected_language=None,
    detected_language_confidence=None,
    source_prior_label=None,
    model_language=None,
    page_geometry=None,
    geometry_issues=None,
    review_reasons=None,
    candidate_reviews=None,
    coverage_assessment=None,
):
    """Embed recovery metadata in the same atomically-written page PDF.

    Keeping the metadata inside the checkpoint avoids a two-file commit window:
    after a crash the PDF and the OCR provenance are either both present or both
    absent.  The temporary page PDF is pipeline-owned, so its subject field is
    reserved for this compact recovery record.  ``canonical_text`` is content-bound
    via UTF-8 SHA-256 (including the empty-text digest).
    """
    status_value = status.value if isinstance(status, enum.Enum) else str(status)
    method_value = (
        ocr_method.value if isinstance(ocr_method, enum.Enum) else str(ocr_method)
    )
    if not isinstance(processing_profile, dict):
        raise ValueError(
            "processing_profile is required; refusing empty page-local provenance"
        )
    if canonical_text is None:
        raise ValueError(
            "canonical_text is required for content-bound checkpoint metadata"
        )
    if not isinstance(canonical_text, str):
        raise ValueError("canonical_text must be a string")
    try:
        text_length = _strict_checkpoint_text_length(text_length)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(has_text, bool):
        raise ValueError("has_text must be a bool")
    if len(canonical_text) != text_length or bool(canonical_text) is not has_text:
        raise ValueError("canonical_text inconsistent with text_length/has_text")
    confidence = _strict_checkpoint_confidence(
        0.0 if ocr_confidence is None else ocr_confidence
    )
    profile = processing_profile
    page_local_issue = _checkpoint_page_local_profile_issue(profile)
    if page_local_issue:
        raise ValueError(page_local_issue)
    per_span_language_enabled = bool(ENABLE_PER_SPAN_LANGUAGE)
    language_analysis_complete = bool(
        not per_span_language_enabled or not has_text or language_data is not None
    )
    if not language_analysis_complete:
        raise ValueError(
            "per-span language analysis is enabled but text-bearing page "
            "analysis is incomplete"
        )
    if not per_span_language_enabled and language_data is not None:
        raise ValueError(
            "language_data cannot be persisted when per-span analysis is disabled"
        )
    text_sha256 = _canonical_text_sha256(canonical_text)
    ocr_lang = str(ocr_language or "")
    if model_language is None:
        model_lang = ocr_lang
    else:
        model_lang = str(model_language or "")
    if detected_language is None:
        detected_lang = ""
    else:
        detected_lang = str(detected_language or "")
    payload = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA_VERSION,
        "document_id": str(document_id),
        "page_num": int(page_num),
        "status": status_value,
        "ocr_method": method_value,
        "ocr_language": ocr_lang,
        "detected_language": detected_lang,
        "detected_language_confidence": (
            None if detected_language_confidence is None
            else _strict_checkpoint_confidence(detected_language_confidence)
        ),
        "source_prior_label": str(source_prior_label or ""),
        "model_language": model_lang,
        "ocr_confidence": confidence,
        "text_length": text_length,
        "has_text": has_text,
        "text_sha256": text_sha256,
        "validation_status": _checkpoint_validation_status(status, has_text),
        "language_analysis_complete": language_analysis_complete,
        "per_span_language_enabled": per_span_language_enabled,
        "language_data": _compact_page_language(language_data),
        "processing_profile": profile,
    }
    # Production callers pass geometry captured from the authorized source.
    # Direct checkpoint helpers used by migration/tests can only assert their
    # own local page extent and are marked accordingly.
    if page_geometry is None:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as local_pdf:
            local_page = local_pdf[0]
            page_geometry = PageGeometry(
                local_page.rect.width,
                local_page.rect.height,
                "checkpoint-local",
            )
    payload["page_geometry"] = asdict(page_geometry)
    payload["geometry_issues"] = list(geometry_issues or [])
    payload["review_reasons"] = list(review_reasons or [])
    payload["candidate_reviews"] = list(candidate_reviews or [])
    payload["coverage_assessment"] = dict(coverage_assessment or {})
    payload["needs_review"] = bool(
        payload["geometry_issues"] or payload["review_reasons"] or payload["candidate_reviews"]
        or payload["coverage_assessment"].get("needs_review")
    ) or (
        page_geometry.source_kind in ("checkpoint-local", "raster-dpi-fallback")
    )
    consistency_issue = _checkpoint_status_consistency_issue(payload)
    if consistency_issue:
        raise ValueError(consistency_issue)
    # Run the same closed semantic envelope used by resume/cache before bytes
    # are embedded.  Type-shaped but empty language summaries must never be
    # allowed to claim analysis completion on a text-bearing checkpoint.
    payload = _parse_page_checkpoint_metadata_payload(
        payload,
        document_id=document_id,
        page_num=page_num,
    )
    encoded = CHECKPOINT_METADATA_SUBJECT_PREFIX + json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    with fitz.open(stream=pdf_bytes, filetype="pdf") as checkpoint:
        if checkpoint.page_count != 1 or checkpoint.needs_pass:
            raise ValueError("Page checkpoint must be a readable single-page PDF")
        checkpoint_page = checkpoint.load_page(0)
        if (
            abs(checkpoint_page.rect.width - page_geometry.width_pt) > 0.5
            or abs(checkpoint_page.rect.height - page_geometry.height_pt) > 0.5
        ):
            raise ValueError("checkpoint page dimensions differ from source geometry")
        current = checkpoint.metadata or {}
        allowed_keys = (
            "title",
            "author",
            "subject",
            "keywords",
            "creator",
            "producer",
            "creationDate",
            "modDate",
            "trapped",
        )
        metadata = {key: str(current.get(key, "") or "") for key in allowed_keys}
        metadata["subject"] = encoded
        checkpoint.set_metadata(metadata)
        return checkpoint.tobytes()


def _parse_page_checkpoint_metadata_payload(payload, *, document_id, page_num):
    """Validate a decoded checkpoint metadata object (fail closed)."""
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint metadata must be a JSON object")
    if payload.get("schema_version") != CHECKPOINT_METADATA_SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint metadata schema")
    try:
        geometry = PageGeometry(**payload["page_geometry"])
        if geometry.source_kind not in ("pdf", "image", "checkpoint-local", "raster-dpi-fallback"):
            raise ValueError("invalid source geometry kind")
        if geometry.source_kind == "pdf" and geometry.source_cropbox is None:
            raise ValueError("PDF checkpoint missing source cropbox")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid checkpoint page geometry") from exc
    issues = payload.get("geometry_issues")
    reasons = payload.get("review_reasons")
    candidates = payload.get("candidate_reviews")
    coverage = payload.get("coverage_assessment", {})
    detected_confidence = payload.get("detected_language_confidence")
    if detected_confidence is not None:
        _strict_checkpoint_confidence(detected_confidence)
    if not isinstance(payload.get("source_prior_label", ""), str):
        raise ValueError("Invalid checkpoint source prior label")
    if not isinstance(coverage, dict):
        raise ValueError("Invalid checkpoint independent coverage")
    if coverage:
        coverage_value = coverage.get("coverage")
        if coverage.get("status") not in {"measured", "unknown"} or not isinstance(
            coverage.get("needs_review"), bool
        ) or not isinstance(coverage.get("reason"), str):
            raise ValueError("Invalid checkpoint independent coverage fields")
        if coverage_value is not None and (
            isinstance(coverage_value, bool)
            or not isinstance(coverage_value, (int, float))
            or not math.isfinite(coverage_value)
            or not 0 <= coverage_value <= 1
        ):
            raise ValueError("Invalid checkpoint independent coverage fraction")
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        raise ValueError("Invalid checkpoint review reasons")
    if not isinstance(candidates, list) or not all(
        isinstance(item, dict) and isinstance(item.get("text"), str)
        and isinstance(item.get("decision"), str) for item in candidates
    ):
        raise ValueError("Invalid checkpoint candidate reviews")
    expected_review = bool(issues or reasons or candidates or coverage.get("needs_review")) or geometry.source_kind in (
        "checkpoint-local", "raster-dpi-fallback"
    )
    if not isinstance(issues, list) or payload.get("needs_review") is not expected_review:
        raise ValueError("Invalid checkpoint geometry review state")
    for issue in issues:
        if (
            not isinstance(issue, dict)
            or not isinstance(issue.get("index"), int)
            or not isinstance(issue.get("text"), str)
            or not isinstance(issue.get("reason"), str)
        ):
            raise ValueError("Invalid checkpoint geometry issue")
    if payload.get("document_id") != str(document_id):
        raise ValueError("Checkpoint metadata document mismatch")
    if payload.get("page_num") != int(page_num):
        raise ValueError("Checkpoint metadata page mismatch")

    validation_status = payload.get("validation_status")
    if validation_status not in {
        "ok",
        "fallback",
        "image_only",
        "failed",
    }:
        raise ValueError("Invalid checkpoint validation status")
    status_value = payload.get("status")
    if not isinstance(status_value, str) or not status_value:
        raise ValueError("Invalid checkpoint pipeline status")
    # Paddle-<lang> is a dynamic status; other statuses must be closed-set.
    if not (
        status_value == PageStatus.PADDLE.value
        or status_value.startswith("Paddle-")
        or status_value in _VALID_CHECKPOINT_PIPELINE_STATUSES
    ):
        raise ValueError(f"Unsupported checkpoint status: {status_value!r}")
    if not isinstance(payload.get("ocr_method"), str):
        raise ValueError("Invalid checkpoint OCR method")
    if not isinstance(payload.get("ocr_language"), str):
        raise ValueError("Invalid checkpoint OCR language")
    # Pre-B v3 resume: model language is the OCR alias; never invent detected "en".
    if "model_language" not in payload:
        payload["model_language"] = payload["ocr_language"]
    elif not isinstance(payload.get("model_language"), str):
        raise ValueError("Invalid checkpoint model language")
    if "detected_language" not in payload:
        payload["detected_language"] = ""
    elif not isinstance(payload.get("detected_language"), str):
        raise ValueError("Invalid checkpoint detected language")
    if not isinstance(payload.get("has_text"), bool):
        raise ValueError("Invalid checkpoint text flag")
    if not isinstance(payload.get("language_analysis_complete"), bool):
        raise ValueError("Checkpoint language analysis flag is invalid")
    if not isinstance(payload.get("per_span_language_enabled"), bool):
        raise ValueError("Checkpoint per-span language flag is invalid")
    if not isinstance(payload.get("processing_profile"), dict):
        raise ValueError("Checkpoint processing profile is invalid")
    page_local_issue = _checkpoint_page_local_profile_issue(
        payload.get("processing_profile"),
    )
    if page_local_issue:
        raise ValueError(page_local_issue)

    confidence = _strict_checkpoint_confidence(payload.get("ocr_confidence"))
    text_length = _strict_checkpoint_text_length(payload.get("text_length"))
    text_sha256 = payload.get("text_sha256")
    text_issue = _checkpoint_text_fields_issue(
        payload.get("has_text"),
        text_length,
        text_sha256,
    )
    if text_issue:
        raise ValueError(text_issue)

    language_data = payload.get("language_data")
    per_span_language_enabled = payload["per_span_language_enabled"]
    language_analysis_complete = payload["language_analysis_complete"]
    has_text = payload["has_text"]
    if language_analysis_complete is not True:
        raise ValueError("Checkpoint language analysis is incomplete")
    if per_span_language_enabled and has_text and language_data is None:
        raise ValueError(
            "Text-bearing checkpoint is missing enabled per-span language data"
        )
    if not per_span_language_enabled and language_data is not None:
        raise ValueError(
            "Checkpoint has language data while per-span analysis is disabled"
        )
    if not has_text and language_data is not None:
        raise ValueError("Text-free checkpoint cannot contain language data")
    if language_data is not None:
        if not isinstance(language_data, dict):
            raise ValueError("Invalid checkpoint language metadata")
        if language_data.get("page_num") != int(page_num):
            raise ValueError("Checkpoint language metadata page mismatch")
        if not isinstance(language_data.get("primary_language"), str):
            raise ValueError("Invalid checkpoint primary language")
        primary_confidence = _strict_checkpoint_confidence(
            language_data.get("primary_confidence")
        )

        languages_detected = language_data.get("languages_detected")
        if not isinstance(languages_detected, list) or not all(
            isinstance(value, str) for value in languages_detected
        ):
            raise ValueError("Invalid checkpoint detected languages")

        language_char_shares = language_data.get("language_char_shares")
        if not isinstance(language_char_shares, dict):
            raise ValueError("Invalid checkpoint language shares")
        # Validate original values before float conversion: bool and numeric
        # strings must fail; only finite int/float in [0, 1] normalize.
        normalized_shares = {}
        for key, value in language_char_shares.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Invalid checkpoint language share type")
            share = float(value)
            if not math.isfinite(share) or not 0.0 <= share <= 1.0:
                raise ValueError("Checkpoint language shares are out of range")
            normalized_shares[str(key)] = share

        language_char_counts = language_data.get("language_char_counts")
        if not isinstance(language_char_counts, dict):
            raise ValueError("Invalid checkpoint language character counts")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in language_char_counts.values()
        ):
            raise ValueError("Invalid checkpoint language character counts")
        normalized_counts = {
            str(key): value for key, value in language_char_counts.items()
        }
        if any(value < 0 for value in normalized_counts.values()):
            raise ValueError("Checkpoint language character counts are negative")

        scripts_detected = language_data.get("scripts_detected")
        if not isinstance(scripts_detected, list) or not all(
            isinstance(value, str) for value in scripts_detected
        ):
            raise ValueError("Invalid checkpoint detected scripts")
        if not isinstance(language_data.get("mixed_script"), bool):
            raise ValueError("Invalid checkpoint mixed-script flag")

        span_count = language_data.get("span_count")
        spans_labeled = language_data.get("spans_labeled")
        if (
            isinstance(span_count, bool)
            or not isinstance(span_count, int)
            or span_count < 0
        ):
            raise ValueError("Invalid checkpoint span count")
        if (
            isinstance(spans_labeled, bool)
            or not isinstance(spans_labeled, int)
            or spans_labeled < 0
            or spans_labeled > span_count
        ):
            raise ValueError("Invalid checkpoint labeled span count")

        language_data["primary_confidence"] = primary_confidence
        language_data["language_char_shares"] = normalized_shares
        language_data["language_char_counts"] = normalized_counts

        primary_language = language_data["primary_language"]
        if not primary_language:
            raise ValueError("Checkpoint primary language is empty")
        if len(set(languages_detected)) != len(languages_detected):
            raise ValueError("Checkpoint detected languages contain duplicates")
        if len(set(scripts_detected)) != len(scripts_detected):
            raise ValueError("Checkpoint detected scripts contain duplicates")
        if language_data["mixed_script"] != (len(scripts_detected) >= 2):
            raise ValueError("Checkpoint mixed-script flag is inconsistent")
        if has_text:
            if span_count < 1:
                raise ValueError(
                    "Text-bearing checkpoint has no analyzed language spans"
                )
            if sum(normalized_counts.values()) < 1:
                raise ValueError(
                    "Text-bearing checkpoint has no analyzed character counts"
                )
            if spans_labeled == 0:
                if (
                    primary_language != "und"
                    or languages_detected
                    or normalized_shares
                    or primary_confidence != 0.0
                ):
                    raise ValueError(
                        "Unlabeled checkpoint language summary is inconsistent"
                    )
            else:
                if primary_language == "und":
                    raise ValueError(
                        "Labeled checkpoint cannot use und as primary language"
                    )
                if primary_language not in languages_detected:
                    raise ValueError("Checkpoint primary language is not detected")
                if primary_language not in normalized_shares:
                    raise ValueError(
                        "Checkpoint primary language has no character share"
                    )
                if normalized_counts.get(primary_language, 0) < 1:
                    raise ValueError(
                        "Checkpoint primary language has no character count"
                    )
                if not math.isclose(
                    sum(normalized_shares.values()),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-5,
                ):
                    raise ValueError(
                        "Checkpoint language character shares do not sum to one"
                    )
                if not set(languages_detected).issubset(normalized_shares):
                    raise ValueError(
                        "Checkpoint detected language has no character share"
                    )
                if not set(languages_detected).issubset(normalized_counts):
                    raise ValueError(
                        "Checkpoint detected language has no character count"
                    )

    payload["ocr_confidence"] = confidence
    payload["text_length"] = text_length
    payload["text_sha256"] = text_sha256
    consistency_issue = _checkpoint_status_consistency_issue(payload)
    if consistency_issue:
        raise ValueError(consistency_issue)
    return payload


def _load_page_checkpoint_metadata_from_document(doc, *, document_id, page_num):
    """Load checkpoint provenance from an already-open fitz document."""
    subject = str((doc.metadata or {}).get("subject", "") or "")
    if not subject.startswith(CHECKPOINT_METADATA_SUBJECT_PREFIX):
        # Older prefixes (V1-V3) contain pixel-space pages and cannot resume.
        return None
    try:
        payload = json.loads(subject[len(CHECKPOINT_METADATA_SUBJECT_PREFIX) :])
    except (TypeError, ValueError) as exc:
        raise ValueError("Malformed checkpoint metadata JSON") from exc
    parsed = _parse_page_checkpoint_metadata_payload(
        payload,
        document_id=document_id,
        page_num=page_num,
    )
    if doc.page_count != 1:
        raise ValueError("checkpoint must contain exactly one page")
    geometry = PageGeometry(**parsed["page_geometry"])
    page_rect = doc[0].rect
    if (
        abs(page_rect.width - geometry.width_pt) > 0.5
        or abs(page_rect.height - geometry.height_pt) > 0.5
    ):
        raise ValueError("checkpoint page dimensions differ from recorded geometry")
    return parsed


def _load_page_checkpoint_metadata(path, *, document_id, page_num):
    """Load and validate embedded OCR provenance from a page checkpoint.

    A checkpoint without the V3 marker is provenance-free legacy state and
    returns ``None``.  Callers must reject that via
    :func:`_checkpoint_resume_incompatibility` so resume/cache always
    reprocess.  A marked but malformed or mismatched checkpoint raises so the
    scheduler reprocesses the page.
    """
    with fitz.open(path) as checkpoint:
        return _load_page_checkpoint_metadata_from_document(
            checkpoint,
            document_id=document_id,
            page_num=page_num,
        )


def _restore_checkpoint_page_language(checkpoint_metadata):
    """Rebuild the compact PageLanguage needed for document aggregation."""
    summary = (
        checkpoint_metadata.get("language_data")
        if isinstance(checkpoint_metadata, dict)
        else None
    )
    if summary is None or not _LANGUAGE_DETECTION_AVAILABLE:
        return None
    recovered_spans = [
        SpanLanguage(
            bbox=[],
            text_sample="",
            language=language,
            confidence=float(summary["primary_confidence"]),
            script="other",
            detection_method="inherited_page",
            char_count=int(char_count),
        )
        for language, char_count in summary["language_char_counts"].items()
        if int(char_count) > 0
    ]
    return PageLanguage(
        page_num=int(summary["page_num"]),
        primary_language=str(summary["primary_language"]),
        primary_confidence=float(summary["primary_confidence"]),
        languages_detected=list(summary["languages_detected"]),
        language_char_shares=dict(summary["language_char_shares"]),
        scripts_detected=list(summary["scripts_detected"]),
        mixed_script=bool(summary["mixed_script"]),
        span_count=int(summary["span_count"]),
        spans_labeled=int(summary["spans_labeled"]),
        spans=recovered_spans,
    )


def _checkpoint_page_local_profile_issue(profile):
    """Return why page-local realized fields are structurally invalid."""
    if not isinstance(profile, dict):
        return "checkpoint processing profile is invalid"
    for key in _CHECKPOINT_PAGE_LOCAL_PROFILE_KEYS:
        if key not in profile:
            return f"checkpoint processing profile missing {key}"
    dpi_raw = profile.get("actual_render_dpi")
    if isinstance(dpi_raw, bool) or not isinstance(dpi_raw, int):
        return "checkpoint actual_render_dpi is unverifiable"
    dpi_value = dpi_raw
    if dpi_value not in _allowed_actual_render_dpis():
        return "checkpoint actual_render_dpi is outside closed set"
    routing_target = profile.get("page_routing_target")
    if (
        not isinstance(routing_target, str)
        or routing_target not in _allowed_page_routing_targets()
    ):
        return "checkpoint page_routing_target is outside closed set"
    if profile.get("execution_device") not in _VALID_CHECKPOINT_EXECUTION_DEVICES:
        return "checkpoint execution_device is outside closed set"
    if profile.get("execution_engine") not in _VALID_CHECKPOINT_EXECUTION_ENGINES:
        return "checkpoint execution_engine is outside closed set"
    if not isinstance(profile.get("gpu_optimization_applied"), bool):
        return "checkpoint gpu_optimization_applied is unverifiable"
    if not isinstance(profile.get("preprocessing_applied"), bool):
        return "checkpoint preprocessing_applied is unverifiable"
    # Applied implies the feature was available on that page.  enabled=True with
    # applied=False remains valid (available but not realized).  Snapshot embed
    # and reuse finalization both call this helper.
    if (
        profile.get("gpu_optimization_applied") is True
        and profile.get("gpu_optimization_enabled") is not True
    ):
        return (
            "checkpoint gpu_optimization_applied requires gpu_optimization_enabled true"
        )
    if (
        profile.get("preprocessing_applied") is True
        and profile.get("preprocessing_enabled") is not True
    ):
        return "checkpoint preprocessing_applied requires preprocessing_enabled true"
    return None


def _checkpoint_profile_incompatibility(stored_profile):
    """Return why a stored processing profile cannot be reused safely."""
    if not isinstance(stored_profile, dict):
        return "checkpoint processing profile is invalid"

    page_local_issue = _checkpoint_page_local_profile_issue(stored_profile)
    if page_local_issue:
        return page_local_issue

    expected = _checkpoint_processing_profile()
    for key, expected_value in expected.items():
        if key not in stored_profile:
            return f"checkpoint processing profile missing {key}"
        if key in _CHECKPOINT_PAGE_LOCAL_PROFILE_KEYS:
            # Page-local realized fields are structurally validated above, not
            # compared to a hypothetical re-route of the same page.
            continue
        if stored_profile[key] != expected_value:
            return (
                "checkpoint processing profile differs from the current runtime "
                f"({key})"
            )
    return None


def _checkpoint_resume_incompatibility(checkpoint_metadata):
    """Return why this checkpoint must be discarded and the page reprocessed.

    Provenance-free legacy checkpoints are always rejected, regardless of
    optional feature flags.  Missing or unverifiable material profile fields
    also fail closed.
    """
    if not isinstance(checkpoint_metadata, dict):
        return "legacy checkpoint has no provenance"
    # Review is an output disposition, not an artifact-integrity failure.
    # Unknown language or uncertain visual coverage can still have a valid
    # searchable checkpoint; only unplaceable text geometry is unsafe to
    # publish as spatially correct OCR.
    if checkpoint_metadata.get("geometry_issues"):
        return "checkpoint has unplaced OCR spans requiring review"
    if checkpoint_metadata.get("page_geometry", {}).get("source_kind") in (
        "checkpoint-local", "raster-dpi-fallback"
    ):
        return "checkpoint geometry was not bound to source"

    unsupported_page_features = (
        (ENABLE_DOCUMENT_INTELLIGENCE, "document intelligence"),
        (ENABLE_HANDWRITING, "handwriting analysis"),
        (ENABLE_SIGNATURE_VERIFICATION, "signature analysis"),
        (ENABLE_VERTICAL_TEXT, "vertical-text analysis"),
        (ENABLE_TABLE_FALLBACK, "table fallback analysis"),
    )
    for enabled, label in unsupported_page_features:
        if enabled:
            return f"{label} state is not persisted in page checkpoints"
    if ENABLE_PER_SPAN_LANGUAGE:
        checkpoint_has_text = checkpoint_metadata.get("has_text") is True
        if checkpoint_has_text and not _LANGUAGE_DETECTION_AVAILABLE:
            return "per-span language analysis runtime is unavailable"
        if checkpoint_metadata.get("per_span_language_enabled") is not True:
            return "checkpoint was produced without per-span language analysis"
        if checkpoint_has_text and LANGUAGE_INCLUDE_SPANS:
            return "verbose language spans require page reprocessing"
        if checkpoint_metadata.get("language_analysis_complete") is not True:
            return "checkpoint language-analysis state is incomplete"

    profile_issue = _checkpoint_profile_incompatibility(
        checkpoint_metadata.get("processing_profile")
    )
    if profile_issue:
        return profile_issue
    return None


_failure_report_lock = threading.Lock()

# Per-document in-process RLocks serialize checkpoint/cache writes with
# final open+validate+insert.  They do NOT protect against a hostile external
# process mutating files on disk.
_checkpoint_artifact_locks_guard = threading.Lock()
_checkpoint_artifact_locks: dict[str, threading.RLock] = {}


def _get_checkpoint_artifact_lock(doc_id: str) -> threading.RLock:
    """Return the per-document checkpoint artifact RLock."""
    key = str(doc_id)
    with _checkpoint_artifact_locks_guard:
        lock = _checkpoint_artifact_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _checkpoint_artifact_locks[key] = lock
        return lock


def _release_checkpoint_artifact_lock(doc_id: str) -> None:
    """Drop the per-document lock entry after the document leaves the registry."""
    key = str(doc_id)
    with _checkpoint_artifact_locks_guard:
        _checkpoint_artifact_locks.pop(key, None)


def log_failure(path, page, error):
    """Writes failure to CSV report."""
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clean_err = str(error).replace(",", ";").replace("\n", " ")
        with _failure_report_lock:
            with open(FAILURE_REPORT, "a", encoding="utf-8") as f:
                f.write(f"{ts},{path},{page},{clean_err}\n")
    except Exception as log_err:
        logger.warning(
            "Failed to write failure report for %s p%s: %s", path, page, log_err
        )


def img_to_bytes(img):
    img_byte_arr = io.BytesIO()
    white_composite_for_ocr(img).save(img_byte_arr, format="PNG", optimize=False)
    return img_byte_arr.getvalue()


def to_plain_list(value):
    """Converts Paddle/numpy structures into Python lists."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return []


def extract_paddle_lines(result):
    """Normalizes PaddleOCR v3/v2 result layouts into (text, box, confidence) triples."""
    if not result:
        return []

    entries = []
    result_item = result[0] if isinstance(result, list) else result

    if isinstance(result_item, dict):
        texts = [str(t).strip() for t in to_plain_list(result_item.get("rec_texts"))]
        scores = to_plain_list(result_item.get("rec_scores"))
        boxes = []
        # Prefer detector-space polygons first for page-coordinate placement.
        for key in ("dt_polys", "dt_boxes", "rec_polys", "rec_boxes"):
            candidate = to_plain_list(result_item.get(key))
            if candidate:
                boxes = candidate
                break

        for idx, txt in enumerate(texts):
            if not txt:
                continue
            box = boxes[idx] if idx < len(boxes) else None
            conf = float(scores[idx]) if idx < len(scores) else 0.0
            entries.append((txt, box, conf))
        return entries

    if isinstance(result_item, list):
        for line in result_item:
            try:
                txt = str(line[1][0]).strip()
                box = line[0]
                conf = float(line[1][1]) if len(line[1]) > 1 else 0.0
                if txt:
                    entries.append((txt, box, conf))
            except Exception as exc:
                logger.debug("Skipping malformed PaddleOCR line entry: %s", exc)
                continue

    return entries


def box_to_rect_and_anchor(box):
    """Builds a rectangle and fallback anchor from polygon/box inputs."""
    box_list = to_plain_list(box)
    if not box_list:
        return None, None

    first = box_list[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        points = []
        for point in box_list:
            point_vals = to_plain_list(point)
            if len(point_vals) < 2:
                continue
            try:
                points.append((float(point_vals[0]), float(point_vals[1])))
            except Exception as exc:
                logger.debug("Skipping invalid box point: %s", exc)
                continue
        if not points:
            return None, None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
        return rect, points[0]

    try:
        x0 = float(box_list[0])
        y0 = float(box_list[1])
        x2 = float(box_list[2])
        y2 = float(box_list[3])
    except Exception as exc:
        logger.debug("Failed to parse box coordinates: %s", exc)
        return None, None

    if x2 <= x0 or y2 <= y0:
        x2 = x0 + max(float(box_list[2]), 1.0)
        y2 = y0 + max(float(box_list[3]), 1.0)

    rect = fitz.Rect(min(x0, x2), min(y0, y2), max(x0, x2), max(y0, y2))
    return rect, (rect.x0, rect.y0)


def _resolve_text_font(lang_code):
    """Return (fontname, fontfile) for the given language.

    When font_selector is available and the font file exists on disk,
    returns a custom fontname and the file path so PyMuPDF embeds the
    correct glyphs.  Otherwise falls back to the Base14 ``helv``.
    """
    if lang_code and _FONT_SELECTOR_AVAILABLE:
        font_path = _get_font_path(lang_code)
        if font_path is not None:
            return str(font_path.stem), str(font_path)
    return "helv", None


_TEXT_METRICS_FONT_CACHE = {}
_TEXT_METRICS_FONT_LOCK = threading.Lock()


def _measure_text_unit_width(txt, fontname, fontfile):
    """Measure text at one point with the font used for PDF insertion.

    ``fitz.get_text_length`` only supports Base14 font names and rejects the
    ``fontfile`` keyword accepted by ``Page.insert_text``.  Custom fonts must
    therefore be measured through a ``fitz.Font`` instance.  Cache the font
    objects because this function runs once per OCR line.
    """
    if not fontfile:
        return fitz.get_text_length(txt, fontname=fontname, fontsize=1.0)

    with _TEXT_METRICS_FONT_LOCK:
        font = _TEXT_METRICS_FONT_CACHE.get(fontfile)
        if font is None:
            font = fitz.Font(fontfile=fontfile)
            _TEXT_METRICS_FONT_CACHE[fontfile] = font
        return font.text_length(txt, fontsize=1.0)


def insert_text_line(page, txt, box, lang_code=None, source_rotation=0):
    """Place invisible OCR text only when it fits its detected PDF-point box.

    When *lang_code* is provided and a matching Noto font file is
    installed, the text is embedded with that font so non-Latin scripts
    render correctly in the searchable PDF text layer.
    """
    fontname, fontfile = _resolve_text_font(lang_code)
    return insert_hidden_text_in_box(
        page, txt, box, fontname=fontname, fontfile=fontfile,
        source_rotation=source_rotation,
    )


try:
    from ocr_local.config.language_config import (
        LANG_MAPPING,
        get_tesseract_code,
        is_paddle_language_supported,
    )
except ImportError:
    LANG_MAPPING = {
        "zh": "ch",
        "zh-cn": "ch",
        "zh-tw": "chinese_cht",
        "ja": "japan",
        "ko": "korean",
        "vi": "vi",
        "en": "en",
        "fr": "fr",
        "de": "german",
        "es": "es",
        "it": "it",
        "pt": "pt",
        "ru": "ru",
        "ar": "ar",
        "hi": "hi",
        "uk": "uk",
        "be": "be",
        "bg": "bg",
        "cs": "cs",
        "pl": "pl",
        "tr": "tr",
        "nl": "nl",
        "sv": "sv",
        "da": "da",
        "fi": "fi",
        "el": "el",
        "hu": "hu",
        "ro": "ro",
        "fa": "fa",
        "ur": "ur",
        "ug": "ug",
        "te": "te",
        "ta": "ta",
        "kn": "kn",
        "ka": "ka",
    }
    _PADDLE_UNSUPPORTED_LANGUAGE_CODES = frozenset({"bn", "el", "kn", "th"})
    _PADDLE_KNOWN_LANGUAGE_CODES = frozenset(LANG_MAPPING) | frozenset(
        LANG_MAPPING.values()
    )

    def is_paddle_language_supported(language_code):
        code = str(language_code or "").strip().lower()
        return bool(
            code
            and code in _PADDLE_KNOWN_LANGUAGE_CODES
            and code not in _PADDLE_UNSUPPORTED_LANGUAGE_CODES
        )

    def get_tesseract_code(language_code):
        return {"el": "ell", "kn": "kan", "th": "tha", "bn": "ben"}.get(
            str(language_code or ""), "eng"
        )


lang_model = None


def load_fasttext():
    global lang_model
    lang_model = load_fasttext_model(
        FASTTEXT_MODEL_PATH,
        logger=logger,
        context="FastText",
    )
    if lang_model is not None:
        logger.info("FastText Language Model Loaded.")


def detect_language(doc_path):
    """Attempts to detect language from the first page text (if any).

    Detect-fail (missing model, short sample, or exception) returns ``und``.
    Never invent ``en`` as a detector default; English weights are a model
    fallback, not a language detection result.
    """
    if not lang_model:
        return "und"
    try:
        with fitz.open(doc_path) as doc:
            # Try getting text from first few pages
            text_sample = ""
            for i in range(min(3, doc.page_count)):
                text_sample += doc[i].get_text()

            if len(text_sample) > 50:
                text_sample = text_sample.replace("\n", " ").casefold()[:200]
                lang_code, confidence = predict_language(
                    lang_model,
                    text_sample,
                    logger=logger,
                    context="Language detection from path FastText predict",
                )
                if confidence >= 0.7:
                    return LANG_MAPPING.get(lang_code, "und")
    except Exception:
        logger.exception("Language detection from path failed")

    return "und"


def detect_language_from_text(text):
    """Detects language from OCR output string."""
    if not lang_model or len(text) < 20:
        return None, 0.0
    try:
        # Create a clean sample (remove digits/symbols)
        sample = text.replace("\n", " ").casefold()[:300]
        lang_code, conf = predict_language(
            lang_model,
            sample,
            logger=logger,
            context="Language detection from text FastText predict",
        )
        mapped = LANG_MAPPING.get(lang_code)

        if mapped and conf > 0.4:  # 40% Confidence threshold
            return mapped, conf
    except Exception as exc:
        logger.debug("Language detection from text failed: %s", exc)
    return None, 0.0


def create_paddle_engine(lang_code, device):
    """Build a Paddle engine with geometry-preserving defaults for OCR overlay.

    Uses OCR_INFERENCE_BACKEND environment variable to select the
    inference backend (paddle, onnx, openvino, auto).  Falls back to
    the native PaddlePaddle constructor if the backend module is not
    available.
    """
    if not _PADDLEOCR_AVAILABLE:
        raise RuntimeError("PaddleOCR dependency not installed")
    try:
        from ocr_inference_backend import create_ocr_engine

        return create_ocr_engine(lang_code, device)
    except ImportError:
        pass

    # Fallback if ocr_inference_backend is not available.
    # Uses PaddleOCR 2.x constructor kwargs: use_angle_cls, use_gpu, show_log.
    use_gpu, gpu_id = get_gpu_device_configuration(device)
    fallback_kwargs = dict(
        use_angle_cls=True,
        lang=lang_code,
        use_gpu=use_gpu,
        show_log=False,
        drop_score=OCR_DROP_SCORE,
    )
    if gpu_id is not None:
        fallback_kwargs["gpu_id"] = gpu_id
    if get_native_threads_per_predictor() is not None:
        fallback_kwargs["cpu_threads"] = get_native_threads_per_predictor()
    if get_recognizer_batch_size() is not None:
        fallback_kwargs["rec_batch_num"] = get_recognizer_batch_size()
    return PaddleOCR(**fallback_kwargs)


def _detect_default_paddle_device():
    """Return the real default Paddle device for this process.

    CPU release images deliberately set ``CUDA_VISIBLE_DEVICES`` to an empty
    value.  Passing ``use_gpu=True`` in that environment makes PaddleOCR fall
    back internally while the application still caches and reports a GPU
    engine.  Resolve the device before engine construction instead.
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices.strip().lower() in {
        "",
        "-1",
        "none",
        "void",
    }:
        return "cpu"

    try:
        import paddle

        if (
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        ):
            return "gpu"
    except Exception as exc:
        logger.debug("Paddle device detection failed; using CPU: %s", exc)
    return "cpu"


_DEFAULT_PADDLE_DEVICE = _detect_default_paddle_device()


def _bounded_env_capacity(name: str, default: int, maximum: int) -> int:
    try:
        return max(1, min(maximum, int(os.environ.get(name, str(default)))))
    except ValueError:
        logger.warning("Invalid %s; using %d", name, default)
        return default


CPU_PADDLE_REPLICAS = _bounded_env_capacity("OCR_CPU_PADDLE_REPLICAS", 2, 4)
GPU_PADDLE_REPLICAS = _bounded_env_capacity("OCR_GPU_PADDLE_REPLICAS", 1, 4)
PADDLE_POOL_MAX_MODELS = _bounded_env_capacity("OCR_ENGINE_POOL_MAX_ENTRIES", 4, 16)
PADDLE_POOL_PENDING_PER_REPLICA = _bounded_env_capacity(
    "OCR_ENGINE_MAX_PENDING", 2, 32
)


def _get_cpu_paddle_executor(replica_id=0):
    """Return one stable owner thread for a bounded CPU predictor replica."""
    if not 0 <= replica_id < CPU_PADDLE_REPLICAS:
        raise ValueError("CPU Paddle replica outside configured capacity")
    with _CPU_PADDLE_EXECUTOR_LOCK:
        if not _CPU_PADDLE_ACCEPTING_WORK:
            raise RuntimeError("CPU Paddle executor is closed for shutdown")
        if replica_id not in _CPU_PADDLE_EXECUTORS:
            _CPU_PADDLE_EXECUTORS[replica_id] = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"PaddleCPU-{replica_id}",
            )
        return _CPU_PADDLE_EXECUTORS[replica_id]


def _call_on_cpu_paddle_thread(fn, *args, replica_id=0, **kwargs):
    """Run *fn* on the selected stable CPU predictor owner thread."""
    if not 0 <= replica_id < CPU_PADDLE_REPLICAS:
        raise ValueError("CPU Paddle replica outside configured capacity")
    # One running and one pending operation per replica. ThreadPoolExecutor's
    # own queue is unbounded, so admission must happen before submit().
    admission = _CPU_PADDLE_ADMISSION[replica_id]
    admission.acquire()
    try:
        with _CPU_PADDLE_EXECUTOR_LOCK:
            if not _CPU_PADDLE_ACCEPTING_WORK:
                raise RuntimeError("CPU Paddle executor is closed for shutdown")
            if replica_id not in _CPU_PADDLE_EXECUTORS:
                _CPU_PADDLE_EXECUTORS[replica_id] = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=f"PaddleCPU-{replica_id}",
                )
            # Submission and shutdown share this lifecycle lock.
            future = _CPU_PADDLE_EXECUTORS[replica_id].submit(fn, *args, **kwargs)
        return future.result()
    finally:
        admission.release()


def _run_paddle_ocr(engine, inference_lock, image, device, *, replica_id=0):
    """Run OCR with serialization and CPU predictor thread affinity.

    PaddlePaddle 3.0 CPU predictors can raise ``could not execute a
    primitive`` when a single predictor is reused sequentially from different
    Python threads.  A lock prevents overlap but not that migration.  CPU
    engines are therefore both constructed and invoked on one owner thread.
    """

    def _invoke():
        with inference_lock:
            return engine.ocr(image)

    if device == "cpu":
        return _call_on_cpu_paddle_thread(_invoke, replica_id=replica_id)
    return _invoke()


def _start_cpu_paddle_executor_lifecycle():
    """Allow CPU predictor submissions for a new pipeline invocation."""
    global _CPU_PADDLE_ACCEPTING_WORK
    with _CPU_PADDLE_EXECUTOR_LOCK:
        if _CPU_PADDLE_EXECUTORS:
            raise RuntimeError("cannot reopen CPU Paddle lifecycle before shutdown")
        _CPU_PADDLE_ACCEPTING_WORK = True


def _close_cpu_paddle_executor_lifecycle():
    """Reject late submissions while pipeline workers are being joined."""
    global _CPU_PADDLE_ACCEPTING_WORK
    with _CPU_PADDLE_EXECUTOR_LOCK:
        _CPU_PADDLE_ACCEPTING_WORK = False


def _shutdown_cpu_paddle_executor():
    """Shut down the CPU owner thread and evict thread-bound predictors."""
    global _CPU_PADDLE_ACCEPTING_WORK
    with _CPU_PADDLE_EXECUTOR_LOCK:
        _CPU_PADDLE_ACCEPTING_WORK = False
        executors = list(_CPU_PADDLE_EXECUTORS.values())
        _CPU_PADDLE_EXECUTORS.clear()
    for executor in executors:
        executor.shutdown(wait=True, cancel_futures=False)
    with _ENGINE_CACHE_LOCK:
        cpu_keys = [key for key in _ENGINE_CACHE if key[0] == "cpu"]
        for key in cpu_keys:
            _ENGINE_CACHE.pop(key, None)


def _engine_cache_key(lang_code, device, replica_id=0):
    """Return the shared PaddleOCR cache key for a language/device pair."""
    return (str(device or "gpu"), str(lang_code), OCR_DROP_SCORE,
            replica_id if device == "cpu" else 0)


def _standalone_pool_for_device(device: str) -> EngineRuntimePool:
    """Return the bounded process-local runtime for one supported device."""
    if device not in {"cpu", "gpu", "gpu:0"}:
        raise ValueError(f"Unsupported standalone Paddle device {device!r}")
    pool_device = "cpu" if device == "cpu" else "gpu"
    with _PADDLE_POOL_LOCK:
        if not _PADDLE_POOL_ACCEPTING_WORK:
            raise RuntimeError("Paddle runtime pool is closed for shutdown")
        pool = _PADDLE_POOLS.get(pool_device)
        if pool is None:
            device_default = (
                CPU_PADDLE_REPLICAS if device == "cpu" else GPU_PADDLE_REPLICAS
            )
            per_key = _bounded_env_capacity(
                "OCR_ENGINE_REPLICAS_PER_KEY", device_default, 4
            )
            per_device = _bounded_env_capacity(
                "OCR_INFERENCE_SLOTS_PER_DEVICE", device_default, 8
            )
            pool = EngineRuntimePool(
                max_entries=PADDLE_POOL_MAX_MODELS,
                max_replicas_per_key=per_key,
                max_replicas_per_device=per_device,
                max_pending_per_replica=PADDLE_POOL_PENDING_PER_REPLICA,
                native_call_timeout_seconds=production_native_call_timeout(),
            )
            _PADDLE_POOLS[pool_device] = pool
        return pool


def _standalone_paddle_key(lang_code: str, device: str) -> EngineKey:
    """Bind predictor reuse to its actual model and configured constructor."""
    actual_language = get_paddle_runtime_code(lang_code)
    visible = (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        or os.environ.get("NVIDIA_VISIBLE_DEVICES")
        or "unbound"
    ).split(",", 1)[0]
    device_id = "cpu" if device == "cpu" else f"gpu:{visible}"
    return EngineKey(
        device=device_id,
        language=actual_language,
        model_family="paddleocr",
        backend=(get_ocr_backend() if device == "cpu" else "paddle"),
        config_identity=(
            "standalone-v1",
            OCR_DROP_SCORE,
            get_detection_limit_side_len(),
            get_native_threads_per_predictor(),
            get_recognizer_batch_size(),
            device,
            True,  # use_angle_cls
            os.environ.get("PADDLEOCR_HOME", ""),
        ),
    )


def _run_pooled_paddle(lang_code: str, image: np.ndarray, device: str):
    """Construct and invoke Paddle only on a pool owner; expose no engine.

    A failed language construction gets one English fallback, whose distinct
    key accurately identifies the model in use. Inference errors retire the
    predictor within EngineRuntimePool before the next call.
    """
    if device == "gpu":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get(
            "NVIDIA_VISIBLE_DEVICES", ""
        )
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if len(tokens) == 1 and tokens[0].lower() not in {"all", "none", "void"}:
            device = "gpu:0"
    pool = _standalone_pool_for_device(device)

    class ModelLoadFailure(RuntimeError):
        pass

    def build(requested_language):
        try:
            return create_paddle_engine(requested_language, device)
        except Exception as exc:
            raise ModelLoadFailure(str(exc)) from exc

    def invoke(engine):
        return engine.ocr(image)

    try:
        return (
            pool.run(
                _standalone_paddle_key(lang_code, device),
                lambda: build(lang_code),
                invoke,
            ),
            get_paddle_runtime_code(lang_code),
        )
    except ModelLoadFailure as exc:
        if get_paddle_runtime_code(lang_code) == "en":
            raise
        logger.warning("Paddle %s failed; trying English model: %s", lang_code, exc)
        return (
            pool.run(
                _standalone_paddle_key("en", device),
                lambda: build("en"),
                invoke,
            ),
            "en",
        )


def _close_standalone_paddle_pools() -> None:
    global _PADDLE_POOL_ACCEPTING_WORK
    with _PADDLE_POOL_LOCK:
        _PADDLE_POOL_ACCEPTING_WORK = False
        pools = list(_PADDLE_POOLS.values())
        _PADDLE_POOLS.clear()
    for pool in pools:
        pool.close()


def _reset_standalone_pools_after_fork() -> None:
    global _PADDLE_POOL_LOCK, _PADDLE_POOLS, _PADDLE_POOL_ACCEPTING_WORK
    _PADDLE_POOL_LOCK = threading.Lock()
    _PADDLE_POOLS = {}
    _PADDLE_POOL_ACCEPTING_WORK = True


class _EngineCacheEntry:
    """Two-value-compatible handle plus immutable actual model language."""

    __slots__ = ("engine", "inference_lock", "actual_language")

    def __init__(self, engine, inference_lock, actual_language):
        self.engine = engine
        self.inference_lock = inference_lock
        self.actual_language = str(actual_language)

    def __iter__(self):
        # Preserve the historic ``engine, lock = entry`` public contract.
        yield self.engine
        yield self.inference_lock

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (self.engine, self.inference_lock)[index]


def _evict_engine_cache_entry(cache_key, expected_entry):
    """Evict only the exact predictor generation that actually failed."""
    if cache_key is None or expected_entry is None:
        return False
    with _ENGINE_CACHE_LOCK:
        if _ENGINE_CACHE.get(cache_key) is not expected_entry:
            return False
        _ENGINE_CACHE.pop(cache_key, None)
        return True


def get_or_create_engine(lang_code, device="gpu", *, replica_id=0):
    """Return a shared PaddleOCR engine for *lang_code*, creating it if needed.

    Thread-safe.  The returned tuple is ``(engine, inference_lock)`` where
    *inference_lock* is a :class:`threading.Lock` that callers **must** hold
    while calling ``engine.ocr()`` to prevent concurrent CUDA session
    corruption.

    If primary language model loading fails, falls back to English.  If
    the English fallback also fails, the entry is cached as ``None`` so
    creation is not retried.

    Each CPU replica owns its predictor and construction lock. GPU engines
    retain one shared construction lock for the single visible device.
    """
    if device == "cpu":
        with _CPU_PADDLE_EXECUTOR_LOCK:
            if not _CPU_PADDLE_ACCEPTING_WORK:
                raise RuntimeError("CPU Paddle executor is closed for shutdown")

    cache_key = _engine_cache_key(lang_code, device, replica_id)

    # Fast path -- already cached (no lock needed for dict read under GIL,
    # but we use the lock to stay safe with the double-check pattern).
    with _ENGINE_CACHE_LOCK:
        if cache_key in _ENGINE_CACHE:
            return _ENGINE_CACHE[cache_key]

    # Only construction for the same owner is serialized. Independent CPU
    # replicas can load and infer concurrently on distinct predictor objects.
    construction_lock = (
        _CPU_PADDLE_INIT_LOCKS[replica_id] if device == "cpu" else model_load_lock
    )
    with construction_lock:
        # Double-check: another thread may have created while we waited.
        with _ENGINE_CACHE_LOCK:
            if cache_key in _ENGINE_CACHE:
                return _ENGINE_CACHE[cache_key]

        # Create the engine (outside _ENGINE_CACHE_LOCK to avoid holding
        # two locks simultaneously).
        entry = None
        try:
            logger.info("Loading shared PaddleOCR model for: %s", lang_code)
            if device == "cpu":
                engine = _call_on_cpu_paddle_thread(
                    create_paddle_engine,
                    lang_code,
                    replica_id=replica_id,
                    device=device,
                )
            else:
                engine = create_paddle_engine(lang_code, device=device)
            entry = _EngineCacheEntry(engine, threading.Lock(), lang_code)
        except Exception as primary_exc:
            logger.warning(
                "Failed to load %s model, falling back to English. Error: %s",
                lang_code,
                primary_exc,
            )
            try:
                if device == "cpu":
                    engine = _call_on_cpu_paddle_thread(
                        create_paddle_engine,
                        "en",
                        replica_id=replica_id,
                        device=device,
                    )
                else:
                    engine = create_paddle_engine("en", device=device)
                entry = _EngineCacheEntry(engine, threading.Lock(), "en")
            except Exception as fallback_exc:
                logger.error(
                    "English fallback also failed for %s. "
                    "All pages will use Tesseract/ImageOnly. Error: %s",
                    lang_code,
                    fallback_exc,
                )
                entry = None  # Cache the failure to avoid retrying

        with _ENGINE_CACHE_LOCK:
            _ENGINE_CACHE[cache_key] = entry

    return entry


# --- Document Intelligence Helpers (Phase 3A) ---


def parse_structure_result(result):
    """Parse PP-StructureV3 result into standardized format.

    Returns dict with 'layout_regions', 'tables', 'key_value_pairs',
    and 'form_fields' lists.  Safe to call with None or empty input.
    """
    if not result:
        return {
            "layout_regions": [],
            "tables": [],
            "key_value_pairs": [],
            "form_fields": [],
        }

    regions = []
    tables = []
    key_value_pairs = []
    form_fields = []
    field_id = 1
    for item in result:
        if not isinstance(item, dict):
            continue
        region = {
            "type": item.get("type", "unknown"),
            "bbox": item.get("bbox", []),
            "confidence": 0.0,
        }
        if item.get("type") == "table" and "res" in item:
            table_data = {
                "html": item["res"].get("html", "")
                if isinstance(item["res"], dict)
                else "",
                "cell_bbox": item["res"].get("cell_bbox", [])
                if isinstance(item["res"], dict)
                else [],
            }
            tables.append(table_data)
            region["table_index"] = len(tables) - 1
        elif "res" in item:
            # Text region — res may be a list of dicts with text/confidence
            if isinstance(item["res"], list):
                texts = []
                confidences = []
                for text_item in item["res"]:
                    if isinstance(text_item, dict):
                        texts.append(text_item.get("text", ""))
                        confidences.append(text_item.get("confidence", 0.0))
                region["text"] = " ".join(texts)
                region["confidence"] = (
                    sum(confidences) / len(confidences) if confidences else 0.0
                )

                # Phase 3C: Extract KV pairs and form fields from text regions
                if ENABLE_KV_EXTRACTION or ENABLE_FORM_DETECTION:
                    kvs, fields, field_id = _extract_forms_and_kvs(
                        item["res"],
                        region.get("bbox", []),
                        field_id,
                    )
                    key_value_pairs.extend(kvs)
                    form_fields.extend(fields)

        regions.append(region)
    return {
        "layout_regions": regions,
        "tables": tables,
        "key_value_pairs": key_value_pairs,
        "form_fields": form_fields,
    }


# --- Form & Key-Value Extraction Helpers (Phase 3C) ---

# Patterns for privilege detection
_PRIVILEGE_KEYWORDS = {
    "attorney-client",
    "attorney client",
    "work product",
    "privileged and confidential",
    "privileged & confidential",
    "attorney work product",
    "do not disclose",
    "legally privileged",
}

PRIVILEGE_HEURISTIC_CONFIDENCE = 0.85

_ESQ_PATTERN = re.compile(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*),\s*Esq\.?")
_LAW_FIRM_PATTERN = re.compile(
    r"\b\w+(?:\s+\w+)*\s+(?:&|and)\s+\w+(?:\s+\w+)*\s+(?:LLP|PLLC|P\.C\.|LLC)\b"
    r"|(?:Law\s+Offices?\s+of\b)"
    r"|(?:Attorneys?\s+at\s+Law\b)",
)

_FORM_FIELD_KEYWORDS = {
    "signature": ("signature", "sign here", "signed by", "authorized signature"),
    "date": ("date signed", "effective date", "date:", "dated"),
    "checkbox": ("☐", "☑", "☒", "[ ]", "[x]", "[X]"),
}


def _extract_forms_and_kvs(res_items, region_bbox, field_id_start):
    """Extract key-value pairs and form fields from PPStructure text results.

    PP-StructureV3 does not natively return form/KV data, so this uses
    heuristic approaches:
      - Colon-separated "Key: Value" patterns
      - Keyword matching for signature/date/checkbox fields

    Args:
        res_items: list of dicts from PPStructure ``item["res"]``
        region_bbox: bounding box of the parent region
        field_id_start: starting field_id counter

    Returns:
        (key_value_pairs, form_fields, next_field_id) tuple
    """
    key_value_pairs = []
    form_fields = []
    field_id = field_id_start

    for text_item in res_items:
        if not isinstance(text_item, dict):
            continue
        text = text_item.get("text", "")
        if not text:
            continue
        bbox = text_item.get("bbox", region_bbox)
        conf = text_item.get("confidence", 0.0)

        # Heuristic 1: Colon-based key-value extraction
        if ENABLE_KV_EXTRACTION and ":" in text:
            parts = text.split(":", 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            if key and len(key) < 80:
                # Approximate bbox split at colon position
                if isinstance(bbox, list) and len(bbox) == 4 and len(text) > 0:
                    ratio = len(parts[0]) / len(text)
                    mid_x = bbox[0] + (bbox[2] - bbox[0]) * ratio
                    key_bbox = [bbox[0], bbox[1], mid_x, bbox[3]]
                    value_bbox = [mid_x, bbox[1], bbox[2], bbox[3]]
                else:
                    key_bbox = bbox
                    value_bbox = bbox
                key_value_pairs.append(
                    {
                        "key": key,
                        "value": value,
                        "key_bbox": key_bbox,
                        "value_bbox": value_bbox,
                        "confidence": conf,
                        "extraction_method": "heuristic_colon",
                    }
                )

        # Heuristic 2: Form field keyword detection
        if ENABLE_FORM_DETECTION:
            text_lower = text.lower()
            for ftype, keywords in _FORM_FIELD_KEYWORDS.items():
                if any(kw in text_lower for kw in keywords):
                    form_fields.append(
                        {
                            "field_id": field_id,
                            "label": text.strip(),
                            "value": None,
                            "field_type": ftype,
                            "bbox": bbox,
                            "confidence": conf,
                            "is_filled": False,
                        }
                    )
                    field_id += 1
                    break  # one match per text item

    return key_value_pairs, form_fields, field_id


def _detect_privilege_indicators(pages_structure):
    """Detect attorney-client privilege indicators across all pages.

    Scans text regions for attorney names (``Name, Esq.``), law firm
    patterns, and privileged keywords.  Returns a dict of findings or
    ``None`` if no indicators are detected.
    """
    attorney_names = set()
    law_firms = set()
    privileged_keywords = set()

    for page in pages_structure:
        if not isinstance(page, dict):
            continue
        for region in page.get("layout_regions", []):
            text = region.get("text", "")
            if not text:
                continue
            text_lower = text.lower()

            # Attorney names ("Name, Esq.")
            for match in _ESQ_PATTERN.finditer(text):
                attorney_names.add(match.group(1))

            # Law firm patterns
            for match in _LAW_FIRM_PATTERN.finditer(text):
                law_firms.add(match.group(0).strip())

            # Privileged keywords
            for keyword in _PRIVILEGE_KEYWORDS:
                if keyword in text_lower:
                    privileged_keywords.add(keyword)

    if attorney_names or law_firms or privileged_keywords:
        return {
            "attorney_names": sorted(attorney_names),
            "law_firm": "; ".join(sorted(law_firms)) if law_firms else None,
            "privileged_keywords": sorted(privileged_keywords),
            "confidence": PRIVILEGE_HEURISTIC_CONFIDENCE,
            "review_required": True,
        }
    return None


def _build_document_summary(pages_structure):
    """Build aggregate statistics from page-level structure data."""
    total_tables = 0
    total_figures = 0
    total_form_fields = 0
    total_key_value_pairs = 0
    layout_types = set()
    has_signatures = False
    has_filled_forms = False

    for page in pages_structure:
        if not isinstance(page, dict):
            continue
        for region in page.get("layout_regions", []):
            rtype = region.get("type", "unknown")
            layout_types.add(rtype)
            if rtype == "table":
                total_tables += 1
            elif rtype == "figure":
                total_figures += 1

        # Phase 3C: form fields
        for field in page.get("form_fields", []):
            total_form_fields += 1
            if field.get("field_type") == "signature":
                has_signatures = True
            if field.get("is_filled"):
                has_filled_forms = True

        # Phase 3C: key-value pairs
        total_key_value_pairs += len(page.get("key_value_pairs", []))

    summary = {
        "total_tables": total_tables,
        "total_figures": total_figures,
        "total_form_fields": total_form_fields,
        "total_key_value_pairs": total_key_value_pairs,
        "layout_types_found": sorted(layout_types),
        "has_signatures": has_signatures,
        "has_filled_forms": has_filled_forms,
    }

    # Phase 3C: Privilege detection
    if ENABLE_PRIVILEGE_DETECTION:
        indicators = _detect_privilege_indicators(pages_structure)
        if indicators:
            summary["privilege_indicators"] = indicators

    return summary


def write_structure_json(doc_id, source_file, pages_structure, output_dir):
    """Write sidecar JSON file with document intelligence results.

    Returns the path to the written JSON file, or None on failure.
    """
    from ocr_local.config.version import __version__

    try:
        structure_dir = os.path.join(output_dir, "EXPORT", "STRUCTURE")
        rel_path = _safe_source_relative_path(source_file)
        source_type = (
            "pdf" if os.path.splitext(source_file)[1].lower() == ".pdf" else "image"
        )
        rel_stem = build_output_rel_stem(source_file, source_type)
        target_dir = resolve_safe_output_subfolder(
            structure_dir,
            os.path.dirname(rel_stem),
        )
        os.makedirs(target_dir, exist_ok=True)

        base_name = os.path.basename(rel_stem)
        json_path = os.path.join(target_dir, f"{base_name}.structure.json")

        doc_structure = {
            "schema_version": "1.0",
            "document_id": doc_id,
            "source_file": rel_path,
            "processing": {
                "pipeline_version": __version__,
                "timestamp": datetime.datetime.now().isoformat(),
                "docintel_mode": DOCINTEL_MODE,
            },
            "pages": pages_structure,
            "document_summary": _build_document_summary(pages_structure),
        }

        write_json_atomic(
            json_path,
            doc_structure,
            allowed_root=structure_dir,
            ensure_ascii=False,
        )

        return json_path
    except Exception as e:
        logger.error("Failed to write structure JSON for %s: %s", doc_id, e)
        return None


class HTMLTableParser(HTMLParser):
    """Extract table data from HTML string into 2D list of cells."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self._row = []
        self._cell = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._in_cell = False
            self._row.append("".join(self._cell).strip())
        elif tag == "tr" and self._row:
            self.rows.append(self._row)

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)


def html_table_to_csv_rows(html_string):
    """Convert HTML table markup to a list of rows (each row a list of cell strings)."""
    parser = HTMLTableParser()
    try:
        parser.feed(html_string or "")
        parser.close()
    except Exception as exc:
        logger.debug("HTML table parsing failed: %s", exc)
        return []
    return parser.rows


_SAFE_TABLE_TAGS = frozenset(
    {"table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "colgroup", "col"}
)
_HTML_TAG_RE = re.compile(r"<(/?)(\w+)([^>]*)>", re.DOTALL)


def _sanitize_table_html(raw_html):
    """Strip all HTML tags except safe table-related elements to prevent XSS."""

    def _replace(match):
        slash, tag, _attrs = match.group(1), match.group(2).lower(), match.group(3)
        if tag in _SAFE_TABLE_TAGS:
            return f"<{slash}{tag}>"  # Strip attributes too
        return ""

    text = _HTML_TAG_RE.sub(_replace, raw_html)
    return text


def write_extracted_tables(doc_id, source_file, pages_structure, output_dir):
    """Export detected tables as standalone HTML and CSV files.

    Writes each table to EXPORT/TABLES/<subfolder>/ mirroring the source
    directory structure.  Mutates pages_structure in-place to add
    'extracted_files' paths to each table dict.

    Returns the number of tables successfully exported.
    """
    tables_dir = os.path.join(output_dir, "EXPORT", "TABLES")
    source_type = (
        "pdf" if os.path.splitext(source_file)[1].lower() == ".pdf" else "image"
    )
    rel_stem = build_output_rel_stem(source_file, source_type)
    target_dir = resolve_safe_output_subfolder(
        tables_dir,
        os.path.dirname(rel_stem),
    )
    base_name = os.path.basename(rel_stem)
    os.makedirs(target_dir, exist_ok=True)

    exported = 0
    for page in pages_structure:
        page_num = page.get("page_num", 0)
        for t_idx, table in enumerate(page.get("tables", [])):
            html_content = table.get("html")
            if not html_content:
                continue
            stem = f"{base_name}_p{page_num}_t{t_idx}"
            try:
                # Sanitize HTML to allow only safe table tags (prevent XSS)
                sanitized_html = _sanitize_table_html(html_content)
                # Write HTML / CSV via atomic leaf writers (replace symlink nodes)
                html_path = os.path.join(target_dir, f"{stem}.html")
                write_text_atomic(
                    html_path,
                    sanitized_html,
                    allowed_root=tables_dir,
                )
                csv_path = os.path.join(target_dir, f"{stem}.csv")
                rows = html_table_to_csv_rows(html_content)
                csv_buffer = io.StringIO()
                csv.writer(csv_buffer).writerows(rows)
                write_text_atomic(
                    csv_path,
                    csv_buffer.getvalue(),
                    allowed_root=tables_dir,
                )
                # Store relative paths in structure for JSON portability
                table["extracted_files"] = {
                    "html": os.path.relpath(html_path, output_dir),
                    "csv": os.path.relpath(csv_path, output_dir),
                }
                exported += 1
            except Exception as exc:
                logger.warning("Table export failed for %s: %s", stem, exc)
    return exported


# --- Core Classes ---


class PageTask:
    def __init__(
        self,
        doc_id: str,
        doc_path: str,
        page_num: int,
        image: Image.Image,
        lang_hint: str = "en",
        source_type: str = "pdf",
        source_snapshot_path: str | None = None,
        geometry: PageGeometry | None = None,
        explicit_lang_hint: str | None = None,
        source_prior_unverified: bool = False,
    ) -> None:
        self.doc_id: str = doc_id
        # doc_path is identity/custody only. Raster, DPI re-extract, and
        # salvage use source_snapshot_path (queued runtime). Never treat
        # doc_path as a /proc fd or other live raster handle.
        self.doc_path: str = doc_path
        self.source_snapshot_path: str = source_snapshot_path or ""
        self.page_num: int = page_num
        self.image: Image.Image = image
        self.lang_hint: str = lang_hint
        self.explicit_lang_hint: str | None = explicit_lang_hint
        self.source_prior_unverified: bool = bool(source_prior_unverified)
        self.source_type: str = source_type
        # Extractors pass source-derived geometry.  The fallback supports
        # directly constructed image tasks in local tooling/tests only.
        if geometry is None and source_type == "pdf":
            # Direct tasks without source metadata use an explicit DPI guess,
            # marked review-required and forbidden from resume admission.
            self.geometry = PageGeometry(
                image.width * 72.0 / DPI,
                image.height * 72.0 / DPI,
                "raster-dpi-fallback",
                source_dpi=(float(DPI), float(DPI)),
            )
        else:
            self.geometry = geometry or PageGeometry.from_image_pixels(
                *image.size, dpi=image.info.get("dpi"), fallback_dpi=DPI
            )
        self.retries: int = 0
        # Actual render DPI of the image currently attached to this task.
        # Updated only after a successful DPI-escalation re-extract.
        self.render_dpi: int = int(DPI)


def _mean_ocr_confidence(lines) -> float:
    """Include genuine confidence zero; ignore only invalid metadata."""
    values = []
    for _text, _box, raw_confidence in lines:
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            continue
        if math.isfinite(confidence) and 0 <= confidence <= 1:
            values.append(confidence)
    return sum(values) / len(values) if values else 0.0


def _tesseract_has_uncertain_word(lines, threshold=0.70) -> bool:
    """A high page mean cannot hide one low-confidence visible word."""
    for text, _box, raw_confidence in lines:
        if not any(char.isalpha() for char in str(text)):
            continue
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            return True
        if not math.isfinite(confidence) or confidence < threshold:
            return True
    return False


def _lowest_tesseract_word_confidence(lines) -> float:
    values = []
    for text, _box, raw_confidence in lines:
        if any(char.isalpha() for char in str(text)):
            try:
                value = float(raw_confidence)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    return min(values) if values else 0.0


def _confirm_attempt_text_language(attempt) -> None:
    """Confirm a label only from the selected fresh OCR text and script."""
    label, confidence = detect_language_from_text(attempt.text_content)
    entry = LANGUAGE_REGISTRY.get(label) if label else None
    if (
        entry is not None
        and confidence >= 0.7
        and text_supports_script(attempt.text_content, entry.script)
    ):
        attempt.detected_language = label
        attempt.detected_language_confidence = confidence
        if label != attempt.model_language:
            attempt.review_reasons.append(
                "confirmed text language differs from selected OCR model"
            )


def _measure_attempt_coverage(attempt, spans, candidate_raster_size):
    """Compare OCR spans to independent visual evidence in source pixels."""
    if attempt.raster_evidence is None:
        attempt.raster_evidence = assess_raster_text_regions(attempt.task.image)
    assessment = assess_candidate_coverage(
        attempt.raster_evidence,
        spans,
        candidate_raster_size=candidate_raster_size,
    )
    return asdict(assessment)


def _page_script_evidence(image: Image.Image, prior_label: str = "und") -> tuple[str, ...]:
    """Bounded, corroborated script probe; never claims a language label."""
    try:
        data = pytesseract.image_to_osd(
            image, output_type=pytesseract.Output.DICT, timeout=3
        )
        script = str(data.get("script") or "").strip().lower()
        confidence = float(data.get("script_conf") or 0)
        if not math.isfinite(confidence):
            return ()
        # A weak confusable OSD result is only a reason to run independent
        # bounded OCR probes; it is never accepted as script evidence itself.
        if script in {"greek", "cyrillic"} and confidence >= 1:
            return probe_image_script_candidates(image)
        if confidence < 10:
            return ()
        # Greek/Cyrillic/Latin OSD has made reproducible cross-script errors.
        prior_entry = LANGUAGE_REGISTRY.get(str(prior_label or "").lower())
        prior_script = prior_entry.script if prior_entry is not None else None
        # Only ambiguous Greek/Cyrillic guesses or a script conflicting with
        # a known page prior justify three costlier probes.
        if (
            prior_script in {"latin", "greek", "cyrillic"}
            and script in {"latin", "greek", "cyrillic"}
            and script != prior_script
        ):
            return probe_image_script_candidates(image)
        if script:
            return (script,)
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError):
        pass
    return ()


def _half_turn_direction_evidence(
    image: Image.Image, language: str
) -> tuple[int, str, float] | None:
    """Compatibility wrapper around the CPU-safe shared direction probe."""
    canonical, traineddata = _resolve_tesseract_route_language(language)
    if canonical != language:
        return None
    script_entry = LANGUAGE_REGISTRY.get(language)
    script = script_entry.script if script_entry is not None else ""
    evidence = probe_half_turn_direction(
        image, traineddata=traineddata, script=script,
        timeout_seconds=min(TESSERACT_TIMEOUT, 5),
    )
    if evidence.orientation is None:
        return None
    return evidence.orientation, evidence.reference_text, evidence.score


def _qualified_half_turn_direction(image: Image.Image, language: str) -> int | None:
    """Compatibility API for callers needing only the qualified direction."""
    evidence = _half_turn_direction_evidence(image, language)
    return evidence[0] if evidence is not None else None


def _needs_half_turn_direction_probe(lines) -> bool:
    """Probe only clear horizontal multi-character detector quads."""
    for text, box, _confidence in lines:
        if len(str(text).strip()) < 4:
            continue
        try:
            if len(box) != 4 or not hasattr(box[0], "__iter__"):
                continue
            p0, p1, _p2, p3 = box
            top = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            side = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
            if top > 3 * side:
                return True
        except (TypeError, ValueError, IndexError):
            continue
    return False


@dataclass
class _WorkerAttempt:
    """Per-queue-item OCR state. worker_thread locals persist across tasks."""

    task: PageTask
    pdf_bytes: bytes | None = None
    text_parts: list[str] = dataclass_field(default_factory=list)
    text_content: str = ""
    status: object = PageStatus.OK
    page_confidence: float = 0.0
    ocr_method: str = ""
    detected_language: str = ""
    detected_language_confidence: float | None = None
    model_language: str = ""
    ocr_language: str = ""
    paddle_lines: list = dataclass_field(default_factory=list)
    language_route: object | None = None
    review_reasons: list[str] = dataclass_field(default_factory=list)
    candidate_reviews: list[dict] = dataclass_field(default_factory=list)
    geometry_issues: list = dataclass_field(default_factory=list)
    raster_evidence: object | None = None
    coverage_assessment: dict = dataclass_field(default_factory=dict)
    coverage_needs_retry: bool = False
    tesseract_dpi_recovery_used: bool = False
    paddle_coverage_recovery_used: bool = False
    img_np: object | None = None
    ocr_img: object | None = None
    w: int = 0
    h: int = 0
    chunk_path: str | None = None
    checkpoint_sha256: str | None = None
    searchable_checkpoint_written: bool = False
    structure_data: object | None = None
    handwriting_data: object | None = None
    signature_data: object | None = None
    vertical_text_data: object | None = None
    table_fallback_data: object | None = None
    language_data: object | None = None
    routing_target: object | None = None
    routing_features: object | None = None
    routing_engine_force: object | None = None
    routing_paddle_device: str = ""
    actual_execution_device: str = "none"
    gpu_optimization_applied: bool = False
    preprocessing_applied: bool = False
    active_paddle_cache_key: object | None = None
    active_paddle_cache_entry: object | None = None
    engine_used: str | None = None
    reading_rotation: int = 0
    orientation_reference_text: str = ""
    half_turn_paddle_recovery_used: bool = False
    source_tesseract_probed: bool = False
    dpi_requeued: bool = False
    dpi_retry_inline: bool = False


def _bind_worker_attempt(task: PageTask) -> _WorkerAttempt:
    """Isolate all attempt fields immediately after queue receive."""
    detected = str(getattr(task, "lang_hint", None) or "").strip() or "und"
    return _WorkerAttempt(
        task=task,
        detected_language=detected,
        model_language="",
        ocr_language="",
        routing_paddle_device=_DEFAULT_PADDLE_DEVICE,
    )


def _clear_attempt_language_fields(attempt: _WorkerAttempt) -> None:
    """Image-only / extract-failed pages persist empty language fields."""
    attempt.detected_language = ""
    attempt.model_language = ""
    attempt.ocr_language = ""


def _sync_attempt_ocr_language_alias(attempt: _WorkerAttempt) -> None:
    """Existing ocr_language consumers read the model language, not FastText."""
    attempt.ocr_language = str(attempt.model_language or "")


def _on_worker_attempt_ready(attempt: _WorkerAttempt) -> None:
    """Hook after attempt bind, before OCR. Tests monkeypatch this."""
    return


def _canonical_attempt_checkpoint_path(temp_dir: str, page_num: int) -> str:
    """Absolute `{temp_dir}/{page_num}.pdf` used by keep-searchable."""
    return os.path.abspath(os.path.join(temp_dir, f"{int(page_num)}.pdf"))


def _keep_searchable_checkpoint(
    attempt: _WorkerAttempt, *, temp_dir: str | None
) -> bool:
    """Keep this attempt's canonical checkpoint; foreign leftovers fail closed."""
    if not attempt.searchable_checkpoint_written:
        return False
    chunk_path = attempt.chunk_path
    if not isinstance(chunk_path, str) or not chunk_path:
        return False
    if not isinstance(temp_dir, str) or not temp_dir:
        return False
    expected = _canonical_attempt_checkpoint_path(temp_dir, attempt.task.page_num)
    supplied = os.path.abspath(chunk_path)
    if os.path.normcase(supplied) != os.path.normcase(expected):
        return False
    try:
        if os.path.islink(expected) or os.path.islink(supplied):
            return False
        if not os.path.isfile(expected):
            return False
        temp_root = os.path.realpath(temp_dir)
        expected_real = os.path.realpath(expected)
        common = os.path.commonpath([temp_root, expected_real])
    except (OSError, ValueError):
        return False
    return os.path.normcase(common) == os.path.normcase(temp_root)


def _cache_hit_arms_keep_searchable(cached_status) -> bool:
    """Cache-hit keep-searchable is searchable OCR only, not IMAGE_ONLY.

    Routing-skip IMAGE_ONLY leaves ``searchable_checkpoint_written`` False.
    A validated cache hit of an IMAGE_ONLY checkpoint must match that policy.
    ``cached_status`` is the checkpoint metadata ``status`` (enum or value).
    """
    return (
        cached_status != PageStatus.IMAGE_ONLY
        and cached_status != PageStatus.IMAGE_ONLY.value
    )


def _re_extract_task_page_at_dpi(task: PageTask, dpi: int):
    """Re-render a task only from its immutable scheduler-owned source spool."""
    snapshot_path = getattr(task, "source_snapshot_path", None)
    if not isinstance(snapshot_path, str) or not snapshot_path:
        raise RuntimeError("page task is missing its immutable source snapshot")
    return re_extract_page_at_dpi(snapshot_path, task.page_num, dpi)


def _recover_incomplete_paddle_page(
    task: PageTask, attempt: _WorkerAttempt, current_doc, language: str
):
    """Bounded default OCR retry for independently observed missing regions.

    The source-bound higher-DPI image is used only for OCR. The original
    background and current searchable result remain authoritative until a
    comparable same-engine candidate improves independent visual coverage.
    """
    if (
        task.source_type != "pdf"
        or attempt.engine_used != "paddle"
        or attempt.coverage_assessment.get("status") != "measured"
        or not attempt.coverage_assessment.get("needs_retry")
    ):
        return current_doc
    attempt.paddle_coverage_recovery_used = True
    original_width, original_height = task.image.size
    for retry_dpi in DPI_SCHEDULE:
        if retry_dpi <= _task_render_dpi(task) or retry_dpi > 600:
            continue
        retry_image = None
        retry_doc = None
        try:
            _check_expected_raster_bytes(task.geometry, retry_dpi)
            retry_image = _re_extract_task_page_at_dpi(task, retry_dpi)
            if retry_image is None:
                break
            retry_result, retry_model = _run_pooled_paddle(
                language, np.array(retry_image), attempt.routing_paddle_device
            )
            retry_lines = extract_paddle_lines(retry_result)
            if not retry_lines:
                continue
            retry_transform = task.geometry.for_raster(*retry_image.size)
            retry_pdf_lines, issues = retry_transform.map_lines_with_review(retry_lines)
            original_pixel_lines, pixel_issues = PageGeometry(
                original_width, original_height, "checkpoint-local"
            ).for_raster(*retry_image.size).map_lines_with_review(retry_lines)
            if issues or pixel_issues or not retry_pdf_lines:
                attempt.candidate_reviews.append({
                    "text": " ".join(line[0] for line in retry_lines),
                    "model_language": retry_model,
                    "confidence": _mean_ocr_confidence(retry_lines),
                    "decision": "unplaced_dpi_candidate",
                })
                continue
            retry_coverage = _measure_attempt_coverage(
                attempt, original_pixel_lines, task.image.size
            )
            retry_confidence = _mean_ocr_confidence(retry_lines)
            retry_doc = fitz.open()
            page = retry_doc.new_page(
                width=task.geometry.width_pt, height=task.geometry.height_pt
            )
            page.insert_image(page.rect, stream=img_to_bytes(task.image))
            retry_parts = []
            for text, box, _confidence in retry_pdf_lines:
                normalized = (
                    normalize_ocr_text(text, language)
                    if _UNICODE_UTILS_AVAILABLE else text
                )
                if not insert_text_line(
                    page, normalized, box, lang_code=language,
                    source_rotation=attempt.reading_rotation,
                ):
                    issues.append({"text": normalized, "reason": "unplaceable DPI OCR text"})
                else:
                    retry_parts.append(normalized)
            retry_text = " ".join(retry_parts)
            if retry_text:
                retry_text += " "
            prior_text = attempt.text_content
            decision = compare_ocr_candidates(
                OCRCandidate(
                    prior_text, "paddle", attempt.page_confidence,
                    attempt.coverage_assessment.get("coverage"), True,
                ),
                OCRCandidate(
                    retry_text, "paddle", retry_confidence,
                    retry_coverage.get("coverage"), not issues,
                ),
            )
            if issues or not decision.use_candidate:
                if retry_text and retry_text != prior_text:
                    attempt.candidate_reviews.append({
                        "text": retry_text,
                        "model_language": retry_model,
                        "confidence": retry_confidence,
                        "independent_coverage": retry_coverage,
                        "decision": decision.reason if not issues else "unplaceable_dpi_candidate",
                    })
                continue
            attempt.candidate_reviews.append({
                "text": prior_text,
                "model_language": attempt.model_language,
                "confidence": attempt.page_confidence,
                "independent_coverage": attempt.coverage_assessment,
                "decision": "superseded_lower_dpi_ocr",
            })
            attempt.review_reasons.append(
                f"independent region recovery at {retry_dpi} DPI: {decision.reason}"
            )
            current_doc.close()
            current_doc = retry_doc
            retry_doc = None
            attempt.text_parts = retry_parts
            attempt.text_content = retry_text
            attempt.paddle_lines = original_pixel_lines
            attempt.page_confidence = retry_confidence
            attempt.coverage_assessment = retry_coverage
            attempt.coverage_needs_retry = bool(retry_coverage["needs_retry"])
            attempt.model_language = retry_model
            _sync_attempt_ocr_language_alias(attempt)
            attempt.preprocessing_applied = False
            attempt.gpu_optimization_applied = False
            task.render_dpi = retry_dpi
            if not attempt.coverage_needs_retry:
                break
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            attempt.review_reasons.append(
                f"independent region retry unavailable: {type(exc).__name__}"
            )
            break
        finally:
            if retry_doc is not None:
                retry_doc.close()
            if retry_image is not None:
                retry_image.close()
    return current_doc


def _recover_half_turn_paddle_page(
    task: PageTask, attempt: _WorkerAttempt, current_doc, language: str
):
    """Try upright same-engine OCR only after decisive raster direction proof."""
    if (
        attempt.engine_used != "paddle"
        or task.geometry.source_rotation != 180
        or attempt.reading_rotation != 180
        or not attempt.orientation_reference_text.strip()
        or attempt.img_np is None
    ):
        return current_doc
    candidate_doc = None
    try:
        rotated_image = np.rot90(attempt.img_np, 2).copy()
        result, actual_model = _run_pooled_paddle(
            language, rotated_image, attempt.routing_paddle_device
        )
        raw_lines = extract_paddle_lines(result)
        if not raw_lines:
            return current_doc
        detector_height, detector_width = attempt.img_np.shape[:2]
        pixel_lines, pdf_lines, issues = map_half_turn_detector_lines_to_source(
            raw_lines, task.geometry.for_raster(detector_width, detector_height)
        )
        if issues or not pdf_lines:
            attempt.candidate_reviews.append({
                "text": " ".join(line[0] for line in raw_lines),
                "model_language": actual_model,
                "decision": "unplaced_half_turn_candidate",
            })
            return current_doc
        candidate_text = " ".join(line[0] for line in pdf_lines) + " "
        def normalize(value):
            return re.findall(
                r"\w+", unicodedata.normalize("NFKC", value).casefold()
            )
        reference = normalize(attempt.orientation_reference_text)
        prior_match = SequenceMatcher(None, normalize(attempt.text_content), reference).ratio()
        candidate_match = SequenceMatcher(None, normalize(candidate_text), reference).ratio()
        confidence = _mean_ocr_confidence(pixel_lines)
        coverage = _measure_attempt_coverage(
            attempt, pixel_lines, (detector_width, detector_height)
        )
        prior_coverage = attempt.coverage_assessment.get("coverage")
        new_coverage = coverage.get("coverage")
        model_script = getattr(LANGUAGE_REGISTRY.get(language), "script", "")
        qualified = (
            candidate_match >= 0.80
            and candidate_match >= prior_match + 0.10
            and confidence >= attempt.page_confidence - 0.02
            and prior_coverage is not None
            and new_coverage is not None
            and new_coverage + 1e-9 >= prior_coverage
            and text_supports_script(candidate_text, model_script)
        )
        attempt.candidate_reviews.append({
            "text": candidate_text,
            "model_language": actual_model,
            "confidence": confidence,
            "confidence_basis": "paddle_recognition_score",
            "independent_coverage": coverage,
            "orientation_reference_agreement": candidate_match,
            "prior_orientation_reference_agreement": prior_match,
            "decision": "qualified_half_turn_candidate" if qualified else "half_turn_candidate_unselected",
        })
        if not qualified:
            return current_doc
        candidate_doc = fitz.open()
        page = candidate_doc.new_page(
            width=task.geometry.width_pt, height=task.geometry.height_pt
        )
        page.insert_image(page.rect, stream=img_to_bytes(task.image))
        parts = []
        for text, box, _score in pdf_lines:
            normalized = normalize_ocr_text(text, language) if _UNICODE_UTILS_AVAILABLE else text
            if not insert_text_line(page, normalized, box, lang_code=language):
                attempt.geometry_issues.append({
                    "text": normalized, "reason": "half-turn Paddle candidate unplaceable"
                })
                return current_doc
            parts.append(normalized)
        embedded = " ".join(parts) + " "
        _canonical_text_after_checkpoint_pdf(
            candidate_doc.write(deflate=True), ocr_text=embedded
        )
        attempt.candidate_reviews.append({
            "text": attempt.text_content,
            "model_language": attempt.model_language,
            "confidence": attempt.page_confidence,
            "confidence_basis": "paddle_recognition_score",
            "independent_coverage": attempt.coverage_assessment,
            "decision": "superseded_source_orientation_preserved_for_review",
        })
        attempt.review_reasons.append(
            "half-turn OCR copy selected by independent raster reading agreement"
        )
        current_doc.close()
        accepted = candidate_doc
        candidate_doc = None
        attempt.text_parts = parts
        attempt.text_content = embedded
        attempt.paddle_lines = pixel_lines
        attempt.page_confidence = confidence
        attempt.coverage_assessment = coverage
        attempt.coverage_needs_retry = bool(coverage.get("needs_retry"))
        attempt.model_language = actual_model
        _sync_attempt_ocr_language_alias(attempt)
        attempt.half_turn_paddle_recovery_used = True
        return accepted
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        attempt.review_reasons.append("half-turn Paddle probe unavailable: " + str(exc))
        return current_doc
    finally:
        if candidate_doc is not None:
            candidate_doc.close()


def _recover_half_turn_punctuation(
    task: PageTask, attempt: _WorkerAttempt, current_doc, language: str
):
    """Verify Paddle punctuation/spacing with one bounded source OCR reading.

    This CPU probe never edits characters. A complete alternative may replace
    Paddle only after its own confidence, visual coverage, script, PDF geometry,
    and identical ordered alphanumeric stream independently qualify it.
    """
    if (
        attempt.engine_used != "paddle" or not attempt.text_content.strip()
        or attempt.source_tesseract_probed
    ):
        return current_doc
    candidate_doc = None
    try:
        canonical, traineddata = _resolve_tesseract_route_language(language)
        if canonical != language:
            return current_doc
        rotation = 180 if attempt.half_turn_paddle_recovery_used else 0
        attempt.source_tesseract_probed = True
        pixel_lines, issues = read_source_tesseract_words(
            task.image, traineddata=traineddata, rotation=rotation,
            timeout_seconds=min(TESSERACT_TIMEOUT, 5),
        )
        source_transform = task.geometry.for_raster(*task.image.size)
        pdf_lines, mapped_issues = source_transform.map_lines_with_review(pixel_lines)
        issues.extend(mapped_issues)
        candidate_text = " ".join(text for text, _, _ in pdf_lines) + " "
        score = _mean_ocr_confidence(pixel_lines)
        script = getattr(LANGUAGE_REGISTRY.get(language), "script", "")
        candidate_audit = {
            "text": candidate_text,
            "model_language": language,
            "runtime_model": traineddata,
            "confidence": score,
            "confidence_basis": "tesseract_word_confidence_percent_scaled",
            "word_lines": [
                {"text": text, "pdf_box": box, "confidence": confidence}
                for text, box, confidence in pdf_lines
            ],
            "source": "bounded_source_raster_tesseract",
            "rotation": rotation,
            "source_raster_size": list(task.image.size),
            "recognition_raster_size": list(
                bounded_source_raster_size(task.image.size)
            ),
            "recognition_max_pixels": 2_000_000,
            "recognition_timeout_seconds": min(TESSERACT_TIMEOUT, 5),
            "page_segmentation_mode": 3,
            "decision": "source_revision_candidate_unselected",
        }
        if candidate_audit["text"]:
            attempt.candidate_reviews.append(candidate_audit)
        detector_size = (
            task.image.size if attempt.half_turn_paddle_recovery_used
            else (attempt.img_np.shape[1], attempt.img_np.shape[0])
        )
        prior_pdf_lines, prior_issues = task.geometry.for_raster(
            *detector_size
        ).map_lines_with_review(attempt.paddle_lines)
        if len(attempt.text_parts) == len(prior_pdf_lines):
            prior_pdf_lines = [
                (text, box, confidence)
                for text, (_raw, box, confidence) in zip(
                    attempt.text_parts, prior_pdf_lines
                )
            ]
        coverage = _measure_attempt_coverage(attempt, pixel_lines, task.image.size)
        decision = compare_source_grounded_text_revision(
            prior_pdf_lines, pdf_lines,
            prior_coverage=attempt.coverage_assessment,
            candidate_coverage=coverage,
            script_supported=(
                not issues and not prior_issues
                and len(pdf_lines) == len(pixel_lines)
                and len(prior_pdf_lines) == len(attempt.text_parts)
                and text_supports_script(candidate_text, script)
            ),
        )
        candidate_audit["decision"] = decision.reason
        candidate_audit["independent_coverage"] = coverage
        if not decision.use_candidate:
            if candidate_text.strip() and (
                unicodedata.normalize("NFKC", candidate_text).casefold().split()
                != unicodedata.normalize("NFKC", attempt.text_content).casefold().split()
            ):
                attempt.review_reasons.append(
                    "source OCR disagrees with selected Paddle reading: "
                    + decision.reason
                )
            return current_doc
        candidate_doc = fitz.open()
        page = candidate_doc.new_page(
            width=task.geometry.width_pt, height=task.geometry.height_pt
        )
        page.insert_image(page.rect, stream=img_to_bytes(task.image))
        parts = []
        for text, box, _confidence in pdf_lines:
            if not insert_text_line(page, text, box, lang_code=language):
                return current_doc
            parts.append(text)
        embedded = " ".join(parts) + " "
        _canonical_text_after_checkpoint_pdf(
            candidate_doc.write(deflate=True), ocr_text=embedded
        )
        attempt.candidate_reviews.append({
            "text": attempt.text_content,
            "model_language": attempt.model_language,
            "confidence": attempt.page_confidence,
            "confidence_basis": "paddle_recognition_score",
            "decision": "superseded_paddle_reading_preserved_for_review",
        })
        attempt.review_reasons.append(
            "source-mapped Tesseract punctuation/spacing reading selected with identical alphanumeric stream"
        )
        current_doc.close()
        accepted = candidate_doc
        candidate_doc = None
        attempt.text_parts = parts
        attempt.text_content = embedded
        attempt.paddle_lines = pixel_lines
        attempt.page_confidence = score
        attempt.coverage_assessment = coverage
        attempt.coverage_needs_retry = bool(coverage.get("needs_retry"))
        attempt.engine_used = "tesseract"
        attempt.status = PageStatus.TESSERACT
        attempt.ocr_method = PageStatus.TESSERACT
        attempt.model_language = language
        _sync_attempt_ocr_language_alias(attempt)
        _confirm_attempt_text_language(attempt)
        attempt.actual_execution_device = "cpu"
        attempt.gpu_optimization_applied = False
        return accepted
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        attempt.review_reasons.append(
            "source punctuation/spacing probe unavailable: " + type(exc).__name__
        )
        return current_doc
    finally:
        if candidate_doc is not None:
            candidate_doc.close()


def _recover_weak_paddle_with_tesseract(
    task: PageTask, attempt: _WorkerAttempt, current_doc, language: str
):
    """Compare one bounded cross-engine reading on weak primary evidence."""
    if (
        attempt.engine_used != "paddle"
        or not attempt.text_content.strip()
        or (
            attempt.paddle_coverage_recovery_used
            and _task_render_dpi(task) != DPI
        )
    ):
        return current_doc
    prior_label, prior_support = detect_language_from_text(attempt.text_content)
    prior_language_support = prior_support if prior_label == language else 0.0
    if not (
        attempt.page_confidence < 0.55
        or (language != "en" and prior_language_support < 0.85)
    ):
        return current_doc
    candidate_doc = None
    try:
        candidate_language, traineddata = _resolve_tesseract_route_language(language)
        if candidate_language != language:
            return current_doc
        attempt.source_tesseract_probed = True
        data = pytesseract.image_to_data(
            attempt.ocr_img, lang=traineddata,
            output_type=pytesseract.Output.DICT, timeout=TESSERACT_TIMEOUT,
        )
        pixel_lines, issues = tesseract_data_to_lines_with_review(data)
        pdf_lines, mapped_issues = tesseract_data_to_lines_with_review(
            data, transform=task.geometry.for_raster(*attempt.ocr_img.size)
        )
        issues.extend(mapped_issues)
        if not pdf_lines:
            return current_doc
        candidate_text = " ".join(line[0] for line in pdf_lines) + " "
        candidate_label, candidate_support = detect_language_from_text(candidate_text)
        script = LANGUAGE_REGISTRY[language].script if language in LANGUAGE_REGISTRY else ""
        candidate_coverage = _measure_attempt_coverage(
            attempt, pixel_lines, attempt.ocr_img.size
        )
        candidate_confidence = _mean_ocr_confidence(pixel_lines)
        decision = compare_cross_engine_candidates(
            OCRCandidate(
                attempt.text_content, "paddle", attempt.page_confidence,
                attempt.coverage_assessment.get("coverage"), True,
                prior_language_support,
            ),
            OCRCandidate(
                candidate_text, "tesseract", candidate_confidence,
                candidate_coverage.get("coverage"),
                not issues and text_supports_script(candidate_text, script),
                candidate_support if candidate_label == language else 0.0,
            ),
        )
        attempt.candidate_reviews.append({
            "text": candidate_text,
            "model_language": candidate_language,
            "confidence": candidate_confidence,
            "confidence_basis": "tesseract_word_confidence_percent_scaled",
            "independent_coverage": candidate_coverage,
            "decision": decision.reason,
        })
        if not decision.use_candidate:
            return current_doc
        candidate_doc = fitz.open()
        page = candidate_doc.new_page(
            width=task.geometry.width_pt, height=task.geometry.height_pt
        )
        page.insert_image(page.rect, stream=img_to_bytes(task.image))
        parts = []
        for word, box, _confidence in pdf_lines:
            if not insert_text_line(page, word, box, lang_code=language):
                issues.append({"text": word, "reason": "cross-engine word unplaceable"})
            else:
                parts.append(word)
        if issues or len(parts) != len(pdf_lines):
            attempt.review_reasons.append("cross-engine candidate had unplaceable spans")
            return current_doc
        embedded = " ".join(parts) + " "
        _canonical_text_after_checkpoint_pdf(
            candidate_doc.write(deflate=True), ocr_text=embedded
        )
        attempt.candidate_reviews.append({
            "text": attempt.text_content,
            "model_language": attempt.model_language,
            "confidence": attempt.page_confidence,
            "confidence_basis": "paddle_recognition_score",
            "independent_coverage": attempt.coverage_assessment,
            "decision": "superseded_primary_preserved_for_review",
        })
        attempt.review_reasons.append("cross-engine reading selected: " + decision.reason)
        current_doc.close()
        accepted_doc = candidate_doc
        candidate_doc = None
        attempt.text_parts = parts
        attempt.text_content = embedded
        attempt.paddle_lines = pixel_lines
        attempt.page_confidence = candidate_confidence
        attempt.coverage_assessment = candidate_coverage
        attempt.coverage_needs_retry = bool(candidate_coverage["needs_retry"])
        attempt.engine_used = "tesseract"
        attempt.status = PageStatus.TESSERACT
        attempt.ocr_method = PageStatus.TESSERACT
        attempt.model_language = candidate_language
        _sync_attempt_ocr_language_alias(attempt)
        _confirm_attempt_text_language(attempt)
        attempt.actual_execution_device = "cpu"
        attempt.gpu_optimization_applied = False
        return accepted_doc
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        attempt.review_reasons.append("cross-engine probe unavailable: " + str(exc))
        return current_doc
    finally:
        if candidate_doc is not None:
            candidate_doc.close()


def _record_rejected_ocr_candidate(task, attempt, reason, doc_state=None):
    """Keep raw recognized text when its searchable checkpoint is rejected."""
    rejected_text = attempt.text_content or " ".join(attempt.text_parts)
    if not rejected_text and attempt.paddle_lines:
        rejected_text = " ".join(
            str(line[0]) for line in attempt.paddle_lines
            if isinstance(line, (tuple, list)) and line
        )
    if not rejected_text.strip():
        return
    digest = hashlib.sha256(rejected_text.encode("utf-8")).hexdigest()
    if any(
        item.get("decision") == "rejected_from_searchable_checkpoint"
        and item.get("text_sha256") == digest
        for item in attempt.candidate_reviews
    ):
        return
    source_fingerprint = getattr(doc_state, "source_fingerprint", None) or {}
    attempt.candidate_reviews.append({
        "text": rejected_text,
        "text_sha256": digest,
        "source": str(attempt.ocr_method or attempt.engine_used or "recognized_ocr"),
        "source_sha256": str(source_fingerprint.get("content_sha256", "")),
        "source_generation": {
            "size_bytes": source_fingerprint.get("size_bytes"),
            "mtime_ns": source_fingerprint.get("mtime_ns"),
        },
        "document_id": task.doc_id,
        "page_num": task.page_num,
        "decision": "rejected_from_searchable_checkpoint",
        "reason": str(reason),
    })
    attempt.review_reasons.append(
        "recognized OCR rejected by PDF checkpoint validation: " + str(reason)
    )


def _write_image_only_checkpoint(
    task: PageTask,
    status: PageStatus,
    *,
    page_routing_target: str | None = None,
    geometry_issues: list | None = None,
    review_reasons: list | None = None,
    candidate_reviews: list | None = None,
):
    """Persist the pristine task pixels as a canonical one-page checkpoint."""
    image = getattr(task, "image", None)
    if image is None:
        return None, None
    width, height = task.geometry.width_pt, task.geometry.height_pt
    checkpoint_doc = fitz.open()
    try:
        page = checkpoint_doc.new_page(width=width, height=height)
        page.insert_image(fitz.Rect(0, 0, width, height), stream=img_to_bytes(image))
        checkpoint_bytes = _embed_page_checkpoint_metadata(
            checkpoint_doc.write(deflate=True),
            document_id=task.doc_id,
            page_num=task.page_num,
            status=status,
            ocr_method=PageStatus.IMAGE_ONLY.value,
            ocr_language="",
            detected_language="",
            model_language="",
            ocr_confidence=0.0,
            text_length=0,
            has_text=False,
            processing_profile=_checkpoint_processing_profile(
                actual_render_dpi=_task_render_dpi(task),
                page_routing_target=(
                    page_routing_target
                    or ("skip" if status == PageStatus.IMAGE_ONLY else "default")
                ),
                execution_device="none",
                execution_engine="ImageOnly",
                gpu_optimization_applied=False,
                preprocessing_applied=False,
            ),
            canonical_text="",
            page_geometry=task.geometry,
            geometry_issues=geometry_issues,
            review_reasons=review_reasons,
            candidate_reviews=candidate_reviews,
        )
    finally:
        checkpoint_doc.close()
    checkpoint_dir = os.path.join(TEMP_FOLDER, task.doc_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{task.page_num}.pdf")
    with _get_checkpoint_artifact_lock(task.doc_id):
        _atomic_write_bytes(checkpoint_path, checkpoint_bytes)
        _, digest = _durable_checkpoint_matching_intended_bytes(
            checkpoint_path, checkpoint_bytes
        )
    return checkpoint_path, digest


class _RouteToTesseract(RuntimeError):
    """Internal control signal for routing directly to the Tesseract branch."""


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _estimate_routing_image_bytes(image: Image.Image, width: int, height: int) -> int:
    """Estimate in-memory page image bytes when no page-image file exists."""
    try:
        bands = len(image.getbands())
    except Exception:
        bands = 3
    return max(0, int(width) * int(height) * max(1, bands))


def _analyze_routing_image_features(image: Image.Image) -> tuple[float, float, bool]:
    """Return text density, complexity, and coarse image-content hints.

    This deliberately uses cheap PIL statistics because page routing runs before
    OCR lines, table sidecars, or handwriting classifiers exist for the page.
    """
    try:
        analysis_img = image
        max_dimension = max(analysis_img.size) if analysis_img.size else 0
        if max_dimension > 256:
            scale = 256 / max_dimension
            analysis_img = image.resize(
                (
                    max(1, int(image.width * scale)),
                    max(1, int(image.height * scale)),
                )
            )
        gray = analysis_img.convert("L")
        hist = gray.histogram()
        total = max(1, gray.width * gray.height)
        dark_ratio = sum(hist[:220]) / total

        gray_stat = ImageStat.Stat(gray)
        contrast = _clamp01((gray_stat.stddev[0] if gray_stat.stddev else 0.0) / 128.0)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        edge_mean = edge_stat.mean[0] if edge_stat.mean else 0.0
        edge_density = _clamp01(edge_mean / 255.0)

        estimated_text_density = _clamp01(dark_ratio)
        complexity_score = _clamp01(
            0.15
            + (contrast * 0.35)
            + (edge_density * 1.8)
            + (estimated_text_density * 0.2)
        )

        has_images = False
        if analysis_img.mode not in ("1", "L"):
            rgb_stat = ImageStat.Stat(analysis_img.convert("RGB"))
            color_spread = max(rgb_stat.stddev) - min(rgb_stat.stddev)
            has_images = color_spread > 12.0 and edge_density > 0.03

        return estimated_text_density, complexity_score, has_images
    except Exception as exc:
        logger.debug(
            "Page routing image analysis failed; using neutral features: %s", exc
        )
        return 0.5, 0.5, False


def _build_page_routing_features(task: PageTask, width: int, height: int, dpi: int):
    text_density, complexity_score, has_images = _analyze_routing_image_features(
        task.image
    )
    features_cls = globals().get("PageFeatures")
    if features_cls is None:
        from ocr_local.infra.page_routing import PageFeatures as features_cls
    return features_cls(
        page_number=task.page_num,
        width=width,
        height=height,
        dpi=dpi,
        file_size_bytes=_estimate_routing_image_bytes(task.image, width, height),
        estimated_text_density=text_density,
        has_tables=False,
        has_images=has_images,
        is_handwritten=False,
        language=task.lang_hint or "",
        complexity_score=complexity_score,
    )


def _routing_target_value(routing_target: Any) -> str | None:
    if routing_target is None:
        return None
    return str(getattr(routing_target, "value", routing_target))


def _routing_target_is_skip(routing_target: Any) -> bool:
    return _routing_target_value(routing_target) == "skip"


def _engine_force_for_routing_target(routing_target: Any) -> str | None:
    target = _routing_target_value(routing_target)
    if target in ("gpu_paddle", "cpu_paddle"):
        return "paddle"
    if target in ("gpu_tesseract", "cpu_tesseract"):
        return "tesseract"
    return None


def _paddle_device_for_routing_target(routing_target: Any) -> str:
    target = _routing_target_value(routing_target)
    return "cpu" if target is not None and target.startswith("cpu_") else "gpu"


def _engine_force_or_fallback_for_routing_target(routing_target: Any) -> str | None:
    force = _engine_force_for_routing_target(routing_target)
    if (
        force is not None
        or routing_target is None
        or _routing_target_is_skip(routing_target)
    ):
        return force
    return "paddle"


def _routing_uses_legacy_tesseract_branch(
    routing_engine_force: str | None,
    enable_engine_registry: bool,
) -> bool:
    return routing_engine_force == "tesseract" and not enable_engine_registry


def _actual_registry_engine_language(
    engine_name: str,
    requested_language: str | None,
) -> str:
    """Return the model language actually used by a registry OCR adapter."""
    if engine_name == "tesseract":
        return _resolve_tesseract_route_language(requested_language)[0]
    if engine_name == "easyocr":
        try:
            from easyocr_engine import map_paddle_to_easyocr

            return map_paddle_to_easyocr(requested_language) or "en"
        except Exception:
            # EasyOCRAdapter uses this same fallback for an unsupported hint.
            return "en"
    return str(requested_language or "en")


def _resolve_tesseract_route_language(
    requested_language: str | None,
) -> tuple[str, str]:
    """Return the canonical OCR language and installed Tesseract model code."""
    requested = str(requested_language or "en").strip().lower()
    canonical = str(LANG_MAPPING.get(requested, requested))
    tesseract_code = get_tesseract_code(canonical)
    if tesseract_code == "eng" and canonical != "en":
        return "en", "eng"
    return canonical, tesseract_code


def _routing_target_supported_by_worker(routing_target: Any) -> bool:
    target = _routing_target_value(routing_target)
    return target in (
        None,
        "skip",
        "gpu_paddle",
        "cpu_paddle",
        "gpu_tesseract",
        "cpu_tesseract",
    )


class DocumentState:
    def __init__(self, path: str, doc_id: str, source_type: str) -> None:
        self.path: str = path
        self.doc_id: str = doc_id
        self.source_type: str = source_type
        self.total_pages: int = 0
        self.processed_pages: int = 0
        self.terminal_pages: set[int] = set()
        self.terminal_statuses: dict[int, str] = {}
        self.finalized: bool = False
        self.completion_fence = DocumentCompletionFence(doc_id)
        self.sealed_terminal_records: dict[int, object] = {}
        self.start_time: float = time.time()
        self.source_fingerprint: dict[str, Any] | None = None
        self.source_snapshot_path: str | None = None
        self.source_authority: _ImmutableSourceAuthority | None = None
        rel_stem = build_output_rel_stem(path, source_type)
        self.output_pdf: str = os.path.join(
            OUTPUT_FOLDER, "EXPORT", "PDF", rel_stem + ".pdf"
        )
        os.makedirs(os.path.dirname(self.output_pdf), exist_ok=True)

        self.output_txt_dir: str = os.path.join(
            OUTPUT_FOLDER, "EXPORT", "TEXT", os.path.dirname(rel_stem)
        )
        os.makedirs(self.output_txt_dir, exist_ok=True)

        # Temp dir for this specific doc
        self.temp_dir: str = os.path.join(TEMP_FOLDER, doc_id)
        # RESUME: Do NOT delete existing temp dir!
        os.makedirs(self.temp_dir, exist_ok=True)

        # Custody chain for forensic audit trail
        self.custody_chain: Any | None = None
        if ENABLE_CUSTODY and _CUSTODY_AVAILABLE:
            self.custody_dir: str = os.path.join(OUTPUT_FOLDER, "EXPORT", "CUSTODY")
            os.makedirs(self.custody_dir, exist_ok=True)
            self.custody_chain = CustodyChain(doc_id, path, self.custody_dir)


# Adaptive batch sizing (opt-in via ENABLE_ADAPTIVE_BATCH)
_adaptive_batch_sizer = None
if ENABLE_ADAPTIVE_BATCH:
    try:
        from ocr_local.infra.adaptive_batch import AdaptiveBatchSizer, BatchResult

        _adaptive_batch_sizer = AdaptiveBatchSizer()
        logger.info(
            "Adaptive batch sizing enabled (strategy=%s)",
            _adaptive_batch_sizer._config.strategy.value,
        )
    except ImportError:
        logger.warning(
            "adaptive_batch module not found, falling back to fixed chunk size"
        )

# --- Global State ---
doc_registry = {}  # {doc_id: DocumentState}
doc_registry_lock = (
    threading.RLock()
)  # Protects doc_registry from concurrent access (maintenance review)
model_load_lock = threading.Lock()  # Prevents race condition during model download/load

# Shared PaddleOCR engine cache (maintenance review) --
# Without this, each of the 12 GPU worker threads maintains its own
# {lang: engine} dict, resulting in O(threads x languages) engine
# instances competing for VRAM.  The shared cache ensures only ONE
# engine per language exists in VRAM across all threads.
# Each engine is paired with a threading.Lock to serialize inference
# calls, since PaddleOCR's internal CUDA session state is not
# guaranteed to be thread-safe for concurrent ocr() calls.
_ENGINE_CACHE: dict = {}  # {(device, requested_lang): _EngineCacheEntry}
_ENGINE_CACHE_LOCK = threading.Lock()
_CPU_PADDLE_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_CPU_PADDLE_EXECUTOR_LOCK = threading.Lock()
_CPU_PADDLE_INIT_LOCKS = [threading.Lock() for _ in range(CPU_PADDLE_REPLICAS)]
_CPU_PADDLE_ADMISSION = [
    threading.BoundedSemaphore(2) for _ in range(CPU_PADDLE_REPLICAS)
]
_CPU_PADDLE_ACCEPTING_WORK = True
_PADDLE_POOL_LOCK = threading.Lock()
_PADDLE_POOLS: dict[str, EngineRuntimePool] = {}
_PADDLE_POOL_ACCEPTING_WORK = True
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_standalone_pools_after_fork)
chunk_queue = queue.Queue(maxsize=CHUNK_QUEUE_SIZE)  # PDF chunks for extractors
image_queue = queue.Queue(maxsize=IMAGE_QUEUE_SIZE)  # Images for workers
assembly_queue: queue.Queue[AssemblyMessage] = queue.Queue(maxsize=RESULT_QUEUE_SIZE)
_RASTER_QUEUE_BUDGET = RasterByteBudget(
    _bounded_env_capacity("OCR_RASTER_QUEUE_BYTE_BUDGET", 512 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
)
_MAX_PAGE_RASTER_BYTES = min(
    _RASTER_QUEUE_BUDGET.limit,
    _bounded_env_capacity("OCR_MAX_PAGE_RASTER_BYTES", 256 * 1024 * 1024, 1024 * 1024 * 1024),
)
_QUEUED_FENCE_TOKENS: dict[int, tuple] = {}
_QUEUED_FENCE_TOKENS_LOCK = threading.Lock()
compression_queue = queue.Queue(
    maxsize=COMPRESSION_QUEUE_SIZE
)  # Finished PDFs for Ghostscript
extractor_process_pool = None

# Signals
stop_event = threading.Event()
hard_stop_event = threading.Event()  # Set AFTER drain completes (triggers thread exit)
finalization_done_event = threading.Event()
finalization_abort_event = threading.Event()
fatal_pipeline_error_event = threading.Event()
assembly_producers_done_event = threading.Event()
_finalization_handoff_lock = threading.RLock()
_compression_handoff_reservation_lock = threading.Lock()
COMPRESSION_HANDOFF_RESERVATION_TIMEOUT_SECONDS = 5.0


def _queue_fence_descriptor(target_queue, item):
    if target_queue is chunk_queue:
        kind, doc_id = "chunk", item[0]
    elif target_queue is image_queue:
        kind, doc_id = "page", item.doc_id
    elif target_queue is assembly_queue:
        kind, doc_id = "assembly", item["doc_id"]
    else:
        return None
    with doc_registry_lock:
        doc = doc_registry.get(doc_id)
    fence = getattr(doc, "completion_fence", None)
    if fence is None:
        return None
    return fence, kind, str(doc_id)


def _reserve_queue_obligation(target_queue, item):
    descriptor = _queue_fence_descriptor(target_queue, item)
    if descriptor is not None:
        fence, kind, generation = descriptor
        fence.reserve(kind, generation)
        with _QUEUED_FENCE_TOKENS_LOCK:
            _QUEUED_FENCE_TOKENS[id(item)] = descriptor
    return descriptor


def _complete_queue_obligation(descriptor):
    if descriptor is not None:
        fence, kind, generation = descriptor
        fence.complete(kind, generation)


def _take_queue_obligation(item):
    with _QUEUED_FENCE_TOKENS_LOCK:
        return _QUEUED_FENCE_TOKENS.pop(id(item), None)


def _cancel_queue_obligation(item, descriptor):
    if descriptor is not None:
        with _QUEUED_FENCE_TOKENS_LOCK:
            _QUEUED_FENCE_TOKENS.pop(id(item), None)
        _complete_queue_obligation(descriptor)


def _raster_task_bytes(task) -> int:
    image = task.image
    return int(image.width) * int(image.height) * max(1, len(image.getbands()))


def _check_expected_raster_bytes(geometry: PageGeometry, dpi: int) -> None:
    """Reject a page before rendering can allocate an oversized RGB raster."""
    if geometry.source_kind == "image" and geometry.source_dpi:
        width = math.ceil(geometry.width_pt * geometry.source_dpi[0] / 72.0) + 2
        height = math.ceil(geometry.height_pt * geometry.source_dpi[1] / 72.0) + 2
    else:
        width = math.ceil(geometry.width_pt * dpi / 72.0) + 2
        height = math.ceil(geometry.height_pt * dpi / 72.0) + 2
    expected = int(width) * int(height) * 3
    if expected <= 0 or expected > _MAX_PAGE_RASTER_BYTES:
        raise ValueError(
            f"page raster exceeds OCR_MAX_PAGE_RASTER_BYTES={_MAX_PAGE_RASTER_BYTES}"
        )


def _release_queued_raster(task) -> None:
    amount = int(getattr(task, "_queued_raster_bytes", 0) or 0)
    if amount:
        task._queued_raster_bytes = 0
        _RASTER_QUEUE_BUDGET.release(amount)


def _interruptible_queue_put(target_queue, item) -> bool:
    """Put with bounded waits so shutdown can stop a stalled producer."""
    reservation = _reserve_queue_obligation(target_queue, item)
    accepted = False
    try:
        if target_queue is image_queue:
            amount = _raster_task_bytes(item)
            if amount > _MAX_PAGE_RASTER_BYTES:
                raise ValueError("decoded page raster exceeds OCR_MAX_PAGE_RASTER_BYTES")
            if not _RASTER_QUEUE_BUDGET.acquire(
                amount,
                cancelled=lambda: stop_event.is_set() or hard_stop_event.is_set(),
            ):
                return False
            item._queued_raster_bytes = amount
        while not hard_stop_event.is_set():
            try:
                target_queue.put(item, timeout=0.5)
                accepted = True
                return True
            except queue.Full:
                if stop_event.is_set():
                    return False
        return False
    finally:
        if not accepted:
            if target_queue is image_queue:
                _release_queued_raster(item)
            _cancel_queue_obligation(item, reservation)


def _set_fatal_pipeline_error() -> None:
    """Serialize fatal-state transitions with final publication handoff."""
    with _finalization_handoff_lock:
        fatal_pipeline_error_event.set()


def _request_finalization_abort() -> None:
    """Atomically stop future publication and mark the pipeline fatal."""
    with _finalization_handoff_lock:
        finalization_abort_event.set()
        fatal_pipeline_error_event.set()


def _raise_on_finalization_barrier(stage: str) -> None:
    """Fail closed when publication/handoff is no longer admissible."""
    if fatal_pipeline_error_event.is_set():
        raise RuntimeError(f"fatal pipeline barrier reached at {stage}")
    if finalization_abort_event.is_set():
        raise RuntimeError(f"finalization abort barrier reached at {stage}")


def _reserve_compression_handoff_capacity() -> None:
    """Wait boundedly for queue capacity before entering the commit lock.

    Every production compression producer holds
    ``_compression_handoff_reservation_lock`` from this check through its
    final ``put_nowait``.  Consumers can only free capacity, so the reservation
    cannot be stolen by another finalizer while publication/custody commits.
    """
    timeout = max(0.0, float(COMPRESSION_HANDOFF_RESERVATION_TIMEOUT_SECONDS))
    deadline = time.monotonic() + timeout
    while compression_queue.full():
        _raise_on_finalization_barrier("compression handoff reservation")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise queue.Full(
                "compression queue capacity unavailable before publication"
            )
        time.sleep(min(0.05, remaining))


# Metrics Counters (thread-safe via locks)
global_pages_processed = 0
_pages_processed_lock = threading.Lock()
global_docs_processed = 0
_docs_processed_lock = threading.Lock()
start_time_global = time.time()

# Page Cache (opt-in via ENABLE_PAGE_CACHE)
_page_cache = None
if ENABLE_PAGE_CACHE:
    try:
        from ocr_local.infra.page_cache import PageCache

        _page_cache = PageCache()
        logger.info(
            "Page cache enabled (max_entries=%d, max_bytes=%d)",
            _page_cache._max_entries,
            _page_cache._max_size_bytes,
        )
    except ImportError:
        logger.warning("page_cache module not found, caching disabled")

# Page Routing (opt-in via ENABLE_PAGE_ROUTING)
_page_router = None
if ENABLE_PAGE_ROUTING:
    try:
        from ocr_local.infra.page_routing import PageRouter

        _page_router = PageRouter()
        logger.info("Page routing enabled")
    except ImportError:
        logger.warning("page_routing module not found, routing disabled")

# GPU Optimization (opt-in via ENABLE_GPU_OPTIMIZATION)
_gpu_optimizer = None
_batch_preprocessor = None
if ENABLE_GPU_OPTIMIZATION:
    try:
        from ocr_local.infra.gpu_optimization import BatchPreprocessor, GpuOptimizer

        _gpu_optimizer = GpuOptimizer()
        logger.info("GPU optimization enabled")
    except ImportError:
        logger.warning("gpu_optimization module not found")

# --- Threads ---


def scheduler_thread():
    """Scans files and creates Extraction Chunks."""
    threading.current_thread().name = "Scheduler"
    logger.debug("Scheduler thread started")
    logger.info("Scheduler started.")

    logger.info("Scheduler scanning source folder...")
    files = []
    skipped = 0
    scan_count = 0
    for root, _, fs in os.walk(SOURCE_FOLDER):
        scan_count += 1
        if scan_count % 100 == 0:
            logger.debug("Scanned %d folders...", scan_count)
        for f in fs:
            input_path = os.path.join(root, f)
            # Discovery is path-only.  Content classification happens only
            # after the bytes have been captured into the immutable spool.
            files.append(input_path)

    logger.info(
        "Found %d candidate source files; classification follows immutable capture.",
        len(files),
    )

    for input_path in files:
        logger.info("Processing: %s", input_path)  # Explicit Start Log
        if stop_event.is_set():
            break

        staged_source_snapshot = None
        source_authority = None
        try:
            # Stream once into a durable scheduler-owned spool.  The digest of
            # these exact bytes is the resume/custody authority; extractors
            # never reopen the mutable source path.
            (
                staged_source_snapshot,
                source_fingerprint,
                captured_header,
            ) = _stage_source_snapshot(input_path)
            source_type, detail = _classify_captured_source(
                input_path,
                captured_header,
            )
            if not source_type:
                skipped += 1
                _set_fatal_pipeline_error()
                if detail:
                    logger.warning("Source skipped: %s (%s)", input_path, detail)
                try:
                    os.remove(staged_source_snapshot)
                except OSError:
                    pass
                staged_source_snapshot = None
                continue
            if detail:
                logger.warning(
                    "Source accepted with extension fallback: %s (%s)",
                    input_path,
                    detail,
                )
            doc_id = build_resume_doc_id(input_path, source_fingerprint)
        except Exception as e:
            _set_fatal_pipeline_error()
            logger.error("Failed to capture/classify source %s: %s", input_path, e)
            log_failure(input_path, 0, f"SOURCE_CAPTURE_FAILED: {e}")
            if staged_source_snapshot:
                try:
                    os.remove(staged_source_snapshot)
                except OSError:
                    pass
            continue

        try:
            doc_state = DocumentState(input_path, doc_id, source_type)
        except Exception as exc:
            _set_fatal_pipeline_error()
            logger.error(
                "Failed to initialize document state for %s: %s",
                input_path,
                exc,
            )
            log_failure(input_path, 0, f"DOCUMENT_STATE_INIT_FAILED: {exc}")
            try:
                os.remove(staged_source_snapshot)
            except OSError:
                pass
            continue
        doc_state.source_fingerprint = dict(source_fingerprint)
        with doc_registry_lock:
            doc_registry[doc_id] = doc_state

        # 1. Get Page Count (Fast)
        try:
            doc_temp_dir = getattr(
                doc_state, "temp_dir", os.path.join(TEMP_FOLDER, doc_id)
            )
            if not hasattr(doc_state, "temp_dir"):
                doc_state.temp_dir = doc_temp_dir

            resume_state = prepare_resume_state(
                doc_temp_dir,
                input_path,
                source_fingerprint,
            )
            if resume_state["status"] == "invalidated":
                logger.warning(
                    "Resume cache invalidated for changed source: %s (ID: %s, removed=%s)",
                    input_path,
                    doc_id,
                    resume_state["removed_entries"],
                )

            source_snapshot_path = _publish_source_snapshot(
                staged_source_snapshot,
                doc_temp_dir,
                input_path,
                source_fingerprint,
            )
            staged_source_snapshot = None
            doc_state.source_snapshot_path = source_snapshot_path
            source_authority = _open_immutable_source_authority(
                source_snapshot_path,
                source_fingerprint,
            )
            doc_state.source_authority = source_authority
            source_runtime_path = source_authority.runtime_path

            # All content-derived scheduler work is performed against the
            # opened immutable generation.  The canonical snapshot pathname is
            # retained only for durable restart admission and never reopened by
            # live classification/page-count/extraction consumers.
            detected_lang = (
                detect_language(source_runtime_path) if source_type == "pdf" else "und"
            )
            doc_state.total_pages = get_source_page_count(
                source_runtime_path,
                source_type,
            )

            if doc_state.total_pages <= 0:
                _set_fatal_pipeline_error()
                reason = (
                    f"SOURCE_PAGE_COUNT_ZERO: {source_type} source reported 0 pages"
                )
                logger.error(
                    "Source rejected for processing: %s (ID: %s) [%s] -> %s",
                    input_path,
                    doc_id,
                    source_type,
                    reason,
                )
                log_failure(input_path, 0, reason)
                if doc_state.custody_chain:
                    doc_state.custody_chain.append_event(
                        "processing_failed",
                        {
                            "stage": "scheduler",
                            "error": reason,
                        },
                    )
                with doc_registry_lock:
                    if doc_id in doc_registry:
                        del doc_registry[doc_id]
                source_authority.close()
                doc_state.source_authority = None
                continue

            logger.info(
                "Scheduling: %s (ID: %s) [%s] [%s] (%d pages)",
                os.path.basename(input_path),
                doc_id,
                detected_lang,
                source_type,
                doc_state.total_pages,
            )

            # Custody: record file ingestion
            if doc_state.custody_chain:
                doc_state.custody_chain.append_event(
                    "file_ingested",
                    {
                        "source_path": input_path,
                        "source_type": source_type,
                        "total_pages": doc_state.total_pages,
                        "detected_language": detected_lang,
                        "file_hash": source_fingerprint["content_sha256"],
                        "document_id": doc_id,
                        "source_fingerprint": source_fingerprint,
                    },
                )

            # 2. Schedule Chunks - Optimized for Granular Resume
            if _adaptive_batch_sizer is not None:
                try:
                    file_size = int(source_fingerprint["size_bytes"])
                    complexity = _adaptive_batch_sizer.compute_complexity(
                        width=0,
                        height=0,  # not known at scheduler level
                        file_size=file_size,
                        dpi=DPI,
                        has_tables=False,
                        has_images=False,
                    )
                    chunk_target_size = _adaptive_batch_sizer.recommend_batch_size(
                        [complexity]
                    )
                    logger.debug(
                        "Adaptive chunk size for %s: %d (complexity=%.4f, file_size=%d)",
                        os.path.basename(input_path),
                        chunk_target_size,
                        complexity.complexity_score,
                        file_size,
                    )
                except Exception as exc:
                    logger.debug("Adaptive batch sizing failed, using default: %s", exc)
                    chunk_target_size = CHUNK_TARGET_SIZE
            else:
                chunk_target_size = CHUNK_TARGET_SIZE

            # Fast Resume: Bulk load existing pages (maintenance review)
            existing_pages: set[int] = set()
            existing_page_metadata: dict[int, dict[str, Any]] = {}
            existing_page_texts: dict[int, str] = {}
            existing_page_digests: dict[int, str] = {}
            if os.path.exists(doc_state.temp_dir):
                logger.info("Scanning existing files in %s...", doc_state.temp_dir)
                for f in os.listdir(doc_state.temp_dir):
                    if f.endswith(".pdf"):
                        try:
                            # Filename format "1.pdf"
                            page_num = int(os.path.splitext(f)[0])
                            checkpoint_path = os.path.join(doc_state.temp_dir, f)
                            if _is_valid_page_checkpoint(checkpoint_path):
                                try:
                                    (
                                        _resume_bytes,
                                        resume_digest,
                                        checkpoint_metadata,
                                        validated_text,
                                    ) = _coherent_checkpoint_snapshot(
                                        checkpoint_path,
                                        document_id=doc_id,
                                        page_num=page_num,
                                    )
                                    del _resume_bytes  # digest/text/meta are bound
                                except (OSError, RuntimeError, ValueError) as exc:
                                    logger.warning(
                                        "Discarding resume checkpoint with "
                                        "coherent snapshot failure %s: %s",
                                        checkpoint_path,
                                        exc,
                                    )
                                    try:
                                        os.remove(checkpoint_path)
                                    except FileNotFoundError:
                                        pass
                                    continue
                                existing_pages.add(page_num)
                                existing_page_metadata[page_num] = checkpoint_metadata
                                existing_page_texts[page_num] = validated_text
                                existing_page_digests[page_num] = resume_digest
                            else:
                                logger.warning(
                                    "Discarding invalid resume checkpoint: %s",
                                    checkpoint_path,
                                )
                                os.remove(checkpoint_path)
                        except Exception as exc:
                            logger.debug(
                                "Skipping non-numeric temp file %s: %s", f, exc
                            )
                logger.info("Found %d existing pages.", len(existing_pages))

            invalid_existing_pages = {
                page
                for page in existing_pages
                if page < 1 or page > doc_state.total_pages
            }
            if invalid_existing_pages:
                logger.warning(
                    "Ignoring out-of-range resume pages for %s: %s",
                    doc_id,
                    sorted(invalid_existing_pages),
                )
                existing_pages.difference_update(invalid_existing_pages)

            # Notify assembler of every RESUMED page (in order) so doc state
            # reflects on-disk progress before any new extraction is dispatched.
            for p in range(1, doc_state.total_pages + 1):
                if p in existing_pages:
                    chunk_path = os.path.join(doc_state.temp_dir, f"{p}.pdf")
                    resume_metadata = existing_page_metadata[p]
                    _interruptible_queue_put(
                        assembly_queue,
                        {
                            "doc_id": doc_id,
                            "page_num": p,
                            "text": existing_page_texts[p],
                            "status": PageStatus.RESUMED,
                            "chunk_path": chunk_path,
                            "checkpoint_sha256": existing_page_digests[p],
                            "structure_data": None,
                            "ocr_confidence": resume_metadata["ocr_confidence"],
                            "ocr_method": resume_metadata["ocr_method"],
                            "ocr_language": resume_metadata["ocr_language"],
                            "detected_language": resume_metadata.get(
                                "detected_language", ""
                            ),
                            "resume_metadata": resume_metadata,
                            "handwriting_data": None,
                            "signature_data": None,
                            "vertical_text_data": None,
                            "table_fallback_data": None,
                            "language_data": _restore_checkpoint_page_language(
                                resume_metadata
                            ),
                            "cache_hit": False,
                        },
                    )

            if existing_pages and doc_state.custody_chain:
                doc_state.custody_chain.append_event(
                    "processing_resumed",
                    {
                        "resumed_pages": sorted(existing_pages),
                        "remaining_pages": doc_state.total_pages - len(existing_pages),
                        "temp_dir": doc_state.temp_dir,
                    },
                )

            # maintenance review: compute gap chunks via set-difference semantics. The
            # helper groups missing pages into contiguous runs and splits
            # each run by chunk_target_size, guaranteeing that every
            # dispatched chunk is (a) contiguous and (b) only covers truly
            # missing pages (never re-extracts resumed pages).
            gap_chunks = compute_resume_gap_chunks(
                doc_state.total_pages,
                existing_pages,
                chunk_target_size,
            )
            for start_gap, end_gap in gap_chunks:
                _interruptible_queue_put(
                    chunk_queue,
                    (
                        doc_id,
                        source_runtime_path,
                        start_gap,
                        end_gap,
                        detected_lang,
                        source_type,
                    ),
                )
            doc_state.completion_fence.close_admission()

        except Exception as e:
            _set_fatal_pipeline_error()
            logger.error("Failed to schedule %s: %s", input_path, e)
            log_failure(input_path, 0, f"SCHEDULER_FAILED: {e}")
            with doc_registry_lock:
                if doc_id in doc_registry and doc_registry[doc_id].custody_chain:
                    doc_registry[doc_id].custody_chain.append_event(
                        "processing_failed",
                        {
                            "stage": "scheduler",
                            "error": str(e),
                        },
                    )
                if doc_id in doc_registry:
                    del doc_registry[doc_id]
            if staged_source_snapshot:
                try:
                    os.remove(staged_source_snapshot)
                except OSError:
                    pass
            if source_authority is not None:
                source_authority.close()
                doc_state.source_authority = None

    logger.info("Scheduler finished queuing all tasks.")


@dataclass
class _SalvageSource:
    """Verified raster path for extractor IMAGE_ONLY salvage. Never unverified."""

    runtime_path: str
    expected_fingerprint: dict[str, Any] | None = None
    owned_fd: int | None = None
    origin: str = ""


def _close_fd_quietly(fd) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _is_this_pid_proc_fd_path(path) -> bool:
    """True for this process's live ``/proc/{pid}/fd/{n}`` generation.

    Overlay treated every ``/proc/.../fd/...`` string as dead. This-pid
    descriptors are the sealed source while this process still holds them.
    Other-pid strings (``/proc/1/fd/999``) are not live and must not admit
    an unverified disk spool.
    """
    text = str(path or "")
    prefix = f"/proc/{os.getpid()}/fd/"
    if not text.startswith(prefix):
        return False
    suffix = text[len(prefix) :]
    return bool(suffix) and suffix.isdigit() and "/" not in suffix


def _content_fingerprint(fingerprint):
    """Return ``(sha256, size_bytes)`` or None when the digest is unusable."""
    if not isinstance(fingerprint, dict):
        return None
    digest = str(fingerprint.get("content_sha256", "") or "")
    try:
        size_bytes = int(fingerprint.get("size_bytes", -1) or -1)
    except (TypeError, ValueError):
        return None
    if size_bytes <= 0 or not _is_valid_checkpoint_sha256(digest):
        return None
    return digest, size_bytes


def _sha256_size_of_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size_bytes = 0
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        size_bytes += len(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest(), size_bytes


def _fd_matches_source_fingerprint(fd, fingerprint) -> bool:
    parsed = _content_fingerprint(fingerprint)
    if parsed is None:
        return False
    expected_digest, expected_size = parsed
    observed_digest, observed_size = _sha256_size_of_fd(fd)
    return observed_digest == expected_digest and observed_size == expected_size


def _open_readonly_fd(path, *, nofollow: bool):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if nofollow:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(str(path), flags)


def _open_regular_readonly_fd(path, *, nofollow: bool):
    fd = _open_readonly_fd(path, nofollow=nofollow)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            _close_fd_quietly(fd)
            return None
        return fd
    except Exception:
        _close_fd_quietly(fd)
        raise


def _lookup_registered_doc_state(doc_id):
    with doc_registry_lock:
        return doc_registry.get(doc_id)


def _salvage_temp_dir(doc_id, doc_state):
    if doc_state is not None:
        registered = getattr(doc_state, "temp_dir", None)
        if isinstance(registered, str) and registered:
            return registered
    return os.path.join(TEMP_FOLDER, str(doc_id))


def _salvage_expected_fingerprint(doc_id, doc_state):
    """Prefer live registry fingerprint, then resume-manifest digest."""
    candidates = []
    if doc_state is not None:
        registered = getattr(doc_state, "source_fingerprint", None)
        if isinstance(registered, dict):
            candidates.append(registered)
    manifest = _load_resume_manifest(_salvage_temp_dir(doc_id, doc_state))
    if isinstance(manifest, dict):
        manifest_fp = manifest.get("source_fingerprint")
        if isinstance(manifest_fp, dict):
            candidates.append(manifest_fp)
    for fingerprint in candidates:
        parsed = _content_fingerprint(fingerprint)
        if parsed is None:
            continue
        digest, size_bytes = parsed
        return {"content_sha256": digest, "size_bytes": size_bytes}
    return None


def _path_is_document_durable_snapshot(path, doc_id, doc_state) -> bool:
    """True when *path* is under ``{temp}/{doc_id}/.source-snapshot/``.

    Original ``DocumentState.path`` and this-pid proc fds are excluded.
    """
    text = str(path or "")
    if not text or _is_this_pid_proc_fd_path(text):
        return False
    snapshot_dir = os.path.join(
        _salvage_temp_dir(doc_id, doc_state), SOURCE_SNAPSHOT_DIRNAME
    )
    original = ""
    if doc_state is not None:
        original = str(getattr(doc_state, "path", "") or "")
    try:
        real_dir = os.path.realpath(snapshot_dir)
        real_path = os.path.realpath(text)
        if original:
            real_original = os.path.realpath(original)
            if os.path.normcase(real_path) == os.path.normcase(real_original):
                return False
        common = os.path.commonpath([real_dir, real_path])
    except (OSError, ValueError):
        return False
    return os.path.normcase(common) == os.path.normcase(real_dir)


def _durable_source_snapshot_candidates(doc_id, doc_state):
    """Directory entries only. Admission still requires fd-hash verify."""
    snapshot_dir = os.path.join(
        _salvage_temp_dir(doc_id, doc_state), SOURCE_SNAPSHOT_DIRNAME
    )
    try:
        names = os.listdir(snapshot_dir)
    except OSError:
        names = []
    candidates = []
    for name in sorted(names):
        if not name.startswith("source"):
            continue
        candidates.append(os.path.join(snapshot_dir, name))
    if doc_state is not None:
        registered = str(getattr(doc_state, "source_snapshot_path", "") or "")
        if (
            registered
            and registered not in candidates
            and _path_is_document_durable_snapshot(registered, doc_id, doc_state)
        ):
            candidates.append(registered)
    return candidates


def _runtime_path_for_held_fd(fd):
    """Raster through the held fd, not a replaceable directory entry."""
    if sys.platform.startswith("linux"):
        return f"/proc/{os.getpid()}/fd/{fd}"
    return ""


def _try_queued_runtime_salvage(doc_id, source_runtime_path, doc_state, fingerprint):
    queued = str(source_runtime_path or "")
    if not queued:
        return None
    original = ""
    if doc_state is not None:
        original = str(getattr(doc_state, "path", "") or "")
        if original:
            try:
                if os.path.normcase(os.path.realpath(queued)) == os.path.normcase(
                    os.path.realpath(original)
                ):
                    return None
            except OSError:
                pass
    this_pid = _is_this_pid_proc_fd_path(queued)
    windows_mapped = os.name == "nt" and _path_is_document_durable_snapshot(
        queued, doc_id, doc_state
    )
    if not this_pid and not windows_mapped:
        return None
    fd = None
    try:
        fd = _open_regular_readonly_fd(queued, nofollow=not this_pid)
        if fd is None:
            return None
        if fingerprint is not None and not _fd_matches_source_fingerprint(
            fd, fingerprint
        ):
            logger.warning(
                "Extractor salvage queued runtime fingerprint mismatch for %s",
                doc_id,
            )
            return None
        if this_pid:
            held = _runtime_path_for_held_fd(fd)
            if held:
                owned = fd
                fd = None
                return _SalvageSource(
                    runtime_path=held,
                    expected_fingerprint=fingerprint,
                    owned_fd=owned,
                    origin="queued_runtime",
                )
        return _SalvageSource(
            runtime_path=queued,
            expected_fingerprint=fingerprint,
            owned_fd=None,
            origin="queued_runtime",
        )
    except OSError as exc:
        logger.debug(
            "Extractor salvage queued runtime path unusable for %s: %s",
            doc_id,
            exc,
        )
        return None
    finally:
        _close_fd_quietly(fd)


def _try_restart_snapshot_salvage(doc_id, doc_state, fingerprint):
    if fingerprint is None:
        logger.debug(
            "Extractor salvage restart refused without source fingerprint for %s",
            doc_id,
        )
        return None
    for candidate in _durable_source_snapshot_candidates(doc_id, doc_state):
        fd = None
        try:
            fd = _open_regular_readonly_fd(candidate, nofollow=True)
            if fd is None:
                continue
            if not _fd_matches_source_fingerprint(fd, fingerprint):
                logger.warning(
                    "Extractor salvage durable snapshot digest mismatch for %s (%s)",
                    doc_id,
                    candidate,
                )
                continue
            runtime = _runtime_path_for_held_fd(fd)
            if runtime:
                owned = fd
                fd = None
                return _SalvageSource(
                    runtime_path=runtime,
                    expected_fingerprint=fingerprint,
                    owned_fd=owned,
                    origin="restart_snapshot",
                )
            return _SalvageSource(
                runtime_path=candidate,
                expected_fingerprint=fingerprint,
                owned_fd=None,
                origin="restart_snapshot",
            )
        except OSError as exc:
            logger.debug(
                "Extractor salvage durable snapshot open failed for %s (%s): %s",
                doc_id,
                candidate,
                exc,
            )
        finally:
            _close_fd_quietly(fd)
    return None


def _select_salvage_source(doc_id, source_runtime_path):
    """Fail-closed IMAGE_ONLY salvage selector.

    Order:
    1. Live ``source_authority.verify()`` → raster ``runtime_path``.
    2. Queued this-pid fd / still-valid ``source_runtime_path``.
    3. Restart only: resume manifest + durable snapshot fd SHA-256.
    Never reopen ``DocumentState.path``. Never treat all ``/proc/`` as dead.
    """
    doc_state = _lookup_registered_doc_state(doc_id)
    fingerprint = _salvage_expected_fingerprint(doc_id, doc_state)
    authority = getattr(doc_state, "source_authority", None) if doc_state else None
    if authority is not None and not getattr(authority, "_closed", True):
        if fingerprint is None:
            logger.warning(
                "Extractor salvage refused live authority without fingerprint for %s",
                doc_id,
            )
            return None
        try:
            authority.verify(fingerprint)
        except Exception as exc:
            if "closed" in str(exc):
                logger.debug(
                    "Extractor salvage live authority closed for %s: %s",
                    doc_id,
                    exc,
                )
            else:
                logger.warning(
                    "Extractor salvage live authority verify failed for %s: %s",
                    doc_id,
                    exc,
                )
                return None
        else:
            runtime = str(getattr(authority, "runtime_path", "") or "")
            if not runtime:
                logger.warning(
                    "Extractor salvage live authority missing runtime_path for %s",
                    doc_id,
                )
                return None
            return _SalvageSource(
                runtime_path=runtime,
                expected_fingerprint=fingerprint,
                owned_fd=None,
                origin="live_authority",
            )
    queued = _try_queued_runtime_salvage(
        doc_id, source_runtime_path, doc_state, fingerprint
    )
    if queued is not None:
        return queued
    return _try_restart_snapshot_salvage(doc_id, doc_state, fingerprint)


def _verify_salvage_runtime_path(runtime_path, expected_fingerprint) -> bool:
    if expected_fingerprint is None:
        return _is_this_pid_proc_fd_path(runtime_path)
    fd = None
    try:
        fd = _open_regular_readonly_fd(
            runtime_path,
            nofollow=not _is_this_pid_proc_fd_path(runtime_path),
        )
        if fd is None:
            return False
        return _fd_matches_source_fingerprint(fd, expected_fingerprint)
    except OSError:
        return False
    finally:
        _close_fd_quietly(fd)


def _raster_salvage_page(runtime_path, page_num, source_type):
    image = None
    try:
        geometry = _capture_source_geometries(runtime_path, page_num, page_num, source_type).get(page_num)
    except (OSError, RuntimeError, ValueError):
        logger.exception("Extractor salvage geometry failed for page %s", page_num)
        return None
    if geometry is not None:
        try:
            _check_expected_raster_bytes(geometry, DPI)
        except ValueError:
            logger.warning("Extractor salvage page %s exceeds raster byte cap", page_num)
            return None
    try:
        for p_num, frame in iter_source_images(
            runtime_path, page_num, page_num, source_type
        ):
            if p_num == page_num:
                image = frame.copy()
                break
    except Exception:
        logger.exception(
            "Extractor salvage raster failed for page %s path %s",
            page_num,
            runtime_path,
        )
        image = None
    if image is None and source_type == "pdf":
        try:
            for p_num, frame in iter_fitz_image_frames(
                runtime_path, page_num, page_num
            ):
                if p_num == page_num:
                    image = frame.copy()
                    break
        except Exception:
            logger.exception(
                "Extractor salvage fitz fallback failed for page %s path %s",
                page_num,
                runtime_path,
            )
            image = None
    return image


def _salvage_extractor_image_only(
    *,
    doc_id,
    input_path,
    page_num,
    lang_hint,
    source_type,
    source_runtime_path,
    expected_fingerprint=None,
):
    """IMAGE_ONLY checkpoint from a selector-verified runtime path only."""
    runtime = str(source_runtime_path or "")
    if not runtime:
        return None, None
    if not _verify_salvage_runtime_path(runtime, expected_fingerprint):
        logger.warning(
            "Extractor salvage refused unverified runtime path for %s page %s",
            doc_id,
            page_num,
        )
        return None, None
    image = _raster_salvage_page(runtime, page_num, source_type)
    if image is None:
        return None, None
    try:
        geometry = _capture_source_geometries(runtime, page_num, page_num, source_type).get(page_num)
        if source_type == "pdf" and geometry is None:
            raise ValueError("missing PDF page geometry")
    except Exception:
        logger.exception("Extractor salvage geometry failed for %s page %s", doc_id, page_num)
        return None, None
    task = PageTask(
        doc_id,
        input_path,
        page_num,
        image,
        lang_hint,
        source_type,
        runtime,
        geometry=geometry,
        source_prior_unverified=(source_type == "pdf"),
    )
    try:
        return _write_image_only_checkpoint(task, PageStatus.IMAGE_ONLY)
    except Exception:
        logger.exception(
            "Extractor salvage write failed for %s page %s", doc_id, page_num
        )
        return None, None


def _extractor_terminal_message(doc_id, page_num, status, chunk_path, digest):
    return {
        "doc_id": doc_id,
        "page_num": page_num,
        "text": "",
        "status": status,
        "chunk_path": chunk_path,
        "checkpoint_sha256": digest,
        "structure_data": None,
        "ocr_confidence": 0.0,
        "ocr_method": status,
        "ocr_language": "",
        "detected_language": "",
        "handwriting_data": None,
        "signature_data": None,
        "vertical_text_data": None,
        "table_fallback_data": None,
        "language_data": None,
        "resume_metadata": None,
        "cache_hit": False,
    }


def extractor_thread(thread_id):
    """Pulls Chunks, converts to Images, puts in Image Queue."""
    threading.current_thread().name = f"Extractor-{thread_id}"
    logger.debug("Extractor %d started", thread_id)

    while not hard_stop_event.is_set():
        try:
            task = chunk_queue.get(timeout=2)
            chunk_reservation = _take_queue_obligation(task)
            doc_id, source_runtime_path, start, end, lang_hint, source_type = task
        except queue.Empty:
            continue

        queued_pages = set()
        # The registry may be removed after this chunk was queued. Keep the
        # failure path self-contained so every page still receives an
        # EXTRACT_FAILED terminal instead of killing the extractor thread on
        # an unbound diagnostic variable.
        input_path = str(source_runtime_path or doc_id or "<unknown>")
        try:
            with doc_registry_lock:
                source_doc_state = doc_registry.get(doc_id)
            if source_doc_state is None:
                raise RuntimeError(f"document state missing for extraction: {doc_id}")
            input_path = source_doc_state.path
            source_geometries = _capture_source_geometries(
                source_runtime_path, start, end, source_type
            )
            if source_type == "pdf" and len(source_geometries) != end - start + 1:
                raise RuntimeError("PDF source geometry capture was incomplete")
            for geometry in source_geometries.values():
                _check_expected_raster_bytes(geometry, DPI)
            process_mode_failed = False
            if EXTRACTOR_MODE == "process" and extractor_process_pool is not None:
                try:
                    # One IPC result at a time bounds compressed payloads even
                    # when a configured chunk contains many pages.
                    for requested_page in range(start, end + 1):
                        payload_pages = extractor_process_pool.submit(
                            _extract_chunk_in_subprocess,
                            source_runtime_path,
                            requested_page,
                            requested_page,
                            source_type,
                            DPI,
                            PDF_CONVERSION_THREADS,
                        ).result()
                        for p_num, payload in payload_pages:
                            if p_num != requested_page or p_num in queued_pages:
                                continue
                            work_task = PageTask(
                                doc_id,
                                input_path,
                                p_num,
                                _decode_image_from_jpeg_bytes(payload),
                                lang_hint,
                                source_type,
                                source_runtime_path,
                                geometry=source_geometries.get(p_num),
                                source_prior_unverified=(source_type == "pdf"),
                            )
                            if not _interruptible_queue_put(image_queue, work_task):
                                raise RuntimeError("raster queue admission stopped")
                            queued_pages.add(p_num)
                    if len(queued_pages) != (end - start + 1):
                        process_mode_failed = True
                        logger.warning(
                            "Process-mode extraction returned a partial range for "
                            "%s pages %d-%d; thread fallback will fill only missing pages.",
                            doc_id,
                            start,
                            end,
                        )
                except Exception as proc_err:
                    logger.warning(
                        "Process-mode extraction failed for %s pages %d-%d (%s); "
                        "falling back to thread-mode for this chunk.",
                        doc_id,
                        start,
                        end,
                        proc_err,
                    )
                    process_mode_failed = True

            if EXTRACTOR_MODE == "thread" or process_mode_failed:
                for p_num, img in iter_source_images(
                    source_runtime_path,
                    start,
                    end,
                    source_type,
                ):
                    if p_num > end:
                        break
                    if p_num in queued_pages:
                        continue
                    # iter_source_images owns and closes its yielded frame when
                    # the generator advances. The queued task needs an
                    # independent image object because OCR runs asynchronously.
                    work_task = PageTask(
                        doc_id,
                        input_path,
                        p_num,
                        img.copy(),
                        lang_hint,
                        source_type,
                        source_runtime_path,
                        geometry=source_geometries.get(p_num),
                        source_prior_unverified=(source_type == "pdf"),
                    )
                    if not _interruptible_queue_put(image_queue, work_task):
                        raise RuntimeError("raster queue admission stopped")
                    queued_pages.add(p_num)

            expected = end - start + 1
            if len(queued_pages) != expected:
                missing = [
                    str(p) for p in range(start, end + 1) if p not in queued_pages
                ]
                raise RuntimeError(
                    f"Incomplete extraction for {source_type} range {start}-{end}; "
                    f"queued={len(queued_pages)}/{expected}; missing={','.join(missing)}"
                )

        except Exception as e:
            logger.error(
                "Extraction failed for %s %d-%d [%s]: %s",
                doc_id,
                start,
                end,
                source_type,
                e,
            )
            salvage = None
            try:
                salvage = _select_salvage_source(doc_id, source_runtime_path)
            except Exception:
                logger.exception(
                    "Extractor salvage source selection failed for %s", doc_id
                )
                salvage = None
            try:
                for p_num in range(start, end + 1):
                    if p_num in queued_pages:
                        continue
                    salvaged_path = None
                    salvaged_digest = None
                    if salvage is not None:
                        salvaged_path, salvaged_digest = _salvage_extractor_image_only(
                            doc_id=doc_id,
                            input_path=input_path,
                            page_num=p_num,
                            lang_hint=lang_hint,
                            source_type=source_type,
                            source_runtime_path=salvage.runtime_path,
                            expected_fingerprint=salvage.expected_fingerprint,
                        )
                    if salvaged_path:
                        _interruptible_queue_put(
                            assembly_queue,
                            _extractor_terminal_message(
                                doc_id,
                                p_num,
                                PageStatus.IMAGE_ONLY,
                                salvaged_path,
                                salvaged_digest,
                            ),
                        )
                        log_failure(
                            input_path,
                            p_num,
                            f"EXTRACT_SALVAGED_IMAGE_ONLY: {e}",
                        )
                        continue
                    _interruptible_queue_put(
                        assembly_queue,
                        _extractor_terminal_message(
                            doc_id,
                            p_num,
                            PageStatus.EXTRACT_FAILED,
                            None,
                            None,
                        ),
                    )
                    log_failure(input_path, p_num, f"EXTRACT_FAILED: {e}")
            finally:
                if salvage is not None:
                    _close_fd_quietly(salvage.owned_fd)
                    salvage.owned_fd = None
        finally:
            chunk_queue.task_done()
            _complete_queue_obligation(chunk_reservation)


def worker_thread(worker_id):
    """Consumes PageTasks, runs OCR, saves temp chunks."""
    global global_pages_processed
    threading.current_thread().name = f"GPU-Worker-{worker_id}"
    logger.debug("Worker %d thread started", worker_id)
    logger.info("GPU Worker %d started.", worker_id)

    # Engine cache is now module-level (_ENGINE_CACHE, maintenance review).
    # Each entry is (engine, inference_lock) or None.

    # Document Intelligence — PPStructure engine (Phase 3A)
    structure_engine = None
    if ENABLE_DOCUMENT_INTELLIGENCE:
        try:
            from paddleocr import PPStructure

            use_table = ENABLE_TABLE_EXTRACTION and DOCINTEL_MODE in (
                "tables_only",
                "full",
            )
            use_layout = ENABLE_LAYOUT_ANALYSIS and DOCINTEL_MODE in (
                "layout_only",
                "full",
            )
            structure_engine = PPStructure(
                show_log=False,
                use_gpu=(_DEFAULT_PADDLE_DEVICE == "gpu"),
                table=use_table,
                layout=use_layout,
                ocr=True,
                image_orientation=True,
            )
            logger.info(
                "Worker %d: PPStructure engine initialized (mode=%s)",
                worker_id,
                DOCINTEL_MODE,
            )
        except Exception as e:
            logger.warning(
                "Worker %d: PPStructure init failed, DocIntel disabled: %s",
                worker_id,
                e,
            )
            structure_engine = None

    _inline_retry_task = None
    _inline_retry_reservation = None
    # An inline DPI retry still owns the queue's original unfinished-task
    # obligation.  Drain it even if shutdown is requested between attempts.
    while _inline_retry_task is not None or not hard_stop_event.is_set():
        if _inline_retry_task is not None:
            task = _inline_retry_task
            page_reservation = _inline_retry_reservation
            _inline_retry_task = None
            _inline_retry_reservation = None
            _pdf_path = task.doc_path
        else:
            try:
                task = image_queue.get(timeout=3)
                page_reservation = _take_queue_obligation(task)
                _release_queued_raster(task)
                _pdf_path = task.doc_path
            except queue.Empty:
                continue

        attempt = _bind_worker_attempt(task)
        _dpi_requeued = (
            False  # Track DPI escalation re-queue to avoid premature task_done()
        )
        _dpi_retry_inline = False
        try:
            _on_worker_attempt_ready(attempt)
            if hasattr(task, "image") and task.image is not None:
                attempt.w, attempt.h = task.image.size

            # --- Page Cache: check before OCR ---
            if _page_cache is not None:
                _cache_key = f"{task.doc_id}:{task.page_num}:{DPI}"
                _cached_bytes = _page_cache.get(_cache_key)
                if _cached_bytes is not None:
                    with doc_registry_lock:
                        _cache_doc_state = doc_registry.get(task.doc_id)
                    if _cache_doc_state is not None:
                        _cache_chunk = os.path.join(
                            _cache_doc_state.temp_dir, f"{task.page_num}.pdf"
                        )
                        try:
                            (
                                _cache_durable_bytes,
                                _cache_digest,
                                _cached_checkpoint_meta,
                                _cached_text,
                            ) = _publish_or_select_cached_checkpoint(
                                _cache_chunk,
                                _cached_bytes,
                                document_id=task.doc_id,
                                page_num=task.page_num,
                                expected_geometry=task.geometry,
                            )
                        except (OSError, RuntimeError, TypeError, ValueError) as exc:
                            logger.info(
                                "Bypassing incompatible page cache entry %s: %s",
                                _cache_key,
                                exc,
                            )
                            _page_cache.remove(_cache_key)
                        else:
                            attempt.chunk_path = _cache_chunk
                            attempt.checkpoint_sha256 = _cache_digest
                            attempt.text_content = _cached_text
                            attempt.status = _cached_checkpoint_meta["status"]
                            attempt.searchable_checkpoint_written = (
                                _cache_hit_arms_keep_searchable(
                                    _cached_checkpoint_meta["status"]
                                )
                            )
                            _interruptible_queue_put(
                                assembly_queue,
                                {
                                    "doc_id": task.doc_id,
                                    "page_num": task.page_num,
                                    "text": attempt.text_content,
                                    "status": attempt.status,
                                    "chunk_path": attempt.chunk_path,
                                    "checkpoint_sha256": attempt.checkpoint_sha256,
                                    "ocr_confidence": _cached_checkpoint_meta[
                                        "ocr_confidence"
                                    ],
                                    "ocr_method": _cached_checkpoint_meta["ocr_method"],
                                    "ocr_language": _cached_checkpoint_meta[
                                        "ocr_language"
                                    ],
                                    "detected_language": _cached_checkpoint_meta.get(
                                        "detected_language", ""
                                    ),
                                    "structure_data": None,
                                    "handwriting_data": None,
                                    "signature_data": None,
                                    "vertical_text_data": None,
                                    "table_fallback_data": None,
                                    "language_data": _restore_checkpoint_page_language(
                                        _cached_checkpoint_meta
                                    ),
                                    # Carry exact embedded provenance so validation
                                    # never falls back to a stale sidecar.
                                    "resume_metadata": _cached_checkpoint_meta,
                                    "cache_hit": True,
                                },
                            )
                            # task_done/cleanup handled by finally block
                            continue

            # --- OCR LOGIC ---
            # (Paddle -> Tesseract -> Image)

            # --- Page Routing (opt-in) ---
            # Route with the DPI that produced the attached image, not the global
            # DPI constant (which can diverge after escalation).
            if _page_router is not None:
                try:
                    attempt.routing_features = _build_page_routing_features(
                        task,
                        attempt.w,
                        attempt.h,
                        _task_render_dpi(task),
                    )
                    decision = _page_router.route_page(attempt.routing_features)
                    attempt.routing_target = decision.target
                    attempt.routing_engine_force = (
                        _engine_force_or_fallback_for_routing_target(
                            attempt.routing_target
                        )
                    )
                    attempt.routing_paddle_device = _paddle_device_for_routing_target(
                        attempt.routing_target
                    )
                    if (
                        attempt.routing_paddle_device == "gpu"
                        and _DEFAULT_PADDLE_DEVICE == "cpu"
                    ):
                        logger.info(
                            "Page routing requested GPU Paddle without an available "
                            "GPU; using the CPU predictor owner thread"
                        )
                        attempt.routing_paddle_device = "cpu"
                    logger.debug(
                        "Page %d routed to %s: %s (density=%.3f complexity=%.3f)",
                        task.page_num,
                        decision.target.value,
                        decision.reason,
                        attempt.routing_features.estimated_text_density,
                        attempt.routing_features.complexity_score,
                    )
                except Exception as exc:
                    logger.debug("Page routing failed, using default path: %s", exc)
                    attempt.routing_target = None  # Fall through to default path
                    attempt.routing_features = None
                    attempt.routing_engine_force = None
                    attempt.routing_paddle_device = _DEFAULT_PADDLE_DEVICE

            if _routing_target_is_skip(attempt.routing_target):
                logger.info(
                    "Page %d of %s skipped by routing (too small)",
                    task.page_num,
                    task.doc_id,
                )
                _clear_attempt_language_fields(attempt)
                attempt.chunk_path, attempt.checkpoint_sha256 = (
                    _write_image_only_checkpoint(task, PageStatus.IMAGE_ONLY)
                )
                _interruptible_queue_put(
                    assembly_queue,
                    {
                        "doc_id": task.doc_id,
                        "page_num": task.page_num,
                        "text": "",
                        "status": PageStatus.IMAGE_ONLY,
                        "chunk_path": attempt.chunk_path,
                        "checkpoint_sha256": attempt.checkpoint_sha256,
                        "structure_data": None,
                        "ocr_confidence": 0.0,
                        "ocr_method": PageStatus.IMAGE_ONLY,
                        "ocr_language": "",
                        "detected_language": "",
                        "handwriting_data": None,
                        "signature_data": None,
                        "vertical_text_data": None,
                        "table_fallback_data": None,
                        "language_data": None,
                        "resume_metadata": None,
                        "cache_hit": False,
                    },
                )
                # task_done/cleanup handled by finally block
                continue

            prior_label = str(task.lang_hint or "und").strip().lower()
            script_evidence = _page_script_evidence(task.image, prior_label)
            attempt.language_route = route_language(
                "und" if task.source_prior_unverified else prior_label,
                explicit_hint=task.explicit_lang_hint,
                script_evidence=script_evidence,
                prior_hint=prior_label if task.source_prior_unverified else None,
            )
            requested_language = attempt.language_route.model_language
            attempt.detected_language = attempt.language_route.detected_label
            if attempt.language_route.needs_review:
                attempt.review_reasons.append(
                    f"language route: {attempt.language_route.reason}"
                )
            if attempt.language_route.engine == "tesseract":
                attempt.routing_engine_force = "tesseract"
                logger.info(
                    "PaddleOCR has no model for language %s; routing %s p%d to "
                    "the language-specific Tesseract model",
                    requested_language,
                    task.doc_id,
                    task.page_num,
                )

            if not _routing_target_supported_by_worker(attempt.routing_target):
                logger.warning(
                    "Page %d routing target %s is not implemented in this worker; using default Paddle path",
                    task.page_num,
                    _routing_target_value(attempt.routing_target),
                )

            # Phase 4D: Optional image preprocessing
            # Use a separate variable so task.image stays pristine for
            # forensic image-only fallback (original pixels are evidence).
            attempt.ocr_img = task.image
            if ENABLE_PREPROCESSING:
                try:
                    from preprocessing import preprocess_for_ocr

                    # The searchable PDF embeds the pristine source image.
                    # Geometry-changing preprocessing would therefore move OCR
                    # boxes away from their visual text unless it also returned
                    # an inverse transform. Keep OCR enhancement geometry-safe.
                    attempt.ocr_img = preprocess_for_ocr(
                        task.image,
                        level=PREPROCESSING_LEVEL,
                        preserve_geometry=True,
                    )
                    attempt.preprocessing_applied = True
                except Exception as preproc_exc:
                    attempt.preprocessing_applied = False
                    logger.warning(
                        "Preprocessing failed for %s p%d: %s",
                        task.doc_id,
                        task.page_num,
                        preproc_exc,
                    )

            # [Paddle Attempt]
            temp_doc = None
            _engine_custody_events = []
            try:
                if _routing_uses_legacy_tesseract_branch(
                    attempt.routing_engine_force,
                    ENABLE_ENGINE_REGISTRY,
                ):
                    raise _RouteToTesseract("page routing selected Tesseract")

                if _batch_preprocessor is not None:
                    try:
                        processed = _batch_preprocessor.preprocess_batch(
                            [attempt.ocr_img]
                        )
                        if processed:
                            attempt.img_np = processed[0]
                            attempt.gpu_optimization_applied = True
                        else:
                            attempt.img_np = np.array(attempt.ocr_img)
                            attempt.gpu_optimization_applied = False
                    except Exception as exc:
                        logger.debug(
                            "Batch preprocessing failed, using raw image: %s", exc
                        )
                        attempt.img_np = np.array(attempt.ocr_img)
                        attempt.gpu_optimization_applied = False
                else:
                    attempt.img_np = np.array(attempt.ocr_img)
                    attempt.gpu_optimization_applied = False

                def _paddle_default_lines():
                    result, attempt.model_language = _run_pooled_paddle(
                        requested_language,
                        attempt.img_np,
                        attempt.routing_paddle_device,
                    )
                    return extract_paddle_lines(result)

                if ENABLE_ENGINE_REGISTRY:
                    attempt.engine_used, attempt.paddle_lines = _resolve_engine_lines(
                        image=attempt.ocr_img,
                        image_np=attempt.img_np,
                        lang_code=requested_language,
                        registry=_ENGINE_REGISTRY,
                        paddle_runner=_paddle_default_lines,
                        page_id=f"{task.doc_id}:{task.page_num}",
                        custody_event_sink=_engine_custody_events.append,
                        force=attempt.routing_engine_force,
                        is_handwritten=(
                            bool(attempt.routing_features.is_handwritten)
                            if attempt.routing_features is not None
                            else False
                        ),
                    )
                else:
                    attempt.engine_used, attempt.paddle_lines = (
                        "paddle",
                        _paddle_default_lines(),
                    )

                # Rebuild Page
                temp_doc = fitz.open()
                page = temp_doc.new_page(
                    width=task.geometry.width_pt, height=task.geometry.height_pt
                )
                page.insert_image(
                    page.rect,
                    stream=img_to_bytes(task.image),
                )

                has_text = False
                if _UNICODE_UTILS_AVAILABLE:
                    attempt.paddle_lines = reorder_cjk_vertical_lines(
                        attempt.paddle_lines, requested_language
                    )

                # Detector coordinates belong to the actual image sent to the
                # selected engine.  GPU preprocessing can resize it separately
                # from the pristine background raster.
                if attempt.engine_used == "paddle":
                    detector_height, detector_width = attempt.img_np.shape[:2]
                else:
                    detector_width, detector_height = attempt.ocr_img.size
                pdf_lines, geometry_issues = task.geometry.for_raster(
                    detector_width, detector_height
                ).map_lines_with_review(attempt.paddle_lines)
                if attempt.engine_used == "paddle":
                    attempt.reading_rotation = task.geometry.source_rotation
                    if (
                        attempt.reading_rotation == 180
                        and _needs_half_turn_direction_probe(pdf_lines)
                    ):
                        orientation_evidence = _half_turn_direction_evidence(
                            task.image, requested_language
                        )
                        qualified = (
                            orientation_evidence[0]
                            if orientation_evidence is not None else None
                        )
                        if orientation_evidence is not None:
                            attempt.orientation_reference_text = orientation_evidence[1]
                        attempt.reading_rotation = qualified or 0
                        if qualified is None:
                            attempt.review_reasons.append(
                                "half-turn glyph direction unresolved by bounded raster OCR"
                            )
                attempt.geometry_issues.extend(geometry_issues)
                if geometry_issues:
                    logger.warning(
                        "Page %s p%d has %d unplaceable OCR spans; review required",
                        task.doc_id, task.page_num, len(geometry_issues),
                    )
                for span_index, (txt, box, _conf) in enumerate(pdf_lines):
                    try:
                        normalized_txt = (
                            normalize_ocr_text(txt, requested_language)
                            if _UNICODE_UTILS_AVAILABLE
                            else txt
                        )
                        if insert_text_line(
                            page, normalized_txt, box, lang_code=requested_language,
                            source_rotation=attempt.reading_rotation,
                        ):
                            attempt.text_parts.append(normalized_txt)
                            has_text = True
                        else:
                            attempt.geometry_issues.append({
                                "index": span_index, "text": normalized_txt,
                                "reason": "hidden text could not fit detector box",
                            })
                    except Exception as insert_error:
                        logger.error("Text insertion error: %s", insert_error)
                        attempt.geometry_issues.append({
                            "index": span_index, "text": str(txt),
                            "reason": f"hidden text insertion error: {insert_error}",
                        })

                if attempt.geometry_issues:
                    raise _RouteToTesseract(
                        "Paddle detector returned unplaceable spans; trying a "
                        "fresh geometry-bearing fallback OCR"
                    )

                # Compute page-level mean confidence from the OCR scores
                attempt.page_confidence = _mean_ocr_confidence(attempt.paddle_lines)
                # Provenance: label ocr_method/status from the engine ACTUALLY used,
                # not a hardcoded "PaddleOCR" -- a registry-routed Tesseract/EasyOCR
                # page must not be recorded as PaddleOCR (forensic provenance integrity).
                attempt.ocr_method, _engine_status_value = _engine_provenance(
                    attempt.engine_used
                )
                _engine_status = PageStatus(_engine_status_value)
                if attempt.engine_used == "paddle":
                    _sync_attempt_ocr_language_alias(attempt)
                    attempt.actual_execution_device = str(
                        attempt.routing_paddle_device or _DEFAULT_PADDLE_DEVICE
                    )
                    if attempt.actual_execution_device not in ("cpu", "gpu"):
                        attempt.actual_execution_device = "cpu"
                else:
                    attempt.model_language = _actual_registry_engine_language(
                        attempt.engine_used,
                        requested_language,
                    )
                    _sync_attempt_ocr_language_alias(attempt)
                    # Registry non-Paddle engines realize on CPU.  If batch GPU
                    # preprocessing already produced img_np that was passed into
                    # the registry runner, preserve gpu_optimization_applied.
                    attempt.actual_execution_device = "cpu"

                # maintenance review: materialize text_content from accumulated parts
                attempt.text_content = " ".join(attempt.text_parts)
                if attempt.text_content:
                    attempt.text_content += " "
                attempt.coverage_assessment = _measure_attempt_coverage(
                    attempt,
                    attempt.paddle_lines,
                    (attempt.img_np.shape[1], attempt.img_np.shape[0]),
                )
                attempt.coverage_needs_retry = bool(
                    attempt.coverage_assessment["needs_retry"]
                )

                if has_text:
                    # --- ADAPTIVE TWO-PASS LOGIC ---
                    # If FastText detects a language different from the initial hint,
                    # we re-run OCR with the correct PaddleOCR language model. img_np
                    # is reused (no redundant numpy conversion). An accepted rerun
                    # replaces attempt.paddle_lines with the rerun lines so
                    # downstream processing (handwriting, table fallback, checkpoint)
                    # describes the second pass, not the discarded first.
                    # Two-pass re-OCR is PaddleOCR-specific: skip it for a non-paddle
                    # registry engine so the selected engine's text/provenance label
                    # are not overwritten by a Paddle re-run.
                    if attempt.engine_used == "paddle":
                        detected_code, conf = detect_language_from_text(
                            attempt.text_content
                        )
                    else:
                        detected_code, conf = None, 0.0

                    # LOG FLAVOR (User Request)
                    if detected_code and conf >= 0.7:
                        attempt.detected_language = detected_code
                        attempt.detected_language_confidence = conf
                        logger.info(
                            "Page Flavor %s p%d: %s (%.2f)",
                            task.doc_id,
                            task.page_num,
                            detected_code,
                            conf,
                        )
                    elif detected_code:
                        attempt.review_reasons.append(
                            f"unconfirmed text-language candidate {detected_code} ({conf:.2f})"
                        )

                    if (
                        detected_code
                        and detected_code != requested_language
                        and not is_paddle_language_supported(detected_code)
                    ):
                        attempt.review_reasons.append(
                            "detected language unsupported by Paddle; prior OCR "
                            "retained pending independent candidate comparison"
                        )

                    if (
                        detected_code
                        and detected_code != requested_language
                        and is_paddle_language_supported(detected_code)
                    ):
                        logger.info(
                            "Re-running %s p%d: Detected %s (was %s)",
                            task.doc_id,
                            task.page_num,
                            detected_code,
                            requested_language,
                        )

                        # Re-Run with Correct Language
                        result_2, rerun_model_language = _run_pooled_paddle(
                            detected_code,
                            attempt.img_np,
                            attempt.routing_paddle_device,
                        )
                        rerun_lines = extract_paddle_lines(result_2)
                        if _UNICODE_UTILS_AVAILABLE:
                            rerun_lines = reorder_cjk_vertical_lines(
                                rerun_lines, detected_code
                            )
                        if rerun_lines:
                            # maintenance review: guarantee rerun_doc is closed on every exit path
                            # (including exceptions in new_page/insert_image/insert_text_line)
                            # unless ownership is transferred to temp_doc.
                            rerun_doc = fitz.open()
                            _rerun_doc_handed_off = False
                            try:
                                rerun_page = rerun_doc.new_page(
                                    width=task.geometry.width_pt,
                                    height=task.geometry.height_pt,
                                )
                                rerun_page.insert_image(
                                    rerun_page.rect,
                                    stream=img_to_bytes(task.image),
                                )

                                _rerun_parts: list[str] = []
                                rerun_pdf_lines, rerun_issues = task.geometry.for_raster(
                                    attempt.img_np.shape[1], attempt.img_np.shape[0]
                                ).map_lines_with_review(rerun_lines)
                                attempt.geometry_issues.extend(rerun_issues)
                                for span_index, (txt, box, _conf) in enumerate(rerun_pdf_lines):
                                    try:
                                        normalized_txt = (
                                            normalize_ocr_text(txt, detected_code)
                                            if _UNICODE_UTILS_AVAILABLE
                                            else txt
                                        )
                                        if insert_text_line(
                                            rerun_page,
                                            normalized_txt,
                                            box,
                                            lang_code=detected_code,
                                            source_rotation=attempt.reading_rotation,
                                        ):
                                            _rerun_parts.append(normalized_txt)
                                        else:
                                            attempt.geometry_issues.append({
                                                "index": span_index,
                                                "text": normalized_txt,
                                                "reason": "rerun hidden text could not fit detector box",
                                            })
                                    except Exception as insert_error:
                                        logger.error(
                                            "Re-run text insertion error: %s",
                                            insert_error,
                                        )
                                        attempt.geometry_issues.append({
                                            "index": span_index, "text": str(txt),
                                            "reason": f"rerun text insertion error: {insert_error}",
                                        })

                                rerun_text = " ".join(_rerun_parts)
                                if rerun_text:
                                    rerun_text += " "
                                candidate_confidence = _mean_ocr_confidence(rerun_lines)
                                rerun_coverage = _measure_attempt_coverage(
                                    attempt,
                                    rerun_lines,
                                    (attempt.img_np.shape[1], attempt.img_np.shape[0]),
                                )
                                candidate_detected_code, candidate_language_conf = (
                                    detect_language_from_text(rerun_text)
                                )
                                rerun_decision = compare_ocr_candidates(
                                    OCRCandidate(
                                        text=attempt.text_content,
                                        engine="paddle",
                                        confidence=attempt.page_confidence,
                                        coverage=attempt.coverage_assessment.get("coverage"),
                                        script_supported=True,
                                        language_support=(
                                            conf if detected_code == requested_language else 0.0
                                        ),
                                    ),
                                    OCRCandidate(
                                        text=rerun_text,
                                        engine="paddle",
                                        confidence=candidate_confidence,
                                        coverage=rerun_coverage.get("coverage"),
                                        script_supported=True,
                                        language_support=(
                                            candidate_language_conf
                                            if candidate_detected_code == detected_code
                                            else 0.0
                                        ),
                                    ),
                                )
                                if rerun_decision.needs_review:
                                    attempt.review_reasons.append(
                                        f"language rerun: {rerun_decision.reason}"
                                    )
                                    attempt.candidate_reviews.append({
                                        "text": rerun_text,
                                        "model_language": rerun_model_language,
                                        "confidence": candidate_confidence,
                                        "independent_coverage": rerun_coverage,
                                        "decision": rerun_decision.reason,
                                    })
                                if rerun_decision.use_candidate and rerun_text.strip():
                                    if rerun_decision.needs_review:
                                        attempt.candidate_reviews.append({
                                            "text": attempt.text_content,
                                            "model_language": requested_language,
                                            "confidence": attempt.page_confidence,
                                            "independent_coverage": attempt.coverage_assessment,
                                            "decision": "superseded_prior_preserved_for_review",
                                        })
                                    temp_doc.close()
                                    temp_doc = rerun_doc
                                    _rerun_doc_handed_off = True
                                    attempt.text_content = rerun_text
                                    # All downstream analyses and checkpoint
                                    # payloads must describe the accepted
                                    # second pass, not the discarded first.
                                    attempt.paddle_lines = rerun_lines
                                    attempt.status = f"Paddle-{rerun_model_language}"
                                    attempt.model_language = rerun_model_language
                                    if (
                                        candidate_detected_code == detected_code
                                        and candidate_language_conf >= 0.7
                                    ):
                                        attempt.detected_language = detected_code
                                        attempt.detected_language_confidence = candidate_language_conf
                                    _sync_attempt_ocr_language_alias(attempt)
                                    # Update confidence from rerun
                                    attempt.page_confidence = candidate_confidence
                                    attempt.coverage_assessment = rerun_coverage
                                    attempt.coverage_needs_retry = bool(
                                        rerun_coverage["needs_retry"]
                                    )
                            finally:
                                if not _rerun_doc_handed_off:
                                    try:
                                        rerun_doc.close()
                                    except Exception:
                                        pass

                    selected_language = (
                        detected_code
                        if attempt.status.startswith("Paddle-") and detected_code
                        else requested_language
                    )
                    temp_doc = _recover_half_turn_paddle_page(
                        task, attempt, temp_doc, selected_language
                    )
                    temp_doc = _recover_incomplete_paddle_page(
                        task, attempt, temp_doc, selected_language
                    )
                    if (
                        attempt.ocr_method != PageStatus.TESSERACT
                        and not attempt.status.startswith("Paddle-")
                    ):  # no two-pass rerun or accepted cross-engine recovery
                        attempt.status = _engine_status
                    temp_doc = _recover_weak_paddle_with_tesseract(
                        task, attempt, temp_doc, selected_language
                    )
                    temp_doc = _recover_half_turn_punctuation(
                        task, attempt, temp_doc, selected_language
                    )
                    attempt.pdf_bytes = temp_doc.write(deflate=True)
                    temp_doc.close()
                else:
                    raise Exception("No text")
            except _EngineAdmissionDenied as e:
                try:
                    if temp_doc:
                        temp_doc.close()
                except Exception as exc:
                    logger.debug(
                        "Failed to close temp_doc after engine admission denial: %s",
                        exc,
                    )
                with doc_registry_lock:
                    _denied_doc_state = doc_registry.get(task.doc_id)
                if _denied_doc_state and _denied_doc_state.custody_chain:
                    for _engine_event in _engine_custody_events:
                        _engine_event_data = _engine_event.copy()
                        _engine_event_type = _engine_event_data.pop(
                            "event", "ENGINE_PROVENANCE"
                        )
                        _denied_doc_state.custody_chain.append_event(
                            _engine_event_type,
                            _engine_event_data,
                        )
                logger.error(
                    "OCR engine admission denied for %s p%d: %s",
                    task.doc_id,
                    task.page_num,
                    e,
                )
                raise
            except Exception as e:
                # maintenance review: detect CUDA/GPU OOM and attempt recovery before fallback
                _err_str = str(e).lower()
                _is_oom = not isinstance(e, _RouteToTesseract) and (
                    "out of memory" in _err_str
                    or "cuda" in type(e).__name__.lower()
                    or "outofmemory" in type(e).__name__.lower()
                    or "cannot allocate" in _err_str
                )
                if _is_oom:
                    logger.error(
                        "GPU OOM on %s p%s — clearing CUDA cache and evicting engine",
                        task.doc_id,
                        task.page_num,
                    )
                    try:
                        import paddle

                        paddle.device.cuda.empty_cache()
                    except Exception as exc:
                        logger.debug("CUDA cache clear failed: %s", exc)
                    # Evict broken engine from shared cache so next request recreates it
                    try:
                        _evict_engine_cache_entry(
                            attempt.active_paddle_cache_key,
                            attempt.active_paddle_cache_entry,
                        )
                    except Exception as exc:
                        logger.debug("Engine cache eviction failed: %s", exc)
                try:
                    if temp_doc:
                        temp_doc.close()
                except Exception as exc:
                    logger.debug(
                        "Failed to close temp_doc after PaddleOCR error: %s", exc
                    )
                if isinstance(e, _RouteToTesseract):
                    logger.info(
                        "Page routing selected Tesseract for %s p%d",
                        task.doc_id,
                        task.page_num,
                    )
                else:
                    logger.warning(
                        "PaddleOCR Failed for %s p%d: %s", task.doc_id, task.page_num, e
                    )
                # [Tesseract Attempt]
                try:
                    temp_doc = None
                    tess_canonical_language, tess_traineddata_code = (
                        _resolve_tesseract_route_language(
                            requested_language
                        )
                    )
                    data = pytesseract.image_to_data(
                        attempt.ocr_img,
                        lang=tess_traineddata_code,
                        output_type=pytesseract.Output.DICT,
                        timeout=TESSERACT_TIMEOUT,
                    )
                    source_transform = task.geometry.for_raster(*attempt.ocr_img.size)
                    tess_pixel_lines, tess_issues = tesseract_data_to_lines_with_review(data)
                    tess_lines, _mapped_issues = tesseract_data_to_lines_with_review(
                        data, transform=source_transform
                    )
                    tess_issues.extend(_mapped_issues)
                    original_coverage = _measure_attempt_coverage(
                        attempt, tess_pixel_lines, attempt.ocr_img.size
                    )
                    original_confidence = _mean_ocr_confidence(tess_pixel_lines)
                    if (
                        task.source_type == "pdf"
                        and task.geometry.source_rotation == 180
                        and (original_confidence < 0.9 or original_coverage["needs_retry"])
                    ):
                        # A weak fallback on a half-turn page gets one bounded
                        # OCR-only orientation candidate. The visual source and
                        # final background never rotate; inverse quads map
                        # recognized words back onto the source pixels.
                        rotated_image = attempt.ocr_img.rotate(180)
                        try:
                            rotated_data = pytesseract.image_to_data(
                                rotated_image,
                                lang=tess_traineddata_code,
                                output_type=pytesseract.Output.DICT,
                                timeout=TESSERACT_TIMEOUT,
                            )
                        except (OSError, RuntimeError, TypeError, ValueError) as exc:
                            attempt.review_reasons.append(
                                f"half-turn fallback unavailable: {type(exc).__name__}"
                            )
                            rotated_data = {"text": []}
                        rotated_raw, rotated_issues = tesseract_data_to_lines_with_review(
                            rotated_data
                        )
                        rotated_pixel, pixel_issues = map_half_turn_words_to_source(
                            rotated_raw,
                            PageGeometry(*attempt.ocr_img.size, "checkpoint-local").for_raster(
                                *attempt.ocr_img.size
                            ),
                        )
                        rotated_pdf, pdf_issues = map_half_turn_words_to_source(
                            rotated_raw, source_transform
                        )
                        rotated_issues.extend(pixel_issues)
                        rotated_issues.extend(pdf_issues)
                        rotated_text = " ".join(word for word, _, _ in rotated_raw)
                        original_text = " ".join(word for word, _, _ in tess_pixel_lines)
                        rotated_confidence = _mean_ocr_confidence(rotated_raw)
                        rotated_coverage = _measure_attempt_coverage(
                            attempt, rotated_pixel, attempt.ocr_img.size
                        )
                        model_script = getattr(
                            LANGUAGE_REGISTRY.get(tess_canonical_language), "script", ""
                        )
                        supported = text_supports_script(rotated_text, model_script)
                        decision = compare_ocr_candidates(
                            OCRCandidate(
                                original_text, "tesseract", original_confidence,
                                original_coverage.get("coverage"), True,
                            ),
                            OCRCandidate(
                                rotated_text, "tesseract", rotated_confidence,
                                rotated_coverage.get("coverage"), supported,
                            ),
                        )
                        if (
                            not decision.use_candidate
                            and not original_text.strip()
                            and rotated_text.strip()
                            and supported
                            and rotated_confidence >= 0.85
                        ):
                            decision = type(decision)(True, "empty_original_half_turn_recovery", True)
                        if decision.use_candidate and rotated_pdf and not rotated_issues:
                            attempt.candidate_reviews.append({
                                "text": original_text,
                                "model_language": tess_canonical_language,
                                "confidence": original_confidence,
                                "independent_coverage": original_coverage,
                                "decision": "superseded_original_orientation",
                            })
                            attempt.review_reasons.append(
                                f"half-turn fallback: {decision.reason}"
                            )
                            tess_pixel_lines = rotated_pixel
                            tess_lines = rotated_pdf
                            tess_issues = []
                            chosen_coverage = rotated_coverage
                        else:
                            if rotated_text != original_text and rotated_text.strip():
                                attempt.candidate_reviews.append({
                                    "text": rotated_text,
                                    "model_language": tess_canonical_language,
                                    "confidence": rotated_confidence,
                                    "independent_coverage": rotated_coverage,
                                    "decision": decision.reason,
                                })
                            chosen_coverage = original_coverage
                    else:
                        chosen_coverage = original_coverage
                    if (
                        task.source_type == "pdf"
                        and task.geometry.source_rotation in (90, 270)
                        and (
                            _mean_ocr_confidence(tess_pixel_lines) < 0.9
                            or chosen_coverage.get("needs_retry")
                        )
                    ):
                        # A sideways source page may need either clockwise or
                        # counterclockwise OCR. Compare both bounded candidates;
                        # page rotation metadata alone never selects one.
                        for angle in (90, 270):
                            try:
                                rotated_image = attempt.ocr_img.rotate(angle, expand=True)
                                rotated_data = pytesseract.image_to_data(
                                    rotated_image,
                                    lang=tess_traineddata_code,
                                    output_type=pytesseract.Output.DICT,
                                    timeout=TESSERACT_TIMEOUT,
                                )
                                rotated_raw, issues = tesseract_data_to_lines_with_review(
                                    rotated_data
                                )
                                rotated_pixel, pixel_issues = map_rotated_words_to_source(
                                    rotated_raw,
                                    PageGeometry(*attempt.ocr_img.size, "checkpoint-local").for_raster(
                                        *attempt.ocr_img.size
                                    ),
                                    angle,
                                )
                                rotated_pdf, pdf_issues = map_rotated_words_to_source(
                                    rotated_raw, source_transform, angle
                                )
                                issues.extend(pixel_issues)
                                issues.extend(pdf_issues)
                                candidate_text = " ".join(text for text, _, _ in rotated_raw)
                                prior_text = " ".join(text for text, _, _ in tess_pixel_lines)
                                candidate_confidence = _mean_ocr_confidence(rotated_raw)
                                candidate_coverage = _measure_attempt_coverage(
                                    attempt, rotated_pixel, attempt.ocr_img.size
                                )
                                # Sideways glyphs may produce no supported
                                # text regions in the source-oriented visual
                                # analyzer. Measure the same source pixels in
                                # the OCR-only upright view before treating an
                                # otherwise high-confidence reading as complete.
                                upright_coverage = asdict(assess_candidate_coverage(
                                    assess_raster_text_regions(rotated_image),
                                    rotated_raw,
                                    candidate_raster_size=rotated_image.size,
                                ))
                                upright_coverage["measurement_orientation"] = angle
                                model_script = getattr(
                                    LANGUAGE_REGISTRY.get(tess_canonical_language),
                                    "script", "",
                                )
                                decision = compare_ocr_candidates(
                                    OCRCandidate(
                                        prior_text, "tesseract",
                                        _mean_ocr_confidence(tess_pixel_lines),
                                        chosen_coverage.get("coverage"), True,
                                    ),
                                    OCRCandidate(
                                        candidate_text, "tesseract", candidate_confidence,
                                        candidate_coverage.get("coverage"),
                                        text_supports_script(candidate_text, model_script),
                                    ),
                                )
                                if (
                                    not decision.use_candidate
                                    and chosen_coverage.get("status") == "unknown"
                                    and upright_coverage.get("status") == "measured"
                                    and (upright_coverage.get("coverage") or 0) >= 0.8
                                    and not upright_coverage.get("needs_retry")
                                    and candidate_confidence >= 0.85
                                    and candidate_confidence
                                    >= _mean_ocr_confidence(tess_pixel_lines) + 0.30
                                    and text_supports_script(candidate_text, model_script)
                                ):
                                    decision = type(decision)(
                                        True, "upright_visual_coverage_and_confidence_gain", True
                                    )
                                if decision.use_candidate and rotated_pdf and not issues:
                                    attempt.candidate_reviews.append({
                                        "text": prior_text,
                                        "model_language": tess_canonical_language,
                                        "confidence": _mean_ocr_confidence(tess_pixel_lines),
                                        "independent_coverage": chosen_coverage,
                                        "decision": "superseded_source_orientation",
                                    })
                                    attempt.review_reasons.append(
                                        f"quarter-turn fallback {angle}: {decision.reason}"
                                    )
                                    tess_pixel_lines = rotated_pixel
                                    tess_lines = rotated_pdf
                                    tess_issues = []
                                    chosen_coverage = (
                                        upright_coverage
                                        if decision.reason == "upright_visual_coverage_and_confidence_gain"
                                        else candidate_coverage
                                    )
                                elif candidate_text and candidate_text != prior_text:
                                    attempt.candidate_reviews.append({
                                        "text": candidate_text,
                                        "model_language": tess_canonical_language,
                                        "confidence": candidate_confidence,
                                        "independent_coverage": candidate_coverage,
                                        "upright_visual_coverage": upright_coverage,
                                        "decision": decision.reason,
                                    })
                            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                                attempt.review_reasons.append(
                                    f"quarter-turn fallback unavailable: {type(exc).__name__}"
                                )
                    # Quality recovery is part of the default legal path.
                    # Legacy ENABLE_DPI_ESCALATION governs general requeueing,
                    # but a weak geometry-bearing Tesseract result gets at most
                    # two source-bound OCR retries here even when that optional
                    # switch is off. The original visual raster is preserved.
                    selected_tess_dpi = _task_render_dpi(task)
                    selected_tess_confidence = _mean_ocr_confidence(tess_pixel_lines)
                    if task.source_type == "pdf" and (
                        (tess_pixel_lines and (
                            selected_tess_confidence < 0.85
                            or _tesseract_has_uncertain_word(tess_pixel_lines)
                        ))
                        or (
                            chosen_coverage.get("status") == "measured"
                            and chosen_coverage.get("needs_retry")
                        )
                    ):
                        attempt.tesseract_dpi_recovery_used = True
                        for retry_dpi in DPI_SCHEDULE:
                            if retry_dpi <= selected_tess_dpi:
                                continue
                            if retry_dpi > 600:
                                break
                            try:
                                _check_expected_raster_bytes(task.geometry, retry_dpi)
                                retry_image = _re_extract_task_page_at_dpi(task, retry_dpi)
                                if retry_image is None:
                                    break
                                retry_data = pytesseract.image_to_data(
                                    retry_image,
                                    lang=tess_traineddata_code,
                                    output_type=pytesseract.Output.DICT,
                                    timeout=TESSERACT_TIMEOUT,
                                )
                                retry_pixel, retry_issues = tesseract_data_to_lines_with_review(
                                    retry_data
                                )
                                retry_pdf, retry_map_issues = tesseract_data_to_lines_with_review(
                                    retry_data,
                                    transform=task.geometry.for_raster(*retry_image.size),
                                )
                                retry_issues.extend(retry_map_issues)
                                retry_text = " ".join(word for word, _, _ in retry_pixel)
                                retry_confidence = _mean_ocr_confidence(retry_pixel)
                                retry_coverage = _measure_attempt_coverage(
                                    attempt, retry_pixel, retry_image.size
                                )
                                model_script = getattr(
                                    LANGUAGE_REGISTRY.get(tess_canonical_language),
                                    "script", "",
                                )
                                script_supported = text_supports_script(
                                    retry_text, model_script
                                )
                                prior_text = " ".join(word for word, _, _ in tess_pixel_lines)
                                prior_coverage = chosen_coverage.get("coverage")
                                new_coverage = retry_coverage.get("coverage")
                                nonregressed_coverage = (
                                    prior_coverage is None
                                    or new_coverage is None
                                    or new_coverage + 1e-9 >= prior_coverage
                                )
                                retry_language, retry_support = detect_language_from_text(
                                    retry_text
                                )
                                grounded_decision = compare_grounded_word_candidates(
                                    OCRCandidate(
                                        prior_text, "tesseract", selected_tess_confidence,
                                        prior_coverage, True,
                                    ),
                                    OCRCandidate(
                                        retry_text, "tesseract", retry_confidence,
                                        new_coverage, script_supported,
                                        retry_support
                                        if retry_language == tess_canonical_language else None,
                                    ),
                                    prior_lines=tess_lines,
                                    candidate_lines=retry_pdf,
                                )
                                gained = (
                                    script_supported
                                    and not retry_issues
                                    and retry_pdf
                                    and nonregressed_coverage
                                    and grounded_decision.use_candidate
                                )
                                if gained:
                                    attempt.candidate_reviews.append({
                                        "text": prior_text,
                                        "model_language": tess_canonical_language,
                                        "confidence": selected_tess_confidence,
                                        "independent_coverage": chosen_coverage,
                                        "decision": "superseded_lower_dpi_ocr",
                                    })
                                    attempt.review_reasons.append(
                                        f"bounded Tesseract DPI recovery: {selected_tess_dpi} to {retry_dpi}; "
                                        f"{grounded_decision.reason}"
                                    )
                                    tess_pixel_lines = retry_pixel
                                    tess_lines = retry_pdf
                                    chosen_coverage = retry_coverage
                                    selected_tess_confidence = retry_confidence
                                    selected_tess_dpi = retry_dpi
                                elif retry_text and retry_text != prior_text:
                                    attempt.candidate_reviews.append({
                                        "text": retry_text,
                                        "model_language": tess_canonical_language,
                                        "confidence": retry_confidence,
                                        "independent_coverage": retry_coverage,
                                        "decision": "dpi_candidate_not_corroborated",
                                    })
                            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                                attempt.review_reasons.append(
                                    f"bounded DPI recovery unavailable: {type(exc).__name__}"
                                )
                                break
                            finally:
                                if "retry_image" in locals() and retry_image is not None:
                                    retry_image.close()
                                    retry_image = None
                            if (
                                selected_tess_confidence >= 0.85
                                and not _tesseract_has_uncertain_word(tess_pixel_lines)
                                and not chosen_coverage.get("needs_retry")
                            ):
                                break
                    # A fresh OCR reading may provide independent evidence of
                    # a different Latin language even when initial OSD only
                    # established Latin script. Probe one installed traineddata
                    # alternative; scores from different traineddata are not
                    # compared as calibrated confidence values.
                    prior_tess_text = " ".join(word for word, _, _ in tess_pixel_lines)
                    text_language, text_support = detect_language_from_text(prior_tess_text)
                    source_entry = LANGUAGE_REGISTRY.get(tess_canonical_language)
                    target_entry = LANGUAGE_REGISTRY.get(text_language)
                    if (
                        text_language and text_language != tess_canonical_language
                        and text_support >= 0.55
                        and source_entry is not None and source_entry.script == "latin"
                        and target_entry is not None and target_entry.script == "latin"
                        and selected_tess_dpi == _task_render_dpi(task)
                    ):
                        alternate_doc = target_entry.tesseract_code
                        if alternate_doc and alternate_doc != tess_traineddata_code:
                            try:
                                alternate_data = pytesseract.image_to_data(
                                    attempt.ocr_img, lang=alternate_doc,
                                    output_type=pytesseract.Output.DICT,
                                    timeout=TESSERACT_TIMEOUT,
                                )
                                alternate_pixel, alternate_issues = (
                                    tesseract_data_to_lines_with_review(alternate_data)
                                )
                                alternate_pdf, mapped_issues = (
                                    tesseract_data_to_lines_with_review(
                                        alternate_data, transform=source_transform
                                    )
                                )
                                alternate_issues.extend(mapped_issues)
                                alternate_text = " ".join(
                                    word for word, _, _ in alternate_pixel
                                )
                                alternate_label, alternate_support = (
                                    detect_language_from_text(alternate_text)
                                )
                                alternate_coverage = _measure_attempt_coverage(
                                    attempt, alternate_pixel, attempt.ocr_img.size
                                )
                                decision = compare_cross_engine_candidates(
                                    OCRCandidate(
                                        prior_tess_text,
                                        f"tesseract:{tess_traineddata_code}",
                                        selected_tess_confidence,
                                        chosen_coverage.get("coverage"), True,
                                        text_support,
                                    ),
                                    OCRCandidate(
                                        alternate_text,
                                        f"tesseract:{alternate_doc}",
                                        _mean_ocr_confidence(alternate_pixel),
                                        alternate_coverage.get("coverage"),
                                        not alternate_issues and text_supports_script(
                                            alternate_text, "latin"
                                        ),
                                        alternate_support
                                        if alternate_label == text_language else 0.0,
                                    ),
                                )
                                attempt.candidate_reviews.append({
                                    "text": alternate_text,
                                    "model_language": text_language,
                                    "runtime_model": alternate_doc,
                                    "confidence": _mean_ocr_confidence(alternate_pixel),
                                    "confidence_basis": "tesseract_word_confidence_percent_scaled",
                                    "independent_coverage": alternate_coverage,
                                    "decision": decision.reason,
                                })
                                if decision.use_candidate and alternate_pdf and not alternate_issues:
                                    attempt.candidate_reviews.append({
                                        "text": prior_tess_text,
                                        "model_language": tess_canonical_language,
                                        "runtime_model": tess_traineddata_code,
                                        "confidence": selected_tess_confidence,
                                        "independent_coverage": chosen_coverage,
                                        "decision": "superseded_prior_traineddata_preserved_for_review",
                                    })
                                    attempt.review_reasons.append(
                                        "independently supported Latin traineddata rerun: "
                                        + decision.reason
                                    )
                                    tess_pixel_lines = alternate_pixel
                                    tess_lines = alternate_pdf
                                    chosen_coverage = alternate_coverage
                                    tess_canonical_language = text_language
                                    tess_traineddata_code = alternate_doc
                                    selected_tess_confidence = _mean_ocr_confidence(
                                        alternate_pixel
                                    )
                            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                                attempt.review_reasons.append(
                                    "Latin traineddata probe unavailable: "
                                    + type(exc).__name__
                                )
                    if selected_tess_dpi != _task_render_dpi(task):
                        task.render_dpi = selected_tess_dpi
                    temp_doc = fitz.open()
                    page = temp_doc.new_page(
                        width=task.geometry.width_pt, height=task.geometry.height_pt
                    )
                    page.insert_image(
                        page.rect,
                        stream=img_to_bytes(task.image),
                    )

                    found_tess = False
                    tess_confs = []
                    _tess_parts: list[str] = []
                    attempt.coverage_assessment = chosen_coverage
                    attempt.coverage_needs_retry = bool(
                        attempt.coverage_assessment["needs_retry"]
                    )
                    attempt.geometry_issues.extend(tess_issues)
                    if tess_issues:
                        logger.warning(
                            "Page %s p%d has %d unplaceable Tesseract spans; review required",
                            task.doc_id, task.page_num, len(tess_issues),
                        )
                    for span_index, (word, box, confidence) in enumerate(tess_lines):
                        if insert_text_line(
                            page, word, box, lang_code=tess_canonical_language
                        ):
                            _tess_parts.append(word)
                            found_tess = True
                            tess_confs.append(confidence)
                        else:
                            attempt.geometry_issues.append({
                                "index": span_index, "text": word,
                                "reason": "Tesseract hidden text could not fit word box",
                            })

                    # maintenance review: materialize text_content from accumulated parts
                    attempt.text_content = " ".join(_tess_parts)
                    if attempt.text_content:
                        attempt.text_content += " "
                    if found_tess:
                        attempt.pdf_bytes = temp_doc.write(deflate=True)
                        attempt.status = PageStatus.TESSERACT
                        attempt.ocr_method = PageStatus.TESSERACT
                        # Record the canonical language label; the Tesseract
                        # traineddata code is passed separately to image_to_data.
                        attempt.model_language = tess_canonical_language
                        _sync_attempt_ocr_language_alias(attempt)
                        _confirm_attempt_text_language(attempt)
                        attempt.page_confidence = (
                            (sum(tess_confs) / len(tess_confs))
                            if tess_confs
                            else 0.0
                        )
                        attempt.actual_execution_device = "cpu"
                        # Legacy Tesseract path runs on raw/preprocessed ocr_img,
                        # not the GPU-batched img_np.
                        attempt.gpu_optimization_applied = False
                    else:
                        raise Exception("No text")
                except Exception as e:
                    logger.warning(
                        "Tesseract Failed for %s p%d: %s", task.doc_id, task.page_num, e
                    )
                    # [Fallback - Image Only]
                    # If Paddle/Tesseract fail, we MUST preserve the image in the PDF
                    if not attempt.pdf_bytes:
                        try:
                            with doc_registry_lock:
                                _fallback_doc_state = doc_registry.get(task.doc_id)
                            _record_rejected_ocr_candidate(
                                task, attempt, e, _fallback_doc_state
                            )
                            temp_doc = fitz.open()
                            page = temp_doc.new_page(
                                width=task.geometry.width_pt,
                                height=task.geometry.height_pt,
                            )
                            page.insert_image(
                                page.rect,
                                stream=img_to_bytes(task.image),
                            )
                            attempt.pdf_bytes = temp_doc.write(deflate=True)
                            attempt.status = PageStatus.IMAGE_ONLY
                            attempt.ocr_method = PageStatus.IMAGE_ONLY
                            _clear_attempt_language_fields(attempt)
                            attempt.page_confidence = 0.0
                            attempt.actual_execution_device = "none"
                            attempt.gpu_optimization_applied = False
                            attempt.preprocessing_applied = False
                            logger.warning(
                                "OCR Failed for %s p%d. Saved as Image Only.",
                                task.doc_id,
                                task.page_num,
                            )
                            log_failure(
                                task.doc_path,
                                task.page_num,
                                "OCR Text Failure (Image Preserved)",
                            )
                        except Exception as inner_e:
                            # If even Image Only fails (e.g. corrupt image stream), raise to outer block
                            raise inner_e
                finally:
                    try:
                        if temp_doc:
                            temp_doc.close()
                    except Exception as exc:
                        logger.debug(
                            "Failed to close temp_doc in finally block: %s", exc
                        )

            if not attempt.pdf_bytes:
                raise RuntimeError("No PDF bytes generated")

            if not attempt.coverage_assessment:
                attempt.coverage_assessment = _measure_attempt_coverage(
                    attempt, [], task.image.size
                )
                attempt.coverage_needs_retry = bool(
                    attempt.coverage_assessment["needs_retry"]
                )
            if attempt.coverage_assessment["needs_review"]:
                attempt.review_reasons.append(
                    "independent raster coverage: "
                    + str(attempt.coverage_assessment["reason"])
                )

            # --- Document Intelligence (Phase 3A) ---
            attempt.structure_data = None
            if structure_engine is not None:
                try:
                    # Reuse OCR numpy array when preprocessing is disabled (same source image).
                    # When preprocessing IS enabled, ocr_img differs from task.image so we must
                    # convert the original image separately for PP-Structure.
                    if attempt.img_np is not None and attempt.ocr_img is task.image:
                        di_img_np = attempt.img_np
                    else:
                        di_img_np = np.array(task.image)
                    structure_result = structure_engine(di_img_np)
                    attempt.structure_data = parse_structure_result(structure_result)
                except Exception as e:
                    logger.warning(
                        "Worker-%d: DocIntel failed for %s p%d: %s",
                        worker_id,
                        task.doc_id,
                        task.page_num,
                        e,
                    )

            # Resolve document state before custody and retry decisions.
            with doc_registry_lock:
                doc_state = doc_registry.get(task.doc_id)
            if not doc_state:
                raise RuntimeError(f"Document state missing for {task.doc_id}")

            # Custody: engine provenance + docintel early; page OCR events
            # (text_length) must wait until canonical checkpoint text is final.
            if doc_state and doc_state.custody_chain:
                for _engine_event in _engine_custody_events:
                    _engine_event_data = _engine_event.copy()
                    _engine_event_type = _engine_event_data.pop(
                        "event", "ENGINE_PROVENANCE"
                    )
                    doc_state.custody_chain.append_event(
                        _engine_event_type,
                        _engine_event_data,
                    )
                if attempt.structure_data is not None:
                    doc_state.custody_chain.append_event(
                        "docintel_analysis",
                        {
                            "page_num": task.page_num,
                            "tables_found": len(
                                attempt.structure_data.get("tables", [])
                            ),
                            "layout_regions": len(
                                attempt.structure_data.get("layout_regions", [])
                            ),
                        },
                    )

            # --- Adaptive DPI Escalation (Phase 4D+) ---
            if (
                ENABLE_DPI_ESCALATION
                and _DPI_ESCALATION_AVAILABLE
                and not attempt.tesseract_dpi_recovery_used
                and not attempt.paddle_coverage_recovery_used
                and task.source_type == "pdf"
                and (
                    should_escalate(
                        attempt.page_confidence,
                        task.retries,
                        DPI_CONFIDENCE_THRESHOLD,
                    )
                    or attempt.coverage_needs_retry
                )
            ):
                next_idx = task.retries + 1
                next_dpi = (
                    DPI_SCHEDULE[next_idx] if next_idx < len(DPI_SCHEDULE) else None
                )
                if next_dpi is not None:
                    logger.info(
                        "Worker-%d: DPI escalation for %s p%d (conf=%.3f < %s, retry %d, DPI %s)",
                        worker_id,
                        task.doc_id,
                        task.page_num,
                        attempt.page_confidence,
                        DPI_CONFIDENCE_THRESHOLD,
                        task.retries + 1,
                        next_dpi,
                    )
                    retry_img = _re_extract_task_page_at_dpi(task, next_dpi)
                    if retry_img is not None:
                        prior_dpi = _task_render_dpi(task)
                        task.retries += 1
                        task.image = retry_img
                        # Only update after successful re-extract.
                        task.render_dpi = int(next_dpi)
                        if doc_state and doc_state.custody_chain:
                            doc_state.custody_chain.append_event(
                                "dpi_escalation",
                                {
                                    "page_num": task.page_num,
                                    "original_dpi": prior_dpi,
                                    "retry_dpi": next_dpi,
                                    "original_confidence": attempt.page_confidence,
                                    "retry_number": task.retries,
                                },
                            )
                        try:
                            retry_reservation = _reserve_queue_obligation(
                                image_queue, task
                            )
                            try:
                                retry_bytes = _raster_task_bytes(task)
                                if not _RASTER_QUEUE_BUDGET.acquire(
                                    retry_bytes, wait=False
                                ):
                                    raise queue.Full
                                task._queued_raster_bytes = retry_bytes
                                image_queue.put_nowait(task)
                            except BaseException:
                                _release_queued_raster(task)
                                _cancel_queue_obligation(task, retry_reservation)
                                raise
                        except queue.Full:
                            # All workers may simultaneously need a retry while
                            # the bounded queue is full.  Never block a consumer
                            # trying to become its own producer: retain this
                            # task in the current worker and carry the original
                            # unfinished-task obligation into the inline pass.
                            _inline_retry_task = task
                            _dpi_retry_inline = True
                            attempt.dpi_retry_inline = True
                        _dpi_requeued = True
                        attempt.dpi_requeued = True
                        continue

            # --- Handwriting Detection (Phase 6B) ---
            attempt.handwriting_data = None
            if ENABLE_HANDWRITING and _HANDWRITING_AVAILABLE and attempt.paddle_lines:
                try:
                    # Convert polygon boxes to [x1,y1,x2,y2] bboxes for handwriting module
                    hw_lines = []
                    for txt, box, conf in attempt.paddle_lines:
                        if box is not None and len(box) >= 4:
                            xs = [pt[0] for pt in box]
                            ys = [pt[1] for pt in box]
                            bbox = [min(xs), min(ys), max(xs), max(ys)]
                        else:
                            bbox = [0, 0, 0, 0]
                        hw_lines.append((txt, conf, bbox))
                    img_size = (
                        task.image.size if hasattr(task.image, "size") else (0, 0)
                    )
                    hw_conf = detect_handwriting_by_confidence(
                        hw_lines,
                        task.page_num,
                        img_size,
                    )
                    hw_geom = detect_handwriting_by_geometry(hw_lines, task.page_num)
                    attempt.handwriting_data = merge_handwriting_signals(
                        hw_conf,
                        hw_geom,
                        None,
                        task.page_num,
                    )
                except Exception as hw_err:
                    logger.warning(
                        "Worker-%d: Handwriting detection failed for %s p%d: %s",
                        worker_id,
                        task.doc_id,
                        task.page_num,
                        hw_err,
                    )

            # --- Signature Verification Research Path (Phase 6D) ---
            attempt.signature_data = None
            if (
                ENABLE_SIGNATURE_VERIFICATION
                and _SIGNATURE_VERIFICATION_AVAILABLE
                and (attempt.structure_data is not None or attempt.paddle_lines)
            ):
                try:
                    attempt.signature_data = analyze_signature_page(
                        task.image,
                        task.page_num,
                        attempt.structure_data,
                        attempt.paddle_lines,
                    )
                except Exception as sig_err:
                    logger.warning(
                        "Worker-%d: Signature verification failed for %s p%d: %s",
                        worker_id,
                        task.doc_id,
                        task.page_num,
                        sig_err,
                    )

            # --- CJK Vertical Text Analysis (Phase 6D) ---
            attempt.vertical_text_data = None
            if (
                ENABLE_VERTICAL_TEXT
                and _VERTICAL_TEXT_AVAILABLE
                and attempt.paddle_lines
            ):
                try:
                    page_w = task.image.size[0] if hasattr(task.image, "size") else 0
                    attempt.vertical_text_data = analyze_page_vertical_text(
                        attempt.paddle_lines,
                        task.page_num,
                        page_w,
                        attempt.ocr_language,
                    )
                except Exception as vt_err:
                    logger.warning(
                        "Worker-%d: Vertical text analysis failed for %s p%d: %s",
                        worker_id,
                        task.doc_id,
                        task.page_num,
                        vt_err,
                    )

            # --- Table Fallback Analysis (Phase 4D) ---
            attempt.table_fallback_data = None
            if (
                ENABLE_TABLE_FALLBACK
                and _TABLE_FALLBACK_AVAILABLE
                and attempt.structure_data
                and attempt.paddle_lines
            ):
                try:
                    table_regions = []
                    for region in attempt.structure_data.get("layout_regions", []):
                        if region.get("type") == "table":
                            rb = region.get("bbox", [0, 0, 0, 0])
                            if len(rb) == 4:
                                x1, y1, x2, y2 = rb
                                table_regions.append(
                                    TableRegion(
                                        bbox=(x1, y1, x2 - x1, y2 - y1),
                                        page_number=task.page_num,
                                        original_confidence=region.get(
                                            "confidence", 0.0
                                        ),
                                        original_engine=attempt.ocr_method or "paddle",
                                    )
                                )
                    if table_regions:
                        attempt.table_fallback_data = analyze_page_tables(
                            task.page_num,
                            table_regions,
                            attempt.paddle_lines,
                        )
                except Exception as tf_err:
                    logger.warning(
                        "Worker-%d: Table fallback analysis failed for %s p%d: %s",
                        worker_id,
                        task.doc_id,
                        task.page_num,
                        tf_err,
                    )

            # --- Per-Span Language Detection (Plan A -- PR A3) ---
            # Build per-line SpanLanguage objects and aggregate them into a
            # PageLanguage with character-count-weighted primary detection.
            # When the operator enabled this output, text-bearing pages fail
            # closed if the detector is unavailable or errors; persisting a
            # checkpoint that falsely claims analysis completion would make
            # the omission permanent across resume/cache reuse.
            attempt.language_data = None
            if ENABLE_PER_SPAN_LANGUAGE and (attempt.text_content or "").strip():
                if not _LANGUAGE_DETECTION_AVAILABLE:
                    raise RuntimeError(
                        "per-span language analysis enabled but module unavailable"
                    )
                if lang_model is None:
                    raise RuntimeError(
                        "per-span language analysis enabled but model unavailable"
                    )
                try:
                    language_lines = [
                        line for line in attempt.paddle_lines if (line[0] or "").strip()
                    ]
                    if not language_lines:
                        language_lines = [
                            (line, [0.0, 0.0, 0.0, 0.0], attempt.page_confidence)
                            for line in attempt.text_content.splitlines()
                            if line.strip()
                        ] or [
                            (
                                attempt.text_content,
                                [0.0, 0.0, 0.0, 0.0],
                                attempt.page_confidence,
                            )
                        ]
                    spans = []
                    for _txt, _box, _conf in language_lines:
                        bbox_list = (
                            list(_box) if _box is not None else [0.0, 0.0, 0.0, 0.0]
                        )
                        spans.append(
                            detect_span_language(
                                text=_txt or "",
                                bbox=bbox_list,
                                fasttext_model=lang_model,
                                short_span_threshold=LANGUAGE_SHORT_SPAN_THRESHOLD,
                                confidence_threshold=LANGUAGE_CONFIDENCE_THRESHOLD,
                            )
                        )
                    attempt.language_data = aggregate_page_from_spans(
                        page_num=task.page_num,
                        spans=spans,
                    )
                    if attempt.language_data is None:
                        raise RuntimeError(
                            "language detector produced no page analysis"
                        )
                except Exception as lang_err:
                    raise RuntimeError(
                        "span language detection failed for "
                        f"{task.doc_id} p{task.page_num}: {lang_err}"
                    ) from lang_err

            # --- Language custody events + Prometheus metrics (Plan A -- PR A4) ---
            # Emit LANGUAGE_DETECTED (and LANGUAGE_MIXED_SCRIPT when applicable)
            # custody events and bump the three language Prometheus counters.
            # All calls are wrapped in try/except: a custody/metrics failure
            # must never block OCR output or page assembly.
            if attempt.language_data is not None:
                _language_provenance = _language_model_provenance()
                try:
                    with doc_registry_lock:
                        _lang_doc_state = doc_registry.get(task.doc_id)
                    if _lang_doc_state is not None and _lang_doc_state.custody_chain:
                        _lang_doc_state.custody_chain.append_event(
                            "LANGUAGE_DETECTED",
                            {
                                "page_num": task.page_num,
                                "primary_language": attempt.language_data.primary_language,
                                "primary_confidence": round(
                                    attempt.language_data.primary_confidence,
                                    4,
                                ),
                                "languages_detected": list(
                                    attempt.language_data.languages_detected,
                                ),
                                "language_char_shares": {
                                    k: round(v, 4)
                                    for k, v in attempt.language_data.language_char_shares.items()
                                },
                                "span_count": attempt.language_data.span_count,
                                "spans_labeled": attempt.language_data.spans_labeled,
                                "detection_engine": "fasttext+script_heuristic",
                                **_language_provenance,
                            },
                        )
                        if attempt.language_data.mixed_script:
                            _script_char_shares: dict[str, int] = {}
                            for _span in attempt.language_data.spans:
                                _script_char_shares[_span.script] = (
                                    _script_char_shares.get(_span.script, 0)
                                    + _span.char_count
                                )
                            _total_chars = sum(_script_char_shares.values()) or 1
                            _lang_doc_state.custody_chain.append_event(
                                "LANGUAGE_MIXED_SCRIPT",
                                {
                                    "page_num": task.page_num,
                                    "scripts_detected": list(
                                        attempt.language_data.scripts_detected,
                                    ),
                                    "script_char_shares": {
                                        k: round(v / _total_chars, 4)
                                        for k, v in _script_char_shares.items()
                                    },
                                    "primary_language": attempt.language_data.primary_language,
                                    "rtl_present": any(
                                        s.script == "arabic"
                                        for s in attempt.language_data.spans
                                    ),
                                },
                            )
                except Exception:
                    # Custody failure must never block OCR output.
                    pass

                try:
                    from api.prometheus import (
                        ocr_language_confidence,
                        ocr_language_detected_total,
                        ocr_language_mixed_script_pages_total,
                    )

                    try:
                        _entry = LANGUAGE_REGISTRY.get(
                            attempt.language_data.primary_language
                        )
                        _tier = _entry.tier if _entry is not None else "core"
                    except Exception:
                        _tier = "core"
                    ocr_language_detected_total.labels(
                        lang=attempt.language_data.primary_language,
                        tier=_tier,
                        level="page",
                    ).inc()
                    ocr_language_confidence.labels(
                        lang=attempt.language_data.primary_language,
                        level="page",
                    ).observe(attempt.language_data.primary_confidence)
                    if (
                        attempt.language_data.mixed_script
                        and attempt.language_data.scripts_detected
                    ):
                        ocr_language_mixed_script_pages_total.labels(
                            primary_script=attempt.language_data.scripts_detected[0],
                        ).inc()
                except Exception:
                    # Prometheus failure must never block OCR output.
                    pass

            # The attempt is now terminal: the DPI decision and all supported
            # page-level analysis are complete. Derive text authority from the
            # checkpoint PDF text layer (same source resume/cache use) before
            # embedding provenance and notifying assembly.
            attempt.chunk_path = os.path.join(
                doc_state.temp_dir, f"{task.page_num}.pdf"
            )
            attempt.text_content = _canonical_text_after_checkpoint_pdf(
                attempt.pdf_bytes,
                ocr_text=attempt.text_content,
            )
            method_value = (
                attempt.ocr_method.value
                if isinstance(attempt.ocr_method, enum.Enum)
                else str(attempt.ocr_method)
            )
            if method_value not in _VALID_CHECKPOINT_EXECUTION_ENGINES:
                # Map status-like labels onto the closed engine set.
                if method_value in {
                    PageStatus.PADDLE.value,
                    PageStatus.OK.value,
                    PageStatus.CACHED.value,
                } or str(method_value).startswith("Paddle"):
                    method_value = "PaddleOCR"
                elif method_value == PageStatus.TESSERACT.value:
                    method_value = "Tesseract"
                elif method_value == PageStatus.EASYOCR.value:
                    method_value = "EasyOCR"
                elif method_value == PageStatus.IMAGE_ONLY.value:
                    method_value = "ImageOnly"
                else:
                    raise RuntimeError(
                        f"unrecognized execution engine for checkpoint: {method_value!r}"
                    )
            routing_target_label = (
                _routing_target_value(attempt.routing_target)
                if attempt.routing_target is not None
                else "default"
            ) or "default"
            page_profile = _checkpoint_processing_profile(
                actual_render_dpi=_task_render_dpi(task),
                page_routing_target=routing_target_label,
                execution_device=str(attempt.actual_execution_device),
                execution_engine=method_value,
                gpu_optimization_applied=bool(attempt.gpu_optimization_applied),
                preprocessing_applied=bool(attempt.preprocessing_applied),
            )
            if method_value == "Tesseract":
                page_profile["runtime_model"] = _resolve_tesseract_route_language(
                    attempt.model_language
                )[1]
                page_profile["ocr_confidence_basis"] = "tesseract_word_confidence_percent_scaled"
            elif method_value == "PaddleOCR":
                page_profile["runtime_model"] = get_paddle_runtime_code(
                    attempt.model_language
                )
                page_profile["ocr_confidence_basis"] = "paddle_recognition_score"
            else:
                page_profile["runtime_model"] = ""
                page_profile["ocr_confidence_basis"] = "none"
            with _get_checkpoint_artifact_lock(task.doc_id):
                attempt.pdf_bytes = _embed_page_checkpoint_metadata(
                    attempt.pdf_bytes,
                    document_id=task.doc_id,
                    page_num=task.page_num,
                    status=attempt.status,
                    ocr_method=attempt.ocr_method,
                    ocr_language=attempt.ocr_language,
                    detected_language=attempt.detected_language,
                    detected_language_confidence=attempt.detected_language_confidence,
                    source_prior_label=(
                        task.lang_hint if task.source_prior_unverified else ""
                    ),
                    model_language=attempt.model_language,
                    ocr_confidence=attempt.page_confidence,
                    text_length=len(attempt.text_content),
                    has_text=bool(attempt.text_content),
                    language_data=attempt.language_data,
                    processing_profile=page_profile,
                    canonical_text=attempt.text_content,
                    page_geometry=task.geometry,
                    geometry_issues=attempt.geometry_issues,
                    review_reasons=attempt.review_reasons,
                    candidate_reviews=attempt.candidate_reviews,
                    coverage_assessment=attempt.coverage_assessment,
                )
                _atomic_write_bytes(attempt.chunk_path, attempt.pdf_bytes)
                # Intended embedded digest is authority; durable snapshot must
                # match exactly before validate/queue (same as cache path).
                durable_bytes, attempt.checkpoint_sha256 = (
                    _durable_checkpoint_matching_intended_bytes(
                        attempt.chunk_path,
                        attempt.pdf_bytes,
                    )
                )
                with fitz.open(stream=durable_bytes, filetype="pdf") as durable_doc:
                    if durable_doc.page_count != 1 or durable_doc.needs_pass:
                        raise RuntimeError(
                            "post-write checkpoint is not a readable single-page PDF"
                        )
                    durable_doc.load_page(0)
                    durable_text = _extract_text_from_pdf_document(durable_doc)
                    if durable_text != attempt.text_content:
                        raise RuntimeError(
                            "post-write checkpoint text diverged from pre-embed "
                            "canonical text"
                        )
                    durable_meta = _load_page_checkpoint_metadata_from_document(
                        durable_doc,
                        document_id=task.doc_id,
                        page_num=task.page_num,
                    )
                    if durable_meta is None:
                        raise RuntimeError("post-write checkpoint missing provenance")
                    _validated_checkpoint_text_value(durable_text, durable_meta)
                # Prefer durable on-disk bytes as the queued cache payload.
                attempt.pdf_bytes = durable_bytes
                attempt.text_content = durable_text
                # Arm keep-searchable only after digest + fitz/text/metadata
                # checks succeed. A raise above must leave the flag False.
                attempt.searchable_checkpoint_written = True

                # Custody page-OCR events use final checkpoint-bound text_length.
                if doc_state and doc_state.custody_chain:
                    if attempt.status == PageStatus.IMAGE_ONLY:
                        doc_state.custody_chain.append_event(
                            "ocr_image_only",
                            {
                                "page_num": task.page_num,
                            },
                        )
                    elif attempt.status == PageStatus.TESSERACT:
                        doc_state.custody_chain.append_event(
                            "ocr_fallback",
                            {
                                "page_num": task.page_num,
                                "engine": "Tesseract",
                                "lang_hint": attempt.ocr_language,
                                "text_length": len(attempt.text_content),
                            },
                        )
                    else:
                        doc_state.custody_chain.append_event(
                            "ocr_primary",
                            {
                                "page_num": task.page_num,
                                "engine": attempt.status,
                                "lang_hint": attempt.ocr_language,
                                "text_length": len(attempt.text_content),
                            },
                        )

                # Cache only accepted terminal attempts. A low-DPI attempt queued
                # for escalation must never short-circuit its own retry.
                if _page_cache is not None:
                    try:
                        _cache_key = f"{task.doc_id}:{task.page_num}:{DPI}"
                        _page_cache.put(
                            _cache_key,
                            attempt.pdf_bytes,
                            metadata={
                                "confidence": attempt.page_confidence,
                                "status": attempt.status,
                                "ocr_language": attempt.ocr_language,
                                "detected_language": attempt.detected_language,
                                "model_language": attempt.model_language,
                            },
                        )
                    except Exception as exc:
                        logger.debug("Page cache store failed (non-fatal): %s", exc)

            # Notify Assembly
            _interruptible_queue_put(
                assembly_queue,
                {
                    "doc_id": task.doc_id,
                    "page_num": task.page_num,
                    "text": attempt.text_content,  # Canonical checkpoint text (empty if ImageOnly)
                    "status": attempt.status,
                    "chunk_path": attempt.chunk_path,
                    "checkpoint_sha256": attempt.checkpoint_sha256,
                    "structure_data": attempt.structure_data,
                    "ocr_confidence": attempt.page_confidence,
                    "ocr_method": attempt.ocr_method,
                    "ocr_language": attempt.ocr_language,
                    "detected_language": attempt.detected_language,
                    "handwriting_data": attempt.handwriting_data,
                    "signature_data": attempt.signature_data,
                    "vertical_text_data": attempt.vertical_text_data,
                    "table_fallback_data": attempt.table_fallback_data,
                    "language_data": attempt.language_data,
                    "resume_metadata": None,
                    "cache_hit": False,
                },
            )

        except Exception as e:
            # CRITICAL FAILURE (Image Processing Failed)
            logger.error(
                "CRITICAL Worker failed on %s p%d: %s", task.doc_id, task.page_num, e
            )
            log_failure(task.doc_path, task.page_num, f"CRITICAL: Image Failure - {e}")
            with doc_registry_lock:
                doc_state = doc_registry.get(task.doc_id)
            if doc_state and doc_state.custody_chain:
                doc_state.custody_chain.append_event(
                    "processing_failed",
                    {
                        "stage": "worker",
                        "page_num": task.page_num,
                        "error": str(e),
                    },
                )

            # Keep a validated searchable checkpoint for THIS attempt. Foreign
            # leftover chunk_path values fail closed to image-only.
            _keep_temp_dir = getattr(doc_state, "temp_dir", None) if doc_state else None
            _keep_searchable = _keep_searchable_checkpoint(
                attempt,
                temp_dir=_keep_temp_dir,
            )
            if _keep_searchable:
                logger.error(
                    "Worker-%d: post-checkpoint failure for %s p%d; "
                    "keeping searchable PDF (%s)",
                    worker_id,
                    task.doc_id,
                    task.page_num,
                    e,
                )
                _interruptible_queue_put(
                    assembly_queue,
                    {
                        "doc_id": task.doc_id,
                        "page_num": task.page_num,
                        "text": attempt.text_content,
                        "status": attempt.status,
                        "chunk_path": attempt.chunk_path,
                        "checkpoint_sha256": attempt.checkpoint_sha256,
                        "structure_data": attempt.structure_data,
                        "ocr_confidence": attempt.page_confidence,
                        "ocr_method": attempt.ocr_method,
                        "ocr_language": attempt.ocr_language,
                        "detected_language": attempt.detected_language,
                        "handwriting_data": attempt.handwriting_data,
                        "signature_data": attempt.signature_data,
                        "vertical_text_data": attempt.vertical_text_data,
                        "table_fallback_data": attempt.table_fallback_data,
                        "language_data": attempt.language_data,
                        "resume_metadata": None,
                        "cache_hit": False,
                    },
                )
            _fallback_chunk = None
            _fallback_digest = None
            if (
                not _keep_searchable
                and hasattr(task, "image")
                and task.image is not None
            ):
                try:
                    _record_rejected_ocr_candidate(task, attempt, e, doc_state)
                    _fallback_chunk, _fallback_digest = _write_image_only_checkpoint(
                        task,
                        PageStatus.CRITICAL_FAILED,
                        page_routing_target=(
                            _routing_target_value(attempt.routing_target)
                            if attempt.routing_target is not None
                            else "default"
                        ),
                        geometry_issues=attempt.geometry_issues,
                        review_reasons=attempt.review_reasons,
                        candidate_reviews=attempt.candidate_reviews,
                    )
                    logger.warning(
                        "Worker-%d: Image-only fallback saved for %s p%d (evidence preserved)",
                        worker_id,
                        task.doc_id,
                        task.page_num,
                    )
                except Exception as fb_err:
                    logger.error(
                        "Worker-%d: Image-only fallback also failed for %s p%d: %s",
                        worker_id,
                        task.doc_id,
                        task.page_num,
                        fb_err,
                    )
                    _fallback_chunk = None
                    _fallback_digest = None
            if doc_state is None:
                _set_fatal_pipeline_error()

            if not _keep_searchable:
                # Notify Assembler — include fallback chunk if we saved one
                _interruptible_queue_put(
                    assembly_queue,
                    {
                        "doc_id": task.doc_id,
                        "page_num": task.page_num,
                        "text": "",
                        "status": PageStatus.CRITICAL_FAILED,
                        "chunk_path": _fallback_chunk,
                        "checkpoint_sha256": _fallback_digest,
                        "structure_data": None,
                        "ocr_confidence": 0.0,
                        "ocr_method": PageStatus.IMAGE_ONLY,
                        "ocr_language": "",
                        "detected_language": "",
                        "handwriting_data": None,
                        "signature_data": None,
                        "vertical_text_data": None,
                        "table_fallback_data": None,
                        "language_data": None,
                        "resume_metadata": None,
                        "cache_hit": False,
                    },
                )

        finally:
            # Every successful Queue.get owns exactly one task_done, including
            # an attempt that enqueues a replacement DPI attempt.  The new put
            # increments unfinished_tasks before this decrement, so join still
            # waits for the retry without leaking the consumed attempt.
            if not _dpi_retry_inline:
                image_queue.task_done()
                _complete_queue_obligation(page_reservation)
            else:
                _inline_retry_reservation = page_reservation
            if not _dpi_requeued:
                # Increment Metric (thread-safe, only when task is truly complete, not re-queued)
                with _pages_processed_lock:
                    global_pages_processed += 1
                if hasattr(task, "image"):
                    del task.image  # Force free RAM only after terminal attempt


def _finalize_and_write_feature(
    doc_obj,
    finalize_fn,
    write_fn,
    doc_path: str,
    output_folder: str,
    version: str,
    feature_name: str,
    log_format_fn,
    doc_id: str,
):
    """Finalize a feature document, write its sidecar JSON, and log the result.

    Shared helper for handwriting, classification, extraction, and NER blocks
    in the assembler thread.  Validation stays inline (too many special cases).
    """
    finalized = finalize_fn(doc_obj)
    rel_dir = os.path.dirname(_safe_source_relative_path(doc_path))
    result_path = write_fn(finalized, output_folder, rel_dir, version)
    if not result_path:
        raise RuntimeError(
            f"enabled {feature_name} writer produced no durable sidecar for {doc_id}"
        )
    logger.info(log_format_fn(finalized, result_path))
    return finalized


def _process_and_write_routing(
    doc_path: str,
    doc_id: str,
    finalized_classification,
    finalized_entities,
    version: str,
    output_folder: str | None = None,
):
    """Derive and write routing sidecars when the feature is enabled."""
    rel_dir = os.path.dirname(_safe_source_relative_path(doc_path))
    routing_doc = derive_document_routing(
        finalized_classification,
        finalized_entities,
    )
    routing_path = write_routing_json(
        routing_doc,
        output_folder or OUTPUT_FOLDER,
        rel_dir,
        version,
    )
    if not routing_path:
        raise RuntimeError(
            f"enabled document routing produced no durable sidecar for {doc_id}"
        )
    logger.info(
        "Routing: %s (routes=%d, review_required=%s) -> %s",
        routing_doc.primary_route,
        len(routing_doc.recommended_routes),
        routing_doc.review_required,
        routing_path,
    )


def _create_finalization_stage(doc):
    """Return an isolated attempt root and staged canonical PDF path."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    stage_root = tempfile.mkdtemp(
        prefix=f".finalize-{doc.doc_id}-",
        dir=OUTPUT_FOLDER,
    )
    output_root_real = os.path.realpath(OUTPUT_FOLDER)
    canonical_pdf_real = os.path.realpath(doc.output_pdf)
    try:
        relative_pdf = os.path.relpath(canonical_pdf_real, output_root_real)
        if relative_pdf == os.pardir or relative_pdf.startswith(os.pardir + os.sep):
            raise ValueError("canonical PDF escapes output root")
        staged_pdf = os.path.join(stage_root, relative_pdf)
        os.makedirs(os.path.dirname(staged_pdf), exist_ok=True)
        return stage_root, staged_pdf
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def _stage_output_path(stage_root, canonical_path):
    """Map one canonical output path into an isolated finalization stage."""
    stage_root_real = os.path.realpath(stage_root)
    output_root_real = os.path.realpath(OUTPUT_FOLDER)
    canonical_real = os.path.realpath(canonical_path)
    try:
        relative = os.path.relpath(canonical_real, output_root_real)
        common = os.path.commonpath([output_root_real, canonical_real])
    except ValueError as exc:
        raise ValueError("canonical output escapes output root") from exc
    if (
        os.path.normcase(common) != os.path.normcase(output_root_real)
        or relative == os.pardir
        or relative.startswith(os.pardir + os.sep)
    ):
        raise ValueError("canonical output escapes output root")
    staged = os.path.realpath(os.path.join(stage_root_real, relative))
    try:
        staged_common = os.path.commonpath([stage_root_real, staged])
    except ValueError as exc:
        raise ValueError("staged output escapes finalization root") from exc
    if os.path.normcase(staged_common) != os.path.normcase(stage_root_real):
        raise ValueError("staged output escapes finalization root")
    return staged


def _publish_finalization_stage(
    stage_root,
    canonical_pdf,
    *,
    post_install_check=None,
):
    """Publish one staged document transaction with PDF visibility last.

    Existing destinations are moved into an attempt-local rollback tree before
    replacement.  An in-process failure restores them in reverse order.  The
    canonical PDF is always the final published path.  ``post_install_check``
    is the rollback-protected commit action: after it returns successfully no
    further fallible/barrier operation occurs before rollback state is retired.
    """
    stage_root_real = os.path.realpath(stage_root)
    output_root_real = os.path.realpath(OUTPUT_FOLDER)
    canonical_pdf_real = os.path.realpath(canonical_pdf)
    rollback_root = os.path.join(stage_root_real, ".rollback")
    staged_files = []
    for root, dirs, files in os.walk(stage_root_real):
        dirs[:] = [name for name in dirs if name != ".rollback"]
        for filename in files:
            source = os.path.join(root, filename)
            relative = os.path.relpath(source, stage_root_real)
            destination = os.path.realpath(os.path.join(output_root_real, relative))
            try:
                common = os.path.commonpath([output_root_real, destination])
            except ValueError as exc:
                raise ValueError("staged output escapes output root") from exc
            if os.path.normcase(common) != os.path.normcase(output_root_real):
                raise ValueError("staged output escapes output root")
            staged_files.append((source, destination, relative))

    pdf_matches = [
        item
        for item in staged_files
        if os.path.normcase(item[1]) == os.path.normcase(canonical_pdf_real)
    ]
    if len(pdf_matches) != 1:
        raise RuntimeError("finalization stage must contain exactly one canonical PDF")
    ordered = [item for item in staged_files if item not in pdf_matches]
    ordered.extend(pdf_matches)

    installed = []
    try:
        for source, destination, relative in ordered:
            _raise_on_finalization_barrier("publication install")
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            backup = os.path.join(rollback_root, relative)
            had_existing = os.path.isfile(destination)
            record = {
                "destination": destination,
                "backup": backup,
                "had_existing": had_existing,
                "installed": False,
            }
            installed.append(record)
            if had_existing:
                os.makedirs(os.path.dirname(backup), exist_ok=True)
                os.replace(destination, backup)
                _fsync_directory(os.path.dirname(destination))
            os.replace(source, destination)
            record["installed"] = True
            _fsync_directory(os.path.dirname(destination))
        _raise_on_finalization_barrier("post-publication rollback window")
        if post_install_check is not None:
            _raise_on_finalization_barrier("publication commit action")
            post_install_check()
    except Exception:
        rollback_errors = []
        for record in reversed(installed):
            destination = record["destination"]
            backup = record["backup"]
            had_existing = record["had_existing"]
            try:
                if record["installed"] and os.path.exists(destination):
                    os.remove(destination)
                if had_existing and os.path.exists(backup):
                    os.replace(backup, destination)
                _fsync_directory(os.path.dirname(destination))
            except Exception as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        if rollback_errors:
            raise RuntimeError(
                "finalization publish failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            )
        raise

    shutil.rmtree(rollback_root, ignore_errors=True)
    return [record["destination"] for record in installed]


def _require_enabled_finalization_dependencies() -> None:
    """Fail closed when an explicitly enabled output surface is unavailable."""
    requirements = (
        (ENABLE_CUSTODY, _CUSTODY_AVAILABLE, "custody"),
        (ENABLE_VALIDATION, _VALIDATION_AVAILABLE, "validation"),
        (ENABLE_NER, _NER_AVAILABLE, "NER"),
        (
            ENABLE_PER_SPAN_LANGUAGE,
            _LANGUAGE_DETECTION_AVAILABLE,
            "per-span language",
        ),
        (ENABLE_HANDWRITING, _HANDWRITING_AVAILABLE, "handwriting"),
        (
            ENABLE_SIGNATURE_VERIFICATION,
            _SIGNATURE_VERIFICATION_AVAILABLE,
            "signature verification",
        ),
        (ENABLE_VERTICAL_TEXT, _VERTICAL_TEXT_AVAILABLE, "vertical text"),
        (ENABLE_TABLE_FALLBACK, _TABLE_FALLBACK_AVAILABLE, "table fallback"),
        (ENABLE_CLASSIFICATION, _CLASSIFICATION_AVAILABLE, "classification"),
        (ENABLE_EXTRACTION, _EXTRACTION_AVAILABLE, "extraction"),
        (ENABLE_EXTRACTION, _ENTITY_OUTPUT_AVAILABLE, "entity output"),
        (ENABLE_SPECIALIST_ROUTING, _ROUTING_AVAILABLE, "document routing"),
        (
            ENABLE_ENTITY_CONSOLIDATION,
            _ENTITY_CONSOLIDATOR_AVAILABLE,
            "entity consolidation",
        ),
        (
            ENABLE_RELATIONSHIP_EXTRACTION,
            _RELATIONSHIP_EXTRACTION_AVAILABLE,
            "relationship extraction",
        ),
        (
            ENABLE_RETRIEVAL_OUTPUT,
            _OUTPUT_ASSEMBLER_AVAILABLE,
            "retrieval output",
        ),
        (
            ENABLE_EXCEPTION_ROUTING,
            _EXCEPTION_ROUTER_AVAILABLE,
            "exception routing",
        ),
    )
    missing = [
        name for enabled, available, name in requirements if enabled and not available
    ]
    if missing:
        raise RuntimeError(
            "enabled finalization dependencies unavailable: " + ", ".join(missing)
        )


def _finalize_doc(doc, doc_id: str, page_data_snap: dict) -> None:
    """Finalize a complete document: merge PDFs, write sidecars, queue for compression.

    Runs inside a ThreadPoolExecutor spawned by the assembler thread so that
    multiple documents can be finalized in parallel without blocking page
    accumulation.

    ``page_data_snap`` is a snapshot dict with keys:
        texts, structure, validation, handwriting, signature,
        vertical_text, table_fallback, classification
    """
    final_pdf = None
    finalization_stage_root = None
    staged_output_pdf = None
    attempt_output_root = None
    try:
        _raise_on_finalization_barrier("finalizer start")
        _require_enabled_finalization_dependencies()
        expected_source_fingerprint = getattr(doc, "source_fingerprint", None)
        if expected_source_fingerprint:
            source_integrity_error = None
            try:
                source_authority = getattr(doc, "source_authority", None)
                if source_authority is None:
                    raise RuntimeError("document has no opened source authority")
                source_authority.verify(expected_source_fingerprint)
                current_source_fingerprint = compute_source_fingerprint(doc.path)
            except Exception as exc:
                current_source_fingerprint = {}
                source_integrity_error = f"source became unreadable: {exc}"
            if source_integrity_error or current_source_fingerprint.get(
                "content_sha256"
            ) != (expected_source_fingerprint.get("content_sha256")):
                message = source_integrity_error or (
                    "source bytes changed during processing; refusing to assemble "
                    "checkpoints from multiple source generations"
                )
                _set_fatal_pipeline_error()
                logger.critical("Source integrity failure for %s: %s", doc_id, message)
                log_failure(doc.path, 0, f"SOURCE_INTEGRITY_FAILED: {message}")
                custody_chain = getattr(doc, "custody_chain", None)
                if custody_chain:
                    custody_chain.append_event(
                        "processing_failed",
                        {
                            "stage": "source_integrity",
                            "error": message,
                            "expected_sha256": expected_source_fingerprint.get(
                                "content_sha256"
                            ),
                            "observed_sha256": current_source_fingerprint.get(
                                "content_sha256"
                            ),
                        },
                    )
                with doc_registry_lock:
                    doc_registry.pop(doc_id, None)
                return

        finalization_stage_root, staged_output_pdf = _create_finalization_stage(doc)
        attempt_output_root = finalization_stage_root
        final_pdf = fitz.open()
        _full_text_parts: list[str] = []  # maintenance review: O(n) list accumulator
        admitted_digests = page_data_snap.get("chunk_digests") or {}
        admitted_no_chunk_failures = set(
            page_data_snap.get("no_chunk_failures") or set()
        )
        if admitted_no_chunk_failures:
            raise ValueError(
                "lossless finalization forbids pages admitted without checkpoints: "
                f"{sorted(admitted_no_chunk_failures)}"
            )
        # Hold the per-document checkpoint artifact RLock across
        # read+hash+open+validate+insert so an in-process cache/write cannot
        # race finalization for the same doc.  Bytes are read once; fitz opens
        # from that same buffer (no path reopen between hash and insert).
        with _get_checkpoint_artifact_lock(doc_id):
            for p in range(1, doc.total_pages + 1):
                chunk_file = os.path.join(doc.temp_dir, f"{(p)}.pdf")
                if os.path.exists(chunk_file):
                    expected_chunk_digest = admitted_digests.get(p)
                    if not _is_valid_checkpoint_sha256(expected_chunk_digest):
                        raise ValueError(
                            "finalization missing admitted checkpoint_sha256 for "
                            f"{doc_id} p{p}"
                        )
                    chunk_bytes = _read_checkpoint_file_bytes(chunk_file)
                    actual_chunk_digest = _checkpoint_bytes_sha256(chunk_bytes)
                    if actual_chunk_digest != expected_chunk_digest:
                        raise ValueError(
                            "finalization chunk byte digest mismatch for "
                            f"{doc_id} p{p}: disk={actual_chunk_digest} "
                            f"snap={expected_chunk_digest}"
                        )
                    with fitz.open(stream=chunk_bytes, filetype="pdf") as c_doc:
                        if c_doc.page_count != 1 or c_doc.needs_pass:
                            raise ValueError(
                                f"finalization chunk is not a readable single-page PDF: {chunk_file}"
                            )
                        c_doc.load_page(0)
                        chunk_meta = _load_page_checkpoint_metadata_from_document(
                            c_doc,
                            document_id=doc_id,
                            page_num=p,
                        )
                        if chunk_meta is None:
                            raise ValueError(
                                f"finalization chunk missing V3 provenance: {chunk_file}"
                            )
                        incompatibility = _checkpoint_resume_incompatibility(chunk_meta)
                        if incompatibility:
                            raise ValueError(
                                f"finalization chunk profile incompatible: {incompatibility}"
                            )
                        chunk_text = _validated_checkpoint_text_from_document(
                            c_doc,
                            chunk_meta,
                        )
                        expected_text = page_data_snap["texts"].get(p, "")
                        if chunk_text != expected_text:
                            raise ValueError(
                                "finalization text snapshot mismatch for "
                                f"{doc_id} p{p}: chunk digest="
                                f"{_canonical_text_sha256(chunk_text)} snap digest="
                                f"{_canonical_text_sha256(expected_text)}"
                            )
                        final_pdf.insert_pdf(c_doc)
                else:
                    raise FileNotFoundError(
                        "finalization lost an admitted canonical checkpoint "
                        f"for {doc_id} p{p}"
                    )
                _full_text_parts.append(
                    page_data_snap["texts"].get(p, "") + "\n\f"
                )  # Standard Page Break (Form Feed)

        full_text = "".join(_full_text_parts)

        if final_pdf.page_count != doc.total_pages:
            raise RuntimeError(
                "lossless finalization page count mismatch: "
                f"output={final_pdf.page_count} source={doc.total_pages}"
            )

        output_page_count = final_pdf.page_count
        final_pdf.save(staged_output_pdf, deflate=True)
        final_pdf.close()
        final_pdf = None

        # Shared version import for validation, NER, and Phase 6 blocks
        from ocr_local.config.version import __version__ as _p6_version

        # --- Processing Integrity Validation (Phase 5) ---
        finalized_validation = None
        if ENABLE_VALIDATION and _VALIDATION_AVAILABLE:
            try:
                # Verify page count
                # Build validation report
                doc_val = DocumentValidation(
                    document_id=doc_id,
                    source_file=doc.path,
                    source_page_count=doc.total_pages,
                    output_page_count=output_page_count,
                )
                # Collect page-level data
                for p in range(1, doc.total_pages + 1):
                    page_data = page_data_snap["validation"].get(p)
                    if page_data:
                        doc_val.pages.append(page_data)
                    else:
                        doc_val.pages.append(
                            {
                                "page_num": p,
                                "ocr_method": PageStatus.UNKNOWN,
                                "ocr_language": "",
                                "ocr_confidence": 0.0,
                                "text_length": 0,
                                "has_text": False,
                                "status": "unknown",
                            }
                        )

                # Compute output PDF hash
                doc_val.output_hash = compute_output_hash(staged_output_pdf)

                # Finalize and classify
                doc_val = finalize_validation(doc_val)
                finalized_validation = doc_val

                # Write sidecar JSON
                rel_dir = os.path.dirname(_safe_source_relative_path(doc.path))
                val_path = write_validation_json(
                    doc_val,
                    attempt_output_root,
                    rel_dir,
                    _p6_version,
                )
                if not val_path:
                    raise RuntimeError(
                        "enabled validation writer produced no durable sidecar"
                    )
                logger.info(
                    "Validation: %s (conf=%.3f) -> %s",
                    doc_val.classification,
                    doc_val.overall_confidence,
                    val_path,
                )
                if not doc_val.page_count_match:
                    raise RuntimeError(
                        "validation page count mismatch: "
                        f"source={doc_val.source_page_count} "
                        f"output={doc_val.output_page_count}"
                    )
            except Exception as val_err:
                raise RuntimeError(
                    f"enabled validation finalization failed for {doc_id}"
                ) from val_err

        # --- Named Entity Recognition (Phase 5A) ---
        finalized_ner = None
        if ENABLE_NER and _NER_AVAILABLE:
            try:
                doc_ner = DocumentNER(document_id=doc_id, source_file=doc.path)
                for p in range(1, doc.total_pages + 1):
                    page_text = page_data_snap["texts"].get(p, "")
                    if page_text.strip():
                        entities = extract_entities(page_text, p)
                        entities.extend(extract_custom_entities(page_text, p))
                        doc_ner.pages.append({"page_num": p, "entities": entities})
                finalized_ner = _finalize_and_write_feature(
                    doc_ner,
                    finalize_ner,
                    write_ner_json,
                    doc.path,
                    os.path.join(attempt_output_root, "EXPORT", "NER"),
                    _p6_version,
                    "NER",
                    lambda d, _path: (
                        f"NER: {d.total_entities} entities extracted for {doc_id}"
                    ),
                    doc_id,
                )
            except Exception as ner_err:
                raise RuntimeError(
                    f"enabled NER finalization failed for {doc_id}"
                ) from ner_err

        # --- Per-page Language Detection Finalization (Plan A -- PR A2) ---
        if ENABLE_PER_SPAN_LANGUAGE and _LANGUAGE_DETECTION_AVAILABLE:
            try:
                page_langs = []
                for p in range(1, doc.total_pages + 1):
                    pl = page_data_snap["language"].get(p)
                    if pl is None:
                        pl = PageLanguage(
                            page_num=p,
                            primary_language="und",
                            primary_confidence=0.0,
                            languages_detected=[],
                            language_char_shares={},
                            scripts_detected=[],
                            mixed_script=False,
                            span_count=0,
                            spans_labeled=0,
                            spans=[],
                        )
                    page_langs.append(pl)
                if page_langs:
                    language_provenance = _language_model_provenance()
                    doc_lang = finalize_document_language(
                        document_id=doc_id,
                        source_file=doc.path,
                        pages=page_langs,
                        fasttext_model_sha256=language_provenance[
                            "fasttext_model_sha256"
                        ],
                        tokenizer_sha256=language_provenance["tokenizer_sha256"],
                        pipeline_version=_p6_version,
                    )
                    lang_path = write_language_json(
                        doc_lang=doc_lang,
                        output_base_dir=attempt_output_root,
                        source_file=_safe_source_relative_path(doc.path),
                        include_spans=LANGUAGE_INCLUDE_SPANS,
                    )
                    if not lang_path:
                        raise RuntimeError(
                            "enabled language writer produced no durable sidecar"
                        )
                    logger.info(
                        "Language: primary=%s conf=%.2f pages=%d -> %s",
                        doc_lang.primary_language,
                        doc_lang.primary_confidence,
                        doc_lang.page_count,
                        lang_path,
                    )

                    # --- Document-level custody + Prometheus (PR A4) ---
                    try:
                        _asm_doc_state = doc
                        _asm_custody = getattr(
                            _asm_doc_state,
                            "custody_chain",
                            None,
                        )
                        if _asm_custody is not None:
                            _asm_custody.append_event(
                                "LANGUAGE_DETECTED",
                                {
                                    "level": "document",
                                    "primary_language": doc_lang.primary_language,
                                    "primary_confidence": round(
                                        doc_lang.primary_confidence,
                                        4,
                                    ),
                                    "languages_detected": list(
                                        doc_lang.languages_detected,
                                    ),
                                    "language_char_shares": {
                                        k: round(v, 4)
                                        for k, v in doc_lang.language_char_shares.items()
                                    },
                                    "page_count": doc_lang.page_count,
                                    "pages_with_mixed_script": doc_lang.pages_with_mixed_script,
                                    "detection_engine": "fasttext+script_heuristic",
                                    **language_provenance,
                                },
                            )
                    except Exception as custody_error:
                        raise RuntimeError(
                            "document language custody persistence failed"
                        ) from custody_error

                    try:
                        from api.prometheus import (
                            ocr_language_confidence as _pc,
                        )
                        from api.prometheus import (
                            ocr_language_detected_total as _pd,
                        )

                        try:
                            from ocr_local.config.language_config import (
                                LANGUAGE_REGISTRY as _REG,
                            )

                            _entry = _REG.get(doc_lang.primary_language)
                            _tier = _entry.tier if _entry is not None else "core"
                        except Exception:
                            _tier = "core"
                        _pd.labels(
                            lang=doc_lang.primary_language,
                            tier=_tier,
                            level="document",
                        ).inc()
                        _pc.labels(
                            lang=doc_lang.primary_language,
                            level="document",
                        ).observe(doc_lang.primary_confidence)
                    except Exception:
                        pass

                    # --- PDF XMP language embedding (PR A4) ---
                    # Write the detected BCP-47 language tag into the output
                    # PDF's XMP metadata so downstream consumers can read the
                    # language label from the PDF itself.  Wrapped in
                    # try/except -- XMP write failure must never block
                    # compression or delivery.
                    if doc_lang.primary_language != "und":
                        try:
                            from ocr_local.config.language_config import (
                                LANGUAGE_REGISTRY as _REG_XMP,
                            )

                            _entry_xmp = _REG_XMP.get(
                                doc_lang.primary_language,
                            )
                            _bcp47 = (
                                _entry_xmp.bcp47
                                if _entry_xmp is not None and _entry_xmp.bcp47
                                else doc_lang.primary_language
                            )
                            _pdf_path = str(staged_output_pdf)
                            if os.path.isfile(_pdf_path):
                                _pdf_doc = fitz.open(_pdf_path)
                                try:
                                    _xmp = _pdf_doc.get_xml_metadata() or ""
                                    if "<dc:language>" not in _xmp:
                                        _lang_xmp = (
                                            "<dc:language><rdf:Bag>"
                                            f"<rdf:li>{_bcp47}</rdf:li>"
                                            "</rdf:Bag></dc:language>"
                                        )
                                        if "</rdf:Description>" in _xmp:
                                            _xmp = _xmp.replace(
                                                "</rdf:Description>",
                                                f"{_lang_xmp}</rdf:Description>",
                                                1,
                                            )
                                        else:
                                            _xmp = _xmp + _lang_xmp
                                        _pdf_doc.set_xml_metadata(_xmp)
                                        try:
                                            _pdf_doc.saveIncr()
                                        except Exception:
                                            # Incremental save can fail on some
                                            # PDFs; fall back to a full save.
                                            _pdf_doc.save(
                                                _pdf_path,
                                                incremental=False,
                                                deflate=True,
                                            )
                                finally:
                                    _pdf_doc.close()
                        except Exception as xmp_err:
                            raise RuntimeError(
                                f"enabled language XMP embedding failed for {doc_id}"
                            ) from xmp_err
            except Exception as lang_err:
                raise RuntimeError(
                    f"enabled language finalization failed for {doc_id}"
                ) from lang_err

        # --- Translation sidecar (Plan B Wave M1) ---
        # Opt-in via PipelineConfig.enable_translation; default False.
        # Fail-open: any exception is logged and the OCR job continues
        # uninterrupted -- a translation failure must NEVER fail the OCR
        # job itself.
        try:
            _cfg_active = _ACTIVE_PIPELINE_CONFIG
            if _cfg_active is not None and getattr(
                _cfg_active,
                "enable_translation",
                False,
            ):
                _target_langs = list(
                    getattr(_cfg_active, "translation_target_languages", []) or []
                )
                if _target_langs:
                    if ENABLE_PRIVILEGE_DETECTION:
                        _structures = list(
                            (page_data_snap.get("structure") or {}).values()
                        )
                        page_data_snap["privilege_flag"] = bool(
                            page_data_snap.get("privilege_flag")
                            or _detect_privilege_indicators(_structures)
                        )
                    from ocr_local.translation.api import (
                        translate_document as _translate_document,
                    )
                    from ocr_local.translation.sidecar import (
                        write_translation_json as _write_translation_json,
                    )

                    _tenant_id = getattr(doc, "tenant_id", None) or "default"
                    _custody_chain = getattr(doc, "custody_chain", None)
                    # Wave M2 PR B14 -- the facade routes through
                    # ``select_engine_for_tenant`` (router_v2) when
                    # ``_tenant_id != "default"``.  Glossary preprocessing
                    # is applied per-page upstream of segmentation so
                    # tenant-overridden terminology reaches the engine.
                    _translation_results = _translate_document(
                        doc_path=str(doc.path),
                        target_languages=_target_langs,
                        tenant_id=_tenant_id,
                        page_data_snap=page_data_snap,
                        custody_chain=_custody_chain,
                        output_dir=attempt_output_root,
                        config=_cfg_active,
                    )
                    for _tdoc in _translation_results:
                        try:
                            _write_translation_json(_tdoc, attempt_output_root)
                        except Exception as _swe:
                            logger.warning(
                                "Translation sidecar write failed for %s "
                                "(target=%s): %s",
                                doc_id,
                                _tdoc.target_language,
                                _swe,
                            )
        except Exception as _te:
            logger.warning(
                "Translation sidecar failed (non-fatal) for %s: %s",
                doc_id,
                _te,
            )
        # --- End translation sidecar ---

        # --- Handwriting Detection Finalization (Phase 6B) ---
        if ENABLE_HANDWRITING and _HANDWRITING_AVAILABLE:
            try:
                doc_hw = DocumentHandwriting(
                    document_id=doc_id,
                    source_file=doc.path,
                )
                for p in range(1, doc.total_pages + 1):
                    page_hw = page_data_snap["handwriting"].get(p)
                    if page_hw is not None:
                        doc_hw.pages.append(page_hw)
                _finalize_and_write_feature(
                    doc_hw,
                    finalize_handwriting,
                    write_handwriting_json,
                    doc.path,
                    attempt_output_root,
                    _p6_version,
                    "Handwriting",
                    lambda d, path: (
                        f"Handwriting: coverage={d.overall_handwriting_coverage:.2f} "
                        f"primary={d.is_primarily_handwritten} -> {path}"
                    ),
                    doc_id,
                )
            except Exception as hw_err:
                raise RuntimeError(
                    f"enabled handwriting finalization failed for {doc_id}"
                ) from hw_err

        # --- Signature Verification Finalization (Phase 6D) ---
        if ENABLE_SIGNATURE_VERIFICATION and _SIGNATURE_VERIFICATION_AVAILABLE:
            try:
                doc_sig = DocumentSignatureVerification(
                    document_id=doc_id,
                    source_file=doc.path,
                )
                for p in range(1, doc.total_pages + 1):
                    page_sig = page_data_snap["signature"].get(p)
                    if page_sig is not None:
                        doc_sig.pages.append(page_sig)
                _finalize_and_write_feature(
                    doc_sig,
                    finalize_signature_verification,
                    write_signature_verification_json,
                    doc.path,
                    attempt_output_root,
                    _p6_version,
                    "SignatureVerification",
                    lambda d, path: (
                        "SignatureVerification: "
                        f"candidate_pages={d.total_candidate_pages} "
                        f"review_pages={d.total_review_pages} -> {path}"
                    ),
                    doc_id,
                )
            except Exception as sig_err:
                raise RuntimeError(
                    f"enabled signature finalization failed for {doc_id}"
                ) from sig_err

        # --- CJK Vertical Text Analysis Finalization (Phase 6D) ---
        if ENABLE_VERTICAL_TEXT and _VERTICAL_TEXT_AVAILABLE:
            try:
                doc_vt = DocumentVerticalText(
                    document_id=doc_id,
                    source_file=doc.path,
                )
                for p in range(1, doc.total_pages + 1):
                    page_vt = page_data_snap["vertical_text"].get(p)
                    if page_vt is not None:
                        doc_vt.pages.append(page_vt)
                _finalize_and_write_feature(
                    doc_vt,
                    finalize_vertical_text,
                    write_vertical_analysis_json,
                    doc.path,
                    attempt_output_root,
                    _p6_version,
                    "VerticalText",
                    lambda d, path: (
                        f"VerticalText: vertical_pages={d.total_vertical_pages} "
                        f"mixed_pages={d.total_mixed_pages} "
                        f"has_vertical={d.has_vertical_content} -> {path}"
                    ),
                    doc_id,
                )
            except Exception as vt_err:
                raise RuntimeError(
                    f"enabled vertical-text finalization failed for {doc_id}"
                ) from vt_err

        # --- Table Fallback Analysis Finalization (Phase 4D) ---
        if ENABLE_TABLE_FALLBACK and _TABLE_FALLBACK_AVAILABLE:
            try:
                page_analyses = []
                for p in range(1, doc.total_pages + 1):
                    page_tf = page_data_snap["table_fallback"].get(p)
                    if page_tf is not None:
                        page_analyses.append(page_tf)
                _finalize_and_write_feature(
                    page_analyses,
                    lambda analyses: finalize_table_fallback(
                        analyses,
                        document_id=doc_id,
                        source_file=doc.path,
                    ),
                    write_table_fallback_json,
                    doc.path,
                    attempt_output_root,
                    _p6_version,
                    "TableFallback",
                    lambda d, path: (
                        f"TableFallback: tables={d.total_tables} "
                        f"triggered={d.total_fallback_triggered} "
                        f"improved={d.total_fallback_improved} -> {path}"
                    ),
                    doc_id,
                )
            except Exception as tf_err:
                raise RuntimeError(
                    f"enabled table-fallback finalization failed for {doc_id}"
                ) from tf_err

        # --- Document Classification (Phase 6A) ---
        finalized_classification = None
        if ENABLE_CLASSIFICATION and _CLASSIFICATION_AVAILABLE:
            try:
                doc_cls = DocumentClassification(
                    document_id=doc_id,
                    source_file=doc.path,
                )
                for p in range(1, doc.total_pages + 1):
                    page_text = page_data_snap["texts"].get(p, "")
                    text_result = classify_page_by_text(page_text, p)

                    # Tier 2: layout features from DocIntel
                    page_struct = page_data_snap["structure"].get(p)
                    layout_result = None
                    if page_struct:
                        layout_result = classify_page_by_layout(
                            page_struct.get("layout_regions", []),
                            page_struct.get("tables", []),
                            page_struct.get("form_fields", []),
                            p,
                        )

                    page_cls = classify_page_ensemble(text_result, layout_result, p)

                    # Cross-reference handwriting detection
                    page_hw = page_data_snap["handwriting"].get(p)
                    if page_hw is not None:
                        has_hw = (
                            page_hw.has_handwriting
                            if hasattr(page_hw, "has_handwriting")
                            else page_hw.get("has_handwriting", False)
                        )
                        if has_hw:
                            page_cls.is_handwritten = True

                    page_data_snap["classification"][p] = page_cls
                    doc_cls.pages.append(page_cls)

                finalized_classification = _finalize_and_write_feature(
                    doc_cls,
                    finalize_classification,
                    write_classification_json,
                    doc.path,
                    attempt_output_root,
                    _p6_version,
                    "Classification",
                    lambda d, path: (
                        f"Classification: {d.document_type} "
                        f"(conf={d.document_confidence:.2f}) -> {path}"
                    ),
                    doc_id,
                )
            except Exception as cls_err:
                raise RuntimeError(
                    f"enabled classification finalization failed for {doc_id}"
                ) from cls_err

        # --- Structured Extraction (Phase 6C) ---
        finalized_extraction = None
        finalized_entities = None
        if ENABLE_EXTRACTION and _EXTRACTION_AVAILABLE:
            try:
                doc_ext = DocumentExtraction(
                    document_id=doc_id,
                    source_file=doc.path,
                )
                for p in range(1, doc.total_pages + 1):
                    page_text = page_data_snap["texts"].get(p, "")
                    if page_text.strip():
                        page_ext = extract_page_fields(page_text, p)
                        doc_ext.pages.append(page_ext)
                finalized_extraction = _finalize_and_write_feature(
                    doc_ext,
                    finalize_extraction,
                    write_extraction_json,
                    doc.path,
                    attempt_output_root,
                    _p6_version,
                    "Extraction",
                    lambda d, path: (
                        f"Extraction: {d.total_fields} fields "
                        f"({d.extraction_engine}) -> {path}"
                    ),
                    doc_id,
                )
                finalized_entities = _finalize_and_write_feature(
                    finalized_extraction,
                    finalize_entity_output,
                    write_entities_json,
                    doc.path,
                    attempt_output_root,
                    _p6_version,
                    "Entities",
                    lambda d, path: (
                        "Entities: "
                        f"{d.total_entities} entities, "
                        f"{d.total_relationships} relationships, "
                        f"{d.total_key_value_pairs} pairs -> {path}"
                    ),
                    doc_id,
                )
            except Exception as ext_err:
                raise RuntimeError(
                    f"enabled extraction finalization failed for {doc_id}"
                ) from ext_err

        # --- Document routing (.routing.json via routing.py) ---
        # Flag name ENABLE_SPECIALIST_ROUTING is historical; it does NOT call
        # specialist_routing.SpecialistRouter (.specialist.json). See wire-or-hide.
        if ENABLE_SPECIALIST_ROUTING and _ROUTING_AVAILABLE:
            if finalized_classification is None:
                raise RuntimeError(
                    "enabled document routing requires finalized classification"
                )
            _process_and_write_routing(
                doc.path,
                doc_id,
                finalized_classification,
                finalized_entities,
                _p6_version,
                output_folder=attempt_output_root,
            )

        # --- Unified Entity Consolidation ---
        if ENABLE_ENTITY_CONSOLIDATION and _ENTITY_CONSOLIDATOR_AVAILABLE:
            try:
                consolidated = consolidate_entities(
                    ner_results=finalized_ner,
                    extraction_results=finalized_extraction,
                    classification_results=finalized_classification,
                    doc_name=os.path.basename(doc.path),
                    pipeline_version=_p6_version,
                )
                # --- Cross-Entity Relationship Extraction ---
                if (
                    ENABLE_RELATIONSHIP_EXTRACTION
                    and _RELATIONSHIP_EXTRACTION_AVAILABLE
                ):
                    page_texts = page_data_snap["texts"]
                    consolidated = extract_and_attach_relationships(
                        consolidated, page_texts
                    )
                    rel_count = len(consolidated.get("relationships", []))
                    logger.info(
                        "Extracted %d entity relationships for %s",
                        rel_count,
                        doc_id,
                    )
                rel_dir = os.path.dirname(_safe_source_relative_path(doc.path))
                cons_path = write_consolidated_entities_json(
                    consolidated,
                    attempt_output_root,
                    rel_dir,
                    doc.path,
                )
                if not cons_path:
                    raise RuntimeError(
                        "enabled entity consolidation produced no durable sidecar"
                    )
                logger.info(
                    "EntityConsolidation: %d entities, %d kv_pairs, class=%s -> %s",
                    consolidated["summary"]["total_entities"],
                    consolidated["summary"]["total_kv_pairs"],
                    consolidated["summary"]["primary_classification"],
                    cons_path,
                )
            except Exception as cons_err:
                raise RuntimeError(
                    f"enabled entity consolidation failed for {doc_id}"
                ) from cons_err

        # Text Output to Separate Folder
        base_name = os.path.splitext(os.path.basename(doc.output_pdf))[0]
        staged_txt_dir = _stage_output_path(
            attempt_output_root,
            doc.output_txt_dir,
        )
        os.makedirs(staged_txt_dir, exist_ok=True)
        txt_path = os.path.join(staged_txt_dir, base_name + ".txt")
        text_export_root = os.path.join(
            attempt_output_root,
            "EXPORT",
            "TEXT",
        )
        write_text_atomic(
            txt_path,
            full_text,
            allowed_root=text_export_root,
        )

        # Document Intelligence — write sidecar JSON (Phase 3A)
        if ENABLE_DOCUMENT_INTELLIGENCE:
            doc_structure_data = page_data_snap["structure"]
            pages_list = []
            for p in range(1, doc.total_pages + 1):
                page_data = doc_structure_data.get(p)
                if page_data is None:
                    raise RuntimeError(
                        f"enabled document intelligence missing page {p} for {doc_id}"
                    )
                page_data = dict(page_data)
                page_data["page_num"] = p
                pages_list.append(page_data)
            # Phase 3B: Export tables before writing JSON
            # (mutates pages_list to add extracted_files paths)
            if EXPORT_TABLES:
                t_count = write_extracted_tables(
                    doc_id,
                    doc.path,
                    pages_list,
                    attempt_output_root,
                )
                if t_count:
                    logger.info("Exported %d tables for %s", t_count, doc.path)

            json_path = write_structure_json(
                doc_id,
                doc.path,
                pages_list,
                attempt_output_root,
            )
            if not json_path:
                raise RuntimeError(
                    "enabled document intelligence produced no durable sidecar"
                )
            logger.info("Structure JSON written: %s", json_path)

        # --- Unified Retrieval Output (maintenance review) ---
        if ENABLE_RETRIEVAL_OUTPUT and _OUTPUT_ASSEMBLER_AVAILABLE:
            try:
                # Build text_by_page list from page data snapshot
                _ret_text_by_page = []
                for p in range(1, doc.total_pages + 1):
                    _ret_text_by_page.append(
                        {
                            "page": p,
                            "text": page_data_snap["texts"].get(p, ""),
                        }
                    )

                # Build entities_data dict from finalized NER (if available)
                _ret_entities_data = None
                if finalized_ner is not None:
                    _ret_entities_data = {
                        "pages": [
                            pg
                            if isinstance(pg, dict)
                            else {
                                "page_num": getattr(pg, "page_num", 0),
                                "entities": getattr(pg, "entities", []),
                            }
                            for pg in getattr(finalized_ner, "pages", [])
                        ],
                    }

                # Build classification_data dict from finalized classification
                _ret_classification_data = None
                if finalized_classification is not None:
                    _ret_classification_data = {
                        "document_type": getattr(
                            finalized_classification, "document_type", ""
                        ),
                        "confidence": getattr(
                            finalized_classification, "document_confidence", 0.0
                        ),
                        "classification_method": "ensemble",
                    }

                # Build extraction_data dict from finalized extraction
                _ret_extraction_data = None
                if finalized_extraction is not None:
                    _ext_pages = []
                    for pg in getattr(finalized_extraction, "pages", []):
                        if isinstance(pg, dict):
                            _ext_pages.append(pg)
                        else:
                            _ext_pages.append(
                                {
                                    "page_num": getattr(pg, "page_num", 0),
                                    "fields": getattr(pg, "fields", []),
                                }
                            )
                    _ret_extraction_data = {"pages": _ext_pages}

                # Build structure_data dict from page data snapshot
                _ret_structure_data = None
                if page_data_snap.get("structure"):
                    _struct_pages = []
                    for p in range(1, doc.total_pages + 1):
                        s_data = page_data_snap["structure"].get(p)
                        if s_data:
                            s_dict = dict(s_data) if isinstance(s_data, dict) else {}
                            s_dict["page_num"] = p
                            _struct_pages.append(s_dict)
                    if _struct_pages:
                        _ret_structure_data = {"pages": _struct_pages}

                # Build validation_data dict from page data snapshot
                _ret_validation_data = None
                if finalized_validation is not None:
                    _validation_languages = sorted(
                        {
                            (
                                page.get("ocr_language", "")
                                if isinstance(page, dict)
                                else getattr(page, "ocr_language", "")
                            )
                            for page in finalized_validation.pages
                            if (
                                page.get("ocr_language", "")
                                if isinstance(page, dict)
                                else getattr(page, "ocr_language", "")
                            )
                        }
                    )
                    _detected_content_languages = set()
                    for _language_page in page_data_snap.get("language", {}).values():
                        if isinstance(_language_page, dict):
                            _page_languages = _language_page.get(
                                "languages_detected",
                                [],
                            )
                            _primary_language = _language_page.get(
                                "primary_language",
                                "",
                            )
                        else:
                            _page_languages = getattr(
                                _language_page,
                                "languages_detected",
                                [],
                            )
                            _primary_language = getattr(
                                _language_page,
                                "primary_language",
                                "",
                            )
                        _detected_content_languages.update(
                            str(language)
                            for language in (_page_languages or [])
                            if language
                        )
                        if _primary_language:
                            _detected_content_languages.add(str(_primary_language))
                    _ret_validation_data = {
                        "page_count": {
                            "source": finalized_validation.source_page_count,
                            "output": finalized_validation.output_page_count,
                            "match": finalized_validation.page_count_match,
                        },
                        "quality": {
                            "classification": finalized_validation.classification,
                            "overall_confidence": finalized_validation.overall_confidence,
                            "text_extraction_rate": finalized_validation.text_extraction_rate,
                            "total_text_length": finalized_validation.total_text_length,
                            "ocr_methods_used": finalized_validation.ocr_methods_used,
                        },
                        "languages": (
                            sorted(_detected_content_languages) or _validation_languages
                        ),
                        "output_hash": finalized_validation.output_hash,
                    }
                elif page_data_snap.get("validation"):
                    _val_pages = []
                    for p in range(1, doc.total_pages + 1):
                        v_data = page_data_snap["validation"].get(p)
                        if v_data:
                            _val_pages.append(v_data)
                    if _val_pages:
                        # Compute summary metrics for the validation data
                        _total_conf = sum(
                            float(vp.get("ocr_confidence", 0.0) or 0.0)
                            for vp in _val_pages
                        )
                        _total_text_len = sum(
                            int(vp.get("text_length", 0) or 0) for vp in _val_pages
                        )
                        _text_pages = sum(1 for vp in _val_pages if vp.get("has_text"))
                        _avg_conf = _total_conf / len(_val_pages) if _val_pages else 0.0
                        _text_rate = (
                            _text_pages / len(_val_pages) if _val_pages else 0.0
                        )
                        _ret_validation_data = {
                            "page_count": {"source": doc.total_pages},
                            "quality": {
                                "overall_confidence": _avg_conf,
                                "text_extraction_rate": _text_rate,
                                "total_text_length": _total_text_len,
                                "ocr_methods_used": list(
                                    {
                                        vp.get("ocr_method", "")
                                        for vp in _val_pages
                                        if vp.get("ocr_method")
                                    }
                                ),
                            },
                        }

                # Build handwriting_data dict from page data snapshot
                _ret_handwriting_data = None
                if page_data_snap.get("handwriting"):
                    _hw_pages = []
                    _total_hw_pages = 0
                    for p in range(1, doc.total_pages + 1):
                        hw = page_data_snap["handwriting"].get(p)
                        if hw is not None:
                            has_hw = (
                                hw.has_handwriting
                                if hasattr(hw, "has_handwriting")
                                else hw.get("has_handwriting", False)
                            )
                            regions = (
                                hw.handwriting_regions
                                if hasattr(hw, "handwriting_regions")
                                else hw.get("handwriting_regions", [])
                            )
                            _hw_pages.append(
                                {
                                    "page_num": p,
                                    "handwriting_regions": (
                                        regions if isinstance(regions, list) else []
                                    ),
                                }
                            )
                            if has_hw:
                                _total_hw_pages += 1
                    if _hw_pages:
                        _ret_handwriting_data = {
                            "pages": _hw_pages,
                            "total_handwritten_pages": _total_hw_pages,
                            "overall_handwriting_coverage": 0.0,
                            "is_primarily_handwritten": (
                                _total_hw_pages > len(_hw_pages) / 2
                            ),
                        }

                # Assemble the retrieval document
                ret_doc = assemble_retrieval_output(
                    document_id=doc_id,
                    source_file=os.path.basename(doc.path),
                    ocr_text=full_text,
                    text_by_page=_ret_text_by_page,
                    entities_data=_ret_entities_data,
                    classification_data=_ret_classification_data,
                    extraction_data=_ret_extraction_data,
                    structure_data=_ret_structure_data,
                    validation_data=_ret_validation_data,
                    handwriting_data=_ret_handwriting_data,
                    pipeline_version=_p6_version,
                )

                # Write JSON and Markdown outputs
                rel_dir = os.path.dirname(_safe_source_relative_path(doc.path))
                json_out = write_retrieval_json(
                    ret_doc,
                    attempt_output_root,
                    rel_dir,
                )
                md_out = write_retrieval_markdown(
                    ret_doc,
                    attempt_output_root,
                    rel_dir,
                )
                if not json_out or not md_out:
                    raise RuntimeError(
                        "enabled retrieval output did not produce both JSON and Markdown"
                    )
                logger.info(
                    "Retrieval output: %s (%d entities, %d kv_pairs) -> %s",
                    ret_doc.classification.get("label", "unknown"),
                    len(ret_doc.entities),
                    len(ret_doc.key_value_pairs),
                    json_out,
                )
                logger.debug("Retrieval Markdown: %s", md_out)
            except Exception as ret_err:
                raise RuntimeError(
                    f"enabled retrieval output failed for {doc_id}"
                ) from ret_err

        # --- Exception Routing (maintenance review) ---
        if ENABLE_EXCEPTION_ROUTING and _EXCEPTION_ROUTER_AVAILABLE:
            try:
                # Build validation_data dict for the router from page-level data
                _er_validation_data = None
                if page_data_snap.get("validation"):
                    _er_val_pages = list(page_data_snap["validation"].values())
                    if _er_val_pages:
                        _er_total_conf = sum(
                            float(vp.get("ocr_confidence", 0.0) or 0.0)
                            for vp in _er_val_pages
                        )
                        _er_avg_conf = (
                            _er_total_conf / len(_er_val_pages)
                            if _er_val_pages
                            else 0.0
                        )
                        _er_image_only = sum(
                            1
                            for vp in _er_val_pages
                            if vp.get("ocr_method") == "IMAGE_ONLY"
                        )
                        # Derive quality classification from average confidence
                        _er_quality_cls = "good"
                        if _er_avg_conf < 0.3:
                            _er_quality_cls = "review_required"
                        elif _er_avg_conf < 0.5:
                            _er_quality_cls = "degraded"
                        elif _er_avg_conf < 0.7:
                            _er_quality_cls = "acceptable"
                        _er_validation_data = {
                            "quality": {
                                "overall_confidence": _er_avg_conf,
                                "classification": _er_quality_cls,
                                "pages_image_only": _er_image_only,
                            },
                        }

                # Build classification_data dict
                _er_classification_data = None
                if finalized_classification is not None:
                    _er_classification_data = {
                        "confidence": getattr(
                            finalized_classification, "document_confidence", 1.0
                        ),
                    }

                # Build handwriting_data dict from page-level data
                _er_handwriting_data = None
                if page_data_snap.get("handwriting"):
                    _er_hw_pages = [
                        v
                        for v in page_data_snap["handwriting"].values()
                        if v is not None
                    ]
                    if _er_hw_pages:
                        _er_any_hw = any(
                            (
                                hw.has_handwriting
                                if hasattr(hw, "has_handwriting")
                                else hw.get("has_handwriting", False)
                            )
                            for hw in _er_hw_pages
                        )
                        _er_hw_count = sum(
                            1
                            for hw in _er_hw_pages
                            if (
                                hw.has_handwriting
                                if hasattr(hw, "has_handwriting")
                                else hw.get("has_handwriting", False)
                            )
                        )
                        _er_handwriting_data = {
                            "document_summary": {
                                "is_primarily_handwritten": (
                                    _er_hw_count > len(_er_hw_pages) / 2
                                ),
                                "handwriting_detected": _er_any_hw,
                            },
                        }

                _router = ExceptionRouter()
                _routing_decision = _router.evaluate(
                    validation_data=_er_validation_data,
                    classification_data=_er_classification_data,
                    handwriting_data=_er_handwriting_data,
                    extracted_text=full_text[:10240] if full_text else "",
                )
                if _routing_decision.should_route:
                    logger.warning(
                        "Exception routing: %s flagged for review — rules: %s",
                        doc_id,
                        ", ".join(_routing_decision.triggered_rules),
                    )
                    # Post-sweep HIGH: enqueue HITL ReviewQueue when available
                    # (sidecar remains the forensic artifact if queue unavailable).
                    try:
                        from api.review_queue import ReviewQueue

                        _er_job_id = (
                            os.environ.get("OCR_JOB_ID")
                            or getattr(doc, "job_id", None)
                            or doc_id
                        )
                        _er_rq = ReviewQueue()
                        _er_ids = _router.enqueue_decision(
                            _routing_decision,
                            job_id=str(_er_job_id),
                            review_queue=_er_rq,
                            quality_classification=str(
                                (
                                    (_er_validation_data or {}).get("quality")
                                    or _er_validation_data
                                    or {}
                                ).get("classification", "")
                                if isinstance(_er_validation_data, dict)
                                else ""
                            ),
                        )
                        logger.info(
                            "Exception routing enqueued review items for %s: %s",
                            _er_job_id,
                            ", ".join(_er_ids) if _er_ids else "(none)",
                        )
                        # LT-13: hold deliverable until review release
                        doc._hitl_deliverable_hold = True
                        doc._hitl_review_ids = list(_er_ids or [])
                        doc._hitl_routing_decision = _routing_decision
                        doc._hitl_job_id = str(_er_job_id)
                    except Exception as _er_enq_err:
                        # LT-13: fail-closed under production / EXCEPTION_ROUTER_FAIL_CLOSED
                        _er_fail_closed = os.environ.get(
                            "EXCEPTION_ROUTER_FAIL_CLOSED", ""
                        ).strip().lower() in ("1", "true", "yes") or (
                            os.environ.get("DEPLOYMENT_ENV", "").strip().lower()
                            in ("production", "prod")
                        )
                        if _er_fail_closed:
                            logger.error(
                                "Exception routing ReviewQueue enqueue FAILED "
                                "(fail-closed) for %s: %s",
                                doc_id,
                                _er_enq_err,
                            )
                            raise
                        logger.warning(
                            "Exception routing ReviewQueue enqueue skipped for %s: %s",
                            doc_id,
                            _er_enq_err,
                        )
                    # Write routing decision sidecar JSON
                    _er_routing_dir = os.path.join(
                        attempt_output_root, "EXPORT", "ROUTING"
                    )
                    _er_source_type = (
                        "pdf"
                        if os.path.splitext(doc.path)[1].lower() == ".pdf"
                        else "image"
                    )
                    _er_rel_stem = build_output_rel_stem(
                        doc.path,
                        _er_source_type,
                    )
                    _er_target_dir = resolve_safe_output_subfolder(
                        _er_routing_dir,
                        os.path.dirname(_er_rel_stem),
                    )
                    os.makedirs(_er_target_dir, exist_ok=True)
                    _er_routing_path = os.path.join(
                        _er_target_dir,
                        os.path.basename(_er_rel_stem) + ".exception-routing.json",
                    )
                    write_json_atomic(
                        _er_routing_path,
                        {
                            "document_id": doc_id,
                            "should_route": True,
                            "triggered_rules": _routing_decision.triggered_rules,
                            "reasons": _routing_decision.reasons,
                            "confidence": _routing_decision.confidence,
                            "metadata": _routing_decision.metadata,
                        },
                        allowed_root=_er_routing_dir,
                        ensure_ascii=False,
                    )
                    logger.info("Exception routing sidecar: %s", _er_routing_path)
                else:
                    logger.debug("Exception routing: %s passed all rules", doc_id)
            except Exception as _er_err:
                raise RuntimeError(
                    f"enabled exception routing failed for {doc_id}"
                ) from _er_err

        # Commit the isolated output transaction only after the PDF and every
        # enabled finalization surface have finished.  The canonical PDF is
        # installed last by _publish_finalization_stage; any exception before
        # this point leaves prior canonical outputs untouched and queues
        # nothing for compression.
        custody_chain = getattr(doc, "custody_chain", None)

        def _commit_published_assembly():
            _raise_on_finalization_barrier("post-publication commit")
            if not os.path.isfile(doc.output_pdf):
                raise RuntimeError("finalization publish did not install canonical PDF")
            assembled_sha256 = _file_sha256(doc.output_pdf)
            _raise_on_finalization_barrier("pre-custody commit")
            if custody_chain is not None:
                custody_chain.append_event(
                    "assembly_complete",
                    {
                        "output_pdf": doc.output_pdf,
                        "output_sha256": assembled_sha256,
                        "total_pages": doc.total_pages,
                        "processing_seconds": round(
                            time.time() - doc.start_time,
                            2,
                        ),
                        "ocr_quality_classification": (
                            finalized_validation.classification
                            if finalized_validation is not None
                            else "not_evaluated"
                        ),
                        "pages_needing_review": [
                            int(page_num)
                            for page_num, page in sorted(
                                page_data_snap.get("validation", {}).items()
                            )
                            if page.get("needs_review")
                        ],
                        "detected_languages": {
                            str(page_num): page.get("detected_language", "")
                            for page_num, page in sorted(
                                page_data_snap.get("validation", {}).items()
                            )
                        },
                        "model_languages": {
                            str(page_num): page.get("model_language", "")
                            for page_num, page in sorted(
                                page_data_snap.get("validation", {}).items()
                            )
                        },
                    },
                )
                if getattr(custody_chain, "integrity_compromised", False):
                    raise RuntimeError(
                        "assembly custody event was not durably persisted"
                    )
            # Capacity was reserved before entering the finalization lock and
            # no other production producer can steal it.  This non-blocking
            # handoff is the last rollback-protected commit operation.
            # LT-13: HITL hold skips compression release; marker gates downloads.
            if getattr(doc, "_hitl_deliverable_hold", False):
                try:
                    from exception_router import write_deliverable_hold_marker

                    write_deliverable_hold_marker(
                        doc.output_pdf,
                        decision=getattr(doc, "_hitl_routing_decision", None),
                        review_ids=list(getattr(doc, "_hitl_review_ids", []) or []),
                        job_id=str(getattr(doc, "_hitl_job_id", "") or ""),
                        document_id=str(doc_id),
                    )
                except Exception as _hold_err:
                    raise RuntimeError(
                        f"LT-13 deliverable hold marker failed for {doc_id}: {_hold_err}"
                    ) from _hold_err
                logger.warning(
                    "LT-13 HITL hold: skipping compression queue for %s",
                    doc.output_pdf,
                )
            else:
                compression_queue.put_nowait((doc.output_pdf, custody_chain))

        with _compression_handoff_reservation_lock:
            _reserve_compression_handoff_capacity()
            with _finalization_handoff_lock:
                _raise_on_finalization_barrier("pre-publication handoff")
                if compression_queue.full():
                    raise queue.Full(
                        "compression queue reservation was lost before publication"
                    )
                if custody_chain is not None and getattr(
                    custody_chain,
                    "integrity_compromised",
                    False,
                ):
                    raise RuntimeError(
                        "custody chain was compromised before assembly publication"
                    )
                _publish_finalization_stage(
                    finalization_stage_root,
                    doc.output_pdf,
                    post_install_check=_commit_published_assembly,
                )
        shutil.rmtree(finalization_stage_root, ignore_errors=True)
        finalization_stage_root = None

        # Cleanup per-document temp directory (maintenance review). Opt-out via KEEP_TEMP_FILES=true
        # for debugging crash-resume or page-level artifacts. Failures are logged but
        # never propagate, so they cannot affect the main pipeline.
        if not KEEP_TEMP_FILES:
            try:
                if doc.temp_dir and os.path.isdir(doc.temp_dir):
                    shutil.rmtree(doc.temp_dir, ignore_errors=False)
            except Exception as _cleanup_err:
                logger.warning(
                    "Temp cleanup failed for %s (%s): %s",
                    doc.doc_id,
                    doc.temp_dir,
                    _cleanup_err,
                )
        logger.info("DONE (Queued for Compression): %s", doc.output_pdf)

        # Increment Metric (thread-safe)
        global global_docs_processed
        with _docs_processed_lock:
            global_docs_processed += 1

        # Adaptive batch feedback
        if _adaptive_batch_sizer is not None:
            try:
                doc_duration = time.time() - doc.start_time
                _adaptive_batch_sizer.record_result(
                    BatchResult(
                        batch_size=doc.total_pages,
                        pages_processed=doc.processed_pages,
                        duration_seconds=doc_duration,
                        success_count=doc.processed_pages,
                        failure_count=doc.total_pages - doc.processed_pages,
                    )
                )
            except Exception as exc:
                logger.debug("Adaptive batch sizer record failed (non-fatal): %s", exc)

        # Fix Memory Leak: Remove from registry
        with doc_registry_lock:
            if doc_id in doc_registry:
                del doc_registry[doc_id]
        _release_checkpoint_artifact_lock(doc_id)

    except Exception as e:
        # Any assembly/finalization failure must fail the pipeline process.
        # Without this signal main() can drain queues, print ALL TASKS COMPLETED,
        # and exit 0 even though no new deliverable was produced.
        _set_fatal_pipeline_error()
        logger.error("Assembly Failed for %s: %s", doc_id, e)
        log_failure(doc.path, 0, f"ASSEMBLY_FAILED: {e}")
        # maintenance review/RES-004: Record specific assembly_failed custody event
        # so the failure is auditable in the chain-of-custody log.
        _asm_custody = doc.custody_chain if hasattr(doc, "custody_chain") else None
        if _asm_custody:
            try:
                _asm_custody.append_event(
                    "assembly_failed",
                    {
                        "error": str(e),
                        "total_pages": getattr(doc, "total_pages", 0),
                        "processed_pages": getattr(doc, "processed_pages", 0),
                    },
                )
            except Exception as exc:
                logger.debug(
                    "Custody write in assembly failure handler failed: %s", exc
                )
        with doc_registry_lock:
            if doc_id in doc_registry:
                del doc_registry[doc_id]
        _release_checkpoint_artifact_lock(doc_id)
    finally:
        try:
            if final_pdf:
                final_pdf.close()
        except Exception as exc:
            logger.debug("Failed to close final_pdf in finally block: %s", exc)
        if finalization_stage_root:
            shutil.rmtree(finalization_stage_root, ignore_errors=True)
        _release_checkpoint_artifact_lock(doc_id)
        if hasattr(doc, "terminal_statuses"):
            doc.terminal_statuses.clear()
        source_authority = getattr(doc, "source_authority", None)
        if source_authority is not None:
            source_authority.close()
            doc.source_authority = None


_ASSEMBLY_NO_CHUNK_STATUSES = frozenset(
    {
        PageStatus.EXTRACT_FAILED.value,
        PageStatus.CRITICAL_FAILED.value,
        PageStatus.SKIPPED.value,
    }
)
_ASSEMBLY_CLOSED_STATUSES = frozenset(status.value for status in PageStatus)
_ASSEMBLY_CORE_FIELDS = (
    "doc_id",
    "page_num",
    "text",
    "status",
    "chunk_path",
    "checkpoint_sha256",
    "resume_metadata",
    "cache_hit",
    "ocr_method",
    "ocr_language",
    "ocr_confidence",
    "structure_data",
    "handwriting_data",
    "signature_data",
    "vertical_text_data",
    "table_fallback_data",
    "language_data",
)


def _freeze_assembly_value(value):
    """Return a stable equality value for duplicate-message comparison."""
    if isinstance(value, enum.Enum):
        return _freeze_assembly_value(value.value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(key), _freeze_assembly_value(item)) for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_assembly_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(repr(_freeze_assembly_value(item)) for item in value))
    if isinstance(value, float):
        return ("float", repr(value))
    if value is None or isinstance(value, (str, int, bool, bytes)):
        return value
    if hasattr(value, "__dict__"):
        return (
            value.__class__.__qualname__,
            _freeze_assembly_value(vars(value)),
        )
    return repr(value)


def _assembly_message_core(msg):
    """Return the complete page/provenance core used for duplicate handling."""
    return tuple(
        (field, _freeze_assembly_value(msg.get(field)))
        for field in _ASSEMBLY_CORE_FIELDS
    )


def _normalize_assembly_status(status):
    value = status.value if isinstance(status, enum.Enum) else str(status)
    if value not in _ASSEMBLY_CLOSED_STATUSES and not value.startswith("Paddle-"):
        raise ValueError(f"unsupported terminal status {value!r}")
    if value == PageStatus.UNKNOWN.value:
        raise ValueError("UNKNOWN is not an admissible terminal status")
    return value


def _admit_assembly_checkpoint(
    doc,
    *,
    page_num,
    status_value,
    msg_text,
    chunk_path,
    checkpoint_sha256,
    is_reuse_message,
    cache_hit,
    resume_metadata,
    language_data,
):
    """Validate one canonical page artifact before accumulator mutation."""
    if chunk_path is None:
        if status_value not in _ASSEMBLY_NO_CHUNK_STATUSES:
            raise ValueError(
                "lossless assembly requires a valid checkpoint for every source page "
                f"(status={status_value})"
            )
        if checkpoint_sha256 is not None or msg_text or is_reuse_message or cache_hit:
            raise ValueError("no-checkpoint failure terminal has incompatible fields")
        if resume_metadata is not None or language_data is not None:
            raise ValueError(
                "no-checkpoint failure terminal carries derived provenance"
            )
        return None

    if not isinstance(chunk_path, str) or not chunk_path:
        raise ValueError("chunk_path must be a non-empty string")
    if not _is_valid_checkpoint_sha256(checkpoint_sha256):
        raise ValueError("missing or invalid checkpoint_sha256 for present chunk")

    expected_path = os.path.abspath(os.path.join(doc.temp_dir, f"{page_num}.pdf"))
    supplied_path = os.path.abspath(chunk_path)
    if os.path.normcase(supplied_path) != os.path.normcase(expected_path):
        raise ValueError("chunk_path is not the canonical page checkpoint path")
    if os.path.islink(expected_path):
        raise ValueError("canonical checkpoint may not be a symbolic link")
    temp_root = os.path.realpath(doc.temp_dir)
    expected_real = os.path.realpath(expected_path)
    try:
        common = os.path.commonpath([temp_root, expected_real])
    except ValueError as exc:
        raise ValueError("canonical checkpoint escapes document temp root") from exc
    if os.path.normcase(common) != os.path.normcase(temp_root):
        raise ValueError("canonical checkpoint escapes document temp root")

    with _get_checkpoint_artifact_lock(doc.doc_id):
        checkpoint_bytes, actual_digest, metadata, checkpoint_text = (
            _coherent_checkpoint_snapshot(
                expected_path,
                document_id=doc.doc_id,
                page_num=page_num,
            )
        )
        del checkpoint_bytes
    if actual_digest != checkpoint_sha256:
        raise ValueError("checkpoint_sha256 does not match canonical checkpoint bytes")
    if checkpoint_text != msg_text:
        raise ValueError("queued text diverges from canonical checkpoint text")
    if not is_reuse_message and metadata.get("status") != status_value:
        raise ValueError("queued status diverges from canonical checkpoint provenance")
    if is_reuse_message and metadata != resume_metadata:
        raise ValueError(
            "queued reuse provenance diverges from canonical checkpoint metadata"
        )
    if cache_hit and metadata.get("status") != status_value:
        raise ValueError("cached status diverges from canonical checkpoint provenance")
    canonical_language_data = _compact_page_language(language_data)
    if canonical_language_data != metadata.get("language_data"):
        raise ValueError(
            "queued language analysis diverges from canonical checkpoint metadata"
        )
    return metadata


def assembler_thread():
    """Tracks completion and merges final PDFs.

    Page accumulation remains single-threaded so that message ordering is
    deterministic.  When all pages for a document arrive, the heavy I/O
    finalization (PDF merge, sidecar writes, compression hand-off) is
    dispatched to a small ThreadPoolExecutor so multiple documents can
    finalize in parallel without blocking incoming assembly messages.
    """
    threading.current_thread().name = "Assembler"
    logger.debug("Assembler thread started")

    _finalize_executor = ThreadPoolExecutor(
        max_workers=NUM_ASSEMBLER_WORKERS,
        thread_name_prefix="Finalizer",
    )

    extracted_texts = {}  # {doc_id: {page_num: text}}
    structure_pages = {}  # {doc_id: {page_num: structure_data}} (Phase 3A)
    validation_pages = {}  # {doc_id: {page_num: dict}} (Phase 5 validation)
    handwriting_pages = {}  # {doc_id: {page_num: PageHandwriting}} (Phase 6B)
    signature_pages = {}  # {doc_id: {page_num: PageSignatureVerification}} (Phase 6D)
    vertical_text_pages = {}  # {doc_id: {page_num: VerticalTextAnalysis}} (Phase 6D)
    table_fallback_pages = {}  # {doc_id: {page_num: PageTableFallbackAnalysis}} (Phase 4D)
    classification_pages = {}  # {doc_id: {page_num: PageClassification}} (Phase 6A)
    language_pages = {}  # {doc_id: {page_num: PageLanguage}} (Plan A -- PR A2)
    page_chunk_digests = {}  # {doc_id: {page_num: checkpoint_sha256}}
    page_no_chunk_failures = {}  # {doc_id: set[page_num] admitted without chunk}
    terminal_message_records = {}  # {doc_id: {page_num: frozen message core}}
    producer_drain_failures = set()

    def _cleanup_doc_dicts(doc_id: str) -> None:
        """Remove all per-document page data from assembler-local dicts.

        Called on every terminal path (success finalization AND failure) so
        that long-running deployments never leak memory for abandoned or
        failed documents.
        """
        extracted_texts.pop(doc_id, None)
        structure_pages.pop(doc_id, None)
        validation_pages.pop(doc_id, None)
        handwriting_pages.pop(doc_id, None)
        signature_pages.pop(doc_id, None)
        vertical_text_pages.pop(doc_id, None)
        table_fallback_pages.pop(doc_id, None)
        classification_pages.pop(doc_id, None)
        language_pages.pop(doc_id, None)
        page_chunk_digests.pop(doc_id, None)
        page_no_chunk_failures.pop(doc_id, None)
        terminal_message_records.pop(doc_id, None)

    def _dispatch_completed_docs() -> None:
        """Seal each proven generation once its own obligations are consumed."""
        global_drain = (
            assembly_producers_done_event.is_set() and assembly_queue.empty()
        )
        if fatal_pipeline_error_event.is_set() or finalization_abort_event.is_set():
            return
        with doc_registry_lock:
            candidates = list(doc_registry.items())
        ready_candidates = []
        for ready_id, ready_doc in candidates:
            if ready_id in producer_drain_failures:
                continue
            if ready_doc.finalized:
                continue
            if ready_doc.total_pages <= 0:
                continue
            fence = getattr(ready_doc, "completion_fence", None)
            fence_ready = fence is not None and fence.ready()
            if not fence_ready and not global_drain:
                continue
            expected_pages = set(range(1, ready_doc.total_pages + 1))
            admitted_pages = set(ready_doc.terminal_pages)
            if admitted_pages != expected_pages:
                missing_pages = sorted(expected_pages - admitted_pages)
                extra_pages = sorted(admitted_pages - expected_pages)
                log_failure(
                    ready_doc.path,
                    0,
                    "FINALIZATION_PAGE_SET_INCOMPLETE_AFTER_PRODUCER_DRAIN: "
                    f"missing={missing_pages}; extra={extra_pages}",
                )
                producer_drain_failures.add(ready_id)
                _cleanup_doc_dicts(ready_id)
                continue
            no_chunk = set(page_no_chunk_failures.get(ready_id, set()))
            if no_chunk:
                log_failure(
                    ready_doc.path,
                    0,
                    "DOCUMENT_CHECKPOINTS_UNAVAILABLE: "
                    f"pages={sorted(no_chunk)}; source retained",
                )
                producer_drain_failures.add(ready_id)
                _cleanup_doc_dicts(ready_id)
                continue
            ready_candidates.append((ready_id, ready_doc, fence_ready))

        # Dispatch each fully proven document independently.  A damaged sibling
        # remains unpublished and raises the pipeline fatal signal after the
        # healthy finalizers finish; it must not destroy already-proven output.
        for ready_id, ready_doc, fence_ready in ready_candidates:
            if fatal_pipeline_error_event.is_set() or finalization_abort_event.is_set():
                return
            if fence_ready and not ready_doc.completion_fence.seal():
                continue
            ready_doc.finalized = True
            logger.info(
                "FINALIZING (document completion proven): %s",
                os.path.basename(ready_doc.path),
            )
            ready_doc.sealed_terminal_records = dict(
                terminal_message_records.get(ready_id, {})
            )
            snapshot = {
                "texts": dict(extracted_texts.get(ready_id, {})),
                "structure": dict(structure_pages.get(ready_id, {})),
                "validation": dict(validation_pages.get(ready_id, {})),
                "handwriting": dict(handwriting_pages.get(ready_id, {})),
                "signature": dict(signature_pages.get(ready_id, {})),
                "vertical_text": dict(vertical_text_pages.get(ready_id, {})),
                "table_fallback": dict(table_fallback_pages.get(ready_id, {})),
                "classification": dict(classification_pages.get(ready_id, {})),
                "language": dict(language_pages.get(ready_id, {})),
                "chunk_digests": dict(page_chunk_digests.get(ready_id, {})),
                "no_chunk_failures": set(page_no_chunk_failures.get(ready_id, set())),
            }
            _cleanup_doc_dicts(ready_id)
            _finalize_executor.submit(
                _finalize_doc,
                ready_doc,
                ready_id,
                snapshot,
            )

    def _finish_assembly_message(reservation) -> None:
        assembly_queue.task_done()
        _complete_queue_obligation(reservation)
        _dispatch_completed_docs()

    try:
        while True:
            try:
                msg: AssemblyMessage = assembly_queue.get(timeout=2)
                assembly_reservation = _take_queue_obligation(msg)

                # Process Message
                doc_id = msg["doc_id"]
                with doc_registry_lock:
                    doc = doc_registry.get(doc_id)
                if not doc:
                    logger.error("ASSEMBLER DROP: ID %s not found", doc_id)
                    _cleanup_doc_dicts(doc_id)  # maintenance review: purge orphaned page data
                    _finish_assembly_message(assembly_reservation)
                    continue
                if doc.finalized:
                    page_num = msg.get("page_num")
                    frozen_message = _assembly_message_core(msg)
                    previous = getattr(doc, "sealed_terminal_records", {}).get(
                        page_num
                    )
                    if previous != frozen_message:
                        _set_fatal_pipeline_error()
                        log_failure(
                            doc.path,
                            page_num if isinstance(page_num, int) else 0,
                            "POST_SEAL_CONFLICTING_TERMINAL_MESSAGE",
                        )
                    else:
                        logger.info(
                            "Exact post-seal duplicate ignored for %s p%s",
                            doc_id,
                            page_num,
                        )
                    _finish_assembly_message(assembly_reservation)
                    continue

                page_num = msg.get("page_num")
                status = msg.get("status", PageStatus.UNKNOWN)
                msg_text = msg.get("text")
                chunk_path = msg.get("chunk_path")
                msg_checkpoint_sha256 = msg.get("checkpoint_sha256")
                admission_label = "CHECKPOINT_ADMISSION_FAILED"
                try:
                    if (
                        isinstance(page_num, bool)
                        or not isinstance(page_num, int)
                        or page_num < 1
                        or page_num > doc.total_pages
                    ):
                        raise ValueError(
                            "page_num must be a non-bool int within document range"
                        )
                    if not isinstance(msg_text, str):
                        raise ValueError("text must be a string")
                    status_value = _normalize_assembly_status(status)
                    cache_hit = msg.get("cache_hit", False)
                    if not isinstance(cache_hit, bool):
                        raise ValueError("cache_hit must be a bool")
                    is_reuse_message = (
                        status_value == PageStatus.RESUMED.value or cache_hit
                    )
                    resume_meta = msg.get("resume_metadata")
                    if status_value == PageStatus.RESUMED.value and cache_hit:
                        raise ValueError("RESUMED and cache_hit are mutually exclusive")
                    if is_reuse_message:
                        admission_label = "REUSE_ADMISSION_FAILED"
                        if not isinstance(resume_meta, dict):
                            raise ValueError("missing resume_metadata")
                        _validated_checkpoint_text_value(msg_text, resume_meta)
                    elif resume_meta is not None:
                        raise ValueError(
                            "resume_metadata is only valid for reused checkpoints"
                        )

                    # Duplicate handling precedes artifact validation and every
                    # accumulator/state mutation.  First admitted message wins.
                    frozen_message = _assembly_message_core(msg)
                    prior_message = terminal_message_records.get(doc_id, {}).get(
                        page_num
                    )
                    if prior_message is not None:
                        if prior_message == frozen_message:
                            logger.info(
                                "Exact duplicate terminal message ignored for %s p%d",
                                doc_id,
                                page_num,
                            )
                        else:
                            _set_fatal_pipeline_error()
                            logger.critical(
                                "ASSEMBLER CONFLICTING_DUPLICATE for %s p%d",
                                doc_id,
                                page_num,
                            )
                            log_failure(
                                doc.path,
                                page_num,
                                "CONFLICTING_DUPLICATE_TERMINAL_MESSAGE",
                            )
                        _finish_assembly_message(assembly_reservation)
                        continue

                    admitted_metadata = _admit_assembly_checkpoint(
                        doc,
                        page_num=page_num,
                        status_value=status_value,
                        msg_text=msg_text,
                        chunk_path=chunk_path,
                        checkpoint_sha256=msg_checkpoint_sha256,
                        is_reuse_message=is_reuse_message,
                        cache_hit=cache_hit,
                        resume_metadata=resume_meta,
                        language_data=msg.get("language_data"),
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    _set_fatal_pipeline_error()
                    logger.critical(
                        "ASSEMBLER %s for %s p%s: %s",
                        admission_label,
                        doc_id,
                        page_num,
                        exc,
                    )
                    log_failure(
                        doc.path,
                        page_num,
                        f"{admission_label}: {exc}",
                    )
                    _finish_assembly_message(assembly_reservation)
                    continue

                # The message is now fully admitted.  Record it before any
                # accumulator mutation so later duplicates cannot mix fields.
                terminal_message_records.setdefault(doc_id, {})[page_num] = (
                    frozen_message
                )

                # Init storage only after admission/duplicate handling.
                extracted_texts.setdefault(doc_id, {})
                structure_pages.setdefault(doc_id, {})
                validation_pages.setdefault(doc_id, {})
                handwriting_pages.setdefault(doc_id, {})
                signature_pages.setdefault(doc_id, {})
                vertical_text_pages.setdefault(doc_id, {})
                table_fallback_pages.setdefault(doc_id, {})
                classification_pages.setdefault(doc_id, {})
                language_pages.setdefault(doc_id, {})

                # Admit per-page chunk digest only after format/provenance checks.
                if chunk_path is not None:
                    page_chunk_digests.setdefault(doc_id, {})[page_num] = (
                        msg_checkpoint_sha256
                    )
                else:
                    page_no_chunk_failures.setdefault(doc_id, set()).add(page_num)

                # Collect Document Intelligence data (Phase 3A)
                msg_structure = msg.get("structure_data")
                if msg_structure and page_num not in structure_pages[doc_id]:
                    structure_pages[doc_id][page_num] = msg_structure

                # Collect handwriting detection data (Phase 6B)
                msg_hw = msg.get("handwriting_data")
                if msg_hw and page_num not in handwriting_pages[doc_id]:
                    handwriting_pages[doc_id][page_num] = msg_hw

                # Collect signature verification data (Phase 6D)
                msg_sig = msg.get("signature_data")
                if msg_sig and page_num not in signature_pages[doc_id]:
                    signature_pages[doc_id][page_num] = msg_sig

                # Collect vertical text analysis data (Phase 6D)
                msg_vt = msg.get("vertical_text_data")
                if msg_vt and page_num not in vertical_text_pages[doc_id]:
                    vertical_text_pages[doc_id][page_num] = msg_vt

                # Collect table fallback analysis data (Phase 4D)
                msg_tf = msg.get("table_fallback_data")
                if msg_tf and page_num not in table_fallback_pages[doc_id]:
                    table_fallback_pages[doc_id][page_num] = msg_tf

                # Collect per-page language detection data (Plan A -- PR A2)
                msg_lang = msg.get("language_data")
                if msg_lang is not None and page_num not in language_pages[doc_id]:
                    language_pages[doc_id][page_num] = msg_lang

                # Collect validation data (Phase 5)
                if (
                    ENABLE_VALIDATION
                    and _VALIDATION_AVAILABLE
                    and page_num not in validation_pages[doc_id]
                ):
                    # Exact embedded checkpoint metadata is the only resumed/cache
                    # provenance authority. Never inherit confidence/language/status
                    # from a prior validation sidecar.
                    checkpoint_metadata = None
                    candidate_metadata = admitted_metadata
                    if isinstance(candidate_metadata, dict):
                        checkpoint_metadata = candidate_metadata

                    if checkpoint_metadata is not None:
                        val_status = checkpoint_metadata["validation_status"]
                        ocr_method = checkpoint_metadata["ocr_method"]
                        ocr_language = checkpoint_metadata["ocr_language"]
                        ocr_confidence = float(checkpoint_metadata["ocr_confidence"])
                        text_length = int(checkpoint_metadata["text_length"])
                        has_text = bool(checkpoint_metadata["has_text"])
                        detected_language = checkpoint_metadata.get(
                            "detected_language", ""
                        )
                        detected_language_confidence = checkpoint_metadata.get(
                            "detected_language_confidence"
                        )
                        source_prior_label = checkpoint_metadata.get("source_prior_label", "")
                        model_language = checkpoint_metadata.get("model_language", "")
                        realized_profile = checkpoint_metadata.get("processing_profile", {})
                        runtime_model = realized_profile.get("runtime_model", "")
                        execution_device = realized_profile.get("execution_device", "")
                        configured_backend = realized_profile.get("inference_backend", "")
                        inference_backend = (
                            "tesseract" if ocr_method == "Tesseract"
                            else configured_backend if ocr_method == "PaddleOCR"
                            else "none"
                        )
                        ocr_confidence_basis = realized_profile.get("ocr_confidence_basis", "")
                        geometry_issues = checkpoint_metadata.get("geometry_issues", [])
                        review_reasons = checkpoint_metadata.get("review_reasons", [])
                        candidate_reviews = checkpoint_metadata.get("candidate_reviews", [])
                        independent_coverage = checkpoint_metadata.get(
                            "coverage_assessment", {}
                        )
                        needs_review = checkpoint_metadata.get("needs_review", False)
                    else:
                        if status in (
                            PageStatus.OK,
                            PageStatus.PADDLE,
                            PageStatus.TESSERACT,
                            PageStatus.EASYOCR,
                        ) or status.startswith("Paddle-"):
                            val_status = (
                                "ok"
                                if status
                                not in (PageStatus.TESSERACT, PageStatus.EASYOCR)
                                else "fallback"
                            )
                        elif status == PageStatus.CACHED:
                            val_status = "ok"
                        elif status == PageStatus.IMAGE_ONLY:
                            val_status = "image_only"
                        elif status in (
                            PageStatus.EXTRACT_FAILED,
                            PageStatus.CRITICAL_FAILED,
                        ):
                            val_status = "failed"
                        elif status == PageStatus.RESUMED:
                            val_status = "ok" if msg_text.strip() else "unknown"
                        else:
                            val_status = "unknown"
                        ocr_method = msg.get(
                            "ocr_method",
                            PageStatus.RESUMED if status == PageStatus.RESUMED else "",
                        )
                        ocr_language = str(msg.get("ocr_language", ""))
                        ocr_confidence = float(msg.get("ocr_confidence", 0.0) or 0.0)
                        text_length = len(msg_text)
                        has_text = bool(msg_text.strip())
                        detected_language = ""
                        detected_language_confidence = None
                        source_prior_label = ""
                        model_language = ""
                        runtime_model = ""
                        execution_device = ""
                        inference_backend = ""
                        ocr_confidence_basis = ""
                        geometry_issues = []
                        review_reasons = [
                            "no validated searchable page checkpoint"
                        ]
                        candidate_reviews = []
                        independent_coverage = {}
                        needs_review = True
                    validation_pages[doc_id][page_num] = {
                        "page_num": page_num,
                        "ocr_method": ocr_method,
                        "ocr_language": ocr_language,
                        "ocr_confidence": ocr_confidence,
                        "text_length": text_length,
                        "has_text": has_text,
                        "status": val_status,
                        "detected_language": detected_language,
                        "detected_language_confidence": detected_language_confidence,
                        "source_prior_label": source_prior_label,
                        "model_language": model_language,
                        "runtime_model": runtime_model,
                        "execution_device": execution_device,
                        "inference_backend": inference_backend,
                        "ocr_confidence_basis": ocr_confidence_basis,
                        "needs_review": needs_review,
                        "review_reasons": review_reasons,
                        "geometry_issues": geometry_issues,
                        "candidate_reviews": candidate_reviews,
                        "independent_coverage": independent_coverage,
                    }

                if status in (PageStatus.EXTRACT_FAILED, PageStatus.CRITICAL_FAILED):
                    logger.error(
                        "ASSEMBLER TERMINAL PAGE %s p%d: %s", doc_id, page_num, status
                    )
                elif status == PageStatus.RESUMED:
                    logger.info(
                        "ASSEMBLER TERMINAL PAGE %s p%d: RESUMED", doc_id, page_num
                    )

                extracted_texts[doc_id][page_num] = msg_text
                doc.terminal_statuses[page_num] = status_value
                doc.terminal_pages.add(page_num)

                doc.processed_pages = len(doc.terminal_pages)

                # Page-count completion is not producer completion.  Keep the
                # first-wins records and accumulators until main proves every
                # scheduler/extractor/worker producer is done and this queue is
                # empty, so a queued conflicting duplicate can still veto.
                if (
                    not doc.finalized
                    and doc.total_pages > 0
                    and doc.processed_pages == doc.total_pages
                ):
                    logger.debug(
                        "FINALIZATION READY (awaiting producer drain): %s",
                        os.path.basename(doc.path),
                    )

                _finish_assembly_message(assembly_reservation)

            except queue.Empty:
                _dispatch_completed_docs()
                if assembly_producers_done_event.is_set():
                    # The event proves no producer can enqueue after this
                    # point.  The pause remains a defensive queue visibility
                    # check, not the source of the lifecycle guarantee.
                    time.sleep(0.05)
                    if not assembly_queue.empty():
                        continue
                    _dispatch_completed_docs()
                    break
                if finalization_abort_event.is_set() and assembly_queue.empty():
                    break
                continue
    finally:
        _finalize_executor.shutdown(wait=True)
        if producer_drain_failures:
            _set_fatal_pipeline_error()
        # maintenance review: purge any remaining per-doc data on shutdown to avoid leaks
        # if the assembler exits before all documents finalize.
        for stale_id in list(extracted_texts.keys()):
            _cleanup_doc_dicts(stale_id)
        with doc_registry_lock:
            remaining_docs = list(doc_registry.values())
        for remaining_doc in remaining_docs:
            if getattr(remaining_doc, "finalized", False):
                continue
            source_authority = getattr(remaining_doc, "source_authority", None)
            if source_authority is not None:
                source_authority.close()
                remaining_doc.source_authority = None


def monitor_thread():
    """Logs queue status and speed metrics periodically."""
    threading.current_thread().name = "Monitor"

    last_pages = 0
    last_time = time.time()

    while not stop_event.is_set():
        time.sleep(MONITOR_SLEEP_SECONDS)  # configurable via MONITOR_SLEEP_SECONDS

        # Calculate Speed -- snapshot counters under their locks
        now = time.time()
        delta_t = now - last_time
        with _pages_processed_lock:
            current_pages = global_pages_processed
        delta_p = current_pages - last_pages

        ppm_inst = (delta_p / delta_t) * 60 if delta_t > 0 else 0

        total_time = now - start_time_global
        ppm_avg = (current_pages / total_time) * 60 if total_time > 0 else 0
        dph_avg = (global_docs_processed / total_time) * 3600 if total_time > 0 else 0

        # Update trackers
        last_pages = current_pages
        last_time = now

        # Safe logging of progress — snapshot registry to avoid holding lock during I/O
        with doc_registry_lock:
            registry_snapshot = dict(doc_registry)
        active_status = []
        for d_id, doc in list(registry_snapshot.items())[
            :3
        ]:  # Limit to first 3 to avoid log spam
            active_status.append(
                f"[{os.path.basename(doc.path)}: {doc.processed_pages}/{doc.total_pages}]"
            )

        status_str = " | ".join(active_status)
        registry_size = len(registry_snapshot)
        logger.info(
            "STATS: Q_Img=%d Q_Asm=%d Registry=%d | Speed: %.1f PPM (Inst) / %.1f PPM (Avg) | Rate: %.1f Docs/Hr | %s",
            image_queue.qsize(),
            assembly_queue.qsize(),
            registry_size,
            ppm_inst,
            ppm_avg,
            dph_avg,
            status_str,
        )

        # Page cache stats (opt-in)
        if _page_cache is not None:
            cache_stats = _page_cache.get_stats()
            logger.info(
                "Cache: hits=%d misses=%d evictions=%d hit_rate=%.1f%%",
                cache_stats.hits,
                cache_stats.misses,
                cache_stats.evictions,
                cache_stats.hit_rate * 100,
            )

        # Page routing stats (opt-in)
        if _page_router is not None:
            routing_stats = _page_router.get_routing_stats()
            if routing_stats:
                logger.info("Routing: %s", routing_stats)

        # Write heartbeat for Docker HEALTHCHECK
        try:
            with open(HEALTHCHECK_FILE, "w") as hf:
                hf.write(f"{now:.0f}\n")
        except (IOError, OSError) as e:
            logger.warning("Failed to write healthcheck heartbeat file: %s", e)


def _stage_atomic_bytes(path, payload):
    """Fsync *payload* to a temporary file beside *path* and return it."""
    parent = os.path.dirname(path)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=parent,
    )
    try:
        if hasattr(os, "fchmod"):
            try:
                target_mode = stat.S_IMODE(os.lstat(path).st_mode)
            except FileNotFoundError:
                # mkstemp defaults to 0600, which would unexpectedly make new
                # sidecars/checkpoints unreadable to shared-volume consumers.
                target_mode = 0o644
            os.fchmod(fd, target_mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def _fsync_directory(path):
    """Persist directory-entry changes where the platform exposes that API."""
    if os.name == "nt":
        # Windows does not allow opening directory handles with os.open().
        # os.replace still provides atomic visibility; file contents are
        # flushed before replacement.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path, payload):
    """Atomically replace *path* with byte *payload*."""
    temp_path = _stage_atomic_bytes(path, payload)
    try:
        os.replace(temp_path, path)
        _fsync_directory(os.path.dirname(path))
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def _atomic_write_json(path, payload):
    """Atomically replace a JSON sidecar in its existing directory."""
    encoded = (
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_FINAL_HASH_TRANSACTION_SCHEMA = 1
_FINAL_HASH_TRANSACTION_DIRNAME = ".final-hash-transactions"
_FINAL_HASH_TRANSACTION_LOCKS = {}
_FINAL_HASH_TRANSACTION_LOCKS_GUARD = threading.Lock()


def _path_within_root(path, root, label):
    """Resolve *path* and fail closed unless it is contained by *root*."""
    root_real = os.path.realpath(root)
    path_real = os.path.realpath(path)
    try:
        common_root = os.path.commonpath([root_real, path_real])
    except ValueError as exc:
        raise ValueError(f"{label} is outside its allowed root: {path}") from exc
    if os.path.normcase(common_root) != os.path.normcase(root_real):
        raise ValueError(f"{label} escapes its allowed root: {path}")
    return path_real


def _canonical_final_output_identity(pdf_path):
    """Return ``(identity_key, pdf_real)`` for locks and transaction dirs.

    The identity is the case-normalized, slash-normalized path of the PDF
    relative to ``EXPORT/PDF``.  The same key feeds the in-process lock map
    and the OS lock filename so Windows path/case aliases cannot race.
    """
    pdf_root = os.path.realpath(os.path.join(OUTPUT_FOLDER, "EXPORT", "PDF"))
    pdf_real = os.path.realpath(pdf_path)
    try:
        common_root = os.path.commonpath([pdf_root, pdf_real])
    except ValueError as exc:
        raise ValueError(f"PDF path is outside output volume: {pdf_path}") from exc
    if os.path.normcase(common_root) != os.path.normcase(pdf_root):
        raise ValueError(f"PDF path escapes output root: {pdf_path}")
    relative_pdf = os.path.relpath(pdf_real, pdf_root)
    identity = os.path.normcase(relative_pdf.replace("\\", "/"))
    return identity, pdf_real


def _final_hash_transaction_location(pdf_path):
    """Return a deterministic transaction directory for one output PDF."""
    identity, _pdf_real = _canonical_final_output_identity(pdf_path)
    transaction_root = os.path.realpath(
        os.path.join(OUTPUT_FOLDER, "EXPORT", _FINAL_HASH_TRANSACTION_DIRNAME)
    )
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return transaction_root, os.path.join(transaction_root, key)


def _cleanup_final_hash_transaction(transaction_root, transaction_dir):
    """Remove only the verified per-PDF transaction directory."""
    transaction_dir = _path_within_root(
        transaction_dir,
        transaction_root,
        "final-hash transaction directory",
    )
    if os.path.isdir(transaction_dir):
        shutil.rmtree(transaction_dir)
        _fsync_directory(transaction_root)


def _read_transaction_blob(transaction_dir, blob_name, expected_hash):
    if not isinstance(blob_name, str) or os.path.basename(blob_name) != blob_name:
        raise ValueError("invalid final-hash transaction blob name")
    blob_path = _path_within_root(
        os.path.join(transaction_dir, blob_name),
        transaction_dir,
        "final-hash transaction blob",
    )
    with open(blob_path, "rb") as handle:
        payload = handle.read()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"final-hash transaction blob digest mismatch: {blob_name}")
    return payload


def _load_final_hash_transaction(journal_path):
    with open(journal_path, encoding="utf-8") as handle:
        journal = json.load(handle)
    if not isinstance(journal, dict):
        raise TypeError("final-hash transaction journal must be an object")
    if journal.get("schema_version") != _FINAL_HASH_TRANSACTION_SCHEMA:
        raise ValueError("unsupported final-hash transaction journal schema")
    targets = journal.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("final-hash transaction journal has no targets")
    return journal


def _apply_final_hash_transaction(journal, transaction_dir, *, use_updated):
    """Apply either all updated or all original sidecar bytes from a journal."""
    export_root = os.path.join(OUTPUT_FOLDER, "EXPORT")
    blob_key = "updated_blob" if use_updated else "original_blob"
    digest_key = "updated_sha256" if use_updated else "original_sha256"
    expected_targets = []
    for item in journal["targets"]:
        if not isinstance(item, dict):
            raise TypeError("final-hash transaction target must be an object")
        target = _path_within_root(
            item.get("path", ""),
            export_root,
            "final-hash sidecar",
        )
        payload = _read_transaction_blob(
            transaction_dir,
            item.get(blob_key),
            item.get(digest_key),
        )
        _atomic_write_bytes(target, payload)
        expected_targets.append((target, item.get(digest_key)))

    for target, expected_hash in expected_targets:
        if _file_sha256(target) != expected_hash:
            raise RuntimeError(
                f"final-hash transaction verification failed for sidecar: {target}"
            )
    return [target for target, _ in expected_targets]


def _recover_final_hash_transaction(pdf_real):
    """Finish or roll back an interrupted multi-sidecar hash transaction."""
    transaction_root, transaction_dir = _final_hash_transaction_location(pdf_real)
    journal_path = os.path.join(transaction_dir, "transaction.json")
    if not os.path.isdir(transaction_dir):
        return None
    if not os.path.isfile(journal_path):
        # Target replacement starts only after the journal is durable, so a
        # journal-less directory contains preparation debris only.
        _cleanup_final_hash_transaction(transaction_root, transaction_dir)
        return None

    journal = _load_final_hash_transaction(journal_path)
    journal_pdf = os.path.realpath(journal.get("pdf_path", ""))
    if os.path.normcase(journal_pdf) != os.path.normcase(pdf_real):
        raise ValueError("final-hash transaction journal names a different PDF")

    export_root = os.path.join(OUTPUT_FOLDER, "EXPORT")
    target_states = []
    for item in journal["targets"]:
        target = _path_within_root(
            item.get("path", ""),
            export_root,
            "final-hash sidecar",
        )
        if not os.path.isfile(target):
            target_states.append("missing")
            continue
        current_target_hash = _file_sha256(target)
        if current_target_hash == item.get("original_sha256"):
            target_states.append("original")
        elif current_target_hash == item.get("updated_sha256"):
            target_states.append("updated")
        else:
            target_states.append("newer")

    newer_count = target_states.count("newer")
    if newer_count == len(target_states):
        # A complete later assembly replaced every sidecar.  The old journal
        # must not overwrite that newer generation.
        _cleanup_final_hash_transaction(transaction_root, transaction_dir)
        return None
    if newer_count:
        raise RuntimeError(
            "interrupted final-hash transaction overlaps a partial newer "
            f"sidecar generation: {target_states}"
        )

    current_hash = _file_sha256(pdf_real)
    use_updated = current_hash == journal.get("pdf_sha256")
    targets = _apply_final_hash_transaction(
        journal,
        transaction_dir,
        use_updated=use_updated,
    )
    _cleanup_final_hash_transaction(transaction_root, transaction_dir)
    if use_updated:
        return current_hash, targets
    return None


def _get_final_hash_transaction_lock(pdf_path):
    identity, _pdf_real = _canonical_final_output_identity(pdf_path)
    with _FINAL_HASH_TRANSACTION_LOCKS_GUARD:
        return _FINAL_HASH_TRANSACTION_LOCKS.setdefault(identity, threading.Lock())


def _lock_final_hash_file_strict(lock_file):
    """Acquire a blocking inter-process lock without a fail-open fallback."""
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        return
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - unsupported production OS
        raise RuntimeError(
            "Final-hash transaction locking is unsupported on this platform"
        ) from exc
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)


def _unlock_final_hash_file_strict(lock_file):
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        return
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - unsupported production OS
        raise RuntimeError(
            "Final-hash transaction unlocking is unsupported on this platform"
        ) from exc
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _locked_final_hash_transaction(pdf_path):
    """Serialize one PDF's transaction across threads and OS processes."""
    with _get_final_hash_transaction_lock(pdf_path):
        transaction_root, transaction_dir = _final_hash_transaction_location(pdf_path)
        os.makedirs(transaction_root, exist_ok=True)
        lock_path = _path_within_root(
            transaction_dir + ".lock",
            transaction_root,
            "final-hash transaction lock",
        )
        with open(lock_path, "a+b") as lock_file:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
                os.fsync(lock_file.fileno())
            _lock_final_hash_file_strict(lock_file)
            try:
                yield
            finally:
                _unlock_final_hash_file_strict(lock_file)


def _refresh_final_output_hashes(pdf_path):
    """Serialize and recover the final PDF-to-sidecar hash binding."""
    with _locked_final_hash_transaction(pdf_path):
        return _refresh_final_output_hashes_locked(pdf_path)


def _compress_and_refresh_final_output(pdf_path):
    """Compress a final PDF and refresh hashes under one transaction lock.

    Ghostscript replacement and post-compression hash/journal refresh must not
    interleave with another producer for the same canonical output identity.
    Callers must not nest this with :func:`_refresh_final_output_hashes` — both
    acquire the same non-reentrant lock.  Any pending journal is recovered under
    the held lock before optimize begins.

    Returns a dict::

        {
            "disposition": "APPLIED" | "SKIPPED_VISUAL_BUDGET" |
                           "SKIPPED_CANDIDATE_REJECTED",
            "final_hash": str,
            "updated_sidecars": list[str],
            "reason": str,
        }

    Both ``SKIPPED_*`` outcomes finalize exact unchanged assembled bytes
    (digest equality verified) and refresh sidecar hashes against the original
    PDF. Any ``FAILED_*`` optimizer result, hash divergence, or refresh failure
    raises (caller treats as fatal).
    """
    with _locked_final_hash_transaction(pdf_path):
        pdf_real = os.path.realpath(pdf_path)
        # Finish or discard an interrupted multi-sidecar transaction before
        # mutating the PDF.  Do not nest the non-reentrant public refresh helper.
        _recover_final_hash_transaction(pdf_real)
        before_hash = _file_sha256(pdf_real)
        result = optimize_pdf_detailed(pdf_path, quality="/prepress")
        disposition = result.disposition

        safe_skip_dispositions = {
            OptimizationDisposition.SKIPPED_VISUAL_BUDGET,
            OptimizationDisposition.SKIPPED_CANDIDATE_REJECTED,
        }
        if disposition in safe_skip_dispositions:
            after_hash = _file_sha256(pdf_real)
            if after_hash != before_hash:
                raise RuntimeError(
                    "optimizer safe skip mutated PDF bytes "
                    f"(before={before_hash}, after={after_hash})"
                )
            final_hash, updated_sidecars = _refresh_final_output_hashes_locked(pdf_path)
            if final_hash != before_hash:
                raise RuntimeError(
                    "optimizer safe skip final hash diverged from assembled bytes "
                    f"(before={before_hash}, final={final_hash})"
                )
            return {
                "disposition": disposition.value,
                "final_hash": final_hash,
                "updated_sidecars": updated_sidecars,
                "reason": result.reason,
            }

        if disposition is not OptimizationDisposition.APPLIED:
            # Never relabel integrity/GS/durability failures as a safe skip.
            raise RuntimeError(
                f"PDF optimization failed ({disposition.value}): {result.reason}"
            )

        final_hash, updated_sidecars = _refresh_final_output_hashes_locked(pdf_path)
        return {
            "disposition": OptimizationDisposition.APPLIED.value,
            "final_hash": final_hash,
            "updated_sidecars": updated_sidecars,
            "reason": result.reason,
        }


def _refresh_final_output_hashes_locked(pdf_path):
    """Bind validation and retrieval metadata to the final compressed PDF.

    Validation and retrieval are assembled before the asynchronous Ghostscript
    stage.  Because compression replaces the PDF, their original hash becomes
    stale.  Refresh every enabled sidecar only after compression completes and
    return the authoritative final digest plus updated paths.
    """
    pdf_root = os.path.realpath(os.path.join(OUTPUT_FOLDER, "EXPORT", "PDF"))
    pdf_real = os.path.realpath(pdf_path)
    try:
        common_root = os.path.commonpath([pdf_root, pdf_real])
    except ValueError as exc:
        raise ValueError(f"PDF path is outside output volume: {pdf_path}") from exc
    if os.path.normcase(common_root) != os.path.normcase(pdf_root):
        raise ValueError(f"PDF path escapes output root: {pdf_path}")
    recovered = _recover_final_hash_transaction(pdf_real)
    if recovered is not None:
        return recovered
    relative_pdf = os.path.relpath(pdf_real, pdf_root)

    relative_stem = os.path.splitext(relative_pdf)[0]
    final_hash = _file_sha256(pdf_real)
    targets = []
    if ENABLE_VALIDATION and _VALIDATION_AVAILABLE:
        targets.append(
            (
                os.path.join(
                    OUTPUT_FOLDER,
                    "EXPORT",
                    "VALIDATION",
                    relative_stem + ".validation.json",
                ),
                "validation",
            )
        )
    if ENABLE_RETRIEVAL_OUTPUT and _OUTPUT_ASSEMBLER_AVAILABLE:
        targets.append(
            (
                os.path.join(
                    OUTPUT_FOLDER,
                    "EXPORT",
                    "RETRIEVAL",
                    relative_stem + ".retrieval.json",
                ),
                "retrieval",
            )
        )

    prepared = []
    for sidecar_path, sidecar_type in targets:
        if not os.path.isfile(sidecar_path):
            raise FileNotFoundError(
                f"enabled {sidecar_type} sidecar missing after finalization: "
                f"{sidecar_path}"
            )
        with open(sidecar_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if sidecar_type == "validation":
            payload["output_hash"] = final_hash
        else:
            metadata = payload.setdefault("metadata", {})
            if not isinstance(metadata, dict):
                raise TypeError("retrieval metadata must be an object")
            metadata["output_hash"] = final_hash

        updated_bytes = (
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
        ).encode("utf-8")
        with open(sidecar_path, "rb") as handle:
            original_bytes = handle.read()
        prepared.append((sidecar_path, original_bytes, updated_bytes))

    if not prepared:
        return final_hash, []

    transaction_root, transaction_dir = _final_hash_transaction_location(pdf_real)
    os.makedirs(transaction_root, exist_ok=True)
    os.makedirs(transaction_dir, exist_ok=False)
    _fsync_directory(transaction_root)
    journal_targets = []
    try:
        for index, (sidecar_path, original_bytes, updated_bytes) in enumerate(prepared):
            original_blob = f"{index}.original"
            updated_blob = f"{index}.updated"
            _atomic_write_bytes(
                os.path.join(transaction_dir, original_blob),
                original_bytes,
            )
            _atomic_write_bytes(
                os.path.join(transaction_dir, updated_blob),
                updated_bytes,
            )
            journal_targets.append(
                {
                    "path": os.path.realpath(sidecar_path),
                    "original_blob": original_blob,
                    "original_sha256": hashlib.sha256(original_bytes).hexdigest(),
                    "updated_blob": updated_blob,
                    "updated_sha256": hashlib.sha256(updated_bytes).hexdigest(),
                }
            )
        journal = {
            "schema_version": _FINAL_HASH_TRANSACTION_SCHEMA,
            "pdf_path": pdf_real,
            "pdf_sha256": final_hash,
            "targets": journal_targets,
        }
        _atomic_write_json(
            os.path.join(transaction_dir, "transaction.json"),
            journal,
        )
    except Exception:
        _cleanup_final_hash_transaction(transaction_root, transaction_dir)
        raise

    try:
        _apply_final_hash_transaction(journal, transaction_dir, use_updated=True)
        for sidecar_path, _, _ in prepared:
            with open(sidecar_path, encoding="utf-8") as handle:
                committed = json.load(handle)
            committed_hash = (
                committed.get("output_hash")
                if "output_hash" in committed
                else committed.get("metadata", {}).get("output_hash")
            )
            if committed_hash != final_hash:
                raise RuntimeError(
                    f"final hash verification failed for sidecar: {sidecar_path}"
                )
        _cleanup_final_hash_transaction(transaction_root, transaction_dir)
    except Exception:
        try:
            _apply_final_hash_transaction(
                journal,
                transaction_dir,
                use_updated=False,
            )
            _cleanup_final_hash_transaction(transaction_root, transaction_dir)
        except Exception as rollback_exc:
            raise RuntimeError(
                "sidecar transaction failed and durable rollback was incomplete"
            ) from rollback_exc
        raise

    return final_hash, [path for path, _, _ in prepared]


def compressor_thread(thread_id):
    """Consumes finished PDFs and runs Ghostscript."""
    threading.current_thread().name = f"Compressor-{thread_id}"
    logger.info("Compressor %d started.", thread_id)

    while not hard_stop_event.is_set():
        try:
            item = compression_queue.get(timeout=2)
            if isinstance(item, tuple):
                pdf_path, custody_chain = item
            else:
                pdf_path, custody_chain = item, None
        except queue.Empty:
            if (
                stop_event.is_set()
                and image_queue.empty()
                and assembly_queue.empty()
                and finalization_done_event.is_set()
            ):
                break
            continue

        try:
            outcome = _compress_and_refresh_final_output(pdf_path)
            disposition = outcome["disposition"]
            final_hash = outcome["final_hash"]
            updated_sidecars = outcome["updated_sidecars"]
            if custody_chain:
                if disposition in {
                    OptimizationDisposition.SKIPPED_VISUAL_BUDGET.value,
                    OptimizationDisposition.SKIPPED_CANDIDATE_REJECTED.value,
                }:
                    custody_chain.append_event(
                        "compression_skipped",
                        {
                            "pdf_path": pdf_path,
                            "reason": outcome.get("reason", ""),
                            "output_sha256": final_hash,
                            "updated_sidecars": updated_sidecars,
                        },
                    )
                else:
                    # APPLIED — verified replacement committed.
                    custody_chain.append_event(
                        "compression_complete",
                        {
                            "pdf_path": pdf_path,
                            "output_sha256": final_hash,
                            "updated_sidecars": updated_sidecars,
                        },
                    )
                if getattr(custody_chain, "integrity_compromised", False):
                    raise RuntimeError("final custody event was not durably persisted")
        except Exception as e:
            _set_fatal_pipeline_error()
            logger.error("Compression logic failed for %s: %s", pdf_path, e)
            log_failure(pdf_path, 0, f"COMPRESSION_OR_FINAL_HASH_FAILED: {e}")
            if custody_chain:
                try:
                    custody_chain.append_event(
                        "compression_failed",
                        {
                            "pdf_path": pdf_path,
                            "error": str(e),
                        },
                    )
                except Exception as custody_exc:
                    logger.error(
                        "Failed to record compression failure in custody chain: %s",
                        custody_exc,
                    )
        finally:
            compression_queue.task_done()


def _join_queue_with_timeout(q: queue.Queue, timeout: float) -> bool:
    """Join a queue with a bounded timeout. Returns True if drained within timeout."""
    t = threading.Thread(target=q.join, daemon=True, name="drain-helper")
    t.start()
    t.join(timeout=timeout)
    return not t.is_alive()


def _join_threads_or_mark_fatal(threads, role, timeout):
    """Join pipeline threads and make a timeout a fatal, observable condition."""
    for thread in threads:
        thread.join(timeout=timeout)
    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        _set_fatal_pipeline_error()
        logger.critical(
            "%s threads survived shutdown timeout and cannot be declared complete: %s",
            role,
            alive,
        )
    return alive


def _graceful_shutdown(signum, frame):
    """Handle SIGTERM/SIGINT for graceful Docker shutdown."""
    sig_name = signal.Signals(signum).name
    logger.warning("Received %s — initiating graceful shutdown...", sig_name)
    stop_event.set()


def _parse_args():
    """Parse command-line arguments for pipeline configuration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Industrial Adaptive OCR Pipeline (Forensic Grade)",
    )
    parser.add_argument(
        "--enable-docintel",
        action="store_true",
        default=False,
        help="Enable Document Intelligence (PP-StructureV3 layout/table analysis)",
    )
    parser.add_argument(
        "--docintel-mode",
        choices=["layout_only", "tables_only", "full"],
        default="full",
        help="Document Intelligence mode (default: full)",
    )
    parser.add_argument(
        "--export-tables",
        action="store_true",
        default=False,
        help="Export detected tables as HTML/CSV files (requires --enable-docintel)",
    )
    parser.add_argument(
        "--enable-form-detection",
        action="store_true",
        default=False,
        help="Enable form field detection (requires --enable-docintel)",
    )
    parser.add_argument(
        "--enable-kv-extraction",
        action="store_true",
        default=False,
        help="Enable key-value pair extraction (requires --enable-docintel)",
    )
    parser.add_argument(
        "--enable-privilege-detection",
        action="store_true",
        default=False,
        help="Enable legal privilege indicator detection (requires --enable-docintel)",
    )
    parser.add_argument(
        "--no-custody",
        action="store_true",
        default=False,
        help="Disable chain-of-custody audit logging",
    )
    parser.add_argument(
        "--enable-preprocessing",
        action="store_true",
        default=False,
        help="Enable image preprocessing before OCR (deskew, denoise, etc.)",
    )
    parser.add_argument(
        "--preprocessing-level",
        choices=["standard", "enhanced", "aggressive"],
        default="standard",
        help="Preprocessing intensity level (default: standard)",
    )
    parser.add_argument(
        "--enable-noise-profiling",
        action="store_true",
        default=False,
        help="Enable noise profiling for adaptive preprocessing parameter selection",
    )
    parser.add_argument(
        "--enable-dpi-escalation",
        action="store_true",
        default=False,
        help="Enable adaptive DPI escalation for low-confidence pages (300->450->600)",
    )
    parser.add_argument(
        "--enable-ner",
        action="store_true",
        default=False,
        help="Enable Named Entity Recognition (spaCy NER + custom regex patterns)",
    )
    parser.add_argument(
        "--enable-handwriting",
        action="store_true",
        default=False,
        help="Enable handwriting detection (confidence + geometry heuristics)",
    )
    parser.add_argument(
        "--enable-signature-verification",
        action="store_true",
        default=False,
        help=(
            "Enable experimental signature verification sidecars. "
            "This never certifies authenticity and only emits review signals."
        ),
    )
    parser.add_argument(
        "--enable-vertical-text",
        action="store_true",
        default=False,
        help="Enable CJK vertical text detection and reading order analysis",
    )
    parser.add_argument(
        "--enable-table-fallback",
        action="store_true",
        default=False,
        help="Enable per-region table OCR fallback analysis (requires --enable-docintel)",
    )
    parser.add_argument(
        "--enable-classification",
        action="store_true",
        default=False,
        help="Enable document classification (text rules + layout features)",
    )
    parser.add_argument(
        "--enable-extraction",
        action="store_true",
        default=False,
        help="Enable structured field extraction (PaddleNLP UIE + regex fallback)",
    )
    parser.add_argument(
        "--enable-specialist-routing",
        action="store_true",
        default=False,
        help="Enable specialist routing sidecar output from classification and entity signals",
    )
    parser.add_argument(
        "--enable-entity-consolidation",
        action="store_true",
        default=False,
        help="Enable unified entity consolidation (merges NER + extraction + classification into .entities.json)",
    )
    parser.add_argument(
        "--enable-relationship-extraction",
        action="store_true",
        default=False,
        help="Enable cross-entity relationship extraction (requires entity consolidation)",
    )
    parser.add_argument(
        "--enable-retrieval-output",
        action="store_true",
        default=False,
        help="Enable unified retrieval output (merges OCR text, entities, classification, extraction into .retrieval.json)",
    )
    parser.add_argument(
        "--enable-exception-routing",
        action="store_true",
        default=False,
        help="Enable confidence-based exception routing (flags low-quality documents for human review)",
    )
    parser.add_argument(
        "--extractor-mode",
        choices=["thread", "process", "auto"],
        default=None,
        help="Override extractor execution mode (env: EXTRACTOR_MODE).",
    )
    parser.add_argument(
        "--extractor-process-workers",
        type=int,
        default=None,
        help="Number of process workers when --extractor-mode=process (env: EXTRACTOR_PROCESS_WORKERS).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Override the source folder for API-submitted isolated jobs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override the output folder for API-submitted isolated jobs.",
    )
    return parser.parse_args()


def _apply_cli_overrides(args, cfg: PipelineConfig) -> None:
    """Apply CLI arguments onto the active PipelineConfig."""
    if args.source:
        cfg.source_folder = args.source
    if args.output:
        cfg.output_folder = args.output
    if args.enable_docintel:
        cfg.enable_document_intelligence = True
    cfg.docintel_mode = args.docintel_mode
    cfg.export_tables = bool(args.export_tables)
    if cfg.export_tables and not cfg.enable_document_intelligence:
        logger.warning(
            "--export-tables requires --enable-docintel; table export disabled."
        )
        cfg.export_tables = False

    docintel_flags = {
        "enable_form_detection": "enable_form_detection",
        "enable_kv_extraction": "enable_kv_extraction",
        "enable_privilege_detection": "enable_privilege_detection",
    }
    for arg_flag, cfg_field in docintel_flags.items():
        if getattr(args, arg_flag):
            if cfg.enable_document_intelligence:
                setattr(cfg, cfg_field, True)
            else:
                logger.warning(
                    "--%s requires --enable-docintel; ignored.",
                    arg_flag.replace("_", "-"),
                )

    if args.no_custody:
        cfg.enable_custody = False

    if args.enable_preprocessing:
        cfg.enable_preprocessing = True
    cfg.preprocessing_level = args.preprocessing_level

    if args.enable_noise_profiling:
        cfg.enable_noise_profiling = True
    if args.enable_dpi_escalation:
        cfg.enable_dpi_escalation = True
    if args.enable_ner:
        cfg.enable_ner = True
    if args.enable_handwriting:
        cfg.enable_handwriting = True
    if args.enable_signature_verification:
        cfg.enable_signature_verification = True
    if args.enable_vertical_text:
        cfg.enable_vertical_text = True
    if args.enable_table_fallback:
        if cfg.enable_document_intelligence:
            cfg.enable_table_fallback = True
        else:
            logger.warning(
                "--enable-table-fallback requires --enable-docintel; ignored."
            )
    if args.enable_classification:
        cfg.enable_classification = True
    if args.enable_extraction:
        cfg.enable_extraction = True
    if args.enable_specialist_routing:
        cfg.enable_specialist_routing = True
    if args.enable_entity_consolidation:
        cfg.enable_entity_consolidation = True
    if args.enable_relationship_extraction:
        cfg.enable_relationship_extraction = True
    if args.enable_retrieval_output:
        cfg.enable_retrieval_output = True
    if args.enable_exception_routing:
        cfg.enable_exception_routing = True

    if args.extractor_mode:
        cfg.extractor_mode = args.extractor_mode
    if args.extractor_process_workers is not None:
        cfg.extractor_process_workers = max(
            1,
            min(args.extractor_process_workers, 64),
        )


def main():
    from ocr_local.config.version import __version__

    global _PADDLE_POOL_ACCEPTING_WORK
    stop_event.clear()
    hard_stop_event.clear()
    finalization_done_event.clear()
    finalization_abort_event.clear()
    fatal_pipeline_error_event.clear()
    assembly_producers_done_event.clear()
    _start_cpu_paddle_executor_lifecycle()
    with _PADDLE_POOL_LOCK:
        if _PADDLE_POOLS:
            raise RuntimeError("cannot reopen Paddle pools before shutdown")
        _PADDLE_POOL_ACCEPTING_WORK = True

    # Parse command-line arguments and apply to the authoritative config object.
    args = _parse_args()
    global extractor_process_pool
    pipeline_cfg = _ACTIVE_PIPELINE_CONFIG
    _apply_cli_overrides(args, pipeline_cfg)
    pipeline_cfg.extractor_mode = _resolve_auto_extractor_mode(
        pipeline_cfg.extractor_mode,
        pipeline_cfg.num_extractors,
    )
    _activate_pipeline_config(pipeline_cfg)

    if ENABLE_CUSTODY and _CUSTODY_AVAILABLE:
        logger.info("Chain-of-custody logging ENABLED")
    elif ENABLE_CUSTODY and not _CUSTODY_AVAILABLE:
        logger.warning("Chain-of-custody requested but custody module not available")

    logger.debug("Script start")
    logger.info("=== STARTING INDUSTRIAL OCR PIPELINE v%s ===", __version__)
    if ENABLE_DOCUMENT_INTELLIGENCE:
        logger.info("Document Intelligence ENABLED (mode=%s)", DOCINTEL_MODE)
    if ENABLE_PREPROCESSING:
        logger.info("Image preprocessing ENABLED (level=%s)", PREPROCESSING_LEVEL)
    if ENABLE_NOISE_PROFILING:
        logger.info("Noise profiling ENABLED (adaptive preprocessing parameters)")
    if ENABLE_DPI_ESCALATION:
        logger.info(
            "Adaptive DPI escalation ENABLED (threshold=%s)", DPI_CONFIDENCE_THRESHOLD
        )
    if ENABLE_NER:
        logger.info("Named Entity Recognition ENABLED")
    if ENABLE_HANDWRITING:
        logger.info("Handwriting detection ENABLED")
    if ENABLE_SIGNATURE_VERIFICATION:
        logger.info("Signature verification ENABLED (experimental review signals only)")
    if ENABLE_VERTICAL_TEXT:
        logger.info("CJK vertical text detection ENABLED")
    if ENABLE_TABLE_FALLBACK:
        logger.info("Table fallback analysis ENABLED")
    if ENABLE_CLASSIFICATION:
        logger.info("Document classification ENABLED")
    if ENABLE_EXTRACTION:
        logger.info("Structured field extraction ENABLED")
    if ENABLE_SPECIALIST_ROUTING:
        logger.info(
            "Document routing ENABLED (.routing.json via routing.py; "
            "not specialist_routing.SpecialistRouter)"
        )
    if ENABLE_ENTITY_CONSOLIDATION:
        logger.info("Unified entity consolidation ENABLED")
    if ENABLE_RELATIONSHIP_EXTRACTION:
        logger.info("Cross-entity relationship extraction ENABLED")
    if ENABLE_RETRIEVAL_OUTPUT:
        if _OUTPUT_ASSEMBLER_AVAILABLE:
            logger.info("Unified retrieval output ENABLED")
        else:
            logger.warning(
                "Retrieval output requested but output_assembler module not available"
            )
    if ENABLE_EXCEPTION_ROUTING:
        if _EXCEPTION_ROUTER_AVAILABLE:
            logger.info("Confidence-based exception routing ENABLED")
        else:
            logger.warning(
                "Exception routing requested but exception_router module not available"
            )
    logger.info(
        "Extractor mode: %s (workers=%s)",
        EXTRACTOR_MODE,
        EXTRACTOR_PROCESS_WORKERS if EXTRACTOR_MODE in ("process", "auto") else "n/a",
    )

    # Register signal handlers for graceful Docker shutdown
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)

    # 0. Warmup / Download Models (Prevent Race Condition)
    logger.info("Checking/Downloading OCR Models (Single Threaded)...")
    try:
        # Run once to ensure files are present
        warm_pool = _standalone_pool_for_device(_DEFAULT_PADDLE_DEVICE)
        warm_pool.run(
            _standalone_paddle_key("en", _DEFAULT_PADDLE_DEVICE),
            lambda: create_paddle_engine("en", _DEFAULT_PADDLE_DEVICE),
            lambda _engine: None,
        )
    except Exception as e:
        logger.error("Model Download Warning: %s", e)

    # GPU Optimization probe (opt-in via ENABLE_GPU_OPTIMIZATION)
    if _gpu_optimizer is not None:
        try:
            caps = _gpu_optimizer.detect_capabilities()
            config = _gpu_optimizer.recommend_config(caps)
            global _batch_preprocessor
            _batch_preprocessor = BatchPreprocessor(config)
            logger.info(
                "GPU optimization: %s", _gpu_optimizer.get_optimization_summary()
            )
        except Exception as gpu_opt_err:
            logger.warning("GPU optimization probe failed: %s", gpu_opt_err)

    if EXTRACTOR_MODE == "process":
        try:
            extractor_process_pool = ProcessPoolExecutor(
                max_workers=EXTRACTOR_PROCESS_WORKERS,
            )
            logger.info(
                "Initialized process-based extractor pool with %s workers.",
                EXTRACTOR_PROCESS_WORKERS,
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize process pool (%s); falling back to thread mode.",
                e,
            )
            pipeline_cfg.extractor_mode = "thread"
            _activate_pipeline_config(pipeline_cfg)
            extractor_process_pool = None

    # maintenance review Phase 3: log the active config after env parsing + CLI overrides.
    logger.info("Pipeline configuration:\n%s", pipeline_cfg)

    # 1. Start Threads
    logger.debug("Starting threads...")

    t_sched = threading.Thread(target=scheduler_thread)
    t_sched.start()

    extractors = []
    for i in range(NUM_EXTRACTORS):
        t = threading.Thread(target=extractor_thread, args=(i + 1,))
        t.start()
        extractors.append(t)

    workers = []
    for i in range(NUM_WORKERS):
        t = threading.Thread(target=worker_thread, args=(i + 1,))
        t.start()
        workers.append(t)

    t_asm = threading.Thread(target=assembler_thread)
    t_asm.start()

    t_mon = threading.Thread(target=monitor_thread, daemon=True)
    t_mon.start()

    compressors = []
    for i in range(NUM_COMPRESSORS):
        t = threading.Thread(target=compressor_thread, args=(i + 1,))
        t.start()
        compressors.append(t)

    # 2. Drain queues — always drain, even on SIGTERM (with bounded timeout)
    #
    # ORDERING NOTE (maintenance review): The assembler thread now dispatches doc
    # finalization to a ThreadPoolExecutor.  Finalizer threads put items
    # into compression_queue, so compression_queue.join() must come AFTER
    # t_asm.join() (which waits for the executor to shut down).
    #
    # R4-RC-15/51 / PB-A04: sequential drains share ONE monotonic deadline so
    # wall-clock cannot exceed SHUTDOWN_DRAIN_TIMEOUT_SECONDS (and therefore
    # remains under stop_grace_period when grace >= drain, default 360 >= 300).
    assembler_shutdown_incomplete = False
    t_sched.join()
    if stop_event.is_set():
        _drain_budget = float(SHUTDOWN_DRAIN_TIMEOUT_SECONDS)
        _drain_deadline = time.monotonic() + max(0.0, _drain_budget)

        def _remaining_drain_budget() -> float:
            return max(0.0, _drain_deadline - time.monotonic())

        logger.info(
            "Shutdown signal received — draining in-flight work "
            "(shared budget: %ds wall-clock across queues)...",
            SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
        )
        _producer_queues_drained = True
        for _q, _qname in [
            (chunk_queue, "chunk"),
            (image_queue, "image"),
        ]:
            _budget = _remaining_drain_budget()
            _drained = _budget > 0 and _join_queue_with_timeout(_q, _budget)
            if not _drained:
                _producer_queues_drained = False
                _request_finalization_abort()
                logger.critical(
                    "%s_queue did not fully drain within remaining budget "
                    "(shared SHUTDOWN_DRAIN_TIMEOUT=%ds)",
                    _qname,
                    SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
                )
        if _producer_queues_drained:
            # No extractor/worker can enqueue another assembly message after
            # both upstream queues report every item task_done.
            assembly_producers_done_event.set()
        _budget = _remaining_drain_budget()
        _drained = _budget > 0 and _join_queue_with_timeout(assembly_queue, _budget)
        if not _drained:
            _request_finalization_abort()
            logger.critical(
                "assembly_queue did not fully drain within remaining budget "
                "(shared SHUTDOWN_DRAIN_TIMEOUT=%ds)",
                SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
            )
        # Wait for assembler + finalization executor before draining compression
        _budget = _remaining_drain_budget()
        _alive_finalizers = _join_threads_or_mark_fatal(
            [t_asm],
            "assembler/finalizer",
            _budget,
        )
        if _alive_finalizers:
            assembler_shutdown_incomplete = True
            _request_finalization_abort()
        else:
            finalization_done_event.set()
        _budget = _remaining_drain_budget()
        _drained = _budget > 0 and _join_queue_with_timeout(
            compression_queue,
            _budget,
        )
        if not _drained:
            _set_fatal_pipeline_error()
            logger.critical(
                "compression_queue did not fully drain within remaining budget "
                "(shared SHUTDOWN_DRAIN_TIMEOUT=%ds)",
                SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
            )
    else:
        chunk_queue.join()
        image_queue.join()
        # This is the explicit producer-done barrier.  Scheduler has joined,
        # chunk extraction is complete, and every OCR task reached task_done;
        # therefore the assembly queue now contains the full terminal stream,
        # including all duplicates.
        assembly_producers_done_event.set()
        assembly_queue.join()
        # Wait for assembler + finalization executor before draining compression
        t_asm.join()
        finalization_done_event.set()
        compression_queue.join()

    stop_event.set()  # Ensure soft-stop is set (for assembler/compressor termination conditions)
    if not assembler_shutdown_incomplete:
        finalization_done_event.set()
    hard_stop_event.set()  # Signal worker loops to exit
    _close_cpu_paddle_executor_lifecycle()

    # 3. Wait for Workers (with timeout to avoid hanging on shutdown)
    _join_threads_or_mark_fatal(workers, "OCR worker", THREAD_JOIN_TIMEOUT)
    _join_threads_or_mark_fatal(extractors, "extractor", THREAD_JOIN_TIMEOUT)

    # t_asm already joined above during drain cascade — skip standalone join

    _join_threads_or_mark_fatal(compressors, "compressor", THREAD_JOIN_TIMEOUT)

    if extractor_process_pool is not None:
        extractor_process_pool.shutdown(wait=True, cancel_futures=True)
        extractor_process_pool = None

    _shutdown_cpu_paddle_executor()
    _close_standalone_paddle_pools()

    if fatal_pipeline_error_event.is_set():
        raise RuntimeError(
            "Pipeline completed with a fatal worker, compression, custody, "
            "or final-hash error"
        )

    logger.info("=== ALL TASKS COMPLETED ===")


if __name__ == "__main__":
    load_fasttext()
    main()
