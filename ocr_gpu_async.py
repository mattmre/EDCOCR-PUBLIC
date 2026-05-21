import csv
import datetime
import enum
import hashlib
import io
import json
import logging
import os
import queue
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from html.parser import HTMLParser
from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# PageStatus enum — type-safe replacement for scattered status string literals
#.  Inherits from ``str`` so comparisons with plain strings remain
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
from PIL import Image

from optimize_pdfs import optimize_pdf  # Direct Import
from pipeline_config import PipelineConfig, create_pipeline_config  # 
from video_utils import get_video_page_count, iter_video_frames

try:
    import fasttext  # Restored for Language Detection
    _FASTTEXT_AVAILABLE = True
except ImportError:
    fasttext = None
    _FASTTEXT_AVAILABLE = False

_DISABLE_PADDLEOCR = os.environ.get("EDCOCR_DISABLE_PADDLEOCR", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

try:
    if _DISABLE_PADDLEOCR:
        raise ImportError("PaddleOCR disabled by EDCOCR_DISABLE_PADDLEOCR")
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

# --- Per-span language detection (Plan A -- PR A2, opt-in) ---
try:
    from ocr_local.features.language_detection import (
        aggregate_page_from_spans,
        detect_span_language,
        finalize_document_language,
        write_language_json,
    )
    _LANGUAGE_DETECTION_AVAILABLE = True
except ImportError:
    _LANGUAGE_DETECTION_AVAILABLE = False

# --- Assembly Queue Message Schema ---


class _AssemblyMessageRequired(TypedDict):
    """Required keys present in every assembly queue message."""
    doc_id: str
    page_num: int
    text: str
    status: str
    chunk_path: str | None


class AssemblyMessage(_AssemblyMessageRequired, total=False):
    """Queue message from worker/extractor threads to the assembler thread.

    Required keys (always present): doc_id, page_num, text, status, chunk_path.
    Optional keys (present in OCR-success paths; None/0.0 in failure paths).
    """
    structure_data: dict | None
    ocr_confidence: float
    ocr_method: str
    handwriting_data: dict | None
    signature_data: dict | None
    vertical_text_data: dict | None
    table_fallback_data: dict | None
    language_data: object | None  # PageLanguage | None (Plan A -- PR A2)


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
    ".mp4", ".avi", ".mov", ".m4v", ".mpg", ".mpeg", ".mkv", ".webm",
}
PHASE1_IMAGE_EXTENSIONS = {
    ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".gif",
    ".webp", ".jp2", ".jpx", ".pnm", ".pbm", ".pgm", ".ppm",
    ".pcx", ".ico", ".svg", ".svgz", ".wmf", ".emf",
}
PHASE2_IMAGE_EXTENSIONS = {".heic", ".heif", ".avif", ".jxl", ".jxr", ".dcx", ".xps"}
STRICT_PDF_SIGNATURE = os.environ.get("OCR_STRICT_PDF_SIGNATURE", "").lower() in (
    "1",
    "true",
    "yes",
)
RESUME_MANIFEST_FILENAME = "resume_manifest.json"
RESUME_MANIFEST_SCHEMA_VERSION = 2

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
    if not header:
        return None

    stripped = header.lstrip()

    if header.startswith(b"%PDF-"):
        return "pdf"
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
    if magic_family and magic_family != ext_family:
        return None, f"Signature mismatch: extension={ext_family} magic={magic_family}"

    if not magic_family:
        if ext_family == "pdf" and strict_pdf_signature:
            return None, "Missing PDF signature (strict mode)"
        return ext_family, "No known signature match; accepted by extension fallback"

    return ext_family, None

def get_source_page_count(path, source_type):
    """Returns total page/frame count for pdf, image, and video inputs."""
    if source_type == "pdf":
        with fitz.open(path) as doc:
            return int(doc.page_count)
    if source_type == "video":
        return get_video_page_count(
            path,
            sample_seconds=VIDEO_FRAME_SAMPLE_SECONDS,
            max_frames=VIDEO_MAX_FRAMES,
        )

    try:
        with Image.open(path) as img:
            return max(1, int(getattr(img, "n_frames", 1) or 1))
    except Exception as exc:
        logger.debug("PIL page count failed for %s, falling back to fitz: %s", path, exc)
        with fitz.open(path) as doc:
            return max(1, int(doc.page_count))

def iter_pil_image_frames(path, start, end):
    """Yields PIL RGB frames for 1-based inclusive page range."""
    with Image.open(path) as src:
        frame_count = max(1, int(getattr(src, "n_frames", 1) or 1))
        upper = min(end, frame_count)
        for page_num in range(start, upper + 1):
            src.seek(page_num - 1)
            # explicitly close copied frames after consumer finishes
            # to prevent PIL Image handle accumulation on multi-frame TIFFs.
            frame = src.convert("RGB").copy()
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
            pix = doc[page_num - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            if pix.n == 1:
                yield Image.frombytes("L", (pix.width, pix.height), pix.samples).convert("RGB")
            else:
                yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

def iter_source_images(path, start, end, source_type):
    """Yields PIL images for the requested page range."""
    if source_type == "pdf":
        if not _PDF2IMAGE_AVAILABLE or convert_from_path is None:
            raise RuntimeError("pdf2image dependency not installed")
        # extract one page at a time to avoid O(chunk_pages * 20MB) peak
        for page_num in range(start, end + 1):
            page_images = convert_from_path(
                path,
                first_page=page_num,
                last_page=page_num,
                dpi=DPI,
                thread_count=1,  # single page -- no benefit from multiple threads
                timeout=POPPLER_TIMEOUT,
            )
            if page_images:
                yield page_images[0].convert("RGB")
        return
    if source_type == "video":
        for frame in iter_video_frames(
            path,
            start,
            end,
            sample_seconds=VIDEO_FRAME_SAMPLE_SECONDS,
            max_frames=VIDEO_MAX_FRAMES,
        ):
            yield frame
        return

    try:
        for frame in iter_pil_image_frames(path, start, end):
            yield frame
    except Exception as pil_err:
        logger.warning("PIL decode fallback for %s: %s", path, pil_err)
        for frame in iter_fitz_image_frames(path, start, end):
            yield frame


def _encode_image_to_jpeg_bytes(img):
    """Serialize PIL image for optional process-based extraction."""
    payload = io.BytesIO()
    img.convert("RGB").save(payload, format="JPEG", quality=JPEG_QUALITY)
    return payload.getvalue()


def _decode_image_from_jpeg_bytes(payload):
    """Decode serialized JPEG payload into a PIL RGB image."""
    with Image.open(io.BytesIO(payload)) as img:
        return img.convert("RGB")


def _extract_chunk_in_subprocess(input_path, start, end, source_type, dpi, pdf_conversion_threads):
    """Extract pages in a worker process and return JPEG payloads."""
    extracted = []

    if source_type == "pdf":
        if not _PDF2IMAGE_AVAILABLE or convert_from_path is None:
            raise RuntimeError("pdf2image dependency not installed")
        # extract one page at a time to avoid O(chunk_pages * 20MB) peak
        for page_num in range(start, end + 1):
            page_images = convert_from_path(
                input_path,
                first_page=page_num,
                last_page=page_num,
                dpi=dpi,
                thread_count=1,  # single page -- no benefit from multiple threads
                timeout=POPPLER_TIMEOUT,
            )
            if page_images:
                extracted.append((page_num, _encode_image_to_jpeg_bytes(page_images[0])))
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
        logger.debug("PIL extraction failed for %s, falling back to fitz: %s", input_path, exc)
        zoom = dpi / 72.0
        with fitz.open(input_path) as doc:
            upper = min(end, int(doc.page_count))
            for page_num in range(start, upper + 1):
                pix = doc[page_num - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                if pix.n == 1:
                    frame = Image.frombytes("L", (pix.width, pix.height), pix.samples).convert("RGB")
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
    """
    Build a stable output stem relative to SOURCE_FOLDER.
    For non-PDF sources, include extension token to prevent basename collisions.
    """
    try:
        rel_path = os.path.relpath(path, SOURCE_FOLDER)
        if rel_path == ".." or rel_path.startswith(f"..{os.sep}"):
            rel_path = os.path.basename(path)
    except (OSError, ValueError):
        rel_path = os.path.basename(path)

    if not rel_path:
        rel_path = os.path.basename(path)

    # Sanitize each path component to handle problematic filenames (TD-012)
    parts = rel_path.replace("\\", "/").split("/")
    parts = [_sanitize_path_segment(p) for p in parts]
    rel_path = os.path.join(*parts) if parts else rel_path
    rel_stem, rel_ext = os.path.splitext(rel_path)
    if source_type == "pdf":
        return rel_stem

    ext_token = rel_ext.lower().lstrip(".")
    safe_ext = "".join(ch if ch.isalnum() else "_" for ch in ext_token)
    if not safe_ext:
        safe_ext = "img"
    return f"{rel_stem}__{safe_ext}"


def _build_sidecar_base_name(source_file):
    """Build sidecar basename with non-PDF extension token to avoid collisions."""
    base_name = _sanitize_path_segment(os.path.splitext(os.path.basename(source_file))[0])
    if not base_name:
        base_name = "document"

    ext = os.path.splitext(source_file)[1].lower().lstrip(".")
    safe_ext = "".join(ch if ch.isalnum() else "_" for ch in ext)
    if safe_ext and safe_ext != "pdf":
        return f"{base_name}__{safe_ext}"
    return base_name


def _validation_sidecar_path_for_source(source_file):
    """Resolve existing validation sidecar path for a source file."""
    validation_dir = os.path.join(OUTPUT_FOLDER, "EXPORT", "VALIDATION")
    try:
        rel_dir = os.path.dirname(os.path.relpath(source_file, SOURCE_FOLDER))
    except (OSError, ValueError):
        rel_dir = ""

    target_dir = (
        os.path.join(validation_dir, rel_dir)
        if rel_dir and rel_dir != "."
        else validation_dir
    )
    base_name = _build_sidecar_base_name(source_file)
    return os.path.join(target_dir, f"{base_name}.validation.json")


def _load_validation_page_cache(source_file):
    """Load prior validation page metadata keyed by page number."""
    sidecar_path = _validation_sidecar_path_for_source(source_file)
    if not os.path.exists(sidecar_path):
        return {}

    try:
        with open(sidecar_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        logger.debug("Failed to load validation sidecar %s: %s", sidecar_path, exc)
        return {}

    pages = payload.get("pages")
    if not isinstance(pages, list):
        return {}

    cached = {}
    for page in pages:
        if not isinstance(page, dict):
            continue
        try:
            page_num = int(page.get("page_num", 0))
        except Exception as exc:
            logger.debug("Invalid page_num in validation sidecar: %s", exc)
            continue
        if page_num <= 0:
            continue
        cached[page_num] = {
            "ocr_method": str(page.get("ocr_method", "")),
            "ocr_language": str(page.get("ocr_language", "")),
            "ocr_confidence": float(page.get("ocr_confidence", 0.0) or 0.0),
            "text_length": int(page.get("text_length", 0) or 0),
            "has_text": bool(page.get("has_text", False)),
            "status": str(page.get("status", "unknown")),
        }
    return cached


def _extract_text_from_chunk_pdf(chunk_path):
    """Best-effort extraction of embedded text from a chunk PDF."""
    if not chunk_path or not os.path.exists(chunk_path):
        return ""

    try:
        with fitz.open(chunk_path) as doc:
            parts = []
            for page in doc:
                text = page.get_text("text")
                if text:
                    parts.append(text)
            return "".join(parts).strip()
    except Exception as exc:
        logger.debug("Failed to extract text from chunk PDF %s: %s", chunk_path, exc)
        return ""

def get_path_based_doc_id(path):
    """Returns a deterministic document ID derived from the file path.

    This is NOT a content hash -- use compute_file_hash() from
    ocr_distributed/ocr_utils.py for content integrity verification.
    """
    return hashlib.sha256(path.encode('utf-8')).hexdigest()[:16]


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
        "mtime_ns": int(getattr(stat_info, "st_mtime_ns", int(stat_info.st_mtime * 1_000_000_000))),
    }


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
    mtime_ns = int(getattr(stat_info, "st_mtime_ns", int(stat_info.st_mtime * 1_000_000_000)))
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

    Crash-safety: writes payload to a uniquely-named temp file in
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
        "updated_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
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
        name
        for name in os.listdir(temp_dir)
        if name != RESUME_MANIFEST_FILENAME
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

    This is the  investigation outcome: the prior in-place algorithm
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


_failure_report_lock = threading.Lock()


def log_failure(path, page, error):
    """Writes failure to CSV report."""
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clean_err = str(error).replace(",", ";").replace("\n", " ")
        with _failure_report_lock:
            with open(FAILURE_REPORT, "a", encoding="utf-8") as f:
                f.write(f"{ts},{path},{page},{clean_err}\n")
    except Exception as log_err:
        logger.warning("Failed to write failure report for %s p%s: %s", path, page, log_err)

def img_to_bytes(img):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=JPEG_QUALITY)
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


def insert_text_line(page, txt, box, lang_code=None):
    """Attempts bbox placement first, then falls back to anchor insertion.

    When *lang_code* is provided and a matching Noto font file is
    installed, the text is embedded with that font so non-Latin scripts
    render correctly in the searchable PDF text layer.
    """
    fontname, fontfile = _resolve_text_font(lang_code)
    font_kwargs = {"fontname": fontname}
    if fontfile:
        font_kwargs["fontfile"] = fontfile

    rect, anchor = box_to_rect_and_anchor(box)
    if rect and rect.width > 1 and rect.height > 1:
        # Width/height-fit direct baseline placement gives tighter geometry than
        # insert_textbox for hidden text overlays.
        unit_width = fitz.get_text_length(txt, fontsize=1.0, **font_kwargs)
        max_by_width = (rect.width * 0.98) / unit_width if unit_width > 0 else rect.height
        max_by_height = rect.height * 0.92
        font_size = max(6.0, min(max_by_width, max_by_height, 240.0))

        baseline = (rect.x0, rect.y0 + font_size)
        page.insert_text(baseline, txt, fontsize=font_size, render_mode=3, **font_kwargs)
        return True

    if anchor:
        page.insert_text(anchor, txt, fontsize=12, render_mode=3, **font_kwargs)
        return True

    return False

try:
    from ocr_local.config.language_config import LANG_MAPPING
except ImportError:
    LANG_MAPPING = {
        'zh': 'ch', 'zh-cn': 'ch', 'zh-tw': 'chinese_cht',
        'ja': 'japan', 'ko': 'korean', 'vi': 'vi',
        'en': 'en', 'fr': 'fr', 'de': 'german', 'es': 'es', 'it': 'it',
        'pt': 'pt', 'ru': 'ru', 'ar': 'ar', 'hi': 'hi',
        'uk': 'uk', 'be': 'be', 'bg': 'bg', 'cs': 'cs', 'pl': 'pl',
        'tr': 'tr', 'nl': 'nl', 'sv': 'sv', 'da': 'da',
        'fi': 'fi', 'el': 'el', 'hu': 'hu', 'ro': 'ro',
        'fa': 'fa', 'ur': 'ur', 'ug': 'ug',
        'te': 'te', 'ta': 'ta', 'kn': 'kn', 'ka': 'ka',
    }
lang_model = None

def load_fasttext():
    global lang_model
    if not _FASTTEXT_AVAILABLE:
        logger.warning("FastText dependency not installed; language detection disabled.")
        return
    try:
        if os.path.exists(FASTTEXT_MODEL_PATH):
            lang_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
            logger.info("FastText Language Model Loaded.")
    except Exception as e:
        logger.warning("FastText Load Failed: %s", e)

def detect_language(doc_path):
    """Attempts to detect language from the first page text (if any)."""
    if not lang_model:
        return 'en'
    try:
        with fitz.open(doc_path) as doc:
            # Try getting text from first few pages
            text_sample = ""
            for i in range(min(3, doc.page_count)):
                text_sample += doc[i].get_text()
            
            if len(text_sample) > 50:
                text_sample = text_sample.replace("\n", " ")[:200]
                prediction = lang_model.predict(text_sample)
                lang_code = prediction[0][0].replace('__label__', '')
                return LANG_MAPPING.get(lang_code, 'en')
    except Exception as exc:
        logger.debug("Language detection from path failed: %s", exc)

    return 'en'

def detect_language_from_text(text):
    """Detects language from OCR output string."""
    if not lang_model or len(text) < 20:
        return None, 0.0
    try:
        # Create a clean sample (remove digits/symbols)
        sample = text.replace("\n", " ")[:300]
        prediction = lang_model.predict(sample)
        lang_code = prediction[0][0].replace('__label__', '')
        conf = prediction[1][0]
        mapped = LANG_MAPPING.get(lang_code)
        
        if mapped and conf > 0.4: # 40% Confidence threshold
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
    return PaddleOCR(
        use_angle_cls=True,
        lang=lang_code,
        use_gpu=(device == "gpu"),
        show_log=False,
    )


def get_or_create_engine(lang_code, device='gpu'):
    """Return a shared PaddleOCR engine for *lang_code*, creating it if needed.

    Thread-safe.  The returned tuple is ``(engine, inference_lock)`` where
    *inference_lock* is a :class:`threading.Lock` that callers **must** hold
    while calling ``engine.ocr()`` to prevent concurrent CUDA session
    corruption.

    If primary language model loading fails, falls back to English.  If
    the English fallback also fails, the entry is cached as ``None`` so
    creation is not retried.

    Uses ``model_load_lock`` for engine construction (same serialization
    as the previous per-thread implementation) and ``_ENGINE_CACHE_LOCK``
    for cache reads/writes.
    """
    # Fast path -- already cached (no lock needed for dict read under GIL,
    # but we use the lock to stay safe with the double-check pattern).
    with _ENGINE_CACHE_LOCK:
        if lang_code in _ENGINE_CACHE:
            return _ENGINE_CACHE[lang_code]

    # Slow path -- must create.  Hold model_load_lock to serialize GPU
    # model loading, then store the result under _ENGINE_CACHE_LOCK.
    with model_load_lock:
        # Double-check: another thread may have created while we waited.
        with _ENGINE_CACHE_LOCK:
            if lang_code in _ENGINE_CACHE:
                return _ENGINE_CACHE[lang_code]

        # Create the engine (outside _ENGINE_CACHE_LOCK to avoid holding
        # two locks simultaneously).
        entry = None
        try:
            logger.info("Loading shared PaddleOCR model for: %s", lang_code)
            engine = create_paddle_engine(lang_code, device=device)
            entry = (engine, threading.Lock())
        except Exception as primary_exc:
            logger.warning(
                "Failed to load %s model, falling back to English. Error: %s",
                lang_code, primary_exc,
            )
            try:
                engine = create_paddle_engine('en', device=device)
                entry = (engine, threading.Lock())
            except Exception as fallback_exc:
                logger.error(
                    "English fallback also failed for %s. "
                    "All pages will use Tesseract/ImageOnly. Error: %s",
                    lang_code, fallback_exc,
                )
                entry = None  # Cache the failure to avoid retrying

        with _ENGINE_CACHE_LOCK:
            _ENGINE_CACHE[lang_code] = entry

    return entry


# --- Document Intelligence Helpers (Phase 3A) ---

def parse_structure_result(result):
    """Parse PP-StructureV3 result into standardized format.

    Returns dict with 'layout_regions', 'tables', 'key_value_pairs',
    and 'form_fields' lists.  Safe to call with None or empty input.
    """
    if not result:
        return {"layout_regions": [], "tables": [], "key_value_pairs": [], "form_fields": []}

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
                "html": item["res"].get("html", "") if isinstance(item["res"], dict) else "",
                "cell_bbox": item["res"].get("cell_bbox", []) if isinstance(item["res"], dict) else [],
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
                        item["res"], region.get("bbox", []), field_id,
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

_ESQ_PATTERN = re.compile(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*),\s*Esq\.?')
_LAW_FIRM_PATTERN = re.compile(
    r'\b\w+(?:\s+\w+)*\s+(?:&|and)\s+\w+(?:\s+\w+)*\s+(?:LLP|PLLC|P\.C\.|LLC)\b'
    r'|(?:Law\s+Offices?\s+of\b)'
    r'|(?:Attorneys?\s+at\s+Law\b)',
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
                key_value_pairs.append({
                    "key": key,
                    "value": value,
                    "key_bbox": key_bbox,
                    "value_bbox": value_bbox,
                    "confidence": conf,
                    "extraction_method": "heuristic_colon",
                })

        # Heuristic 2: Form field keyword detection
        if ENABLE_FORM_DETECTION:
            text_lower = text.lower()
            for ftype, keywords in _FORM_FIELD_KEYWORDS.items():
                if any(kw in text_lower for kw in keywords):
                    form_fields.append({
                        "field_id": field_id,
                        "label": text.strip(),
                        "value": None,
                        "field_type": ftype,
                        "bbox": bbox,
                        "confidence": conf,
                        "is_filled": False,
                    })
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
    import json

    from ocr_local.config.version import __version__

    try:
        structure_dir = os.path.join(output_dir, "EXPORT", "STRUCTURE")
        # Mirror subfolder structure from source
        rel_path = os.path.relpath(source_file, SOURCE_FOLDER)
        rel_dir = os.path.dirname(rel_path)
        target_dir = os.path.join(structure_dir, rel_dir) if rel_dir != "." else structure_dir
        os.makedirs(target_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(source_file))[0]
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

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc_structure, f, indent=2, ensure_ascii=False)

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


_SAFE_TABLE_TAGS = frozenset({"table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "colgroup", "col"})
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
    try:
        rel_path = os.path.relpath(source_file, SOURCE_FOLDER)
    except ValueError:
        rel_path = os.path.basename(source_file)
    # Sanitize each path segment to prevent path traversal
    parts = rel_path.replace("\\", "/").split("/")
    parts = [_sanitize_path_segment(p) for p in parts]
    rel_dir = os.path.join(*parts[:-1]) if len(parts) > 1 else ""
    target_dir = os.path.join(tables_dir, rel_dir) if rel_dir and rel_dir != "." else tables_dir
    base_name = os.path.splitext(parts[-1])[0] if parts else "unknown"
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
                # Write HTML
                html_path = os.path.join(target_dir, f"{stem}.html")
                with open(html_path, "w", encoding="utf-8") as fh:
                    fh.write(sanitized_html)
                # Write CSV
                csv_path = os.path.join(target_dir, f"{stem}.csv")
                rows = html_table_to_csv_rows(html_content)
                with open(csv_path, "w", encoding="utf-8", newline="") as fc:
                    csv.writer(fc).writerows(rows)
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
    def __init__(self, doc_id: str, doc_path: str, page_num: int, image: Image.Image, lang_hint: str = 'en', source_type: str = 'pdf') -> None:
        self.doc_id: str = doc_id
        self.doc_path: str = doc_path
        self.page_num: int = page_num
        self.image: Image.Image = image
        self.lang_hint: str = lang_hint
        self.source_type: str = source_type
        self.retries: int = 0

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
        self.start_time: float = time.time()
        rel_stem = build_output_rel_stem(path, source_type)
        self.output_pdf: str = os.path.join(OUTPUT_FOLDER, "EXPORT", "PDF", rel_stem + ".pdf")
        os.makedirs(os.path.dirname(self.output_pdf), exist_ok=True)

        self.output_txt_dir: str = os.path.join(OUTPUT_FOLDER, "EXPORT", "TEXT", os.path.dirname(rel_stem))
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
        logger.info("Adaptive batch sizing enabled (strategy=%s)", _adaptive_batch_sizer._config.strategy.value)
    except ImportError:
        logger.warning("adaptive_batch module not found, falling back to fixed chunk size")

# --- Global State ---
doc_registry = {}  # {doc_id: DocumentState}
doc_registry_lock = threading.RLock()  # Protects doc_registry from concurrent access
model_load_lock = threading.Lock() # Prevents race condition during model download/load

# Shared PaddleOCR engine cache --
# Without this, each of the 12 GPU worker threads maintains its own
# {lang: engine} dict, resulting in O(threads x languages) engine
# instances competing for VRAM.  The shared cache ensures only ONE
# engine per language exists in VRAM across all threads.
# Each engine is paired with a threading.Lock to serialize inference
# calls, since PaddleOCR's internal CUDA session state is not
# guaranteed to be thread-safe for concurrent ocr() calls.
_ENGINE_CACHE: dict = {}            # {lang: (engine, inference_lock)}
_ENGINE_CACHE_LOCK = threading.Lock()
chunk_queue = queue.Queue(maxsize=CHUNK_QUEUE_SIZE) # PDF chunks for extractors
image_queue = queue.Queue(maxsize=IMAGE_QUEUE_SIZE) # Images for workers
assembly_queue: queue.Queue[AssemblyMessage] = queue.Queue(maxsize=RESULT_QUEUE_SIZE)
compression_queue = queue.Queue(maxsize=COMPRESSION_QUEUE_SIZE) # Finished PDFs for Ghostscript
extractor_process_pool = None

# Signals
stop_event = threading.Event()
hard_stop_event = threading.Event()  # Set AFTER drain completes (triggers thread exit)
finalization_done_event = threading.Event()

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
        logger.info("Page cache enabled (max_entries=%d, max_bytes=%d)",
                     _page_cache._max_entries, _page_cache._max_size_bytes)
    except ImportError:
        logger.warning("page_cache module not found, caching disabled")

# Page Routing (opt-in via ENABLE_PAGE_ROUTING)
_page_router = None
if ENABLE_PAGE_ROUTING:
    try:
        from ocr_local.infra.page_routing import PageFeatures, PageRouter
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
            source_type, detail = classify_source_file(input_path)
            if source_type:
                files.append((input_path, source_type))
                if detail:
                    logger.warning("Source accepted with extension fallback: %s (%s)", input_path, detail)
            else:
                skipped += 1
                if detail:
                    logger.warning("Source skipped: %s (%s)", input_path, detail)

    logger.info("Found %d supported documents. Skipped %d unsupported/planned files.", len(files), skipped)

    for input_path, source_type in files:
        logger.info("Processing: %s", input_path) # Explicit Start Log
        if stop_event.is_set():
            break

        try:
            source_fingerprint = compute_source_fingerprint_fast(input_path)
            doc_id = build_resume_doc_id(input_path, source_fingerprint)
        except Exception as e:
            logger.error("Failed to fingerprint source %s: %s", input_path, e)
            log_failure(input_path, 0, f"FINGERPRINT_FAILED: {e}")
            continue
        
        # Detect Language (Best Effort)
        detected_lang = detect_language(input_path) if source_type == "pdf" else "en"

        doc_state = DocumentState(input_path, doc_id, source_type)
        with doc_registry_lock:
            doc_registry[doc_id] = doc_state

        # 1. Get Page Count (Fast)
        try:
            doc_temp_dir = getattr(doc_state, "temp_dir", os.path.join(TEMP_FOLDER, doc_id))
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

            doc_state.total_pages = get_source_page_count(input_path, source_type)

            if doc_state.total_pages <= 0:
                reason = (
                    f"SOURCE_PAGE_COUNT_ZERO: {source_type} source reported 0 pages"
                )
                logger.error(
                    "Source rejected for processing: %s (ID: %s) [%s] -> %s",
                    input_path, doc_id, source_type, reason,
                )
                log_failure(input_path, 0, reason)
                if doc_state.custody_chain:
                    doc_state.custody_chain.append_event("processing_failed", {
                        "stage": "scheduler",
                        "error": reason,
                    })
                with doc_registry_lock:
                    if doc_id in doc_registry:
                        del doc_registry[doc_id]
                continue


            logger.info(
                "Scheduling: %s (ID: %s) [%s] [%s] (%d pages)",
                os.path.basename(input_path), doc_id, detected_lang, source_type, doc_state.total_pages,
            )

            # Custody: record file ingestion
            if doc_state.custody_chain:
                doc_state.custody_chain.append_event("file_ingested", {
                    "source_path": input_path,
                    "source_type": source_type,
                    "total_pages": doc_state.total_pages,
                    "detected_language": detected_lang,
                    "file_hash": doc_id,
                    "source_fingerprint": source_fingerprint,
                })
            
            # 2. Schedule Chunks - Optimized for Granular Resume
            if _adaptive_batch_sizer is not None:
                try:
                    file_size = os.path.getsize(input_path) if os.path.exists(input_path) else 0
                    complexity = _adaptive_batch_sizer.compute_complexity(
                        width=0, height=0,  # not known at scheduler level
                        file_size=file_size,
                        dpi=DPI,
                        has_tables=False,
                        has_images=False,
                    )
                    chunk_target_size = _adaptive_batch_sizer.recommend_batch_size([complexity])
                    logger.debug(
                        "Adaptive chunk size for %s: %d (complexity=%.4f, file_size=%d)",
                        os.path.basename(input_path), chunk_target_size,
                        complexity.complexity_score, file_size,
                    )
                except Exception as exc:
                    logger.debug("Adaptive batch sizing failed, using default: %s", exc)
                    chunk_target_size = CHUNK_TARGET_SIZE
            else:
                chunk_target_size = CHUNK_TARGET_SIZE
            
            # Fast Resume: Bulk load existing pages
            existing_pages: set[int] = set()
            if os.path.exists(doc_state.temp_dir):
                logger.info("Scanning existing files in %s...", doc_state.temp_dir)
                for f in os.listdir(doc_state.temp_dir):
                    if f.endswith(".pdf"):
                        try:
                            # Filename format "1.pdf"
                            existing_pages.add(int(os.path.splitext(f)[0]))
                        except Exception as exc:
                            logger.debug("Skipping non-numeric temp file %s: %s", f, exc)
                logger.info("Found %d existing pages.", len(existing_pages))

            # Notify assembler of every RESUMED page (in order) so doc state
            # reflects on-disk progress before any new extraction is dispatched.
            for p in range(1, doc_state.total_pages + 1):
                if p in existing_pages:
                    chunk_path = os.path.join(doc_state.temp_dir, f"{p}.pdf")
                    assembly_queue.put({
                        "doc_id": doc_id,
                        "page_num": p,
                        "text": "",
                        "status": PageStatus.RESUMED,
                        "chunk_path": chunk_path,
                        "structure_data": None,
                        "ocr_confidence": 0.0,
                        "ocr_method": PageStatus.RESUMED,
                        "handwriting_data": None,
                        "signature_data": None,
                        "vertical_text_data": None,
                        "table_fallback_data": None,
                    })

            # compute gap chunks via set-difference semantics. The
            # helper groups missing pages into contiguous runs and splits
            # each run by chunk_target_size, guaranteeing that every
            # dispatched chunk is (a) contiguous and (b) only covers truly
            # missing pages (never re-extracts resumed pages).
            gap_chunks = compute_resume_gap_chunks(
                doc_state.total_pages, existing_pages, chunk_target_size,
            )
            for start_gap, end_gap in gap_chunks:
                chunk_queue.put(
                    (doc_id, input_path, start_gap, end_gap, detected_lang, source_type)
                )
                 
        except Exception as e:
            logger.error("Failed to schedule %s: %s", input_path, e)
            log_failure(input_path, 0, f"SCHEDULER_FAILED: {e}")
            with doc_registry_lock:
                if doc_id in doc_registry and doc_registry[doc_id].custody_chain:
                    doc_registry[doc_id].custody_chain.append_event("processing_failed", {
                        "stage": "scheduler",
                        "error": str(e),
                    })
                if doc_id in doc_registry:
                    del doc_registry[doc_id]

    logger.info("Scheduler finished queuing all tasks.")

def extractor_thread(thread_id):
    """Pulls Chunks, converts to Images, puts in Image Queue."""
    threading.current_thread().name = f"Extractor-{thread_id}"
    logger.debug("Extractor %d started", thread_id)

    while not hard_stop_event.is_set():
        try:
            task = chunk_queue.get(timeout=2)
            doc_id, input_path, start, end, lang_hint, source_type = task
        except queue.Empty:
            continue

        queued_pages = set()
        try:
            process_mode_failed = False
            if EXTRACTOR_MODE == "process" and extractor_process_pool is not None:
                try:
                    payload_pages = extractor_process_pool.submit(
                        _extract_chunk_in_subprocess,
                        input_path, start, end, source_type, DPI, PDF_CONVERSION_THREADS,
                    ).result()
                    for p_num, payload in payload_pages:
                        if p_num > end:
                            break
                        work_task = PageTask(
                            doc_id,
                            input_path,
                            p_num,
                            _decode_image_from_jpeg_bytes(payload),
                            lang_hint,
                            source_type,
                        )
                        image_queue.put(work_task)
                        queued_pages.add(p_num)
                except Exception as proc_err:
                    logger.warning(
                        "Process-mode extraction failed for %s pages %d-%d (%s); "
                        "falling back to thread-mode for this chunk.",
                        doc_id, start, end, proc_err,
                    )
                    process_mode_failed = True

            if EXTRACTOR_MODE == "thread" or process_mode_failed:
                for i, img in enumerate(iter_source_images(input_path, start, end, source_type)):
                    p_num = start + i
                    if p_num > end:
                        break
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
                    )
                    image_queue.put(work_task)
                    queued_pages.add(p_num)

            expected = end - start + 1
            if len(queued_pages) != expected:
                missing = [str(p) for p in range(start, end + 1) if p not in queued_pages]
                raise RuntimeError(
                    f"Incomplete extraction for {source_type} range {start}-{end}; "
                    f"queued={len(queued_pages)}/{expected}; missing={','.join(missing)}"
                )

        except Exception as e:
            logger.error("Extraction failed for %s %d-%d [%s]: %s", doc_id, start, end, source_type, e)
            for p_num in range(start, end + 1):
                if p_num in queued_pages:
                    continue
                assembly_queue.put({
                    "doc_id": doc_id,
                    "page_num": p_num,
                    "text": "",
                    "status": PageStatus.EXTRACT_FAILED,
                    "chunk_path": None,
                    "structure_data": None,
                    "ocr_confidence": 0.0,
                    "ocr_method": PageStatus.EXTRACT_FAILED,
                    "handwriting_data": None,
                    "signature_data": None,
                    "vertical_text_data": None,
                    "table_fallback_data": None,
                })
                log_failure(input_path, p_num, f"EXTRACT_FAILED: {e}")
        finally:
            chunk_queue.task_done()

def worker_thread(worker_id):
    """Consumes PageTasks, runs OCR, saves temp chunks."""
    global global_pages_processed
    threading.current_thread().name = f"GPU-Worker-{worker_id}"
    logger.debug("Worker %d thread started", worker_id)
    logger.info("GPU Worker %d started.", worker_id)
    
    # Engine cache is now module-level (_ENGINE_CACHE, ).
    # Each entry is (engine, inference_lock) or None.

    # Document Intelligence — PPStructure engine (Phase 3A)
    structure_engine = None
    if ENABLE_DOCUMENT_INTELLIGENCE:
        try:
            from paddleocr import PPStructure
            use_table = ENABLE_TABLE_EXTRACTION and DOCINTEL_MODE in ("tables_only", "full")
            use_layout = ENABLE_LAYOUT_ANALYSIS and DOCINTEL_MODE in ("layout_only", "full")
            structure_engine = PPStructure(
                show_log=False,
                use_gpu=True,
                table=use_table,
                layout=use_layout,
                ocr=True,
                image_orientation=True,
            )
            logger.info("Worker %d: PPStructure engine initialized (mode=%s)", worker_id, DOCINTEL_MODE)
        except Exception as e:
            logger.warning("Worker %d: PPStructure init failed, DocIntel disabled: %s", worker_id, e)
            structure_engine = None

    def get_engine(lang_code):
        """Return a PaddleOCR engine for *lang_code* from the shared cache."""
        entry = get_or_create_engine(lang_code, device='gpu')
        if entry is None:
            raise RuntimeError(
                f"No PaddleOCR engine available for {lang_code} "
                f"(primary and English fallback both failed)"
            )
        return entry  # (engine, inference_lock)

    while not hard_stop_event.is_set():
        try:
            task = image_queue.get(timeout=3)
            _pdf_path = task.doc_path
        except queue.Empty:
            continue
            
        _dpi_requeued = False  # Track DPI escalation re-queue to avoid premature task_done()
        try:
            # --- Page Cache: check before OCR ---
            if _page_cache is not None:
                _cache_key = f"{task.doc_id}:{task.page_num}:{DPI}"
                _cached_bytes = _page_cache.get(_cache_key)
                if _cached_bytes is not None:
                    with doc_registry_lock:
                        _cache_doc_state = doc_registry.get(task.doc_id)
                    if _cache_doc_state is not None:
                        _cache_chunk = os.path.join(_cache_doc_state.temp_dir, f"{task.page_num}.pdf")
                        with open(_cache_chunk, "wb") as _cf:
                            _cf.write(_cached_bytes)
                        _cached_meta = _page_cache.get_metadata(_cache_key) or {}
                        assembly_queue.put({
                            "doc_id": task.doc_id,
                            "page_num": task.page_num,
                            "text": "",
                            "status": PageStatus.CACHED,
                            "chunk_path": _cache_chunk,
                            "ocr_confidence": _cached_meta.get("confidence", 0.0),
                            "ocr_method": PageStatus.CACHED,
                            "structure_data": None,
                            "handwriting_data": None,
                            "signature_data": None,
                            "vertical_text_data": None,
                            "table_fallback_data": None,
                        })
                        # task_done/cleanup handled by finally block
                        continue

            # --- OCR LOGIC ---
            # (Paddle -> Tesseract -> Image)
            pdf_bytes = None
            _text_parts: list[str] = []  # O(n) list accumulator instead of O(n²) += concat
            text_content = ""             # Materialized after each OCR pass from _text_parts
            status = PageStatus.OK
            page_confidence = 0.0  # Mean OCR confidence for this page
            ocr_method = ""        # Tracking for validation
            paddle_lines = []      # Populated on Paddle success; empty on fallback paths
            img_np = None          # Numpy array of ocr_img; reused by DocIntel when preprocessing is disabled
            w, h = task.image.size

            # --- Page Routing (opt-in) ---
            routing_target = None
            if _page_router is not None:
                try:
                    features = PageFeatures(
                        page_number=task.page_num,
                        width=w,
                        height=h,
                        dpi=DPI,
                        file_size_bytes=0,
                        estimated_text_density=0.0,
                        has_tables=False,
                        has_images=False,
                        is_handwritten=False,
                        language=task.lang_hint or "",
                        complexity_score=0.0,
                    )
                    decision = _page_router.route_page(features)
                    routing_target = decision.target
                    logger.debug(
                        "Page %d routed to %s: %s",
                        task.page_num, decision.target.value, decision.reason,
                    )
                except Exception as exc:
                    logger.debug("Page routing failed, using default path: %s", exc)
                    routing_target = None  # Fall through to default path

            if routing_target is not None and routing_target.value == "skip":
                logger.info(
                    "Page %d of %s skipped by routing (too small)",
                    task.page_num, task.doc_id,
                )
                assembly_queue.put({
                    "doc_id": task.doc_id,
                    "page_num": task.page_num,
                    "text": "",
                    "status": PageStatus.SKIPPED,
                    "chunk_path": None,
                    "structure_data": None,
                    "ocr_confidence": 0.0,
                    "ocr_method": PageStatus.SKIPPED,
                    "handwriting_data": None,
                    "signature_data": None,
                    "vertical_text_data": None,
                    "table_fallback_data": None,
                })
                # task_done/cleanup handled by finally block
                continue

            # Phase 4D: Optional image preprocessing
            # Use a separate variable so task.image stays pristine for
            # forensic image-only fallback (original pixels are evidence).
            ocr_img = task.image
            if ENABLE_PREPROCESSING:
                try:
                    from preprocessing import preprocess_for_ocr
                    ocr_img = preprocess_for_ocr(task.image, level=PREPROCESSING_LEVEL)
                except Exception as preproc_exc:
                    logger.warning(
                        "Preprocessing failed for %s p%d: %s",
                        task.doc_id, task.page_num, preproc_exc,
                    )

            # [Paddle Attempt]
            temp_doc = None
            try:
                ocr_engine, _infer_lock = get_engine(task.lang_hint)
                if _batch_preprocessor is not None:
                    try:
                        processed = _batch_preprocessor.preprocess_batch([ocr_img])
                        img_np = processed[0] if processed else np.array(ocr_img)
                    except Exception as exc:
                        logger.debug("Batch preprocessing failed, using raw image: %s", exc)
                        img_np = np.array(ocr_img)
                else:
                    img_np = np.array(ocr_img)
                with _infer_lock:
                    result = ocr_engine.ocr(img_np)
                
                # Rebuild Page
                temp_doc = fitz.open()
                page = temp_doc.new_page(width=w, height=h)
                page.insert_image(fitz.Rect(0, 0, w, h), stream=img_to_bytes(task.image))
                
                has_text = False
                paddle_lines = extract_paddle_lines(result)
                if _UNICODE_UTILS_AVAILABLE:
                    paddle_lines = reorder_cjk_vertical_lines(paddle_lines, task.lang_hint)

                for txt, box, _conf in paddle_lines:
                    try:
                        normalized_txt = (
                            normalize_ocr_text(txt, task.lang_hint)
                            if _UNICODE_UTILS_AVAILABLE
                            else txt
                        )
                        if insert_text_line(page, normalized_txt, box, lang_code=task.lang_hint):
                            _text_parts.append(normalized_txt)
                            has_text = True
                    except Exception as insert_error:
                        logger.error("Text insertion error: %s", insert_error)

                # Compute page-level mean confidence from PaddleOCR scores
                paddle_confs = [c for _, _, c in paddle_lines if c > 0]
                page_confidence = sum(paddle_confs) / len(paddle_confs) if paddle_confs else 0.0
                ocr_method = "PaddleOCR"

                # materialize text_content from accumulated parts
                text_content = " ".join(_text_parts)
                if text_content:
                    text_content += " "

                if has_text:
                    # --- ADAPTIVE TWO-PASS LOGIC ---
                    # If FastText detects a language different from the initial hint,
                    # we re-run OCR with the correct PaddleOCR language model. img_np
                    # is reused (no redundant numpy conversion). paddle_lines is NOT
                    # replaced — downstream processing (handwriting, table fallback)
                    # uses first-pass lines intentionally, as the second pass is only
                    # for building a better text layer in the final PDF.
                    detected_code, conf = detect_language_from_text(text_content)
                    
                    # LOG FLAVOR (User Request)
                    if detected_code:
                         logger.info("Page Flavor %s p%d: %s (%.2f)", task.doc_id, task.page_num, detected_code, conf)

                    if detected_code and detected_code != task.lang_hint:
                        logger.info("Re-running %s p%d: Detected %s (was %s)", task.doc_id, task.page_num, detected_code, task.lang_hint)
                        
                        # Re-Run with Correct Language
                        ocr_engine_2, _infer_lock_2 = get_engine(detected_code)
                        with _infer_lock_2:
                            result_2 = ocr_engine_2.ocr(img_np)
                        rerun_lines = extract_paddle_lines(result_2)
                        if _UNICODE_UTILS_AVAILABLE:
                            rerun_lines = reorder_cjk_vertical_lines(rerun_lines, detected_code)
                        if rerun_lines:
                            # guarantee rerun_doc is closed on every exit path
                            # (including exceptions in new_page/insert_image/insert_text_line)
                            # unless ownership is transferred to temp_doc.
                            rerun_doc = fitz.open()
                            _rerun_doc_handed_off = False
                            try:
                                rerun_page = rerun_doc.new_page(width=w, height=h)
                                rerun_page.insert_image(fitz.Rect(0, 0, w, h), stream=img_to_bytes(task.image))

                                _rerun_parts: list[str] = []
                                for txt, box, _conf in rerun_lines:
                                    try:
                                        normalized_txt = (
                                            normalize_ocr_text(txt, detected_code)
                                            if _UNICODE_UTILS_AVAILABLE
                                            else txt
                                        )
                                        if insert_text_line(rerun_page, normalized_txt, box, lang_code=detected_code):
                                            _rerun_parts.append(normalized_txt)
                                    except Exception as insert_error:
                                        logger.error("Re-run text insertion error: %s", insert_error)

                                rerun_text = " ".join(_rerun_parts)
                                if rerun_text:
                                    rerun_text += " "
                                if rerun_text.strip():
                                    temp_doc.close()
                                    temp_doc = rerun_doc
                                    _rerun_doc_handed_off = True
                                    text_content = rerun_text
                                    status = f"Paddle-{detected_code}"
                                    # Update confidence from rerun
                                    rerun_confs = [c for _, _, c in rerun_lines if c > 0]
                                    page_confidence = sum(rerun_confs) / len(rerun_confs) if rerun_confs else 0.0
                            finally:
                                if not _rerun_doc_handed_off:
                                    try:
                                        rerun_doc.close()
                                    except Exception:
                                        pass

                    if not status.startswith("Paddle-"): # If we didn't re-run
                        status = PageStatus.PADDLE
                    pdf_bytes = temp_doc.write()
                    temp_doc.close()
                else:
                    raise Exception("No text")
            except Exception as e:
                # detect CUDA/GPU OOM and attempt recovery before fallback
                _err_str = str(e).lower()
                _is_oom = (
                    "out of memory" in _err_str
                    or "cuda" in type(e).__name__.lower()
                    or "outofmemory" in type(e).__name__.lower()
                    or "cannot allocate" in _err_str
                )
                if _is_oom:
                    logger.error(
                        "GPU OOM on %s p%s — clearing CUDA cache and evicting engine",
                        task.doc_id, task.page_num,
                    )
                    try:
                        import paddle
                        paddle.device.cuda.empty_cache()
                    except Exception as exc:
                        logger.debug("CUDA cache clear failed: %s", exc)
                    # Evict broken engine from shared cache so next request recreates it
                    try:
                        with _ENGINE_CACHE_LOCK:
                            _ENGINE_CACHE.pop(task.lang_hint, None)
                    except Exception as exc:
                        logger.debug("Engine cache eviction failed: %s", exc)
                try:
                    if temp_doc:
                        temp_doc.close()
                except Exception as exc:
                    logger.debug("Failed to close temp_doc after PaddleOCR error: %s", exc)
                logger.warning("PaddleOCR Failed for %s p%d: %s", task.doc_id, task.page_num, e)
                # [Tesseract Attempt]
                try:
                    temp_doc = None
                    data = pytesseract.image_to_data(ocr_img, output_type=pytesseract.Output.DICT, timeout=TESSERACT_TIMEOUT)
                    temp_doc = fitz.open()
                    page = temp_doc.new_page(width=w, height=h)
                    page.insert_image(fitz.Rect(0, 0, w, h), stream=img_to_bytes(task.image))
                    
                    found_tess = False
                    tess_confs = []
                    _tess_parts: list[str] = []
                    for i in range(len(data['text'])):
                         try:
                             conf_val = int(float(data['conf'][i]))
                         except Exception as exc:
                             logger.debug("Invalid Tesseract confidence value at index %d: %s", i, exc)
                             conf_val = -1
                         if conf_val > 0 and data['text'][i].strip():
                             _tess_fn, _tess_ff = _resolve_text_font(task.lang_hint)
                             _tess_fkw = {"fontname": _tess_fn}
                             if _tess_ff:
                                 _tess_fkw["fontfile"] = _tess_ff
                             page.insert_text((data['left'][i], data['top'][i]), data['text'][i], fontsize=12, render_mode=3, **_tess_fkw)
                             _tess_parts.append(data['text'][i])
                             found_tess = True
                             tess_confs.append(conf_val)

                    # materialize text_content from accumulated parts
                    text_content = " ".join(_tess_parts)
                    if text_content:
                        text_content += " "
                    if found_tess:
                        pdf_bytes = temp_doc.write()
                        status = PageStatus.TESSERACT
                        ocr_method = PageStatus.TESSERACT
                        page_confidence = (sum(tess_confs) / len(tess_confs) / 100.0) if tess_confs else 0.0
                    else:
                        raise Exception("No text")
                except Exception as e:
                    logger.warning("Tesseract Failed for %s p%d: %s", task.doc_id, task.page_num, e)
                    # [Fallback - Image Only]
                    # If Paddle/Tesseract fail, we MUST preserve the image in the PDF
                    if not pdf_bytes:
                        try:
                            temp_doc = fitz.open()
                            page = temp_doc.new_page(width=w, height=h)
                            page.insert_image(fitz.Rect(0, 0, w, h), stream=img_to_bytes(task.image))
                            pdf_bytes = temp_doc.write()
                            status = PageStatus.IMAGE_ONLY
                            ocr_method = PageStatus.IMAGE_ONLY
                            page_confidence = 0.0
                            logger.warning("OCR Failed for %s p%d. Saved as Image Only.", task.doc_id, task.page_num)
                            log_failure(task.doc_path, task.page_num, "OCR Text Failure (Image Preserved)")
                        except Exception as inner_e:
                            # If even Image Only fails (e.g. corrupt image stream), raise to outer block
                            raise inner_e
                finally:
                    try:
                        if temp_doc:
                            temp_doc.close()
                    except Exception as exc:
                        logger.debug("Failed to close temp_doc in finally block: %s", exc)

            if not pdf_bytes:
                raise RuntimeError("No PDF bytes generated")

            # --- Document Intelligence (Phase 3A) ---
            structure_data = None
            if structure_engine is not None:
                try:
                    # Reuse OCR numpy array when preprocessing is disabled (same source image).
                    # When preprocessing IS enabled, ocr_img differs from task.image so we must
                    # convert the original image separately for PP-Structure.
                    if img_np is not None and ocr_img is task.image:
                        di_img_np = img_np
                    else:
                        di_img_np = np.array(task.image)
                    structure_result = structure_engine(di_img_np)
                    structure_data = parse_structure_result(structure_result)
                except Exception as e:
                    logger.warning(
                        "Worker-%d: DocIntel failed for %s p%d: %s",
                        worker_id, task.doc_id, task.page_num, e,
                    )

            # --- SAVE TEMP CHUNK ---
            # IMPORTANT: Write to disk immediately to free RAM
            with doc_registry_lock:
                doc_state = doc_registry.get(task.doc_id)
            if not doc_state:
                raise RuntimeError(f"Document state missing for {task.doc_id}")

            chunk_path = os.path.join(doc_state.temp_dir, f"{task.page_num}.pdf")
            with open(chunk_path, "wb") as f:
                f.write(pdf_bytes)

            # --- Page Cache: store after successful OCR ---
            if _page_cache is not None:
                try:
                    _cache_key = f"{task.doc_id}:{task.page_num}:{DPI}"
                    _page_cache.put(
                        _cache_key,
                        pdf_bytes,
                        metadata={"confidence": page_confidence, "status": status},
                    )
                except Exception as exc:
                    logger.debug("Page cache store failed (non-fatal): %s", exc)

            # Custody: record OCR result
            if doc_state and doc_state.custody_chain:
                if status == PageStatus.IMAGE_ONLY:
                    doc_state.custody_chain.append_event("ocr_image_only", {
                        "page_num": task.page_num,
                    })
                elif status == PageStatus.TESSERACT:
                    doc_state.custody_chain.append_event("ocr_fallback", {
                        "page_num": task.page_num,
                        "engine": "Tesseract",
                        "text_length": len(text_content),
                    })
                else:
                    doc_state.custody_chain.append_event("ocr_primary", {
                        "page_num": task.page_num,
                        "engine": status,
                        "lang_hint": task.lang_hint,
                        "text_length": len(text_content),
                    })
                if structure_data is not None:
                    doc_state.custody_chain.append_event("docintel_analysis", {
                        "page_num": task.page_num,
                        "tables_found": len(structure_data.get("tables", [])),
                        "layout_regions": len(structure_data.get("layout_regions", [])),
                    })

            # --- Adaptive DPI Escalation (Phase 4D+) ---
            if (ENABLE_DPI_ESCALATION and _DPI_ESCALATION_AVAILABLE
                    and task.source_type == "pdf"
                    and should_escalate(page_confidence, task.retries, DPI_CONFIDENCE_THRESHOLD)):
                next_idx = task.retries + 1
                next_dpi = DPI_SCHEDULE[next_idx] if next_idx < len(DPI_SCHEDULE) else None
                if next_dpi is not None:
                    logger.info(
                        "Worker-%d: DPI escalation for %s p%d (conf=%.3f < %s, retry %d, DPI %s)",
                        worker_id, task.doc_id, task.page_num,
                        page_confidence, DPI_CONFIDENCE_THRESHOLD,
                        task.retries + 1, next_dpi,
                    )
                    retry_img = re_extract_page_at_dpi(task.doc_path, task.page_num, next_dpi)
                    if retry_img is not None:
                        task.retries += 1
                        task.image = retry_img
                        if doc_state and doc_state.custody_chain:
                            doc_state.custody_chain.append_event("dpi_escalation", {
                                "page_num": task.page_num,
                                "original_dpi": DPI,
                                "retry_dpi": next_dpi,
                                "original_confidence": page_confidence,
                                "retry_number": task.retries,
                            })
                        image_queue.put(task)
                        _dpi_requeued = True  # Skip task_done() in finally
                        continue

            # --- Handwriting Detection (Phase 6B) ---
            handwriting_data = None
            if ENABLE_HANDWRITING and _HANDWRITING_AVAILABLE and paddle_lines:
                try:
                    # Convert polygon boxes to [x1,y1,x2,y2] bboxes for handwriting module
                    hw_lines = []
                    for txt, box, conf in paddle_lines:
                        if box is not None and len(box) >= 4:
                            xs = [pt[0] for pt in box]
                            ys = [pt[1] for pt in box]
                            bbox = [min(xs), min(ys), max(xs), max(ys)]
                        else:
                            bbox = [0, 0, 0, 0]
                        hw_lines.append((txt, conf, bbox))
                    img_size = (task.image.size if hasattr(task.image, "size") else (0, 0))
                    hw_conf = detect_handwriting_by_confidence(
                        hw_lines, task.page_num, img_size,
                    )
                    hw_geom = detect_handwriting_by_geometry(hw_lines, task.page_num)
                    handwriting_data = merge_handwriting_signals(
                        hw_conf, hw_geom, None, task.page_num,
                    )
                except Exception as hw_err:
                    logger.warning(
                        "Worker-%d: Handwriting detection failed for %s p%d: %s",
                        worker_id, task.doc_id, task.page_num, hw_err,
                    )

            # --- Signature Verification Research Path (Phase 6D) ---
            signature_data = None
            if (
                ENABLE_SIGNATURE_VERIFICATION
                and _SIGNATURE_VERIFICATION_AVAILABLE
                and (structure_data is not None or paddle_lines)
            ):
                try:
                    signature_data = analyze_signature_page(
                        task.image,
                        task.page_num,
                        structure_data,
                        paddle_lines,
                    )
                except Exception as sig_err:
                    logger.warning(
                        "Worker-%d: Signature verification failed for %s p%d: %s",
                        worker_id, task.doc_id, task.page_num, sig_err,
                    )

            # --- CJK Vertical Text Analysis (Phase 6D) ---
            vertical_text_data = None
            if ENABLE_VERTICAL_TEXT and _VERTICAL_TEXT_AVAILABLE and paddle_lines:
                try:
                    page_w = task.image.size[0] if hasattr(task.image, "size") else 0
                    vertical_text_data = analyze_page_vertical_text(
                        paddle_lines, task.page_num, page_w, task.lang_hint,
                    )
                except Exception as vt_err:
                    logger.warning(
                        "Worker-%d: Vertical text analysis failed for %s p%d: %s",
                        worker_id, task.doc_id, task.page_num, vt_err,
                    )

            # --- Table Fallback Analysis (Phase 4D) ---
            table_fallback_data = None
            if (
                ENABLE_TABLE_FALLBACK
                and _TABLE_FALLBACK_AVAILABLE
                and structure_data
                and paddle_lines
            ):
                try:
                    table_regions = []
                    for region in structure_data.get("layout_regions", []):
                        if region.get("type") == "table":
                            rb = region.get("bbox", [0, 0, 0, 0])
                            if len(rb) == 4:
                                x1, y1, x2, y2 = rb
                                table_regions.append(TableRegion(
                                    bbox=(x1, y1, x2 - x1, y2 - y1),
                                    page_number=task.page_num,
                                    original_confidence=region.get("confidence", 0.0),
                                    original_engine=ocr_method or "paddle",
                                ))
                    if table_regions:
                        table_fallback_data = analyze_page_tables(
                            task.page_num, table_regions, paddle_lines,
                        )
                except Exception as tf_err:
                    logger.warning(
                        "Worker-%d: Table fallback analysis failed for %s p%d: %s",
                        worker_id, task.doc_id, task.page_num, tf_err,
                    )

            # --- Per-Span Language Detection (Plan A -- PR A3) ---
            # Build per-line SpanLanguage objects and aggregate them into a
            # PageLanguage with character-count-weighted primary detection
            # and mixed-script awareness.  Gracefully degrades to no
            # language data when FastText is not loaded or the feature
            # flag is off.
            language_data = None
            if (
                ENABLE_PER_SPAN_LANGUAGE
                and _LANGUAGE_DETECTION_AVAILABLE
                and paddle_lines
            ):
                try:
                    if lang_model is not None:
                        spans = []
                        for _txt, _box, _conf in paddle_lines:
                            bbox_list = (
                                list(_box)
                                if _box is not None and not isinstance(_box, list)
                                else (list(_box) if _box is not None else [0.0, 0.0, 0.0, 0.0])
                            )
                            span_lang = detect_span_language(
                                text=_txt or "",
                                bbox=bbox_list,
                                fasttext_model=lang_model,
                                short_span_threshold=LANGUAGE_SHORT_SPAN_THRESHOLD,
                                confidence_threshold=LANGUAGE_CONFIDENCE_THRESHOLD,
                            )
                            spans.append(span_lang)
                        language_data = aggregate_page_from_spans(
                            page_num=task.page_num,
                            spans=spans,
                        )
                    # else: fasttext not loaded -- skip silently.
                except Exception as lang_err:
                    logger.warning(
                        "Worker-%d: Span language detection failed for %s p%d: %s",
                        worker_id, task.doc_id, task.page_num, lang_err,
                    )

            # --- Language custody events + Prometheus metrics (Plan A -- PR A4) ---
            # Emit LANGUAGE_DETECTED (and LANGUAGE_MIXED_SCRIPT when applicable)
            # custody events and bump the three language Prometheus counters.
            # All calls are wrapped in try/except: a custody/metrics failure
            # must never block OCR output or page assembly.
            if language_data is not None:
                _model_sha = getattr(lang_model, "_sha256", "") or ""
                try:
                    with doc_registry_lock:
                        _lang_doc_state = doc_registry.get(task.doc_id)
                    if _lang_doc_state is not None and _lang_doc_state.custody_chain:
                        _lang_doc_state.custody_chain.append_event(
                            "LANGUAGE_DETECTED",
                            {
                                "page_num": task.page_num,
                                "primary_language": language_data.primary_language,
                                "primary_confidence": round(
                                    language_data.primary_confidence, 4,
                                ),
                                "languages_detected": list(
                                    language_data.languages_detected,
                                ),
                                "language_char_shares": {
                                    k: round(v, 4)
                                    for k, v in language_data.language_char_shares.items()
                                },
                                "span_count": language_data.span_count,
                                "spans_labeled": language_data.spans_labeled,
                                "detection_engine": "fasttext+script_heuristic",
                                "detector_model_sha256": _model_sha,
                                "tokenizer_sha256": _model_sha,
                                "fasttext_model_sha256": _model_sha,
                            },
                        )
                        if language_data.mixed_script:
                            _script_char_shares: dict[str, int] = {}
                            for _span in language_data.spans:
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
                                        language_data.scripts_detected,
                                    ),
                                    "script_char_shares": {
                                        k: round(v / _total_chars, 4)
                                        for k, v in _script_char_shares.items()
                                    },
                                    "primary_language": language_data.primary_language,
                                    "rtl_present": any(
                                        s.script == "arabic"
                                        for s in language_data.spans
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
                        from ocr_local.config.language_config import LANGUAGE_REGISTRY
                        _entry = LANGUAGE_REGISTRY.get(language_data.primary_language)
                        _tier = _entry.tier if _entry is not None else "core"
                    except Exception:
                        _tier = "core"
                    ocr_language_detected_total.labels(
                        lang=language_data.primary_language,
                        tier=_tier,
                        level="page",
                    ).inc()
                    ocr_language_confidence.labels(
                        lang=language_data.primary_language,
                        level="page",
                    ).observe(language_data.primary_confidence)
                    if (
                        language_data.mixed_script
                        and language_data.scripts_detected
                    ):
                        ocr_language_mixed_script_pages_total.labels(
                            primary_script=language_data.scripts_detected[0],
                        ).inc()
                except Exception:
                    # Prometheus failure must never block OCR output.
                    pass

            # Notify Assembly
            assembly_queue.put({
                "doc_id": task.doc_id,
                "page_num": task.page_num,
                "text": text_content, # Empty if ImageOnly
                "status": status,
                "chunk_path": chunk_path,
                "structure_data": structure_data,
                "ocr_confidence": page_confidence,
                "ocr_method": ocr_method,
                "handwriting_data": handwriting_data,
                "signature_data": signature_data,
                "vertical_text_data": vertical_text_data,
                "table_fallback_data": table_fallback_data,
                "language_data": language_data,
            })

        except Exception as e:
            # CRITICAL FAILURE (Image Processing Failed)
            logger.error("CRITICAL Worker failed on %s p%d: %s", task.doc_id, task.page_num, e)
            log_failure(task.doc_path, task.page_num, f"CRITICAL: Image Failure - {e}")
            with doc_registry_lock:
                doc_state = doc_registry.get(task.doc_id)
            if doc_state and doc_state.custody_chain:
                doc_state.custody_chain.append_event("processing_failed", {
                    "stage": "worker",
                    "page_num": task.page_num,
                    "error": str(e),
                })

            # Preserve evidence via image-only fallback before
            # sending CRITICAL_FAILED.  If the PIL image is still available,
            # create a single-page PDF containing just the image so the
            # assembler can include it rather than leaving a gap.
            _fallback_chunk = None
            if hasattr(task, "image") and task.image is not None and doc_state:
                try:
                    _fb_w, _fb_h = task.image.size
                    _fb_doc = fitz.open()
                    _fb_page = _fb_doc.new_page(width=_fb_w, height=_fb_h)
                    _fb_page.insert_image(
                        fitz.Rect(0, 0, _fb_w, _fb_h),
                        stream=img_to_bytes(task.image),
                    )
                    _fb_bytes = _fb_doc.write()
                    _fb_doc.close()
                    _fallback_chunk = os.path.join(
                        doc_state.temp_dir, f"{task.page_num}.pdf",
                    )
                    with open(_fallback_chunk, "wb") as _fb_f:
                        _fb_f.write(_fb_bytes)
                    logger.warning(
                        "Worker-%d: Image-only fallback saved for %s p%d (evidence preserved)",
                        worker_id, task.doc_id, task.page_num,
                    )
                except Exception as fb_err:
                    logger.error(
                        "Worker-%d: Image-only fallback also failed for %s p%d: %s",
                        worker_id, task.doc_id, task.page_num, fb_err,
                    )
                    _fallback_chunk = None

            # Notify Assembler — include fallback chunk if we saved one
            assembly_queue.put({
                "doc_id": task.doc_id,
                "page_num": task.page_num,
                "text": "",
                "status": PageStatus.CRITICAL_FAILED,
                "chunk_path": _fallback_chunk,
                "structure_data": None,
                "ocr_confidence": 0.0,
                "ocr_method": PageStatus.CRITICAL_FAILED,
                "handwriting_data": None,
                "signature_data": None,
                "vertical_text_data": None,
                "table_fallback_data": None,
                "language_data": None,
            })

        finally:
            if not _dpi_requeued:
                image_queue.task_done()
                # Increment Metric (thread-safe, only when task is truly complete, not re-queued)
                with _pages_processed_lock:
                    global_pages_processed += 1
            if hasattr(task, "image"):
                del task.image  # Force free RAM


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
    try:
        finalized = finalize_fn(doc_obj)
        rel_dir = os.path.dirname(os.path.relpath(doc_path, SOURCE_FOLDER))
        result_path = write_fn(finalized, output_folder, rel_dir, version)
        if result_path:
            logger.info(log_format_fn(finalized, result_path))
        return finalized
    except Exception as exc:
        logger.warning("%s failed for %s: %s", feature_name, doc_id, exc)
        return None


def _process_and_write_routing(
    doc_path: str,
    doc_id: str,
    finalized_classification,
    finalized_entities,
    version: str,
):
    """Derive and write routing sidecars when the feature is enabled."""
    try:
        rel_dir = os.path.dirname(os.path.relpath(doc_path, SOURCE_FOLDER))
        routing_doc = derive_document_routing(
            finalized_classification,
            finalized_entities,
        )
        routing_path = write_routing_json(
            routing_doc,
            OUTPUT_FOLDER,
            rel_dir,
            version,
        )
        if routing_path:
            logger.info(
                "Routing: %s (routes=%d, review_required=%s) -> %s",
                routing_doc.primary_route,
                len(routing_doc.recommended_routes),
                routing_doc.review_required,
                routing_path,
            )
    except Exception as routing_err:
        logger.warning("Routing failed for %s: %s", doc_id, routing_err)


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
    try:
        final_pdf = fitz.open()
        _full_text_parts: list[str] = []  # O(n) list accumulator

        for p in range(1, doc.total_pages + 1):
            chunk_file = os.path.join(doc.temp_dir, f"{(p)}.pdf")
            if os.path.exists(chunk_file):
                try:
                    with fitz.open(chunk_file) as c_doc:
                        final_pdf.insert_pdf(c_doc)
                except Exception as exc:
                    logger.error("Corrupt chunk %d for %s: %s", p, doc_id, exc)
                    log_failure(doc.path, p, "Corrupt Chunk File")
            else:
                logger.error("Missing chunk %d for %s", p, doc_id)
                # If page was never terminal-acknowledged, treat as crash/missing.
                if p not in doc.terminal_pages:
                    log_failure(doc.path, p, "Missing Chunk/Worker Crash")
                else:
                    terminal_status = doc.terminal_statuses.get(p, PageStatus.UNKNOWN)
                    if terminal_status not in (PageStatus.EXTRACT_FAILED, PageStatus.CRITICAL_FAILED):
                        log_failure(
                            doc.path,
                            p,
                            f"Missing chunk after terminal status {terminal_status}",
                        )
            _full_text_parts.append(page_data_snap["texts"].get(p, "") + "\n\f")  # Standard Page Break (Form Feed)

        full_text = "".join(_full_text_parts)

        if final_pdf.page_count == 0:
            placeholder = final_pdf.new_page()
            placeholder.insert_text((36, 36), "No renderable OCR pages. See failures.csv.")

        final_pdf.save(doc.output_pdf, deflate=True)


        # Shared version import for validation, NER, and Phase 6 blocks
        from ocr_local.config.version import __version__ as _p6_version

        # --- Processing Integrity Validation (Phase 5) ---
        if ENABLE_VALIDATION and _VALIDATION_AVAILABLE:
            try:
                # Verify page count
                output_page_count = final_pdf.page_count

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
                        doc_val.pages.append({
                            "page_num": p,
                            "ocr_method": PageStatus.UNKNOWN,
                            "ocr_language": "",
                            "ocr_confidence": 0.0,
                            "text_length": 0,
                            "has_text": False,
                            "status": "unknown",
                        })

                # Compute output PDF hash
                doc_val.output_hash = compute_output_hash(doc.output_pdf)

                # Finalize and classify
                doc_val = finalize_validation(doc_val)

                # Write sidecar JSON
                rel_dir = os.path.dirname(
                    os.path.relpath(doc.path, SOURCE_FOLDER)
                )
                val_path = write_validation_json(
                    doc_val, OUTPUT_FOLDER, rel_dir, _p6_version,
                )
                if val_path:
                    logger.info(
                        "Validation: %s (conf=%.3f) -> %s",
                        doc_val.classification, doc_val.overall_confidence, val_path,
                    )
                if not doc_val.page_count_match:
                    log_failure(
                        doc.path,
                        0,
                        "PAGE_COUNT_MISMATCH: "
                        f"source={doc_val.source_page_count} "
                        f"output={doc_val.output_page_count}",
                    )
                    logger.warning(
                        "Page count mismatch for %s: source=%s output=%s",
                        doc_id, doc_val.source_page_count, doc_val.output_page_count,
                    )
            except Exception as val_err:
                logger.warning("Validation failed for %s: %s", doc_id, val_err)

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
                    os.path.join(OUTPUT_FOLDER, "EXPORT", "NER"),
                    _p6_version,
                    "NER",
                    lambda d, _path: (
                        f"NER: {d.total_entities} entities extracted for {doc_id}"
                    ),
                    doc_id,
                )
            except Exception as ner_err:
                logger.warning("NER extraction failed for %s: %s", doc_id, ner_err)

        # --- Per-page Language Detection Finalization (Plan A -- PR A2) ---
        if ENABLE_PER_SPAN_LANGUAGE and _LANGUAGE_DETECTION_AVAILABLE:
            try:
                page_langs = []
                for p in range(1, doc.total_pages + 1):
                    pl = page_data_snap["language"].get(p)
                    if pl is not None:
                        page_langs.append(pl)
                if page_langs:
                    doc_lang = finalize_document_language(
                        document_id=doc_id,
                        source_file=doc.path,
                        pages=page_langs,
                        fasttext_model_sha256="",
                        tokenizer_sha256="",
                        pipeline_version=_p6_version,
                    )
                    lang_path = write_language_json(
                        doc_lang=doc_lang,
                        output_base_dir=OUTPUT_FOLDER,
                        source_file=os.path.relpath(doc.path, SOURCE_FOLDER)
                        if doc.path.startswith(SOURCE_FOLDER)
                        else doc.path,
                        include_spans=LANGUAGE_INCLUDE_SPANS,
                    )
                    if lang_path:
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
                            _asm_doc_state, "custody_chain", None,
                        )
                        if _asm_custody is not None:
                            _asm_custody.append_event(
                                "LANGUAGE_DETECTED",
                                {
                                    "level": "document",
                                    "primary_language": doc_lang.primary_language,
                                    "primary_confidence": round(
                                        doc_lang.primary_confidence, 4,
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
                                },
                            )
                    except Exception:
                        pass

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
                            _pdf_path = str(doc.output_pdf)
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
                            logger.debug(
                                "XMP language embedding failed for %s: %s",
                                doc_id, xmp_err,
                            )
            except Exception as lang_err:
                logger.warning(
                    "Language finalization failed for %s: %s", doc_id, lang_err,
                )

        # --- Translation sidecar (Plan B Wave M1) ---
        # Opt-in via PipelineConfig.enable_translation; default False.
        # Fail-open: any exception is logged and the OCR job continues
        # uninterrupted -- a translation failure must NEVER fail the OCR
        # job itself.
        try:
            _cfg_active = _ACTIVE_PIPELINE_CONFIG
            if _cfg_active is not None and getattr(
                _cfg_active, "enable_translation", False,
            ):
                _target_langs = list(
                    getattr(_cfg_active, "translation_target_languages", []) or []
                )
                if _target_langs:
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
                        output_dir=OUTPUT_FOLDER,
                        config=_cfg_active,
                    )
                    for _tdoc in _translation_results:
                        try:
                            _write_translation_json(_tdoc, OUTPUT_FOLDER)
                        except Exception as _swe:
                            logger.warning(
                                "Translation sidecar write failed for %s "
                                "(target=%s): %s",
                                doc_id, _tdoc.target_language, _swe,
                            )
        except Exception as _te:
            logger.warning(
                "Translation sidecar failed (non-fatal) for %s: %s",
                doc_id, _te,
            )
        # --- End translation sidecar ---

        # --- Handwriting Detection Finalization (Phase 6B) ---
        if ENABLE_HANDWRITING and _HANDWRITING_AVAILABLE:
            try:
                doc_hw = DocumentHandwriting(
                    document_id=doc_id, source_file=doc.path,
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
                    OUTPUT_FOLDER,
                    _p6_version,
                    "Handwriting",
                    lambda d, path: (
                        f"Handwriting: coverage={d.overall_handwriting_coverage:.2f} "
                        f"primary={d.is_primarily_handwritten} -> {path}"
                    ),
                    doc_id,
                )
            except Exception as hw_err:
                logger.warning("Handwriting detection failed for %s: %s", doc_id, hw_err)

        # --- Signature Verification Finalization (Phase 6D) ---
        if ENABLE_SIGNATURE_VERIFICATION and _SIGNATURE_VERIFICATION_AVAILABLE:
            try:
                doc_sig = DocumentSignatureVerification(
                    document_id=doc_id, source_file=doc.path,
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
                    OUTPUT_FOLDER,
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
                logger.warning("Signature verification failed for %s: %s", doc_id, sig_err)

        # --- CJK Vertical Text Analysis Finalization (Phase 6D) ---
        if ENABLE_VERTICAL_TEXT and _VERTICAL_TEXT_AVAILABLE:
            try:
                doc_vt = DocumentVerticalText(
                    document_id=doc_id, source_file=doc.path,
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
                    OUTPUT_FOLDER,
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
                logger.warning("Vertical text analysis failed for %s: %s", doc_id, vt_err)

        # --- Table Fallback Analysis Finalization (Phase 4D) ---
        if ENABLE_TABLE_FALLBACK and _TABLE_FALLBACK_AVAILABLE:
            try:
                page_analyses = []
                for p in range(1, doc.total_pages + 1):
                    page_tf = page_data_snap["table_fallback"].get(p)
                    if page_tf is not None:
                        page_analyses.append(page_tf)
                if page_analyses:
                    _finalize_and_write_feature(
                        page_analyses,
                        lambda analyses: finalize_table_fallback(
                            analyses,
                            document_id=doc_id,
                            source_file=doc.path,
                        ),
                        write_table_fallback_json,
                        doc.path,
                        OUTPUT_FOLDER,
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
                logger.warning("Table fallback analysis failed for %s: %s", doc_id, tf_err)

        # --- Document Classification (Phase 6A) ---
        finalized_classification = None
        if ENABLE_CLASSIFICATION and _CLASSIFICATION_AVAILABLE:
            try:
                doc_cls = DocumentClassification(
                    document_id=doc_id, source_file=doc.path,
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
                        has_hw = (page_hw.has_handwriting
                                  if hasattr(page_hw, "has_handwriting")
                                  else page_hw.get("has_handwriting", False))
                        if has_hw:
                            page_cls.is_handwritten = True

                    page_data_snap["classification"][p] = page_cls
                    doc_cls.pages.append(page_cls)

                finalized_classification = _finalize_and_write_feature(
                    doc_cls,
                    finalize_classification,
                    write_classification_json,
                    doc.path,
                    OUTPUT_FOLDER,
                    _p6_version,
                    "Classification",
                    lambda d, path: (
                        f"Classification: {d.document_type} "
                        f"(conf={d.document_confidence:.2f}) -> {path}"
                    ),
                    doc_id,
                )
            except Exception as cls_err:
                logger.warning("Classification failed for %s: %s", doc_id, cls_err)

        # --- Structured Extraction (Phase 6C) ---
        finalized_extraction = None
        finalized_entities = None
        if ENABLE_EXTRACTION and _EXTRACTION_AVAILABLE:
            try:
                doc_ext = DocumentExtraction(
                    document_id=doc_id, source_file=doc.path,
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
                    OUTPUT_FOLDER,
                    _p6_version,
                    "Extraction",
                    lambda d, path: (
                        f"Extraction: {d.total_fields} fields "
                        f"({d.extraction_engine}) -> {path}"
                    ),
                    doc_id,
                )
                if finalized_extraction and _ENTITY_OUTPUT_AVAILABLE:
                    finalized_entities = _finalize_and_write_feature(
                        finalized_extraction,
                        finalize_entity_output,
                        write_entities_json,
                        doc.path,
                        OUTPUT_FOLDER,
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
                logger.warning("Extraction failed for %s: %s", doc_id, ext_err)

        # --- Specialist Routing (Phase 6 follow-through) ---
        if (
            ENABLE_SPECIALIST_ROUTING
            and _ROUTING_AVAILABLE
            and finalized_classification is not None
        ):
            _process_and_write_routing(
                doc.path,
                doc_id,
                finalized_classification,
                finalized_entities,
                _p6_version,
            )

        # --- Unified Entity Consolidation ---
        if (
            ENABLE_ENTITY_CONSOLIDATION
            and _ENTITY_CONSOLIDATOR_AVAILABLE
            and (finalized_ner or finalized_extraction or finalized_classification)
        ):
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
                    and consolidated
                ):
                    try:
                        page_texts = page_data_snap["texts"]
                        consolidated = extract_and_attach_relationships(
                            consolidated, page_texts
                        )
                        rel_count = len(consolidated.get("relationships", []))
                        logger.info(
                            "Extracted %d entity relationships for %s",
                            rel_count, doc_id,
                        )
                    except Exception as exc:
                        logger.exception(
                            "Relationship extraction failed for %s: %s", doc_id, exc
                        )
                rel_dir = os.path.dirname(
                    os.path.relpath(doc.path, SOURCE_FOLDER)
                )
                cons_path = write_consolidated_entities_json(
                    consolidated,
                    OUTPUT_FOLDER,
                    rel_dir,
                    doc.path,
                )
                if cons_path:
                    logger.info(
                        "EntityConsolidation: %d entities, %d kv_pairs, "
                        "class=%s -> %s",
                        consolidated["summary"]["total_entities"],
                        consolidated["summary"]["total_kv_pairs"],
                        consolidated["summary"]["primary_classification"],
                        cons_path,
                    )
            except Exception as cons_err:
                logger.warning(
                    "Entity consolidation failed for %s: %s", doc_id, cons_err,
                )

        # Text Output to Separate Folder
        base_name = os.path.splitext(os.path.basename(doc.output_pdf))[0]
        os.makedirs(doc.output_txt_dir, exist_ok=True)
        txt_path = os.path.join(doc.output_txt_dir, base_name + ".txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        # Document Intelligence — write sidecar JSON (Phase 3A)
        if ENABLE_DOCUMENT_INTELLIGENCE and page_data_snap["structure"]:
            doc_structure_data = page_data_snap["structure"]
            if doc_structure_data:
                pages_list = []
                for p in range(1, doc.total_pages + 1):
                    page_data = doc_structure_data.get(p)
                    if page_data:
                        page_data["page_num"] = p
                        pages_list.append(page_data)
                    else:
                        logger.warning(
                            "Page %d structure data missing for %s — using empty placeholder",
                            p, doc.path,
                        )
                        pages_list.append({
                            "page_num": p,
                            "layout_regions": [],
                            "tables": [],
                            "key_value_pairs": [],
                            "form_fields": [],
                        })
                # Phase 3B: Export tables before writing JSON
                # (mutates pages_list to add extracted_files paths)
                if EXPORT_TABLES:
                    t_count = write_extracted_tables(
                        doc_id, doc.path, pages_list, OUTPUT_FOLDER,
                    )
                    if t_count:
                        logger.info("Exported %d tables for %s", t_count, doc.path)

                json_path = write_structure_json(
                    doc_id, doc.path, pages_list, OUTPUT_FOLDER,
                )
                if json_path:
                    logger.info("Structure JSON written: %s", json_path)

        # --- Unified Retrieval Output ---
        if ENABLE_RETRIEVAL_OUTPUT and _OUTPUT_ASSEMBLER_AVAILABLE:
            try:
                # Build text_by_page list from page data snapshot
                _ret_text_by_page = []
                for p in range(1, doc.total_pages + 1):
                    _ret_text_by_page.append({
                        "page": p,
                        "text": page_data_snap["texts"].get(p, ""),
                    })

                # Build entities_data dict from finalized NER (if available)
                _ret_entities_data = None
                if finalized_ner is not None:
                    _ret_entities_data = {
                        "pages": [
                            pg if isinstance(pg, dict) else {"page_num": getattr(pg, "page_num", 0), "entities": getattr(pg, "entities", [])}
                            for pg in getattr(finalized_ner, "pages", [])
                        ],
                    }

                # Build classification_data dict from finalized classification
                _ret_classification_data = None
                if finalized_classification is not None:
                    _ret_classification_data = {
                        "document_type": getattr(finalized_classification, "document_type", ""),
                        "confidence": getattr(finalized_classification, "document_confidence", 0.0),
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
                            _ext_pages.append({
                                "page_num": getattr(pg, "page_num", 0),
                                "fields": getattr(pg, "fields", []),
                            })
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
                if page_data_snap.get("validation"):
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
                            int(vp.get("text_length", 0) or 0)
                            for vp in _val_pages
                        )
                        _text_pages = sum(
                            1 for vp in _val_pages if vp.get("has_text")
                        )
                        _avg_conf = (
                            _total_conf / len(_val_pages)
                            if _val_pages else 0.0
                        )
                        _text_rate = (
                            _text_pages / len(_val_pages)
                            if _val_pages else 0.0
                        )
                        _ret_validation_data = {
                            "page_count": {"source": doc.total_pages},
                            "quality": {
                                "overall_confidence": _avg_conf,
                                "text_extraction_rate": _text_rate,
                                "total_text_length": _total_text_len,
                                "ocr_methods_used": list({
                                    vp.get("ocr_method", "")
                                    for vp in _val_pages
                                    if vp.get("ocr_method")
                                }),
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
                            _hw_pages.append({
                                "page_num": p,
                                "handwriting_regions": (
                                    regions if isinstance(regions, list) else []
                                ),
                            })
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
                rel_dir = os.path.dirname(
                    os.path.relpath(doc.path, SOURCE_FOLDER)
                )
                json_out = write_retrieval_json(ret_doc, OUTPUT_FOLDER, rel_dir)
                md_out = write_retrieval_markdown(ret_doc, OUTPUT_FOLDER, rel_dir)
                if json_out:
                    logger.info(
                        "Retrieval output: %s (%d entities, %d kv_pairs) -> %s",
                        ret_doc.classification.get("label", "unknown"),
                        len(ret_doc.entities),
                        len(ret_doc.key_value_pairs),
                        json_out,
                    )
                if md_out:
                    logger.debug("Retrieval Markdown: %s", md_out)
            except Exception as ret_err:
                logger.warning(
                    "Retrieval output assembly failed for %s: %s",
                    doc_id, ret_err,
                )

        # --- Exception Routing ---
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
                            if _er_val_pages else 0.0
                        )
                        _er_image_only = sum(
                            1 for vp in _er_val_pages
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
                        v for v in page_data_snap["handwriting"].values()
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
                            1 for hw in _er_hw_pages
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
                    # Write routing decision sidecar JSON
                    _er_routing_dir = os.path.join(
                        OUTPUT_FOLDER, "EXPORT", "ROUTING"
                    )
                    os.makedirs(_er_routing_dir, exist_ok=True)
                    _er_routing_path = os.path.join(
                        _er_routing_dir,
                        os.path.splitext(os.path.basename(doc.path))[0]
                        + ".exception-routing.json",
                    )
                    with open(_er_routing_path, "w", encoding="utf-8") as _er_f:
                        json.dump(
                            {
                                "document_id": doc_id,
                                "should_route": True,
                                "triggered_rules": _routing_decision.triggered_rules,
                                "reasons": _routing_decision.reasons,
                                "confidence": _routing_decision.confidence,
                                "metadata": _routing_decision.metadata,
                            },
                            _er_f,
                            indent=2,
                        )
                    logger.info("Exception routing sidecar: %s", _er_routing_path)
                else:
                    logger.debug(
                        "Exception routing: %s passed all rules", doc_id
                    )
            except Exception as _er_err:
                logger.warning(
                    "Exception routing failed for %s: %s", doc_id, _er_err
                )

        # Cleanup per-document temp directory. Opt-out via KEEP_TEMP_FILES=true
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
                _adaptive_batch_sizer.record_result(BatchResult(
                    batch_size=doc.total_pages,
                    pages_processed=doc.processed_pages,
                    duration_seconds=doc_duration,
                    success_count=doc.processed_pages,
                    failure_count=doc.total_pages - doc.processed_pages,
                ))
            except Exception as exc:
                logger.debug("Adaptive batch sizer record failed (non-fatal): %s", exc)

        # Custody: record assembly completion
        custody_chain = doc.custody_chain if hasattr(doc, 'custody_chain') else None
        if custody_chain:
            custody_chain.append_event("assembly_complete", {
                "output_pdf": doc.output_pdf,
                "total_pages": doc.total_pages,
                "processing_seconds": round(time.time() - doc.start_time, 2),
            })

        # Hand off to Compressor (Non-Blocking)
        compression_queue.put((doc.output_pdf, custody_chain))

        # Fix Memory Leak: Remove from registry
        with doc_registry_lock:
            if doc_id in doc_registry:
                del doc_registry[doc_id]


    except Exception as e:
        logger.error("Assembly Failed for %s: %s", doc_id, e)
        log_failure(doc.path, 0, f"ASSEMBLY_FAILED: {e}")
        # Record specific assembly_failed custody event
        # so the failure is auditable in the chain-of-custody log.
        _asm_custody = (
            doc.custody_chain
            if hasattr(doc, 'custody_chain') else None
        )
        if _asm_custody:
            try:
                _asm_custody.append_event("assembly_failed", {
                    "error": str(e),
                    "total_pages": getattr(doc, 'total_pages', 0),
                    "processed_pages": getattr(doc, 'processed_pages', 0),
                })
            except Exception as exc:
                logger.debug("Custody write in assembly failure handler failed: %s", exc)
        # If the output PDF was already saved to disk before the
        # exception, rescue it into the compression queue so the
        # partial result is not silently lost.
        if (
            hasattr(doc, 'output_pdf')
            and doc.output_pdf
            and os.path.isfile(doc.output_pdf)
        ):
            try:
                compression_queue.put((doc.output_pdf, _asm_custody))
                logger.info(
                    "Assembly failed for %s but output PDF rescued to compression queue",
                    doc_id,
                )
            except Exception as exc:
                logger.debug("Best-effort rescue to compression queue failed: %s", exc)
        with doc_registry_lock:
            if doc_id in doc_registry:
                del doc_registry[doc_id]
    finally:
        try:
            if final_pdf:
                final_pdf.close()
        except Exception as exc:
            logger.debug("Failed to close final_pdf in finally block: %s", exc)
        if hasattr(doc, "terminal_statuses"):
            doc.terminal_statuses.clear()


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

    extracted_texts = {} # {doc_id: {page_num: text}}
    structure_pages = {} # {doc_id: {page_num: structure_data}} (Phase 3A)
    validation_pages = {} # {doc_id: {page_num: dict}} (Phase 5 validation)
    validation_page_cache = {} # {doc_id: {page_num: prior validation dict}}
    handwriting_pages = {} # {doc_id: {page_num: PageHandwriting}} (Phase 6B)
    signature_pages = {} # {doc_id: {page_num: PageSignatureVerification}} (Phase 6D)
    vertical_text_pages = {} # {doc_id: {page_num: VerticalTextAnalysis}} (Phase 6D)
    table_fallback_pages = {} # {doc_id: {page_num: PageTableFallbackAnalysis}} (Phase 4D)
    classification_pages = {} # {doc_id: {page_num: PageClassification}} (Phase 6A)
    language_pages = {} # {doc_id: {page_num: PageLanguage}} (Plan A -- PR A2)

    def _cleanup_doc_dicts(doc_id: str) -> None:
        """Remove all per-document page data from assembler-local dicts.

        Called on every terminal path (success finalization AND failure) so
        that long-running deployments never leak memory for abandoned or
        failed documents.
        """
        extracted_texts.pop(doc_id, None)
        structure_pages.pop(doc_id, None)
        validation_pages.pop(doc_id, None)
        validation_page_cache.pop(doc_id, None)
        handwriting_pages.pop(doc_id, None)
        signature_pages.pop(doc_id, None)
        vertical_text_pages.pop(doc_id, None)
        table_fallback_pages.pop(doc_id, None)
        classification_pages.pop(doc_id, None)
        language_pages.pop(doc_id, None)

    try:
        while True:
            try:
                msg: AssemblyMessage = assembly_queue.get(timeout=2)

                # Process Message
                doc_id = msg['doc_id']
                with doc_registry_lock:
                    doc = doc_registry.get(doc_id)
                if not doc:
                    logger.error("ASSEMBLER DROP: ID %s not found", doc_id)
                    _cleanup_doc_dicts(doc_id)  # purge orphaned page data
                    assembly_queue.task_done()
                    continue

                # Init storage if needed
                if doc_id not in extracted_texts:
                    extracted_texts[doc_id] = {}
                if doc_id not in structure_pages:
                    structure_pages[doc_id] = {}
                if doc_id not in validation_pages:
                    validation_pages[doc_id] = {}
                if doc_id not in handwriting_pages:
                    handwriting_pages[doc_id] = {}
                if doc_id not in signature_pages:
                    signature_pages[doc_id] = {}
                if doc_id not in vertical_text_pages:
                    vertical_text_pages[doc_id] = {}
                if doc_id not in table_fallback_pages:
                    table_fallback_pages[doc_id] = {}
                if doc_id not in classification_pages:
                    classification_pages[doc_id] = {}
                if doc_id not in language_pages:
                    language_pages[doc_id] = {}

                page_num = msg.get('page_num')
                status = msg.get('status', PageStatus.UNKNOWN)
                msg_text = msg.get('text', '')
                chunk_path = msg.get("chunk_path")

                # Resume pages carry a pre-rendered chunk; recover embedded text for
                # downstream text/validation parity when message text is empty.
                if status == PageStatus.RESUMED and not msg_text.strip():
                    recovered_text = _extract_text_from_chunk_pdf(chunk_path)
                    if recovered_text:
                        msg_text = recovered_text

                # Collect Document Intelligence data (Phase 3A)
                msg_structure = msg.get('structure_data')
                if msg_structure and page_num not in structure_pages[doc_id]:
                    structure_pages[doc_id][page_num] = msg_structure

                # Collect handwriting detection data (Phase 6B)
                msg_hw = msg.get('handwriting_data')
                if msg_hw and page_num not in handwriting_pages[doc_id]:
                    handwriting_pages[doc_id][page_num] = msg_hw

                # Collect signature verification data (Phase 6D)
                msg_sig = msg.get('signature_data')
                if msg_sig and page_num not in signature_pages[doc_id]:
                    signature_pages[doc_id][page_num] = msg_sig

                # Collect vertical text analysis data (Phase 6D)
                msg_vt = msg.get('vertical_text_data')
                if msg_vt and page_num not in vertical_text_pages[doc_id]:
                    vertical_text_pages[doc_id][page_num] = msg_vt

                # Collect table fallback analysis data (Phase 4D)
                msg_tf = msg.get('table_fallback_data')
                if msg_tf and page_num not in table_fallback_pages[doc_id]:
                    table_fallback_pages[doc_id][page_num] = msg_tf

                # Collect per-page language detection data (Plan A -- PR A2)
                msg_lang = msg.get('language_data')
                if msg_lang is not None and page_num not in language_pages[doc_id]:
                    language_pages[doc_id][page_num] = msg_lang

                # Collect validation data (Phase 5)
                if ENABLE_VALIDATION and _VALIDATION_AVAILABLE and page_num not in validation_pages[doc_id]:
                    prior_page = None
                    if status == PageStatus.RESUMED:
                        if doc_id not in validation_page_cache:
                            validation_page_cache[doc_id] = _load_validation_page_cache(doc.path)
                        prior_page = validation_page_cache[doc_id].get(page_num)

                    # Map pipeline status to validation status (or reuse prior page
                    # metrics for resumed pages when available).
                    if prior_page:
                        val_status = prior_page.get("status", "unknown")
                        if val_status not in {"pending", "ok", "fallback", "image_only", "failed", "unknown"}:
                            val_status = "unknown"
                        ocr_method = prior_page.get("ocr_method", PageStatus.RESUMED)
                        ocr_language = prior_page.get("ocr_language", "")
                        ocr_confidence = float(prior_page.get("ocr_confidence", 0.0) or 0.0)
                        text_length = int(prior_page.get("text_length", len(msg_text)) or 0)
                        has_text = bool(prior_page.get("has_text", bool(msg_text.strip())))
                    else:
                        if status in (PageStatus.OK, PageStatus.PADDLE, PageStatus.TESSERACT) or status.startswith("Paddle-"):
                            val_status = "ok" if status != PageStatus.TESSERACT else "fallback"
                        elif status == PageStatus.CACHED:
                            val_status = "ok"
                        elif status == PageStatus.IMAGE_ONLY:
                            val_status = "image_only"
                        elif status in (PageStatus.EXTRACT_FAILED, PageStatus.CRITICAL_FAILED):
                            val_status = "failed"
                        elif status == PageStatus.RESUMED:
                            val_status = "ok" if msg_text.strip() else "unknown"
                        else:
                            val_status = "unknown"
                        ocr_method = msg.get("ocr_method", PageStatus.RESUMED if status == PageStatus.RESUMED else "")
                        ocr_language = ""
                        ocr_confidence = float(msg.get("ocr_confidence", 0.0) or 0.0)
                        text_length = len(msg_text)
                        has_text = bool(msg_text.strip())
                    validation_pages[doc_id][page_num] = {
                        "page_num": page_num,
                        "ocr_method": ocr_method,
                        "ocr_language": ocr_language,
                        "ocr_confidence": ocr_confidence,
                        "text_length": text_length,
                        "has_text": has_text,
                        "status": val_status,
                    }

                if status in (PageStatus.EXTRACT_FAILED, PageStatus.CRITICAL_FAILED):
                    logger.error("ASSEMBLER TERMINAL PAGE %s p%d: %s", doc_id, page_num, status)
                elif status == PageStatus.RESUMED:
                    logger.info("ASSEMBLER TERMINAL PAGE %s p%d: RESUMED", doc_id, page_num)

                if page_num not in extracted_texts[doc_id]:
                    extracted_texts[doc_id][page_num] = msg_text
                elif msg_text:
                    # Keep last non-empty text if duplicate terminal message arrives.
                    extracted_texts[doc_id][page_num] = msg_text

                if page_num in doc.terminal_statuses and doc.terminal_statuses[page_num] != status:
                    logger.warning(
                        "Terminal status changed for %s p%d: %s -> %s",
                        doc_id, page_num, doc.terminal_statuses[page_num], status,
                    )
                doc.terminal_statuses[page_num] = status

                if page_num not in doc.terminal_pages:
                    doc.terminal_pages.add(page_num)
                else:
                    logger.warning("Duplicate terminal message ignored for %s p%d", doc_id, page_num)

                doc.processed_pages = len(doc.terminal_pages)

                # Check Completion — dispatch finalization to thread pool
                if (not doc.finalized) and doc.total_pages > 0 and doc.processed_pages >= doc.total_pages:
                    doc.finalized = True
                    logger.info("FINALIZING (dispatched): %s", os.path.basename(doc.path))
                    # Snapshot the accumulated page data before cleanup.
                    _snap = {
                        "texts": extracted_texts.get(doc_id, {}),
                        "structure": structure_pages.get(doc_id, {}),
                        "validation": validation_pages.get(doc_id, {}),
                        "handwriting": handwriting_pages.get(doc_id, {}),
                        "signature": signature_pages.get(doc_id, {}),
                        "vertical_text": vertical_text_pages.get(doc_id, {}),
                        "table_fallback": table_fallback_pages.get(doc_id, {}),
                        "classification": classification_pages.get(doc_id, {}),
                        "language": language_pages.get(doc_id, {}),
                    }
                    # single cleanup call removes doc from all 9 local dicts
                    _cleanup_doc_dicts(doc_id)
                    _finalize_executor.submit(_finalize_doc, doc, doc_id, _snap)

                assembly_queue.task_done()

            except queue.Empty:
                if image_queue.empty() and assembly_queue.empty() and stop_event.is_set():
                    # brief drain pause to close empty-check / put() race window
                    time.sleep(0.05)
                    if not assembly_queue.empty():
                        continue  # A message arrived during the pause — keep processing
                    break
                continue
    finally:
        _finalize_executor.shutdown(wait=True)
        # purge any remaining per-doc data on shutdown to avoid leaks
        # if the assembler exits before all documents finalize.
        for stale_id in list(extracted_texts.keys()):
            _cleanup_doc_dicts(stale_id)


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
        for d_id, doc in list(registry_snapshot.items())[:3]: # Limit to first 3 to avoid log spam
             active_status.append(f"[{os.path.basename(doc.path)}: {doc.processed_pages}/{doc.total_pages}]")

        status_str = " | ".join(active_status)
        registry_size = len(registry_snapshot)
        logger.info(
            "STATS: Q_Img=%d Q_Asm=%d Registry=%d | Speed: %.1f PPM (Inst) / %.1f PPM (Avg) | Rate: %.1f Docs/Hr | %s",
            image_queue.qsize(), assembly_queue.qsize(), registry_size,
            ppm_inst, ppm_avg, dph_avg, status_str,
        )

        # Page cache stats (opt-in)
        if _page_cache is not None:
            cache_stats = _page_cache.get_stats()
            logger.info(
                "Cache: hits=%d misses=%d evictions=%d hit_rate=%.1f%%",
                cache_stats.hits, cache_stats.misses,
                cache_stats.evictions, cache_stats.hit_rate * 100,
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
            optimize_pdf(pdf_path, quality="/prepress")
            if custody_chain:
                custody_chain.append_event("compression_complete", {
                    "pdf_path": pdf_path,
                })
        except Exception as e:
            logger.error("Compression logic failed for %s: %s", pdf_path, e)
        finally:
            compression_queue.task_done()

def _join_queue_with_timeout(q: queue.Queue, timeout: float) -> bool:
    """Join a queue with a bounded timeout. Returns True if drained within timeout."""
    t = threading.Thread(target=q.join, daemon=True, name="drain-helper")
    t.start()
    t.join(timeout=timeout)
    return not t.is_alive()


def _graceful_shutdown(signum, frame):
    """Handle SIGTERM/SIGINT for graceful Docker shutdown."""
    sig_name = signal.Signals(signum).name
    logger.warning("Received %s — initiating graceful shutdown...", sig_name)
    stop_event.set()

def _parse_args():
    """Parse command-line arguments for pipeline configuration."""
    import argparse
    parser = argparse.ArgumentParser(
        description="EDCOCR Pipeline (Forensic-Grade)",
    )
    parser.add_argument(
        "--enable-docintel", action="store_true", default=False,
        help="Enable Document Intelligence (PP-StructureV3 layout/table analysis)",
    )
    parser.add_argument(
        "--docintel-mode", choices=["layout_only", "tables_only", "full"],
        default="full",
        help="Document Intelligence mode (default: full)",
    )
    parser.add_argument(
        "--export-tables", action="store_true", default=False,
        help="Export detected tables as HTML/CSV files (requires --enable-docintel)",
    )
    parser.add_argument(
        "--enable-form-detection", action="store_true", default=False,
        help="Enable form field detection (requires --enable-docintel)",
    )
    parser.add_argument(
        "--enable-kv-extraction", action="store_true", default=False,
        help="Enable key-value pair extraction (requires --enable-docintel)",
    )
    parser.add_argument(
        "--enable-privilege-detection", action="store_true", default=False,
        help="Enable legal privilege indicator detection (requires --enable-docintel)",
    )
    parser.add_argument(
        "--no-custody", action="store_true", default=False,
        help="Disable chain-of-custody audit logging",
    )
    parser.add_argument(
        "--enable-preprocessing", action="store_true", default=False,
        help="Enable image preprocessing before OCR (deskew, denoise, etc.)",
    )
    parser.add_argument(
        "--preprocessing-level", choices=["standard", "enhanced", "aggressive"],
        default="standard",
        help="Preprocessing intensity level (default: standard)",
    )
    parser.add_argument(
        "--enable-noise-profiling", action="store_true", default=False,
        help="Enable noise profiling for adaptive preprocessing parameter selection",
    )
    parser.add_argument(
        "--enable-dpi-escalation", action="store_true", default=False,
        help="Enable adaptive DPI escalation for low-confidence pages (300->450->600)",
    )
    parser.add_argument(
        "--enable-ner", action="store_true", default=False,
        help="Enable Named Entity Recognition (spaCy NER + custom regex patterns)",
    )
    parser.add_argument(
        "--enable-handwriting", action="store_true", default=False,
        help="Enable handwriting detection (confidence + geometry heuristics)",
    )
    parser.add_argument(
        "--enable-signature-verification", action="store_true", default=False,
        help=(
            "Enable experimental signature verification sidecars. "
            "This never certifies authenticity and only emits review signals."
        ),
    )
    parser.add_argument(
        "--enable-vertical-text", action="store_true", default=False,
        help="Enable CJK vertical text detection and reading order analysis",
    )
    parser.add_argument(
        "--enable-table-fallback", action="store_true", default=False,
        help="Enable per-region table OCR fallback analysis (requires --enable-docintel)",
    )
    parser.add_argument(
        "--enable-classification", action="store_true", default=False,
        help="Enable document classification (text rules + layout features)",
    )
    parser.add_argument(
        "--enable-extraction", action="store_true", default=False,
        help="Enable structured field extraction (PaddleNLP UIE + regex fallback)",
    )
    parser.add_argument(
        "--enable-specialist-routing", action="store_true", default=False,
        help="Enable specialist routing sidecar output from classification and entity signals",
    )
    parser.add_argument(
        "--enable-entity-consolidation", action="store_true", default=False,
        help="Enable unified entity consolidation (merges NER + extraction + classification into .entities.json)",
    )
    parser.add_argument(
        "--enable-relationship-extraction", action="store_true", default=False,
        help="Enable cross-entity relationship extraction (requires entity consolidation)",
    )
    parser.add_argument(
        "--enable-retrieval-output", action="store_true", default=False,
        help="Enable unified retrieval output (merges OCR text, entities, classification, extraction into .retrieval.json)",
    )
    parser.add_argument(
        "--enable-exception-routing", action="store_true", default=False,
        help="Enable confidence-based exception routing (flags low-quality documents for human review)",
    )
    parser.add_argument(
        "--extractor-mode", choices=["thread", "process", "auto"], default=None,
        help="Override extractor execution mode (env: EXTRACTOR_MODE).",
    )
    parser.add_argument(
        "--extractor-process-workers", type=int, default=None,
        help="Number of process workers when --extractor-mode=process (env: EXTRACTOR_PROCESS_WORKERS).",
    )
    parser.add_argument(
        "--source", default=None,
        help="Override the source folder for API-submitted isolated jobs.",
    )
    parser.add_argument(
        "--output", default=None,
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
        logger.warning("--export-tables requires --enable-docintel; table export disabled.")
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
            logger.warning("--enable-table-fallback requires --enable-docintel; ignored.")
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

    stop_event.clear()
    hard_stop_event.clear()
    finalization_done_event.clear()

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
        logger.info("Adaptive DPI escalation ENABLED (threshold=%s)", DPI_CONFIDENCE_THRESHOLD)
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
        logger.info("Specialist routing ENABLED")
    if ENABLE_ENTITY_CONSOLIDATION:
        logger.info("Unified entity consolidation ENABLED")
    if ENABLE_RELATIONSHIP_EXTRACTION:
        logger.info("Cross-entity relationship extraction ENABLED")
    if ENABLE_RETRIEVAL_OUTPUT:
        if _OUTPUT_ASSEMBLER_AVAILABLE:
            logger.info("Unified retrieval output ENABLED")
        else:
            logger.warning("Retrieval output requested but output_assembler module not available")
    if ENABLE_EXCEPTION_ROUTING:
        if _EXCEPTION_ROUTER_AVAILABLE:
            logger.info("Confidence-based exception routing ENABLED")
        else:
            logger.warning("Exception routing requested but exception_router module not available")
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
        create_paddle_engine('en', device='gpu')
    except Exception as e:
        logger.error("Model Download Warning: %s", e)

    # GPU Optimization probe (opt-in via ENABLE_GPU_OPTIMIZATION)
    if _gpu_optimizer is not None:
        try:
            caps = _gpu_optimizer.detect_capabilities()
            config = _gpu_optimizer.recommend_config(caps)
            global _batch_preprocessor
            _batch_preprocessor = BatchPreprocessor(config)
            logger.info("GPU optimization: %s", _gpu_optimizer.get_optimization_summary())
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

    # Phase 3: log the active config after env parsing + CLI overrides.
    logger.info("Pipeline configuration:\n%s", pipeline_cfg)

    # 1. Start Threads
    logger.debug("Starting threads...")

    t_sched = threading.Thread(target=scheduler_thread)
    t_sched.start()
    
    extractors = []
    for i in range(NUM_EXTRACTORS):
        t = threading.Thread(target=extractor_thread, args=(i+1,))
        t.start()
        extractors.append(t)

    workers = []
    for i in range(NUM_WORKERS):
        t = threading.Thread(target=worker_thread, args=(i+1,))
        t.start()
        workers.append(t)

    t_asm = threading.Thread(target=assembler_thread)
    t_asm.start()

    t_mon = threading.Thread(target=monitor_thread, daemon=True)
    t_mon.start()
    
    compressors = []
    for i in range(NUM_COMPRESSORS):
        t = threading.Thread(target=compressor_thread, args=(i+1,))
        t.start()
        compressors.append(t)
    
    # 2. Drain queues — always drain, even on SIGTERM (with bounded timeout)
    #
    # ORDERING NOTE: The assembler thread now dispatches doc
    # finalization to a ThreadPoolExecutor.  Finalizer threads put items
    # into compression_queue, so compression_queue.join() must come AFTER
    # t_asm.join() (which waits for the executor to shut down).
    t_sched.join()
    if stop_event.is_set():
        logger.info(
            "Shutdown signal received — draining in-flight work "
            "(timeout: %ds per queue)...",
            SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
        )
        for _q, _qname in [
            (chunk_queue, "chunk"),
            (image_queue, "image"),
            (assembly_queue, "assembly"),
        ]:
            _drained = _join_queue_with_timeout(_q, SHUTDOWN_DRAIN_TIMEOUT_SECONDS)
            if not _drained:
                logger.warning(
                    "%s_queue did not fully drain within %ds on shutdown",
                    _qname, SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
                )
        # Wait for assembler + finalization executor before draining compression
        t_asm.join(timeout=SHUTDOWN_DRAIN_TIMEOUT_SECONDS)
        _drained = _join_queue_with_timeout(
            compression_queue, SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
        )
        if not _drained:
            logger.warning(
                "compression_queue did not fully drain within %ds on shutdown",
                SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
            )
    else:
        chunk_queue.join()
        image_queue.join()
        assembly_queue.join()
        stop_event.set()
        # Wait for assembler + finalization executor before draining compression
        t_asm.join()
        finalization_done_event.set()
        compression_queue.join()

    stop_event.set()       # Ensure soft-stop is set (for assembler/compressor termination conditions)
    finalization_done_event.set()
    hard_stop_event.set()  # Signal worker loops to exit


    # 3. Wait for Workers (with timeout to avoid hanging on shutdown)
    for t in workers:
        t.join(timeout=THREAD_JOIN_TIMEOUT)

    for t in extractors:
        t.join(timeout=THREAD_JOIN_TIMEOUT)

    # t_asm already joined above during drain cascade — skip standalone join

    for t in compressors:
        t.join(timeout=THREAD_JOIN_TIMEOUT)

    if extractor_process_pool is not None:
        extractor_process_pool.shutdown(wait=True, cancel_futures=True)
        extractor_process_pool = None
    
    logger.info("=== ALL TASKS COMPLETED ===")

if __name__ == "__main__":
    load_fasttext()
    main()
