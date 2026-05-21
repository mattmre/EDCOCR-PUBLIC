"""OCR utility functions extracted from ocr_gpu_async.py for distributed use.

These functions are pure/parameterized and can be used by both the monolithic
pipeline (ocr_gpu_async.py) and distributed Celery tasks.
"""

import hashlib
import io
import logging
import os

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from video_utils import get_video_page_count, iter_video_frames

from .constants import (
    DPI_DEFAULT,
    PDF_EXTENSIONS,
    PHASE1_IMAGE_EXTENSIONS,
    PHASE2_IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)

logger = logging.getLogger(__name__)
STRICT_PDF_SIGNATURE = os.environ.get("OCR_STRICT_PDF_SIGNATURE", "").lower() in (
    "1",
    "true",
    "yes",
)

# ---------------------------------------------------------------------------
# Environment-variable parsing helpers (canonical location)
# ---------------------------------------------------------------------------

_env_logger = logging.getLogger(__name__ + ".env")


def get_env_int(
    env_key: str,
    default: int,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    """Parse an integer from an environment variable with optional bounds clamping.

    Catches ValueError and TypeError, logs a warning on bad input, and always
    returns a value within [min_val, max_val] when bounds are specified.
    """
    raw = os.environ.get(env_key, str(default))
    try:
        value = int(raw)
    except (ValueError, TypeError):
        _env_logger.warning("Invalid %s=%r; using default %s", env_key, raw, default)
        value = default
    if min_val is not None:
        value = max(min_val, value)
    if max_val is not None:
        value = min(max_val, value)
    return value


def get_env_float(
    env_key: str,
    default: float,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Parse a float from an environment variable with optional bounds clamping.

    Catches ValueError and TypeError, logs a warning on bad input, and always
    returns a value within [min_val, max_val] when bounds are specified.
    """
    raw = os.environ.get(env_key, str(default))
    try:
        value = float(raw)
    except (ValueError, TypeError):
        _env_logger.warning("Invalid %s=%r; using default %s", env_key, raw, default)
        value = default
    if min_val is not None:
        value = max(min_val, value)
    if max_val is not None:
        value = min(max_val, value)
    return value


def read_file_header(path, max_bytes=4096):
    """Reads a small leading byte window for signature checks."""
    try:
        with open(path, "rb") as fh:
            return fh.read(max_bytes)
    except Exception:
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


# Text-like extensions accepted by the coordinator for ingest but not by OCR
# workers (workers only process pdf/image/video).  Used when
# ``include_coordinator_types=True`` is passed to ``classify_source_file``.
COORDINATOR_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json"}


def classify_source_file(
    path,
    strict_pdf_signature=None,
    include_coordinator_types=False,
):
    """Classify a source file for routing.

    Args:
        path: Path to the file.
        strict_pdf_signature: Whether to require a PDF magic-byte signature.
            Defaults to the ``STRICT_PDF_SIGNATURE`` module constant.
        include_coordinator_types: If ``True``, also accept coordinator-only
            text types (``.txt``, ``.md``, ``.csv``, ``.json``) that are valid
            ingest targets but not OCR worker inputs.  Returns ``"text"`` for
            these extensions.

    Returns:
        ``(source_type_or_none, warning_or_reason_or_none)`` where
        *source_type* is one of ``"pdf"``, ``"image"``, ``"video"``,
        ``"text"`` (coordinator only), or ``None`` for unsupported files.
    """
    if strict_pdf_signature is None:
        strict_pdf_signature = STRICT_PDF_SIGNATURE

    ext = os.path.splitext(path)[1].lower()
    if ext in PHASE2_IMAGE_EXTENSIONS:
        return None, f"Phase-2 extension not yet enabled ({ext})"

    if ext in PDF_EXTENSIONS:
        ext_family = "pdf"
    elif ext in VIDEO_EXTENSIONS:
        ext_family = "video"
    elif ext in PHASE1_IMAGE_EXTENSIONS:
        ext_family = "image"
    elif include_coordinator_types and ext in COORDINATOR_TEXT_EXTENSIONS:
        # Text files are valid coordinator ingest targets (1-page passthrough)
        # but have no magic-byte signature to validate.
        return "text", None
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
    """Returns total page/frame count for pdf, image, text, and video inputs."""
    if source_type == "text":
        return 1
    if source_type == "pdf":
        with fitz.open(path) as doc:
            return int(doc.page_count)
    if source_type == "video":
        return get_video_page_count(path)

    try:
        with Image.open(path) as img:
            return max(1, int(getattr(img, "n_frames", 1) or 1))
    except Exception:
        with fitz.open(path) as doc:
            return max(1, int(doc.page_count))


def iter_pil_image_frames(path, start, end):
    """Yields PIL RGB frames for 1-based inclusive page range."""
    with Image.open(path) as src:
        frame_count = max(1, int(getattr(src, "n_frames", 1) or 1))
        upper = min(end, frame_count)
        for page_num in range(start, upper + 1):
            src.seek(page_num - 1)
            yield src.convert("RGB").copy()


def iter_fitz_image_frames(path, start, end, dpi=DPI_DEFAULT):
    """Fallback renderer for formats Pillow cannot decode directly.

    Args:
        path: Path to the image/document file.
        start: First page number (1-based inclusive).
        end: Last page number (1-based inclusive).
        dpi: Resolution for rendering (default 300).
    """
    zoom = dpi / 72.0
    with fitz.open(path) as doc:
        upper = min(end, int(doc.page_count))
        for page_num in range(start, upper + 1):
            pix = doc[page_num - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            if pix.n == 1:
                yield Image.frombytes("L", (pix.width, pix.height), pix.samples).convert("RGB")
            else:
                yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def iter_source_images(path, start, end, source_type, dpi=DPI_DEFAULT, pdf_threads=1):
    """Yields PIL images for the requested page range.

    Args:
        path: Path to the source file.
        start: First page number (1-based inclusive).
        end: Last page number (1-based inclusive).
        source_type: Either 'pdf', 'image', or 'video'.
        dpi: Resolution for PDF rendering (default 300).
        pdf_threads: Thread count for pdf2image conversion.
    """
    if source_type == "pdf":
        from pdf2image import convert_from_path
        images = convert_from_path(path, first_page=start, last_page=end, dpi=dpi, thread_count=pdf_threads)
        for img in images:
            yield img.convert("RGB")
        return
    if source_type == "video":
        for frame in iter_video_frames(path, start, end):
            yield frame
        return

    try:
        for frame in iter_pil_image_frames(path, start, end):
            yield frame
    except Exception as pil_err:
        logger.warning(f"PIL decode fallback for {path}: {pil_err}")
        for frame in iter_fitz_image_frames(path, start, end, dpi=dpi):
            yield frame


def sanitize_path_segment(segment: str) -> str:
    """Remove or replace characters problematic on Windows/Linux filesystems."""
    sanitized_chars = []
    for ch in segment:
        if ch in '<>:"|?*\x00' or (ord(ch) < 32):
            sanitized_chars.append("_")
        else:
            sanitized_chars.append(ch)
    return "".join(sanitized_chars).strip(". ")


def build_output_rel_stem(path, source_type, source_folder):
    """Build a stable output stem relative to source_folder.

    For non-PDF sources, includes extension token to prevent basename collisions.

    Args:
        path: Absolute path to the source file.
        source_type: Either 'pdf' or 'image'.
        source_folder: Root source directory for computing relative paths.
    """
    rel_path = os.path.relpath(path, source_folder)
    # Sanitize each path component to handle problematic filenames
    parts = rel_path.replace("\\", "/").split("/")
    parts = [sanitize_path_segment(p) for p in parts]
    rel_path = os.path.join(*parts) if parts else rel_path
    rel_stem, rel_ext = os.path.splitext(rel_path)
    if source_type == "pdf":
        return rel_stem

    ext_token = rel_ext.lower().lstrip(".")
    safe_ext = "".join(ch if ch.isalnum() else "_" for ch in ext_token)
    if not safe_ext:
        safe_ext = "img"
    return f"{rel_stem}__{safe_ext}"


def build_sidecar_base_name(source_file: str) -> str:
    """Build a collision-safe basename for sidecar JSON outputs.

    For non-PDF sources, append an extension token (e.g. "__jpg") so files
    with the same basename but different source extensions do not overwrite
    each other.
    """
    base_name = sanitize_path_segment(os.path.splitext(os.path.basename(source_file))[0])
    if not base_name:
        base_name = "document"

    ext = os.path.splitext(source_file)[1].lower().lstrip(".")
    safe_ext = "".join(ch if ch.isalnum() else "_" for ch in ext)
    if safe_ext and safe_ext != "pdf":
        return f"{base_name}__{safe_ext}"
    return base_name


def get_file_hash(path):
    """Generates a stable unique ID for a file path."""
    return hashlib.sha256(path.encode('utf-8')).hexdigest()[:16]


def img_to_bytes(img, fmt='JPEG', quality=85):
    """Convert PIL Image to bytes.

    Args:
        img: PIL Image object.
        fmt: Image format string (default 'JPEG').
        quality: JPEG quality 1-100 (default 85).
    """
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=fmt, quality=quality)
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
            except Exception:
                continue

    return entries


def box_to_rect_and_anchor(box):
    """Builds a rectangle and fallback anchor from polygon/box inputs.

    Returns:
        (fitz.Rect or None, anchor_point_tuple or None)
    """
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
            except Exception:
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
    except Exception:
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
    if lang_code:
        try:
            from font_selector import get_font_path
            font_path = get_font_path(lang_code)
            if font_path is not None:
                return str(font_path.stem), str(font_path)
        except ImportError:
            pass
    return "helv", None


def insert_text_line(page, txt, box, lang_code=None):
    """Attempts bbox placement first, then falls back to anchor insertion.

    When *lang_code* is provided and a matching Noto font file is
    installed, the text is embedded with that font so non-Latin scripts
    render correctly in the searchable PDF text layer.

    Args:
        page: A fitz.Page object.
        txt: Text string to insert.
        box: Bounding box (polygon or flat coordinates).
        lang_code: Optional language code for font selection.

    Returns:
        True if text was inserted, False otherwise.
    """
    fontname, fontfile = _resolve_text_font(lang_code)
    font_kwargs = {"fontname": fontname}
    if fontfile:
        font_kwargs["fontfile"] = fontfile

    rect, anchor = box_to_rect_and_anchor(box)
    if rect and rect.width > 1 and rect.height > 1:
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


def create_paddle_engine(lang_code, device='gpu'):
    """Build a PaddleOCR engine with geometry-preserving defaults.

    Uses OCR_INFERENCE_BACKEND environment variable to select the
    inference backend (paddle, onnx, openvino, auto).  Falls back to
    the native PaddleOCR 2.9.1 constructor if the backend module is
    not available.

    Args:
        lang_code: PaddleOCR language code (e.g. 'en', 'ch', 'japan').
        device: Compute device ('gpu' or 'cpu').
    """
    try:
        from ocr_inference_backend import create_ocr_engine

        return create_ocr_engine(lang_code, device)
    except ImportError:
        pass

    # Fallback if ocr_inference_backend is not available (PaddleOCR 2.9.1 API)
    from paddleocr import PaddleOCR

    return PaddleOCR(
        use_angle_cls=True,
        lang=lang_code,
        use_gpu=(device == "gpu"),
        show_log=False,
    )
