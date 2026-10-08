"""OCR utility functions extracted from ocr_gpu_async.py for distributed use.

These functions are pure/parameterized and can be used by both the monolithic
pipeline (ocr_gpu_async.py) and distributed Celery tasks.
"""

import base64
import hashlib
import io
import json
import logging
import os
import stat
import tempfile
import threading

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from ocr_local.ocr_overlay import insert_hidden_text_in_box
from ocr_local.pdf_geometry import PageGeometry
from ocr_local.raster_source import white_composite_for_ocr
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
# Environment-variable parsing helpers (maintenance review: canonical location)
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
    # B01: PE/ELF never OCR-admitted via extension fallback.
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
        # Coordinator text ingest still refuses PE/ELF. Honest text has no
        # required magic; executables spoofing .txt/.md/.csv/.json must not
        # skip the same denylist used for pdf/image/video.
        magic_family = detect_magic_family(path)
        if magic_family == "executable":
            return None, "Executable binary signature refused (PE/ELF; B01)"
        return "text", None
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


def _safe_expansion_limit(
    name: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: int = 1_000_000,
) -> int:
    """Read a positive expansion limit without allowing malformed env input."""
    return get_env_int(
        name,
        default,
        min_val=minimum,
        max_val=maximum,
    )


def get_source_page_count(path, source_type):
    """Return bounded page/frame count for coordinator and standalone ingest."""
    max_pdf_pages = _safe_expansion_limit("MAX_PDF_PAGES", 5000)
    max_image_pixels = _safe_expansion_limit(
        "MAX_IMAGE_PIXELS", 100_000_000, minimum=0, maximum=2_000_000_000
    )
    if source_type == "text":
        return 1
    if source_type == "pdf":
        with fitz.open(path) as doc:
            count = int(doc.page_count)
            if count > max_pdf_pages:
                raise ValueError(
                    f"PDF has {count} pages which exceeds MAX_PDF_PAGES={max_pdf_pages}"
                )
            return count
    if source_type == "video":
        max_video_frames = _safe_expansion_limit("VIDEO_MAX_FRAMES", 300)
        return get_video_page_count(
            path,
            max_frames=min(max_video_frames, max_pdf_pages),
        )

    try:
        with Image.open(path) as img:
            width, height = img.size
            frames = max(1, int(getattr(img, "n_frames", 1) or 1))
            total_pixels = int(width) * int(height) * frames
            if max_image_pixels > 0 and total_pixels > max_image_pixels:
                raise ValueError(
                    f"Image has {total_pixels} pixels which exceeds "
                    f"MAX_IMAGE_PIXELS={max_image_pixels}"
                )
            if frames > max_pdf_pages:
                raise ValueError(
                    f"Image has {frames} frames which exceeds "
                    f"MAX_PDF_PAGES={max_pdf_pages}"
                )
            return frames
    except OSError:
        with fitz.open(path) as doc:
            count = max(1, int(doc.page_count))
            if count > max_pdf_pages:
                raise ValueError(
                    f"Image document has {count} pages which exceeds "
                    f"MAX_PDF_PAGES={max_pdf_pages}"
                )
            return count


def iter_pil_image_frames(path, start, end):
    """Yields PIL RGB frames for 1-based inclusive page range."""
    with Image.open(path) as src:
        frame_count = max(1, int(getattr(src, "n_frames", 1) or 1))
        upper = min(end, frame_count)
        for page_num in range(start, upper + 1):
            src.seek(page_num - 1)
            yield white_composite_for_ocr(src)


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
            pix = doc[page_num - 1].get_pixmap(
                matrix=fitz.Matrix(zoom, zoom), alpha=False
            )
            if pix.n == 1:
                yield Image.frombytes(
                    "L", (pix.width, pix.height), pix.samples
                ).convert("RGB")
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
        pdf_threads: Retained for caller compatibility. PyMuPDF rendering is
            bounded to the current process and does not start worker threads.
    """
    for raster, geometry in iter_source_pages(path, start, end, source_type, dpi=dpi):
        # Preserve the established PIL-only iterator contract while carrying
        # geometry from the very page/frame that produced this raster.
        raster.info["_ocr_page_geometry"] = geometry
        yield raster


def iter_source_pages(path, start, end, source_type, dpi=DPI_DEFAULT):
    """Yield each raster with its physical page geometry from one open source.

    A caller must not re-open a mutable source solely to recover geometry.
    In particular, PDF geometry comes from the same page that is rendered.
    """
    if source_type == "pdf":
        zoom = dpi / 72.0
        with fitz.open(path) as doc:
            upper = min(end, int(doc.page_count))
            for page_num in range(start, upper + 1):
                page = doc[page_num - 1]
                geometry = PageGeometry.from_pdf_page(page)
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                if pix.n == 1:
                    raster = Image.frombytes(
                        "L", (pix.width, pix.height), pix.samples
                    ).convert("RGB")
                else:
                    raster = Image.frombytes(
                        "RGB", (pix.width, pix.height), pix.samples
                    )
                yield raster, geometry
        return
    if source_type == "video":
        for frame in iter_video_frames(path, start, end):
            yield white_composite_for_ocr(frame), PageGeometry.from_image_pixels(
                *frame.size, dpi=frame.info.get("dpi"), fallback_dpi=dpi
            )
        return
    try:
        with Image.open(path) as src:
            frame_count = max(1, int(getattr(src, "n_frames", 1) or 1))
            for page_num in range(start, min(end, frame_count) + 1):
                src.seek(page_num - 1)
                geometry = PageGeometry.from_image_pixels(
                    *src.size, dpi=src.info.get("dpi"), fallback_dpi=dpi
                )
                yield white_composite_for_ocr(src), geometry
    except OSError as pil_err:
        logger.warning("PIL decode fallback for %s: %s", path, pil_err)
        zoom = dpi / 72.0
        with fitz.open(path) as doc:
            for page_num in range(start, min(end, int(doc.page_count)) + 1):
                page = doc[page_num - 1]
                geometry = PageGeometry.from_pdf_page(page)
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                raster = Image.frombytes(
                    "L" if pix.n == 1 else "RGB",
                    (pix.width, pix.height),
                    pix.samples,
                ).convert("RGB")
                yield raster, geometry


# Characters that cannot appear in portable Windows/Linux path segments.
# Slash/backslash are included so a single segment never smuggles separators.
_UNSAFE_PATH_CHARS = frozenset('<>:"|?*/\\')
# Windows device / reserved basenames (matched case-insensitively on the
# device stem before the first '.').
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
)
# Versioned identity-marker namespace.  Portable pass-through names may never
# contain "~", so a raw name equal to another name's encoded form is always
# identity-encoded (prefix-free).
_PATH_ENCODE_MARKER = "~p1h"
_PATH_SEGMENT_DIGEST_BYTES = 16  # 128 bits of SHA-256 identity
_PATH_SEGMENT_MAX_UTF8_BYTES = 160  # portable bound for every encoded segment
_DEFAULT_FILE_MODE = 0o644


def _utf8_byte_len(value: str) -> int:
    return len(value.encode("utf-8"))


def _truncate_utf8(value: str, max_bytes: int) -> str:
    """Truncate *value* to at most *max_bytes* UTF-8 bytes without partial chars."""
    if max_bytes <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    truncated = encoded[:max_bytes]
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    return ""


def _raw_identity_digest(raw: str) -> str:
    """Return a 128-bit lowercase base32 identity digest of the original raw text.

    Uses the first 128 bits of SHA-256, encoded as lowercase base32 without
    padding (26 characters).  Lowercase base32 is filesystem case-safe on
    Windows while remaining shorter than a 32-char hex form.
    """
    digest = hashlib.sha256(raw.encode("utf-8")).digest()[:_PATH_SEGMENT_DIGEST_BYTES]
    return base64.b32encode(digest).decode("ascii").rstrip("=").lower()


def _compose_identity_segment(slug: str, raw: str) -> str:
    """Join readable slug + versioned marker + digest, bounding UTF-8 length.

    Only the readable slug is truncated; the marker and digest are never cut.
    """
    suffix = f"{_PATH_ENCODE_MARKER}{_raw_identity_digest(raw)}"
    max_slug_bytes = _PATH_SEGMENT_MAX_UTF8_BYTES - _utf8_byte_len(suffix)
    return f"{_truncate_utf8(slug, max_slug_bytes)}{suffix}"


def _is_windows_reserved_name(name: str) -> bool:
    """Return True when *name* collides with a Windows reserved device name."""
    if not name:
        return False
    device = name.split(".", 1)[0]
    return device.casefold() in _WINDOWS_RESERVED_NAMES


def _is_portable_pass_through(segment: str) -> bool:
    """Return True for ordinary lowercase portable ASCII names left unchanged.

    Pass-through names must never contain the identity marker character ``~``,
    must fit the documented UTF-8 byte cap, and must not be Windows-reserved.
    """
    if not segment:
        return False
    if _utf8_byte_len(segment) > _PATH_SEGMENT_MAX_UTF8_BYTES:
        return False
    if "~" in segment:
        return False
    if segment != segment.rstrip("."):
        return False
    if _is_windows_reserved_name(segment):
        return False
    for ch in segment:
        # Strict portable set: lowercase ASCII letters, digits, dot, underscore, hyphen.
        if not (("a" <= ch <= "z") or ("0" <= ch <= "9") or ch in "._-"):
            return False
    return True


def _sanitize_to_slug(raw: str, empty_fallback: str) -> str:
    """Build a readable lowercase ASCII slug (lossy; identity digest is separate)."""
    sanitized_chars: list[str] = []
    for ch in raw:
        code = ord(ch)
        if (
            ch in _UNSAFE_PATH_CHARS
            or ch == "~"
            or ch == "\x00"
            or code < 32
            or code > 126
        ):
            sanitized_chars.append("_")
        else:
            sanitized_chars.append(ch)
    joined = "".join(sanitized_chars)
    stripped = joined.strip(". ")
    lowered = stripped.casefold()
    # Keep only portable ASCII in the slug so truncation is byte-stable.
    ascii_chars: list[str] = []
    for ch in lowered:
        if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch in "._-":
            ascii_chars.append(ch)
        else:
            ascii_chars.append("_")
    base = "".join(ascii_chars)
    if not base:
        base = empty_fallback
    if _is_windows_reserved_name(base):
        base = f"_{base}"
    return base


def sanitize_path_segment(segment: str) -> str:
    """Remove or replace characters problematic on Windows/Linux filesystems.

    This is the low-level character cleaner.  Output path builders should use
    :func:`encode_path_segment` instead so lossy normalization cannot collide
    distinct source names onto the same stem.
    """
    sanitized_chars = []
    for ch in segment:
        code = ord(ch)
        if ch in _UNSAFE_PATH_CHARS or ch == "\x00" or code < 32:
            sanitized_chars.append("_")
        else:
            sanitized_chars.append(ch)
    return "".join(sanitized_chars).strip(". ")


def encode_path_segment(segment: str, *, empty_fallback: str = "_") -> str:
    """Encode one path segment with prefix-free, collision-resistant identity.

    Ordinary lowercase portable ASCII names (``[a-z0-9._-]``, no ``~``, within
    the 160-byte UTF-8 cap, not Windows-reserved) are preserved unchanged.

    Every other input is identity-encoded as::

        {readable_slug}~p1h{sha256_base32_26}

    where the digest covers the exact raw segment (128 bits of SHA-256 as
    lowercase base32 without padding).  Only the slug is truncated to keep
    the complete segment within 160 UTF-8 bytes.  Because pass-through names
    can never contain ``~``, a literal raw name equal to another name's
    encoded form always re-encodes to a different result.

    Args:
        segment: Raw source path segment (file stem or directory name).
        empty_fallback: Replacement slug when the sanitized form is empty
            (``"_"`` for directories, ``"document"`` for file stems).
    """
    raw = segment if isinstance(segment, str) else str(segment)
    if _is_portable_pass_through(raw):
        return raw
    slug = _sanitize_to_slug(raw, empty_fallback)
    return _compose_identity_segment(slug, raw)


def _extension_token(ext_body: str) -> tuple[str, bool]:
    """Return ``(token, is_portable)`` for a raw extension body (no leading dot).

    Portable tokens are non-empty lowercase ASCII alphanumerics.  Non-portable
    tokens are sanitized for readability; callers must attach complete-filename
    identity so case / punctuation variants cannot collide.
    """
    if (
        ext_body
        and ext_body == ext_body.lower()
        and ext_body.isascii()
        and ext_body.isalnum()
    ):
        return ext_body, True

    chars: list[str] = []
    for ch in ext_body:
        if ch.isascii() and ch.isalnum():
            chars.append(ch.lower())
        else:
            chars.append("_")
    token = "".join(chars)
    if not token:
        token = "ext"
    return token, False


def _is_token_shaped_portable_stem(stem: str) -> bool:
    """Return True when *stem* collides with the non-PDF ``stem__ext`` namespace.

    Non-PDF portable sources encode as ``{stem}__{ext_token}`` (for example
    ``report.jpg`` → ``report__jpg``).  A portable PDF whose stem already ends
    with ``__`` plus a lowercase alphanumeric extension-like token would pass
    through as that same base and collide across types.  Reserve that shape:
    callers must identity-encode the complete raw PDF basename instead.
    """
    marker = "__"
    idx = stem.rfind(marker)
    if idx < 0:
        return False
    token = stem[idx + len(marker) :]
    if not token:
        return False
    return bool(token == token.lower() and token.isascii() and token.isalnum())


def _build_encoded_file_base(raw_filename: str) -> str:
    """Encode a complete source basename into a collision-safe output file base.

    Shared by :func:`build_output_rel_stem` and :func:`build_sidecar_base_name`
    so PDF / text / validation / sidecar families agree for the same raw source.

    Rules:
    - ``report.pdf`` → ``report`` (portable stem + exact lowercase ``.pdf``)
    - ``report.jpg`` → ``report__jpg`` (portable stem + portable lowercase ext)
    - ``report__jpg.pdf`` is identity-encoded (token-shaped PDF stem reserved)
    - ``report.PDF`` / ``report.JPG`` differ via identity of the full raw name
    - no extension is distinct from a literal ``.img`` extension
    - extension tokens and the complete base honor the 160-byte UTF-8 cap
    - literal encoded-name forms remain prefix-free via the ``~`` marker
    """
    raw = os.path.basename(raw_filename)
    stem, ext = os.path.splitext(raw)
    ext_body = ext[1:] if ext.startswith(".") else ""
    is_pdf = ext_body.casefold() == "pdf"

    if is_pdf:
        # Portable happy path: exact ".pdf" + portable stem that is NOT already
        # in the non-PDF ``stem__ext`` token namespace.
        if (
            ext == ".pdf"
            and _is_portable_pass_through(stem)
            and not _is_token_shaped_portable_stem(stem)
        ):
            return stem
        # Identity over the complete raw basename (case, token-shaped, etc.).
        slug = _sanitize_to_slug(stem, "document")
        return _compose_identity_segment(slug, raw)

    # Non-PDF: append an extension token.  Missing extension is synthetic
    # ``img`` and always identity-encoded so it cannot collide with ``*.img``.
    if not ext_body:
        slug = _sanitize_to_slug(stem, "document")
        return _compose_identity_segment(f"{slug}__img", raw)

    token, token_portable = _extension_token(ext_body)
    if token_portable and ext == f".{ext_body}" and _is_portable_pass_through(stem):
        candidate = f"{stem}__{token}"
        if _utf8_byte_len(candidate) <= _PATH_SEGMENT_MAX_UTF8_BYTES:
            return candidate

    slug = _sanitize_to_slug(stem, "document")
    readable = f"{slug}__{token}"
    return _compose_identity_segment(readable, raw)


def _is_path_lexically_within(path: str, root: str) -> bool:
    """Return True when *path* is nested under *root* using abspath (no symlinks)."""
    try:
        path_abs = os.path.abspath(path)
        root_abs = os.path.abspath(root)
        common = os.path.commonpath([root_abs, path_abs])
    except (OSError, ValueError):
        return False
    return os.path.normcase(common) == os.path.normcase(root_abs)


def _is_path_within_root(path: str, root: str) -> bool:
    """Return True when *path* is the same as or nested under *root* (realpath)."""
    try:
        path_real = os.path.realpath(path)
        root_real = os.path.realpath(root)
        common = os.path.commonpath([root_real, path_real])
    except (OSError, ValueError):
        return False
    return os.path.normcase(common) == os.path.normcase(root_real)


def _extended_fs_path(path: str) -> str:
    """Return a Windows extended-length path when needed; otherwise *path*.

    Identity-encoded segments are long by design (marker + 128-bit digest).
    Nested output trees under pytest or deep source mirrors can exceed the
    legacy 260-character MAX_PATH limit.  The ``\\\\?\\`` prefix allows the
    atomic writer to complete without changing the returned logical path.
    """
    if os.name != "nt":
        return path
    abs_path = os.path.abspath(path)
    if abs_path.startswith("\\\\?\\"):
        return abs_path
    if abs_path.startswith("\\\\"):
        # UNC: \\server\share\... -> \\?\UNC\server\share\...
        return "\\\\?\\UNC\\" + abs_path[2:]
    return "\\\\?\\" + abs_path


def _destination_file_mode(path: str) -> int:
    """Choose POSIX mode for an atomic replace without following leaf symlinks.

    Preserve the mode only for an existing regular file.  Missing destinations
    and non-regular leaves (including symlinks) use ``0o644``.
    """
    try:
        st = os.lstat(_extended_fs_path(path) if os.name == "nt" else path)
    except OSError:
        return _DEFAULT_FILE_MODE
    if stat.S_ISREG(st.st_mode):
        return stat.S_IMODE(st.st_mode)
    return _DEFAULT_FILE_MODE


def _fsync_directory(path: str) -> None:
    """Best-effort directory fsync (no-op on platforms that disallow it)."""
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _apply_temp_file_mode(fd: int, temp_path: str, dest_mode: int) -> None:
    """Apply *dest_mode* to the temp file before ``os.replace``.

    On POSIX, mode application is required: ``fchmod`` is preferred and
    ``chmod`` is used only when ``fchmod`` is unavailable
    (``AttributeError`` / ``NotImplementedError``).  Any ``OSError`` from the
    chosen path is fatal so a write never completes with an unapplied mode.

    On Windows, POSIX mode bits are not meaningful for ordinary files; failures
    are swallowed as an unsupported-operation best effort.
    """
    if os.name == "nt":
        try:
            os.fchmod(fd, dest_mode)
        except (AttributeError, NotImplementedError, OSError):
            try:
                os.chmod(temp_path, dest_mode)
            except (AttributeError, NotImplementedError, OSError):
                pass
        return

    try:
        os.fchmod(fd, dest_mode)
    except (AttributeError, NotImplementedError):
        # fchmod genuinely unsupported on this interpreter/platform.
        os.chmod(temp_path, dest_mode)


def write_text_atomic(
    path: str,
    data: str,
    *,
    allowed_root: str | None = None,
    encoding: str = "utf-8",
) -> str:
    """Atomically write text without following a final-leaf symlink.

    When *allowed_root* is set, performs:

    1. Lexical absolute-root containment **before** any mutation
    2. Realpath containment **before** ``os.makedirs`` (``realpath`` resolves
       existing symlink components even when the final parent tail does not
       yet exist, so a parent symlink escape cannot create outside directories)
    3. Post-create realpath recheck for TOCTOU races

    Destination mode is selected via ``lstat`` without following symlinks:
    existing regular files keep their mode; new files and leaf-symlink
    replacements use ``0o644``.  The mode is applied to the temp file before
    ``os.replace``; on POSIX a mode-application failure is fatal.

    Returns:
        Absolute path of the written leaf (normal form, without Windows
        extended-length prefixes).
    """
    if not isinstance(data, str):
        raise TypeError(f"write_text_atomic expects str data, got {type(data)!r}")

    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path)
    leaf = os.path.basename(abs_path)
    if not leaf or leaf in {".", ".."}:
        raise ValueError(f"invalid output leaf name: {path!r}")
    if os.sep in leaf or (os.altsep and os.altsep in leaf):
        raise ValueError(f"output leaf must be a single path segment: {path!r}")

    if allowed_root is not None:
        # Lexical containment before any mutation.
        if not _is_path_lexically_within(parent, allowed_root):
            raise ValueError(f"output path parent escapes allowed root: {path}")
        # Realpath containment before mkdir.  realpath resolves existing
        # symlink components even when the nonexistent tail remains.
        if not _is_path_within_root(parent, allowed_root):
            raise ValueError(f"output path parent escapes allowed root: {path}")

    fs_parent = _extended_fs_path(parent)
    fs_abs_path = _extended_fs_path(abs_path)
    os.makedirs(fs_parent, exist_ok=True)

    if allowed_root is not None:
        # Post-create realpath recheck for races (parent replaced by symlink).
        if not _is_path_within_root(parent, allowed_root):
            raise ValueError(f"output path parent escapes allowed root: {path}")

    dest_mode = _destination_file_mode(abs_path)

    payload = data.encode(encoding)
    fd = None
    temp_path = None
    try:
        # Keep the temp basename short.  Embedding the full leaf name in the
        # prefix can push deep identity-encoded paths over Windows MAX_PATH
        # even when the final leaf itself is still valid.
        fd, temp_path = tempfile.mkstemp(
            prefix=".ocrw.",
            suffix=".tmp",
            dir=fs_parent,
        )
        # Apply chosen mode to the temp before replace; never inherit symlink mode.
        _apply_temp_file_mode(fd, temp_path, dest_mode)
        with os.fdopen(fd, "wb") as handle:
            fd = None  # ownership transferred to the file object
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # os.replace replaces a symlink node; it does not follow it.
        os.replace(temp_path, fs_abs_path)
        temp_path = None
        _fsync_directory(parent)
        return abs_path
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise


def write_json_atomic(
    path: str,
    payload,
    *,
    allowed_root: str | None = None,
    encoding: str = "utf-8",
    indent: int | None = 2,
    ensure_ascii: bool = False,
    default=str,
    **json_kwargs,
) -> str:
    """Atomically serialize *payload* as JSON using :func:`write_text_atomic`."""
    text = json.dumps(
        payload,
        indent=indent,
        ensure_ascii=ensure_ascii,
        default=default,
        **json_kwargs,
    )
    return write_text_atomic(
        path,
        text,
        allowed_root=allowed_root,
        encoding=encoding,
    )


def resolve_safe_output_subfolder(root: str, subfolder: str) -> str:
    """Resolve a mirrored, collision-safe *subfolder* beneath *root*.

    A lone ``.`` remains the legacy spelling for the output root.  Dot tokens
    anywhere else are rejected instead of being normalized because they do
    not represent source-directory names.  Every real source segment is kept
    via :func:`encode_path_segment` so empty / reserved / lossy names cannot
    collide with distinct siblings — matching :func:`build_output_rel_stem`.

    Raises:
        ValueError: If the subfolder contains a traversal token or resolves
            outside *root*.
    """
    if not subfolder or subfolder == ".":
        return root

    raw_parts = [part for part in subfolder.replace("\\", "/").split("/") if part]
    if any(part in {".", ".."} for part in raw_parts):
        raise ValueError(f"path traversal blocked in output subfolder: {subfolder}")

    safe_parts = [encode_path_segment(part, empty_fallback="_") for part in raw_parts]
    target = os.path.join(root, *safe_parts) if safe_parts else root
    root_real = os.path.realpath(root)
    target_real = os.path.realpath(target)
    try:
        common_root = os.path.commonpath([root_real, target_real])
    except ValueError as exc:
        raise ValueError(f"output subfolder is outside its root: {subfolder}") from exc
    if os.path.normcase(common_root) != os.path.normcase(root_real):
        raise ValueError(f"output subfolder escapes its root: {subfolder}")
    return target


# Production source_type spellings from classify_source_file / pipeline callers.
_ACCEPTED_SOURCE_TYPES = frozenset({"pdf", "image", "video", "text"})


def _filename_ext_body(filename: str) -> str:
    """Return the extension body (no leading dot) for a basename, or empty."""
    _, ext = os.path.splitext(os.path.basename(filename))
    if ext.startswith("."):
        return ext[1:]
    return ""


def _validate_source_type_for_filename(filename: str, source_type) -> None:
    """Fail-closed validation that *source_type* agrees with the raw filename.

    Contract:
    - *source_type* must be one of ``pdf`` / ``image`` / ``video`` / ``text``
      (the spellings emitted by :func:`classify_source_file` and production
      pipeline callers).
    - ``source_type == 'pdf'`` requires a case-insensitive ``.pdf`` extension.
    - Non-PDF source types reject a case-insensitive ``.pdf`` filename.
    - Accepted image / video / text / other non-PDF extensions (and missing
      extensions) remain filename-driven and align with
      :func:`build_sidecar_base_name`.  Missing extension is explicitly allowed
      for non-PDF types (encoded as synthetic identity ``__img``).
    """
    if not isinstance(source_type, str) or source_type not in _ACCEPTED_SOURCE_TYPES:
        raise ValueError(
            f"unsupported source_type {source_type!r}; "
            f"expected one of {sorted(_ACCEPTED_SOURCE_TYPES)}"
        )

    ext_body = _filename_ext_body(filename)
    is_pdf_name = ext_body.casefold() == "pdf"

    if source_type == "pdf":
        if not is_pdf_name:
            raise ValueError(
                f"source_type 'pdf' requires a .pdf filename, got {filename!r}"
            )
        return

    # image / video / text — reject PDF names; everything else is filename-driven.
    if is_pdf_name:
        raise ValueError(
            f"source_type {source_type!r} rejects .pdf filenames, got {filename!r}"
        )


def build_output_rel_stem(path, source_type, source_folder):
    """Build a stable, collision-resistant output stem relative to source_folder.

    Directory segments use :func:`encode_path_segment`.  The file base uses the
    same private complete-filename helper as :func:`build_sidecar_base_name`, so
    PDF and sidecar families agree for the same raw source when *source_type*
    is consistent with the filename.

    Args:
        path: Absolute path to the source file.
        source_type: One of ``'pdf'``, ``'image'``, ``'video'``, or ``'text'``
            (production spellings from :func:`classify_source_file`).  Validated
            fail-closed against the filename: PDF type requires a
            case-insensitive ``.pdf`` extension; non-PDF types reject ``.pdf``.
            Encoding of accepted non-PDF extensions (and no-extension names)
            remains filename-driven and matches :func:`build_sidecar_base_name`.
        source_folder: Root source directory for computing relative paths.

    Raises:
        ValueError: If *source_type* is unsupported or disagrees with the
            filename extension.
    """
    try:
        rel_path = os.path.relpath(path, source_folder)
        if rel_path == ".." or rel_path.startswith(f"..{os.sep}"):
            rel_path = os.path.basename(path)
    except (OSError, ValueError):
        rel_path = os.path.basename(path)

    raw_parts = [part for part in rel_path.replace("\\", "/").split("/") if part]
    if not raw_parts:
        raw_parts = [os.path.basename(path)]

    raw_filename = raw_parts[-1]
    _validate_source_type_for_filename(raw_filename, source_type)

    safe_dirs = [
        encode_path_segment(part, empty_fallback="_") for part in raw_parts[:-1]
    ]
    safe_base = _build_encoded_file_base(raw_filename)
    if safe_dirs:
        return os.path.join(*safe_dirs, safe_base)
    return safe_base


def build_sidecar_base_name(source_file: str) -> str:
    """Build a collision-safe basename for sidecar JSON outputs.

    Delegates to the same complete-filename helper as
    :func:`build_output_rel_stem` so PDF / text / validation / sidecar outputs
    agree for the same raw source.  Non-PDF sources append an extension token
    (for example ``__jpg``); PDF sources keep the encoded stem only, except
    token-shaped portable PDF stems (``report__jpg.pdf``) which are
    identity-encoded to avoid cross-type collisions.
    """
    return _build_encoded_file_base(os.path.basename(source_file))


def get_file_hash(path):
    """Generates a stable unique ID for a file path."""
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


def img_to_bytes(img, fmt="JPEG", quality=85):
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


_TEXT_METRICS_FONT_CACHE = {}
_TEXT_METRICS_FONT_LOCK = threading.Lock()


def _measure_text_unit_width(txt, fontname, fontfile):
    """Measure text at one point with the font used for PDF insertion.

    ``fitz.get_text_length`` supports Base14 names but not a custom
    ``fontfile``.  Use and cache ``fitz.Font`` for custom font metrics so the
    distributed worker matches the font ultimately embedded in the PDF.
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
    """Insert invisible searchable text only inside grounded PDF geometry.

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
    return insert_hidden_text_in_box(
        page, txt, box, fontname=fontname, fontfile=fontfile,
        source_rotation=source_rotation,
    )


def create_paddle_engine(lang_code, device="gpu"):
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

    from ocr_local.config.language_config import get_paddle_runtime_code
    from ocr_local.infra.ocr_inference_backend import (
        get_gpu_device_configuration,
        get_native_threads_per_predictor,
        get_recognizer_batch_size,
    )

    use_gpu, gpu_id = get_gpu_device_configuration(device)
    kwargs = dict(
        use_angle_cls=True,
        lang=get_paddle_runtime_code(lang_code),
        use_gpu=use_gpu,
        show_log=False,
        drop_score=0.0,
    )
    if gpu_id is not None:
        kwargs["gpu_id"] = gpu_id
    if (threads := get_native_threads_per_predictor()) is not None:
        kwargs["cpu_threads"] = threads
    if (batch := get_recognizer_batch_size()) is not None:
        kwargs["rec_batch_num"] = batch
    return PaddleOCR(**kwargs)
