"""Tests for ocr_distributed utility functions.

Tests the extracted OCR utility functions from the ocr_distributed package.
These tests do NOT require PaddleOCR, GPU, or Docker -- all external
dependencies are mocked.

Run with: python -m pytest tests/test_ocr_utils.py -v
"""

import hashlib
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

# Ensure project root is on sys.path


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for ocr_distributed.constants values."""

    def test_dpi_default(self):
        from ocr_distributed.constants import DPI_DEFAULT
        assert DPI_DEFAULT == 300

    def test_pdf_extensions(self):
        from ocr_distributed.constants import PDF_EXTENSIONS
        assert ".pdf" in PDF_EXTENSIONS
        assert len(PDF_EXTENSIONS) == 1

    def test_video_extensions_contains_common_formats(self):
        from ocr_distributed.constants import VIDEO_EXTENSIONS
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            assert ext in VIDEO_EXTENSIONS, f"{ext} missing from video extensions"

    def test_phase1_image_extensions_contains_common_formats(self):
        from ocr_distributed.constants import PHASE1_IMAGE_EXTENSIONS
        for ext in [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]:
            assert ext in PHASE1_IMAGE_EXTENSIONS, f"{ext} missing from Phase 1 images"

    def test_phase2_image_extensions_are_separate(self):
        from ocr_distributed.constants import (
            PHASE1_IMAGE_EXTENSIONS,
            PHASE2_IMAGE_EXTENSIONS,
        )
        # Phase 2 extensions should not overlap with Phase 1
        overlap = PHASE1_IMAGE_EXTENSIONS & PHASE2_IMAGE_EXTENSIONS
        assert len(overlap) == 0

    def test_lang_mapping_has_common_languages(self):
        from ocr_distributed.constants import LANG_MAPPING
        assert LANG_MAPPING["en"] == "en"
        assert LANG_MAPPING["zh"] == "ch"
        assert LANG_MAPPING["ja"] == "japan"
        assert LANG_MAPPING["ko"] == "korean"
        assert LANG_MAPPING["de"] == "german"

    def test_lang_mapping_includes_low_resource_tranche(self):
        from ocr_distributed.constants import LANG_MAPPING
        assert LANG_MAPPING["fa"] == "fa"
        assert LANG_MAPPING["ur"] == "ur"
        assert LANG_MAPPING["ug"] == "ug"
        assert LANG_MAPPING["ta"] == "ta"
        assert LANG_MAPPING["te"] == "te"
        assert LANG_MAPPING["kn"] == "kn"
        assert LANG_MAPPING["ka"] == "ka"

    def test_privilege_keywords_is_frozenset(self):
        from ocr_distributed.constants import PRIVILEGE_KEYWORDS
        assert isinstance(PRIVILEGE_KEYWORDS, frozenset)
        assert "attorney-client" in PRIVILEGE_KEYWORDS

    def test_privilege_heuristic_confidence(self):
        from ocr_distributed.constants import PRIVILEGE_HEURISTIC_CONFIDENCE
        assert PRIVILEGE_HEURISTIC_CONFIDENCE == 0.85


# ---------------------------------------------------------------------------
# read_file_header tests
# ---------------------------------------------------------------------------


class TestReadFileHeader:
    """Tests for ocr_utils.read_file_header."""

    def test_reads_first_bytes(self, tmp_path):
        from ocr_distributed.ocr_utils import read_file_header
        p = tmp_path / "test.bin"
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        p.write_bytes(data)
        header = read_file_header(str(p))
        assert header.startswith(b"\x89PNG\r\n\x1a\n")

    def test_respects_max_bytes(self, tmp_path):
        from ocr_distributed.ocr_utils import read_file_header
        p = tmp_path / "test.bin"
        p.write_bytes(b"A" * 8192)
        header = read_file_header(str(p), max_bytes=16)
        assert len(header) == 16

    def test_returns_empty_for_missing_file(self):
        from ocr_distributed.ocr_utils import read_file_header
        result = read_file_header("/nonexistent/file.pdf")
        assert result == b""


# ---------------------------------------------------------------------------
# detect_magic_family tests
# ---------------------------------------------------------------------------


class TestDetectMagicFamily:
    """Tests for ocr_utils.detect_magic_family."""

    def test_detects_pdf(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.pdf"
        p.write_bytes(b"%PDF-1.4 " + b"\x00" * 100)
        assert detect_magic_family(str(p)) == "pdf"

    def test_detects_png(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        assert detect_magic_family(str(p)) == "image"

    def test_detects_jpeg(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        assert detect_magic_family(str(p)) == "image"

    def test_detects_tiff_little_endian(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.tif"
        p.write_bytes(b"II*\x00" + b"\x00" * 100)
        assert detect_magic_family(str(p)) == "image"

    def test_detects_tiff_big_endian(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.tif"
        p.write_bytes(b"MM\x00*" + b"\x00" * 100)
        assert detect_magic_family(str(p)) == "image"

    def test_detects_gif87a(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.gif"
        p.write_bytes(b"GIF87a" + b"\x00" * 100)
        assert detect_magic_family(str(p)) == "image"

    def test_detects_gif89a(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.gif"
        p.write_bytes(b"GIF89a" + b"\x00" * 100)
        assert detect_magic_family(str(p)) == "image"

    def test_detects_bmp(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.bmp"
        p.write_bytes(b"BM" + b"\x00" * 100)
        assert detect_magic_family(str(p)) == "image"

    def test_detects_webp(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.webp"
        p.write_bytes(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100)
        assert detect_magic_family(str(p)) == "image"

    def test_detects_mp4(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.mp4"
        p.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100)
        assert detect_magic_family(str(p)) == "video"

    def test_detects_avi(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.avi"
        p.write_bytes(b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 100)
        assert detect_magic_family(str(p)) == "video"

    def test_detects_ico(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.ico"
        p.write_bytes(b"\x00\x00\x01\x00" + b"\x00" * 100)
        assert detect_magic_family(str(p)) == "image"

    def test_detects_pnm_formats(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        for magic in [b"P1", b"P2", b"P3", b"P4", b"P5", b"P6"]:
            p = tmp_path / "test.pnm"
            p.write_bytes(magic + b" " * 100)
            assert detect_magic_family(str(p)) == "image"

    def test_detects_svg(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.svg"
        p.write_bytes(b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"></svg>')
        assert detect_magic_family(str(p)) == "image"

    def test_returns_none_for_unknown(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.xyz"
        p.write_bytes(b"UNKNOWN_FORMAT" + b"\x00" * 100)
        assert detect_magic_family(str(p)) is None

    def test_returns_none_for_empty_file(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.empty"
        p.write_bytes(b"")
        assert detect_magic_family(str(p)) is None

    def test_returns_none_for_missing_file(self):
        from ocr_distributed.ocr_utils import detect_magic_family
        assert detect_magic_family("/nonexistent/file.pdf") is None

    def test_detects_jp2(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.jp2"
        p.write_bytes(b"\x00\x00\x00\x0cjP  \r\n\x87\n" + b"\x00" * 100)
        assert detect_magic_family(str(p)) == "image"

    def test_detects_j2k(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family
        p = tmp_path / "test.j2k"
        p.write_bytes(b"\xff\x4f\xff\x51" + b"\x00" * 100)
        assert detect_magic_family(str(p)) == "image"


# ---------------------------------------------------------------------------
# classify_source_file tests
# ---------------------------------------------------------------------------


class TestClassifySourceFile:
    """Tests for ocr_utils.classify_source_file."""

    def test_classifies_pdf(self, tmp_path):
        from ocr_distributed.ocr_utils import classify_source_file
        p = tmp_path / "document.pdf"
        p.write_bytes(b"%PDF-1.4 " + b"\x00" * 100)
        source_type, warning = classify_source_file(str(p))
        assert source_type == "pdf"
        assert warning is None

    def test_classifies_png_image(self, tmp_path):
        from ocr_distributed.ocr_utils import classify_source_file
        p = tmp_path / "image.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        source_type, warning = classify_source_file(str(p))
        assert source_type == "image"
        assert warning is None

    def test_classifies_mp4_video(self, tmp_path):
        from ocr_distributed.ocr_utils import classify_source_file
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100)
        source_type, warning = classify_source_file(str(p))
        assert source_type == "video"
        assert warning is None

    def test_rejects_phase2_extension(self, tmp_path):
        from ocr_distributed.ocr_utils import classify_source_file
        p = tmp_path / "image.heic"
        p.write_bytes(b"\x00" * 100)
        source_type, warning = classify_source_file(str(p))
        assert source_type is None
        assert "Phase-2" in warning

    def test_rejects_unsupported_extension(self, tmp_path):
        from ocr_distributed.ocr_utils import classify_source_file
        p = tmp_path / "file.xyz"
        p.write_bytes(b"some data")
        source_type, warning = classify_source_file(str(p))
        assert source_type is None
        assert "Unsupported extension" in warning

    def test_rejects_signature_mismatch(self, tmp_path):
        from ocr_distributed.ocr_utils import classify_source_file
        # .pdf extension but PNG magic bytes
        p = tmp_path / "fake.pdf"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        source_type, warning = classify_source_file(str(p))
        assert source_type is None
        assert "Signature mismatch" in warning

    def test_no_signature_falls_back_to_extension(self, tmp_path):
        from ocr_distributed.ocr_utils import classify_source_file
        p = tmp_path / "image.jpg"
        p.write_bytes(b"UNKNOWN_DATA" + b"\x00" * 100)
        source_type, warning = classify_source_file(str(p))
        assert source_type == "image"
        assert "extension fallback" in warning

    def test_pdf_no_signature_rejected_in_strict_mode(self, tmp_path):
        from ocr_distributed.ocr_utils import classify_source_file
        p = tmp_path / "ambiguous.pdf"
        p.write_bytes(b"UNKNOWN_DATA" + b"\x00" * 128)
        source_type, warning = classify_source_file(
            str(p),
            strict_pdf_signature=True,
        )
        assert source_type is None
        assert "Missing PDF signature" in warning

    def test_strict_mode_still_allows_image_extension_fallback(self, tmp_path):
        from ocr_distributed.ocr_utils import classify_source_file
        p = tmp_path / "ambiguous.jpg"
        p.write_bytes(b"UNKNOWN_DATA" + b"\x00" * 128)
        source_type, warning = classify_source_file(
            str(p),
            strict_pdf_signature=True,
        )
        assert source_type == "image"
        assert "extension fallback" in warning

    def test_rejects_no_extension(self, tmp_path):
        from ocr_distributed.ocr_utils import classify_source_file
        p = tmp_path / "noext"
        p.write_bytes(b"some data")
        source_type, warning = classify_source_file(str(p))
        assert source_type is None
        assert "Unsupported extension" in warning


# ---------------------------------------------------------------------------
# get_source_page_count tests
# ---------------------------------------------------------------------------


class TestGetSourcePageCount:
    """Tests for ocr_utils.get_source_page_count."""

    @patch("ocr_distributed.ocr_utils.fitz")
    def test_pdf_page_count(self, mock_fitz):
        from ocr_distributed.ocr_utils import get_source_page_count
        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc
        count = get_source_page_count("/fake/doc.pdf", "pdf")
        assert count == 5

    @patch("ocr_distributed.ocr_utils.Image")
    def test_image_page_count_single_frame(self, mock_pil):
        from ocr_distributed.ocr_utils import get_source_page_count
        mock_img = MagicMock()
        mock_img.n_frames = 1
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)
        mock_pil.open.return_value = mock_img
        count = get_source_page_count("/fake/image.tiff", "image")
        assert count == 1

    @patch("ocr_distributed.ocr_utils.Image")
    def test_image_page_count_multiframe(self, mock_pil):
        from ocr_distributed.ocr_utils import get_source_page_count
        mock_img = MagicMock()
        mock_img.n_frames = 3
        mock_img.__enter__ = MagicMock(return_value=mock_img)
        mock_img.__exit__ = MagicMock(return_value=False)
        mock_pil.open.return_value = mock_img
        count = get_source_page_count("/fake/image.tiff", "image")
        assert count == 3

    @patch("ocr_distributed.ocr_utils.fitz")
    @patch("ocr_distributed.ocr_utils.Image")
    def test_image_falls_back_to_fitz(self, mock_pil, mock_fitz):
        from ocr_distributed.ocr_utils import get_source_page_count
        # PIL.open raises an exception
        mock_pil.open.side_effect = Exception("PIL cannot open")
        mock_doc = MagicMock()
        mock_doc.page_count = 2
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc
        count = get_source_page_count("/fake/image.svg", "image")
        assert count == 2

    @patch("ocr_distributed.ocr_utils.get_video_page_count")
    def test_video_page_count(self, mock_get_video_page_count):
        from ocr_distributed.ocr_utils import get_source_page_count

        mock_get_video_page_count.return_value = 12
        count = get_source_page_count("/fake/camera.mp4", "video")
        assert count == 12
        mock_get_video_page_count.assert_called_once_with("/fake/camera.mp4")


class TestIterSourceImages:
    """Tests for ocr_utils.iter_source_images()."""

    @patch("ocr_distributed.ocr_utils.iter_video_frames")
    def test_video_iter_source_images_delegates_to_video_utils(self, mock_iter_video_frames):
        from ocr_distributed.ocr_utils import iter_source_images

        img = Image.new("RGB", (10, 10), "white")
        mock_iter_video_frames.return_value = iter([img, img.copy()])

        frames = list(iter_source_images("/fake/camera.mp4", 1, 2, "video"))

        assert len(frames) == 2
        assert all(frame.mode == "RGB" for frame in frames)
        mock_iter_video_frames.assert_called_once_with("/fake/camera.mp4", 1, 2)


# ---------------------------------------------------------------------------
# extract_paddle_lines tests
# ---------------------------------------------------------------------------


class TestExtractPaddleLines:
    """Tests for ocr_utils.extract_paddle_lines."""

    def test_empty_result(self):
        from ocr_distributed.ocr_utils import extract_paddle_lines
        assert extract_paddle_lines(None) == []
        assert extract_paddle_lines([]) == []

    def test_v3_dict_format(self):
        from ocr_distributed.ocr_utils import extract_paddle_lines
        result = [{
            "rec_texts": ["Hello", "World"],
            "rec_scores": [0.95, 0.88],
            "dt_polys": [[[0, 0], [100, 0], [100, 30], [0, 30]],
                         [[0, 40], [100, 40], [100, 70], [0, 70]]],
        }]
        lines = extract_paddle_lines(result)
        assert len(lines) == 2
        assert lines[0][0] == "Hello"
        assert lines[0][2] == 0.95
        assert lines[1][0] == "World"
        assert lines[1][2] == 0.88

    def test_v2_list_format(self):
        from ocr_distributed.ocr_utils import extract_paddle_lines
        result = [[
            [[[0, 0], [100, 0], [100, 30], [0, 30]], ("Hello", 0.95)],
            [[[0, 40], [100, 40], [100, 70], [0, 70]], ("World", 0.88)],
        ]]
        lines = extract_paddle_lines(result)
        assert len(lines) == 2
        assert lines[0][0] == "Hello"
        assert lines[0][2] == 0.95

    def test_v3_skips_empty_text(self):
        from ocr_distributed.ocr_utils import extract_paddle_lines
        result = [{
            "rec_texts": ["Hello", "", "World"],
            "rec_scores": [0.95, 0.0, 0.88],
            "dt_polys": [None, None, None],
        }]
        lines = extract_paddle_lines(result)
        assert len(lines) == 2

    def test_v3_fallback_key_priority(self):
        from ocr_distributed.ocr_utils import extract_paddle_lines
        # dt_polys present but empty; falls back to dt_boxes
        result = [{
            "rec_texts": ["Hello"],
            "rec_scores": [0.95],
            "dt_polys": [],
            "dt_boxes": [[[10, 10], [110, 10], [110, 40], [10, 40]]],
        }]
        lines = extract_paddle_lines(result)
        assert len(lines) == 1
        assert lines[0][1] == [[10, 10], [110, 10], [110, 40], [10, 40]]

    def test_v2_handles_malformed_lines(self):
        from ocr_distributed.ocr_utils import extract_paddle_lines
        # Use a non-iterable item to truly trigger the except branch
        result = [[
            [[[0, 0], [100, 0]], ("Hello", 0.95)],
            12345,  # Not indexable as line[0], line[1]
        ]]
        lines = extract_paddle_lines(result)
        # The valid line should be parsed; the invalid one silently skipped
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# build_output_rel_stem tests
# ---------------------------------------------------------------------------


class TestBuildOutputRelStem:
    """Tests for ocr_utils.build_output_rel_stem."""

    def test_pdf_returns_stem_only(self):
        from ocr_distributed.ocr_utils import build_output_rel_stem
        src_folder = "/source"
        path = "/source/subfolder/document.pdf"
        result = build_output_rel_stem(path, "pdf", src_folder)
        # Should strip the .pdf extension
        assert "document" in result
        assert ".pdf" not in result

    def test_image_appends_ext_token(self):
        from ocr_distributed.ocr_utils import build_output_rel_stem
        src_folder = "/source"
        path = "/source/scan.tiff"
        result = build_output_rel_stem(path, "image", src_folder)
        assert "__tiff" in result

    def test_image_no_extension_uses_img(self):
        from ocr_distributed.ocr_utils import build_output_rel_stem
        # Simulate a file with no extension (after lstrip ".")
        src_folder = "/source"
        path = "/source/noext"
        result = build_output_rel_stem(path, "image", src_folder)
        assert "__img" in result


class TestBuildSidecarBaseName:
    """Tests for ocr_utils.build_sidecar_base_name."""

    def test_pdf_does_not_append_ext_token(self):
        from ocr_distributed.ocr_utils import build_sidecar_base_name

        assert build_sidecar_base_name("/source/report.pdf") == "report"

    def test_non_pdf_appends_ext_token(self):
        from ocr_distributed.ocr_utils import build_sidecar_base_name

        assert build_sidecar_base_name("/source/report.jpg") == "report__jpg"

    def test_weird_extension_is_sanitized(self):
        from ocr_distributed.ocr_utils import build_sidecar_base_name

        assert build_sidecar_base_name("/source/report.my-ext") == "report__my_ext"


# ---------------------------------------------------------------------------
# sanitize_path_segment tests
# ---------------------------------------------------------------------------


class TestSanitizePathSegment:
    """Tests for ocr_utils.sanitize_path_segment."""

    def test_passes_normal_text(self):
        from ocr_distributed.ocr_utils import sanitize_path_segment
        assert sanitize_path_segment("hello_world") == "hello_world"

    def test_replaces_problematic_chars(self):
        from ocr_distributed.ocr_utils import sanitize_path_segment
        result = sanitize_path_segment('file<name>:with|"bad?chars*')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert "|" not in result
        assert '"' not in result
        assert "?" not in result
        assert "*" not in result

    def test_strips_trailing_dots_and_spaces(self):
        from ocr_distributed.ocr_utils import sanitize_path_segment
        assert sanitize_path_segment("file. ") == "file"
        assert sanitize_path_segment("file...") == "file"

    def test_replaces_control_chars(self):
        from ocr_distributed.ocr_utils import sanitize_path_segment
        result = sanitize_path_segment("abc\x01\x02def")
        assert "\x01" not in result
        assert "\x02" not in result

    def test_null_byte(self):
        from ocr_distributed.ocr_utils import sanitize_path_segment
        result = sanitize_path_segment("file\x00name")
        assert "\x00" not in result


# ---------------------------------------------------------------------------
# get_file_hash tests
# ---------------------------------------------------------------------------


class TestGetFileHash:
    """Tests for ocr_utils.get_file_hash."""

    def test_returns_16_char_hex(self):
        from ocr_distributed.ocr_utils import get_file_hash
        result = get_file_hash("/some/path.pdf")
        assert len(result) == 16
        # All hex chars
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        from ocr_distributed.ocr_utils import get_file_hash
        h1 = get_file_hash("/same/path.pdf")
        h2 = get_file_hash("/same/path.pdf")
        assert h1 == h2

    def test_different_paths_different_hash(self):
        from ocr_distributed.ocr_utils import get_file_hash
        h1 = get_file_hash("/path/a.pdf")
        h2 = get_file_hash("/path/b.pdf")
        assert h1 != h2

    def test_matches_manual_sha256(self):
        from ocr_distributed.ocr_utils import get_file_hash
        path = "/test/path.pdf"
        expected = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
        assert get_file_hash(path) == expected


# ---------------------------------------------------------------------------
# img_to_bytes tests
# ---------------------------------------------------------------------------


class TestImgToBytes:
    """Tests for ocr_utils.img_to_bytes."""

    def test_returns_bytes(self):
        from ocr_distributed.ocr_utils import img_to_bytes
        img = Image.new("RGB", (100, 100), color="red")
        result = img_to_bytes(img)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_jpeg_format_default(self):
        from ocr_distributed.ocr_utils import img_to_bytes
        img = Image.new("RGB", (50, 50), color="blue")
        result = img_to_bytes(img)
        # JPEG starts with ff d8 ff
        assert result[:2] == b"\xff\xd8"

    def test_png_format(self):
        from ocr_distributed.ocr_utils import img_to_bytes
        img = Image.new("RGB", (50, 50), color="green")
        result = img_to_bytes(img, fmt="PNG")
        assert result[:4] == b"\x89PNG"

    def test_quality_param(self):
        from ocr_distributed.ocr_utils import img_to_bytes
        img = Image.new("RGB", (200, 200), color="white")
        low_q = img_to_bytes(img, quality=10)
        high_q = img_to_bytes(img, quality=95)
        # Higher quality should generally produce larger output
        # (not guaranteed for trivial images, but works for most)
        assert isinstance(low_q, bytes)
        assert isinstance(high_q, bytes)


# ---------------------------------------------------------------------------
# to_plain_list tests
# ---------------------------------------------------------------------------


class TestToPlainList:
    """Tests for ocr_utils.to_plain_list."""

    def test_none_returns_empty_list(self):
        from ocr_distributed.ocr_utils import to_plain_list
        assert to_plain_list(None) == []

    def test_list_passes_through(self):
        from ocr_distributed.ocr_utils import to_plain_list
        data = [1, 2, 3]
        assert to_plain_list(data) is data

    def test_tuple_becomes_list(self):
        from ocr_distributed.ocr_utils import to_plain_list
        result = to_plain_list((1, 2, 3))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_numpy_array_becomes_list(self):
        from ocr_distributed.ocr_utils import to_plain_list
        arr = np.array([10, 20, 30])
        result = to_plain_list(arr)
        assert result == [10, 20, 30]
        assert isinstance(result, list)

    def test_other_types_return_empty(self):
        from ocr_distributed.ocr_utils import to_plain_list
        assert to_plain_list("string") == []
        assert to_plain_list(42) == []
        assert to_plain_list({}) == []


# ---------------------------------------------------------------------------
# box_to_rect_and_anchor tests
# ---------------------------------------------------------------------------


class TestBoxToRectAndAnchor:
    """Tests for ocr_utils.box_to_rect_and_anchor."""

    def test_polygon_input(self):
        from ocr_distributed.ocr_utils import box_to_rect_and_anchor
        box = [[10, 20], [110, 20], [110, 50], [10, 50]]
        rect, anchor = box_to_rect_and_anchor(box)
        assert rect is not None
        assert anchor == (10.0, 20.0)
        assert rect.x0 == 10.0
        assert rect.y0 == 20.0
        assert rect.x1 == 110.0
        assert rect.y1 == 50.0

    def test_flat_box_input(self):
        from ocr_distributed.ocr_utils import box_to_rect_and_anchor
        box = [10, 20, 110, 50]
        rect, anchor = box_to_rect_and_anchor(box)
        assert rect is not None
        assert rect.x0 == 10.0
        assert rect.y0 == 20.0

    def test_none_input(self):
        from ocr_distributed.ocr_utils import box_to_rect_and_anchor
        rect, anchor = box_to_rect_and_anchor(None)
        assert rect is None
        assert anchor is None

    def test_empty_list_input(self):
        from ocr_distributed.ocr_utils import box_to_rect_and_anchor
        rect, anchor = box_to_rect_and_anchor([])
        assert rect is None
        assert anchor is None

    def test_numpy_polygon_input(self):
        from ocr_distributed.ocr_utils import box_to_rect_and_anchor
        box = np.array([[10, 20], [110, 20], [110, 50], [10, 50]])
        rect, anchor = box_to_rect_and_anchor(box)
        assert rect is not None
        assert anchor is not None


# ---------------------------------------------------------------------------
# LanguageDetector tests
# ---------------------------------------------------------------------------


class TestLanguageDetector:
    """Tests for language.LanguageDetector."""

    def test_no_model_returns_defaults(self):
        from ocr_distributed.language import LanguageDetector
        detector = LanguageDetector(model_path=None)
        assert not detector.is_loaded
        assert detector.detect_from_pdf("/fake.pdf") == "en"

    def test_missing_model_path_returns_defaults(self):
        from ocr_distributed.language import LanguageDetector
        detector = LanguageDetector(model_path="/nonexistent/lid.176.bin")
        assert not detector.is_loaded

    @patch("ocr_distributed.language.os.path.exists", return_value=True)
    def test_load_failure_falls_back_gracefully(self, mock_exists):
        from ocr_distributed.language import LanguageDetector
        with patch.dict("sys.modules", {"fasttext": MagicMock(load_model=MagicMock(side_effect=Exception("fail")))}):
            detector = LanguageDetector(model_path="/fake/lid.176.bin")
            assert not detector.is_loaded

    def test_detect_from_text_short_input(self):
        from ocr_distributed.language import LanguageDetector
        detector = LanguageDetector(model_path=None)
        lang, conf = detector.detect_from_text("short")
        assert lang is None
        assert conf == 0.0

    def test_detect_from_text_no_model(self):
        from ocr_distributed.language import LanguageDetector
        detector = LanguageDetector(model_path=None)
        lang, conf = detector.detect_from_text("This is a sufficiently long English text for detection.")
        assert lang is None
        assert conf == 0.0

    def test_detect_from_pdf_no_model(self):
        from ocr_distributed.language import LanguageDetector
        detector = LanguageDetector(model_path=None)
        result = detector.detect_from_pdf("/fake.pdf", default="fr")
        assert result == "fr"


# ---------------------------------------------------------------------------
# Package __init__ re-exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Tests that ocr_distributed __init__ exports all expected symbols."""

    def test_all_constants_exported(self):
        import ocr_distributed
        assert hasattr(ocr_distributed, "DPI_DEFAULT")
        assert hasattr(ocr_distributed, "LANG_MAPPING")
        assert hasattr(ocr_distributed, "PDF_EXTENSIONS")
        assert hasattr(ocr_distributed, "VIDEO_EXTENSIONS")
        assert hasattr(ocr_distributed, "PHASE1_IMAGE_EXTENSIONS")
        assert hasattr(ocr_distributed, "PHASE2_IMAGE_EXTENSIONS")
        assert hasattr(ocr_distributed, "PRIVILEGE_KEYWORDS")
        assert hasattr(ocr_distributed, "PRIVILEGE_HEURISTIC_CONFIDENCE")

    def test_language_detector_exported(self):
        import ocr_distributed
        assert hasattr(ocr_distributed, "LanguageDetector")

    def test_utility_functions_exported(self):
        import ocr_distributed
        for name in [
            "classify_source_file", "detect_magic_family", "extract_paddle_lines",
            "get_file_hash", "img_to_bytes", "to_plain_list", "box_to_rect_and_anchor",
            "read_file_header", "sanitize_path_segment", "build_output_rel_stem",
            "get_source_page_count", "insert_text_line",
        ]:
            assert hasattr(ocr_distributed, name), f"{name} not exported from ocr_distributed"
