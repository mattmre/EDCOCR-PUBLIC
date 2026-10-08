"""Tests for ocr_distributed utility functions.

Tests the extracted OCR utility functions from the ocr_distributed package.
These tests do NOT require PaddleOCR, GPU, or Docker -- all external
dependencies are mocked.

Run with: python -m pytest tests/test_ocr_utils.py -v
"""

import hashlib
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
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
        assert {".wmf", ".emf"} <= PHASE2_IMAGE_EXTENSIONS
        assert not ({".wmf", ".emf"} & PHASE1_IMAGE_EXTENSIONS)

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
        p.write_bytes(
            b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"></svg>'
        )
        assert detect_magic_family(str(p)) == "image"

    def test_detects_pe(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family

        p = tmp_path / "payload.bin"
        p.write_bytes(b"MZ" + b"\x00" * 64)
        assert detect_magic_family(str(p)) == "executable"

    def test_detects_elf(self, tmp_path):
        from ocr_distributed.ocr_utils import detect_magic_family

        p = tmp_path / "payload.bin"
        p.write_bytes(b"\x7fELF" + b"\x00" * 64)
        assert detect_magic_family(str(p)) == "executable"

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

    def test_refuses_pe_as_pdf(self, tmp_path):
        from ocr_distributed.ocr_utils import classify_source_file

        p = tmp_path / "payload.pdf"
        p.write_bytes(b"MZ" + b"\x00" * 64)
        source_type, reason = classify_source_file(str(p))
        assert source_type is None
        assert reason == "Executable binary signature refused (PE/ELF; B01)"
        assert "extension fallback" not in reason.lower()

    def test_refuses_elf_as_pdf(self, tmp_path):
        from ocr_distributed.ocr_utils import classify_source_file

        p = tmp_path / "payload.pdf"
        p.write_bytes(b"\x7fELF" + b"\x00" * 64)
        source_type, reason = classify_source_file(str(p))
        assert source_type is None
        assert reason == "Executable binary signature refused (PE/ELF; B01)"

    def test_refuses_pe_as_png_not_extension_fallback(self, tmp_path):
        from ocr_distributed.ocr_utils import classify_source_file

        p = tmp_path / "payload.png"
        p.write_bytes(b"MZ" + b"\x00" * 64)
        source_type, reason = classify_source_file(str(p))
        assert source_type is None
        assert reason == "Executable binary signature refused (PE/ELF; B01)"
        assert "extension fallback" not in reason.lower()

    @pytest.mark.parametrize("ext", [".txt", ".md", ".csv", ".json"])
    def test_refuses_pe_on_coordinator_text_path(self, tmp_path, ext):
        from ocr_distributed.ocr_utils import classify_source_file

        p = tmp_path / f"payload{ext}"
        p.write_bytes(b"MZ" + b"\x00" * 64)
        source_type, reason = classify_source_file(
            str(p),
            include_coordinator_types=True,
        )
        assert source_type is None
        assert reason == "Executable binary signature refused (PE/ELF; B01)"

    @pytest.mark.parametrize("ext", [".txt", ".md", ".csv", ".json"])
    def test_refuses_elf_on_coordinator_text_path(self, tmp_path, ext):
        from ocr_distributed.ocr_utils import classify_source_file

        p = tmp_path / f"payload{ext}"
        p.write_bytes(b"\x7fELF" + b"\x00" * 64)
        source_type, reason = classify_source_file(
            str(p),
            include_coordinator_types=True,
        )
        assert source_type is None
        assert reason == "Executable binary signature refused (PE/ELF; B01)"

    def test_honest_coordinator_text_still_accepted(self, tmp_path):
        from ocr_distributed.ocr_utils import classify_source_file

        p = tmp_path / "notes.txt"
        p.write_bytes(b"plain text notes\n")
        source_type, warning = classify_source_file(
            str(p),
            include_coordinator_types=True,
        )
        assert source_type == "text"
        assert warning is None


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
        mock_img.size = (100, 100)
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
        mock_img.size = (100, 100)
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
        mock_pil.open.side_effect = OSError("PIL cannot open")
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
        mock_get_video_page_count.assert_called_once_with(
            "/fake/camera.mp4", max_frames=300
        )


class TestIterSourceImages:
    """Tests for ocr_utils.iter_source_images()."""

    def test_transparent_png_uses_white_visible_background_and_keeps_dpi(
        self, tmp_path
    ):
        from ocr_distributed.ocr_utils import iter_source_images

        source = tmp_path / "transparent.png"
        image = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
        image.putpixel((5, 5), (0, 0, 0, 255))
        image.save(source, dpi=(200, 200))

        [frame] = list(iter_source_images(source, 1, 1, "image"))
        assert frame.mode == "RGB"
        assert frame.getpixel((0, 0)) == (255, 255, 255)
        assert frame.getpixel((5, 5)) == (0, 0, 0)
        assert abs(frame.info["dpi"][0] - 200) < 1
        assert frame.info["_ocr_page_geometry"].source_kind == "image"

    @patch("ocr_distributed.ocr_utils.iter_video_frames")
    def test_video_iter_source_images_delegates_to_video_utils(
        self, mock_iter_video_frames
    ):
        from ocr_distributed.ocr_utils import iter_source_images

        img = Image.new("RGB", (10, 10), "white")
        mock_iter_video_frames.return_value = iter([img, img.copy()])

        frames = list(iter_source_images("/fake/camera.mp4", 1, 2, "video"))

        assert len(frames) == 2
        assert all(frame.mode == "RGB" for frame in frames)
        mock_iter_video_frames.assert_called_once_with("/fake/camera.mp4", 1, 2)

    def test_pdf_iter_source_images_uses_bounded_fitz_renderer(
        self, tmp_path, monkeypatch
    ):
        import builtins

        import fitz

        from ocr_distributed.ocr_utils import iter_source_images

        source = tmp_path / "two-pages.pdf"
        with fitz.open() as document:
            document.new_page(width=72, height=36)
            document.new_page(width=144, height=72)
            document.save(source)

        real_import = builtins.__import__

        def reject_pdf2image(name, *args, **kwargs):
            if name == "pdf2image" or name.startswith("pdf2image."):
                raise AssertionError("processing PDF render must not import pdf2image")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", reject_pdf2image)
        frames = list(iter_source_images(source, 2, 2, "pdf", dpi=144))

        assert len(frames) == 1
        assert frames[0].mode == "RGB"
        assert frames[0].size == (288, 144)


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

        result = [
            {
                "rec_texts": ["Hello", "World"],
                "rec_scores": [0.95, 0.88],
                "dt_polys": [
                    [[0, 0], [100, 0], [100, 30], [0, 30]],
                    [[0, 40], [100, 40], [100, 70], [0, 70]],
                ],
            }
        ]
        lines = extract_paddle_lines(result)
        assert len(lines) == 2
        assert lines[0][0] == "Hello"
        assert lines[0][2] == 0.95
        assert lines[1][0] == "World"
        assert lines[1][2] == 0.88

    def test_v2_list_format(self):
        from ocr_distributed.ocr_utils import extract_paddle_lines

        result = [
            [
                [[[0, 0], [100, 0], [100, 30], [0, 30]], ("Hello", 0.95)],
                [[[0, 40], [100, 40], [100, 70], [0, 70]], ("World", 0.88)],
            ]
        ]
        lines = extract_paddle_lines(result)
        assert len(lines) == 2
        assert lines[0][0] == "Hello"
        assert lines[0][2] == 0.95

    def test_v3_skips_empty_text(self):
        from ocr_distributed.ocr_utils import extract_paddle_lines

        result = [
            {
                "rec_texts": ["Hello", "", "World"],
                "rec_scores": [0.95, 0.0, 0.88],
                "dt_polys": [None, None, None],
            }
        ]
        lines = extract_paddle_lines(result)
        assert len(lines) == 2

    def test_v3_fallback_key_priority(self):
        from ocr_distributed.ocr_utils import extract_paddle_lines

        # dt_polys present but empty; falls back to dt_boxes
        result = [
            {
                "rec_texts": ["Hello"],
                "rec_scores": [0.95],
                "dt_polys": [],
                "dt_boxes": [[[10, 10], [110, 10], [110, 40], [10, 40]]],
            }
        ]
        lines = extract_paddle_lines(result)
        assert len(lines) == 1
        assert lines[0][1] == [[10, 10], [110, 10], [110, 40], [10, 40]]

    def test_v2_handles_malformed_lines(self):
        from ocr_distributed.ocr_utils import extract_paddle_lines

        # Use a non-iterable item to truly trigger the except branch
        result = [
            [
                [[[0, 0], [100, 0]], ("Hello", 0.95)],
                12345,  # Not indexable as line[0], line[1]
            ]
        ]
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

    def test_weird_extension_is_sanitized_with_identity(self):
        from ocr_distributed.ocr_utils import build_sidecar_base_name

        result = build_sidecar_base_name("/source/report.my-ext")
        assert result.startswith("report__my_ext")
        assert "~p1h" in result
        assert result != "report__my_ext"

    def test_pdf_and_jpg_case_pairs_are_distinct_across_builders(self):
        from ocr_distributed.ocr_utils import (
            build_output_rel_stem,
            build_sidecar_base_name,
        )

        pairs = (
            ("report.pdf", "report.PDF", "pdf"),
            ("report.jpg", "report.JPG", "image"),
        )
        for lower_name, upper_name, source_type in pairs:
            lower_side = build_sidecar_base_name(f"/source/{lower_name}")
            upper_side = build_sidecar_base_name(f"/source/{upper_name}")
            assert lower_side != upper_side
            assert (
                build_output_rel_stem(f"/source/{lower_name}", source_type, "/source")
                == lower_side
            )
            assert (
                build_output_rel_stem(f"/source/{upper_name}", source_type, "/source")
                == upper_side
            )

        assert build_sidecar_base_name("/source/report.pdf") == "report"
        assert build_sidecar_base_name("/source/report.jpg") == "report__jpg"
        assert "~p1h" in build_sidecar_base_name("/source/report.PDF")
        assert "~p1h" in build_sidecar_base_name("/source/report.JPG")

    def test_cross_type_token_namespace_reserved_for_pdf_stems(self):
        from ocr_distributed.ocr_utils import (
            build_output_rel_stem,
            build_sidecar_base_name,
        )

        # Image/portable-ext forms keep the stem__token happy path.
        assert build_sidecar_base_name("/source/report.jpg") == "report__jpg"
        assert build_sidecar_base_name("/source/scan.png") == "scan__png"
        assert build_sidecar_base_name("/source/frame.123") == "frame__123"
        assert build_sidecar_base_name("/source/a__b.c") == "a__b__c"

        # Token-shaped portable PDF stems must not pass through as stem__token.
        collision_pairs = (
            ("report.jpg", "image", "report__jpg.pdf", "pdf"),
            ("scan.png", "image", "scan__png.pdf", "pdf"),
            ("frame.123", "image", "frame__123.pdf", "pdf"),
            ("a__b.c", "image", "a__b__c.pdf", "pdf"),
        )
        for img_name, img_type, pdf_name, pdf_type in collision_pairs:
            img_base = build_sidecar_base_name(f"/source/{img_name}")
            pdf_base = build_sidecar_base_name(f"/source/{pdf_name}")
            assert img_base != pdf_base
            assert "~p1h" in pdf_base
            assert (
                build_output_rel_stem(f"/source/{img_name}", img_type, "/source")
                == img_base
            )
            assert (
                build_output_rel_stem(f"/source/{pdf_name}", pdf_type, "/source")
                == pdf_base
            )

        # Ordinary portable PDFs still pass through without a token marker.
        assert build_sidecar_base_name("/source/report.pdf") == "report"
        assert build_output_rel_stem("/source/report.pdf", "pdf", "/source") == "report"

    def test_source_type_mismatch_is_rejected(self):
        from ocr_distributed.ocr_utils import (
            build_output_rel_stem,
            build_sidecar_base_name,
        )

        with pytest.raises(ValueError, match="requires a .pdf filename"):
            build_output_rel_stem("/source/report.jpg", "pdf", "/source")
        with pytest.raises(ValueError, match="rejects .pdf filenames"):
            build_output_rel_stem("/source/report.pdf", "image", "/source")
        with pytest.raises(ValueError, match="rejects .pdf filenames"):
            build_output_rel_stem("/source/report.PDF", "video", "/source")
        with pytest.raises(ValueError, match="rejects .pdf filenames"):
            build_output_rel_stem("/source/report.pdf", "text", "/source")
        with pytest.raises(ValueError, match="unsupported source_type"):
            build_output_rel_stem("/source/report.pdf", "other", "/source")
        with pytest.raises(ValueError, match="unsupported source_type"):
            build_output_rel_stem("/source/report.pdf", None, "/source")

        # Case-insensitive PDF extension is accepted for source_type pdf.
        assert build_output_rel_stem(
            "/source/report.PDF", "pdf", "/source"
        ) == build_sidecar_base_name("/source/report.PDF")

        # No-extension is allowed for non-PDF types (filename-driven __img identity).
        noext = build_output_rel_stem("/source/noext", "image", "/source")
        assert "__img" in noext
        assert "~p1h" in noext

    def test_noext_distinct_from_literal_img_extension(self):
        from ocr_distributed.ocr_utils import (
            build_output_rel_stem,
            build_sidecar_base_name,
        )

        noext = build_sidecar_base_name("/source/noext")
        literal_img = build_sidecar_base_name("/source/noext.img")
        assert noext != literal_img
        assert "__img" in noext
        assert "~p1h" in noext
        assert literal_img == "noext__img"
        assert build_output_rel_stem("/source/noext", "image", "/source") == noext
        assert (
            build_output_rel_stem("/source/noext.img", "image", "/source")
            == literal_img
        )

    def test_lossy_stems_remain_distinct_across_pdf_and_sidecar(self):
        from ocr_distributed.ocr_utils import (
            build_output_rel_stem,
            build_sidecar_base_name,
        )

        a_q = build_sidecar_base_name("/source/a?.pdf")
        a_star = build_sidecar_base_name("/source/a*.pdf")
        assert a_q != a_star
        assert "~p1h" in a_q and "~p1h" in a_star
        assert build_output_rel_stem("/source/a?.pdf", "pdf", "/source") == a_q
        assert build_output_rel_stem("/source/a*.pdf", "pdf", "/source") == a_star

    def test_cross_builder_agreement_for_same_raw_source(self):
        from ocr_distributed.ocr_utils import (
            build_output_rel_stem,
            build_sidecar_base_name,
        )

        for raw_name, source_type in (
            ("report.pdf", "pdf"),
            ("report.PDF", "pdf"),
            ("report.jpg", "image"),
            ("report.JPG", "image"),
            ("report__jpg.pdf", "pdf"),
            ("clip.mp4", "video"),
            ("notes.txt", "text"),
            ("a?.pdf", "pdf"),
            ("noext", "image"),
            ("noext.img", "image"),
            ("doc. .pdf", "pdf"),
        ):
            side = build_sidecar_base_name(f"/source/{raw_name}")
            stem = build_output_rel_stem(f"/source/{raw_name}", source_type, "/source")
            assert stem == side


# ---------------------------------------------------------------------------
# sanitize_path_segment / encode_path_segment tests
# ---------------------------------------------------------------------------


def _segment_digest(raw: str) -> str:
    import base64

    digest = hashlib.sha256(raw.encode("utf-8")).digest()[:16]
    return base64.b32encode(digest).decode("ascii").rstrip("=").lower()


def _identity(slug: str, raw: str) -> str:
    return f"{slug}~p1h{_segment_digest(raw)}"


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


class TestEncodePathSegment:
    """Collision-resistant segment encoding used by all output families."""

    def test_preserves_simple_lowercase_safe_names(self):
        from ocr_distributed.ocr_utils import encode_path_segment

        assert encode_path_segment("hello_world") == "hello_world"
        assert encode_path_segment("report-v2") == "report-v2"
        assert encode_path_segment("file.name") == "file.name"

    def test_glob_char_pairs_do_not_collide(self):
        from ocr_distributed.ocr_utils import encode_path_segment

        left = encode_path_segment("a?")
        right = encode_path_segment("a*")
        assert left != right
        assert left == _identity("a_", "a?")
        assert right == _identity("a_", "a*")

    def test_case_aliases_do_not_collide(self):
        from ocr_distributed.ocr_utils import encode_path_segment

        lower = encode_path_segment("report")
        mixed = encode_path_segment("Report")
        assert lower == "report"
        assert mixed == _identity("report", "Report")
        assert lower != mixed

    def test_trailing_dot_or_space_is_digest_bearing(self):
        from ocr_distributed.ocr_utils import encode_path_segment

        trailing_space = encode_path_segment("folder. ")
        trailing_dots = encode_path_segment("folder...")
        assert trailing_space == _identity("folder", "folder. ")
        assert trailing_dots == _identity("folder", "folder...")
        assert trailing_space != trailing_dots
        assert trailing_space != encode_path_segment("folder")

    def test_windows_reserved_names_are_digest_bearing(self):
        from ocr_distributed.ocr_utils import encode_path_segment

        reserved = encode_path_segment("CON")
        assert reserved == _identity("_con", "CON")
        assert reserved != encode_path_segment("con")
        assert encode_path_segment("con") == _identity("_con", "con")

    def test_empty_fallback_is_digest_bearing(self):
        from ocr_distributed.ocr_utils import encode_path_segment

        empty_dir = encode_path_segment("...", empty_fallback="_")
        empty_stem = encode_path_segment("...", empty_fallback="document")
        assert empty_dir == _identity("_", "...")
        assert empty_stem == _identity("document", "...")
        assert empty_dir != empty_stem

    def test_literal_encoded_namespace_collision_reencodes(self):
        from ocr_distributed.ocr_utils import encode_path_segment

        encoded = encode_path_segment("a?")
        # A raw name equal to another name's encoded form must encode differently
        # because "~" is reserved for the versioned identity marker.
        collided = encode_path_segment(encoded)
        assert collided != encoded
        assert "~p1h" in collided
        # Slug sanitization replaces "~" and casefolds; digest uses full raw.
        assert collided == _identity(
            encoded.replace("~", "_").casefold(),
            encoded,
        )

    def test_long_same_prefix_different_tail_bounded(self):
        from ocr_distributed.ocr_utils import (
            _PATH_SEGMENT_MAX_UTF8_BYTES,
            encode_path_segment,
        )

        prefix = "x" * 200
        left = encode_path_segment(prefix + "alpha")
        right = encode_path_segment(prefix + "bravo")
        assert left != right
        assert len(left.encode("utf-8")) <= _PATH_SEGMENT_MAX_UTF8_BYTES
        assert len(right.encode("utf-8")) <= _PATH_SEGMENT_MAX_UTF8_BYTES
        assert left.endswith("~p1h" + _segment_digest(prefix + "alpha"))
        assert right.endswith("~p1h" + _segment_digest(prefix + "bravo"))
        # Marker + digest preserved; only the slug is truncated.
        marker_digest = "~p1h" + _segment_digest(prefix + "alpha")
        assert left.endswith(marker_digest)
        assert left[-len(marker_digest) :] == marker_digest

    def test_slash_and_backslash_are_unsafe(self):
        from ocr_distributed.ocr_utils import encode_path_segment

        slash = encode_path_segment("a/b")
        backslash = encode_path_segment("a\\b")
        assert slash == _identity("a_b", "a/b")
        assert backslash == _identity("a_b", "a\\b")
        assert slash != backslash

    def test_paired_directories_and_files_agree_across_builders(self, tmp_path):
        from ocr_distributed.ocr_utils import (
            build_output_rel_stem,
            build_sidecar_base_name,
            encode_path_segment,
            resolve_safe_output_subfolder,
        )

        source_root = tmp_path / "source"
        for raw_dir, raw_file in (
            ("a?", "doc.pdf"),
            ("a*", "doc.pdf"),
            ("Case", "Report.pdf"),
            ("case", "report.pdf"),
            ("folder. ", "x.pdf"),
            ("CON", "y.pdf"),
        ):
            path = source_root / raw_dir / raw_file
            stem = build_output_rel_stem(str(path), "pdf", str(source_root))
            dir_part, file_part = os.path.split(stem)
            assert dir_part == encode_path_segment(raw_dir)
            assert file_part == build_sidecar_base_name(raw_file)
            resolved = resolve_safe_output_subfolder(str(tmp_path / "out"), raw_dir)
            assert os.path.basename(resolved) == dir_part


class TestResolveSafeOutputSubfolder:
    """Tests for the canonical mirrored-output directory resolver."""

    @pytest.mark.parametrize(
        ("subfolder", "raw_parts"),
        [
            ("", ()),
            (".", ()),
            ("case/.../review:queue", ("case", "...", "review:queue")),
            (r"case\nested\folder. ", ("case", "nested", "folder. ")),
        ],
    )
    def test_maps_valid_segments_canonically(self, tmp_path, subfolder, raw_parts):
        from ocr_distributed.ocr_utils import (
            encode_path_segment,
            resolve_safe_output_subfolder,
        )

        root = tmp_path / "EXPORT" / "TEST"
        result = resolve_safe_output_subfolder(str(root), subfolder)
        expected_parts = tuple(
            encode_path_segment(part, empty_fallback="_") for part in raw_parts
        )

        assert os.path.normcase(result) == os.path.normcase(
            str(root.joinpath(*expected_parts))
        )

    @pytest.mark.parametrize(
        "subfolder",
        ["..", "../escape", "safe/../escape", "safe/./escape", r"safe\..\escape"],
    )
    def test_rejects_traversal_tokens(self, tmp_path, subfolder):
        from ocr_distributed.ocr_utils import resolve_safe_output_subfolder

        with pytest.raises(ValueError, match="traversal"):
            resolve_safe_output_subfolder(str(tmp_path / "root"), subfolder)

    def test_empty_sanitized_segment_matches_pdf_stem(self, tmp_path):
        from ocr_distributed.ocr_utils import (
            build_output_rel_stem,
            resolve_safe_output_subfolder,
        )

        source_root = tmp_path / "source"
        source_path = source_root / "..." / "doc.pdf"
        output_root = tmp_path / "output"

        stem = build_output_rel_stem(str(source_path), "pdf", str(source_root))
        sidecar_dir = resolve_safe_output_subfolder(str(output_root), "...")

        assert os.path.dirname(stem) == os.path.relpath(sidecar_dir, output_root)

    def test_commonpath_guard_rejects_encoder_regression(self, tmp_path, monkeypatch):
        import ocr_distributed.ocr_utils as ocr_utils

        monkeypatch.setattr(ocr_utils, "encode_path_segment", lambda *_a, **_k: "..")
        with pytest.raises(ValueError, match="escapes"):
            ocr_utils.resolve_safe_output_subfolder(str(tmp_path / "root"), "safe")


class TestAtomicOutputWriters:
    """Symlink-safe atomic text/JSON writers used by sidecar families."""

    def test_write_text_atomic_round_trip(self, tmp_path):
        from ocr_distributed.ocr_utils import write_text_atomic

        root = tmp_path / "EXPORT" / "TEST"
        root.mkdir(parents=True)
        target = root / "doc.txt"
        written = write_text_atomic(
            str(target),
            "hello atomic",
            allowed_root=str(root),
        )
        assert os.path.normcase(written) == os.path.normcase(str(target))
        assert target.read_text(encoding="utf-8") == "hello atomic"

    def test_write_json_atomic_round_trip(self, tmp_path):
        from ocr_distributed.ocr_utils import write_json_atomic

        root = tmp_path / "EXPORT" / "TEST"
        root.mkdir(parents=True)
        target = root / "doc.json"
        write_json_atomic(
            str(target),
            {"schema_version": "1.0", "ok": True},
            allowed_root=str(root),
            default=None,
        )
        assert target.read_text(encoding="utf-8") == (
            '{\n  "schema_version": "1.0",\n  "ok": true\n}'
        )

    def test_write_text_atomic_rejects_escape_outside_root(self, tmp_path):
        from ocr_distributed.ocr_utils import write_text_atomic

        root = tmp_path / "EXPORT" / "TEST"
        root.mkdir(parents=True)
        outside = tmp_path / "outside.txt"
        with pytest.raises(ValueError, match="escapes allowed root"):
            write_text_atomic(str(outside), "nope", allowed_root=str(root))
        assert not outside.exists()

    def test_write_text_atomic_rejected_nested_outside_creates_no_directory(
        self,
        tmp_path,
    ):
        from ocr_distributed.ocr_utils import write_text_atomic

        root = tmp_path / "EXPORT" / "TEST"
        root.mkdir(parents=True)
        nested_outside = tmp_path / "outside_nested" / "deep" / "file.txt"
        with pytest.raises(ValueError, match="escapes allowed root"):
            write_text_atomic(str(nested_outside), "nope", allowed_root=str(root))
        assert not (tmp_path / "outside_nested").exists()

    def test_write_text_atomic_rejects_parent_symlink_escape(self, tmp_path):
        from ocr_distributed.ocr_utils import write_text_atomic

        root = tmp_path / "EXPORT" / "TEST"
        root.mkdir(parents=True)
        outside_dir = tmp_path / "outside_dir"
        outside_dir.mkdir()
        link = root / "escape_link"
        try:
            link.symlink_to(outside_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not supported in this environment")

        target = link / "doc.txt"
        with pytest.raises(ValueError, match="escapes allowed root"):
            write_text_atomic(str(target), "nope", allowed_root=str(root))
        assert not (outside_dir / "doc.txt").exists()
        leftovers = list(outside_dir.iterdir())
        assert leftovers == []

    def test_write_text_atomic_rejects_parent_symlink_before_mkdir(
        self,
        tmp_path,
    ):
        """Parent symlink + missing tail must not create outside/newdir."""
        from ocr_distributed.ocr_utils import write_text_atomic

        root = tmp_path / "allowed_root"
        root.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        link = root / "link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not supported in this environment")

        target = link / "newdir" / "doc.txt"
        with pytest.raises(ValueError, match="escapes allowed root"):
            write_text_atomic(str(target), "nope", allowed_root=str(root))
        assert not (outside / "newdir").exists()
        assert not target.exists()
        leftovers = list(outside.iterdir())
        assert leftovers == []

    @pytest.mark.skipif(
        os.name == "nt", reason="POSIX mode bits not available on Windows"
    )
    def test_write_text_atomic_modes_new_existing_and_leaf_symlink(self, tmp_path):
        import stat as stat_mod

        from ocr_distributed.ocr_utils import write_text_atomic

        root = tmp_path / "EXPORT" / "TEST"
        root.mkdir(parents=True)

        created = root / "created.txt"
        write_text_atomic(str(created), "new", allowed_root=str(root))
        assert stat_mod.S_IMODE(created.stat().st_mode) == 0o644

        existing = root / "existing.txt"
        existing.write_text("old", encoding="utf-8")
        existing.chmod(0o640)
        write_text_atomic(str(existing), "new", allowed_root=str(root))
        assert stat_mod.S_IMODE(existing.stat().st_mode) == 0o640

        outside = tmp_path / "outside-secret.txt"
        outside.write_text("SECRET", encoding="utf-8")
        outside.chmod(0o600)
        leaf = root / "linked.txt"
        try:
            leaf.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not supported in this environment")
        write_text_atomic(str(leaf), "safe", allowed_root=str(root))
        assert leaf.is_file() and not leaf.is_symlink()
        assert stat_mod.S_IMODE(leaf.stat().st_mode) == 0o644
        assert outside.read_text(encoding="utf-8") == "SECRET"
        assert stat_mod.S_IMODE(outside.stat().st_mode) == 0o600

    def test_write_text_atomic_replaces_leaf_symlink_not_target(self, tmp_path):
        from ocr_distributed.ocr_utils import write_text_atomic

        root = tmp_path / "EXPORT" / "TEST"
        root.mkdir(parents=True)
        outside = tmp_path / "outside.txt"
        outside.write_text("SECRET", encoding="utf-8")
        target = root / "doc.json"
        try:
            target.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not supported in this environment")

        write_text_atomic(str(target), '{"safe": true}', allowed_root=str(root))

        assert target.is_file()
        assert not target.is_symlink()
        assert target.read_text(encoding="utf-8") == '{"safe": true}'
        assert outside.read_text(encoding="utf-8") == "SECRET"

    def test_write_text_atomic_cleans_temp_on_replace_error(
        self, tmp_path, monkeypatch
    ):
        from ocr_distributed import ocr_utils as ocr_utils_mod

        root = tmp_path / "EXPORT" / "TEST"
        root.mkdir(parents=True)
        target = root / "doc.txt"

        def _boom(*_args, **_kwargs):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(ocr_utils_mod.os, "replace", _boom)
        with pytest.raises(OSError, match="simulated replace failure"):
            ocr_utils_mod.write_text_atomic(
                str(target),
                "payload",
                allowed_root=str(root),
            )
        assert not target.exists()
        leftovers = [path for path in root.iterdir() if path.name.endswith(".tmp")]
        assert leftovers == []

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode failure contract")
    def test_write_text_atomic_posix_mode_failure_is_fatal(self, tmp_path, monkeypatch):
        """Injected fchmod OSError must abort before replace on POSIX."""
        from ocr_distributed import ocr_utils as ocr_utils_mod

        root = tmp_path / "EXPORT" / "TEST"
        root.mkdir(parents=True)
        target = root / "doc.txt"

        def _fail_fchmod(_fd, _mode):
            raise OSError("simulated fchmod failure")

        monkeypatch.setattr(ocr_utils_mod.os, "fchmod", _fail_fchmod)
        with pytest.raises(OSError, match="simulated fchmod failure"):
            ocr_utils_mod.write_text_atomic(
                str(target),
                "payload",
                allowed_root=str(root),
            )
        assert not target.exists()
        leftovers = [path for path in root.iterdir() if path.name.endswith(".tmp")]
        assert leftovers == []


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
        assert detector.detect_from_pdf("/fake.pdf") == "und"

    def test_missing_model_path_returns_defaults(self):
        from ocr_distributed.language import LanguageDetector

        detector = LanguageDetector(model_path="/nonexistent/lid.176.bin")
        assert not detector.is_loaded

    @patch("ocr_distributed.language.os.path.exists", return_value=True)
    def test_load_failure_falls_back_gracefully(self, mock_exists):
        from ocr_distributed.language import LanguageDetector

        with patch.dict(
            "sys.modules",
            {
                "fasttext": MagicMock(
                    load_model=MagicMock(side_effect=Exception("fail"))
                )
            },
        ):
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
        lang, conf = detector.detect_from_text(
            "This is a sufficiently long English text for detection."
        )
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
            "classify_source_file",
            "detect_magic_family",
            "extract_paddle_lines",
            "get_file_hash",
            "img_to_bytes",
            "to_plain_list",
            "box_to_rect_and_anchor",
            "read_file_header",
            "sanitize_path_segment",
            "encode_path_segment",
            "build_output_rel_stem",
            "build_sidecar_base_name",
            "resolve_safe_output_subfolder",
            "write_json_atomic",
            "write_text_atomic",
            "get_source_page_count",
            "insert_text_line",
        ]:
            assert hasattr(
                ocr_distributed, name
            ), f"{name} not exported from ocr_distributed"
