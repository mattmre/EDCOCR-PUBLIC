"""Coordinator OCR lines use the shared grounded searchable-text adapter."""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz

from ocr_distributed.ocr_utils import _resolve_text_font, insert_text_line

# ---------------------------------------------------------------------------
# Source inspection tests: verify coordinator tasks.py has correct call sites
# ---------------------------------------------------------------------------


class TestCoordinatorCallSites(unittest.TestCase):
    """Verify coordinator/jobs/tasks.py passes lang_code at all text insertion call sites."""

    @classmethod
    def setUpClass(cls):
        """Read coordinator tasks.py source once for all tests."""
        tasks_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "coordinator", "jobs", "tasks.py",
        )
        with open(tasks_path, "r", encoding="utf-8") as f:
            cls.source = f.read()

    def test_insert_text_line_passes_language_and_source_rotation(self):
        """The page overlay carries font and Paddle reading direction."""
        self.assertIn("insert_text_line(", self.source)
        self.assertIn("lang_code=lang_code", self.source)
        self.assertIn("geometry.source_rotation", self.source)

    def test_tesseract_fallback_uses_grounded_word_boxes(self):
        """Tesseract words are mapped to page geometry before shared insert."""
        self.assertIn("tesseract_data_to_lines_with_review(", self.source)
        self.assertIn("transform=transform", self.source)

    def test_no_hardcoded_helv_in_insert_text(self):
        """There should be no hardcoded fontname='helv' in page.insert_text calls."""
        # After the fix, the Tesseract fallback uses font_kwargs from
        # _resolve_text_font rather than hardcoding fontname="helv"
        self.assertNotIn('fontname="helv"', self.source)

    def test_font_resolution_is_owned_by_shared_adapter(self):
        """The coordinator does not bypass geometry and glyph validation."""
        self.assertNotIn("_resolve_text_font", self.source)

    def test_import_block_includes_shared_adapter(self):
        self.assertIn("insert_text_line,", self.source)


# ---------------------------------------------------------------------------
# Functional tests: _resolve_text_font behavior
# ---------------------------------------------------------------------------


class TestResolveTextFontFromDistributed(unittest.TestCase):
    """Test _resolve_text_font from ocr_distributed for coordinator use."""

    def test_none_returns_helv(self):
        """None lang_code should return helv with no fontfile."""
        fontname, fontfile = _resolve_text_font(None)
        self.assertEqual(fontname, "helv")
        self.assertIsNone(fontfile)

    def test_empty_string_returns_helv(self):
        """Empty string should return helv."""
        fontname, fontfile = _resolve_text_font("")
        self.assertEqual(fontname, "helv")
        self.assertIsNone(fontfile)

    def test_english_without_fonts_installed(self):
        """English should fall back to helv when Noto fonts not installed."""
        fontname, fontfile = _resolve_text_font("en")
        # If fonts are not installed on the test system, falls back to helv
        if fontfile is None:
            self.assertEqual(fontname, "helv")
        else:
            self.assertTrue(os.path.exists(fontfile))

    def test_returns_tuple_of_two(self):
        """Should always return a 2-tuple."""
        result = _resolve_text_font("en")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @patch("font_selector.get_font_path")
    def test_custom_font_when_available(self, mock_get_path):
        """When font file exists, should return custom fontname and path."""
        mock_path = MagicMock(spec=Path)
        mock_path.stem = "NotoSansCJKjp-Regular"
        mock_path.__str__ = lambda s: "/app/fonts/noto/NotoSansCJKjp-Regular.otf"
        mock_get_path.return_value = mock_path

        fontname, fontfile = _resolve_text_font("japan")
        self.assertEqual(fontname, "NotoSansCJKjp-Regular")
        self.assertEqual(fontfile, "/app/fonts/noto/NotoSansCJKjp-Regular.otf")


# ---------------------------------------------------------------------------
# Integration tests: insert_text_line with lang_code on PDF pages
# ---------------------------------------------------------------------------


class TestInsertTextLineWithLangCode(unittest.TestCase):
    """Test that insert_text_line works with lang_code for coordinator use."""

    def test_paddle_bbox_with_lang_code(self):
        """insert_text_line with bbox and lang_code should produce valid text."""
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        result = insert_text_line(
            page, "Test text",
            [[100, 100], [300, 100], [300, 120], [100, 120]],
            lang_code="en",
        )
        self.assertTrue(result)

        text = page.get_text()
        self.assertIn("Test text", text)
        doc.close()

    def test_without_lang_code_backward_compat(self):
        """Calling without lang_code should still work."""
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        result = insert_text_line(
            page, "No lang",
            [[100, 100], [300, 100], [300, 120], [100, 120]],
        )
        self.assertTrue(result)
        doc.close()

    def test_tesseract_fallback_pattern(self):
        """Tesseract text stays searchable inside its measured word box."""
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        lang_code = "en"
        page_text = "Sample tesseract output"

        assert insert_text_line(
            page, page_text, [72, 72, 300, 96], lang_code=lang_code
        )

        text = page.get_text()
        self.assertIn("Sample tesseract output", text)
        doc.close()

    def test_tesseract_fallback_cjk_pattern(self):
        """CJK insertion preserves glyphs or fails closed if font absent."""
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        lang_code = "japan"
        inserted = insert_text_line(
            page, "日本語", [72, 72, 300, 96], lang_code=lang_code
        )
        if inserted:
            self.assertIn("日本語", page.get_text())
        else:
            self.assertNotIn("日本語", page.get_text())
        doc.close()

    def test_multiple_languages_same_page(self):
        """Multiple calls with different lang_codes on same page should work."""
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        insert_text_line(
            page, "English",
            [[100, 100], [300, 100], [300, 120], [100, 120]],
            lang_code="en",
        )
        insert_text_line(
            page, "Arabic",
            [[100, 150], [300, 150], [300, 170], [100, 170]],
            lang_code="ar",
        )

        text = page.get_text()
        self.assertIn("English", text)
        self.assertIn("Arabic", text)
        hits = page.search_for("Arabic")
        self.assertEqual(len(hits), 1)
        self.assertTrue(fitz.Rect(100, 150, 300, 170).contains(hits[0]))
        doc.close()


if __name__ == "__main__":
    unittest.main()
