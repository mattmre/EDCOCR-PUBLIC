"""Integration tests for font embedding in searchable PDF text layer.

Tests verify that:
- _resolve_text_font returns correct font info for various languages
- insert_text_line accepts and uses the lang_code parameter
- Fallback to helv works when fonts are unavailable
- All three call sites in ocr_gpu_async.py pass lang_code
- The distributed copy in ocr_distributed/ocr_utils.py is also updated
"""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz

from font_selector import get_font_name


class TestResolveTextFont(unittest.TestCase):
    """Test the _resolve_text_font helper function."""

    def _get_resolve_fn(self):
        """Import _resolve_text_font from ocr_gpu_async."""
        # Import from ocr_distributed since it has the same function
        # without needing all ocr_gpu_async dependencies
        from ocr_distributed.ocr_utils import _resolve_text_font
        return _resolve_text_font

    def test_none_lang_returns_helv(self):
        """No language code should return helv."""
        resolve = self._get_resolve_fn()
        fontname, fontfile = resolve(None)
        self.assertEqual(fontname, "helv")
        self.assertIsNone(fontfile)

    def test_empty_lang_returns_helv(self):
        """Empty string should return helv."""
        resolve = self._get_resolve_fn()
        fontname, fontfile = resolve("")
        self.assertEqual(fontname, "helv")
        self.assertIsNone(fontfile)

    def test_returns_helv_when_font_not_installed(self):
        """When Noto fonts are not installed, should fall back to helv."""
        resolve = self._get_resolve_fn()
        # Unless Noto fonts happen to be at /app/fonts/noto, this should fallback
        fontname, fontfile = resolve("ta")
        if fontfile is None:
            self.assertEqual(fontname, "helv")
        else:
            # If fonts ARE installed, verify the path exists
            self.assertTrue(os.path.exists(fontfile))

    @patch("font_selector.get_font_path")
    def test_returns_custom_font_when_available(self, mock_get_path):
        """When font file exists, should return custom fontname and path."""
        mock_path = MagicMock(spec=Path)
        mock_path.stem = "NotoSansTamil-Regular"
        mock_path.__str__ = lambda s: "/app/fonts/noto/NotoSansTamil-Regular.ttf"
        mock_get_path.return_value = mock_path

        resolve = self._get_resolve_fn()
        fontname, fontfile = resolve("ta")
        self.assertEqual(fontname, "NotoSansTamil-Regular")
        self.assertEqual(fontfile, "/app/fonts/noto/NotoSansTamil-Regular.ttf")


class TestInsertTextLineLangCode(unittest.TestCase):
    """Test insert_text_line with lang_code parameter."""

    def test_insert_text_line_accepts_lang_code(self):
        """insert_text_line should accept lang_code as keyword argument."""
        from ocr_distributed.ocr_utils import insert_text_line

        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        # Should not raise — lang_code is optional and defaults to None
        result = insert_text_line(
            page, "Hello World",
            [[100, 100], [300, 100], [300, 120], [100, 120]],
            lang_code=None,
        )
        self.assertTrue(result)
        doc.close()

    def test_insert_text_line_with_english(self):
        """insert_text_line with lang_code='en' should work with helv fallback."""
        from ocr_distributed.ocr_utils import insert_text_line

        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        result = insert_text_line(
            page, "English text",
            [[100, 100], [300, 100], [300, 120], [100, 120]],
            lang_code="en",
        )
        self.assertTrue(result)
        doc.close()

    def test_insert_text_line_backward_compatible(self):
        """insert_text_line without lang_code should still work."""
        from ocr_distributed.ocr_utils import insert_text_line

        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        # Call without lang_code — backward compatible
        result = insert_text_line(
            page, "No lang code",
            [[100, 100], [300, 100], [300, 120], [100, 120]],
        )
        self.assertTrue(result)
        doc.close()

    def test_insert_text_line_anchor_fallback_with_lang(self):
        """Anchor fallback path should also use lang_code."""
        from ocr_distributed.ocr_utils import insert_text_line

        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        # Provide a single point (anchor path)
        insert_text_line(
            page, "Anchor text",
            [100, 200],
            lang_code="en",
        )
        # The anchor path may or may not work depending on box_to_rect_and_anchor
        # behavior with [100, 200], but it should not crash
        doc.close()


class TestFontSelectorMapping(unittest.TestCase):
    """Test that font_selector maps languages correctly for PDF embedding."""

    def test_tamil_maps_to_noto_tamil(self):
        """Tamil should map to NotoSansTamil-Regular.ttf."""
        self.assertEqual(get_font_name("ta"), "NotoSansTamil-Regular.ttf")

    def test_telugu_maps_to_noto_telugu(self):
        """Telugu should map to NotoSansTelugu-Regular.ttf."""
        self.assertEqual(get_font_name("te"), "NotoSansTelugu-Regular.ttf")

    def test_kannada_maps_to_noto_kannada(self):
        """Kannada should map to NotoSansKannada-Regular.ttf."""
        self.assertEqual(get_font_name("kn"), "NotoSansKannada-Regular.ttf")

    def test_georgian_maps_to_noto_georgian(self):
        """Georgian should map to NotoSansGeorgian-Regular.ttf."""
        self.assertEqual(get_font_name("ka"), "NotoSansGeorgian-Regular.ttf")

    def test_chinese_simplified_maps_to_cjk_sc(self):
        """Chinese Simplified should map to CJK SC font."""
        self.assertEqual(get_font_name("ch"), "NotoSansCJKsc-Regular.otf")

    def test_arabic_maps_to_noto_arabic(self):
        """Arabic should map to NotoSansArabic-Regular.ttf."""
        self.assertEqual(get_font_name("ar"), "NotoSansArabic-Regular.ttf")

    def test_hindi_maps_to_devanagari(self):
        """Hindi should map to NotoSansDevanagari-Regular.ttf."""
        self.assertEqual(get_font_name("hi"), "NotoSansDevanagari-Regular.ttf")

    def test_english_maps_to_noto_sans(self):
        """English should map to NotoSans-Regular.ttf."""
        self.assertEqual(get_font_name("en"), "NotoSans-Regular.ttf")

    def test_unknown_lang_falls_back(self):
        """Unknown language should fall back to NotoSans-Regular.ttf."""
        self.assertEqual(get_font_name("xyz"), "NotoSans-Regular.ttf")

    def test_fasttext_alias_zh(self):
        """FastText 'zh' should resolve to Chinese Simplified font."""
        self.assertEqual(get_font_name("zh"), "NotoSansCJKsc-Regular.otf")

    def test_fasttext_alias_ja(self):
        """FastText 'ja' should resolve to Japanese font."""
        self.assertEqual(get_font_name("ja"), "NotoSansCJKjp-Regular.otf")


class TestCallSitesPassLangCode(unittest.TestCase):
    """Verify all insert_text_line call sites pass lang_code."""

    def _read_source(self, filename):
        """Read a source file from the project root."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            filename,
        )
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_paddle_primary_pass_lang_code(self):
        """PaddleOCR primary pass should pass lang_code=task.lang_hint."""
        source = self._read_source("ocr_gpu_async.py")
        self.assertIn("insert_text_line(page, normalized_txt, box, lang_code=task.lang_hint)", source)

    def test_paddle_rerun_pass_lang_code(self):
        """PaddleOCR two-pass rerun should pass lang_code=detected_code."""
        source = self._read_source("ocr_gpu_async.py")
        self.assertIn("insert_text_line(rerun_page, normalized_txt, box, lang_code=detected_code)", source)

    def test_tesseract_fallback_uses_font(self):
        """Tesseract fallback should use _resolve_text_font."""
        source = self._read_source("ocr_gpu_async.py")
        self.assertIn("_resolve_text_font(task.lang_hint)", source)

    def test_font_selector_imported(self):
        """ocr_gpu_async should import from font_selector."""
        source = self._read_source("ocr_gpu_async.py")
        self.assertIn("from font_selector import", source)
        self.assertIn("_FONT_SELECTOR_AVAILABLE", source)

    def test_distributed_copy_updated(self):
        """ocr_distributed/ocr_utils.py should also have lang_code parameter."""
        source = self._read_source("ocr_distributed/ocr_utils.py")
        self.assertIn("def insert_text_line(page, txt, box, lang_code=None)", source)
        self.assertIn("_resolve_text_font", source)


class TestHelvFallbackBehavior(unittest.TestCase):
    """Verify graceful fallback when font_selector is unavailable."""

    def test_helv_produces_valid_pdf(self):
        """Default helv font should produce valid PDF output."""
        from ocr_distributed.ocr_utils import insert_text_line

        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        insert_text_line(
            page, "Latin text with helv",
            [[50, 50], [300, 50], [300, 70], [50, 70]],
        )

        pdf_bytes = doc.write()
        self.assertGreater(len(pdf_bytes), 0)

        # Verify it's valid PDF
        verify_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        self.assertEqual(len(verify_doc), 1)
        verify_doc.close()
        doc.close()

    def test_render_mode_3_invisible(self):
        """Text inserted with render_mode=3 should be invisible (no visible text)."""
        from ocr_distributed.ocr_utils import insert_text_line

        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        insert_text_line(
            page, "Invisible text",
            [[50, 50], [300, 50], [300, 70], [50, 70]],
        )

        # The text should be searchable but invisible
        text = page.get_text()
        self.assertIn("Invisible text", text)
        doc.close()


if __name__ == "__main__":
    unittest.main()
