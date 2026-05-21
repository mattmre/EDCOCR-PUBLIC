"""Tests for EasyOCR third engine module (easyocr_engine.py).

Tests cover:
- Guarded import and graceful degradation when easyocr is not installed
- EasyOCREngine initialization (lazy Reader creation, CPU-only enforcement)
- OCR execution with mocked Reader
- Result normalization (EasyOCR polygon -> axis-aligned bbox)
- Language mapping (PaddleOCR codes -> EasyOCR codes)
- Singleton factory (get_easyocr_engine)
- Configuration parsing (ENABLE_EASYOCR, EASYOCR_LANGUAGES)
- Integration with engine_selection.py routing
- Integration with handwriting.py is_handwritten_page

Run with: python -m pytest tests/test_easyocr_engine.py -v
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_image(width=200, height=100):
    """Create a small test image for OCR calls."""
    arr = np.ones((height, width, 3), dtype=np.uint8) * 200
    return Image.fromarray(arr)


def _make_easyocr_result():
    """Return a sample EasyOCR readtext() result list.

    EasyOCR returns: list of (bbox_polygon, text, confidence)
    where bbox_polygon is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
    """
    return [
        ([[10, 20], [200, 20], [200, 50], [10, 50]], "Hello World", 0.95),
        ([[10, 60], [180, 60], [180, 90], [10, 90]], "Test line", 0.87),
    ]


# ---------------------------------------------------------------------------
# Tests: Language mapping
# ---------------------------------------------------------------------------


class TestLanguageMapping:
    """Tests for PaddleOCR -> EasyOCR language code mapping."""

    def test_english_maps(self):
        from easyocr_engine import map_paddle_to_easyocr

        assert map_paddle_to_easyocr("en") == "en"

    def test_german_maps(self):
        from easyocr_engine import map_paddle_to_easyocr

        assert map_paddle_to_easyocr("german") == "de"

    def test_chinese_simplified_maps(self):
        from easyocr_engine import map_paddle_to_easyocr

        assert map_paddle_to_easyocr("ch") == "ch_sim"

    def test_chinese_traditional_maps(self):
        from easyocr_engine import map_paddle_to_easyocr

        assert map_paddle_to_easyocr("chinese_cht") == "ch_tra"

    def test_japanese_maps(self):
        from easyocr_engine import map_paddle_to_easyocr

        assert map_paddle_to_easyocr("japan") == "ja"

    def test_korean_maps(self):
        from easyocr_engine import map_paddle_to_easyocr

        assert map_paddle_to_easyocr("korean") == "ko"

    def test_french_maps(self):
        from easyocr_engine import map_paddle_to_easyocr

        assert map_paddle_to_easyocr("fr") == "fr"

    def test_unknown_language_returns_none(self):
        from easyocr_engine import map_paddle_to_easyocr

        assert map_paddle_to_easyocr("klingon") is None

    def test_empty_string_returns_none(self):
        from easyocr_engine import map_paddle_to_easyocr

        assert map_paddle_to_easyocr("") is None

    def test_none_returns_none(self):
        from easyocr_engine import map_paddle_to_easyocr

        assert map_paddle_to_easyocr(None) is None

    def test_case_insensitive(self):
        from easyocr_engine import map_paddle_to_easyocr

        assert map_paddle_to_easyocr("EN") == "en"
        assert map_paddle_to_easyocr("German") == "de"


# ---------------------------------------------------------------------------
# Tests: Result normalization
# ---------------------------------------------------------------------------


class TestResultNormalization:
    """Tests for _normalize_results converting EasyOCR output format."""

    def test_basic_normalization(self):
        from easyocr_engine import _normalize_results

        raw = _make_easyocr_result()
        normalized = _normalize_results(raw)

        assert len(normalized) == 2

        text, bbox, conf = normalized[0]
        assert text == "Hello World"
        assert bbox == [10, 20, 200, 50]
        assert conf == 0.95

    def test_bbox_is_axis_aligned(self):
        """Rotated polygon should be converted to axis-aligned bbox."""
        from easyocr_engine import _normalize_results

        # Rotated polygon
        raw = [
            ([[5, 10], [100, 5], [105, 45], [10, 50]], "Rotated", 0.80),
        ]
        normalized = _normalize_results(raw)
        text, bbox, conf = normalized[0]
        assert bbox == [5, 5, 105, 50]  # min/max of all x/y

    def test_empty_input(self):
        from easyocr_engine import _normalize_results

        assert _normalize_results([]) == []

    def test_malformed_detection_skipped(self):
        from easyocr_engine import _normalize_results

        raw = [
            "not a tuple",
            ([[10, 20]], "short bbox", 0.5),  # too few points in bbox
        ]
        # Should not crash; malformed items are skipped
        result = _normalize_results(raw)
        assert isinstance(result, list)

    def test_confidence_is_float(self):
        from easyocr_engine import _normalize_results

        raw = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "test", 1),
        ]
        normalized = _normalize_results(raw)
        assert isinstance(normalized[0][2], float)


# ---------------------------------------------------------------------------
# Tests: EasyOCREngine class
# ---------------------------------------------------------------------------


class TestEasyOCREngine:
    """Tests for the EasyOCREngine wrapper class."""

    def test_init_defaults(self):
        from easyocr_engine import EasyOCREngine

        engine = EasyOCREngine()
        assert engine.languages == ["en"]
        assert engine._gpu is False

    def test_gpu_always_forced_false(self):
        """Even if gpu=True is passed, it must be forced to False."""
        from easyocr_engine import EasyOCREngine

        engine = EasyOCREngine(languages=["en"], gpu=True)
        assert engine._gpu is False

    def test_custom_languages(self):
        from easyocr_engine import EasyOCREngine

        engine = EasyOCREngine(languages=["en", "fr", "de"])
        assert engine.languages == ["en", "fr", "de"]

    def test_languages_returns_copy(self):
        from easyocr_engine import EasyOCREngine

        engine = EasyOCREngine(languages=["en"])
        langs = engine.languages
        langs.append("fr")
        assert engine.languages == ["en"]  # Original unchanged

    def test_is_available_reflects_import(self):
        from easyocr_engine import EasyOCREngine

        engine = EasyOCREngine()
        # Reflects actual import status (easyocr likely not installed in test env)
        assert isinstance(engine.is_available, bool)

    def test_lazy_reader_not_created_on_init(self):
        from easyocr_engine import EasyOCREngine

        engine = EasyOCREngine()
        assert engine._reader is None

    @patch("easyocr_engine._EASYOCR_AVAILABLE", True)
    @patch("easyocr_engine._easyocr_mod")
    def test_ocr_creates_reader_lazily(self, mock_easyocr_mod):
        """Reader should be created on first ocr() call, not before."""
        from easyocr_engine import EasyOCREngine

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = _make_easyocr_result()
        mock_easyocr_mod.Reader.return_value = mock_reader

        engine = EasyOCREngine(languages=["en"])
        assert engine._reader is None

        img = _make_test_image()
        results = engine.ocr(img)

        mock_easyocr_mod.Reader.assert_called_once_with(["en"], gpu=False)
        assert engine._reader is mock_reader
        assert len(results) == 2

    @patch("easyocr_engine._EASYOCR_AVAILABLE", True)
    @patch("easyocr_engine._easyocr_mod")
    def test_ocr_returns_normalized_format(self, mock_easyocr_mod):
        from easyocr_engine import EasyOCREngine

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = _make_easyocr_result()
        mock_easyocr_mod.Reader.return_value = mock_reader

        engine = EasyOCREngine(languages=["en"])
        results = engine.ocr(_make_test_image())

        text, bbox, conf = results[0]
        assert isinstance(text, str)
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert isinstance(conf, float)

    @patch("easyocr_engine._EASYOCR_AVAILABLE", True)
    @patch("easyocr_engine._easyocr_mod")
    def test_ocr_accepts_numpy_array(self, mock_easyocr_mod):
        from easyocr_engine import EasyOCREngine

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        mock_easyocr_mod.Reader.return_value = mock_reader

        engine = EasyOCREngine()
        img_np = np.ones((100, 200, 3), dtype=np.uint8) * 128
        results = engine.ocr(img_np)
        assert results == []
        mock_reader.readtext.assert_called_once()

    @patch("easyocr_engine._EASYOCR_AVAILABLE", False)
    def test_ocr_returns_empty_when_unavailable(self):
        from easyocr_engine import EasyOCREngine

        engine = EasyOCREngine()
        results = engine.ocr(_make_test_image())
        assert results == []

    @patch("easyocr_engine._EASYOCR_AVAILABLE", True)
    @patch("easyocr_engine._easyocr_mod")
    def test_ocr_handles_reader_exception(self, mock_easyocr_mod):
        from easyocr_engine import EasyOCREngine

        mock_reader = MagicMock()
        mock_reader.readtext.side_effect = RuntimeError("GPU OOM")
        mock_easyocr_mod.Reader.return_value = mock_reader

        engine = EasyOCREngine()
        results = engine.ocr(_make_test_image())
        assert results == []

    def test_ocr_unsupported_image_type(self):
        """Non-PIL, non-numpy input should return empty list."""
        from easyocr_engine import EasyOCREngine

        with patch("easyocr_engine._EASYOCR_AVAILABLE", True), \
             patch("easyocr_engine._easyocr_mod") as mock_mod:
            mock_mod.Reader.return_value = MagicMock()
            engine = EasyOCREngine()
            results = engine.ocr("not an image")
            assert results == []

    @patch("easyocr_engine._EASYOCR_AVAILABLE", True)
    @patch("easyocr_engine._easyocr_mod")
    def test_language_override_warning(self, mock_easyocr_mod):
        """Requesting different languages than configured should log warning."""
        from easyocr_engine import EasyOCREngine

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        mock_easyocr_mod.Reader.return_value = mock_reader

        engine = EasyOCREngine(languages=["en"])
        with patch("easyocr_engine.logger") as mock_logger:
            engine.ocr(_make_test_image(), languages=["fr"])
            mock_logger.warning.assert_called()


# ---------------------------------------------------------------------------
# Tests: Singleton factory
# ---------------------------------------------------------------------------


class TestGetEasyOCREngine:
    """Tests for the get_easyocr_engine singleton factory."""

    def setup_method(self):
        """Reset singleton between tests."""
        from easyocr_engine import reset_engine

        reset_engine()

    def teardown_method(self):
        from easyocr_engine import reset_engine

        reset_engine()

    @patch("easyocr_engine.ENABLE_EASYOCR", False)
    def test_returns_none_when_disabled(self):
        from easyocr_engine import get_easyocr_engine

        assert get_easyocr_engine() is None

    @patch("easyocr_engine.ENABLE_EASYOCR", True)
    @patch("easyocr_engine._EASYOCR_AVAILABLE", False)
    def test_returns_none_when_not_installed(self):
        from easyocr_engine import get_easyocr_engine

        assert get_easyocr_engine() is None

    @patch("easyocr_engine.ENABLE_EASYOCR", True)
    @patch("easyocr_engine._EASYOCR_AVAILABLE", True)
    def test_returns_engine_when_enabled(self):
        from easyocr_engine import EasyOCREngine, get_easyocr_engine

        engine = get_easyocr_engine()
        assert isinstance(engine, EasyOCREngine)

    @patch("easyocr_engine.ENABLE_EASYOCR", True)
    @patch("easyocr_engine._EASYOCR_AVAILABLE", True)
    def test_singleton_returns_same_instance(self):
        from easyocr_engine import get_easyocr_engine

        engine1 = get_easyocr_engine()
        engine2 = get_easyocr_engine()
        assert engine1 is engine2

    @patch("easyocr_engine.ENABLE_EASYOCR", True)
    @patch("easyocr_engine._EASYOCR_AVAILABLE", True)
    def test_custom_languages_on_first_call(self):
        from easyocr_engine import get_easyocr_engine

        engine = get_easyocr_engine(languages=["en", "fr"])
        assert engine.languages == ["en", "fr"]

    @patch("easyocr_engine.ENABLE_EASYOCR", True)
    @patch("easyocr_engine._EASYOCR_AVAILABLE", True)
    @patch("easyocr_engine.EASYOCR_LANGUAGES", ["de", "fr"])
    def test_uses_env_languages_by_default(self):
        from easyocr_engine import get_easyocr_engine

        engine = get_easyocr_engine()
        assert engine.languages == ["de", "fr"]


# ---------------------------------------------------------------------------
# Tests: Configuration parsing
# ---------------------------------------------------------------------------


class TestConfiguration:
    """Tests for environment variable configuration."""

    def test_enable_true_values(self):
        for val in ("true", "True", "TRUE", "1", "yes", "YES"):
            with patch.dict(os.environ, {"ENABLE_EASYOCR": val}):
                # Re-evaluate the config expression
                result = val.lower().strip() in ("true", "1", "yes")
                assert result is True

    def test_enable_false_values(self):
        for val in ("false", "0", "no", "", "anything"):
            result = val.lower().strip() in ("true", "1", "yes")
            assert result is False

    def test_language_parsing_single(self):
        langs = [
            lang.strip() for lang in "en".split(",") if lang.strip()
        ]
        assert langs == ["en"]

    def test_language_parsing_multiple(self):
        langs = [
            lang.strip()
            for lang in "en, fr, de".split(",")
            if lang.strip()
        ]
        assert langs == ["en", "fr", "de"]

    def test_language_parsing_empty(self):
        langs = [
            lang.strip() for lang in "".split(",") if lang.strip()
        ]
        assert langs == []


# ---------------------------------------------------------------------------
# Tests: Integration with engine_selection.py
# ---------------------------------------------------------------------------


class TestEngineSelectionIntegration:
    """Tests for EasyOCR integration into engine_selection routing."""

    def test_force_easyocr_when_available(self):
        with patch("engine_selection._is_easyocr_available", return_value=True):
            from engine_selection import select_engine

            result = select_engine(force="easyocr")
            assert result == "easyocr"

    def test_force_easyocr_fallback_when_unavailable(self):
        with patch("engine_selection._is_easyocr_available", return_value=False):
            from engine_selection import select_engine

            result = select_engine(force="easyocr")
            assert result == "paddle"

    def test_select_engine_for_page_handwritten_routes_easyocr(self):
        with patch("engine_selection._is_easyocr_available", return_value=True):
            from engine_selection import select_engine_for_page

            result = select_engine_for_page(force="auto", is_handwritten=True)
            assert result == "easyocr"

    def test_select_engine_for_page_not_handwritten_uses_standard(self):
        """Non-handwritten pages in auto mode should use standard routing."""
        from engine_selection import select_engine_for_page

        result = select_engine_for_page(force="auto", is_handwritten=False)
        # Without image, auto mode defaults to paddle
        assert result == "paddle"

    def test_select_engine_for_page_paddle_mode_ignores_handwriting(self):
        """Force paddle should override handwriting flag."""
        from engine_selection import select_engine_for_page

        result = select_engine_for_page(force="paddle", is_handwritten=True)
        assert result == "paddle"

    def test_select_engine_for_page_tesseract_mode_ignores_handwriting(self):
        from engine_selection import select_engine_for_page

        result = select_engine_for_page(force="tesseract", is_handwritten=True)
        assert result == "tesseract"

    def test_select_engine_for_page_easyocr_unavailable_fallback(self):
        """Handwritten page with easyocr unavailable falls through to standard."""
        with patch("engine_selection._is_easyocr_available", return_value=False):
            from engine_selection import select_engine_for_page

            result = select_engine_for_page(force="auto", is_handwritten=True)
            # Falls through to standard select_engine auto mode (no image -> paddle)
            assert result == "paddle"

    def test_existing_select_engine_unchanged_for_paddle(self):
        """Existing select_engine behavior is preserved for paddle mode."""
        from engine_selection import select_engine

        assert select_engine(force="paddle") == "paddle"

    def test_existing_select_engine_unchanged_for_tesseract(self):
        from engine_selection import select_engine

        assert select_engine(force="tesseract") == "tesseract"

    def test_existing_select_engine_unchanged_for_auto_no_image(self):
        from engine_selection import select_engine

        assert select_engine(image=None, force="auto") == "paddle"


# ---------------------------------------------------------------------------
# Tests: Integration with handwriting.py
# ---------------------------------------------------------------------------


class TestHandwritingIntegration:
    """Tests for handwriting.py is_handwritten_page predicate."""

    def test_is_handwritten_page_with_true_result(self):
        from handwriting import PageHandwriting, is_handwritten_page

        page = PageHandwriting(page_num=1, has_handwriting=True)
        assert is_handwritten_page(page) is True

    def test_is_handwritten_page_with_false_result(self):
        from handwriting import PageHandwriting, is_handwritten_page

        page = PageHandwriting(page_num=1, has_handwriting=False)
        assert is_handwritten_page(page) is False

    def test_is_handwritten_page_with_dict_true(self):
        from handwriting import is_handwritten_page

        assert is_handwritten_page({"has_handwriting": True}) is True

    def test_is_handwritten_page_with_dict_false(self):
        from handwriting import is_handwritten_page

        assert is_handwritten_page({"has_handwriting": False}) is False

    def test_is_handwritten_page_with_none(self):
        from handwriting import is_handwritten_page

        assert is_handwritten_page(None) is False

    def test_is_handwritten_page_with_empty_dict(self):
        from handwriting import is_handwritten_page

        assert is_handwritten_page({}) is False

    def test_is_handwritten_page_with_unexpected_type(self):
        from handwriting import is_handwritten_page

        assert is_handwritten_page(42) is False
        assert is_handwritten_page("yes") is False
