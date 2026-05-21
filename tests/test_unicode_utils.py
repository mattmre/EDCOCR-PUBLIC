"""
Unit tests for Unicode normalization and RTL support module (unicode_utils.py).

Tests cover:
- NFC normalization (precomposed vs decomposed, Korean jamo, Arabic)
- NFKC normalization (compatibility forms, fullwidth, ligatures)
- RTL language detection (known RTL codes, non-RTL codes)
- RTL character detection (Arabic, Hebrew, mixed text)
- RTL ratio calculation
- RTL text reordering (with and without python-bidi)
- Pipeline integration helper (normalize_ocr_text)
- Edge cases (empty strings, None-like, pure ASCII)

Run with: python -m pytest tests/test_unicode_utils.py -v
"""

import unicodedata
from unittest import mock

from unicode_utils import (
    CJK_VERTICAL_LANGUAGES,
    RTL_LANGUAGES,
    bidi_available,
    get_rtl_ratio,
    has_cjk_chars,
    has_rtl_chars,
    is_cjk_vertical_language,
    is_rtl_language,
    normalize_nfc,
    normalize_nfkc,
    normalize_ocr_text,
    reorder_cjk_vertical_lines,
    reorder_rtl_text,
)

# ---------------------------------------------------------------------------
# Tests: NFC normalization
# ---------------------------------------------------------------------------


class TestNormalizeNFC:
    def test_precomposed_unchanged(self):
        """Already NFC text should pass through unchanged."""
        text = "\u00e9"  # e-with-acute (precomposed)
        assert normalize_nfc(text) == "\u00e9"

    def test_decomposed_to_precomposed(self):
        """NFD decomposed text should be composed to NFC."""
        text = "e\u0301"  # e + combining acute
        result = normalize_nfc(text)
        assert result == "\u00e9"
        assert len(result) == 1  # Single character after composition

    def test_ascii_unchanged(self):
        assert normalize_nfc("hello world") == "hello world"

    def test_empty_string(self):
        assert normalize_nfc("") == ""

    def test_none_returns_empty(self):
        # Our function treats falsy values as empty
        assert normalize_nfc(None) == ""

    def test_korean_jamo_composition(self):
        """Korean Jamo sequences should compose to Hangul syllables."""
        # Hangul Jamo: 한 decomposed
        decomposed = "\u1112\u1161\u11AB"  # ㅎ + ㅏ + ㄴ
        result = normalize_nfc(decomposed)
        expected = "\ud55c"  # 한 (precomposed Hangul syllable)
        assert result == expected

    def test_mixed_scripts(self):
        """Text with mixed scripts should normalize correctly."""
        text = "Hello e\u0301 World"
        result = normalize_nfc(text)
        assert result == "Hello \u00e9 World"

    def test_arabic_presentation_forms(self):
        """NFC does not change Arabic presentation forms (those are NFKC)."""
        # Arabic letter Alef: already NFC
        text = "\u0627\u0644\u0639\u0631\u0628\u064a\u0629"  # Arabic word
        result = normalize_nfc(text)
        assert result == text

    def test_idempotent(self):
        """Applying NFC twice should give the same result."""
        text = "caf\u00e9 na\u00efve"
        result1 = normalize_nfc(text)
        result2 = normalize_nfc(result1)
        assert result1 == result2

    def test_combining_diacritics(self):
        """Multiple combining diacritics should compose where possible."""
        # o + combining tilde + combining acute
        text = "o\u0303\u0301"
        result = normalize_nfc(text)
        # NFC should compose to the extent possible
        assert unicodedata.is_normalized("NFC", result)


# ---------------------------------------------------------------------------
# Tests: NFKC normalization
# ---------------------------------------------------------------------------


class TestNormalizeNFKC:
    def test_fullwidth_to_ascii(self):
        """Fullwidth Latin letters should normalize to ASCII."""
        text = "\uff28\uff45\uff4c\uff4c\uff4f"  # Fullwidth "Hello"
        result = normalize_nfkc(text)
        assert result == "Hello"

    def test_ligature_decomposition(self):
        """Ligatures should decompose to component characters."""
        text = "\ufb01"  # fi ligature
        result = normalize_nfkc(text)
        assert result == "fi"

    def test_superscript_normalization(self):
        """Superscript digits should normalize to regular digits."""
        text = "x\u00b2"  # x-squared
        result = normalize_nfkc(text)
        assert result == "x2"

    def test_empty_string(self):
        assert normalize_nfkc("") == ""

    def test_none_returns_empty(self):
        assert normalize_nfkc(None) == ""


# ---------------------------------------------------------------------------
# Tests: RTL language detection
# ---------------------------------------------------------------------------


class TestIsRTLLanguage:
    def test_arabic_is_rtl(self):
        assert is_rtl_language("ar") is True

    def test_hebrew_is_rtl(self):
        assert is_rtl_language("he") is True

    def test_farsi_is_rtl(self):
        assert is_rtl_language("fa") is True

    def test_urdu_is_rtl(self):
        assert is_rtl_language("ur") is True

    def test_english_not_rtl(self):
        assert is_rtl_language("en") is False

    def test_chinese_not_rtl(self):
        assert is_rtl_language("ch") is False

    def test_empty_string(self):
        assert is_rtl_language("") is False

    def test_none(self):
        assert is_rtl_language(None) is False

    def test_case_insensitive(self):
        assert is_rtl_language("AR") is True
        assert is_rtl_language("Ar") is True

    def test_whitespace_stripped(self):
        assert is_rtl_language("  ar  ") is True

    def test_rtl_languages_set(self):
        assert "ar" in RTL_LANGUAGES
        assert "he" in RTL_LANGUAGES
        assert "fa" in RTL_LANGUAGES
        assert "ur" in RTL_LANGUAGES


# ---------------------------------------------------------------------------
# Tests: RTL character detection
# ---------------------------------------------------------------------------


class TestHasRTLChars:
    def test_arabic_text(self):
        # Arabic characters
        assert has_rtl_chars("\u0627\u0644\u0639\u0631\u0628\u064a\u0629") is True

    def test_hebrew_text(self):
        assert has_rtl_chars("\u05e9\u05dc\u05d5\u05dd") is True  # "shalom"

    def test_latin_text(self):
        assert has_rtl_chars("Hello World") is False

    def test_empty_string(self):
        assert has_rtl_chars("") is False

    def test_none(self):
        assert has_rtl_chars(None) is False

    def test_mixed_ltr_rtl(self):
        # Mix of Latin and Arabic
        text = "Hello \u0627\u0644\u0639\u0631\u0628\u064a\u0629 World"
        assert has_rtl_chars(text) is True

    def test_numbers_only(self):
        assert has_rtl_chars("12345") is False

    def test_arabic_numbers(self):
        # Arabic-Indic digits are AN (Arabic Number) in bidi category
        assert has_rtl_chars("\u0660\u0661\u0662") is True


# ---------------------------------------------------------------------------
# Tests: RTL ratio
# ---------------------------------------------------------------------------


class TestGetRTLRatio:
    def test_pure_arabic(self):
        text = "\u0627\u0644\u0639\u0631\u0628\u064a\u0629"
        ratio = get_rtl_ratio(text)
        assert ratio == 1.0

    def test_pure_latin(self):
        assert get_rtl_ratio("Hello World") == 0.0

    def test_empty_string(self):
        assert get_rtl_ratio("") == 0.0

    def test_numbers_only(self):
        assert get_rtl_ratio("12345") == 0.0

    def test_mixed_content(self):
        """Mix of Arabic and Latin should give a fractional ratio."""
        # 3 Arabic letters + 3 Latin letters
        text = "\u0627\u0644\u0639abc"
        ratio = get_rtl_ratio(text)
        assert 0.0 < ratio < 1.0


# ---------------------------------------------------------------------------
# Tests: RTL text reordering
# ---------------------------------------------------------------------------


class TestReorderRTLText:
    def test_empty_string(self):
        assert reorder_rtl_text("") == ""

    def test_none_returns_empty(self):
        assert reorder_rtl_text(None) == ""

    def test_ltr_text_unchanged(self):
        text = "Hello World"
        result = reorder_rtl_text(text)
        # For LTR text, result should be the same regardless of bidi availability
        assert result == text

    @mock.patch("unicode_utils._BIDI_AVAILABLE", False)
    def test_graceful_degradation_without_bidi(self):
        """Without python-bidi, text should be returned unchanged."""
        text = "\u0627\u0644\u0639\u0631\u0628\u064a\u0629"
        result = reorder_rtl_text(text)
        assert result == text

    @mock.patch("unicode_utils._BIDI_AVAILABLE", True)
    def test_reordering_with_mocked_bidi(self):
        """With python-bidi available, get_display should be called."""
        with mock.patch("unicode_utils.get_display", return_value="reordered") as mock_display:
            # Need to also mock the import inside the function
            with mock.patch.dict("sys.modules", {"bidi": mock.MagicMock(), "bidi.algorithm": mock.MagicMock(get_display=mock_display)}):
                result = reorder_rtl_text("some rtl text")
                assert result == "reordered"

    @mock.patch("unicode_utils._BIDI_AVAILABLE", True)
    def test_reordering_exception_fallback(self):
        """If get_display raises an exception, return original text."""
        with mock.patch("unicode_utils.get_display", side_effect=RuntimeError("bidi error")):
            with mock.patch.dict("sys.modules", {"bidi": mock.MagicMock(), "bidi.algorithm": mock.MagicMock(get_display=mock.MagicMock(side_effect=RuntimeError("bidi error")))}):
                text = "some text"
                result = reorder_rtl_text(text)
                # Should return original on exception
                assert result == text


# ---------------------------------------------------------------------------
# Tests: Pipeline integration
# ---------------------------------------------------------------------------


class TestNormalizeOCRText:
    def test_basic_normalization(self):
        text = "e\u0301"  # e + combining acute
        result = normalize_ocr_text(text, "en")
        assert result == "\u00e9"

    def test_empty_text(self):
        assert normalize_ocr_text("", "en") == ""

    def test_none_text(self):
        assert normalize_ocr_text(None, "en") == ""

    def test_no_lang_code(self):
        result = normalize_ocr_text("hello", "")
        assert result == "hello"

    def test_rtl_language_normalization(self):
        """Arabic text should still get NFC normalized."""
        text = "\u0627\u0644\u0639\u0631\u0628\u064a\u0629"
        result = normalize_ocr_text(text, "ar")
        assert unicodedata.is_normalized("NFC", result)


# ---------------------------------------------------------------------------
# Tests: bidi_available helper
# ---------------------------------------------------------------------------


class TestBidiAvailable:
    def test_returns_bool(self):
        result = bidi_available()
        assert isinstance(result, bool)

    @mock.patch("unicode_utils._BIDI_AVAILABLE", True)
    def test_true_when_available(self):
        assert bidi_available() is True

    @mock.patch("unicode_utils._BIDI_AVAILABLE", False)
    def test_false_when_unavailable(self):
        assert bidi_available() is False


class TestCJKVerticalSupport:
    def test_cjk_vertical_language_detection(self):
        assert is_cjk_vertical_language("japan") is True
        assert is_cjk_vertical_language("ch") is True
        assert is_cjk_vertical_language("chinese_cht") is True
        assert is_cjk_vertical_language("en") is False
        assert "japan" in CJK_VERTICAL_LANGUAGES

    def test_has_cjk_chars(self):
        assert has_cjk_chars("縦書き") is True
        assert has_cjk_chars("テスト") is True
        assert has_cjk_chars("Hello") is False

    def test_reorder_cjk_vertical_lines_right_to_left(self):
        lines = [
            ("二", [[100, 50], [120, 50], [120, 100], [100, 100]], 0.9),
            ("一", [[200, 10], [220, 10], [220, 60], [200, 60]], 0.9),
            ("三", [[100, 105], [120, 105], [120, 155], [100, 155]], 0.9),
            ("二", [[200, 65], [220, 65], [220, 115], [200, 115]], 0.9),
        ]

        reordered = reorder_cjk_vertical_lines(lines, "japan")

        assert [line[0] for line in reordered] == ["一", "二", "二", "三"]

    def test_reorder_cjk_vertical_lines_noop_for_horizontal(self):
        lines = [
            ("横書き", [[10, 10], [110, 10], [110, 30], [10, 30]], 0.9),
            ("テスト", [[10, 40], [110, 40], [110, 60], [10, 60]], 0.9),
        ]

        assert reorder_cjk_vertical_lines(lines, "japan") == lines

    def test_reorder_cjk_vertical_lines_preserves_non_vertical_context(self):
        lines = [
            ("Header", [[20, 10], [180, 10], [180, 30], [20, 30]], 0.9),
            ("二", [[100, 50], [120, 50], [120, 100], [100, 100]], 0.9),
            ("一", [[200, 10], [220, 10], [220, 60], [200, 60]], 0.9),
            ("三", [[100, 105], [120, 105], [120, 155], [100, 155]], 0.9),
            ("二", [[200, 65], [220, 65], [220, 115], [200, 115]], 0.9),
            ("Footer", [[20, 190], [180, 190], [180, 210], [20, 210]], 0.9),
        ]

        reordered = reorder_cjk_vertical_lines(lines, "japan")

        assert [line[0] for line in reordered] == ["Header", "一", "二", "二", "三", "Footer"]

    def test_reorder_cjk_vertical_lines_noop_for_non_cjk_language(self):
        lines = [
            ("A", [[200, 10], [220, 10], [220, 60], [200, 60]], 0.9),
            ("B", [[100, 10], [120, 10], [120, 60], [100, 60]], 0.9),
        ]

        assert reorder_cjk_vertical_lines(lines, "en") == lines
