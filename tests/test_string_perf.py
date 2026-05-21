"""Regression tests for O(n) string join replacing O(n^2) += concat.

Verifies that the list-accumulator + join pattern produces identical
output to the old += concatenation for all three OCR text-building
paths (PaddleOCR first-pass, PaddleOCR second-pass/rerun, Tesseract
fallback) and the assembler full_text builder.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Helpers: simulate old (+=) and new (list join) text building
# ---------------------------------------------------------------------------

def _old_paddle_concat(lines: list[tuple[str, list, float]]) -> str:
    """Original O(n^2) += concatenation pattern from worker_thread PaddleOCR path."""
    text_content = ""
    for txt, _box, _conf in lines:
        # Simulates: if insert_text_line(...): text_content += normalized_txt + " "
        text_content += txt + " "
    return text_content


def _new_paddle_join(lines: list[tuple[str, list, float]]) -> str:
    """New O(n) list-accumulator pattern."""
    _text_parts: list[str] = []
    for txt, _box, _conf in lines:
        _text_parts.append(txt)
    text_content = " ".join(_text_parts)
    if text_content:
        text_content += " "
    return text_content


def _old_tesseract_concat(words: list[str]) -> str:
    """Original O(n^2) += concatenation pattern from Tesseract fallback path."""
    text_content = ""
    for word in words:
        text_content += word + " "
    return text_content


def _new_tesseract_join(words: list[str]) -> str:
    """New O(n) list-accumulator pattern for Tesseract path."""
    _tess_parts: list[str] = []
    for word in words:
        _tess_parts.append(word)
    text_content = " ".join(_tess_parts)
    if text_content:
        text_content += " "
    return text_content


def _old_assembler_concat(page_texts: dict[int, str], total_pages: int) -> str:
    """Original O(n^2) += concatenation in _finalize_doc assembler loop."""
    full_text = ""
    for p in range(1, total_pages + 1):
        full_text += page_texts.get(p, "") + "\n\f"
    return full_text


def _new_assembler_join(page_texts: dict[int, str], total_pages: int) -> str:
    """New O(n) list-accumulator pattern for assembler loop."""
    _full_text_parts: list[str] = []
    for p in range(1, total_pages + 1):
        _full_text_parts.append(page_texts.get(p, "") + "\n\f")
    full_text = "".join(_full_text_parts)
    return full_text


# ---------------------------------------------------------------------------
# Mock data generators
# ---------------------------------------------------------------------------

def _make_paddle_lines(n: int) -> list[tuple[str, list, float]]:
    """Generate n mock PaddleOCR line tuples: (text, bbox, confidence)."""
    return [(f"word{i}", [0, 0, 100, 20], 0.95) for i in range(n)]


def _make_tesseract_words(n: int) -> list[str]:
    """Generate n mock Tesseract word strings."""
    return [f"tess{i}" for i in range(n)]


def _make_page_texts(n: int) -> dict[int, str]:
    """Generate mock page text dict for n pages."""
    return {p: f"Page {p} text content here." for p in range(1, n + 1)}


# ---------------------------------------------------------------------------
# PaddleOCR path tests
# ---------------------------------------------------------------------------

class TestPaddleTextBuild:
    """Verify PaddleOCR text_content output is identical between old and new patterns."""

    @pytest.mark.parametrize("n", [0, 1, 5, 10, 50, 100, 500])
    def test_paddle_output_identical(self, n: int) -> None:
        lines = _make_paddle_lines(n)
        old = _old_paddle_concat(lines)
        new = _new_paddle_join(lines)
        assert old == new, f"Mismatch at n={n}: old={old!r} new={new!r}"

    def test_paddle_empty_lines(self) -> None:
        old = _old_paddle_concat([])
        new = _new_paddle_join([])
        assert old == new == ""

    def test_paddle_single_line(self) -> None:
        lines = [("hello", [0, 0, 50, 10], 0.99)]
        old = _old_paddle_concat(lines)
        new = _new_paddle_join(lines)
        assert old == new == "hello "

    def test_paddle_unicode_text(self) -> None:
        lines = [
            ("\u4f60\u597d", [0, 0, 50, 10], 0.95),
            ("\u4e16\u754c", [0, 20, 50, 30], 0.92),
        ]
        old = _old_paddle_concat(lines)
        new = _new_paddle_join(lines)
        assert old == new

    def test_paddle_trailing_space_preserved(self) -> None:
        """The old pattern always ends with a trailing space."""
        lines = _make_paddle_lines(3)
        result = _new_paddle_join(lines)
        assert result.endswith(" ")

    def test_paddle_words_space_separated(self) -> None:
        lines = _make_paddle_lines(3)
        result = _new_paddle_join(lines)
        # "word0 word1 word2 " -- each word separated by exactly one space
        assert result == "word0 word1 word2 "


# ---------------------------------------------------------------------------
# Tesseract path tests
# ---------------------------------------------------------------------------

class TestTesseractTextBuild:
    """Verify Tesseract text_content output is identical between old and new patterns."""

    @pytest.mark.parametrize("n", [0, 1, 5, 10, 50, 100, 500])
    def test_tesseract_output_identical(self, n: int) -> None:
        words = _make_tesseract_words(n)
        old = _old_tesseract_concat(words)
        new = _new_tesseract_join(words)
        assert old == new, f"Mismatch at n={n}: old={old!r} new={new!r}"

    def test_tesseract_empty(self) -> None:
        old = _old_tesseract_concat([])
        new = _new_tesseract_join([])
        assert old == new == ""

    def test_tesseract_single_word(self) -> None:
        old = _old_tesseract_concat(["invoice"])
        new = _new_tesseract_join(["invoice"])
        assert old == new == "invoice "


# ---------------------------------------------------------------------------
# Assembler full_text tests
# ---------------------------------------------------------------------------

class TestAssemblerTextBuild:
    """Verify assembler full_text output is identical between old and new patterns."""

    @pytest.mark.parametrize("n", [0, 1, 5, 10, 50, 100, 500])
    def test_assembler_output_identical(self, n: int) -> None:
        page_texts = _make_page_texts(n)
        old = _old_assembler_concat(page_texts, n)
        new = _new_assembler_join(page_texts, n)
        assert old == new, f"Mismatch at n={n}"

    def test_assembler_empty_pages(self) -> None:
        """Pages with no text should still produce form feed separators."""
        old = _old_assembler_concat({}, 3)
        new = _new_assembler_join({}, 3)
        assert old == new == "\n\f\n\f\n\f"

    def test_assembler_sparse_pages(self) -> None:
        """Only some pages have text; missing pages get empty string."""
        page_texts = {1: "First page", 3: "Third page"}
        old = _old_assembler_concat(page_texts, 4)
        new = _new_assembler_join(page_texts, 4)
        assert old == new

    def test_assembler_form_feed_separators(self) -> None:
        page_texts = {1: "Page one"}
        result = _new_assembler_join(page_texts, 1)
        assert result == "Page one\n\f"
