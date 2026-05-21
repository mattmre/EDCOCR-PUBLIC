"""
Unit tests for CJK vertical text detection module (vertical_text.py).

Tests cover:
- CJK Unicode character detection
- Bounding box geometry analysis
- Page-level text direction classification
- Column grouping for vertical text
- Reading order sorting (vertical + mixed)
- Vertical text crop rotation
- Full text extraction with vertical awareness
- Dataclass defaults and construction
- JSON output format and file paths
- Graceful degradation (None inputs, empty results, no CJK)
- Constants verification

Run with: python -m pytest tests/test_vertical_text.py -v
"""

import json
import os
from unittest import mock

from version import __version__
from vertical_text import (
    CJK_LANGUAGES,
    CJK_UNICODE_RANGES,
    COLUMN_GROUPING_TOLERANCE,
    VERTICAL_ASPECT_RATIO_THRESHOLD,
    DocumentVerticalText,
    VerticalTextAnalysis,
    analyze_page_vertical_text,
    classify_page_text_direction,
    contains_cjk,
    extract_text_vertical_aware,
    finalize_vertical_text,
    group_vertical_columns,
    is_cjk_char,
    is_vertical_text_box,
    rotate_vertical_crop,
    sort_mixed_reading_order,
    sort_vertical_reading_order,
    write_vertical_analysis_json,
)

# ---------------------------------------------------------------------------
# Helpers: build OCR result tuples and mock box points
# ---------------------------------------------------------------------------


def _make_box(x1, y1, x2, y2):
    """Create a 4-point polygon from corner coordinates (PaddleOCR format)."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _make_horizontal_line(text, x1, y1, width=400, height=30, conf=0.90):
    """Create a horizontal OCR line (width >> height)."""
    box = _make_box(x1, y1, x1 + width, y1 + height)
    return (text, box, conf)


def _make_vertical_line(text, x1, y1, width=30, height=200, conf=0.90):
    """Create a vertical OCR line (height >> width, aspect ratio > 2.0)."""
    box = _make_box(x1, y1, x1 + width, y1 + height)
    return (text, box, conf)


def _make_cjk_vertical_line(x_center, y_top, text_len=3, conf=0.90):
    """Create a CJK vertical text line with realistic proportions."""
    text = "\u4e2d" * text_len  # CJK character repeated
    width = 30
    height = text_len * 40 + 20  # Each character ~40px + padding
    x1 = x_center - width // 2
    box = _make_box(x1, y_top, x1 + width, y_top + height)
    return (text, box, conf)


# ---------------------------------------------------------------------------
# Tests: is_cjk_char
# ---------------------------------------------------------------------------


class TestIsCjkChar:
    def test_cjk_unified_ideograph(self):
        assert is_cjk_char("\u4e2d") is True  # Chinese "middle"

    def test_hiragana(self):
        assert is_cjk_char("\u3042") is True  # Hiragana "a"

    def test_katakana(self):
        assert is_cjk_char("\u30a2") is True  # Katakana "a"

    def test_hangul_syllable(self):
        assert is_cjk_char("\uac00") is True  # Hangul "ga"

    def test_fullwidth_form(self):
        assert is_cjk_char("\uff01") is True  # Fullwidth exclamation

    def test_latin_char(self):
        assert is_cjk_char("A") is False

    def test_digit(self):
        assert is_cjk_char("5") is False

    def test_empty_string(self):
        assert is_cjk_char("") is False

    def test_multi_char_string(self):
        assert is_cjk_char("AB") is False

    def test_cjk_symbols(self):
        assert is_cjk_char("\u3001") is True  # Ideographic comma

    def test_extension_a(self):
        assert is_cjk_char("\u3400") is True  # First char of Extension A


# ---------------------------------------------------------------------------
# Tests: contains_cjk
# ---------------------------------------------------------------------------


class TestContainsCjk:
    def test_pure_cjk(self):
        assert contains_cjk("\u4e2d\u6587") is True

    def test_mixed_text(self):
        assert contains_cjk("Hello \u4e16\u754c") is True

    def test_latin_only(self):
        assert contains_cjk("Hello world") is False

    def test_empty_string(self):
        assert contains_cjk("") is False

    def test_none_input(self):
        assert contains_cjk(None) is False

    def test_numbers_only(self):
        assert contains_cjk("12345") is False

    def test_hangul(self):
        assert contains_cjk("\uac00\ub098\ub2e4") is True


# ---------------------------------------------------------------------------
# Tests: is_vertical_text_box
# ---------------------------------------------------------------------------


class TestIsVerticalTextBox:
    def test_clearly_vertical(self):
        """Box with height >> width (aspect ratio > 2.0)."""
        box = _make_box(100, 100, 130, 400)  # width=30, height=300, ratio=10
        assert is_vertical_text_box(box, text="ab") is True

    def test_clearly_horizontal(self):
        """Box with width >> height (aspect ratio < 1)."""
        box = _make_box(100, 100, 500, 130)  # width=400, height=30, ratio=0.075
        assert is_vertical_text_box(box, text="Hello world") is False

    def test_square_box(self):
        """Square box: aspect ratio = 1.0 < 2.0 threshold."""
        box = _make_box(0, 0, 100, 100)  # ratio=1.0
        assert is_vertical_text_box(box, text="ab") is False

    def test_exactly_at_threshold(self):
        """Aspect ratio exactly at 2.0 threshold."""
        box = _make_box(0, 0, 50, 100)  # width=50, height=100, ratio=2.0
        assert is_vertical_text_box(box, text="ab") is True

    def test_just_below_threshold(self):
        """Aspect ratio just below 2.0."""
        box = _make_box(0, 0, 51, 100)  # width=51, height=100, ratio~1.96
        assert is_vertical_text_box(box, text="ab") is False

    def test_single_character_excluded(self):
        """Single character is ambiguous and excluded by default."""
        box = _make_box(100, 100, 130, 400)  # vertical box
        assert is_vertical_text_box(box, text="A") is False

    def test_single_char_with_min_chars_1(self):
        """Allow single chars when min_chars=1."""
        box = _make_box(100, 100, 130, 400)
        assert is_vertical_text_box(box, text="A", min_chars=1) is True

    def test_empty_box_points(self):
        assert is_vertical_text_box(None) is False
        assert is_vertical_text_box([]) is False

    def test_insufficient_points(self):
        assert is_vertical_text_box([[0, 0], [1, 1]]) is False

    def test_zero_width_box(self):
        """Degenerate box with zero width."""
        box = _make_box(100, 100, 100, 300)  # width=0
        assert is_vertical_text_box(box, text="ab") is False

    def test_no_text_defaults_to_geometry_only(self):
        """When text is empty, only geometry is considered (min_chars skipped)."""
        box = _make_box(100, 100, 130, 400)  # vertical
        assert is_vertical_text_box(box, text="") is True  # geometry-only, no text filter


# ---------------------------------------------------------------------------
# Tests: classify_page_text_direction
# ---------------------------------------------------------------------------


class TestClassifyPageDirection:
    def test_all_horizontal(self):
        lines = [
            _make_horizontal_line("Line 1", 10, 10),
            _make_horizontal_line("Line 2", 10, 50),
            _make_horizontal_line("Line 3", 10, 90),
        ]
        result = classify_page_text_direction(lines)
        assert result["direction"] == "horizontal"
        assert result["vertical_count"] == 0
        assert result["horizontal_count"] == 3

    def test_all_vertical(self):
        lines = [
            _make_vertical_line("\u4e2d\u6587\u6d4b", 100, 10),
            _make_vertical_line("\u4e2d\u6587\u6d4b", 200, 10),
            _make_vertical_line("\u4e2d\u6587\u6d4b", 300, 10),
        ]
        result = classify_page_text_direction(lines)
        assert result["direction"] == "vertical"
        assert result["vertical_count"] == 3
        assert result["horizontal_count"] == 0
        assert result["vertical_ratio"] == 1.0

    def test_mixed_page(self):
        """2 vertical + 2 horizontal = 50% -> mixed (>= 20%, < 60%)."""
        lines = [
            _make_vertical_line("\u4e2d\u6587\u6d4b", 100, 10),
            _make_vertical_line("\u4e2d\u6587\u6d4b", 200, 10),
            _make_horizontal_line("English text", 10, 300),
            _make_horizontal_line("More text here", 10, 340),
        ]
        result = classify_page_text_direction(lines)
        assert result["direction"] == "mixed"
        assert result["vertical_count"] == 2
        assert result["horizontal_count"] == 2

    def test_empty_results(self):
        result = classify_page_text_direction([])
        assert result["direction"] == "horizontal"
        assert result["vertical_ratio"] == 0.0

    def test_none_results(self):
        result = classify_page_text_direction(None)
        assert result["direction"] == "horizontal"

    def test_threshold_boundary_60_percent(self):
        """Exactly 60% vertical -> 'vertical' (>= 0.6 threshold)."""
        lines = [
            _make_vertical_line("\u4e2d\u6587\u6d4b", 100, 10),
            _make_vertical_line("\u4e2d\u6587\u6d4b", 200, 10),
            _make_vertical_line("\u4e2d\u6587\u6d4b", 300, 10),
            _make_horizontal_line("English text one", 10, 300),
            _make_horizontal_line("English text two", 10, 340),
        ]
        result = classify_page_text_direction(lines)
        assert result["direction"] == "vertical"
        assert result["vertical_ratio"] == 0.6


# ---------------------------------------------------------------------------
# Tests: group_vertical_columns
# ---------------------------------------------------------------------------


class TestGroupVerticalColumns:
    def test_single_column(self):
        """All boxes at similar x -> one column."""
        boxes = [
            (_make_box(95, 10, 125, 200), "text1", 0.9),
            (_make_box(100, 220, 130, 410), "text2", 0.9),
            (_make_box(98, 430, 128, 620), "text3", 0.9),
        ]
        columns = group_vertical_columns(boxes, page_width=1000)
        assert len(columns) == 1
        assert len(columns[0]) == 3

    def test_two_columns(self):
        """Two groups far apart -> two columns."""
        boxes = [
            (_make_box(800, 10, 830, 200), "col1_a", 0.9),
            (_make_box(805, 220, 835, 410), "col1_b", 0.9),
            (_make_box(400, 10, 430, 200), "col2_a", 0.9),
            (_make_box(405, 220, 435, 410), "col2_b", 0.9),
        ]
        columns = group_vertical_columns(boxes, page_width=1000)
        assert len(columns) == 2

    def test_empty_input(self):
        assert group_vertical_columns([], page_width=1000) == []

    def test_column_items_sorted_by_y(self):
        """Items within each column should be sorted top-to-bottom."""
        boxes = [
            (_make_box(100, 300, 130, 500), "bottom", 0.9),
            (_make_box(100, 10, 130, 200), "top", 0.9),
            (_make_box(100, 150, 130, 340), "middle", 0.9),
        ]
        columns = group_vertical_columns(boxes, page_width=1000)
        assert len(columns) == 1
        texts = [item[1] for item in columns[0]]
        assert texts == ["top", "middle", "bottom"]

    def test_custom_tolerance(self):
        """Wider tolerance groups more columns together."""
        boxes = [
            (_make_box(100, 10, 130, 200), "a", 0.9),
            (_make_box(180, 10, 210, 200), "b", 0.9),  # 80px apart
        ]
        # Default tolerance at 5% of 1000 = 50px -> too far apart
        cols_default = group_vertical_columns(boxes, page_width=1000)
        assert len(cols_default) == 2

        # Wider tolerance = 10% of 1000 = 100px -> grouped together
        cols_wide = group_vertical_columns(boxes, page_width=1000, tolerance=0.1)
        assert len(cols_wide) == 1

    def test_zero_page_width(self):
        """Zero page width falls back to minimum tolerance."""
        boxes = [
            (_make_box(100, 10, 130, 200), "a", 0.9),
        ]
        columns = group_vertical_columns(boxes, page_width=0)
        assert len(columns) == 1


# ---------------------------------------------------------------------------
# Tests: sort_vertical_reading_order
# ---------------------------------------------------------------------------


class TestSortVerticalReadingOrder:
    def test_right_to_left_order(self):
        """Columns with higher x appear first (right-to-left)."""
        col_right = [
            (_make_box(800, 10, 830, 200), "right_top", 0.9),
            (_make_box(800, 220, 830, 410), "right_bottom", 0.9),
        ]
        col_left = [
            (_make_box(200, 10, 230, 200), "left_top", 0.9),
            (_make_box(200, 220, 230, 410), "left_bottom", 0.9),
        ]
        result = sort_vertical_reading_order([col_left, col_right])
        texts = [item[1] for item in result]
        assert texts == ["right_top", "right_bottom", "left_top", "left_bottom"]

    def test_single_column(self):
        column = [
            (_make_box(100, 10, 130, 200), "top", 0.9),
            (_make_box(100, 220, 130, 410), "bottom", 0.9),
        ]
        result = sort_vertical_reading_order([column])
        texts = [item[1] for item in result]
        assert texts == ["top", "bottom"]

    def test_empty_columns(self):
        assert sort_vertical_reading_order([]) == []

    def test_three_columns(self):
        col1 = [(_make_box(900, 10, 930, 200), "col1", 0.9)]
        col2 = [(_make_box(500, 10, 530, 200), "col2", 0.9)]
        col3 = [(_make_box(100, 10, 130, 200), "col3", 0.9)]
        result = sort_vertical_reading_order([col3, col1, col2])
        texts = [item[1] for item in result]
        assert texts == ["col1", "col2", "col3"]

    def test_column_with_empty_list(self):
        """Empty column should be handled gracefully."""
        result = sort_vertical_reading_order([[]])
        assert result == []


# ---------------------------------------------------------------------------
# Tests: sort_mixed_reading_order
# ---------------------------------------------------------------------------


class TestSortMixedReadingOrder:
    def test_horizontal_only(self):
        """All horizontal lines sorted normally."""
        lines = [
            _make_horizontal_line("Line 3", 10, 90),
            _make_horizontal_line("Line 1", 10, 10),
            _make_horizontal_line("Line 2", 10, 50),
        ]
        result = sort_mixed_reading_order(lines, page_width=1000)
        texts = [item[0] for item in result]
        assert texts == ["Line 1", "Line 2", "Line 3"]

    def test_vertical_only(self):
        """All vertical lines grouped and sorted right-to-left."""
        lines = [
            _make_vertical_line("\u4e2d\u6587\u6d4b", 200, 10),
            _make_vertical_line("\u4e2d\u6587\u6d4b", 800, 10),
        ]
        result = sort_mixed_reading_order(lines, page_width=1000)
        # Right column (x=800) should come before left column (x=200)
        assert len(result) == 2

    def test_interleaved_by_y_position(self):
        """Horizontal and vertical text interleaved by vertical position."""
        lines = [
            _make_horizontal_line("Bottom text", 10, 500),
            _make_vertical_line("\u4e2d\u6587\u6d4b", 800, 10),
            _make_horizontal_line("Top text", 10, 5),
        ]
        result = sort_mixed_reading_order(lines, page_width=1000)
        assert len(result) == 3
        # Top text (y=5) should come first
        assert result[0][0] == "Top text"

    def test_empty_input(self):
        assert sort_mixed_reading_order([], page_width=1000) == []

    def test_single_line(self):
        lines = [_make_horizontal_line("Only line", 10, 10)]
        result = sort_mixed_reading_order(lines, page_width=1000)
        assert len(result) == 1
        assert result[0][0] == "Only line"


# ---------------------------------------------------------------------------
# Tests: rotate_vertical_crop
# ---------------------------------------------------------------------------


class TestRotateVerticalCrop:
    def test_ccw_rotation(self):
        """Counter-clockwise 90 rotation swaps dimensions."""
        import vertical_text as vt_module

        if not vt_module._PIL_AVAILABLE:
            return  # skip if PIL not available

        from PIL import Image
        img = Image.new("RGB", (30, 200), color="white")
        rotated = rotate_vertical_crop(img, direction="ccw")
        assert rotated.size == (200, 30)

    def test_cw_rotation(self):
        """Clockwise 90 rotation swaps dimensions."""
        import vertical_text as vt_module

        if not vt_module._PIL_AVAILABLE:
            return

        from PIL import Image
        img = Image.new("RGB", (30, 200), color="white")
        rotated = rotate_vertical_crop(img, direction="cw")
        assert rotated.size == (200, 30)

    def test_square_image(self):
        """Square image remains same size after rotation."""
        import vertical_text as vt_module

        if not vt_module._PIL_AVAILABLE:
            return

        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        rotated = rotate_vertical_crop(img, direction="ccw")
        assert rotated.size == (100, 100)

    def test_none_image(self):
        result = rotate_vertical_crop(None)
        assert result is None

    def test_unknown_direction(self):
        """Unknown direction returns original image unchanged."""
        import vertical_text as vt_module

        if not vt_module._PIL_AVAILABLE:
            return

        from PIL import Image
        img = Image.new("RGB", (30, 200), color="white")
        result = rotate_vertical_crop(img, direction="invalid")
        assert result.size == (30, 200)

    def test_pil_unavailable(self):
        """When PIL is not available, returns original input."""
        import vertical_text as vt_module

        with mock.patch.object(vt_module, "_PIL_AVAILABLE", False):
            result = rotate_vertical_crop("not_an_image")
        assert result == "not_an_image"


# ---------------------------------------------------------------------------
# Tests: extract_text_vertical_aware
# ---------------------------------------------------------------------------


class TestExtractTextVerticalAware:
    def test_horizontal_page_extraction(self):
        lines = [
            _make_horizontal_line("Line 3", 10, 90),
            _make_horizontal_line("Line 1", 10, 10),
            _make_horizontal_line("Line 2", 10, 50),
        ]
        text = extract_text_vertical_aware(lines, page_width=1000)
        assert text == "Line 1\nLine 2\nLine 3"

    def test_vertical_page_extraction(self):
        lines = [
            _make_vertical_line("\u5de6", 200, 10),  # left column
            _make_vertical_line("\u53f3", 800, 10),  # right column
            _make_vertical_line("\u4e2d", 500, 10),  # middle column
        ]
        text = extract_text_vertical_aware(lines, page_width=1000)
        # Right-to-left order: right, middle, left
        lines_out = text.strip().split("\n")
        assert len(lines_out) == 3

    def test_empty_results(self):
        assert extract_text_vertical_aware([], page_width=1000) == ""

    def test_none_results(self):
        assert extract_text_vertical_aware(None, page_width=1000) == ""

    def test_strips_whitespace(self):
        lines = [_make_horizontal_line("  Hello  ", 10, 10)]
        text = extract_text_vertical_aware(lines, page_width=1000)
        assert text == "Hello"


# ---------------------------------------------------------------------------
# Tests: VerticalTextAnalysis dataclass
# ---------------------------------------------------------------------------


class TestVerticalTextAnalysis:
    def test_defaults(self):
        a = VerticalTextAnalysis(page_number=1)
        assert a.page_number == 1
        assert a.direction == "horizontal"
        assert a.vertical_ratio == 0.0
        assert a.vertical_line_count == 0
        assert a.horizontal_line_count == 0
        assert a.column_count == 0
        assert a.lang_detected == ""
        assert a.reading_order_applied is False

    def test_custom_values(self):
        a = VerticalTextAnalysis(
            page_number=3,
            direction="vertical",
            vertical_ratio=0.85,
            vertical_line_count=17,
            horizontal_line_count=3,
            column_count=4,
            lang_detected="japan",
            reading_order_applied=True,
        )
        assert a.direction == "vertical"
        assert a.column_count == 4

    def test_document_vertical_text_defaults(self):
        d = DocumentVerticalText(document_id="d1", source_file="test.pdf")
        assert d.pages == []
        assert d.total_vertical_pages == 0
        assert d.total_mixed_pages == 0
        assert d.has_vertical_content is False


# ---------------------------------------------------------------------------
# Tests: analyze_page_vertical_text
# ---------------------------------------------------------------------------


class TestAnalyzePageVerticalText:
    def test_horizontal_page(self):
        lines = [
            _make_horizontal_line("Line 1", 10, 10),
            _make_horizontal_line("Line 2", 10, 50),
        ]
        result = analyze_page_vertical_text(lines, page_number=1, page_width=1000)
        assert result.direction == "horizontal"
        assert result.column_count == 0
        assert result.reading_order_applied is False

    def test_vertical_page(self):
        lines = [
            _make_vertical_line("\u4e2d\u6587\u6d4b", 800, 10),
            _make_vertical_line("\u4e2d\u6587\u6d4b", 500, 10),
            _make_vertical_line("\u4e2d\u6587\u6d4b", 200, 10),
        ]
        result = analyze_page_vertical_text(lines, page_number=1, page_width=1000, lang="ch")
        assert result.direction == "vertical"
        assert result.vertical_line_count == 3
        assert result.column_count >= 1
        assert result.reading_order_applied is True
        assert result.lang_detected == "ch"

    def test_empty_results(self):
        result = analyze_page_vertical_text([], page_number=1, page_width=1000)
        assert result.direction == "horizontal"
        assert result.page_number == 1


# ---------------------------------------------------------------------------
# Tests: finalize_vertical_text
# ---------------------------------------------------------------------------


class TestFinalizeVerticalText:
    def test_finalize_empty_document(self):
        doc = DocumentVerticalText(document_id="d1", source_file="empty.pdf")
        result = finalize_vertical_text(doc)
        assert result.total_vertical_pages == 0
        assert result.total_mixed_pages == 0
        assert result.has_vertical_content is False

    def test_finalize_with_vertical_pages(self):
        doc = DocumentVerticalText(document_id="d2", source_file="cjk.pdf")
        doc.pages = [
            VerticalTextAnalysis(page_number=1, direction="vertical"),
            VerticalTextAnalysis(page_number=2, direction="horizontal"),
            VerticalTextAnalysis(page_number=3, direction="mixed"),
        ]
        result = finalize_vertical_text(doc)
        assert result.total_vertical_pages == 1
        assert result.total_mixed_pages == 1
        assert result.has_vertical_content is True

    def test_finalize_no_vertical_content(self):
        doc = DocumentVerticalText(document_id="d3", source_file="english.pdf")
        doc.pages = [
            VerticalTextAnalysis(page_number=1, direction="horizontal"),
            VerticalTextAnalysis(page_number=2, direction="horizontal"),
        ]
        result = finalize_vertical_text(doc)
        assert result.has_vertical_content is False

    def test_finalize_with_dict_pages(self):
        doc = DocumentVerticalText(document_id="d4", source_file="dict.pdf")
        doc.pages = [
            {"direction": "vertical", "page_number": 1},
            {"direction": "mixed", "page_number": 2},
        ]
        result = finalize_vertical_text(doc)
        assert result.total_vertical_pages == 1
        assert result.total_mixed_pages == 1
        assert result.has_vertical_content is True


# ---------------------------------------------------------------------------
# Tests: write_vertical_analysis_json
# ---------------------------------------------------------------------------


class TestWriteJson:
    def test_json_file_created(self, tmp_path):
        doc = DocumentVerticalText(document_id="test_id", source_file="subfolder/doc.pdf")
        doc.pages = []
        doc = finalize_vertical_text(doc)
        result = write_vertical_analysis_json(doc, str(tmp_path), "subfolder", "0.9.0")
        assert result is not None
        assert os.path.exists(result)
        assert result.endswith(".vertical.json")

    def test_json_schema_structure(self, tmp_path):
        doc = DocumentVerticalText(document_id="schema_test", source_file="test.pdf")
        p = VerticalTextAnalysis(
            page_number=1, direction="vertical", vertical_ratio=0.8,
            vertical_line_count=8, horizontal_line_count=2, column_count=3,
        )
        doc.pages = [p]
        doc = finalize_vertical_text(doc)
        result_path = write_vertical_analysis_json(doc, str(tmp_path), "", "0.9.0")
        assert result_path is not None

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0"
        assert data["document_id"] == "schema_test"
        assert data["source_file"] == "test.pdf"
        assert "processing" in data
        assert data["processing"]["pipeline_version"] == "0.9.0"
        assert data["processing"]["analysis_engine"] == "geometry_aspect_ratio"
        assert "timestamp" in data["processing"]
        assert "document_summary" in data
        assert data["document_summary"]["total_vertical_pages"] == 1
        assert data["document_summary"]["has_vertical_content"] is True
        assert "pages" in data
        assert len(data["pages"]) == 1
        assert data["pages"][0]["direction"] == "vertical"

    def test_json_subfolder_creation(self, tmp_path):
        doc = DocumentVerticalText(document_id="sub_test", source_file="deep/nested/doc.pdf")
        doc.pages = []
        doc = finalize_vertical_text(doc)
        result = write_vertical_analysis_json(doc, str(tmp_path), "deep/nested", "0.9.0")
        assert result is not None
        assert os.path.exists(result)
        expected_dir = os.path.join(str(tmp_path), "EXPORT", "VERTICAL", "deep", "nested")
        assert os.path.isdir(expected_dir)

    def test_path_traversal_protection(self, tmp_path):
        """Subfolder with '..' segments -- dots are stripped by sanitization."""
        doc = DocumentVerticalText(document_id="traversal_test", source_file="evil.pdf")
        doc.pages = []
        doc = finalize_vertical_text(doc)
        result = write_vertical_analysis_json(doc, str(tmp_path), "../../etc", "0.9.0")
        assert result is not None
        vertical_dir = os.path.join(str(tmp_path), "EXPORT", "VERTICAL")
        assert os.path.realpath(result).startswith(os.path.realpath(vertical_dir))

    def test_empty_subfolder(self, tmp_path):
        doc = DocumentVerticalText(document_id="root_test", source_file="root_doc.pdf")
        doc.pages = []
        doc = finalize_vertical_text(doc)
        result = write_vertical_analysis_json(doc, str(tmp_path), ".", "0.9.0")
        assert result is not None
        assert os.path.exists(result)
        expected_dir = os.path.join(str(tmp_path), "EXPORT", "VERTICAL")
        assert os.path.dirname(result) == expected_dir

    def test_non_pdf_uses_ext_token(self, tmp_path):
        doc = DocumentVerticalText(document_id="img_doc", source_file="img/photo.tiff")
        doc.pages = []
        doc = finalize_vertical_text(doc)
        result = write_vertical_analysis_json(doc, str(tmp_path), "", __version__)
        assert result is not None
        assert os.path.basename(result) == "photo__tiff.vertical.json"


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_none_ocr_results_classify(self):
        result = classify_page_text_direction(None)
        assert result["direction"] == "horizontal"

    def test_none_ocr_results_extract(self):
        assert extract_text_vertical_aware(None, page_width=1000) == ""

    def test_none_ocr_results_analyze(self):
        result = analyze_page_vertical_text(None, page_number=1, page_width=1000)
        assert result.direction == "horizontal"

    def test_malformed_box_in_classify(self):
        """Lines with malformed box data should not crash."""
        lines = [
            ("text", None, 0.9),
            ("text2", [[0, 0]], 0.9),
        ]
        result = classify_page_text_direction(lines)
        assert result["direction"] == "horizontal"
        assert result["horizontal_count"] == 2

    def test_empty_group_columns(self):
        assert group_vertical_columns([], page_width=1000) == []

    def test_sort_empty_columns(self):
        assert sort_vertical_reading_order([]) == []

    def test_mixed_sort_empty(self):
        assert sort_mixed_reading_order([], page_width=1000) == []

    def test_write_json_with_none_pages(self, tmp_path):
        """Document with no pages should still produce valid JSON."""
        doc = DocumentVerticalText(document_id="empty", source_file="empty.pdf")
        doc = finalize_vertical_text(doc)
        result = write_vertical_analysis_json(doc, str(tmp_path), "", "0.9.0")
        assert result is not None
        with open(result, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["pages"] == []


# ---------------------------------------------------------------------------
# Tests: Constants verification
# ---------------------------------------------------------------------------


class TestConstants:
    def test_vertical_aspect_ratio_threshold(self):
        assert VERTICAL_ASPECT_RATIO_THRESHOLD == 2.0

    def test_column_grouping_tolerance(self):
        assert COLUMN_GROUPING_TOLERANCE == 0.05

    def test_cjk_languages_set(self):
        assert CJK_LANGUAGES == {"ch", "chinese_cht", "japan", "korean"}

    def test_cjk_unicode_ranges_count(self):
        assert len(CJK_UNICODE_RANGES) == 7

    def test_unicode_ranges_are_valid(self):
        """All ranges should have start <= end."""
        for start, end in CJK_UNICODE_RANGES:
            assert start <= end, f"Invalid range: {start:#x}-{end:#x}"


# ---------------------------------------------------------------------------
# Tests: End-to-end pipeline
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline_vertical_document(self, tmp_path):
        """Full pipeline: analyze pages -> finalize -> write JSON."""
        # Page 1: vertical CJK content
        page1_lines = [
            _make_vertical_line("\u4e2d\u6587\u6d4b", 800, 10),
            _make_vertical_line("\u4e2d\u6587\u6d4b", 500, 10),
            _make_vertical_line("\u4e2d\u6587\u6d4b", 200, 10),
        ]
        analysis1 = analyze_page_vertical_text(page1_lines, 1, page_width=1000, lang="ch")

        # Page 2: horizontal content
        page2_lines = [
            _make_horizontal_line("English text", 10, 10),
            _make_horizontal_line("More text", 10, 50),
        ]
        analysis2 = analyze_page_vertical_text(page2_lines, 2, page_width=1000, lang="en")

        doc = DocumentVerticalText(document_id="e2e_doc", source_file="evidence/cjk_scan.pdf")
        doc.pages = [analysis1, analysis2]
        doc = finalize_vertical_text(doc)

        assert doc.total_vertical_pages == 1
        assert doc.total_mixed_pages == 0
        assert doc.has_vertical_content is True

        result_path = write_vertical_analysis_json(doc, str(tmp_path), "evidence", "0.9.0")
        assert result_path is not None
        assert os.path.exists(result_path)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["document_summary"]["total_vertical_pages"] == 1
        assert data["document_summary"]["has_vertical_content"] is True
        assert len(data["pages"]) == 2

    def test_full_pipeline_no_vertical(self, tmp_path):
        """All horizontal document produces correct summary."""
        doc = DocumentVerticalText(document_id="horiz_doc", source_file="english.pdf")
        for i in range(3):
            lines = [_make_horizontal_line(f"Line {j}", 10, 10 + j * 40) for j in range(5)]
            analysis = analyze_page_vertical_text(lines, i + 1, page_width=1000)
            doc.pages.append(analysis)

        doc = finalize_vertical_text(doc)
        assert doc.has_vertical_content is False

        result_path = write_vertical_analysis_json(doc, str(tmp_path), "", "0.9.0")
        assert result_path is not None
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["document_summary"]["has_vertical_content"] is False
