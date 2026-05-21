"""
Unit tests for per-region table OCR fallback module (table_fallback.py).

Tests cover:
- Dataclass defaults and construction
- Table region cropping (padding, edge clamping, invalid inputs)
- Confidence assessment (inside bbox, outside, partial overlap, empty)
- Preprocessing strategies (enhance, binarize, denoise, graceful degradation)
- Retry strategy building (engine alternation, preprocessing escalation)
- Fallback trigger logic (threshold comparison)
- Result evaluation (confidence comparison, text length tiebreaker)
- Page-level analysis (multiple tables, none, mixed quality)
- Document-level finalization (summary aggregation)
- JSON output (file creation, path safety, schema)
- Graceful degradation (None inputs, missing deps)
- Constants (threshold values, max retries)

Run with: python -m pytest tests/test_table_fallback.py -v
"""

import json
import os
from unittest import mock

from PIL import Image

from table_fallback import (
    ENABLE_TABLE_FALLBACK,
    TABLE_FALLBACK_CONFIDENCE_THRESHOLD,
    TABLE_FALLBACK_MAX_RETRIES,
    DocumentTableFallbackSummary,
    PageTableFallbackAnalysis,
    TableFallbackResult,
    TableRegion,
    analyze_page_tables,
    assess_table_confidence,
    build_retry_strategy,
    crop_table_region,
    evaluate_fallback_result,
    finalize_table_fallback,
    preprocess_table_region,
    should_trigger_fallback,
    write_table_fallback_json,
)
from version import __version__

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(width=800, height=600, color="white"):
    """Create a simple PIL test image."""
    return Image.new("RGB", (width, height), color=color)


def _make_ocr_line(x1, y1, x2, y2, text, confidence):
    """Create a flat-format OCR line: (text, [x1, y1, x2, y2], confidence)."""
    return (text, [x1, y1, x2, y2], confidence)


def _make_ocr_line_poly(points, text, confidence):
    """Create a polygon-format OCR line: (text, [[x1,y1],...], confidence)."""
    return (text, points, confidence)


def _make_table_region(bbox=(100, 100, 200, 150), page=1, conf=0.5, text="table", engine="paddle"):
    """Create a TableRegion with sensible defaults."""
    return TableRegion(
        bbox=bbox,
        page_number=page,
        original_confidence=conf,
        original_text=text,
        original_engine=engine,
    )


# ---------------------------------------------------------------------------
# Tests: Dataclass defaults
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_table_region_defaults(self):
        r = TableRegion(bbox=(0, 0, 100, 50))
        assert r.bbox == (0, 0, 100, 50)
        assert r.page_number == 0
        assert r.original_confidence == 0.0
        assert r.original_text == ""
        assert r.original_engine == ""

    def test_table_fallback_result_defaults(self):
        r = TableFallbackResult()
        assert r.final_text == ""
        assert r.final_confidence == 0.0
        assert r.final_engine == ""
        assert r.attempts == []
        assert r.improved is False
        assert r.fallback_applied is False

    def test_page_analysis_defaults(self):
        a = PageTableFallbackAnalysis()
        assert a.page_number == 0
        assert a.table_count == 0
        assert a.fallback_triggered == 0
        assert a.fallback_improved == 0
        assert a.results == []

    def test_document_summary_defaults(self):
        s = DocumentTableFallbackSummary()
        assert s.document_id == ""
        assert s.source_file == ""
        assert s.total_pages == 0
        assert s.total_tables == 0
        assert s.total_fallback_triggered == 0
        assert s.total_fallback_improved == 0
        assert s.pages == []

    def test_table_region_with_values(self):
        r = TableRegion(
            bbox=(10, 20, 300, 400),
            page_number=3,
            original_confidence=0.75,
            original_text="Hello",
            original_engine="tesseract",
        )
        assert r.bbox == (10, 20, 300, 400)
        assert r.page_number == 3
        assert r.original_confidence == 0.75
        assert r.original_text == "Hello"
        assert r.original_engine == "tesseract"


# ---------------------------------------------------------------------------
# Tests: crop_table_region
# ---------------------------------------------------------------------------


class TestCropTableRegion:
    def test_basic_crop(self):
        img = _make_image(800, 600)
        cropped = crop_table_region(img, (100, 100, 200, 150), padding=0)
        assert cropped is not None
        assert cropped.size == (200, 150)

    def test_crop_with_padding(self):
        img = _make_image(800, 600)
        cropped = crop_table_region(img, (100, 100, 200, 150), padding=10)
        assert cropped is not None
        # With padding: (90, 90) to (310, 260) -> 220 x 170
        assert cropped.size == (220, 170)

    def test_crop_padding_clamped_at_edge(self):
        img = _make_image(800, 600)
        # Table near top-left corner
        cropped = crop_table_region(img, (5, 5, 100, 80), padding=20)
        assert cropped is not None
        # Left/top clamped to 0, right = min(800, 5+100+20)=125, bottom = min(600, 5+80+20)=105
        assert cropped.size == (125, 105)

    def test_crop_none_image(self):
        result = crop_table_region(None, (10, 10, 50, 50))
        assert result is None

    def test_crop_zero_width(self):
        img = _make_image(800, 600)
        result = crop_table_region(img, (100, 100, 0, 50))
        assert result is None

    def test_crop_negative_height(self):
        img = _make_image(800, 600)
        result = crop_table_region(img, (100, 100, 50, -10))
        assert result is None

    def test_crop_full_image(self):
        img = _make_image(200, 100)
        cropped = crop_table_region(img, (0, 0, 200, 100), padding=0)
        assert cropped is not None
        assert cropped.size == (200, 100)


# ---------------------------------------------------------------------------
# Tests: assess_table_confidence
# ---------------------------------------------------------------------------


class TestAssessTableConfidence:
    def test_lines_inside_bbox(self):
        bbox = (50, 50, 300, 200)
        lines = [
            _make_ocr_line(100, 80, 250, 100, "Row 1", 0.9),
            _make_ocr_line(100, 130, 250, 150, "Row 2", 0.8),
        ]
        conf = assess_table_confidence(lines, bbox)
        assert abs(conf - 0.85) < 0.001

    def test_lines_outside_bbox(self):
        bbox = (50, 50, 100, 100)
        lines = [
            _make_ocr_line(400, 400, 500, 420, "Outside", 0.95),
        ]
        conf = assess_table_confidence(lines, bbox)
        assert conf == 0.0

    def test_partial_overlap_center_inside(self):
        bbox = (0, 0, 200, 200)
        lines = [
            _make_ocr_line(50, 50, 150, 80, "Inside center", 0.7),
        ]
        conf = assess_table_confidence(lines, bbox)
        assert abs(conf - 0.7) < 0.001

    def test_empty_lines(self):
        bbox = (0, 0, 100, 100)
        assert assess_table_confidence([], bbox) == 0.0

    def test_empty_bbox(self):
        lines = [_make_ocr_line(10, 10, 50, 30, "text", 0.9)]
        assert assess_table_confidence(lines, ()) == 0.0

    def test_polygon_format(self):
        bbox = (0, 0, 500, 500)
        lines = [
            _make_ocr_line_poly(
                [[100, 100], [200, 100], [200, 130], [100, 130]],
                "Poly text",
                0.88,
            ),
        ]
        conf = assess_table_confidence(lines, bbox)
        assert abs(conf - 0.88) < 0.001

    def test_mixed_inside_outside(self):
        bbox = (0, 0, 200, 200)
        lines = [
            _make_ocr_line(50, 50, 150, 80, "Inside", 0.9),
            _make_ocr_line(500, 500, 600, 520, "Outside", 0.3),
        ]
        conf = assess_table_confidence(lines, bbox)
        # Only the inside line counts
        assert abs(conf - 0.9) < 0.001


# ---------------------------------------------------------------------------
# Tests: preprocess_table_region
# ---------------------------------------------------------------------------


class TestPreprocessTableRegion:
    def test_enhance_strategy(self):
        img = _make_image(100, 100, "gray")
        result = preprocess_table_region(img, strategy="enhance")
        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    def test_binarize_strategy_with_cv2(self):
        img = _make_image(100, 100, "gray")
        # This may or may not have OpenCV; either way it should not crash
        result = preprocess_table_region(img, strategy="binarize")
        assert result is not None
        assert isinstance(result, Image.Image)

    def test_denoise_strategy_with_cv2(self):
        img = _make_image(100, 100, "gray")
        result = preprocess_table_region(img, strategy="denoise")
        assert result is not None
        assert isinstance(result, Image.Image)

    def test_unknown_strategy_falls_back_to_enhance(self):
        img = _make_image(100, 100, "gray")
        result = preprocess_table_region(img, strategy="unknown_strategy")
        assert result is not None
        assert isinstance(result, Image.Image)

    def test_none_image_returns_none(self):
        result = preprocess_table_region(None, strategy="enhance")
        assert result is None

    def test_binarize_without_cv2(self):
        img = _make_image(100, 100)
        with mock.patch("table_fallback._CV2_AVAILABLE", False):
            result = preprocess_table_region(img, strategy="binarize")
        assert result is not None
        assert isinstance(result, Image.Image)

    def test_denoise_without_cv2(self):
        img = _make_image(100, 100)
        with mock.patch("table_fallback._CV2_AVAILABLE", False):
            result = preprocess_table_region(img, strategy="denoise")
        assert result is not None
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# Tests: build_retry_strategy
# ---------------------------------------------------------------------------


class TestBuildRetryStrategy:
    def test_paddle_original_tries_tesseract_first(self):
        steps = build_retry_strategy("paddle", 0.5)
        assert len(steps) >= 1
        assert steps[0]["engine"] == "tesseract"
        assert steps[0]["preprocessing"] == "enhance"

    def test_tesseract_original_tries_paddle_first(self):
        steps = build_retry_strategy("tesseract", 0.5)
        assert len(steps) >= 1
        assert steps[0]["engine"] == "paddle"

    def test_low_confidence_uses_binarize(self):
        steps = build_retry_strategy("paddle", 0.3)
        assert len(steps) == 2
        assert steps[1]["preprocessing"] == "binarize"

    def test_moderate_confidence_uses_denoise(self):
        steps = build_retry_strategy("paddle", 0.5)
        assert len(steps) == 2
        assert steps[1]["preprocessing"] == "denoise"

    def test_strategy_limited_to_max_retries(self):
        steps = build_retry_strategy("paddle", 0.1)
        assert len(steps) <= TABLE_FALLBACK_MAX_RETRIES

    def test_second_step_uses_original_engine(self):
        steps = build_retry_strategy("paddle", 0.5)
        assert len(steps) == 2
        assert steps[1]["engine"] == "paddle"


# ---------------------------------------------------------------------------
# Tests: should_trigger_fallback
# ---------------------------------------------------------------------------


class TestShouldTriggerFallback:
    def test_below_threshold(self):
        assert should_trigger_fallback(0.3) is True

    def test_above_threshold(self):
        assert should_trigger_fallback(0.9) is False

    def test_exact_threshold(self):
        # At exactly the threshold, should NOT trigger (not strictly below)
        assert should_trigger_fallback(TABLE_FALLBACK_CONFIDENCE_THRESHOLD) is False

    def test_custom_threshold_below(self):
        assert should_trigger_fallback(0.7, threshold=0.8) is True

    def test_custom_threshold_above(self):
        assert should_trigger_fallback(0.9, threshold=0.8) is False

    def test_zero_confidence(self):
        assert should_trigger_fallback(0.0) is True

    def test_confidence_one(self):
        assert should_trigger_fallback(1.0) is False


# ---------------------------------------------------------------------------
# Tests: evaluate_fallback_result
# ---------------------------------------------------------------------------


class TestEvaluateFallbackResult:
    def test_higher_confidence_uses_fallback(self):
        result = evaluate_fallback_result(0.4, "original", 0.8, "better")
        assert result["use_fallback"] is True
        assert result["reason"] == "higher confidence"
        assert result["confidence_delta"] > 0

    def test_lower_confidence_keeps_original(self):
        result = evaluate_fallback_result(0.8, "original", 0.3, "worse")
        assert result["use_fallback"] is False
        assert result["reason"] == "fallback did not improve confidence"

    def test_close_confidence_more_text_uses_fallback(self):
        result = evaluate_fallback_result(0.7, "short", 0.72, "longer text here")
        assert result["use_fallback"] is True
        assert "more text" in result["reason"]

    def test_close_confidence_less_text_keeps_original(self):
        result = evaluate_fallback_result(0.7, "longer original text", 0.72, "short")
        assert result["use_fallback"] is False

    def test_empty_original_text(self):
        result = evaluate_fallback_result(0.5, "", 0.52, "new text")
        assert result["use_fallback"] is True

    def test_empty_new_text(self):
        result = evaluate_fallback_result(0.5, "original text", 0.52, "")
        assert result["use_fallback"] is False

    def test_both_empty_text_close_confidence(self):
        result = evaluate_fallback_result(0.5, "", 0.52, "")
        assert result["use_fallback"] is False

    def test_confidence_delta_calculated(self):
        result = evaluate_fallback_result(0.4, "a", 0.7, "b")
        assert abs(result["confidence_delta"] - 0.3) < 0.001

    def test_none_text_handled(self):
        result = evaluate_fallback_result(0.5, None, 0.52, None)
        assert "use_fallback" in result


# ---------------------------------------------------------------------------
# Tests: analyze_page_tables
# ---------------------------------------------------------------------------


class TestAnalyzePageTables:
    def test_no_tables(self):
        analysis = analyze_page_tables(1, [], [])
        assert analysis.page_number == 1
        assert analysis.table_count == 0
        assert analysis.fallback_triggered == 0
        assert analysis.results == []

    def test_single_good_table(self):
        region = _make_table_region(
            bbox=(0, 0, 500, 500), conf=0.9, engine="paddle"
        )
        lines = [_make_ocr_line(100, 100, 200, 120, "Good text", 0.95)]
        analysis = analyze_page_tables(1, [region], lines)
        assert analysis.table_count == 1
        assert analysis.fallback_triggered == 0

    def test_single_bad_table_triggers_fallback(self):
        region = _make_table_region(
            bbox=(0, 0, 500, 500), conf=0.3, engine="paddle"
        )
        lines = [_make_ocr_line(100, 100, 200, 120, "Bad text", 0.2)]
        analysis = analyze_page_tables(1, [region], lines)
        assert analysis.table_count == 1
        assert analysis.fallback_triggered == 1
        assert len(analysis.results) == 1
        result = analysis.results[0]
        assert result["fallback_applied"] is True
        assert len(result["attempts"]) > 0

    def test_multiple_tables_mixed_quality(self):
        good = _make_table_region(bbox=(0, 0, 200, 200), conf=0.9, engine="paddle")
        bad = _make_table_region(bbox=(300, 0, 200, 200), conf=0.3, engine="paddle")
        lines = [
            _make_ocr_line(50, 50, 150, 80, "Good", 0.95),
            _make_ocr_line(350, 50, 450, 80, "Bad", 0.2),
        ]
        analysis = analyze_page_tables(1, [good, bad], lines)
        assert analysis.table_count == 2
        assert analysis.fallback_triggered == 1

    def test_fallback_uses_original_confidence_if_no_lines(self):
        region = _make_table_region(
            bbox=(0, 0, 100, 100), conf=0.3, engine="tesseract"
        )
        # No OCR lines overlap the bbox, so original_confidence is used
        analysis = analyze_page_tables(1, [region], [])
        assert analysis.fallback_triggered == 1


# ---------------------------------------------------------------------------
# Tests: finalize_table_fallback
# ---------------------------------------------------------------------------


class TestFinalization:
    def test_empty_pages(self):
        summary = finalize_table_fallback([])
        assert summary.total_pages == 0
        assert summary.total_tables == 0

    def test_single_page_summary(self):
        page = PageTableFallbackAnalysis(
            page_number=1,
            table_count=3,
            fallback_triggered=1,
            fallback_improved=1,
        )
        summary = finalize_table_fallback(
            [page], document_id="doc1", source_file="test.pdf"
        )
        assert summary.total_pages == 1
        assert summary.total_tables == 3
        assert summary.total_fallback_triggered == 1
        assert summary.total_fallback_improved == 1
        assert summary.document_id == "doc1"

    def test_multi_page_aggregation(self):
        pages = [
            PageTableFallbackAnalysis(
                page_number=1, table_count=2, fallback_triggered=1, fallback_improved=0
            ),
            PageTableFallbackAnalysis(
                page_number=2, table_count=5, fallback_triggered=3, fallback_improved=2
            ),
        ]
        summary = finalize_table_fallback(pages)
        assert summary.total_pages == 2
        assert summary.total_tables == 7
        assert summary.total_fallback_triggered == 4
        assert summary.total_fallback_improved == 2

    def test_dict_page_input(self):
        pages = [
            {"table_count": 2, "fallback_triggered": 1, "fallback_improved": 0},
        ]
        summary = finalize_table_fallback(pages)
        assert summary.total_tables == 2
        assert summary.total_fallback_triggered == 1


# ---------------------------------------------------------------------------
# Tests: write_table_fallback_json
# ---------------------------------------------------------------------------


class TestWriteJson:
    def test_basic_write(self, tmp_path):
        summary = DocumentTableFallbackSummary(
            document_id="doc1",
            source_file="test.pdf",
            total_pages=1,
            total_tables=2,
        )
        result = write_table_fallback_json(
            summary, str(tmp_path), ".", __version__
        )
        assert result is not None
        assert os.path.isfile(result)

        with open(result, encoding="utf-8") as f:
            data = json.load(f)
        assert data["schema_version"] == "1.0"
        assert data["document_id"] == "doc1"
        assert data["processing"]["table_fallback_enabled"] is True

    def test_write_with_subfolder(self, tmp_path):
        summary = DocumentTableFallbackSummary(
            document_id="doc2",
            source_file="folder/test.pdf",
        )
        result = write_table_fallback_json(
            summary, str(tmp_path), "sub/folder", __version__
        )
        assert result is not None
        assert "sub" in result or "folder" in result

    def test_write_path_traversal_blocked(self, tmp_path):
        summary = DocumentTableFallbackSummary(
            document_id="bad",
            source_file="test.pdf",
        )
        result = write_table_fallback_json(
            summary, str(tmp_path), "../../etc", __version__
        )
        # Should be blocked or safely contained
        if result is not None:
            resolved = os.path.realpath(result)
            fallback_dir = os.path.realpath(
                os.path.join(str(tmp_path), "EXPORT", "TABLE_FALLBACK")
            )
            assert resolved.startswith(fallback_dir)

    def test_json_schema_fields(self, tmp_path):
        summary = DocumentTableFallbackSummary(
            document_id="doc3",
            source_file="report.pdf",
            total_pages=2,
            total_tables=4,
            total_fallback_triggered=1,
            total_fallback_improved=1,
        )
        result = write_table_fallback_json(
            summary, str(tmp_path), ".", __version__
        )
        assert result is not None
        with open(result, encoding="utf-8") as f:
            data = json.load(f)

        assert "processing" in data
        assert "document_summary" in data
        assert data["document_summary"]["total_tables"] == 4
        assert data["processing"]["confidence_threshold"] == TABLE_FALLBACK_CONFIDENCE_THRESHOLD
        assert data["processing"]["max_retries"] == TABLE_FALLBACK_MAX_RETRIES


# ---------------------------------------------------------------------------
# Tests: graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_crop_with_invalid_bbox_type(self):
        img = _make_image(100, 100)
        # Pass something that will cause unpacking to fail
        result = crop_table_region(img, "not_a_tuple")
        assert result is None

    def test_assess_confidence_none_lines(self):
        result = assess_table_confidence(None, (0, 0, 100, 100))
        assert result == 0.0

    def test_assess_confidence_malformed_line(self):
        # Line with fewer than 3 elements
        lines = [([0, 0, 50, 50], "text")]
        result = assess_table_confidence(lines, (0, 0, 100, 100))
        assert result == 0.0

    def test_analyze_with_none_regions(self):
        analysis = analyze_page_tables(1, None, [])
        assert analysis.table_count == 0

    def test_preprocess_exception_returns_original(self):
        img = _make_image(50, 50)
        with mock.patch("table_fallback._preprocess_enhance", side_effect=RuntimeError("boom")):
            result = preprocess_table_region(img, "enhance")
        # Should return original image on exception
        assert result is img


# ---------------------------------------------------------------------------
# Tests: constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_disabled(self):
        # Default should be disabled (opt-in)
        assert ENABLE_TABLE_FALLBACK is False

    def test_confidence_threshold_value(self):
        assert TABLE_FALLBACK_CONFIDENCE_THRESHOLD == 0.6

    def test_max_retries_value(self):
        assert TABLE_FALLBACK_MAX_RETRIES == 2

    def test_threshold_is_float(self):
        assert isinstance(TABLE_FALLBACK_CONFIDENCE_THRESHOLD, float)

    def test_max_retries_is_int(self):
        assert isinstance(TABLE_FALLBACK_MAX_RETRIES, int)


# ---------------------------------------------------------------------------
# Tests: retry_table_with_engine (strategy only)
# ---------------------------------------------------------------------------


class TestRetryStrategyEdgeCases:
    def test_unknown_engine_treated_as_non_paddle(self):
        steps = build_retry_strategy("easyocr", 0.5)
        # If original is not "paddle", alternate should be "paddle"
        assert steps[0]["engine"] == "paddle"

    def test_empty_engine_string(self):
        steps = build_retry_strategy("", 0.5)
        assert steps[0]["engine"] == "paddle"

    def test_very_low_confidence_strategy(self):
        steps = build_retry_strategy("paddle", 0.1)
        assert len(steps) >= 1
        # Very low confidence should produce binarize on second step
        if len(steps) > 1:
            assert steps[1]["preprocessing"] == "binarize"


# ---------------------------------------------------------------------------
# Tests: evaluate edge cases
# ---------------------------------------------------------------------------


class TestEvaluateEdgeCases:
    def test_identical_confidence_identical_text(self):
        result = evaluate_fallback_result(0.5, "same", 0.5, "same")
        assert result["use_fallback"] is False
        assert result["confidence_delta"] == 0.0

    def test_identical_confidence_empty_both(self):
        result = evaluate_fallback_result(0.5, "", 0.5, "")
        assert result["use_fallback"] is False

    def test_zero_confidence_both(self):
        result = evaluate_fallback_result(0.0, "", 0.0, "")
        assert result["use_fallback"] is False

    def test_max_confidence_improvement(self):
        result = evaluate_fallback_result(0.0, "", 1.0, "perfect")
        assert result["use_fallback"] is True
        assert result["confidence_delta"] == 1.0
