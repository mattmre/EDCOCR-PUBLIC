"""
Unit tests for handwriting detection module (handwriting.py).

Tests cover:
- Dataclass defaults and construction
- Confidence heuristic detection
- Geometry analysis detection
- Image variance detection (with mocked OpenCV)
- Signal merging / weighted voting
- Finalization and summary stats
- JSON output format and file paths
- Graceful degradation (missing OpenCV, empty input)

Run with: python -m pytest tests/test_handwriting.py -v
"""

import json
import os
from unittest import mock

from handwriting import (
    HANDWRITING_CONFIDENCE_THRESHOLD,
    HANDWRITING_PAGE_THRESHOLD,
    DocumentHandwriting,
    HandwritingRegion,
    PageHandwriting,
    detect_handwriting_by_confidence,
    detect_handwriting_by_geometry,
    detect_handwriting_by_image,
    finalize_handwriting,
    merge_handwriting_signals,
    write_handwriting_json,
)
from version import __version__

# ---------------------------------------------------------------------------
# Helpers: build paddle_lines tuples
# ---------------------------------------------------------------------------


def _make_line(text, confidence, x1, y1, x2, y2):
    """Shortcut to create a single paddle_lines tuple."""
    return (text, confidence, [x1, y1, x2, y2])


def _make_printed_lines(n, start_y=0, spacing=50, height=40):
    """Generate *n* evenly-spaced printed lines (high confidence)."""
    lines = []
    for i in range(n):
        y1 = start_y + i * spacing
        y2 = y1 + height
        lines.append(_make_line(f"Printed line {i}", 0.92, 100, y1, 800, y2))
    return lines


def _make_handwritten_lines(n, start_y=0, spacing=50, height=40):
    """Generate *n* evenly-spaced handwritten lines (low confidence)."""
    lines = []
    for i in range(n):
        y1 = start_y + i * spacing
        y2 = y1 + height
        lines.append(_make_line(f"Handwritten line {i}", 0.35, 100, y1, 800, y2))
    return lines


# ---------------------------------------------------------------------------
# Tests: Dataclass defaults
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_handwriting_region_defaults(self):
        r = HandwritingRegion()
        assert r.bbox == []
        assert r.confidence == 0.0
        assert r.text == ""
        assert r.ocr_confidence == 0.0
        assert r.detection_method == ""

    def test_page_handwriting_defaults(self):
        p = PageHandwriting(page_num=1)
        assert p.page_num == 1
        assert p.has_handwriting is False
        assert p.handwriting_coverage == 0.0
        assert p.handwriting_regions == []
        assert p.printed_line_count == 0
        assert p.handwritten_line_count == 0
        assert p.detection_methods_used == []

    def test_document_handwriting_defaults(self):
        d = DocumentHandwriting(document_id="doc1", source_file="test.pdf")
        assert d.document_id == "doc1"
        assert d.source_file == "test.pdf"
        assert d.pages == []
        assert d.total_handwritten_pages == 0
        assert d.total_pages_with_mixed == 0
        assert d.overall_handwriting_coverage == 0.0
        assert d.is_primarily_handwritten is False


# ---------------------------------------------------------------------------
# Tests: Confidence-based detection
# ---------------------------------------------------------------------------


class TestConfidenceDetection:
    def test_all_printed_lines(self):
        lines = _make_printed_lines(5)
        result = detect_handwriting_by_confidence(lines, page_num=1)
        assert result.has_handwriting is False
        assert result.printed_line_count == 5
        assert result.handwritten_line_count == 0
        assert result.handwriting_regions == []

    def test_all_handwritten_lines(self):
        lines = _make_handwritten_lines(5)
        result = detect_handwriting_by_confidence(lines, page_num=1)
        assert result.has_handwriting is True
        assert result.handwritten_line_count == 5
        assert result.printed_line_count == 0
        assert len(result.handwriting_regions) == 5

    def test_mixed_lines(self):
        # 3 printed + 2 handwritten = 40% handwritten = at HANDWRITING_PAGE_THRESHOLD
        lines = _make_printed_lines(3) + _make_handwritten_lines(2, start_y=150)
        result = detect_handwriting_by_confidence(lines, page_num=1)
        assert result.printed_line_count == 3
        assert result.handwritten_line_count == 2
        # 2/5 = 0.4 which meets >= 0.40 threshold
        assert result.has_handwriting is True

    def test_empty_lines(self):
        result = detect_handwriting_by_confidence([], page_num=1)
        assert result.has_handwriting is False
        assert result.printed_line_count == 0
        assert result.handwritten_line_count == 0

    def test_single_printed_line(self):
        lines = [_make_line("Hello world", 0.95, 10, 10, 200, 50)]
        result = detect_handwriting_by_confidence(lines, page_num=1)
        assert result.printed_line_count == 1
        assert result.handwritten_line_count == 0
        assert result.has_handwriting is False

    def test_single_handwritten_line(self):
        lines = [_make_line("Scrawled note", 0.30, 10, 10, 200, 50)]
        result = detect_handwriting_by_confidence(lines, page_num=1)
        assert result.handwritten_line_count == 1
        assert result.printed_line_count == 0
        # 1/1 = 1.0 >= 0.40 threshold
        assert result.has_handwriting is True

    def test_threshold_boundary_exact(self):
        """Line at exactly HANDWRITING_CONFIDENCE_THRESHOLD (0.65) is printed (>= threshold)."""
        lines = [_make_line("Boundary line", 0.65, 0, 0, 100, 40)]
        result = detect_handwriting_by_confidence(lines, page_num=1)
        # The code uses `confidence < HANDWRITING_CONFIDENCE_THRESHOLD`, so 0.65 is NOT below
        assert result.printed_line_count == 1
        assert result.handwritten_line_count == 0

    def test_threshold_boundary_below(self):
        """Line at 0.649 is below threshold and flagged as handwritten."""
        lines = [_make_line("Just below", 0.649, 0, 0, 100, 40)]
        result = detect_handwriting_by_confidence(lines, page_num=1)
        assert result.handwritten_line_count == 1
        assert result.printed_line_count == 0

    def test_handwriting_coverage_calculation(self):
        """Verify coverage ratio: handwritten bbox area / page area."""
        # One handwritten line with bbox [0, 0, 100, 50] -> area = 5000
        # Page image_size = (200, 100) -> area = 20000
        # Expected coverage = 5000 / 20000 = 0.25
        lines = [_make_line("Handwritten", 0.30, 0, 0, 100, 50)]
        result = detect_handwriting_by_confidence(lines, page_num=1, image_size=(200, 100))
        assert result.handwriting_coverage == 0.25

    def test_page_threshold_not_met(self):
        """Few handwritten lines below HANDWRITING_PAGE_THRESHOLD -> has_handwriting=False."""
        # 1 handwritten + 9 printed = 10% < 40% threshold
        lines = _make_handwritten_lines(1) + _make_printed_lines(9, start_y=50)
        result = detect_handwriting_by_confidence(lines, page_num=1)
        assert result.handwritten_line_count == 1
        assert result.printed_line_count == 9
        # 1/10 = 0.10 < 0.40
        assert result.has_handwriting is False

    def test_page_threshold_met(self):
        """Enough handwritten lines to meet threshold."""
        # 5 handwritten + 5 printed = 50% >= 40% threshold
        lines = _make_handwritten_lines(5) + _make_printed_lines(5, start_y=250)
        result = detect_handwriting_by_confidence(lines, page_num=1)
        assert result.handwritten_line_count == 5
        assert result.printed_line_count == 5
        assert result.has_handwriting is True

    def test_regions_have_correct_detection_method(self):
        lines = _make_handwritten_lines(2)
        result = detect_handwriting_by_confidence(lines, page_num=1)
        for region in result.handwriting_regions:
            assert region["detection_method"] == "confidence_heuristic"
        assert "confidence_heuristic" in result.detection_methods_used


# ---------------------------------------------------------------------------
# Tests: Geometry-based detection
# ---------------------------------------------------------------------------


class TestGeometryDetection:
    def test_regular_spacing_printed(self):
        """Evenly spaced lines with uniform heights -> printed."""
        lines = _make_printed_lines(6, start_y=0, spacing=50, height=40)
        result = detect_handwriting_by_geometry(lines, page_num=1)
        assert result["is_handwritten"] is False
        assert result["spacing_cv"] < 0.45
        assert result["height_cv"] < 0.35

    def test_irregular_spacing_handwritten(self):
        """Variable spacing between lines -> handwriting indicator."""
        lines = [
            _make_line("Line 1", 0.50, 100, 0, 800, 40),
            _make_line("Line 2", 0.50, 100, 120, 800, 160),   # large gap
            _make_line("Line 3", 0.50, 100, 165, 800, 205),   # tiny gap
            _make_line("Line 4", 0.50, 100, 400, 800, 440),   # large gap
            _make_line("Line 5", 0.50, 100, 445, 800, 485),   # tiny gap
        ]
        result = detect_handwriting_by_geometry(lines, page_num=1)
        # The spacing CV should exceed the threshold due to wide variance
        assert result["spacing_cv"] > 0.0

    def test_variable_heights_handwritten(self):
        """Variable character heights -> high height_cv."""
        lines = [
            _make_line("Tall", 0.50, 100, 0, 800, 80),    # h=80
            _make_line("Short", 0.50, 100, 90, 800, 105),  # h=15
            _make_line("Mid", 0.50, 100, 115, 800, 155),   # h=40
            _make_line("Tiny", 0.50, 100, 165, 800, 175),  # h=10
            _make_line("Big", 0.50, 100, 185, 800, 280),   # h=95
        ]
        result = detect_handwriting_by_geometry(lines, page_num=1)
        assert result["height_cv"] > 0.35
        assert result["is_handwritten"] is True

    def test_uniform_heights_printed(self):
        """Uniform heights -> low height_cv."""
        lines = _make_printed_lines(6, spacing=50, height=40)
        result = detect_handwriting_by_geometry(lines, page_num=1)
        # All lines have identical height (40), so height_cv should be 0.0
        assert result["height_cv"] == 0.0

    def test_too_few_lines(self):
        """Less than 3 lines -> safe default (is_handwritten=False)."""
        lines = _make_printed_lines(2)
        result = detect_handwriting_by_geometry(lines, page_num=1)
        assert result["is_handwritten"] is False

    def test_single_line(self):
        lines = [_make_line("One line", 0.90, 0, 0, 500, 40)]
        result = detect_handwriting_by_geometry(lines, page_num=1)
        assert result["is_handwritten"] is False

    def test_empty_lines_geometry(self):
        result = detect_handwriting_by_geometry([], page_num=1)
        assert result["is_handwritten"] is False
        assert result["spacing_cv"] == 0.0
        assert result["height_cv"] == 0.0

    def test_returns_expected_keys(self):
        lines = _make_printed_lines(5)
        result = detect_handwriting_by_geometry(lines, page_num=1)
        assert "is_handwritten" in result
        assert "spacing_cv" in result
        assert "height_cv" in result


# ---------------------------------------------------------------------------
# Tests: Image variance detection (mocked OpenCV)
# ---------------------------------------------------------------------------


class TestImageVarianceDetection:
    def test_with_cv2_available(self):
        """When cv2 is available and processes normally, returns meaningful result."""
        import handwriting as hw_module

        # Build a mock cv2 module
        mock_cv2 = mock.MagicMock()
        mock_cv2.Canny.return_value = "edges"
        # Simulate 10 contours, each with >= 5 points
        mock_contours = [mock.MagicMock() for _ in range(10)]
        for c in mock_contours:
            c.__len__ = mock.MagicMock(return_value=6)
        mock_cv2.findContours.return_value = (mock_contours, None)
        # Return varying stroke widths to simulate handwriting
        widths = [(None, (w, 5.0), None) for w in [2.0, 5.0, 1.0, 8.0, 3.0, 6.0, 2.0, 9.0, 1.0, 4.0]]
        mock_cv2.minAreaRect.side_effect = widths
        mock_cv2.COLOR_BGR2GRAY = 6
        mock_cv2.RETR_EXTERNAL = 0
        mock_cv2.CHAIN_APPROX_SIMPLE = 1

        mock_np = mock.MagicMock()
        mock_np.array.return_value = mock.MagicMock(shape=(100, 200))
        mock_np.ndarray = type(mock.MagicMock())

        # Create a mock PIL image
        mock_image = mock.MagicMock()
        mock_image.convert.return_value = mock_image

        with mock.patch.object(hw_module, "_CV2_AVAILABLE", True), \
             mock.patch.object(hw_module, "cv2", mock_cv2, create=True), \
             mock.patch.object(hw_module, "np", mock_np, create=True):
            result = detect_handwriting_by_image(mock_image, page_num=1)

        assert "is_handwritten" in result
        assert "stroke_variance" in result
        assert isinstance(result["stroke_variance"], float)

    def test_without_cv2_available(self):
        """When cv2 is unavailable, returns safe default."""
        import handwriting as hw_module

        with mock.patch.object(hw_module, "_CV2_AVAILABLE", False):
            result = detect_handwriting_by_image(None, page_num=1)

        assert result["is_handwritten"] is False
        assert result["stroke_variance"] == 0.0

    def test_image_as_numpy_array(self):
        """Passing a numpy array (not PIL) should work when cv2 is available."""
        import handwriting as hw_module

        mock_cv2 = mock.MagicMock()
        mock_cv2.Canny.return_value = "edges"
        mock_cv2.cvtColor.return_value = "gray"
        mock_cv2.findContours.return_value = ([], None)
        mock_cv2.COLOR_BGR2GRAY = 6
        mock_cv2.RETR_EXTERNAL = 0
        mock_cv2.CHAIN_APPROX_SIMPLE = 1

        import numpy as real_np
        test_array = real_np.zeros((100, 200, 3), dtype=real_np.uint8)

        with mock.patch.object(hw_module, "_CV2_AVAILABLE", True), \
             mock.patch.object(hw_module, "cv2", mock_cv2, create=True), \
             mock.patch.object(hw_module, "np", real_np, create=True):
            result = detect_handwriting_by_image(test_array, page_num=1)

        # Fewer than 5 contours -> returns default
        assert result["is_handwritten"] is False
        assert result["stroke_variance"] == 0.0

    def test_image_processing_error(self):
        """When cv2 raises an exception, returns graceful default."""
        import handwriting as hw_module

        mock_cv2 = mock.MagicMock()
        mock_cv2.Canny.side_effect = RuntimeError("OpenCV internal error")

        mock_np = mock.MagicMock()
        mock_np.array.return_value = mock.MagicMock(shape=(100, 200))
        mock_np.ndarray = type(mock.MagicMock())

        mock_image = mock.MagicMock()
        mock_image.convert.return_value = mock_image

        with mock.patch.object(hw_module, "_CV2_AVAILABLE", True), \
             mock.patch.object(hw_module, "cv2", mock_cv2, create=True), \
             mock.patch.object(hw_module, "np", mock_np, create=True):
            result = detect_handwriting_by_image(mock_image, page_num=1)

        assert result["is_handwritten"] is False
        assert result["stroke_variance"] == 0.0


# ---------------------------------------------------------------------------
# Tests: Signal merging
# ---------------------------------------------------------------------------


class TestSignalMerging:
    def _make_confidence_result(self, has_handwriting, page_num=1, regions=None, methods=None):
        """Create a PageHandwriting to use as confidence_result."""
        p = PageHandwriting(page_num=page_num)
        p.has_handwriting = has_handwriting
        p.handwriting_regions = regions or []
        p.detection_methods_used = methods or (["confidence_heuristic"] if has_handwriting else [])
        p.printed_line_count = 0 if has_handwriting else 5
        p.handwritten_line_count = 5 if has_handwriting else 0
        p.handwriting_coverage = 0.5 if has_handwriting else 0.0
        return p

    def test_all_signals_agree_handwritten(self):
        conf = self._make_confidence_result(has_handwriting=True)
        geo = {"is_handwritten": True, "spacing_cv": 0.6, "height_cv": 0.5}
        img = {"is_handwritten": True, "stroke_variance": 0.4}

        merged = merge_handwriting_signals(conf, geo, img, page_num=1)
        # Score = 0.5 + 0.3 + 0.2 = 1.0 >= 0.5
        assert merged.has_handwriting is True

    def test_all_signals_agree_printed(self):
        conf = self._make_confidence_result(has_handwriting=False)
        geo = {"is_handwritten": False}
        img = {"is_handwritten": False}

        merged = merge_handwriting_signals(conf, geo, img, page_num=1)
        # Score = 0.0 < 0.5
        assert merged.has_handwriting is False

    def test_confidence_overrides_geometry(self):
        """Confidence weight (0.5) >= threshold (0.5) alone, so it wins."""
        conf = self._make_confidence_result(has_handwriting=True)
        geo = {"is_handwritten": False}
        img = {"is_handwritten": False}

        merged = merge_handwriting_signals(conf, geo, img, page_num=1)
        # Score = 0.5 >= 0.5
        assert merged.has_handwriting is True

    def test_geometry_and_image_override_confidence(self):
        """Geometry (0.3) + Image (0.2) = 0.5 >= 0.5 threshold."""
        conf = self._make_confidence_result(has_handwriting=False)
        geo = {"is_handwritten": True}
        img = {"is_handwritten": True}

        merged = merge_handwriting_signals(conf, geo, img, page_num=1)
        # Score = 0.0 + 0.3 + 0.2 = 0.5 >= 0.5
        assert merged.has_handwriting is True

    def test_none_geometry_result(self):
        """geometry_result=None -> uses only confidence and image."""
        conf = self._make_confidence_result(has_handwriting=True)
        img = {"is_handwritten": False}

        merged = merge_handwriting_signals(conf, None, img, page_num=1)
        # Score = 0.5 >= 0.5
        assert merged.has_handwriting is True

    def test_none_image_result(self):
        """image_result=None -> uses only confidence and geometry."""
        conf = self._make_confidence_result(has_handwriting=False)
        geo = {"is_handwritten": True}

        merged = merge_handwriting_signals(conf, geo, None, page_num=1)
        # Score = 0.0 + 0.3 = 0.3 < 0.5
        assert merged.has_handwriting is False

    def test_all_none_signals(self):
        """Only confidence_result available, geometry and image are None."""
        conf = self._make_confidence_result(has_handwriting=True)

        merged = merge_handwriting_signals(conf, None, None, page_num=1)
        # Score = 0.5 >= 0.5
        assert merged.has_handwriting is True

    def test_merged_detection_methods(self):
        """detection_methods_used should list all methods that contributed."""
        conf = self._make_confidence_result(has_handwriting=True)
        geo = {"is_handwritten": True}
        img = {"is_handwritten": True}

        merged = merge_handwriting_signals(conf, geo, img, page_num=1)
        assert "confidence_heuristic" in merged.detection_methods_used
        assert "geometry" in merged.detection_methods_used
        assert "image_variance" in merged.detection_methods_used


# ---------------------------------------------------------------------------
# Tests: Finalization
# ---------------------------------------------------------------------------


class TestFinalization:
    def test_finalize_empty_document(self):
        doc = DocumentHandwriting(document_id="d1", source_file="empty.pdf")
        result = finalize_handwriting(doc)
        assert result.total_handwritten_pages == 0
        assert result.total_pages_with_mixed == 0
        assert result.overall_handwriting_coverage == 0.0
        assert result.is_primarily_handwritten is False

    def test_finalize_all_handwritten(self):
        doc = DocumentHandwriting(document_id="d2", source_file="hw.pdf")
        doc.pages = [
            PageHandwriting(page_num=1, has_handwriting=True, handwriting_coverage=0.8,
                            handwritten_line_count=10, printed_line_count=0),
            PageHandwriting(page_num=2, has_handwriting=True, handwriting_coverage=0.9,
                            handwritten_line_count=8, printed_line_count=0),
        ]
        result = finalize_handwriting(doc)
        assert result.total_handwritten_pages == 2
        assert result.is_primarily_handwritten is True
        assert result.overall_handwriting_coverage == round((0.8 + 0.9) / 2, 4)

    def test_finalize_mixed_pages(self):
        doc = DocumentHandwriting(document_id="d3", source_file="mixed.pdf")
        doc.pages = [
            PageHandwriting(page_num=1, has_handwriting=True, handwriting_coverage=0.6,
                            handwritten_line_count=5, printed_line_count=3),
            PageHandwriting(page_num=2, has_handwriting=False, handwriting_coverage=0.0,
                            handwritten_line_count=0, printed_line_count=10),
            PageHandwriting(page_num=3, has_handwriting=True, handwriting_coverage=0.7,
                            handwritten_line_count=7, printed_line_count=2),
        ]
        result = finalize_handwriting(doc)
        assert result.total_handwritten_pages == 2
        # Pages with both handwritten AND printed lines
        assert result.total_pages_with_mixed == 2  # page 1 and page 3
        # 2/3 > 1.5 -> is_primarily_handwritten
        assert result.is_primarily_handwritten is True

    def test_finalize_no_handwriting(self):
        doc = DocumentHandwriting(document_id="d4", source_file="typed.pdf")
        doc.pages = [
            PageHandwriting(page_num=1, has_handwriting=False, handwriting_coverage=0.0,
                            printed_line_count=10, handwritten_line_count=0),
            PageHandwriting(page_num=2, has_handwriting=False, handwriting_coverage=0.0,
                            printed_line_count=12, handwritten_line_count=0),
        ]
        result = finalize_handwriting(doc)
        assert result.total_handwritten_pages == 0
        assert result.is_primarily_handwritten is False
        assert result.overall_handwriting_coverage == 0.0

    def test_overall_coverage_calculation(self):
        """Average coverage across all pages."""
        doc = DocumentHandwriting(document_id="d5", source_file="avg.pdf")
        doc.pages = [
            PageHandwriting(page_num=1, has_handwriting=True, handwriting_coverage=0.2,
                            handwritten_line_count=2, printed_line_count=0),
            PageHandwriting(page_num=2, has_handwriting=True, handwriting_coverage=0.6,
                            handwritten_line_count=6, printed_line_count=0),
            PageHandwriting(page_num=3, has_handwriting=False, handwriting_coverage=0.0,
                            printed_line_count=5, handwritten_line_count=0),
        ]
        result = finalize_handwriting(doc)
        expected = round((0.2 + 0.6 + 0.0) / 3, 4)
        assert result.overall_handwriting_coverage == expected

    def test_pages_with_mixed_content(self):
        """Pages with has_handwriting=True but also printed lines -> mixed."""
        doc = DocumentHandwriting(document_id="d6", source_file="mix.pdf")
        doc.pages = [
            PageHandwriting(page_num=1, has_handwriting=True, handwriting_coverage=0.4,
                            handwritten_line_count=4, printed_line_count=6),
        ]
        result = finalize_handwriting(doc)
        assert result.total_pages_with_mixed == 1


# ---------------------------------------------------------------------------
# Tests: Finalization with dict pages
# ---------------------------------------------------------------------------


class TestFinalizationDictPages:
    def test_finalize_with_dict_pages(self):
        """finalize_handwriting should accept dict pages as well as dataclass instances."""
        doc = DocumentHandwriting(document_id="d7", source_file="dictpages.pdf")
        doc.pages = [
            {
                "page_num": 1,
                "has_handwriting": True,
                "handwriting_coverage": 0.5,
                "handwritten_line_count": 5,
                "printed_line_count": 3,
            },
            {
                "page_num": 2,
                "has_handwriting": False,
                "handwriting_coverage": 0.0,
                "handwritten_line_count": 0,
                "printed_line_count": 10,
            },
        ]
        result = finalize_handwriting(doc)
        assert result.total_handwritten_pages == 1
        assert result.total_pages_with_mixed == 1
        assert result.is_primarily_handwritten is False


# ---------------------------------------------------------------------------
# Tests: write_handwriting_json
# ---------------------------------------------------------------------------


class TestWriteJson:
    def test_json_file_created(self, tmp_path):
        doc = DocumentHandwriting(document_id="test_id", source_file="subfolder/doc.pdf")
        doc.pages = []
        doc = finalize_handwriting(doc)
        result = write_handwriting_json(doc, str(tmp_path), "subfolder", "0.5.0")
        assert result is not None
        assert os.path.exists(result)
        assert result.endswith(".handwriting.json")

    def test_non_pdf_uses_ext_token(self, tmp_path):
        doc = DocumentHandwriting(document_id="img_doc", source_file="img/photo.tiff")
        doc.pages = []
        doc = finalize_handwriting(doc)
        result = write_handwriting_json(doc, str(tmp_path), "", __version__)
        assert result is not None
        assert os.path.basename(result) == "photo__tiff.handwriting.json"

    def test_json_schema_structure(self, tmp_path):
        doc = DocumentHandwriting(document_id="schema_test", source_file="test.pdf")
        p = PageHandwriting(page_num=1, has_handwriting=True, handwriting_coverage=0.3,
                            handwritten_line_count=3, printed_line_count=7)
        doc.pages = [p]
        doc = finalize_handwriting(doc)
        result_path = write_handwriting_json(doc, str(tmp_path), "", "0.5.0")
        assert result_path is not None

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0"
        assert data["document_id"] == "schema_test"
        assert data["source_file"] == "test.pdf"
        assert "processing" in data
        assert data["processing"]["pipeline_version"] == "0.5.0"
        assert "detection_engine" in data["processing"]
        assert "timestamp" in data["processing"]
        assert "document_summary" in data
        assert "total_handwritten_pages" in data["document_summary"]
        assert "total_pages_with_mixed" in data["document_summary"]
        assert "overall_handwriting_coverage" in data["document_summary"]
        assert "is_primarily_handwritten" in data["document_summary"]
        assert "pages" in data
        assert len(data["pages"]) == 1

    def test_json_subfolder_creation(self, tmp_path):
        doc = DocumentHandwriting(document_id="sub_test", source_file="deep/nested/doc.pdf")
        doc.pages = []
        doc = finalize_handwriting(doc)
        result = write_handwriting_json(doc, str(tmp_path), "deep/nested", "0.5.0")
        assert result is not None
        assert os.path.exists(result)
        expected_dir = os.path.join(str(tmp_path), "EXPORT", "HANDWRITING", "deep", "nested")
        assert os.path.isdir(expected_dir)

    def test_path_traversal_protection(self, tmp_path):
        """Subfolder with '..' segments -- dots are stripped by sanitization so
        '../../etc' becomes just 'etc' (a safe child dir), not a traversal escape."""
        doc = DocumentHandwriting(document_id="traversal_test", source_file="evil.pdf")
        doc.pages = []
        doc = finalize_handwriting(doc)
        result = write_handwriting_json(doc, str(tmp_path), "../../etc", "0.5.0")
        # The _sanitize_path_segment strips leading/trailing dots, so '..' becomes ''
        # and is filtered out. The remaining 'etc' is a safe subdirectory name.
        assert result is not None
        handwriting_dir = os.path.join(str(tmp_path), "EXPORT", "HANDWRITING")
        # Verify the resolved path stays under HANDWRITING dir
        assert os.path.realpath(result).startswith(os.path.realpath(handwriting_dir))
        assert os.path.basename(result) == "evil.handwriting.json"

    def test_empty_subfolder(self, tmp_path):
        doc = DocumentHandwriting(document_id="root_test", source_file="root_doc.pdf")
        doc.pages = []
        doc = finalize_handwriting(doc)
        result = write_handwriting_json(doc, str(tmp_path), ".", "0.5.0")
        assert result is not None
        assert os.path.exists(result)
        expected_dir = os.path.join(str(tmp_path), "EXPORT", "HANDWRITING")
        assert os.path.dirname(result) == expected_dir


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_empty_paddle_lines_all_functions(self):
        """All functions handle empty input without error."""
        conf = detect_handwriting_by_confidence([], page_num=1)
        assert conf.has_handwriting is False

        geo = detect_handwriting_by_geometry([], page_num=1)
        assert geo["is_handwritten"] is False

        import handwriting as hw_module
        with mock.patch.object(hw_module, "_CV2_AVAILABLE", False):
            img = detect_handwriting_by_image(None, page_num=1)
        assert img["is_handwritten"] is False

    def test_none_image_for_image_detection(self):
        """None image with cv2 available -> safe default (not a PIL image or ndarray)."""
        import handwriting as hw_module

        mock_cv2 = mock.MagicMock()
        mock_np = mock.MagicMock()
        mock_np.ndarray = type(mock.MagicMock())

        with mock.patch.object(hw_module, "_CV2_AVAILABLE", True), \
             mock.patch.object(hw_module, "cv2", mock_cv2, create=True), \
             mock.patch.object(hw_module, "np", mock_np, create=True):
            result = detect_handwriting_by_image(None, page_num=1)

        # None has no .convert and is not np.ndarray -> returns default
        assert result["is_handwritten"] is False
        assert result["stroke_variance"] == 0.0

    def test_large_input_no_crash(self):
        """1000 lines should complete without error."""
        lines = []
        for i in range(1000):
            conf = 0.30 if i % 3 == 0 else 0.90
            lines.append(_make_line(f"Line {i}", conf, 0, i * 30, 800, i * 30 + 25))
        result = detect_handwriting_by_confidence(lines, page_num=1)
        assert result.printed_line_count + result.handwritten_line_count == 1000

    def test_negative_confidence_values(self):
        """Lines with negative confidence should be handled gracefully."""
        lines = [
            _make_line("Negative conf", -0.5, 0, 0, 100, 40),
            _make_line("Normal", 0.90, 0, 50, 100, 90),
        ]
        result = detect_handwriting_by_confidence(lines, page_num=1)
        # Negative confidence is < 0.65 threshold, so counted as handwritten
        assert result.handwritten_line_count == 1
        assert result.printed_line_count == 1


# ---------------------------------------------------------------------------
# Tests: Integration-style (end-to-end pipeline)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline_handwritten_document(self, tmp_path):
        """Full pipeline: detect -> merge -> finalize -> write JSON."""
        # Page 1: all handwritten
        page1_lines = _make_handwritten_lines(8)
        conf1 = detect_handwriting_by_confidence(page1_lines, page_num=1, image_size=(1000, 1000))
        geo1 = detect_handwriting_by_geometry(page1_lines, page_num=1)
        merged1 = merge_handwriting_signals(conf1, geo1, None, page_num=1)

        # Page 2: all printed
        page2_lines = _make_printed_lines(10)
        conf2 = detect_handwriting_by_confidence(page2_lines, page_num=2, image_size=(1000, 1000))
        geo2 = detect_handwriting_by_geometry(page2_lines, page_num=2)
        merged2 = merge_handwriting_signals(conf2, geo2, None, page_num=2)

        doc = DocumentHandwriting(document_id="e2e_doc", source_file="evidence/scan.pdf")
        doc.pages = [merged1, merged2]
        doc = finalize_handwriting(doc)

        assert doc.total_handwritten_pages == 1
        assert doc.is_primarily_handwritten is False

        result_path = write_handwriting_json(doc, str(tmp_path), "evidence", "0.5.0")
        assert result_path is not None
        assert os.path.exists(result_path)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["document_summary"]["total_handwritten_pages"] == 1
        assert data["document_summary"]["is_primarily_handwritten"] is False
        assert len(data["pages"]) == 2

    def test_full_pipeline_primarily_handwritten(self, tmp_path):
        """When majority of pages are handwritten, is_primarily_handwritten=True."""
        doc = DocumentHandwriting(document_id="hw_heavy", source_file="notes.pdf")
        # 3 handwritten pages, 1 printed
        for i in range(1, 4):
            lines = _make_handwritten_lines(6)
            conf = detect_handwriting_by_confidence(lines, page_num=i, image_size=(1000, 1000))
            doc.pages.append(conf)
        printed_lines = _make_printed_lines(8)
        conf_p = detect_handwriting_by_confidence(printed_lines, page_num=4, image_size=(1000, 1000))
        doc.pages.append(conf_p)

        doc = finalize_handwriting(doc)
        assert doc.total_handwritten_pages == 3
        assert doc.is_primarily_handwritten is True

        result_path = write_handwriting_json(doc, str(tmp_path), "", "0.5.0")
        assert result_path is not None
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["document_summary"]["is_primarily_handwritten"] is True


# ---------------------------------------------------------------------------
# Tests: Constants verification
# ---------------------------------------------------------------------------


class TestConstants:
    def test_handwriting_confidence_threshold(self):
        assert HANDWRITING_CONFIDENCE_THRESHOLD == 0.65

    def test_handwriting_page_threshold(self):
        assert HANDWRITING_PAGE_THRESHOLD == 0.40
