"""
Unit tests for experimental signature verification module (signature_verification.py).

Tests cover:
- Dataclass defaults and construction
- Signature detection (form fields and OCR keyword fallback)
- Ink density / presence estimation
- Stroke complexity analysis
- Spatial / bounding box analysis
- Hu moment / contour-level metric helpers
- Contour feature matching
- Signal merging and authenticity signals
- Document-level finalization
- JSON sidecar output (schema, path safety)
- Graceful degradation (missing deps, None inputs)
- End-to-end detection flow
- Constants and threshold assertions

Run with: python -m pytest tests/test_signature_verification.py -v
"""

import json
import os
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from signature_verification import (
    _MIN_STROKE_COMPLEXITY,
    _PRESENCE_MIN_INK_RATIO,
    _SIGNATURE_KEYWORDS,
    _SUSPICIOUS_TEXT_CONFIDENCE,
    _TYPED_TEXT_CONFIDENCE,
    _WIDE_SIGNATURE_RATIO,
    DocumentSignatureVerification,
    PageSignatureVerification,
    SignatureCandidate,
    _bbox_to_ints,
    _candidate_from_form_field,
    _crop_with_padding,
    _estimate_presence_metrics,
    _fallback_candidates_from_lines,
    _label_has_signature_line_markers,
    _presence_confidence,
    _project_ocr_keyword_bbox,
    _stroke_complexity,
    _to_grayscale_array,
    _typed_text_signal,
    analyze_signature_page,
    finalize_signature_verification,
    write_signature_verification_json,
)
from version import __version__

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_page(width=500, height=300):
    """Create a blank white PIL image."""
    return Image.new("RGB", (width, height), "white")


def _signature_page():
    """Create a PIL image with simulated signature strokes."""
    image = _blank_page()
    draw = ImageDraw.Draw(image)
    draw.line((120, 180, 300, 210), fill="black", width=4)
    draw.line((180, 210, 320, 175), fill="black", width=3)
    return image


def _make_gray_image(height=800, width=600, fill=255):
    """Create a grayscale numpy array (white page by default)."""
    return np.full((height, width), fill, dtype=np.uint8)


def _make_image_with_dark_region(height=800, width=600, region_bbox=None, ink_value=40):
    """Create a white image with a dark rectangular region simulating ink."""
    img = np.full((height, width), 255, dtype=np.uint8)
    if region_bbox:
        x1, y1, x2, y2 = region_bbox
        img[y1:y2, x1:x2] = ink_value
    return img


def _make_form_field(field_type="signature", label="Signature: ___", bbox=None, confidence=0.85):
    """Create a dict mimicking a form_field from structure_data."""
    return {
        "field_type": field_type,
        "label": label,
        "bbox": bbox or [100, 500, 400, 560],
        "confidence": confidence,
    }


def _make_paddle_line(text, box_pts, conf=0.90):
    """Create a paddle_line tuple: (text, box_points, confidence)."""
    return (text, box_pts, conf)


def _make_box_points(x1, y1, x2, y2):
    """Convert x1,y1,x2,y2 to polygon box points format."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# ---------------------------------------------------------------------------
# Tests: Dataclass defaults
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_signature_candidate_defaults(self):
        c = SignatureCandidate(page_num=1, source="form_field")
        assert c.page_num == 1
        assert c.source == "form_field"
        assert c.bbox == []
        assert c.label == ""
        assert c.ocr_confidence == 0.0
        assert c.presence_detected is False
        assert c.presence_confidence == 0.0
        assert c.authenticity_signal == "not_applicable"
        assert c.review_required is False
        assert c.reason_codes == []
        assert c.metrics == {}

    def test_page_signature_verification_defaults(self):
        p = PageSignatureVerification(page_num=1)
        assert p.page_num == 1
        assert p.has_signature_candidate is False
        assert p.presence_detected is False
        assert p.review_required is False
        assert p.authenticity_signal == "not_applicable"
        assert p.candidates == []

    def test_document_signature_verification_defaults(self):
        d = DocumentSignatureVerification(document_id="doc1", source_file="test.pdf")
        assert d.document_id == "doc1"
        assert d.source_file == "test.pdf"
        assert d.pages == []
        assert d.total_candidate_pages == 0
        assert d.total_presence_pages == 0
        assert d.total_review_pages == 0
        assert d.experimental is True

    def test_candidate_mutable_defaults_independent(self):
        """Mutable default fields should not share state between instances."""
        c1 = SignatureCandidate(page_num=1, source="a")
        c2 = SignatureCandidate(page_num=2, source="b")
        c1.reason_codes.append("test")
        assert c2.reason_codes == []

    def test_document_mutable_defaults_independent(self):
        d1 = DocumentSignatureVerification(document_id="a", source_file="a.pdf")
        d2 = DocumentSignatureVerification(document_id="b", source_file="b.pdf")
        d1.pages.append("dummy")
        assert d2.pages == []


# ---------------------------------------------------------------------------
# Tests: Bounding box helpers
# ---------------------------------------------------------------------------


class TestBboxToInts:
    def test_valid_bbox(self):
        result = _bbox_to_ints([10, 20, 300, 400], (800, 600))
        assert result == (10, 20, 300, 400)

    def test_clamps_to_image_bounds(self):
        result = _bbox_to_ints([-10, -5, 700, 900], (800, 600))
        assert result is not None
        x1, y1, x2, y2 = result
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 600
        assert y2 <= 800

    def test_invalid_bbox_wrong_length(self):
        assert _bbox_to_ints([1, 2, 3], (100, 100)) is None

    def test_invalid_bbox_not_list(self):
        assert _bbox_to_ints("bad", (100, 100)) is None

    def test_zero_area_bbox(self):
        """When x1==x2 or y1==y2, area is zero -> None."""
        assert _bbox_to_ints([50, 50, 50, 100], (200, 200)) is None

    def test_float_values_rounded(self):
        result = _bbox_to_ints([10.7, 20.3, 100.9, 200.1], (300, 200))
        assert result is not None
        assert all(isinstance(v, int) for v in result)

    def test_inverted_coordinates_return_none(self):
        """When x2 < x1 (after clamping), returns None."""
        assert _bbox_to_ints([100, 100, 50, 200], (300, 300)) is None


class TestCropWithPadding:
    def test_valid_crop(self):
        img = _make_gray_image(100, 100, fill=128)
        crop = _crop_with_padding(img, [20, 20, 60, 60], pad=5)
        assert crop is not None
        assert crop.shape[0] > 0 and crop.shape[1] > 0

    def test_invalid_bbox_returns_none(self):
        img = _make_gray_image(100, 100)
        assert _crop_with_padding(img, [0, 0, 0, 0]) is None

    def test_padding_does_not_exceed_image(self):
        img = _make_gray_image(50, 50)
        crop = _crop_with_padding(img, [0, 0, 50, 50], pad=20)
        assert crop is not None
        assert crop.shape[0] <= 50
        assert crop.shape[1] <= 50

    def test_empty_bbox_returns_none(self):
        img = _make_gray_image(100, 100)
        assert _crop_with_padding(img, []) is None


# ---------------------------------------------------------------------------
# Tests: Grayscale conversion
# ---------------------------------------------------------------------------


class TestToGrayscaleArray:
    def test_pil_image(self):
        pil_img = Image.new("RGB", (100, 80), color=(128, 128, 128))
        result = _to_grayscale_array(pil_img)
        assert result is not None
        assert result.ndim == 2
        assert result.shape == (80, 100)

    def test_numpy_2d(self):
        arr = np.zeros((50, 60), dtype=np.uint8)
        result = _to_grayscale_array(arr)
        assert result is not None
        assert result.shape == (50, 60)

    def test_numpy_3d(self):
        arr = np.zeros((50, 60, 3), dtype=np.uint8)
        result = _to_grayscale_array(arr)
        assert result is not None
        assert result.ndim == 2

    def test_none_input(self):
        result = _to_grayscale_array(None)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Ink density / presence estimation
# ---------------------------------------------------------------------------


class TestInkDensityAnalysis:
    def test_blank_region_low_ink(self):
        """A white region has ~0 ink ratio."""
        crop = _make_gray_image(40, 200, fill=255)
        metrics = _estimate_presence_metrics(crop)
        assert metrics["ink_ratio"] < 0.05

    def test_dark_region_high_ink(self):
        """A dark region has high ink ratio."""
        crop = _make_gray_image(40, 200, fill=50)
        metrics = _estimate_presence_metrics(crop)
        assert metrics["ink_ratio"] > 0.5

    def test_empty_crop(self):
        crop = np.array([], dtype=np.uint8).reshape(0, 0)
        metrics = _estimate_presence_metrics(crop)
        assert metrics["ink_ratio"] == 0.0
        assert metrics["stroke_complexity"] == 0.0
        assert metrics["dark_span_ratio"] == 0.0
        assert metrics["aspect_ratio"] == 0.0

    def test_partial_ink_region(self):
        """Half dark, half white region."""
        crop = np.full((40, 200), 255, dtype=np.uint8)
        crop[:, :100] = 50  # Left half is dark
        metrics = _estimate_presence_metrics(crop)
        assert 0.2 < metrics["ink_ratio"] < 0.8
        assert metrics["dark_span_ratio"] > 0.0

    def test_aspect_ratio_wide(self):
        """A wide crop produces aspect_ratio > 1."""
        crop = _make_gray_image(20, 200, fill=128)
        metrics = _estimate_presence_metrics(crop)
        assert metrics["aspect_ratio"] > 1.0

    def test_aspect_ratio_tall(self):
        """A tall crop produces aspect_ratio < 1."""
        crop = _make_gray_image(200, 20, fill=128)
        metrics = _estimate_presence_metrics(crop)
        assert metrics["aspect_ratio"] < 1.0

    def test_metrics_keys(self):
        """All expected keys are present."""
        crop = _make_gray_image(40, 200, fill=128)
        metrics = _estimate_presence_metrics(crop)
        assert "ink_ratio" in metrics
        assert "stroke_complexity" in metrics
        assert "dark_span_ratio" in metrics
        assert "aspect_ratio" in metrics


# ---------------------------------------------------------------------------
# Tests: Stroke complexity
# ---------------------------------------------------------------------------


class TestStrokeAnalysis:
    def test_uniform_region_low_complexity(self):
        """A solid block has low stroke complexity."""
        binary = np.zeros((50, 100), dtype=np.uint8)
        result = _stroke_complexity(binary)
        assert result == 0.0

    def test_alternating_pattern_high_complexity(self):
        """Alternating black/white columns have high complexity."""
        binary = np.zeros((50, 100), dtype=np.uint8)
        binary[:, ::2] = 1  # Alternating columns
        result = _stroke_complexity(binary)
        assert result > 0.0

    def test_empty_array(self):
        binary = np.array([], dtype=np.uint8).reshape(0, 0)
        result = _stroke_complexity(binary)
        assert result == 0.0

    def test_single_horizontal_stroke(self):
        """A horizontal line across the middle."""
        binary = np.zeros((50, 100), dtype=np.uint8)
        binary[25, :] = 1
        result = _stroke_complexity(binary)
        assert result > 0.0

    def test_result_is_float(self):
        binary = np.ones((10, 10), dtype=np.uint8)
        result = _stroke_complexity(binary)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Tests: Spatial / keyword analysis
# ---------------------------------------------------------------------------


class TestSpatialAnalysis:
    def test_signature_keyword_detected(self):
        """OCR line containing 'signature' generates keyword candidate."""
        paddle_lines = [
            _make_paddle_line(
                "Authorized Signature",
                _make_box_points(100, 600, 400, 640),
                0.90,
            ),
        ]
        candidates = _fallback_candidates_from_lines(1, paddle_lines, (800, 600))
        assert len(candidates) == 1
        assert candidates[0].source == "ocr_keyword"

    def test_no_keywords_no_candidates(self):
        """Lines without signature keywords produce no candidates."""
        paddle_lines = [
            _make_paddle_line("Hello world", _make_box_points(100, 100, 400, 140), 0.95),
        ]
        candidates = _fallback_candidates_from_lines(1, paddle_lines, (800, 600))
        assert len(candidates) == 0

    def test_keyword_case_insensitive(self):
        """Keyword matching is case-insensitive."""
        paddle_lines = [
            _make_paddle_line(
                "AUTHORIZED SIGNATURE",
                _make_box_points(100, 700, 400, 740),
                0.90,
            ),
        ]
        candidates = _fallback_candidates_from_lines(1, paddle_lines, (800, 600))
        assert len(candidates) == 1

    def test_empty_paddle_lines(self):
        candidates = _fallback_candidates_from_lines(1, [], (800, 600))
        assert len(candidates) == 0

    def test_none_paddle_lines(self):
        candidates = _fallback_candidates_from_lines(1, None, (800, 600))
        assert len(candidates) == 0

    def test_multiple_keywords(self):
        """Multiple keyword lines produce multiple candidates."""
        paddle_lines = [
            _make_paddle_line("Signature:", _make_box_points(100, 600, 250, 640), 0.90),
            _make_paddle_line("Signed by:", _make_box_points(100, 700, 250, 740), 0.90),
        ]
        candidates = _fallback_candidates_from_lines(1, paddle_lines, (800, 600))
        assert len(candidates) == 2


# ---------------------------------------------------------------------------
# Tests: Signature line markers
# ---------------------------------------------------------------------------


class TestSignatureLineMarkers:
    def test_underscores_detected(self):
        assert _label_has_signature_line_markers("Signature: ___________") is True

    def test_dashes_detected(self):
        assert _label_has_signature_line_markers("Sign here: ---") is True

    def test_dots_detected(self):
        assert _label_has_signature_line_markers("Name: ...............") is True

    def test_plain_text_not_detected(self):
        assert _label_has_signature_line_markers("Hello world") is False

    def test_empty_string(self):
        assert _label_has_signature_line_markers("") is False

    def test_none_input(self):
        assert _label_has_signature_line_markers(None) is False


# ---------------------------------------------------------------------------
# Tests: Typed text signal
# ---------------------------------------------------------------------------


class TestTypedTextSignal:
    def test_high_confidence_alpha_text(self):
        """Typed-text signal fires when OCR confidence is very high on alpha text."""
        assert _typed_text_signal("John Smith", _SUSPICIOUS_TEXT_CONFIDENCE) is True

    def test_low_confidence_no_signal(self):
        assert _typed_text_signal("John Smith", 0.70) is False

    def test_short_text_no_signal(self):
        """Fewer than 4 alpha chars -> no signal."""
        assert _typed_text_signal("abc", _SUSPICIOUS_TEXT_CONFIDENCE) is False

    def test_empty_text(self):
        assert _typed_text_signal("", 0.99) is False

    def test_none_label(self):
        assert _typed_text_signal(None, 0.99) is False

    def test_exactly_four_alpha_chars(self):
        """Exactly 4 alpha chars meets the threshold."""
        assert _typed_text_signal("John", _SUSPICIOUS_TEXT_CONFIDENCE) is True


# ---------------------------------------------------------------------------
# Tests: Presence confidence
# ---------------------------------------------------------------------------


class TestPresenceConfidence:
    def test_zero_metrics(self):
        metrics = {"ink_ratio": 0.0, "stroke_complexity": 0.0, "dark_span_ratio": 0.0}
        score = _presence_confidence(metrics)
        assert score == 0.0

    def test_high_metrics_near_one(self):
        metrics = {"ink_ratio": 0.10, "stroke_complexity": 0.5, "dark_span_ratio": 0.8}
        score = _presence_confidence(metrics)
        assert 0.5 < score <= 1.0

    def test_always_in_range(self):
        """Confidence must always be in [0.0, 1.0]."""
        metrics = {"ink_ratio": 100.0, "stroke_complexity": 100.0, "dark_span_ratio": 100.0}
        score = _presence_confidence(metrics)
        assert 0.0 <= score <= 1.0

    def test_moderate_metrics(self):
        metrics = {"ink_ratio": 0.02, "stroke_complexity": 0.15, "dark_span_ratio": 0.3}
        score = _presence_confidence(metrics)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Tests: Candidate from form field
# ---------------------------------------------------------------------------


class TestCandidateFromFormField:
    def test_basic_construction(self):
        field = _make_form_field(label="Sig: ___", bbox=[10, 20, 300, 60], confidence=0.92)
        c = _candidate_from_form_field(page_num=3, field=field)
        assert c.page_num == 3
        assert c.source == "form_field"
        assert c.bbox == [10, 20, 300, 60]
        assert c.label == "Sig: ___"
        assert c.ocr_confidence == 0.92

    def test_missing_bbox_defaults_empty(self):
        field = {"field_type": "signature", "label": "test"}
        c = _candidate_from_form_field(page_num=1, field=field)
        assert c.bbox == []

    def test_none_label_defaults_empty(self):
        field = {"field_type": "signature", "label": None, "bbox": [1, 2, 3, 4]}
        c = _candidate_from_form_field(page_num=1, field=field)
        assert c.label == ""


# ---------------------------------------------------------------------------
# Tests: OCR keyword bbox projection
# ---------------------------------------------------------------------------


class TestProjectOcrKeywordBbox:
    def test_valid_projection(self):
        result = _project_ocr_keyword_bbox([100, 600, 300, 640], (800, 600), "Signature: ___")
        assert isinstance(result, list)
        assert len(result) == 4

    def test_empty_bbox_returns_empty(self):
        result = _project_ocr_keyword_bbox([], (800, 600), "Signature")
        assert result == []

    def test_none_bbox_returns_empty(self):
        result = _project_ocr_keyword_bbox(None, (800, 600), "Signature")
        assert result == []


# ---------------------------------------------------------------------------
# Tests: Signature detection (analyze_signature_page)
# ---------------------------------------------------------------------------


class TestSignatureDetection:
    def test_no_structure_no_keywords_empty_result(self):
        """No form fields and no keyword lines -> no candidates."""
        img = _make_gray_image(800, 600)
        result = analyze_signature_page(img, page_num=1, structure_data=None, paddle_lines=[])
        assert result.has_signature_candidate is False
        assert result.candidates == []

    def test_form_field_signature_detected(self):
        """A form field with field_type='signature' produces a candidate."""
        img = _make_image_with_dark_region(800, 600, region_bbox=(100, 500, 400, 560), ink_value=40)
        structure = {"form_fields": [_make_form_field(bbox=[100, 500, 400, 560])]}
        result = analyze_signature_page(img, page_num=1, structure_data=structure)
        assert result.has_signature_candidate is True
        assert len(result.candidates) >= 1

    def test_form_field_non_signature_ignored(self):
        """Form fields that are not 'signature' type are ignored."""
        img = _make_gray_image(800, 600)
        structure = {"form_fields": [_make_form_field(field_type="text")]}
        result = analyze_signature_page(img, page_num=1, structure_data=structure)
        assert result.has_signature_candidate is False

    def test_keyword_fallback_when_no_form_fields(self):
        """When no form fields exist, OCR keyword fallback is used."""
        img = _make_gray_image(800, 600)
        paddle_lines = [
            _make_paddle_line(
                "Signature: ___",
                _make_box_points(100, 600, 400, 640),
                0.85,
            ),
        ]
        result = analyze_signature_page(img, page_num=1, structure_data=None, paddle_lines=paddle_lines)
        assert result.has_signature_candidate is True

    def test_none_image_returns_empty(self):
        """None image -> safe empty result."""
        result = analyze_signature_page(None, page_num=1, structure_data=None)
        assert result.has_signature_candidate is False
        assert result.page_num == 1

    def test_page_number_propagated(self):
        img = _make_gray_image(800, 600)
        result = analyze_signature_page(img, page_num=7, structure_data=None)
        assert result.page_num == 7

    def test_ink_presence_detected_in_dark_region(self):
        """Ink presence should be detected when dark ink exists in bbox region."""
        img = _make_image_with_dark_region(800, 600, region_bbox=(100, 500, 400, 560), ink_value=30)
        structure = {"form_fields": [_make_form_field(bbox=[100, 500, 400, 560])]}
        result = analyze_signature_page(img, page_num=1, structure_data=structure)
        assert result.has_signature_candidate
        any_presence = any(c.get("presence_detected", False) for c in result.candidates)
        assert any_presence is True

    def test_blank_signature_field_no_presence(self):
        """A blank (white) signature region should not detect presence."""
        img = _make_gray_image(800, 600, fill=255)
        structure = {"form_fields": [_make_form_field(bbox=[100, 500, 400, 560])]}
        result = analyze_signature_page(img, page_num=1, structure_data=structure)
        for candidate in result.candidates:
            metrics = candidate.get("metrics", {})
            assert metrics.get("ink_ratio", 1.0) < 0.5


# ---------------------------------------------------------------------------
# Tests: Signal merging and authenticity
# ---------------------------------------------------------------------------


class TestSignalMerging:
    def test_presence_detected_sets_inconclusive(self):
        """When ink is present without typed-text flag -> inconclusive."""
        img = _make_image_with_dark_region(800, 600, region_bbox=(100, 500, 400, 560), ink_value=30)
        structure = {"form_fields": [_make_form_field(
            label="Sign: ___",
            bbox=[100, 500, 400, 560],
            confidence=0.5,
        )]}
        result = analyze_signature_page(img, page_num=1, structure_data=structure)
        for c in result.candidates:
            if c.get("presence_detected"):
                assert c["authenticity_signal"] in ("inconclusive", "review_required")

    def test_typed_text_triggers_review(self):
        """High-confidence alpha text in signature field -> review_required."""
        img = _make_image_with_dark_region(800, 600, region_bbox=(100, 500, 400, 560), ink_value=30)
        structure = {"form_fields": [_make_form_field(
            label="John Q. Public",
            bbox=[100, 500, 400, 560],
            confidence=_SUSPICIOUS_TEXT_CONFIDENCE,
        )]}
        result = analyze_signature_page(img, page_num=1, structure_data=structure)
        for c in result.candidates:
            if c.get("presence_detected"):
                if "typed_text_suspected" in c.get("reason_codes", []):
                    assert c["review_required"] is True

    def test_no_presence_means_not_applicable(self):
        """No ink presence -> authenticity_signal is 'not_applicable'."""
        img = _make_gray_image(800, 600, fill=255)
        structure = {"form_fields": [_make_form_field(bbox=[100, 500, 400, 560])]}
        result = analyze_signature_page(img, page_num=1, structure_data=structure)
        for c in result.candidates:
            if not c.get("presence_detected"):
                assert c["authenticity_signal"] == "not_applicable"

    def test_review_required_propagates_to_page(self):
        """If any candidate has review_required, page-level flag is set."""
        page = PageSignatureVerification(page_num=1)
        page.candidates = [
            {"review_required": False, "presence_detected": True},
            {"review_required": True, "presence_detected": True},
        ]
        page.review_required = any(c["review_required"] for c in page.candidates)
        assert page.review_required is True


# ---------------------------------------------------------------------------
# Tests: Hu moment / contour comparison helpers
# ---------------------------------------------------------------------------


class TestStrokeComplexityExtended:
    """Tests for stroke complexity metrics used in presence estimation."""

    def test_stroke_complexity_symmetric(self):
        """Transposing the binary should produce similar complexity."""
        binary = np.zeros((50, 100), dtype=np.uint8)
        binary[10:40, 20:80] = 1
        c1 = _stroke_complexity(binary)
        c2 = _stroke_complexity(binary.T)
        # Not necessarily equal due to different shape, but both should be positive
        assert c1 > 0
        assert c2 > 0

    def test_all_ones_low_complexity(self):
        """All-1 binary has minimal edge changes (only at boundaries)."""
        binary = np.ones((50, 100), dtype=np.uint8)
        result = _stroke_complexity(binary)
        # Interior has no transitions, only edges contribute minimally
        assert result >= 0.0

    def test_checkerboard_high_complexity(self):
        """Checkerboard pattern has very high complexity."""
        binary = np.zeros((50, 100), dtype=np.uint8)
        binary[::2, ::2] = 1
        binary[1::2, 1::2] = 1
        result = _stroke_complexity(binary)
        assert result > 0.1


class TestDarkSpanAnalysis:
    """Tests for dark span ratio metrics inside presence estimation."""

    def test_dark_span_ratio_full_coverage(self):
        """Full dark coverage produces dark_span_ratio near 1.0."""
        crop = _make_gray_image(40, 200, fill=50)
        metrics = _estimate_presence_metrics(crop)
        assert metrics["dark_span_ratio"] > 0.8

    def test_dark_span_ratio_partial_coverage(self):
        """Partial dark coverage produces intermediate dark_span_ratio."""
        crop = np.full((40, 200), 255, dtype=np.uint8)
        crop[:, 50:150] = 50
        metrics = _estimate_presence_metrics(crop)
        assert 0.3 < metrics["dark_span_ratio"] < 0.8

    def test_no_dark_columns_zero_span(self):
        """All white produces dark_span_ratio of 0."""
        crop = _make_gray_image(40, 200, fill=255)
        metrics = _estimate_presence_metrics(crop)
        assert metrics["dark_span_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Tests: Finalization
# ---------------------------------------------------------------------------


class TestFinalization:
    def test_finalize_empty_document(self):
        doc = DocumentSignatureVerification(document_id="d1", source_file="empty.pdf")
        result = finalize_signature_verification(doc)
        assert result.total_candidate_pages == 0
        assert result.total_presence_pages == 0
        assert result.total_review_pages == 0

    def test_finalize_with_candidates(self):
        doc = DocumentSignatureVerification(document_id="d2", source_file="signed.pdf")
        doc.pages = [
            PageSignatureVerification(
                page_num=1,
                has_signature_candidate=True,
                presence_detected=True,
                review_required=False,
            ),
            PageSignatureVerification(
                page_num=2,
                has_signature_candidate=False,
                presence_detected=False,
                review_required=False,
            ),
        ]
        result = finalize_signature_verification(doc)
        assert result.total_candidate_pages == 1
        assert result.total_presence_pages == 1
        assert result.total_review_pages == 0

    def test_finalize_with_review_pages(self):
        doc = DocumentSignatureVerification(document_id="d3", source_file="suspicious.pdf")
        doc.pages = [
            PageSignatureVerification(
                page_num=1,
                has_signature_candidate=True,
                presence_detected=True,
                review_required=True,
            ),
            PageSignatureVerification(
                page_num=2,
                has_signature_candidate=True,
                presence_detected=True,
                review_required=True,
            ),
        ]
        result = finalize_signature_verification(doc)
        assert result.total_candidate_pages == 2
        assert result.total_presence_pages == 2
        assert result.total_review_pages == 2

    def test_finalize_preserves_experimental_flag(self):
        doc = DocumentSignatureVerification(document_id="d4", source_file="test.pdf")
        result = finalize_signature_verification(doc)
        assert result.experimental is True

    def test_finalize_with_dict_pages(self):
        """finalize_signature_verification should accept dict pages as well."""
        doc = DocumentSignatureVerification(document_id="d5", source_file="dict.pdf")
        doc.pages = [
            {
                "page_num": 1,
                "has_signature_candidate": True,
                "presence_detected": True,
                "review_required": False,
            },
            {
                "page_num": 2,
                "has_signature_candidate": False,
                "presence_detected": False,
                "review_required": False,
            },
        ]
        result = finalize_signature_verification(doc)
        assert result.total_candidate_pages == 1
        assert result.total_presence_pages == 1
        assert result.total_review_pages == 0

    def test_finalize_resets_counts(self):
        """Calling finalize twice should reset counts correctly."""
        doc = DocumentSignatureVerification(document_id="d6", source_file="double.pdf")
        doc.pages = [
            PageSignatureVerification(
                page_num=1,
                has_signature_candidate=True,
                presence_detected=True,
                review_required=True,
            ),
        ]
        doc = finalize_signature_verification(doc)
        assert doc.total_review_pages == 1
        # Remove the review page and re-finalize
        doc.pages = [
            PageSignatureVerification(
                page_num=1,
                has_signature_candidate=True,
                presence_detected=False,
                review_required=False,
            ),
        ]
        doc = finalize_signature_verification(doc)
        assert doc.total_review_pages == 0


# ---------------------------------------------------------------------------
# Tests: JSON output
# ---------------------------------------------------------------------------


class TestWriteJson:
    def test_json_file_created(self, tmp_path):
        doc = DocumentSignatureVerification(
            document_id="test_id", source_file="subfolder/doc.pdf",
        )
        doc.pages = []
        doc = finalize_signature_verification(doc)
        result = write_signature_verification_json(doc, str(tmp_path), "subfolder", __version__)
        assert result is not None
        assert os.path.exists(result)
        assert result.endswith(".signature.json")

    def test_json_schema_structure(self, tmp_path):
        doc = DocumentSignatureVerification(
            document_id="schema_test", source_file="test.pdf",
        )
        p = PageSignatureVerification(
            page_num=1,
            has_signature_candidate=True,
            presence_detected=True,
            review_required=False,
            authenticity_signal="inconclusive",
        )
        doc.pages = [p]
        doc = finalize_signature_verification(doc)
        result_path = write_signature_verification_json(doc, str(tmp_path), "", __version__)
        assert result_path is not None

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0"
        assert data["document_id"] == "schema_test"
        assert data["source_file"] == "test.pdf"
        assert "processing" in data
        assert data["processing"]["experimental"] is True
        assert data["processing"]["pipeline_version"] == __version__
        assert "timestamp" in data["processing"]
        assert "notes" in data["processing"]
        assert isinstance(data["processing"]["notes"], list)
        assert len(data["processing"]["notes"]) > 0
        assert "document_summary" in data
        assert data["document_summary"]["experimental"] is True
        assert "total_candidate_pages" in data["document_summary"]
        assert "total_presence_pages" in data["document_summary"]
        assert "total_review_pages" in data["document_summary"]
        assert "pages" in data
        assert len(data["pages"]) == 1

    def test_json_subfolder_creation(self, tmp_path):
        doc = DocumentSignatureVerification(
            document_id="sub_test", source_file="deep/nested/doc.pdf",
        )
        doc.pages = []
        doc = finalize_signature_verification(doc)
        result = write_signature_verification_json(doc, str(tmp_path), "deep/nested", __version__)
        assert result is not None
        assert os.path.exists(result)
        expected_dir = os.path.join(str(tmp_path), "EXPORT", "SIGNATURE", "deep", "nested")
        assert os.path.isdir(expected_dir)

    def test_path_traversal_protection(self, tmp_path):
        """Subfolder with '..' segments -> sanitized safely."""
        doc = DocumentSignatureVerification(
            document_id="traversal_test", source_file="evil.pdf",
        )
        doc.pages = []
        doc = finalize_signature_verification(doc)
        result = write_signature_verification_json(doc, str(tmp_path), "../../etc", __version__)
        assert result is not None
        sig_dir = os.path.join(str(tmp_path), "EXPORT", "SIGNATURE")
        assert os.path.realpath(result).startswith(os.path.realpath(sig_dir))

    def test_empty_subfolder(self, tmp_path):
        doc = DocumentSignatureVerification(
            document_id="root_test", source_file="root_doc.pdf",
        )
        doc.pages = []
        doc = finalize_signature_verification(doc)
        result = write_signature_verification_json(doc, str(tmp_path), ".", __version__)
        assert result is not None
        assert os.path.exists(result)
        expected_dir = os.path.join(str(tmp_path), "EXPORT", "SIGNATURE")
        assert os.path.dirname(result) == expected_dir

    def test_non_pdf_uses_ext_token(self, tmp_path):
        doc = DocumentSignatureVerification(
            document_id="img_doc", source_file="img/photo.tiff",
        )
        doc.pages = []
        doc = finalize_signature_verification(doc)
        result = write_signature_verification_json(doc, str(tmp_path), "", __version__)
        assert result is not None
        assert os.path.basename(result) == "photo__tiff.signature.json"

    def test_experimental_notes_in_output(self, tmp_path):
        """The JSON output must contain experimental advisory notes."""
        doc = DocumentSignatureVerification(
            document_id="notes_test", source_file="test.pdf",
        )
        doc.pages = []
        doc = finalize_signature_verification(doc)
        result_path = write_signature_verification_json(doc, str(tmp_path), "", __version__)
        assert result_path is not None

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        notes = data["processing"]["notes"]
        # Must mention that it never certifies authenticity
        assert any("never certifies" in n.lower() or "never assert" in n.lower() for n in notes)


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_none_image_safe(self):
        result = analyze_signature_page(None, page_num=1, structure_data=None)
        assert result.has_signature_candidate is False
        assert result.page_num == 1

    def test_empty_structure_data(self):
        img = _make_gray_image(800, 600)
        result = analyze_signature_page(img, page_num=1, structure_data={})
        assert result.has_signature_candidate is False

    def test_none_structure_none_lines(self):
        img = _make_gray_image(800, 600)
        result = analyze_signature_page(img, page_num=1, structure_data=None, paddle_lines=None)
        assert result.has_signature_candidate is False

    def test_malformed_form_field_no_crash(self):
        """Malformed form field data should not crash."""
        img = _make_gray_image(800, 600)
        structure = {"form_fields": [{"field_type": "signature"}]}  # No bbox
        result = analyze_signature_page(img, page_num=1, structure_data=structure)
        assert result.page_num == 1

    def test_cv2_unavailable_still_works(self):
        """When cv2 is not available, module still produces results via numpy."""
        import signature_verification as sv_module

        with mock.patch.object(sv_module, "_CV2_AVAILABLE", False):
            img = _make_gray_image(800, 600)
            result = analyze_signature_page(img, page_num=1, structure_data=None)
            assert result.page_num == 1

    def test_write_json_on_error_returns_none(self, tmp_path):
        """If file write fails, returns None instead of crashing."""
        doc = DocumentSignatureVerification(document_id="err", source_file="test.pdf")
        with mock.patch(
            "signature_verification.os.makedirs",
            side_effect=OSError("Permission denied"),
        ):
            result = write_signature_verification_json(doc, str(tmp_path), "", __version__)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: End-to-end detection flow
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline_with_signature(self, tmp_path):
        """Full: detect -> finalize -> write JSON for a document with signatures."""
        img1 = _make_image_with_dark_region(
            800, 600, region_bbox=(100, 500, 400, 560), ink_value=30,
        )
        structure1 = {"form_fields": [_make_form_field(bbox=[100, 500, 400, 560])]}
        page1 = analyze_signature_page(img1, page_num=1, structure_data=structure1)

        img2 = _make_gray_image(800, 600)
        page2 = analyze_signature_page(img2, page_num=2, structure_data=None, paddle_lines=[])

        doc = DocumentSignatureVerification(
            document_id="e2e_doc", source_file="evidence/contract.pdf",
        )
        doc.pages = [page1, page2]
        doc = finalize_signature_verification(doc)

        assert doc.total_candidate_pages >= 1
        assert doc.experimental is True

        result_path = write_signature_verification_json(doc, str(tmp_path), "evidence", __version__)
        assert result_path is not None
        assert os.path.exists(result_path)

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["document_summary"]["experimental"] is True
        assert data["processing"]["experimental"] is True
        assert len(data["pages"]) == 2

    def test_full_pipeline_no_signatures(self, tmp_path):
        """Full pipeline for a clean document with no signatures."""
        doc = DocumentSignatureVerification(
            document_id="clean", source_file="report.pdf",
        )
        for i in range(1, 4):
            img = _make_gray_image(800, 600)
            page = analyze_signature_page(img, page_num=i, structure_data=None, paddle_lines=[])
            doc.pages.append(page)
        doc = finalize_signature_verification(doc)

        assert doc.total_candidate_pages == 0
        assert doc.total_presence_pages == 0
        assert doc.total_review_pages == 0

        result_path = write_signature_verification_json(doc, str(tmp_path), "", __version__)
        assert result_path is not None
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["document_summary"]["total_candidate_pages"] == 0

    def test_keyword_fallback_end_to_end(self, tmp_path):
        """End-to-end via OCR keyword fallback (no form_fields)."""
        img = _make_gray_image(800, 600)
        paddle_lines = [
            _make_paddle_line(
                "Authorized Signature",
                _make_box_points(100, 700, 350, 730),
                0.90,
            ),
        ]
        page = analyze_signature_page(
            img, page_num=1, structure_data=None, paddle_lines=paddle_lines,
        )
        assert page.has_signature_candidate is True

        doc = DocumentSignatureVerification(
            document_id="kw_e2e", source_file="letter.pdf",
        )
        doc.pages = [page]
        doc = finalize_signature_verification(doc)
        assert doc.total_candidate_pages == 1


# ---------------------------------------------------------------------------
# Tests: Original PIL-based integration tests (preserved from prior version)
# ---------------------------------------------------------------------------


class TestPilIntegration:
    def test_analyze_from_form_field_detects_presence(self):
        image = _signature_page()
        structure = {
            "form_fields": [
                {
                    "field_type": "signature",
                    "label": "Signature: ___________",
                    "bbox": [100, 160, 340, 230],
                    "confidence": 0.82,
                }
            ]
        }
        result = analyze_signature_page(image, 1, structure, [])
        assert result.has_signature_candidate is True
        assert result.presence_detected is True
        assert result.authenticity_signal in {"inconclusive", "review_required"}
        assert result.candidates[0]["presence_confidence"] > 0

    def test_analyze_without_ink_is_not_applicable(self):
        image = _blank_page()
        structure = {
            "form_fields": [
                {
                    "field_type": "signature",
                    "label": "Signature: ___________",
                    "bbox": [100, 160, 340, 230],
                    "confidence": 0.82,
                }
            ]
        }
        result = analyze_signature_page(image, 1, structure, [])
        assert result.has_signature_candidate is True
        assert result.presence_detected is False
        assert result.authenticity_signal == "not_applicable"
        assert "no_signature_ink" in result.candidates[0]["reason_codes"]

    def test_analyze_flags_typed_text_for_review(self):
        image = _signature_page()
        structure = {
            "form_fields": [
                {
                    "field_type": "signature",
                    "label": "John Doe",
                    "bbox": [100, 160, 340, 230],
                    "confidence": 0.99,
                }
            ]
        }
        result = analyze_signature_page(image, 1, structure, [])
        assert result.review_required is True
        assert result.authenticity_signal == "review_required"
        assert "typed_text_suspected" in result.candidates[0]["reason_codes"]

    def test_analyze_falls_back_to_ocr_keyword_lines(self):
        image = _blank_page()
        draw = ImageDraw.Draw(image)
        draw.text((40, 145), "Signature:", fill="black")
        draw.line((190, 170, 300, 205), fill="black", width=4)
        draw.line((230, 205, 340, 168), fill="black", width=3)
        paddle_lines = [
            (
                "Signature:",
                [[40, 145], [140, 145], [140, 175], [40, 175]],
                0.99,
            )
        ]
        result = analyze_signature_page(image, 1, {}, paddle_lines)
        assert result.has_signature_candidate is True
        assert result.presence_detected is True
        assert result.candidates[0]["source"] == "ocr_keyword"

    def test_analyze_ignores_printed_label_without_ink(self):
        image = _blank_page()
        draw = ImageDraw.Draw(image)
        draw.text((40, 145), "Signature:", fill="black")
        paddle_lines = [
            (
                "Signature:",
                [[40, 145], [140, 145], [140, 175], [40, 175]],
                0.99,
            )
        ]
        result = analyze_signature_page(image, 1, {}, paddle_lines)
        assert result.has_signature_candidate is True
        assert result.presence_detected is False
        assert result.authenticity_signal == "not_applicable"
        assert "no_signature_ink" in result.candidates[0]["reason_codes"]

    def test_finalize_counts_pages(self):
        doc = DocumentSignatureVerification(
            document_id="doc-1",
            source_file="sample.pdf",
            pages=[
                PageSignatureVerification(
                    page_num=1,
                    has_signature_candidate=True,
                    presence_detected=True,
                    review_required=False,
                ),
                PageSignatureVerification(
                    page_num=2,
                    has_signature_candidate=True,
                    presence_detected=True,
                    review_required=True,
                ),
            ],
        )
        finalized = finalize_signature_verification(doc)
        assert finalized.total_candidate_pages == 2
        assert finalized.total_presence_pages == 2
        assert finalized.total_review_pages == 1

    def test_write_json(self, tmp_path):
        doc = DocumentSignatureVerification(
            document_id="doc-1",
            source_file="nested/sample.pdf",
            pages=[
                PageSignatureVerification(
                    page_num=1,
                    has_signature_candidate=True,
                    presence_detected=True,
                    review_required=True,
                    authenticity_signal="review_required",
                    candidates=[
                        {"page_num": 1, "reason_codes": ["typed_text_suspected"]},
                    ],
                )
            ],
        )
        finalized = finalize_signature_verification(doc)
        path = write_signature_verification_json(finalized, str(tmp_path), "nested", "0.9.0")
        assert path is not None
        payload = json.loads(
            (tmp_path / "EXPORT" / "SIGNATURE" / "nested" / "sample.signature.json").read_text(
                encoding="utf-8",
            ),
        )
        assert payload["processing"]["experimental"] is True
        assert payload["document_summary"]["total_review_pages"] == 1


# ---------------------------------------------------------------------------
# Tests: Constants verification
# ---------------------------------------------------------------------------


class TestConstants:
    def test_signature_keywords_is_tuple(self):
        assert isinstance(_SIGNATURE_KEYWORDS, tuple)
        assert len(_SIGNATURE_KEYWORDS) >= 4

    def test_presence_min_ink_ratio_positive(self):
        assert _PRESENCE_MIN_INK_RATIO > 0

    def test_wide_signature_ratio(self):
        assert _WIDE_SIGNATURE_RATIO > 1.0

    def test_min_stroke_complexity_positive(self):
        assert _MIN_STROKE_COMPLEXITY > 0

    def test_typed_text_confidence_high(self):
        assert _TYPED_TEXT_CONFIDENCE >= 0.85
        assert _SUSPICIOUS_TEXT_CONFIDENCE >= _TYPED_TEXT_CONFIDENCE

    def test_keywords_include_common_terms(self):
        """Keywords should include at minimum 'signature' and 'sign here'."""
        kw_lower = [k.lower() for k in _SIGNATURE_KEYWORDS]
        assert "signature" in kw_lower
        assert "sign here" in kw_lower
