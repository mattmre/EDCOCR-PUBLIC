"""
Unit tests for processing integrity validation (validation.py).

Tests cover:
- PageValidation and DocumentValidation dataclass defaults
- Quality classification logic
- File hash computation
- Finalization summary statistics
- Validation JSON writing
- Integration with extract_paddle_lines confidence return

Run with: python -m pytest tests/test_processing_validation.py -v
"""

import hashlib
import json
import os
import tempfile

import pytest

# Add project root to path
from validation import (
    DocumentValidation,
    PageValidation,
    classify_quality,
    compute_file_hash,
    finalize_validation,
    write_validation_json,
)
from version import __version__

# ---------------------------------------------------------------------------
# Tests: PageValidation defaults
# ---------------------------------------------------------------------------

class TestPageValidation:
    def test_page_validation_defaults(self):
        pv = PageValidation(page_num=1)
        assert pv.page_num == 1
        assert pv.ocr_method == ""
        assert pv.ocr_language == ""
        assert pv.ocr_confidence == 0.0
        assert pv.text_length == 0
        assert pv.has_text is False
        assert pv.status == "pending"

    def test_page_validation_with_values(self):
        pv = PageValidation(
            page_num=3,
            ocr_method="PaddleOCR",
            ocr_confidence=0.92,
            text_length=500,
            has_text=True,
            status="ok",
        )
        assert pv.page_num == 3
        assert pv.ocr_method == "PaddleOCR"
        assert pv.ocr_confidence == 0.92
        assert pv.text_length == 500
        assert pv.has_text is True
        assert pv.status == "ok"


# ---------------------------------------------------------------------------
# Tests: DocumentValidation defaults
# ---------------------------------------------------------------------------

class TestDocumentValidation:
    def test_document_validation_defaults(self):
        dv = DocumentValidation(document_id="abc123", source_file="test.pdf")
        assert dv.document_id == "abc123"
        assert dv.source_file == "test.pdf"
        assert dv.source_page_count == 0
        assert dv.output_page_count == 0
        assert dv.page_count_match is True
        assert dv.pages == []
        assert dv.classification == ""
        assert dv.overall_confidence == 0.0
        assert dv.pages_with_text == 0
        assert dv.pages_image_only == 0
        assert dv.pages_failed == 0
        assert dv.text_extraction_rate == 0.0
        assert dv.total_text_length == 0
        assert dv.ocr_methods_used == []
        assert dv.output_hash == ""


# ---------------------------------------------------------------------------
# Tests: classify_quality
# ---------------------------------------------------------------------------

class TestClassifyQuality:
    def test_classify_quality_high(self):
        """All pages have text, high confidence -> high_quality."""
        dv = DocumentValidation(document_id="d1", source_file="f.pdf")
        dv.text_extraction_rate = 1.0
        dv.overall_confidence = 0.92
        assert classify_quality(dv) == "high_quality"

    def test_classify_quality_acceptable(self):
        """Decent extraction rate and confidence -> acceptable."""
        dv = DocumentValidation(document_id="d2", source_file="f.pdf")
        dv.text_extraction_rate = 0.85
        dv.overall_confidence = 0.65
        assert classify_quality(dv) == "acceptable"

    def test_classify_quality_degraded_by_rate(self):
        """Moderate extraction rate -> degraded."""
        dv = DocumentValidation(document_id="d3", source_file="f.pdf")
        dv.text_extraction_rate = 0.55
        dv.overall_confidence = 0.30
        assert classify_quality(dv) == "degraded"

    def test_classify_quality_degraded_by_confidence(self):
        """Low rate but moderate confidence -> degraded."""
        dv = DocumentValidation(document_id="d4", source_file="f.pdf")
        dv.text_extraction_rate = 0.10
        dv.overall_confidence = 0.45
        assert classify_quality(dv) == "degraded"

    def test_classify_quality_review_required(self):
        """Very low metrics -> review_required."""
        dv = DocumentValidation(document_id="d5", source_file="f.pdf")
        dv.text_extraction_rate = 0.10
        dv.overall_confidence = 0.10
        assert classify_quality(dv) == "review_required"

    def test_classify_quality_zero_metrics(self):
        """All zeros -> review_required."""
        dv = DocumentValidation(document_id="d6", source_file="f.pdf")
        dv.text_extraction_rate = 0.0
        dv.overall_confidence = 0.0
        assert classify_quality(dv) == "review_required"

    def test_classify_quality_boundary_high(self):
        """Exact boundary: rate=1.0, conf=0.85 -> high_quality."""
        dv = DocumentValidation(document_id="d7", source_file="f.pdf")
        dv.text_extraction_rate = 1.0
        dv.overall_confidence = 0.85
        assert classify_quality(dv) == "high_quality"

    def test_classify_quality_boundary_acceptable(self):
        """Exact boundary: rate=0.80, conf=0.60 -> acceptable."""
        dv = DocumentValidation(document_id="d8", source_file="f.pdf")
        dv.text_extraction_rate = 0.80
        dv.overall_confidence = 0.60
        assert classify_quality(dv) == "acceptable"


# ---------------------------------------------------------------------------
# Tests: compute_file_hash
# ---------------------------------------------------------------------------

class TestComputeFileHash:
    def test_compute_file_hash_returns_sha256(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"test content for hashing")
            f.flush()
            path = f.name
        try:
            result = compute_file_hash(path)
            assert len(result) == 64
            assert all(c in "0123456789abcdef" for c in result)
        finally:
            os.unlink(path)

    def test_compute_file_hash_deterministic(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"deterministic content")
            f.flush()
            path = f.name
        try:
            h1 = compute_file_hash(path)
            h2 = compute_file_hash(path)
            assert h1 == h2
        finally:
            os.unlink(path)

    def test_compute_file_hash_matches_hashlib(self):
        content = b"verify against hashlib directly"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            result = compute_file_hash(path)
            expected = hashlib.sha256(content).hexdigest()
            assert result == expected
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests: finalize_validation
# ---------------------------------------------------------------------------

class TestFinalizeValidation:
    def test_finalize_validation_computes_summary(self):
        dv = DocumentValidation(
            document_id="d1",
            source_file="doc.pdf",
            source_page_count=3,
            output_page_count=3,
        )
        dv.pages = [
            {"page_num": 1, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.90, "text_length": 200, "has_text": True, "status": "ok"},
            {"page_num": 2, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.85, "text_length": 150, "has_text": True, "status": "ok"},
            {"page_num": 3, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.88, "text_length": 180, "has_text": True, "status": "ok"},
        ]
        result = finalize_validation(dv)
        assert result.pages_with_text == 3
        assert result.pages_image_only == 0
        assert result.pages_failed == 0
        assert result.total_text_length == 530
        assert abs(result.overall_confidence - 0.8767) < 0.01
        assert abs(result.text_extraction_rate - 1.0) < 0.01
        assert result.classification == "high_quality"
        assert result.page_count_match is True

    def test_finalize_validation_counts_methods(self):
        dv = DocumentValidation(
            document_id="d2",
            source_file="mixed.pdf",
            source_page_count=3,
            output_page_count=3,
        )
        dv.pages = [
            {"page_num": 1, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.90, "text_length": 200, "has_text": True, "status": "ok"},
            {"page_num": 2, "ocr_method": "Tesseract", "ocr_language": "",
             "ocr_confidence": 0.70, "text_length": 100, "has_text": True, "status": "fallback"},
            {"page_num": 3, "ocr_method": "ImageOnly", "ocr_language": "",
             "ocr_confidence": 0.0, "text_length": 0, "has_text": False, "status": "image_only"},
        ]
        result = finalize_validation(dv)
        assert "PaddleOCR" in result.ocr_methods_used
        assert "Tesseract" in result.ocr_methods_used
        assert "ImageOnly" in result.ocr_methods_used
        assert result.pages_with_text == 2
        assert result.pages_image_only == 1

    def test_finalize_validation_empty_pages(self):
        dv = DocumentValidation(document_id="d3", source_file="empty.pdf")
        dv.pages = []
        result = finalize_validation(dv)
        assert result.classification == "review_required"
        assert result.overall_confidence == 0.0
        assert result.text_extraction_rate == 0.0

    def test_finalize_validation_page_count_mismatch(self):
        dv = DocumentValidation(
            document_id="d4",
            source_file="mismatch.pdf",
            source_page_count=5,
            output_page_count=4,
        )
        dv.pages = [
            {"page_num": i, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.90, "text_length": 100, "has_text": True, "status": "ok"}
            for i in range(1, 5)
        ]
        result = finalize_validation(dv)
        assert result.page_count_match is False

    def test_finalize_validation_page_count_match(self):
        dv = DocumentValidation(
            document_id="d5",
            source_file="match.pdf",
            source_page_count=2,
            output_page_count=2,
        )
        dv.pages = [
            {"page_num": 1, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.90, "text_length": 100, "has_text": True, "status": "ok"},
            {"page_num": 2, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.85, "text_length": 80, "has_text": True, "status": "ok"},
        ]
        result = finalize_validation(dv)
        assert result.page_count_match is True

    def test_text_extraction_rate_calculation(self):
        dv = DocumentValidation(
            document_id="d6",
            source_file="partial.pdf",
            source_page_count=4,
            output_page_count=4,
        )
        dv.pages = [
            {"page_num": 1, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.90, "text_length": 100, "has_text": True, "status": "ok"},
            {"page_num": 2, "ocr_method": "ImageOnly", "ocr_language": "",
             "ocr_confidence": 0.0, "text_length": 0, "has_text": False, "status": "image_only"},
            {"page_num": 3, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.80, "text_length": 50, "has_text": True, "status": "ok"},
            {"page_num": 4, "ocr_method": "ImageOnly", "ocr_language": "",
             "ocr_confidence": 0.0, "text_length": 0, "has_text": False, "status": "image_only"},
        ]
        result = finalize_validation(dv)
        assert abs(result.text_extraction_rate - 0.50) < 0.01

    def test_validation_with_mixed_ocr_methods(self):
        dv = DocumentValidation(
            document_id="d7",
            source_file="mixed_all.pdf",
            source_page_count=4,
            output_page_count=4,
        )
        dv.pages = [
            {"page_num": 1, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.95, "text_length": 300, "has_text": True, "status": "ok"},
            {"page_num": 2, "ocr_method": "Tesseract", "ocr_language": "",
             "ocr_confidence": 0.60, "text_length": 100, "has_text": True, "status": "fallback"},
            {"page_num": 3, "ocr_method": "ImageOnly", "ocr_language": "",
             "ocr_confidence": 0.0, "text_length": 0, "has_text": False, "status": "image_only"},
            {"page_num": 4, "ocr_method": "EXTRACT_FAILED", "ocr_language": "",
             "ocr_confidence": 0.0, "text_length": 0, "has_text": False, "status": "failed"},
        ]
        result = finalize_validation(dv)
        assert result.pages_with_text == 2
        assert result.pages_image_only == 1
        assert result.pages_failed == 1
        assert len(result.ocr_methods_used) == 4
        # Only pages with conf > 0 count toward overall_confidence
        assert abs(result.overall_confidence - 0.775) < 0.01


# ---------------------------------------------------------------------------
# Tests: write_validation_json
# ---------------------------------------------------------------------------

class TestWriteValidationJson:
    def test_write_validation_json_creates_file(self, tmp_path):
        dv = DocumentValidation(
            document_id="test_id",
            source_file="subfolder/document.pdf",
            source_page_count=1,
            output_page_count=1,
        )
        dv.pages = [
            {"page_num": 1, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.90, "text_length": 100, "has_text": True, "status": "ok"},
        ]
        dv = finalize_validation(dv)
        result = write_validation_json(dv, str(tmp_path), "subfolder", "0.4.0")
        assert result is not None
        assert os.path.exists(result)
        assert result.endswith(".validation.json")

    def test_write_validation_json_non_pdf_uses_ext_token(self, tmp_path):
        dv = DocumentValidation(
            document_id="token_test",
            source_file="evidence/photo.jpg",
            source_page_count=1,
            output_page_count=1,
        )
        dv.pages = [
            {"page_num": 1, "ocr_method": "ImageOnly", "ocr_language": "",
             "ocr_confidence": 0.0, "text_length": 0, "has_text": False, "status": "image_only"},
        ]
        dv = finalize_validation(dv)
        result = write_validation_json(dv, str(tmp_path), "", __version__)
        assert result is not None
        assert os.path.basename(result) == "photo__jpg.validation.json"

    def test_write_validation_json_schema_valid(self, tmp_path):
        dv = DocumentValidation(
            document_id="schema_test",
            source_file="test.pdf",
            source_page_count=2,
            output_page_count=2,
        )
        dv.pages = [
            {"page_num": 1, "ocr_method": "PaddleOCR", "ocr_language": "",
             "ocr_confidence": 0.85, "text_length": 200, "has_text": True, "status": "ok"},
            {"page_num": 2, "ocr_method": "Tesseract", "ocr_language": "",
             "ocr_confidence": 0.70, "text_length": 100, "has_text": True, "status": "fallback"},
        ]
        dv = finalize_validation(dv)
        result_path = write_validation_json(dv, str(tmp_path), "", "0.4.0")
        assert result_path is not None

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Verify schema structure
        assert data["schema_version"] == "1.0"
        assert data["document_id"] == "schema_test"
        assert data["source_file"] == "test.pdf"
        assert "processing" in data
        assert data["processing"]["pipeline_version"] == "0.4.0"
        assert "page_count" in data
        assert data["page_count"]["source"] == 2
        assert data["page_count"]["output"] == 2
        assert data["page_count"]["match"] is True
        assert "quality" in data
        assert data["quality"]["classification"] in (
            "high_quality", "acceptable", "degraded", "review_required"
        )
        assert "pages" in data
        assert len(data["pages"]) == 2
        assert "output_hash" in data

    def test_write_validation_json_creates_directories(self, tmp_path):
        dv = DocumentValidation(document_id="dir_test", source_file="deep/nested/doc.pdf")
        dv.pages = []
        dv = finalize_validation(dv)
        result = write_validation_json(dv, str(tmp_path), "deep/nested", "0.4.0")
        assert result is not None
        assert os.path.exists(result)
        # Verify nested directory was created
        expected_dir = os.path.join(str(tmp_path), "EXPORT", "VALIDATION", "deep", "nested")
        assert os.path.isdir(expected_dir)

    def test_write_validation_json_no_subfolder(self, tmp_path):
        dv = DocumentValidation(document_id="root_test", source_file="root_doc.pdf")
        dv.pages = []
        dv = finalize_validation(dv)
        result = write_validation_json(dv, str(tmp_path), ".", "0.4.0")
        assert result is not None
        assert os.path.exists(result)
        expected_dir = os.path.join(str(tmp_path), "EXPORT", "VALIDATION")
        assert os.path.dirname(result) == expected_dir


# ---------------------------------------------------------------------------
# Tests: extract_paddle_lines confidence integration
# ---------------------------------------------------------------------------

class TestExtractPaddleLinesConfidence:
    """Test that extract_paddle_lines returns confidence as third tuple element."""

    @pytest.fixture(autouse=True)
    def _load_function(self):
        """Load extract_paddle_lines from ocr_gpu_async.py."""
        try:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            from ocr_gpu_async import extract_paddle_lines
            self.extract_paddle_lines = extract_paddle_lines
        except ImportError:
            # Fallback: regex extraction
            import re
            import types

            import numpy as np
            from PIL import Image

            src_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "ocr_gpu_async.py"
            )
            with open(src_path, "r", encoding="utf-8") as f:
                source = f.read()

            mod = types.ModuleType("ocr_utils_val")
            mod.__file__ = src_path
            mod.os = os
            mod.np = np
            mod.Image = Image

            for func_name in ["to_plain_list", "extract_paddle_lines"]:
                pattern = rf'^(def {func_name}\(.*?\n(?:(?:    .*\n|[ \t]*\n)*))'
                match = re.search(pattern, source, re.MULTILINE)
                if match:
                    func_source = match.group(1)
                    exec(compile(func_source, src_path, "exec"), mod.__dict__)

            self.extract_paddle_lines = mod.extract_paddle_lines

    def test_v2_returns_confidence(self):
        result = [[
            [[[10, 10], [100, 10], [100, 30], [10, 30]], ("Hello", 0.95)],
        ]]
        lines = self.extract_paddle_lines(result)
        assert len(lines) == 1
        assert len(lines[0]) == 3
        assert lines[0][0] == "Hello"
        assert abs(lines[0][2] - 0.95) < 0.01

    def test_v2_confidence_default_zero_when_missing(self):
        """If line[1] has no confidence index, default to 0.0."""
        result = [[
            [[[10, 10], [100, 10], [100, 30], [10, 30]], ("Only text",)],
        ]]
        lines = self.extract_paddle_lines(result)
        assert len(lines) == 1
        assert lines[0][2] == 0.0

    def test_dict_format_returns_confidence_with_scores(self):
        result = [{
            "rec_texts": ["Line A", "Line B"],
            "rec_scores": [0.88, 0.76],
            "dt_polys": [
                [[0, 0], [100, 0], [100, 30], [0, 30]],
                [[0, 40], [100, 40], [100, 70], [0, 70]],
            ],
        }]
        lines = self.extract_paddle_lines(result)
        assert len(lines) == 2
        assert abs(lines[0][2] - 0.88) < 0.01
        assert abs(lines[1][2] - 0.76) < 0.01

    def test_dict_format_confidence_default_without_scores(self):
        result = [{
            "rec_texts": ["No scores here"],
            "dt_polys": [[[0, 0], [100, 0], [100, 30], [0, 30]]],
        }]
        lines = self.extract_paddle_lines(result)
        assert len(lines) == 1
        assert lines[0][2] == 0.0
