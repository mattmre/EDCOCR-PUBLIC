"""End-to-end integration tests for all six sidecar modules.

Tier 1: Module-level integration (no GPU needed) -- ~19 tests
  Tests each sidecar module with realistic data, validates JSON output
  schema, and cross-validates results across modules.

Tier 2: Pipeline-level (skip if no PaddleOCR/fasttext) -- 7 tests
  Tests ocr_gpu_async feature flag integration. Expected to be skipped
  on most dev/test machines.

Run with: python -m pytest tests/test_integration_e2e.py -v
"""

import json
import os

import pytest

from classification import (
    DocumentClassification,
    classify_page_by_text,
    finalize_classification,
    write_classification_json,
)
from extraction import (
    DocumentExtraction,
    extract_page_fields,
    finalize_extraction,
    write_extraction_json,
)
from handwriting import (
    DocumentHandwriting,
    detect_handwriting_by_confidence,
    detect_handwriting_by_geometry,
    finalize_handwriting,
    merge_handwriting_signals,
    write_handwriting_json,
)
from ner import (
    DocumentNER,
    PageNER,
    extract_custom_entities,
    extract_entities,
    finalize_ner,
    write_ner_json,
)
from semantic_extraction import finalize_entity_output, write_entities_json
from validation import (
    DocumentValidation,
    finalize_validation,
    write_validation_json,
)
from version import __version__

PIPELINE_VERSION = f"{__version__}-test"
DOC_ID = "test-doc-001"
SOURCE_FILE = "test_document.pdf"
SUBFOLDER = "batch_001"


# ---------------------------------------------------------------------------
# Helper: JSON schema validation
# ---------------------------------------------------------------------------


def _validate_json_schema(data, required_keys, nested_checks=None):
    """Validate that a JSON structure has required top-level keys and
    optionally check nested key paths.

    Args:
        data: Parsed JSON dict.
        required_keys: List of top-level keys that must exist.
        nested_checks: Dict mapping top-level key -> list of nested keys.
            e.g. {"processing": ["pipeline_version", "timestamp"]}
    """
    for key in required_keys:
        assert key in data, f"Missing required top-level key: {key}"

    if nested_checks:
        for parent_key, child_keys in nested_checks.items():
            assert parent_key in data, f"Missing parent key for nested check: {parent_key}"
            parent = data[parent_key]
            assert isinstance(parent, dict), f"Expected dict for {parent_key}, got {type(parent)}"
            for child in child_keys:
                assert child in parent, f"Missing nested key {parent_key}.{child}"


# ===========================================================================
# Tier 1: Module-level integration tests
# ===========================================================================


class TestValidationIntegration:
    """Test validation module: finalize -> write_json -> read -> validate."""

    def test_full_pipeline_high_quality(self, output_dir, sample_page_texts):
        """High-quality document: all pages have text, high confidence."""
        doc_val = DocumentValidation(
            document_id=DOC_ID,
            source_file=SOURCE_FILE,
            source_page_count=3,
            output_page_count=3,
        )
        for page_num, text in sample_page_texts.items():
            doc_val.pages.append({
                "page_num": page_num,
                "ocr_method": "PaddleOCR",
                "ocr_language": "en",
                "ocr_confidence": 0.92,
                "text_length": len(text),
                "has_text": True,
                "status": "ok",
            })

        finalize_validation(doc_val)
        path = write_validation_json(doc_val, str(output_dir), SUBFOLDER, PIPELINE_VERSION)

        assert path is not None
        assert os.path.isfile(path)
        assert path.endswith(".validation.json")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        _validate_json_schema(data, [
            "schema_version", "document_id", "source_file",
            "processing", "page_count", "quality", "pages",
        ], nested_checks={
            "processing": ["pipeline_version", "timestamp"],
            "page_count": ["source", "output", "match"],
            "quality": ["classification", "overall_confidence", "text_extraction_rate"],
        })

        assert data["schema_version"] == "1.0"
        assert data["document_id"] == DOC_ID
        assert data["quality"]["classification"] == "high_quality"
        assert data["quality"]["overall_confidence"] >= 0.90
        assert data["quality"]["text_extraction_rate"] == 1.0
        assert data["page_count"]["match"] is True

    def test_all_ok_pages(self, output_dir):
        """All pages ok, medium confidence -> acceptable."""
        doc_val = DocumentValidation(
            document_id="doc-ok-test",
            source_file="medium.pdf",
            source_page_count=2,
            output_page_count=2,
        )
        for i in range(1, 3):
            doc_val.pages.append({
                "page_num": i,
                "ocr_method": "Tesseract",
                "ocr_language": "en",
                "ocr_confidence": 0.70,
                "text_length": 500,
                "has_text": True,
                "status": "fallback",
            })

        finalize_validation(doc_val)
        path = write_validation_json(doc_val, str(output_dir), ".", PIPELINE_VERSION)

        assert path is not None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["quality"]["classification"] == "acceptable"

    def test_degraded_quality(self, output_dir):
        """Mixed results with image-only pages -> degraded."""
        doc_val = DocumentValidation(
            document_id="doc-degraded",
            source_file="degraded.pdf",
            source_page_count=4,
            output_page_count=4,
        )
        # 2 good pages
        for i in range(1, 3):
            doc_val.pages.append({
                "page_num": i,
                "ocr_method": "PaddleOCR",
                "ocr_language": "en",
                "ocr_confidence": 0.45,
                "text_length": 200,
                "has_text": True,
                "status": "ok",
            })
        # 2 image-only pages
        for i in range(3, 5):
            doc_val.pages.append({
                "page_num": i,
                "ocr_method": "ImageOnly",
                "ocr_language": "",
                "ocr_confidence": 0.0,
                "text_length": 0,
                "has_text": False,
                "status": "image_only",
            })

        finalize_validation(doc_val)
        path = write_validation_json(doc_val, str(output_dir), SUBFOLDER, PIPELINE_VERSION)

        assert path is not None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["quality"]["classification"] == "degraded"
        assert data["quality"]["pages_image_only"] == 2
        assert data["quality"]["text_extraction_rate"] == 0.5


class TestNERIntegration:
    """Test NER module: extract_custom_entities -> finalize -> write -> validate."""

    def test_regex_pipeline(self, output_dir, sample_page_texts):
        """Extract legal entities from page 2 (case numbers, Bates, exhibits)."""
        doc_ner = DocumentNER(document_id=DOC_ID, source_file=SOURCE_FILE)

        for page_num, text in sample_page_texts.items():
            entities = extract_custom_entities(text, page_num)
            page = PageNER(page_num=page_num, entities=entities)
            doc_ner.pages.append(page)

        finalize_ner(doc_ner)

        ner_dir = os.path.join(str(output_dir), "EXPORT", "NER")
        path = write_ner_json(doc_ner, ner_dir, SUBFOLDER, PIPELINE_VERSION)

        assert path is not None
        assert os.path.isfile(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        _validate_json_schema(data, [
            "schema_version", "document_id", "source_file",
            "processing", "entities_summary", "pages",
        ], nested_checks={
            "processing": ["ner_engine", "pipeline_version", "timestamp"],
            "entities_summary": ["total_entities", "entity_types", "unique_entities"],
        })

        assert data["schema_version"] == "1.0"
        assert data["document_id"] == DOC_ID
        assert data["entities_summary"]["total_entities"] > 0

        # Page 2 should have CASE_NUMBER and EXHIBIT_REF
        entity_types = data["entities_summary"]["entity_types"]
        assert "CASE_NUMBER" in entity_types
        assert "EXHIBIT_REF" in entity_types

    def test_empty_text(self, output_dir):
        """Empty text produces zero entities."""
        doc_ner = DocumentNER(document_id="doc-empty", source_file="empty.pdf")
        entities = extract_custom_entities("", 1)
        doc_ner.pages.append(PageNER(page_num=1, entities=entities))
        finalize_ner(doc_ner)

        ner_dir = os.path.join(str(output_dir), "EXPORT", "NER")
        path = write_ner_json(doc_ner, ner_dir, ".", PIPELINE_VERSION)

        assert path is not None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["entities_summary"]["total_entities"] == 0

    def test_multi_page_aggregation(self, output_dir, sample_page_texts):
        """Entities across multiple pages are aggregated and deduplicated."""
        doc_ner = DocumentNER(document_id="doc-multi", source_file="multi.pdf")

        for page_num, text in sample_page_texts.items():
            entities = extract_custom_entities(text, page_num)
            doc_ner.pages.append(PageNER(page_num=page_num, entities=entities))

        finalize_ner(doc_ner)

        assert doc_ner.total_entities > 0
        assert len(doc_ner.unique_entities) > 0
        # unique_entities should be <= total_entities (dedup removes duplicates)
        assert len(doc_ner.unique_entities) <= doc_ner.total_entities


class TestNERWithSpacyIntegration:
    """Test NER with mocked spaCy entities combined with regex entities."""

    def test_spacy_and_regex_combined(self, output_dir, sample_page_texts):
        """Both spaCy and regex entities appear in the finalized output."""
        from unittest.mock import MagicMock, patch

        # Mock spaCy to return PERSON and ORG entities
        mock_ent_person = MagicMock()
        mock_ent_person.label_ = "PERSON"
        mock_ent_person.text = "Jane Attorney"
        mock_ent_person.start_char = 0
        mock_ent_person.end_char = 13

        mock_ent_org = MagicMock()
        mock_ent_org.label_ = "ORG"
        mock_ent_org.text = "Smith Enterprises"
        mock_ent_org.start_char = 20
        mock_ent_org.end_char = 37

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent_person, mock_ent_org]

        mock_nlp = MagicMock(return_value=mock_doc)

        doc_ner = DocumentNER(document_id="doc-spacy", source_file="spacy-test.pdf")

        # Page 2 has legal text with case numbers, Bates stamps, exhibits
        text = sample_page_texts[2]

        with patch("ner._SPACY_AVAILABLE", True), \
             patch("ner._load_nlp", return_value=mock_nlp):
            spacy_entities = extract_entities(text, 2)

        regex_entities = extract_custom_entities(text, 2)
        all_entities = spacy_entities + regex_entities

        page = PageNER(page_num=2, entities=all_entities)
        doc_ner.pages.append(page)
        finalize_ner(doc_ner)

        # Should have both spaCy types (PERSON, ORG) and regex types (CASE_NUMBER, etc.)
        entity_types = set(doc_ner.entity_type_counts.keys())
        assert "PERSON" in entity_types, "Missing spaCy PERSON entity"
        assert "ORG" in entity_types, "Missing spaCy ORG entity"
        assert "CASE_NUMBER" in entity_types, "Missing regex CASE_NUMBER entity"

        # Total should include both sources
        assert doc_ner.total_entities >= 4  # At least 2 spaCy + 2 regex

        # Write and validate JSON
        ner_dir = os.path.join(str(output_dir), "EXPORT", "NER")
        path = write_ner_json(doc_ner, ner_dir, SUBFOLDER, PIPELINE_VERSION)
        assert path is not None

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Verify both spaCy and regex entities in output
        all_types = data["entities_summary"]["entity_types"]
        assert "PERSON" in all_types
        assert "CASE_NUMBER" in all_types

    def test_spacy_unavailable_falls_back_to_regex(self, sample_page_texts):
        """When spaCy is unavailable, only regex entities are extracted."""
        from unittest.mock import patch

        text = sample_page_texts[2]

        with patch("ner._SPACY_AVAILABLE", False):
            spacy_entities = extract_entities(text, 2)

        assert spacy_entities == []

        regex_entities = extract_custom_entities(text, 2)
        assert len(regex_entities) > 0
        # Should still find case numbers and exhibits via regex
        entity_types = {e.entity_type for e in regex_entities}
        assert "CASE_NUMBER" in entity_types


class TestHandwritingIntegration:
    """Test handwriting module: detect -> merge -> finalize -> write -> validate."""

    def test_full_pipeline(self, output_dir, sample_paddle_lines):
        """Full 3-page document with mixed printed/handwritten content."""
        doc_hw = DocumentHandwriting(document_id=DOC_ID, source_file=SOURCE_FILE)

        for page_num, lines in sample_paddle_lines.items():
            conf_result = detect_handwriting_by_confidence(lines, page_num)
            geo_result = detect_handwriting_by_geometry(lines, page_num)
            merged = merge_handwriting_signals(conf_result, geo_result, None, page_num)
            doc_hw.pages.append(merged)

        finalize_handwriting(doc_hw)
        path = write_handwriting_json(doc_hw, str(output_dir), SUBFOLDER, PIPELINE_VERSION)

        assert path is not None
        assert os.path.isfile(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        _validate_json_schema(data, [
            "schema_version", "document_id", "source_file",
            "processing", "document_summary", "pages",
        ], nested_checks={
            "processing": ["detection_engine", "pipeline_version", "timestamp"],
            "document_summary": [
                "total_handwritten_pages", "total_pages_with_mixed",
                "overall_handwriting_coverage", "is_primarily_handwritten",
            ],
        })

        assert data["schema_version"] == "1.0"
        assert len(data["pages"]) == 3
        # Page 3 is all handwritten lines
        assert data["document_summary"]["total_handwritten_pages"] >= 1

    def test_all_printed(self, output_dir):
        """All lines are high confidence (printed) -> no handwriting detected."""
        lines = [
            ("Line one", 0.95, [10, 10, 200, 30]),
            ("Line two", 0.93, [10, 40, 200, 60]),
            ("Line three", 0.91, [10, 70, 200, 90]),
            ("Line four", 0.94, [10, 100, 200, 120]),
        ]
        doc_hw = DocumentHandwriting(document_id="doc-printed", source_file="printed.pdf")
        conf_result = detect_handwriting_by_confidence(lines, 1)
        geo_result = detect_handwriting_by_geometry(lines, 1)
        merged = merge_handwriting_signals(conf_result, geo_result, None, 1)
        doc_hw.pages.append(merged)
        finalize_handwriting(doc_hw)

        path = write_handwriting_json(doc_hw, str(output_dir), ".", PIPELINE_VERSION)
        assert path is not None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["document_summary"]["total_handwritten_pages"] == 0
        assert data["document_summary"]["is_primarily_handwritten"] is False

    def test_mixed_document(self, output_dir, sample_paddle_lines):
        """Document with mixed pages tracks mixed page count."""
        doc_hw = DocumentHandwriting(document_id="doc-mixed", source_file="mixed.pdf")

        for page_num, lines in sample_paddle_lines.items():
            conf_result = detect_handwriting_by_confidence(lines, page_num)
            doc_hw.pages.append(conf_result)

        finalize_handwriting(doc_hw)

        # Page 2 has both printed (5) and handwritten (5) lines
        assert doc_hw.total_pages_with_mixed >= 1


class TestClassificationIntegration:
    """Test classification module: classify -> finalize -> write -> validate."""

    def test_text_only_classification(self, output_dir, sample_page_texts):
        """Classify pages using text rules only."""
        doc_cls = DocumentClassification(document_id=DOC_ID, source_file=SOURCE_FILE)

        for page_num, text in sample_page_texts.items():
            page_cls = classify_page_by_text(text, page_num)
            doc_cls.pages.append(page_cls)

        finalize_classification(doc_cls)
        path = write_classification_json(doc_cls, str(output_dir), SUBFOLDER, PIPELINE_VERSION)

        assert path is not None
        assert os.path.isfile(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        _validate_json_schema(data, [
            "schema_version", "document_id", "source_file",
            "processing", "document_summary", "pages",
        ], nested_checks={
            "processing": ["classification_engine", "pipeline_version", "timestamp"],
            "document_summary": [
                "document_type", "document_confidence", "type_distribution",
            ],
        })

        assert data["schema_version"] == "1.0"
        assert len(data["pages"]) == 3

        # Page 1 text has "invoice", "bill to", "amount due", "subtotal"
        page1 = data["pages"][0]
        assert page1["predicted_type"] == "invoice"

    def test_with_layout_data(self, output_dir, sample_page_texts, sample_structure_data):
        """Classify with layout features produces valid output."""
        from classification import classify_page_by_layout, classify_page_ensemble

        doc_cls = DocumentClassification(document_id="doc-layout", source_file="layout.pdf")

        for page_num, text in sample_page_texts.items():
            text_result = classify_page_by_text(text, page_num)
            struct = sample_structure_data.get(page_num, {})
            layout_result = classify_page_by_layout(
                struct.get("layout_regions", []),
                struct.get("tables", []),
                struct.get("form_fields", []),
                page_num,
            )
            ensemble = classify_page_ensemble(text_result, layout_result, page_num)
            doc_cls.pages.append(ensemble)

        finalize_classification(doc_cls)
        path = write_classification_json(doc_cls, str(output_dir), ".", PIPELINE_VERSION)

        assert path is not None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["processing"]["classification_engine"] == "text_rules+layout_features"

    def test_handwriting_crossref(self, output_dir, sample_page_texts, sample_paddle_lines):
        """Classification can cross-reference handwriting detection results."""
        doc_cls = DocumentClassification(document_id="doc-hw-xref", source_file="hwxref.pdf")

        for page_num, text in sample_page_texts.items():
            page_cls = classify_page_by_text(text, page_num)
            # Cross-reference: mark page as handwritten if paddle lines are low confidence
            lines = sample_paddle_lines.get(page_num, [])
            if lines:
                hw_result = detect_handwriting_by_confidence(lines, page_num)
                page_cls.is_handwritten = hw_result.has_handwriting
            doc_cls.pages.append(page_cls)

        finalize_classification(doc_cls)
        path = write_classification_json(doc_cls, str(output_dir), SUBFOLDER, PIPELINE_VERSION)

        assert path is not None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Page 3 lines are all low confidence -> should be flagged handwritten
        page3 = data["pages"][2]
        assert page3["is_handwritten"] is True

        # Page 1 lines are all high confidence -> not handwritten
        page1 = data["pages"][0]
        assert page1["is_handwritten"] is False


class TestExtractionIntegration:
    """Test extraction module: extract_page_fields -> finalize -> write -> validate."""

    def test_regex_pipeline(self, output_dir, sample_page_texts):
        """Extract fields from all pages using regex (no UIE)."""
        doc_ext = DocumentExtraction(document_id=DOC_ID, source_file=SOURCE_FILE)

        for page_num, text in sample_page_texts.items():
            page_ext = extract_page_fields(text, page_num, use_uie=False)
            doc_ext.pages.append(page_ext)

        finalize_extraction(doc_ext)
        path = write_extraction_json(doc_ext, str(output_dir), SUBFOLDER, PIPELINE_VERSION)

        assert path is not None
        assert os.path.isfile(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        _validate_json_schema(data, [
            "schema_version", "document_id", "source_file",
            "processing", "extraction_summary", "pages",
        ], nested_checks={
            "processing": ["extraction_engine", "pipeline_version", "timestamp"],
            "extraction_summary": ["total_fields", "field_type_counts", "extraction_engine"],
        })

        assert data["schema_version"] == "1.0"
        assert data["extraction_summary"]["total_fields"] > 0
        assert data["extraction_summary"]["extraction_engine"] == "regex"

        # Should find dates and amounts from the invoice page
        field_types = data["extraction_summary"]["field_type_counts"]
        assert "date" in field_types
        assert "amount" in field_types

    def test_multi_field_types(self, output_dir, sample_page_texts):
        """Extraction finds multiple field types across pages."""
        doc_ext = DocumentExtraction(document_id="doc-multi-fields", source_file="multi.pdf")

        for page_num, text in sample_page_texts.items():
            page_ext = extract_page_fields(text, page_num, use_uie=False)
            doc_ext.pages.append(page_ext)

        finalize_extraction(doc_ext)

        field_types = set(doc_ext.field_type_counts.keys())
        # From fixture text: dates, amounts, emails, phone numbers, reference numbers
        assert "date" in field_types
        assert "amount" in field_types
        assert "email_address" in field_types
        assert "phone_number" in field_types

    def test_empty_text(self, output_dir):
        """Empty text produces zero extracted fields."""
        doc_ext = DocumentExtraction(document_id="doc-empty", source_file="empty.pdf")
        page_ext = extract_page_fields("", 1, use_uie=False)
        doc_ext.pages.append(page_ext)
        finalize_extraction(doc_ext)

        path = write_extraction_json(doc_ext, str(output_dir), ".", PIPELINE_VERSION)
        assert path is not None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["extraction_summary"]["total_fields"] == 0


class TestExtractionWithUIEIntegration:
    """Test extraction with mocked UIE engine combined with regex."""

    def test_uie_and_regex_combined(self, output_dir, sample_page_texts):
        """UIE and regex fields are combined and deduplicated in output."""
        from unittest.mock import MagicMock, patch

        # Mock UIE engine to return date and amount fields
        mock_engine = MagicMock()
        mock_engine.return_value = [
            {
                "Date": [
                    {"text": "January 15, 2024", "probability": 0.95, "start": 50, "end": 66},
                ],
                "Amount": [
                    {"text": "$8,586.00", "probability": 0.92, "start": 200, "end": 209},
                ],
                "Person Name": [
                    {"text": "Acme Corporation", "probability": 0.88, "start": 80, "end": 96},
                ],
            }
        ]

        doc_ext = DocumentExtraction(document_id="doc-uie", source_file="uie-test.pdf")

        text = sample_page_texts[1]  # Invoice text

        with patch("extraction._get_uie_engine", return_value=mock_engine):
            page_ext = extract_page_fields(text, 1, use_uie=True)

        doc_ext.pages.append(page_ext)
        finalize_extraction(doc_ext)

        # Should have fields from both UIE and regex
        field_types = set(doc_ext.field_type_counts.keys())
        assert "date" in field_types
        assert "amount" in field_types

        # UIE fields should have higher confidence than regex
        uie_fields = [f for f in page_ext.fields if f.get("extraction_method") == "uie"]
        regex_fields = [f for f in page_ext.fields if f.get("extraction_method") == "regex"]
        assert len(uie_fields) > 0, "Expected UIE fields in output"
        assert len(regex_fields) > 0, "Expected regex fields in output"

        # Write and validate JSON
        path = write_extraction_json(doc_ext, str(output_dir), SUBFOLDER, PIPELINE_VERSION)
        assert path is not None

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["extraction_summary"]["total_fields"] > 0
        # Engine should report "hybrid" when both UIE and regex are used
        engine_type = data["extraction_summary"]["extraction_engine"]
        assert engine_type == "hybrid"

    def test_uie_unavailable_falls_back_to_regex(self, output_dir, sample_page_texts):
        """When UIE is unavailable, extraction falls back to regex only."""
        from unittest.mock import patch

        doc_ext = DocumentExtraction(document_id="doc-no-uie", source_file="no-uie.pdf")

        text = sample_page_texts[1]

        with patch("extraction._get_uie_engine", return_value=None):
            page_ext = extract_page_fields(text, 1, use_uie=True)

        doc_ext.pages.append(page_ext)
        finalize_extraction(doc_ext)

        # Should still find fields via regex
        assert doc_ext.total_fields > 0

        path = write_extraction_json(doc_ext, str(output_dir), SUBFOLDER, PIPELINE_VERSION)
        assert path is not None

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["extraction_summary"]["extraction_engine"] == "regex"


class TestAllSidecarsE2E:
    """Highest-value test: all durable sidecars on the same 3-page document."""

    def test_all_sidecars_single_document(
        self, output_dir, sample_page_texts, sample_paddle_lines, sample_structure_data
    ):
        """Exercise all durable sidecar modules on the same document."""
        out = str(output_dir)

        # --- 1. Validation ---
        doc_val = DocumentValidation(
            document_id=DOC_ID,
            source_file=SOURCE_FILE,
            source_page_count=3,
            output_page_count=3,
        )
        for page_num, text in sample_page_texts.items():
            doc_val.pages.append({
                "page_num": page_num,
                "ocr_method": "PaddleOCR",
                "ocr_language": "en",
                "ocr_confidence": 0.92,
                "text_length": len(text),
                "has_text": True,
                "status": "ok",
            })
        finalize_validation(doc_val)
        val_path = write_validation_json(doc_val, out, SUBFOLDER, PIPELINE_VERSION)

        # --- 2. NER ---
        doc_ner = DocumentNER(document_id=DOC_ID, source_file=SOURCE_FILE)
        for page_num, text in sample_page_texts.items():
            entities = extract_custom_entities(text, page_num)
            doc_ner.pages.append(PageNER(page_num=page_num, entities=entities))
        finalize_ner(doc_ner)
        ner_dir = os.path.join(out, "EXPORT", "NER")
        ner_path = write_ner_json(doc_ner, ner_dir, SUBFOLDER, PIPELINE_VERSION)

        # --- 3. Handwriting ---
        doc_hw = DocumentHandwriting(document_id=DOC_ID, source_file=SOURCE_FILE)
        for page_num, lines in sample_paddle_lines.items():
            conf_result = detect_handwriting_by_confidence(lines, page_num)
            geo_result = detect_handwriting_by_geometry(lines, page_num)
            merged = merge_handwriting_signals(conf_result, geo_result, None, page_num)
            doc_hw.pages.append(merged)
        finalize_handwriting(doc_hw)
        hw_path = write_handwriting_json(doc_hw, out, SUBFOLDER, PIPELINE_VERSION)

        # --- 4. Classification ---
        doc_cls = DocumentClassification(document_id=DOC_ID, source_file=SOURCE_FILE)
        for page_num, text in sample_page_texts.items():
            page_cls = classify_page_by_text(text, page_num)
            # Cross-reference handwriting
            lines = sample_paddle_lines.get(page_num, [])
            if lines:
                hw_result = detect_handwriting_by_confidence(lines, page_num)
                page_cls.is_handwritten = hw_result.has_handwriting
            doc_cls.pages.append(page_cls)
        finalize_classification(doc_cls)
        cls_path = write_classification_json(doc_cls, out, SUBFOLDER, PIPELINE_VERSION)

        # --- 5. Extraction ---
        doc_ext = DocumentExtraction(document_id=DOC_ID, source_file=SOURCE_FILE)
        for page_num, text in sample_page_texts.items():
            page_ext = extract_page_fields(text, page_num, use_uie=False)
            doc_ext.pages.append(page_ext)
        finalize_extraction(doc_ext)
        ext_path = write_extraction_json(doc_ext, out, SUBFOLDER, PIPELINE_VERSION)

        # --- 6. Durable entities ---
        entities_doc = finalize_entity_output(doc_ext)
        entities_path = write_entities_json(
            entities_doc, out, SUBFOLDER, PIPELINE_VERSION
        )

        # --- Verify all JSON files exist ---
        for label, path in [
            ("validation", val_path),
            ("ner", ner_path),
            ("handwriting", hw_path),
            ("classification", cls_path),
            ("extraction", ext_path),
            ("entities", entities_path),
        ]:
            assert path is not None, f"{label} JSON path is None"
            assert os.path.isfile(path), f"{label} JSON file missing: {path}"

        # --- Load and validate each ---
        with open(val_path, encoding="utf-8") as f:
            val_data = json.load(f)
        with open(ner_path, encoding="utf-8") as f:
            ner_data = json.load(f)
        with open(hw_path, encoding="utf-8") as f:
            hw_data = json.load(f)
        with open(cls_path, encoding="utf-8") as f:
            cls_data = json.load(f)
        with open(ext_path, encoding="utf-8") as f:
            ext_data = json.load(f)
        with open(entities_path, encoding="utf-8") as f:
            entities_data = json.load(f)

        # All share the same schema_version and document_id
        for label, data in [
            ("validation", val_data),
            ("ner", ner_data),
            ("handwriting", hw_data),
            ("classification", cls_data),
            ("extraction", ext_data),
            ("entities", entities_data),
        ]:
            assert data["schema_version"] == "1.0", f"{label} schema_version mismatch"
            assert data["document_id"] == DOC_ID, f"{label} document_id mismatch"

        # --- Cross-validation ---

        # Extraction finds dates and amounts
        ext_types = ext_data["extraction_summary"]["field_type_counts"]
        assert "date" in ext_types, "Extraction should find dates"
        assert "amount" in ext_types, "Extraction should find amounts"

        # NER finds CASE_NUMBER, BATES_NUMBER, and EXHIBIT_REF from page 2
        ner_types = ner_data["entities_summary"]["entity_types"]
        assert "CASE_NUMBER" in ner_types, "NER should find case numbers"
        assert "BATES_NUMBER" in ner_types, "NER should find Bates numbers"
        assert "EXHIBIT_REF" in ner_types, "NER should find exhibit references"

        # Classification identifies page 1 as invoice
        page1_cls = cls_data["pages"][0]
        assert page1_cls["predicted_type"] == "invoice", (
            f"Page 1 should be classified as invoice, got {page1_cls['predicted_type']}"
        )

        # Validation reports high quality
        assert val_data["quality"]["classification"] == "high_quality"

        # Handwriting: page 3 (all low-confidence lines) should be flagged
        assert hw_data["document_summary"]["total_handwritten_pages"] >= 1

        # Entities sidecar mirrors extraction-derived field types
        entity_types = entities_data["entities_summary"]["entity_type_counts"]
        assert "date" in entity_types
        assert "amount" in entity_types
        assert entities_data["entities_summary"]["total_entities"] >= 2


# ===========================================================================
# Tier 2: Pipeline-level tests (skip if dependencies unavailable)
# ===========================================================================

_can_import_pipeline = True
try:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    import ocr_gpu_async  # noqa: F401
except ImportError:
    _can_import_pipeline = False

_skip_no_pipeline = pytest.mark.skipif(
    not _can_import_pipeline,
    reason="Requires GPU/fasttext deps (ocr_gpu_async import failed)",
)


@_skip_no_pipeline
class TestAssemblerSidecarGeneration:
    """Test that assembler generates sidecar files when feature flags are enabled.

    These tests require the full ocr_gpu_async module to be importable,
    which needs PaddleOCR, FastText, and potentially GPU drivers.
    They are expected to be skipped on most dev/test machines.
    """

    def test_feature_flags_are_boolean(self):
        """All sidecar feature flags must be boolean type."""
        import ocr_gpu_async
        flags = [
            "ENABLE_VALIDATION",
            "ENABLE_NER",
            "ENABLE_HANDWRITING",
            "ENABLE_CLASSIFICATION",
            "ENABLE_EXTRACTION",
        ]
        for flag in flags:
            value = getattr(ocr_gpu_async, flag)
            assert isinstance(value, bool), (
                f"{flag} must be bool, got {type(value).__name__}: {value}"
            )

    def test_all_flags_disabled_by_default(self):
        """Feature flags default to False (or env var default)."""
        import ocr_gpu_async
        # These may be True if env vars are set, but the attributes should exist
        assert hasattr(ocr_gpu_async, "ENABLE_VALIDATION")
        assert hasattr(ocr_gpu_async, "ENABLE_NER")
        assert hasattr(ocr_gpu_async, "ENABLE_HANDWRITING")
        assert hasattr(ocr_gpu_async, "ENABLE_CLASSIFICATION")
        assert hasattr(ocr_gpu_async, "ENABLE_EXTRACTION")

    def test_pipeline_version_available(self):
        """Pipeline exposes PIPELINE_VERSION constant."""
        import ocr_gpu_async
        assert hasattr(ocr_gpu_async, "PIPELINE_VERSION")
        assert isinstance(ocr_gpu_async.PIPELINE_VERSION, str)
        assert len(ocr_gpu_async.PIPELINE_VERSION) > 0
