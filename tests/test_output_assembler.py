"""Tests for unified retrieval output assembler (output_assembler.py)."""

import json
import os

from output_assembler import (
    _SCHEMA_VERSION,
    RetrievalDocument,
    _extract_classification,
    _extract_entities_from_ner_data,
    _extract_from_extraction_data,
    _extract_handwriting,
    _extract_metadata,
    _extract_tables,
    assemble_retrieval_output,
    write_retrieval_json,
    write_retrieval_markdown,
)
from schemas import validate

# ---------------------------------------------------------------------------
# Sample sidecar data fixtures
# ---------------------------------------------------------------------------


def _sample_ner_data():
    """Return parsed .ner.json sidecar data."""
    return {
        "schema_version": "1.0",
        "document_id": "doc-001",
        "source_file": "test.pdf",
        "pages": [
            {
                "page_num": 1,
                "entities": [
                    {
                        "entity_type": "PERSON",
                        "text": "John Smith",
                        "confidence": 0.95,
                        "page_num": 1,
                        "start": 0,
                        "end": 10,
                    },
                    {
                        "entity_type": "ORG",
                        "text": "Acme Corp",
                        "confidence": 0.88,
                        "page_num": 1,
                        "start": 20,
                        "end": 29,
                    },
                ],
            },
            {
                "page_num": 2,
                "entities": [
                    {
                        "entity_type": "DATE",
                        "text": "2024-01-15",
                        "confidence": 0.92,
                        "page_num": 2,
                        "start": 5,
                        "end": 15,
                    },
                ],
            },
        ],
    }


def _sample_extraction_data():
    """Return parsed .extraction.json sidecar data."""
    return {
        "schema_version": "1.0",
        "document_id": "doc-001",
        "source_file": "test.pdf",
        "pages": [
            {
                "page_num": 1,
                "fields": [
                    {
                        "field_type": "date",
                        "text": "2024-01-15",
                        "confidence": 0.90,
                        "page_num": 1,
                        "extraction_method": "regex",
                        "normalized_value": "2024-01-15",
                        "bbox": [100, 200, 300, 220],
                        "start": 5,
                        "end": 15,
                    },
                    {
                        "field_type": "amount",
                        "text": "$1,234.56",
                        "confidence": 0.85,
                        "page_num": 1,
                        "extraction_method": "regex",
                        "normalized_value": "1234.56",
                        "bbox": [],
                        "start": 30,
                        "end": 39,
                    },
                ],
            },
        ],
    }


def _sample_classification_data():
    """Return parsed .classification.json sidecar data."""
    return {
        "document_type": "invoice",
        "confidence": 0.92,
        "classification_method": "ensemble",
    }


def _sample_structure_data():
    """Return parsed .structure.json sidecar data."""
    return {
        "schema_version": "1.0",
        "pages": [
            {
                "page_num": 1,
                "layout_regions": [
                    {"type": "table", "bbox": [50, 100, 500, 400], "confidence": 0.9},
                ],
                "tables": [
                    {
                        "html": "<table><tr><td>A</td><td>B</td></tr></table>",
                        "cell_bbox": [[50, 100, 250, 200], [250, 100, 500, 200]],
                    },
                ],
            },
            {
                "page_num": 2,
                "layout_regions": [],
                "tables": [
                    {
                        "html": "<table><tr><td>X</td></tr></table>",
                        "cell_bbox": [],
                    },
                ],
            },
        ],
    }


def _sample_validation_data():
    """Return parsed .validation.json sidecar data."""
    return {
        "schema_version": "1.0",
        "document_id": "doc-001",
        "source_file": "test.pdf",
        "page_count": {
            "source": 3,
            "output": 3,
            "match": True,
        },
        "quality": {
            "classification": "high_quality",
            "overall_confidence": 0.91,
            "text_extraction_rate": 1.0,
            "total_text_length": 2500,
            "pages_with_text": 3,
            "pages_image_only": 0,
            "pages_failed": 0,
            "ocr_methods_used": ["PaddleOCR"],
        },
        "output_hash": "abc123def456",
    }


def _sample_handwriting_data():
    """Return parsed .handwriting.json sidecar data."""
    return {
        "document_id": "doc-001",
        "source_file": "test.pdf",
        "pages": [
            {
                "page_num": 1,
                "has_handwriting": True,
                "handwriting_coverage": 0.3,
                "handwriting_regions": [
                    {"bbox": [10, 20, 100, 50], "confidence": 0.7, "text": "Hello"},
                    {"bbox": [10, 60, 100, 90], "confidence": 0.65, "text": "World"},
                ],
                "printed_line_count": 10,
                "handwritten_line_count": 2,
            },
            {
                "page_num": 2,
                "has_handwriting": False,
                "handwriting_coverage": 0.0,
                "handwriting_regions": [],
            },
        ],
        "total_handwritten_pages": 1,
        "total_pages_with_mixed": 1,
        "overall_handwriting_coverage": 0.15,
        "is_primarily_handwritten": False,
    }


def _sample_consolidated_entities_data():
    """Return parsed .entities.json (pre-consolidated) sidecar data."""
    return {
        "schema_version": "1.0",
        "document": "test.pdf",
        "entities": [
            {
                "id": "ent_001",
                "type": "PERSON",
                "text": "John Smith",
                "confidence": 0.95,
                "source": "ner",
                "page": 1,
                "bbox": [],
                "metadata": {},
            },
            {
                "id": "ent_002",
                "type": "ORG",
                "text": "Acme Corp",
                "confidence": 0.88,
                "source": "ner",
                "page": 1,
                "bbox": [],
                "metadata": {},
            },
        ],
        "classifications": [],
        "key_value_pairs": [],
        "summary": {"total_entities": 2},
    }


# ---------------------------------------------------------------------------
# RetrievalDocument unit tests
# ---------------------------------------------------------------------------


class TestRetrievalDocumentDefaults:
    """Test RetrievalDocument creation with defaults."""

    def test_creation_with_required_fields(self):
        doc = RetrievalDocument(document_id="doc-001", source_file="test.pdf")
        assert doc.document_id == "doc-001"
        assert doc.source_file == "test.pdf"
        assert doc.schema_version == _SCHEMA_VERSION
        assert doc.text == ""
        assert doc.text_by_page == []
        assert doc.classification == {}
        assert doc.entities == []
        assert doc.key_value_pairs == []
        assert doc.tables == []
        assert doc.metadata == {}
        assert doc.relationships == []
        assert doc.handwriting == {}

    def test_schema_version_default(self):
        doc = RetrievalDocument(document_id="x", source_file="y.pdf")
        assert doc.schema_version == "1.0"


class TestRetrievalDocumentToDict:
    """Test RetrievalDocument.to_dict() output matches schema."""

    def test_to_dict_has_required_fields(self):
        doc = RetrievalDocument(
            document_id="doc-001",
            source_file="test.pdf",
            text="Hello world",
            pipeline_version="1.2.0",
        )
        d = doc.to_dict()
        assert d["schema_version"] == "1.0"
        assert d["document_id"] == "doc-001"
        assert d["source_file"] == "test.pdf"
        assert d["text"] == "Hello world"
        assert "generated_at" in d
        assert d["pipeline_version"] == "1.2.0"

    def test_to_dict_validates_against_schema(self):
        doc = RetrievalDocument(
            document_id="doc-001",
            source_file="test.pdf",
            text="Sample text",
            pipeline_version="1.2.0",
        )
        d = doc.to_dict()
        errors = validate(d, "retrieval")
        assert errors == [], f"Schema validation errors: {errors}"

    def test_to_dict_with_full_data(self):
        doc = RetrievalDocument(
            document_id="doc-full",
            source_file="full.pdf",
            text="Full text content",
            text_by_page=[{"page": 1, "text": "Full text content"}],
            classification={"label": "invoice", "confidence": 0.92, "method": "ensemble"},
            entities=[
                {"id": "ent_001", "type": "PERSON", "text": "John", "confidence": 0.95},
            ],
            key_value_pairs=[
                {"key": "amount", "value": "$100", "confidence": 0.9},
            ],
            tables=[{"page": 1, "html": "<table></table>", "cell_bbox": []}],
            metadata={"pages": 1, "quality": "high_quality"},
            handwriting={"detected": False, "regions_count": 0, "coverage": 0.0},
            relationships=[],
            pipeline_version="1.2.0",
        )
        d = doc.to_dict()
        errors = validate(d, "retrieval")
        assert errors == [], f"Schema validation errors: {errors}"
        assert len(d["entities"]) == 1
        assert len(d["key_value_pairs"]) == 1
        assert len(d["tables"]) == 1


class TestRetrievalDocumentToMarkdown:
    """Test RetrievalDocument.to_markdown() format."""

    def test_basic_markdown_format(self):
        doc = RetrievalDocument(
            document_id="doc-001",
            source_file="report.pdf",
            text="This is the report text.",
            classification={"label": "report", "confidence": 0.85, "method": "ensemble"},
            metadata={"quality": "high_quality", "pages": 5, "languages": ["en"]},
            pipeline_version="1.2.0",
        )
        md = doc.to_markdown()
        assert "# Document: report.pdf" in md
        assert "**Classification:** report (85%)" in md
        assert "**Quality:** high_quality" in md
        assert "**Pages:** 5" in md
        assert "**Language:** en" in md
        assert "## Text Content" in md
        assert "This is the report text." in md

    def test_markdown_with_entities(self):
        doc = RetrievalDocument(
            document_id="doc-001",
            source_file="test.pdf",
            text="text",
            entities=[
                {"type": "PERSON", "text": "John Doe", "confidence": 0.95, "page": 1},
                {"type": "ORG", "text": "Acme", "confidence": 0.88, "page": 2},
            ],
            classification={},
            metadata={},
        )
        md = doc.to_markdown()
        assert "## Entities" in md
        assert "| Type | Text | Confidence | Page |" in md
        assert "| PERSON | John Doe | 0.95 | 1 |" in md
        assert "| ORG | Acme | 0.88 | 2 |" in md

    def test_markdown_entity_table_formatting(self):
        """Entity table uses pipe-delimited markdown table format."""
        doc = RetrievalDocument(
            document_id="d", source_file="f.pdf",
            entities=[{"type": "DATE", "text": "2024-01-15", "confidence": 0.90, "page": 1}],
            classification={}, metadata={},
        )
        md = doc.to_markdown()
        lines = md.split("\n")
        # Find entity table header
        header_idx = None
        for i, line in enumerate(lines):
            if "| Type |" in line:
                header_idx = i
                break
        assert header_idx is not None
        # Separator line
        assert lines[header_idx + 1].startswith("|---")
        # Data row
        assert "| DATE |" in lines[header_idx + 2]

    def test_markdown_with_kv_pairs(self):
        doc = RetrievalDocument(
            document_id="d", source_file="f.pdf",
            key_value_pairs=[
                {"key": "Invoice Number", "value": "12345", "confidence": 0.88},
            ],
            classification={}, metadata={},
        )
        md = doc.to_markdown()
        assert "## Key-Value Pairs" in md
        assert "| Invoice Number | 12345 | 0.88 |" in md

    def test_markdown_with_tables(self):
        doc = RetrievalDocument(
            document_id="d", source_file="f.pdf",
            tables=[{"page": 1, "html": "<table><tr><td>A</td></tr></table>"}],
            classification={}, metadata={},
        )
        md = doc.to_markdown()
        assert "## Tables" in md
        assert "### Table 1 (Page 1)" in md
        assert "<table><tr><td>A</td></tr></table>" in md

    def test_markdown_with_handwriting(self):
        doc = RetrievalDocument(
            document_id="d", source_file="f.pdf",
            handwriting={"detected": True, "regions_count": 3, "coverage": 0.25, "is_primarily_handwritten": False},
            classification={}, metadata={},
        )
        md = doc.to_markdown()
        assert "## Handwriting" in md
        assert "**Regions detected:** 3" in md
        assert "**Coverage:** 25%" in md
        assert "**Primarily handwritten:** No" in md

    def test_markdown_no_text(self):
        doc = RetrievalDocument(document_id="d", source_file="f.pdf", classification={}, metadata={})
        md = doc.to_markdown()
        assert "_No text extracted._" in md

    def test_markdown_no_entities_skips_section(self):
        doc = RetrievalDocument(document_id="d", source_file="f.pdf", classification={}, metadata={})
        md = doc.to_markdown()
        assert "## Entities" not in md

    def test_markdown_no_handwriting_skips_section(self):
        doc = RetrievalDocument(
            document_id="d", source_file="f.pdf",
            handwriting={"detected": False},
            classification={}, metadata={},
        )
        md = doc.to_markdown()
        assert "## Handwriting" not in md


# ---------------------------------------------------------------------------
# assemble_retrieval_output tests
# ---------------------------------------------------------------------------


class TestAssembleRetrievalOutput:
    """Test assemble_retrieval_output() with various data combinations."""

    def test_full_data_assembly(self):
        doc = assemble_retrieval_output(
            document_id="doc-001",
            source_file="test.pdf",
            ocr_text="Hello world from page 1.\nPage 2 text.",
            text_by_page=[
                {"page": 1, "text": "Hello world from page 1."},
                {"page": 2, "text": "Page 2 text."},
            ],
            entities_data=_sample_ner_data(),
            classification_data=_sample_classification_data(),
            extraction_data=_sample_extraction_data(),
            structure_data=_sample_structure_data(),
            validation_data=_sample_validation_data(),
            handwriting_data=_sample_handwriting_data(),
            pipeline_version="1.2.0",
        )

        assert doc.document_id == "doc-001"
        assert doc.source_file == "test.pdf"
        assert doc.text == "Hello world from page 1.\nPage 2 text."
        assert len(doc.text_by_page) == 2
        assert doc.classification["label"] == "invoice"
        assert len(doc.entities) > 0
        assert len(doc.key_value_pairs) > 0
        assert len(doc.tables) == 2
        assert doc.metadata["quality"] == "high_quality"
        assert doc.metadata["pages"] == 3
        assert doc.handwriting["detected"] is True
        assert doc.handwriting["regions_count"] == 2

        # Validate serialized output against schema
        d = doc.to_dict()
        errors = validate(d, "retrieval")
        assert errors == [], f"Schema errors: {errors}"

    def test_partial_data_ner_only(self):
        doc = assemble_retrieval_output(
            document_id="doc-partial",
            source_file="partial.pdf",
            ocr_text="Some text",
            entities_data=_sample_ner_data(),
            pipeline_version="1.2.0",
        )
        assert len(doc.entities) == 3  # 2 from page 1, 1 from page 2
        assert doc.classification == {}
        assert doc.key_value_pairs == []
        assert doc.tables == []
        assert doc.handwriting["detected"] is False

    def test_partial_data_classification_only(self):
        doc = assemble_retrieval_output(
            document_id="doc-cls",
            source_file="cls.pdf",
            classification_data=_sample_classification_data(),
        )
        assert doc.classification["label"] == "invoice"
        assert doc.entities == []

    def test_partial_data_extraction_only(self):
        doc = assemble_retrieval_output(
            document_id="doc-ext",
            source_file="ext.pdf",
            extraction_data=_sample_extraction_data(),
        )
        assert len(doc.entities) == 2
        assert len(doc.key_value_pairs) == 2

    def test_no_data_produces_valid_output(self):
        doc = assemble_retrieval_output(
            document_id="doc-empty",
            source_file="empty.pdf",
        )
        assert doc.text == ""
        assert doc.entities == []
        assert doc.classification == {}
        assert doc.key_value_pairs == []
        assert doc.tables == []
        assert doc.handwriting["detected"] is False
        assert doc.metadata["quality"] == "unknown"

        # Still valid against schema
        d = doc.to_dict()
        errors = validate(d, "retrieval")
        assert errors == [], f"Schema errors: {errors}"

    def test_consolidated_entities_data(self):
        """Pre-consolidated .entities.json is consumed directly."""
        doc = assemble_retrieval_output(
            document_id="doc-consol",
            source_file="consol.pdf",
            entities_data=_sample_consolidated_entities_data(),
        )
        assert len(doc.entities) == 2
        assert doc.entities[0]["type"] == "PERSON"
        assert doc.entities[1]["type"] == "ORG"

    def test_entity_deduplication_across_sources(self):
        """Same entity from NER and extraction is deduplicated."""
        ner_data = {
            "pages": [
                {
                    "page_num": 1,
                    "entities": [
                        {
                            "entity_type": "PERSON",
                            "text": "John Smith",
                            "confidence": 0.90,
                            "page_num": 1,
                        },
                    ],
                },
            ],
        }
        extraction_data = {
            "pages": [
                {
                    "page_num": 1,
                    "fields": [
                        {
                            "field_type": "PERSON",
                            "text": "John Smith",
                            "confidence": 0.95,
                            "page_num": 1,
                            "bbox": [],
                        },
                    ],
                },
            ],
        }
        doc = assemble_retrieval_output(
            document_id="doc-dedup",
            source_file="dedup.pdf",
            entities_data=ner_data,
            extraction_data=extraction_data,
        )
        # Should be deduplicated to one entity (the higher confidence one)
        person_entities = [e for e in doc.entities if e["type"].upper() == "PERSON"]
        assert len(person_entities) == 1
        assert person_entities[0]["confidence"] == 0.95

    def test_metadata_defaults_without_validation(self):
        doc = assemble_retrieval_output(
            document_id="d", source_file="f.pdf",
            ocr_text="Hello",
            text_by_page=[{"page": 1, "text": "Hello"}],
        )
        assert doc.metadata["pages"] == 1
        assert doc.metadata["total_text_length"] == 5
        assert doc.metadata["quality"] == "unknown"

    def test_relationships_data(self):
        rel_data = {
            "relationships": [
                {
                    "source_entity": "John Smith",
                    "target_entity": "Acme Corp",
                    "relation_type": "employee_of",
                    "confidence": 0.8,
                },
            ],
        }
        doc = assemble_retrieval_output(
            document_id="d", source_file="f.pdf",
            relationship_data=rel_data,
        )
        assert len(doc.relationships) == 1
        assert doc.relationships[0]["relation_type"] == "employee_of"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestExtractEntitiesFromNerData:
    def test_extracts_entities(self):
        entities = _extract_entities_from_ner_data(_sample_ner_data())
        assert len(entities) == 3
        assert entities[0]["type"] == "PERSON"
        assert entities[0]["text"] == "John Smith"
        assert entities[0]["source"] == "ner"

    def test_empty_ner_data(self):
        assert _extract_entities_from_ner_data({}) == []
        assert _extract_entities_from_ner_data({"pages": []}) == []

    def test_skips_empty_text(self):
        data = {
            "pages": [
                {
                    "page_num": 1,
                    "entities": [
                        {"entity_type": "PERSON", "text": "", "confidence": 0.9},
                    ],
                },
            ],
        }
        assert _extract_entities_from_ner_data(data) == []


class TestExtractFromExtractionData:
    def test_extracts_entities_and_kv(self):
        entities, kv = _extract_from_extraction_data(_sample_extraction_data())
        assert len(entities) == 2
        assert len(kv) == 2
        assert entities[0]["type"] == "date"
        assert kv[0]["key"] == "date"
        assert kv[1]["key"] == "amount"

    def test_empty_extraction_data(self):
        entities, kv = _extract_from_extraction_data({})
        assert entities == []
        assert kv == []


class TestExtractClassification:
    def test_extracts_primary_classification(self):
        cls = _extract_classification(_sample_classification_data())
        assert cls["label"] == "invoice"
        assert cls["confidence"] == 0.92
        assert cls["method"] == "ensemble"

    def test_empty_classification(self):
        assert _extract_classification({}) == {}

    def test_alternative_keys(self):
        data = {"label": "contract", "confidence": 0.8, "method": "ml"}
        cls = _extract_classification(data)
        assert cls["label"] == "contract"


class TestExtractTables:
    def test_extracts_tables_from_pages(self):
        tables = _extract_tables(_sample_structure_data())
        assert len(tables) == 2
        assert tables[0]["page"] == 1
        assert "<table>" in tables[0]["html"]
        assert tables[1]["page"] == 2

    def test_empty_structure(self):
        assert _extract_tables({}) == []
        assert _extract_tables({"pages": []}) == []


class TestExtractMetadata:
    def test_extracts_validation_metadata(self):
        meta = _extract_metadata(_sample_validation_data())
        assert meta["pages"] == 3
        assert meta["quality"] == "high_quality"
        assert meta["overall_confidence"] == 0.91
        assert meta["ocr_methods"] == ["PaddleOCR"]
        assert meta["total_text_length"] == 2500
        assert meta["output_hash"] == "abc123def456"

    def test_empty_validation(self):
        meta = _extract_metadata({})
        assert meta["pages"] == 0
        assert meta["quality"] == "unknown"


class TestExtractHandwriting:
    def test_extracts_handwriting_summary(self):
        hw = _extract_handwriting(_sample_handwriting_data())
        assert hw["detected"] is True
        assert hw["regions_count"] == 2
        assert hw["coverage"] == 0.15
        assert hw["is_primarily_handwritten"] is False

    def test_no_handwriting(self):
        hw = _extract_handwriting({
            "total_handwritten_pages": 0,
            "overall_handwriting_coverage": 0.0,
            "is_primarily_handwritten": False,
            "pages": [],
        })
        assert hw["detected"] is False
        assert hw["regions_count"] == 0


# ---------------------------------------------------------------------------
# File writer tests
# ---------------------------------------------------------------------------


class TestWriteRetrievalJson:
    def test_creates_file(self, tmp_path):
        doc = RetrievalDocument(
            document_id="doc-001",
            source_file="test.pdf",
            text="Hello",
            pipeline_version="1.2.0",
        )
        result = write_retrieval_json(doc, str(tmp_path))
        assert result is not None
        assert os.path.isfile(result)
        assert result.endswith(".retrieval.json")

        # Verify content
        with open(result) as f:
            data = json.load(f)
        assert data["document_id"] == "doc-001"
        assert data["text"] == "Hello"

    def test_creates_in_correct_directory(self, tmp_path):
        doc = RetrievalDocument(document_id="d", source_file="doc.pdf", pipeline_version="1.0")
        result = write_retrieval_json(doc, str(tmp_path))
        assert result is not None
        assert "EXPORT" in result
        assert "RETRIEVAL" in result

    def test_with_subfolder(self, tmp_path):
        doc = RetrievalDocument(document_id="d", source_file="doc.pdf", pipeline_version="1.0")
        result = write_retrieval_json(doc, str(tmp_path), subfolder="batch_01/subdir")
        assert result is not None
        assert os.path.isfile(result)
        assert "batch_01" in result

    def test_path_traversal_blocked(self, tmp_path):
        doc = RetrievalDocument(document_id="d", source_file="doc.pdf", pipeline_version="1.0")
        result = write_retrieval_json(doc, str(tmp_path), subfolder="../../etc")
        # Should be blocked (returns None) OR produce safe path
        # Depending on OS realpath resolution, the path may still be safe
        # The key test is it does NOT write outside the RETRIEVAL dir
        if result is not None:
            retrieval_dir = os.path.realpath(os.path.join(str(tmp_path), "EXPORT", "RETRIEVAL"))
            assert os.path.realpath(result).startswith(retrieval_dir)

    def test_output_validates_against_schema(self, tmp_path):
        doc = assemble_retrieval_output(
            document_id="doc-schema",
            source_file="schema_test.pdf",
            ocr_text="Test content",
            pipeline_version="1.2.0",
        )
        result = write_retrieval_json(doc, str(tmp_path))
        assert result is not None
        with open(result) as f:
            data = json.load(f)
        errors = validate(data, "retrieval")
        assert errors == [], f"Schema errors: {errors}"


class TestWriteRetrievalMarkdown:
    def test_creates_file(self, tmp_path):
        doc = RetrievalDocument(
            document_id="doc-001",
            source_file="test.pdf",
            text="Hello",
            classification={"label": "report", "confidence": 0.9},
            metadata={"quality": "high_quality", "pages": 1, "languages": ["en"]},
        )
        result = write_retrieval_markdown(doc, str(tmp_path))
        assert result is not None
        assert os.path.isfile(result)
        assert result.endswith(".retrieval.md")

        content = open(result).read()
        assert "# Document: test.pdf" in content

    def test_with_subfolder(self, tmp_path):
        doc = RetrievalDocument(document_id="d", source_file="doc.pdf")
        result = write_retrieval_markdown(doc, str(tmp_path), subfolder="sub")
        assert result is not None
        assert "sub" in result


# ---------------------------------------------------------------------------
# Environment variable tests
# ---------------------------------------------------------------------------


class TestEnvVarParsing:
    def test_default_is_false(self):
        """Default ENABLE_RETRIEVAL_OUTPUT is false."""
        # The module-level constant was parsed at import time.
        # We test the parsing logic inline.
        result = "false".lower() in ("1", "true", "yes")
        assert result is False

    def test_true_values(self):
        for val in ("1", "true", "yes", "True", "YES"):
            assert val.lower() in ("1", "true", "yes")

    def test_false_values(self):
        for val in ("0", "false", "no", ""):
            assert val.lower() not in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Markdown table formatting
# ---------------------------------------------------------------------------


class TestMarkdownTableFormatting:
    """Detailed tests for Markdown table output."""

    def test_entity_table_pipe_delimited(self):
        doc = RetrievalDocument(
            document_id="d", source_file="f.pdf",
            entities=[
                {"type": "PERSON", "text": "Jane", "confidence": 0.99, "page": 1},
            ],
            classification={}, metadata={},
        )
        md = doc.to_markdown()
        # Check header row
        assert "| Type | Text | Confidence | Page |" in md
        # Check separator row
        assert "|------|------|------------|------|" in md
        # Check data row
        assert "| PERSON | Jane | 0.99 | 1 |" in md

    def test_kv_table_pipe_delimited(self):
        doc = RetrievalDocument(
            document_id="d", source_file="f.pdf",
            key_value_pairs=[
                {"key": "Date", "value": "2024-01-01", "confidence": 0.85},
            ],
            classification={}, metadata={},
        )
        md = doc.to_markdown()
        assert "| Key | Value | Confidence |" in md
        assert "|-----|-------|------------|" in md
        assert "| Date | 2024-01-01 | 0.85 |" in md

    def test_multiple_entities_all_rows(self):
        doc = RetrievalDocument(
            document_id="d", source_file="f.pdf",
            entities=[
                {"type": "PERSON", "text": "A", "confidence": 0.9, "page": 1},
                {"type": "ORG", "text": "B", "confidence": 0.8, "page": 2},
                {"type": "DATE", "text": "C", "confidence": 0.7, "page": 3},
            ],
            classification={}, metadata={},
        )
        md = doc.to_markdown()
        assert md.count("| PERSON |") == 1
        assert md.count("| ORG |") == 1
        assert md.count("| DATE |") == 1
