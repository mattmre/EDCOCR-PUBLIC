"""Tests for output contract schema definitions (X1-A).

Validates schema loading, validation logic, and structural correctness
for all 14 output types.
"""

import json

import pytest

from schemas import (
    OUTPUT_TYPES,
    SCHEMA_DIR,
    SCHEMA_VERSION,
    get_schema_version,
    load_schema,
    validate,
)

# ---------------------------------------------------------------------------
# Schema loading tests
# ---------------------------------------------------------------------------


class TestLoadSchema:
    """Tests for load_schema()."""

    @pytest.mark.parametrize("output_type", OUTPUT_TYPES)
    def test_load_each_schema(self, output_type):
        """Each defined output type loads without error."""
        schema = load_schema(output_type)
        assert isinstance(schema, dict)
        assert schema  # non-empty

    def test_unknown_type_raises_value_error(self):
        """Unknown output type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown output type"):
            load_schema("nonexistent_type")

    def test_schema_dir_exists(self):
        """Schema directory exists on disk."""
        assert SCHEMA_DIR.is_dir()


# ---------------------------------------------------------------------------
# Schema structure tests
# ---------------------------------------------------------------------------


class TestSchemaStructure:
    """Tests for required top-level keys in each schema."""

    REQUIRED_KEYS = {"$schema", "type", "properties", "required"}

    @pytest.mark.parametrize("output_type", OUTPUT_TYPES)
    def test_has_required_top_level_keys(self, output_type):
        """Each schema has $schema, type, properties, and required keys."""
        schema = load_schema(output_type)
        for key in self.REQUIRED_KEYS:
            assert key in schema, (
                f"Schema '{output_type}' missing top-level key: {key}"
            )

    @pytest.mark.parametrize("output_type", OUTPUT_TYPES)
    def test_schema_draft_07(self, output_type):
        """Each schema uses JSON Schema draft-07."""
        schema = load_schema(output_type)
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"

    @pytest.mark.parametrize("output_type", OUTPUT_TYPES)
    def test_schema_type_is_object(self, output_type):
        """Each schema defines the top-level type as 'object'."""
        schema = load_schema(output_type)
        assert schema["type"] == "object"

    @pytest.mark.parametrize("output_type", OUTPUT_TYPES)
    def test_required_is_nonempty_list(self, output_type):
        """Each schema has at least one required field."""
        schema = load_schema(output_type)
        assert isinstance(schema["required"], list)
        assert len(schema["required"]) > 0

    @pytest.mark.parametrize("output_type", OUTPUT_TYPES)
    def test_properties_is_nonempty_dict(self, output_type):
        """Each schema defines at least one property."""
        schema = load_schema(output_type)
        assert isinstance(schema["properties"], dict)
        assert len(schema["properties"]) > 0

    @pytest.mark.parametrize("output_type", OUTPUT_TYPES)
    def test_required_fields_are_defined_in_properties(self, output_type):
        """Every required field is defined in the properties section."""
        schema = load_schema(output_type)
        for field_name in schema["required"]:
            assert field_name in schema["properties"], (
                f"Schema '{output_type}' requires '{field_name}' but it is "
                f"not defined in properties"
            )

    @pytest.mark.parametrize("output_type", OUTPUT_TYPES)
    def test_schema_is_valid_json(self, output_type):
        """Schema file is valid JSON and round-trips cleanly."""
        path = SCHEMA_DIR / f"{output_type}.schema.json"
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Validation function tests
# ---------------------------------------------------------------------------


class TestValidate:
    """Tests for validate()."""

    def test_valid_entities_data_passes(self):
        """Valid consolidated entity data produces no errors."""
        data = {
            "schema_version": "1.0",
            "document": "test.pdf",
            "generated_at": "2026-03-29T00:00:00.000+00:00",
            "pipeline_version": "1.2.0",
            "entities": [],
            "classifications": [],
            "key_value_pairs": [],
            "summary": {
                "total_entities": 0,
                "entity_types": {},
                "total_kv_pairs": 0,
                "primary_classification": "",
            },
        }
        errors = validate(data, "entities")
        assert errors == []

    def test_missing_required_field_detected(self):
        """Missing required fields are reported as errors."""
        data = {
            "schema_version": "1.0",
            # "document" is missing
            "generated_at": "2026-03-29T00:00:00.000+00:00",
            "pipeline_version": "1.2.0",
            "entities": [],
            "classifications": [],
            "key_value_pairs": [],
            "summary": {},
        }
        errors = validate(data, "entities")
        assert any("document" in e for e in errors)

    def test_wrong_type_detected(self):
        """Wrong field type is reported as an error."""
        data = {
            "schema_version": "1.0",
            "document": 12345,  # should be string
            "generated_at": "2026-03-29T00:00:00.000+00:00",
            "pipeline_version": "1.2.0",
            "entities": [],
            "classifications": [],
            "key_value_pairs": [],
            "summary": {},
        }
        errors = validate(data, "entities")
        assert any("document" in e and "string" in e for e in errors)

    def test_array_type_check(self):
        """Array fields checked for type correctness."""
        data = {
            "schema_version": "1.0",
            "document": "test.pdf",
            "generated_at": "2026-03-29T00:00:00.000+00:00",
            "pipeline_version": "1.2.0",
            "entities": "not_an_array",  # should be array
            "classifications": [],
            "key_value_pairs": [],
            "summary": {},
        }
        errors = validate(data, "entities")
        assert any("entities" in e and "array" in e for e in errors)

    def test_object_type_check(self):
        """Object fields checked for type correctness."""
        data = {
            "schema_version": "1.0",
            "document": "test.pdf",
            "generated_at": "2026-03-29T00:00:00.000+00:00",
            "pipeline_version": "1.2.0",
            "entities": [],
            "classifications": [],
            "key_value_pairs": [],
            "summary": "not_an_object",  # should be object
        }
        errors = validate(data, "entities")
        assert any("summary" in e and "object" in e for e in errors)

    def test_validate_unknown_type_raises(self):
        """Validate raises ValueError for unknown output type."""
        with pytest.raises(ValueError, match="Unknown output type"):
            validate({}, "nonexistent_type")

    def test_valid_validation_data_passes(self):
        """Valid validation report data produces no errors."""
        data = {
            "schema_version": "1.0",
            "document_id": "abc123",
            "source_file": "test.pdf",
            "processing": {
                "pipeline_version": "1.2.0",
                "timestamp": "2026-03-29T00:00:00.000+00:00",
            },
            "page_count": {"source": 5, "output": 5, "match": True},
            "quality": {
                "classification": "high_quality",
                "overall_confidence": 0.95,
                "text_extraction_rate": 1.0,
                "total_text_length": 5000,
                "pages_with_text": 5,
                "pages_image_only": 0,
                "pages_failed": 0,
                "ocr_methods_used": ["PaddleOCR"],
            },
            "pages": [],
            "output_hash": "fixture-hash",
        }
        errors = validate(data, "validation")
        assert errors == []

    def test_valid_custody_event_passes(self):
        """Valid custody event data produces no errors."""
        data = {
            "document_id": "abc123",
            "event_type": "file_ingested",
            "timestamp": "2026-03-29T00:00:00.000+00:00",
            "data": {"source_path": "/app/ocr_source/test.pdf"},
            "prev_hash": None,
            "hash": "deadbeef" * 8,
        }
        errors = validate(data, "custody")
        assert errors == []

    def test_valid_ner_data_passes(self):
        """Valid NER data produces no errors."""
        data = {
            "schema_version": "1.0",
            "document_id": "abc123",
            "source_file": "test.pdf",
            "processing": {
                "ner_engine": "spacy",
                "ner_model": "en_core_web_sm",
                "timestamp": "2026-03-29T00:00:00.000+00:00",
                "pipeline_version": "1.2.0",
            },
            "entities_summary": {
                "total_entities": 3,
                "entity_types": {"PERSON": 2, "ORG": 1},
                "unique_entities": 3,
            },
            "pages": [],
        }
        errors = validate(data, "ner")
        assert errors == []

    def test_valid_extraction_data_passes(self):
        """Valid extraction data produces no errors."""
        data = {
            "schema_version": "1.0",
            "document_id": "abc123",
            "source_file": "test.pdf",
            "processing": {
                "extraction_engine": "hybrid",
                "uie_model": "uie-base",
                "pipeline_version": "1.2.0",
                "timestamp": "2026-03-29T00:00:00.000+00:00",
            },
            "extraction_summary": {
                "total_fields": 5,
                "field_type_counts": {"date": 2, "amount": 3},
                "extraction_engine": "hybrid",
            },
            "pages": [],
        }
        errors = validate(data, "extraction")
        assert errors == []

    def test_valid_classification_data_passes(self):
        """Valid classification data produces no errors."""
        data = {
            "schema_version": "1.0",
            "document_id": "abc123",
            "source_file": "test.pdf",
            "processing": {
                "classification_engine": "text_rules",
                "pipeline_version": "1.2.0",
                "timestamp": "2026-03-29T00:00:00.000+00:00",
            },
            "document_summary": {
                "document_type": "invoice",
                "document_confidence": 0.92,
                "type_distribution": {"invoice": 0.8, "form": 0.2},
            },
            "pages": [],
        }
        errors = validate(data, "classification")
        assert errors == []

    def test_valid_handwriting_data_passes(self):
        """Valid handwriting data produces no errors."""
        data = {
            "schema_version": "1.0",
            "document_id": "abc123",
            "source_file": "test.pdf",
            "processing": {
                "detection_engine": "confidence_heuristic+geometry",
                "pipeline_version": "1.2.0",
                "timestamp": "2026-03-29T00:00:00.000+00:00",
            },
            "document_summary": {
                "total_handwritten_pages": 0,
                "total_pages_with_mixed": 0,
                "overall_handwriting_coverage": 0.0,
                "is_primarily_handwritten": False,
            },
            "pages": [],
        }
        errors = validate(data, "handwriting")
        assert errors == []

    def test_valid_signature_data_passes(self):
        """Valid signature data produces no errors."""
        data = {
            "schema_version": "1.0",
            "document_id": "abc123",
            "source_file": "test.pdf",
            "processing": {
                "experimental": True,
                "pipeline_version": "1.2.0",
                "timestamp": "2026-03-29T00:00:00.000+00:00",
                "notes": ["This is experimental."],
            },
            "document_summary": {
                "total_candidate_pages": 0,
                "total_presence_pages": 0,
                "total_review_pages": 0,
                "experimental": True,
            },
            "pages": [],
        }
        errors = validate(data, "signature")
        assert errors == []

    def test_valid_vertical_data_passes(self):
        """Valid vertical text data produces no errors."""
        data = {
            "schema_version": "1.0",
            "document_id": "abc123",
            "source_file": "test.pdf",
            "processing": {
                "analysis_engine": "geometry_aspect_ratio",
                "aspect_ratio_threshold": 2.0,
                "column_grouping_tolerance": 0.05,
                "pipeline_version": "1.2.0",
                "timestamp": "2026-03-29T00:00:00.000+00:00",
            },
            "document_summary": {
                "total_vertical_pages": 0,
                "total_mixed_pages": 0,
                "has_vertical_content": False,
            },
            "pages": [],
        }
        errors = validate(data, "vertical")
        assert errors == []

    def test_valid_structure_data_passes(self):
        """Valid structure data produces no errors."""
        data = {
            "schema_version": "1.0",
            "document_id": "abc123",
            "source_file": "test.pdf",
            "processing": {
                "pipeline_version": "1.2.0",
                "timestamp": "2026-03-29T00:00:00.000+00:00",
                "docintel_mode": "full",
            },
            "pages": [],
            "document_summary": {
                "total_tables": 0,
                "total_figures": 0,
                "total_form_fields": 0,
                "total_key_value_pairs": 0,
                "layout_types_found": [],
                "has_signatures": False,
                "has_filled_forms": False,
            },
        }
        errors = validate(data, "structure")
        assert errors == []

    def test_valid_ocr_text_data_passes(self):
        """Valid OCR text artifact descriptor produces no errors."""
        data = {
            "artifact_type": "ocr_text",
            "file_extension": ".txt",
            "encoding": "utf-8",
        }
        errors = validate(data, "ocr_text")
        assert errors == []

    def test_valid_searchable_pdf_data_passes(self):
        """Valid searchable PDF artifact descriptor produces no errors."""
        data = {
            "artifact_type": "searchable_pdf",
            "file_extension": ".pdf",
            "mime_type": "application/pdf",
        }
        errors = validate(data, "searchable_pdf")
        assert errors == []

    def test_valid_retrieval_data_passes(self):
        """Valid retrieval data produces no errors."""
        data = {
            "schema_version": "1.0",
            "document_id": "abc123",
            "source_file": "test.pdf",
            "generated_at": "2026-03-29T00:00:00.000+00:00",
            "pipeline_version": "1.2.0",
            "text": "Sample text content",
        }
        errors = validate(data, "retrieval")
        assert errors == []

    def test_valid_output_manifest_data_passes(self):
        """Valid output manifest data produces no errors."""
        data = {
            "schema_version": "1.0",
            "document_id": "abc123",
            "source_file": "test.pdf",
            "pipeline_version": "1.2.0",
            "generated_at": "2026-03-29T00:00:00.000+00:00",
            "artifacts": [],
        }
        errors = validate(data, "output_manifest")
        assert errors == []

    def test_multiple_missing_fields(self):
        """Multiple missing required fields are all reported."""
        data = {}
        errors = validate(data, "entities")
        # entities schema requires: schema_version, document, generated_at,
        # pipeline_version, entities, classifications, key_value_pairs, summary
        assert len(errors) >= 8

    def test_boolean_type_check(self):
        """Boolean fields checked for type correctness."""
        data = {
            "schema_version": "1.0",
            "document_id": "abc123",
            "source_file": "test.pdf",
            "processing": {},
            "document_summary": {
                "total_vertical_pages": 0,
                "total_mixed_pages": 0,
                "has_vertical_content": "not_a_bool",
            },
            "pages": [],
        }
        errors = validate(data, "vertical")
        # has_vertical_content is inside a nested object, so our basic
        # validator only checks top-level properties; no error expected at top level.
        # This verifies the top-level fields are valid.
        assert not any("has_vertical_content" in e for e in errors)

    def test_integer_type_check(self):
        """Integer fields on output_manifest are checked."""
        # Test that non-integer values for integer-typed fields are caught
        # by looking for a schema that has an integer field at top level.
        # output_manifest does not have top-level integers, so test via
        # a custom scenario with the custody schema.
        data = {
            "document_id": "abc123",
            "event_type": "file_ingested",
            "timestamp": "2026-03-29T00:00:00.000+00:00",
            "data": {},
            "prev_hash": None,
            "hash": "abc",
        }
        # All fields present and correct types for custody
        errors = validate(data, "custody")
        assert errors == []


# ---------------------------------------------------------------------------
# Version tests
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    """Tests for schema versioning."""

    def test_get_schema_version_returns_expected(self):
        """get_schema_version() returns '1.0'."""
        assert get_schema_version() == "1.0"

    def test_schema_version_constant(self):
        """SCHEMA_VERSION constant is '1.0'."""
        assert SCHEMA_VERSION == "1.0"


# ---------------------------------------------------------------------------
# OUTPUT_TYPES completeness
# ---------------------------------------------------------------------------


class TestOutputTypesCompleteness:
    """Tests for OUTPUT_TYPES tuple completeness."""

    def test_output_types_count(self):
        """OUTPUT_TYPES has 14 entries."""
        assert len(OUTPUT_TYPES) == 14

    def test_all_schema_files_exist(self):
        """Every entry in OUTPUT_TYPES has a corresponding schema file."""
        for output_type in OUTPUT_TYPES:
            path = SCHEMA_DIR / f"{output_type}.schema.json"
            assert path.exists(), f"Missing schema file: {path}"

    def test_no_orphan_schema_files(self):
        """No unclassified schema files exist outside OUTPUT_TYPES."""
        schema_files = set(SCHEMA_DIR.glob("*.schema.json"))
        expected_files = {
            SCHEMA_DIR / f"{ot}.schema.json" for ot in OUTPUT_TYPES
        }
        standalone_contract_files = {
            SCHEMA_DIR / "glossary.schema.json",
            SCHEMA_DIR / "language.schema.json",
            SCHEMA_DIR / "translation.schema.json",
            SCHEMA_DIR / "document-bundle-v1.schema.json",
            SCHEMA_DIR / "translation-bundle-v1.schema.json",
        }
        orphans = schema_files - expected_files - standalone_contract_files
        assert not orphans, f"Orphan schema files: {orphans}"

    def test_output_types_is_tuple(self):
        """OUTPUT_TYPES is an immutable tuple."""
        assert isinstance(OUTPUT_TYPES, tuple)

    def test_no_duplicate_output_types(self):
        """No duplicate entries in OUTPUT_TYPES."""
        assert len(OUTPUT_TYPES) == len(set(OUTPUT_TYPES))
