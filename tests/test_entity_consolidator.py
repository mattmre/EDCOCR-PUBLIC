"""Tests for unified entity consolidation (entity_consolidator.py)."""

import json
import os

import pytest

from entity_consolidator import (
    _classifications_from_result,
    _entities_from_extraction,
    _entities_from_ner,
    consolidate_entities,
    merge_duplicate_entities,
    write_consolidated_entities_json,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeDocumentNER:
    """Minimal stand-in for ner.DocumentNER."""

    def __init__(self, pages=None):
        self.document_id = "doc-ner-001"
        self.source_file = "test_doc.pdf"
        self.pages = pages or []
        self.total_entities = 0
        self.entity_type_counts = {}
        self.unique_entities = []


class _FakeDocumentExtraction:
    """Minimal stand-in for extraction.DocumentExtraction."""

    def __init__(self, pages=None):
        self.document_id = "doc-ext-001"
        self.source_file = "test_doc.pdf"
        self.pages = pages or []
        self.total_fields = 0
        self.field_type_counts = {}
        self.extraction_engine = "regex"


class _FakeDocumentClassification:
    """Minimal stand-in for classification.DocumentClassification."""

    def __init__(self):
        self.document_id = "doc-cls-001"
        self.source_file = "test_doc.pdf"
        self.document_type = "invoice"
        self.document_confidence = 0.92
        self.document_labels = [
            {"label": "invoice", "confidence": 0.92, "source": "classification"},
            {"label": "receipt", "confidence": 0.45, "source": "classification"},
        ]
        self.pages = []
        self.type_distribution = {}
        self.document_type_scores = {}
        self.custom_profile_matches = []


def _make_ner_doc():
    """Create a DocumentNER with sample entities."""
    doc = _FakeDocumentNER(pages=[
        {
            "page_num": 1,
            "entities": [
                {
                    "type": "PERSON",
                    "text": "John Smith",
                    "confidence": 0.95,
                    "page_num": 1,
                    "start": 0,
                    "end": 10,
                },
                {
                    "type": "ORG",
                    "text": "Acme Corp",
                    "confidence": 0.90,
                    "page_num": 1,
                    "start": 15,
                    "end": 24,
                },
                {
                    "type": "DATE",
                    "text": "January 15, 2024",
                    "confidence": 1.0,
                    "page_num": 1,
                    "start": 30,
                    "end": 46,
                },
            ],
        },
        {
            "page_num": 2,
            "entities": [
                {
                    "type": "MONEY",
                    "text": "$5,000.00",
                    "confidence": 1.0,
                    "page_num": 2,
                    "start": 0,
                    "end": 9,
                },
            ],
        },
    ])
    return doc


def _make_extraction_doc():
    """Create a DocumentExtraction with sample fields."""
    doc = _FakeDocumentExtraction(pages=[
        {
            "page_num": 1,
            "fields": [
                {
                    "field_type": "date",
                    "text": "2024-01-15",
                    "confidence": 0.97,
                    "page_num": 1,
                    "start": 30,
                    "end": 40,
                    "extraction_method": "regex",
                    "normalized_value": "2024-01-15",
                },
                {
                    "field_type": "reference_number",
                    "text": "INV-2024-001",
                    "confidence": 0.98,
                    "page_num": 1,
                    "start": 50,
                    "end": 62,
                    "extraction_method": "uie",
                    "normalized_value": "",
                },
                {
                    "field_type": "amount",
                    "text": "$5,000.00",
                    "confidence": 0.96,
                    "page_num": 1,
                    "start": 70,
                    "end": 79,
                    "extraction_method": "regex",
                    "normalized_value": "",
                },
                {
                    "field_type": "person_name",
                    "text": "John Smith",
                    "confidence": 0.93,
                    "page_num": 1,
                    "start": 0,
                    "end": 10,
                    "extraction_method": "uie",
                    "normalized_value": "",
                },
            ],
        },
    ])
    return doc


def _make_classification_doc():
    """Create a DocumentClassification with sample results."""
    return _FakeDocumentClassification()


# ---------------------------------------------------------------------------
# Tests: NER adapter
# ---------------------------------------------------------------------------


class TestEntitiesFromNER:
    """Test _entities_from_ner adapter."""

    def test_extracts_entities_from_ner_pages(self):
        doc = _make_ner_doc()
        entities = _entities_from_ner(doc)

        assert len(entities) == 4
        assert all(e["source"] == "ner" for e in entities)

    def test_entity_ids_are_sequential(self):
        doc = _make_ner_doc()
        entities = _entities_from_ner(doc)

        ids = [e["id"] for e in entities]
        assert ids == ["ner_001", "ner_002", "ner_003", "ner_004"]

    def test_entity_types_preserved(self):
        doc = _make_ner_doc()
        entities = _entities_from_ner(doc)

        types = [e["type"] for e in entities]
        assert types == ["PERSON", "ORG", "DATE", "MONEY"]

    def test_page_numbers_preserved(self):
        doc = _make_ner_doc()
        entities = _entities_from_ner(doc)

        pages = [e["page"] for e in entities]
        assert pages == [1, 1, 1, 2]

    def test_empty_pages_produces_empty_list(self):
        doc = _FakeDocumentNER(pages=[])
        entities = _entities_from_ner(doc)
        assert entities == []

    def test_skips_empty_text_entities(self):
        doc = _FakeDocumentNER(pages=[
            {
                "page_num": 1,
                "entities": [
                    {"type": "PERSON", "text": "", "confidence": 0.9, "page_num": 1, "start": 0, "end": 0},
                    {"type": "ORG", "text": "Valid Org", "confidence": 0.8, "page_num": 1, "start": 5, "end": 14},
                ],
            },
        ])
        entities = _entities_from_ner(doc)
        assert len(entities) == 1
        assert entities[0]["text"] == "Valid Org"

    def test_handles_entity_type_from_entity_type_key(self):
        """Entities may use 'entity_type' instead of 'type'."""
        doc = _FakeDocumentNER(pages=[
            {
                "page_num": 1,
                "entities": [
                    {"entity_type": "CASE_NUMBER", "text": "24-CV-1234", "confidence": 1.0, "page_num": 1, "start": 0, "end": 10},
                ],
            },
        ])
        entities = _entities_from_ner(doc)
        assert len(entities) == 1
        assert entities[0]["type"] == "CASE_NUMBER"


# ---------------------------------------------------------------------------
# Tests: Extraction adapter
# ---------------------------------------------------------------------------


class TestEntitiesFromExtraction:
    """Test _entities_from_extraction adapter."""

    def test_extracts_entities_and_kv_pairs(self):
        doc = _make_extraction_doc()
        entities, kv_pairs = _entities_from_extraction(doc)

        assert len(entities) == 4
        assert len(kv_pairs) == 4
        assert all(e["source"] == "extraction" for e in entities)
        assert all(kv["source"] == "extraction" for kv in kv_pairs)

    def test_extraction_entity_ids_are_sequential(self):
        doc = _make_extraction_doc()
        entities, _ = _entities_from_extraction(doc)

        ids = [e["id"] for e in entities]
        assert ids == ["ext_001", "ext_002", "ext_003", "ext_004"]

    def test_kv_pairs_use_field_type_as_key(self):
        doc = _make_extraction_doc()
        _, kv_pairs = _entities_from_extraction(doc)

        keys = [kv["key"] for kv in kv_pairs]
        assert "date" in keys
        assert "reference_number" in keys

    def test_normalized_value_in_metadata(self):
        doc = _make_extraction_doc()
        entities, _ = _entities_from_extraction(doc)

        date_entity = next(e for e in entities if e["type"] == "date")
        assert date_entity["metadata"]["normalized_value"] == "2024-01-15"

    def test_empty_extraction_produces_empty(self):
        doc = _FakeDocumentExtraction(pages=[])
        entities, kv_pairs = _entities_from_extraction(doc)
        assert entities == []
        assert kv_pairs == []

    def test_bbox_preserved_when_present(self):
        doc = _FakeDocumentExtraction(pages=[
            {
                "page_num": 1,
                "fields": [
                    {
                        "field_type": "amount",
                        "text": "$100",
                        "confidence": 0.9,
                        "page_num": 1,
                        "start": 0,
                        "end": 4,
                        "extraction_method": "semantic",
                        "normalized_value": "",
                        "bbox": [10, 20, 100, 40],
                    },
                ],
            },
        ])
        entities, _ = _entities_from_extraction(doc)
        assert entities[0]["bbox"] == [10, 20, 100, 40]


# ---------------------------------------------------------------------------
# Tests: Classification adapter
# ---------------------------------------------------------------------------


class TestClassificationsFromResult:
    """Test _classifications_from_result adapter."""

    def test_extracts_primary_classification(self):
        doc = _make_classification_doc()
        classifications = _classifications_from_result(doc)

        assert len(classifications) >= 1
        assert classifications[0]["label"] == "invoice"
        assert classifications[0]["confidence"] == 0.92

    def test_includes_additional_labels(self):
        doc = _make_classification_doc()
        classifications = _classifications_from_result(doc)

        labels = [c["label"] for c in classifications]
        assert "receipt" in labels

    def test_empty_classification_returns_empty(self):
        doc = _FakeDocumentClassification()
        doc.document_type = "other"
        doc.document_confidence = 0.0
        doc.document_labels = []
        classifications = _classifications_from_result(doc)
        assert classifications == []

    def test_deduplicates_primary_and_labels(self):
        doc = _FakeDocumentClassification()
        # Primary is "invoice" and document_labels also contains "invoice"
        classifications = _classifications_from_result(doc)
        invoice_count = sum(1 for c in classifications if c["label"] == "invoice")
        assert invoice_count == 1


# ---------------------------------------------------------------------------
# Tests: Deduplication
# ---------------------------------------------------------------------------


class TestMergeDuplicateEntities:
    """Test merge_duplicate_entities."""

    def test_deduplicates_same_type_and_text(self):
        entities = [
            {"id": "ner_001", "type": "PERSON", "text": "John Smith", "confidence": 0.95, "source": "ner", "page": 1, "bbox": [], "metadata": {}},
            {"id": "ext_001", "type": "person_name", "text": "John Smith", "confidence": 0.93, "source": "extraction", "page": 1, "bbox": [], "metadata": {}},
        ]
        result = merge_duplicate_entities(entities)
        # Both normalize to ("person", "john smith") -- different type casing
        # Since "PERSON" != "person_name" when lowered, these are separate
        # types and should NOT be merged.
        assert len(result) == 2

    def test_deduplicates_exact_type_match(self):
        entities = [
            {"id": "ner_001", "type": "DATE", "text": "January 15, 2024", "confidence": 1.0, "source": "ner", "page": 1, "bbox": [], "metadata": {}},
            {"id": "ext_001", "type": "DATE", "text": "january 15, 2024", "confidence": 0.9, "source": "extraction", "page": 1, "bbox": [], "metadata": {}},
        ]
        result = merge_duplicate_entities(entities)
        assert len(result) == 1
        # Higher confidence wins
        assert result[0]["confidence"] == 1.0

    def test_keeps_different_entities(self):
        entities = [
            {"id": "ner_001", "type": "PERSON", "text": "Jane Doe", "confidence": 0.95, "source": "ner", "page": 1, "bbox": [], "metadata": {}},
            {"id": "ner_002", "type": "ORG", "text": "Acme Corp", "confidence": 0.90, "source": "ner", "page": 1, "bbox": [], "metadata": {}},
        ]
        result = merge_duplicate_entities(entities)
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        result = merge_duplicate_entities([])
        assert result == []

    def test_renumbers_ids_after_dedup(self):
        entities = [
            {"id": "ner_001", "type": "PERSON", "text": "Alice", "confidence": 0.9, "source": "ner", "page": 1, "bbox": [], "metadata": {}},
            {"id": "ner_002", "type": "PERSON", "text": "Alice", "confidence": 0.8, "source": "ner", "page": 2, "bbox": [], "metadata": {}},
            {"id": "ner_003", "type": "ORG", "text": "Beta Inc", "confidence": 0.7, "source": "ner", "page": 1, "bbox": [], "metadata": {}},
        ]
        result = merge_duplicate_entities(entities)
        assert len(result) == 2
        ids = [e["id"] for e in result]
        assert ids == ["ent_001", "ent_002"]

    def test_extraction_wins_tie_on_confidence(self):
        """When confidence is equal, extraction source takes priority."""
        entities = [
            {"id": "ner_001", "type": "date", "text": "2024-01-15", "confidence": 0.95, "source": "ner", "page": 1, "bbox": [], "metadata": {}},
            {"id": "ext_001", "type": "date", "text": "2024-01-15", "confidence": 0.95, "source": "extraction", "page": 1, "bbox": [], "metadata": {}},
        ]
        result = merge_duplicate_entities(entities)
        assert len(result) == 1
        assert result[0]["source"] == "extraction"


# ---------------------------------------------------------------------------
# Tests: Full consolidation
# ---------------------------------------------------------------------------


class TestConsolidateEntities:
    """Test consolidate_entities full pipeline."""

    def test_consolidation_from_all_sources(self):
        ner = _make_ner_doc()
        ext = _make_extraction_doc()
        cls = _make_classification_doc()

        result = consolidate_entities(
            ner_results=ner,
            extraction_results=ext,
            classification_results=cls,
            doc_name="test_doc.pdf",
            pipeline_version="0.9.0-test",
        )

        assert result["schema_version"] == "1.0"
        assert result["document"] == "test_doc.pdf"
        assert result["pipeline_version"] == "0.9.0-test"
        assert "generated_at" in result
        assert len(result["entities"]) > 0
        assert len(result["classifications"]) >= 1
        assert len(result["key_value_pairs"]) > 0
        assert result["summary"]["total_entities"] == len(result["entities"])
        assert result["summary"]["primary_classification"] == "invoice"

    def test_consolidation_ner_only(self):
        ner = _make_ner_doc()
        result = consolidate_entities(ner_results=ner, doc_name="ner_only.pdf")

        assert len(result["entities"]) == 4
        assert result["classifications"] == []
        assert result["key_value_pairs"] == []
        assert result["summary"]["total_entities"] == 4

    def test_consolidation_extraction_only(self):
        ext = _make_extraction_doc()
        result = consolidate_entities(extraction_results=ext, doc_name="ext_only.pdf")

        assert len(result["entities"]) == 4
        assert len(result["key_value_pairs"]) == 4
        assert result["classifications"] == []

    def test_consolidation_classification_only(self):
        cls = _make_classification_doc()
        result = consolidate_entities(classification_results=cls, doc_name="cls_only.pdf")

        assert result["entities"] == []
        assert len(result["classifications"]) >= 1
        assert result["summary"]["primary_classification"] == "invoice"

    def test_consolidation_all_none(self):
        result = consolidate_entities(doc_name="empty.pdf")

        assert result["entities"] == []
        assert result["classifications"] == []
        assert result["key_value_pairs"] == []
        assert result["summary"]["total_entities"] == 0
        assert result["summary"]["total_kv_pairs"] == 0

    def test_entity_types_in_summary(self):
        ner = _make_ner_doc()
        result = consolidate_entities(ner_results=ner, doc_name="types.pdf")

        types = result["summary"]["entity_types"]
        assert "PERSON" in types
        assert "ORG" in types
        assert "DATE" in types
        assert "MONEY" in types


# ---------------------------------------------------------------------------
# Tests: JSON output
# ---------------------------------------------------------------------------


class TestWriteConsolidatedEntitiesJson:
    """Test write_consolidated_entities_json."""

    def test_creates_json_file(self, tmp_path):
        result = consolidate_entities(
            ner_results=_make_ner_doc(),
            doc_name="test_doc.pdf",
            pipeline_version="0.9.0-test",
        )

        json_path = write_consolidated_entities_json(
            result,
            str(tmp_path),
            "",
            "test_doc.pdf",
        )

        assert json_path is not None
        assert os.path.exists(json_path)
        assert json_path.endswith(".entities.json")

    def test_json_content_is_valid(self, tmp_path):
        result = consolidate_entities(
            ner_results=_make_ner_doc(),
            extraction_results=_make_extraction_doc(),
            classification_results=_make_classification_doc(),
            doc_name="test_doc.pdf",
            pipeline_version="0.9.0-test",
        )

        json_path = write_consolidated_entities_json(
            result,
            str(tmp_path),
            "batch_001",
            "test_doc.pdf",
        )

        assert json_path is not None
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0"
        assert data["document"] == "test_doc.pdf"
        assert isinstance(data["entities"], list)
        assert isinstance(data["classifications"], list)
        assert isinstance(data["key_value_pairs"], list)
        assert isinstance(data["summary"], dict)
        assert "total_entities" in data["summary"]
        assert "entity_types" in data["summary"]
        assert "total_kv_pairs" in data["summary"]
        assert "primary_classification" in data["summary"]

    def test_subfolder_creates_nested_path(self, tmp_path):
        result = consolidate_entities(doc_name="nested.pdf")

        json_path = write_consolidated_entities_json(
            result,
            str(tmp_path),
            "sub1/sub2",
            "nested.pdf",
        )

        assert json_path is not None
        assert os.path.exists(json_path)
        assert "sub1" in json_path or "sub2" in json_path

    def test_path_traversal_blocked(self, tmp_path):
        result = consolidate_entities(doc_name="traversal.pdf")

        json_path = write_consolidated_entities_json(
            result,
            str(tmp_path),
            "../../etc",
            "traversal.pdf",
        )

        # Should still write but inside the entities dir
        assert json_path is not None
        entities_root = os.path.join(str(tmp_path), "EXPORT", "ENTITIES")
        resolved = os.path.realpath(json_path)
        assert resolved.startswith(os.path.realpath(entities_root))

    def test_empty_consolidated_still_writes(self, tmp_path):
        result = consolidate_entities(doc_name="empty.pdf")

        json_path = write_consolidated_entities_json(
            result,
            str(tmp_path),
            "",
            "empty.pdf",
        )

        assert json_path is not None
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["summary"]["total_entities"] == 0

    def test_non_pdf_filename_handled(self, tmp_path):
        result = consolidate_entities(doc_name="photo.jpg")

        json_path = write_consolidated_entities_json(
            result,
            str(tmp_path),
            "",
            "photo.jpg",
        )

        assert json_path is not None
        assert os.path.basename(json_path) == "photo__jpg.entities.json"


# ---------------------------------------------------------------------------
# Tests: Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Validate the consolidated output conforms to the expected schema."""

    def test_all_required_top_level_keys_present(self):
        result = consolidate_entities(
            ner_results=_make_ner_doc(),
            extraction_results=_make_extraction_doc(),
            classification_results=_make_classification_doc(),
            doc_name="schema.pdf",
            pipeline_version="0.9.0",
        )

        required_keys = {
            "schema_version", "document", "generated_at", "pipeline_version",
            "entities", "classifications", "key_value_pairs", "summary",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_entity_has_required_fields(self):
        result = consolidate_entities(
            ner_results=_make_ner_doc(),
            doc_name="entity_fields.pdf",
        )

        for ent in result["entities"]:
            assert "id" in ent
            assert "type" in ent
            assert "text" in ent
            assert "confidence" in ent
            assert "source" in ent
            assert "page" in ent
            assert "bbox" in ent
            assert "metadata" in ent

    def test_classification_has_required_fields(self):
        result = consolidate_entities(
            classification_results=_make_classification_doc(),
            doc_name="cls_fields.pdf",
        )

        for cls in result["classifications"]:
            assert "label" in cls
            assert "confidence" in cls
            assert "method" in cls

    def test_kv_pair_has_required_fields(self):
        result = consolidate_entities(
            extraction_results=_make_extraction_doc(),
            doc_name="kv_fields.pdf",
        )

        for kv in result["key_value_pairs"]:
            assert "key" in kv
            assert "value" in kv
            assert "confidence" in kv
            assert "page" in kv
            assert "source" in kv

    def test_summary_has_required_fields(self):
        result = consolidate_entities(doc_name="summary.pdf")

        summary = result["summary"]
        assert "total_entities" in summary
        assert "entity_types" in summary
        assert "total_kv_pairs" in summary
        assert "primary_classification" in summary


# ---------------------------------------------------------------------------
# Tests: Integration with real dataclass instances
# ---------------------------------------------------------------------------


class TestWithRealDataclasses:
    """Test using actual dataclass instances from ner.py and extraction.py."""

    def test_with_ner_entity_dataclass(self):
        """Test with ner.Entity dataclass instances in page data."""
        try:
            from ner import Entity
        except ImportError:
            pytest.skip("ner module not available")

        entity = Entity(
            entity_type="PERSON",
            text="Ada Lovelace",
            confidence=0.99,
            page_num=1,
            start=0,
            end=12,
        )
        doc = _FakeDocumentNER(pages=[
            {"page_num": 1, "entities": [entity]},
        ])
        entities = _entities_from_ner(doc)

        assert len(entities) == 1
        assert entities[0]["type"] == "PERSON"
        assert entities[0]["text"] == "Ada Lovelace"
        assert entities[0]["confidence"] == 0.99

    def test_with_extraction_field_dataclass(self):
        """Test with extraction.ExtractedField dataclass instances."""
        try:
            from extraction import ExtractedField
        except ImportError:
            pytest.skip("extraction module not available")

        field_obj = ExtractedField(
            field_type="date",
            text="2024-06-15",
            confidence=0.97,
            page_num=1,
            start=10,
            end=20,
            extraction_method="regex",
            normalized_value="2024-06-15",
        )
        doc = _FakeDocumentExtraction(pages=[
            {"page_num": 1, "fields": [field_obj]},
        ])
        entities, kv_pairs = _entities_from_extraction(doc)

        assert len(entities) == 1
        assert entities[0]["type"] == "date"
        assert entities[0]["metadata"]["normalized_value"] == "2024-06-15"
        assert len(kv_pairs) == 1
        assert kv_pairs[0]["key"] == "date"

    def test_end_to_end_with_real_modules(self):
        """Full consolidation using real module dataclasses."""
        try:
            from extraction import (
                DocumentExtraction,
                PageExtraction,
                finalize_extraction,
            )
            from ner import DocumentNER, Entity
        except ImportError:
            pytest.skip("ner/extraction modules not available")

        # Build NER doc
        ner_doc = DocumentNER(document_id="e2e-test", source_file="e2e.pdf")
        ner_doc.pages = [
            {"page_num": 1, "entities": [
                Entity(entity_type="PERSON", text="Test User", confidence=0.9, page_num=1, start=0, end=9),
            ]},
        ]

        # Build extraction doc
        ext_doc = DocumentExtraction(document_id="e2e-test", source_file="e2e.pdf")
        ext_doc.pages = [
            PageExtraction(page_num=1, fields=[
                {
                    "field_type": "email_address",
                    "text": "test@example.com",
                    "confidence": 0.95,
                    "page_num": 1,
                    "start": 20,
                    "end": 36,
                    "extraction_method": "regex",
                    "normalized_value": "",
                },
            ]),
        ]
        finalize_extraction(ext_doc)

        result = consolidate_entities(
            ner_results=ner_doc,
            extraction_results=ext_doc,
            doc_name="e2e.pdf",
            pipeline_version="0.9.0-test",
        )

        assert result["summary"]["total_entities"] == 2
        sources = {e["source"] for e in result["entities"]}
        assert "ner" in sources
        assert "extraction" in sources
