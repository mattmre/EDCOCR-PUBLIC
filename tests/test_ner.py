"""
Unit tests for Named Entity Recognition module (ner.py).

Tests cover:
- Entity extraction with mocked spaCy
- Custom regex entity patterns (CASE_NUMBER, BATES_NUMBER, EXHIBIT_REF)
- Finalization: dedup, aggregation, summary computation
- JSON output format and file paths
- Graceful degradation when spaCy is not available

Run with: python -m pytest tests/test_ner.py -v
"""

import json
import os
from unittest import mock

# Add project root to path
from ner import (
    DocumentNER,
    Entity,
    PageNER,
    _entity_to_dict,
    extract_custom_entities,
    extract_entities,
    finalize_ner,
    write_ner_json,
)
from version import __version__

# ---------------------------------------------------------------------------
# Helpers: Mock spaCy objects
# ---------------------------------------------------------------------------


class MockSpacyEntity:
    """Mimics a spaCy Span entity."""

    def __init__(self, text, label_, start_char, end_char):
        self.text = text
        self.label_ = label_
        self.start_char = start_char
        self.end_char = end_char


class MockSpacyDoc:
    """Mimics a spaCy Doc with .ents attribute."""

    def __init__(self, ents):
        self.ents = ents


class MockNLP:
    """Callable mock that returns a MockSpacyDoc."""

    def __init__(self, ents):
        self._ents = ents

    def __call__(self, text):
        return MockSpacyDoc(self._ents)


# ---------------------------------------------------------------------------
# Tests: Entity dataclass
# ---------------------------------------------------------------------------


class TestEntityDataclass:
    def test_entity_defaults(self):
        e = Entity(entity_type="PERSON", text="John Smith")
        assert e.entity_type == "PERSON"
        assert e.text == "John Smith"
        assert e.confidence == 0.0
        assert e.page_num == 0
        assert e.start == 0
        assert e.end == 0

    def test_entity_with_values(self):
        e = Entity(
            entity_type="ORG",
            text="Acme Corp",
            confidence=0.95,
            page_num=3,
            start=100,
            end=109,
        )
        assert e.entity_type == "ORG"
        assert e.text == "Acme Corp"
        assert e.confidence == 0.95
        assert e.page_num == 3
        assert e.start == 100
        assert e.end == 109

    def test_page_ner_defaults(self):
        p = PageNER(page_num=1)
        assert p.page_num == 1
        assert p.entities == []

    def test_document_ner_defaults(self):
        d = DocumentNER(document_id="doc1", source_file="test.pdf")
        assert d.document_id == "doc1"
        assert d.source_file == "test.pdf"
        assert d.pages == []
        assert d.total_entities == 0
        assert d.entity_type_counts == {}
        assert d.unique_entities == []


# ---------------------------------------------------------------------------
# Tests: Entity extraction with mocked spaCy
# ---------------------------------------------------------------------------


class TestEntityExtraction:
    """Test extract_entities with mocked spaCy."""

    def test_extract_entities_returns_entities(self):
        mock_ents = [
            MockSpacyEntity("John Smith", "PERSON", 10, 20),
            MockSpacyEntity("Acme Corp", "ORG", 30, 39),
        ]
        mock_nlp = MockNLP(mock_ents)

        import ner as ner_module
        with mock.patch.object(ner_module, "_SPACY_AVAILABLE", True), \
             mock.patch.object(ner_module, "_load_nlp", return_value=mock_nlp):
            result = extract_entities("John Smith works at Acme Corp.", 1)

        assert len(result) == 2
        assert result[0].entity_type == "PERSON"
        assert result[0].text == "John Smith"
        assert result[0].page_num == 1
        assert result[0].start == 10
        assert result[0].end == 20
        assert result[1].entity_type == "ORG"
        assert result[1].text == "Acme Corp"

    def test_extract_entities_filters_types(self):
        """Only extracts PERSON, ORG, GPE, DATE, MONEY -- ignores others."""
        mock_ents = [
            MockSpacyEntity("John", "PERSON", 0, 4),
            MockSpacyEntity("something", "NORP", 5, 14),
            MockSpacyEntity("USA", "GPE", 15, 18),
        ]
        mock_nlp = MockNLP(mock_ents)

        import ner as ner_module
        with mock.patch.object(ner_module, "_SPACY_AVAILABLE", True), \
             mock.patch.object(ner_module, "_load_nlp", return_value=mock_nlp):
            result = extract_entities("John something USA", 2)

        assert len(result) == 2
        types = [e.entity_type for e in result]
        assert "PERSON" in types
        assert "GPE" in types
        assert "NORP" not in types

    def test_extract_entities_date_and_money(self):
        mock_ents = [
            MockSpacyEntity("January 1, 2024", "DATE", 0, 15),
            MockSpacyEntity("$1,000,000", "MONEY", 20, 30),
        ]
        mock_nlp = MockNLP(mock_ents)

        import ner as ner_module
        with mock.patch.object(ner_module, "_SPACY_AVAILABLE", True), \
             mock.patch.object(ner_module, "_load_nlp", return_value=mock_nlp):
            result = extract_entities("January 1, 2024 -- $1,000,000", 5)

        assert len(result) == 2
        assert result[0].entity_type == "DATE"
        assert result[1].entity_type == "MONEY"

    def test_extract_entities_confidence_is_1(self):
        """spaCy does not expose per-entity confidence, so default to 1.0."""
        mock_ents = [MockSpacyEntity("Test Person", "PERSON", 0, 11)]
        mock_nlp = MockNLP(mock_ents)

        import ner as ner_module
        with mock.patch.object(ner_module, "_SPACY_AVAILABLE", True), \
             mock.patch.object(ner_module, "_load_nlp", return_value=mock_nlp):
            result = extract_entities("Test Person", 1)

        assert result[0].confidence == 1.0

    def test_extract_entities_empty_text(self):
        mock_nlp = MockNLP([])

        import ner as ner_module
        with mock.patch.object(ner_module, "_SPACY_AVAILABLE", True), \
             mock.patch.object(ner_module, "_load_nlp", return_value=mock_nlp):
            result = extract_entities("", 1)

        assert result == []

    def test_extract_entities_spacy_exception(self):
        """spaCy error should return empty list, not crash."""

        def failing_nlp(text):
            raise RuntimeError("model crashed")

        import ner as ner_module
        with mock.patch.object(ner_module, "_SPACY_AVAILABLE", True), \
             mock.patch.object(ner_module, "_load_nlp", return_value=failing_nlp):
            result = extract_entities("some text", 1)

        assert result == []


# ---------------------------------------------------------------------------
# Tests: Custom regex entities
# ---------------------------------------------------------------------------


class TestCustomEntities:
    """Test regex patterns for CASE_NUMBER, BATES_NUMBER, EXHIBIT_REF."""

    # --- CASE_NUMBER ---

    def test_case_number_standard(self):
        text = "See Case No. 2024-CV-1234 for details."
        result = extract_custom_entities(text, 1)
        case_entities = [e for e in result if e.entity_type == "CASE_NUMBER"]
        assert len(case_entities) >= 1
        assert "2024-CV-1234" in case_entities[0].text

    def test_case_number_docket(self):
        text = "Refer to Docket No. 23-45678 filed on Monday."
        result = extract_custom_entities(text, 2)
        case_entities = [e for e in result if e.entity_type == "CASE_NUMBER"]
        assert len(case_entities) >= 1
        assert "23-45678" in case_entities[0].text

    def test_case_number_matter(self):
        text = "Matter Number 2024-ABC-99999 is pending."
        result = extract_custom_entities(text, 1)
        case_entities = [e for e in result if e.entity_type == "CASE_NUMBER"]
        assert len(case_entities) >= 1

    def test_case_number_not_present(self):
        text = "This is a normal sentence with no case numbers."
        result = extract_custom_entities(text, 1)
        case_entities = [e for e in result if e.entity_type == "CASE_NUMBER"]
        assert len(case_entities) == 0

    # --- BATES_NUMBER ---

    def test_bates_number_standard(self):
        text = "Document stamped ABC001234 was produced."
        result = extract_custom_entities(text, 3)
        bates_entities = [e for e in result if e.entity_type == "BATES_NUMBER"]
        assert len(bates_entities) >= 1
        assert "ABC001234" in bates_entities[0].text

    def test_bates_number_with_space(self):
        text = "Bates number XYZ 000001 referenced in deposition."
        result = extract_custom_entities(text, 1)
        bates_entities = [e for e in result if e.entity_type == "BATES_NUMBER"]
        assert len(bates_entities) >= 1

    def test_bates_number_longer_prefix(self):
        text = "Production ABCDEF12345678 is complete."
        result = extract_custom_entities(text, 1)
        bates_entities = [e for e in result if e.entity_type == "BATES_NUMBER"]
        assert len(bates_entities) >= 1

    def test_bates_number_not_present(self):
        text = "Short AB12 tokens should not match."
        result = extract_custom_entities(text, 1)
        bates_entities = [e for e in result if e.entity_type == "BATES_NUMBER"]
        # AB12 has only 2 alpha + 2 digits -- below threshold
        assert len(bates_entities) == 0

    def test_bates_number_rejects_state_zip(self):
        """State abbreviation + zip code should not match as Bates number."""
        text = "Springfield, IL 62704 is the mailing address."
        result = extract_custom_entities(text, 1)
        bates_entities = [e for e in result if e.entity_type == "BATES_NUMBER"]
        assert len(bates_entities) == 0, (
            f"Expected no Bates match for 'IL 62704', got: {bates_entities}"
        )

    def test_bates_number_rejects_common_phrases(self):
        """Common two-letter word + year should not match as Bates number."""
        text = "The meeting is on 2024 schedule and in 2025 plan."
        result = extract_custom_entities(text, 1)
        bates_entities = [e for e in result if e.entity_type == "BATES_NUMBER"]
        assert len(bates_entities) == 0, (
            f"Expected no Bates match for 'on 2024', got: {bates_entities}"
        )

    # --- EXHIBIT_REF ---

    def test_exhibit_ref_letter(self):
        text = "As shown in Exhibit A, the data is clear."
        result = extract_custom_entities(text, 1)
        exhibit_entities = [e for e in result if e.entity_type == "EXHIBIT_REF"]
        assert len(exhibit_entities) >= 1
        assert exhibit_entities[0].text == "A"

    def test_exhibit_ref_number(self):
        text = "Please see Exhibit 5 for the chart."
        result = extract_custom_entities(text, 2)
        exhibit_entities = [e for e in result if e.entity_type == "EXHIBIT_REF"]
        assert len(exhibit_entities) >= 1

    def test_exhibit_ref_abbreviated(self):
        text = "Per Ex. B, the evidence was admitted."
        result = extract_custom_entities(text, 1)
        exhibit_entities = [e for e in result if e.entity_type == "EXHIBIT_REF"]
        assert len(exhibit_entities) >= 1

    def test_exhibit_ref_with_number_suffix(self):
        text = "Exhibit No. 42 was entered into the record."
        result = extract_custom_entities(text, 1)
        exhibit_entities = [e for e in result if e.entity_type == "EXHIBIT_REF"]
        assert len(exhibit_entities) >= 1

    def test_exhibit_ref_not_present(self):
        text = "There are no exhibit references here."
        result = extract_custom_entities(text, 1)
        exhibit_entities = [e for e in result if e.entity_type == "EXHIBIT_REF"]
        assert len(exhibit_entities) == 0

    # --- Mixed ---

    def test_multiple_custom_entity_types(self):
        text = (
            "Case No. 2024-CV-5678 involves document ABC001234. "
            "See Exhibit A for supporting evidence."
        )
        result = extract_custom_entities(text, 1)
        types_found = {e.entity_type for e in result}
        assert "CASE_NUMBER" in types_found
        assert "BATES_NUMBER" in types_found
        assert "EXHIBIT_REF" in types_found

    def test_custom_entities_have_offsets(self):
        text = "Case No. 2024-CV-1234"
        result = extract_custom_entities(text, 1)
        case_entities = [e for e in result if e.entity_type == "CASE_NUMBER"]
        assert len(case_entities) >= 1
        e = case_entities[0]
        assert e.start >= 0
        assert e.end > e.start
        assert e.page_num == 1
        assert e.confidence == 1.0

    def test_custom_entities_page_num_preserved(self):
        text = "Exhibit B referenced here."
        result = extract_custom_entities(text, 42)
        for e in result:
            if e.entity_type == "EXHIBIT_REF":
                assert e.page_num == 42


# ---------------------------------------------------------------------------
# Tests: finalize_ner
# ---------------------------------------------------------------------------


class TestFinalizeNER:
    def test_finalize_empty_document(self):
        doc = DocumentNER(document_id="d1", source_file="empty.pdf")
        result = finalize_ner(doc)
        assert result.total_entities == 0
        assert result.entity_type_counts == {}
        assert result.unique_entities == []

    def test_finalize_counts_entities(self):
        doc = DocumentNER(document_id="d2", source_file="test.pdf")
        doc.pages = [{
            "page_num": 1,
            "entities": [
                Entity(entity_type="PERSON", text="John Smith", page_num=1),
                Entity(entity_type="ORG", text="Acme Corp", page_num=1),
                Entity(entity_type="PERSON", text="Jane Doe", page_num=1),
            ],
        }]
        result = finalize_ner(doc)
        assert result.total_entities == 3
        assert result.entity_type_counts["PERSON"] == 2
        assert result.entity_type_counts["ORG"] == 1

    def test_finalize_deduplicates(self):
        doc = DocumentNER(document_id="d3", source_file="dup.pdf")
        doc.pages = [
            {
                "page_num": 1,
                "entities": [
                    Entity(entity_type="PERSON", text="John Smith", page_num=1),
                ],
            },
            {
                "page_num": 2,
                "entities": [
                    Entity(entity_type="PERSON", text="John Smith", page_num=2),
                    Entity(entity_type="PERSON", text="john smith", page_num=2),
                ],
            },
        ]
        result = finalize_ner(doc)
        # Total count should be 3 (no dedup on total)
        assert result.total_entities == 3
        # Unique should dedup by (type, normalized text)
        person_uniques = [u for u in result.unique_entities if u["type"] == "PERSON"]
        assert len(person_uniques) == 1

    def test_finalize_multiple_types(self):
        doc = DocumentNER(document_id="d4", source_file="multi.pdf")
        doc.pages = [{
            "page_num": 1,
            "entities": [
                Entity(entity_type="PERSON", text="Alice", page_num=1),
                Entity(entity_type="GPE", text="New York", page_num=1),
                Entity(entity_type="CASE_NUMBER", text="Case No. 2024-CV-1234", page_num=1),
            ],
        }]
        result = finalize_ner(doc)
        assert result.total_entities == 3
        assert "PERSON" in result.entity_type_counts
        assert "GPE" in result.entity_type_counts
        assert "CASE_NUMBER" in result.entity_type_counts

    def test_finalize_with_page_ner_objects(self):
        """finalize_ner should accept PageNER objects as well as dicts."""
        doc = DocumentNER(document_id="d5", source_file="obj.pdf")
        page = PageNER(page_num=1)
        page.entities = [
            Entity(entity_type="ORG", text="TestCo", page_num=1),
        ]
        doc.pages = [page]
        result = finalize_ner(doc)
        assert result.total_entities == 1
        assert result.entity_type_counts["ORG"] == 1

    def test_finalize_sorted_type_counts(self):
        doc = DocumentNER(document_id="d6", source_file="sorted.pdf")
        doc.pages = [{
            "page_num": 1,
            "entities": [
                Entity(entity_type="PERSON", text="A", page_num=1),
                Entity(entity_type="DATE", text="2024-01-01", page_num=1),
                Entity(entity_type="MONEY", text="$100", page_num=1),
            ],
        }]
        result = finalize_ner(doc)
        keys = list(result.entity_type_counts.keys())
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Tests: _entity_to_dict
# ---------------------------------------------------------------------------


class TestEntityToDict:
    def test_entity_object_to_dict(self):
        e = Entity(
            entity_type="PERSON", text="Test", confidence=0.9,
            page_num=1, start=0, end=4,
        )
        d = _entity_to_dict(e)
        assert d["type"] == "PERSON"
        assert d["text"] == "Test"
        assert d["confidence"] == 0.9
        assert d["page_num"] == 1
        assert d["start"] == 0
        assert d["end"] == 4

    def test_dict_passthrough(self):
        d = {"type": "ORG", "text": "Corp"}
        result = _entity_to_dict(d)
        assert result == d

    def test_unknown_type_returns_empty(self):
        result = _entity_to_dict(42)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: write_ner_json
# ---------------------------------------------------------------------------


class TestWriteNERJson:
    def test_write_creates_file(self, tmp_path):
        doc = DocumentNER(document_id="test_id", source_file="subfolder/doc.pdf")
        doc.pages = [{
            "page_num": 1,
            "entities": [
                Entity(entity_type="PERSON", text="John", page_num=1, start=0, end=4),
            ],
        }]
        doc = finalize_ner(doc)
        result = write_ner_json(doc, str(tmp_path), "subfolder", "0.4.0")
        assert result is not None
        assert os.path.exists(result)
        assert result.endswith(".ner.json")

    def test_write_non_pdf_uses_ext_token(self, tmp_path):
        doc = DocumentNER(document_id="img_doc", source_file="folder/photo.png")
        doc.pages = []
        doc = finalize_ner(doc)
        result = write_ner_json(doc, str(tmp_path), "", __version__)
        assert result is not None
        assert os.path.basename(result) == "photo__png.ner.json"

    def test_write_json_schema_valid(self, tmp_path):
        doc = DocumentNER(document_id="schema_test", source_file="test.pdf")
        doc.pages = [{
            "page_num": 1,
            "entities": [
                Entity(entity_type="PERSON", text="Jane Doe", confidence=1.0,
                       page_num=1, start=0, end=8),
                Entity(entity_type="ORG", text="ACME", confidence=1.0,
                       page_num=1, start=20, end=24),
            ],
        }]
        doc = finalize_ner(doc)
        result_path = write_ner_json(doc, str(tmp_path), "", "0.4.0")
        assert result_path is not None

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0"
        assert data["document_id"] == "schema_test"
        assert data["source_file"] == "test.pdf"
        assert "processing" in data
        assert data["processing"]["pipeline_version"] == "0.4.0"
        assert data["processing"]["ner_engine"] in ("spacy", "regex_only")
        assert "entities_summary" in data
        assert data["entities_summary"]["total_entities"] == 2
        assert "PERSON" in data["entities_summary"]["entity_types"]
        assert "pages" in data
        assert len(data["pages"]) == 1
        assert len(data["pages"][0]["entities"]) == 2

    def test_write_creates_directories(self, tmp_path):
        doc = DocumentNER(document_id="dir_test", source_file="deep/nested/doc.pdf")
        doc.pages = []
        doc = finalize_ner(doc)
        result = write_ner_json(doc, str(tmp_path), "deep/nested", "0.4.0")
        assert result is not None
        assert os.path.exists(result)
        expected_dir = os.path.join(str(tmp_path), "deep", "nested")
        assert os.path.isdir(expected_dir)

    def test_write_no_subfolder(self, tmp_path):
        doc = DocumentNER(document_id="root_test", source_file="root_doc.pdf")
        doc.pages = []
        doc = finalize_ner(doc)
        result = write_ner_json(doc, str(tmp_path), ".", "0.4.0")
        assert result is not None
        assert os.path.exists(result)
        assert os.path.dirname(result) == str(tmp_path)

    def test_write_entity_offsets_preserved(self, tmp_path):
        doc = DocumentNER(document_id="offset_test", source_file="offsets.pdf")
        doc.pages = [{
            "page_num": 1,
            "entities": [
                Entity(entity_type="PERSON", text="Alice", confidence=1.0,
                       page_num=1, start=50, end=55),
            ],
        }]
        doc = finalize_ner(doc)
        result_path = write_ner_json(doc, str(tmp_path), "", "0.4.0")

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entity = data["pages"][0]["entities"][0]
        assert entity["start"] == 50
        assert entity["end"] == 55

    def test_path_traversal_protection(self, tmp_path):
        """Subfolder with '..' segments is neutralized by sanitization."""
        doc = DocumentNER(document_id="traversal_test", source_file="evil.pdf")
        doc = finalize_ner(doc)
        ner_dir = str(tmp_path / "NER")
        result = write_ner_json(doc, ner_dir, "../../etc", "0.5.0")
        assert result is not None
        assert os.path.realpath(result).startswith(os.path.realpath(ner_dir))


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_extract_entities_without_spacy(self):
        """When _SPACY_AVAILABLE is False, extract_entities returns empty list."""
        import ner as ner_module
        with mock.patch.object(ner_module, "_SPACY_AVAILABLE", False):
            result = extract_entities("John Smith works at ACME Corp.", 1)
        assert result == []

    def test_custom_entities_always_work(self):
        """Custom regex entities should work regardless of spaCy availability."""
        import ner as ner_module
        with mock.patch.object(ner_module, "_SPACY_AVAILABLE", False):
            result = extract_custom_entities("Case No. 2024-CV-1234", 1)
        case_entities = [e for e in result if e.entity_type == "CASE_NUMBER"]
        assert len(case_entities) >= 1

    def test_nlp_load_returns_none_without_spacy(self):
        """_load_nlp should return None when spaCy is not available."""
        import ner as ner_module
        with mock.patch.object(ner_module, "_SPACY_AVAILABLE", False):
            from ner import _load_nlp
            # Reset cache
            original_cache = ner_module._nlp_cache
            ner_module._nlp_cache = None
            try:
                result = _load_nlp()
                assert result is None
            finally:
                ner_module._nlp_cache = original_cache

    def test_write_ner_json_engine_fallback(self, tmp_path):
        """When spaCy is not available, engine should be 'regex_only'."""
        import ner as ner_module
        doc = DocumentNER(document_id="fallback_test", source_file="fallback.pdf")
        doc.pages = []
        doc = finalize_ner(doc)

        with mock.patch.object(ner_module, "_SPACY_AVAILABLE", False):
            result_path = write_ner_json(doc, str(tmp_path), "", "0.4.0")

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["processing"]["ner_engine"] == "regex_only"
        assert data["processing"]["ner_model"] == "n/a"

    def test_finalize_ner_works_without_spacy(self):
        """finalize_ner only uses dataclasses, no spaCy dependency."""
        doc = DocumentNER(document_id="no_spacy", source_file="doc.pdf")
        doc.pages = [{
            "page_num": 1,
            "entities": [
                Entity(entity_type="CASE_NUMBER", text="Case No. 2024-CV-1234", page_num=1),
            ],
        }]
        result = finalize_ner(doc)
        assert result.total_entities == 1
        assert result.entity_type_counts["CASE_NUMBER"] == 1
