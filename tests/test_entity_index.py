"""Tests for api.entity_index — SQLite-backed entity search index."""

from __future__ import annotations

import pytest

from api.entity_index import EntityIndex, IndexedEntity, IndexedExtraction


@pytest.fixture()
def index(tmp_path):
    """Create an EntityIndex backed by a temporary SQLite database."""
    db_path = str(tmp_path / "test_entity_index.db")
    return EntityIndex(db_path=db_path)


@pytest.fixture()
def populated_index(index):
    """An EntityIndex pre-populated with sample data."""
    # Job 1 entities
    index.index_entities("job_aaa111bbb222", "document1.pdf", [
        {"type": "PERSON", "text": "John Doe", "confidence": 0.95, "source": "ner", "page": 1},
        {"type": "DATE", "text": "2026-01-15", "confidence": 0.88, "source": "extraction", "page": 1},
        {"type": "PERSON", "text": "Jane Smith", "confidence": 0.72, "source": "ner", "page": 2},
    ])
    # Job 1 extractions
    index.index_extractions("job_aaa111bbb222", "document1.pdf", [
        {"key": "invoice_number", "value": "INV-2026-001", "confidence": 0.91, "page": 1},
        {"key": "date", "value": "2026-01-15", "confidence": 0.88, "page": 1},
        {"key": "amount", "value": "$1,500.00", "confidence": 0.85, "page": 2},
    ])
    # Job 2 entities
    index.index_entities("job_ccc333ddd444", "document2.pdf", [
        {"type": "ORG", "text": "Acme Corp", "confidence": 0.90, "source": "ner", "page": 1},
        {"type": "PERSON", "text": "Bob Jones", "confidence": 0.80, "source": "ner", "page": 1},
    ])
    # Job 2 extractions
    index.index_extractions("job_ccc333ddd444", "document2.pdf", [
        {"key": "invoice_number", "value": "INV-2026-002", "confidence": 0.93, "page": 1},
    ])
    return index


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestIndexedEntity:
    def test_to_dict(self):
        entity = IndexedEntity(
            entity_id="ent_abc123def456",
            job_id="job_aaa111bbb222",
            entity_type="PERSON",
            text="John Doe",
            confidence=0.95,
            source="ner",
            page=1,
            document_name="doc.pdf",
            indexed_at="2026-03-29T00:00:00.000+00:00",
        )
        d = entity.to_dict()
        assert d["entity_id"] == "ent_abc123def456"
        assert d["entity_type"] == "PERSON"
        assert d["text"] == "John Doe"
        assert d["confidence"] == 0.95
        assert d["source"] == "ner"
        assert d["page"] == 1

    def test_defaults(self):
        entity = IndexedEntity(
            entity_id="ent_000",
            job_id="job_000",
            entity_type="DATE",
            text="today",
        )
        assert entity.confidence == 0.0
        assert entity.source == ""
        assert entity.page == 0
        assert entity.document_name == ""


class TestIndexedExtraction:
    def test_to_dict(self):
        ext = IndexedExtraction(
            extraction_id="ext_abc123def456",
            job_id="job_aaa111bbb222",
            field_name="amount",
            field_value="$100",
            confidence=0.88,
            page=2,
            document_name="doc.pdf",
            indexed_at="2026-03-29T00:00:00.000+00:00",
        )
        d = ext.to_dict()
        assert d["extraction_id"] == "ext_abc123def456"
        assert d["field_name"] == "amount"
        assert d["field_value"] == "$100"
        assert d["confidence"] == 0.88

    def test_defaults(self):
        ext = IndexedExtraction(
            extraction_id="ext_000",
            job_id="job_000",
            field_name="date",
            field_value="2026-01-01",
        )
        assert ext.confidence == 0.0
        assert ext.page == 0


# ---------------------------------------------------------------------------
# EntityIndex core tests
# ---------------------------------------------------------------------------


class TestEntityIndexBasic:
    def test_empty_index(self, index):
        """Empty index returns empty results."""
        results, total = index.search_entities()
        assert results == []
        assert total == 0

        results, total = index.search_extractions()
        assert results == []
        assert total == 0

    def test_stats_empty(self, index):
        """Stats on empty index returns all zeros."""
        stats = index.stats()
        assert stats["total_entities"] == 0
        assert stats["total_extractions"] == 0
        assert stats["unique_entity_types"] == 0
        assert stats["unique_field_names"] == 0
        assert stats["jobs_indexed"] == 0

    def test_index_entities_returns_count(self, index):
        """index_entities returns number of entities indexed."""
        count = index.index_entities("job_001", "doc.pdf", [
            {"type": "PERSON", "text": "Alice", "confidence": 0.9, "source": "ner", "page": 1},
            {"type": "DATE", "text": "2026-01-01", "confidence": 0.8, "source": "extraction", "page": 1},
        ])
        assert count == 2

    def test_index_entities_empty_list(self, index):
        """Indexing an empty list returns 0."""
        count = index.index_entities("job_001", "doc.pdf", [])
        assert count == 0

    def test_index_extractions_returns_count(self, index):
        """index_extractions returns number of extractions indexed."""
        count = index.index_extractions("job_001", "doc.pdf", [
            {"key": "amount", "value": "$100", "confidence": 0.85, "page": 1},
        ])
        assert count == 1

    def test_index_extractions_empty_list(self, index):
        """Indexing an empty list returns 0."""
        count = index.index_extractions("job_001", "doc.pdf", [])
        assert count == 0


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------


class TestEntitySearch:
    def test_search_all(self, populated_index):
        """Search without filters returns all entities."""
        results, total = populated_index.search_entities()
        assert total == 5
        assert len(results) == 5

    def test_search_by_type(self, populated_index):
        """Filter by entity type."""
        results, total = populated_index.search_entities(entity_type="PERSON")
        assert total == 3
        texts = {r.text for r in results}
        assert "John Doe" in texts
        assert "Jane Smith" in texts
        assert "Bob Jones" in texts

    def test_search_by_type_no_match(self, populated_index):
        """Filter by non-existent entity type returns empty."""
        results, total = populated_index.search_entities(entity_type="LOCATION")
        assert total == 0
        assert results == []

    def test_search_by_text_query(self, populated_index):
        """Text LIKE search."""
        results, total = populated_index.search_entities(text_query="John")
        assert total == 1
        assert results[0].text == "John Doe"

    def test_search_by_text_query_partial(self, populated_index):
        """Partial text match via LIKE."""
        results, total = populated_index.search_entities(text_query="Doe")
        assert total == 1
        assert results[0].text == "John Doe"

    def test_search_by_job_id(self, populated_index):
        """Filter by job ID."""
        results, total = populated_index.search_entities(job_id="job_ccc333ddd444")
        assert total == 2
        job_ids = {r.job_id for r in results}
        assert job_ids == {"job_ccc333ddd444"}

    def test_search_by_min_confidence(self, populated_index):
        """Filter by minimum confidence."""
        results, total = populated_index.search_entities(min_confidence=0.9)
        assert total == 2
        for r in results:
            assert r.confidence >= 0.9

    def test_search_combined_filters(self, populated_index):
        """Multiple filters combined."""
        results, total = populated_index.search_entities(
            entity_type="PERSON",
            min_confidence=0.8,
        )
        assert total == 2
        for r in results:
            assert r.entity_type == "PERSON"
            assert r.confidence >= 0.8


class TestExtractionSearch:
    def test_search_all(self, populated_index):
        """Search without filters returns all extractions."""
        results, total = populated_index.search_extractions()
        assert total == 4
        assert len(results) == 4

    def test_search_by_field_name(self, populated_index):
        """Filter by field name."""
        results, total = populated_index.search_extractions(field_name="invoice_number")
        assert total == 2
        for r in results:
            assert r.field_name == "invoice_number"

    def test_search_by_value_query(self, populated_index):
        """Value LIKE search."""
        results, total = populated_index.search_extractions(value_query="INV-2026")
        assert total == 2
        for r in results:
            assert "INV-2026" in r.field_value

    def test_search_by_value_query_specific(self, populated_index):
        """Specific value search."""
        results, total = populated_index.search_extractions(value_query="$1,500")
        assert total == 1
        assert results[0].field_value == "$1,500.00"

    def test_search_by_job_id(self, populated_index):
        """Filter by job ID."""
        results, total = populated_index.search_extractions(job_id="job_ccc333ddd444")
        assert total == 1

    def test_search_by_min_confidence(self, populated_index):
        """Filter by minimum confidence."""
        results, total = populated_index.search_extractions(min_confidence=0.9)
        assert total == 2
        for r in results:
            assert r.confidence >= 0.9

    def test_search_combined_filters(self, populated_index):
        """Multiple filters combined."""
        results, total = populated_index.search_extractions(
            field_name="invoice_number",
            min_confidence=0.92,
        )
        assert total == 1
        assert results[0].field_value == "INV-2026-002"


# ---------------------------------------------------------------------------
# Pagination tests
# ---------------------------------------------------------------------------


class TestPagination:
    def test_entity_limit(self, populated_index):
        """Limit controls page size."""
        results, total = populated_index.search_entities(limit=2)
        assert len(results) == 2
        assert total == 5  # total is still full count

    def test_entity_offset(self, populated_index):
        """Offset skips results."""
        all_results, _ = populated_index.search_entities(limit=100)
        results, total = populated_index.search_entities(limit=2, offset=2)
        assert len(results) == 2
        assert total == 5
        # Offset results should differ from first page
        first_page, _ = populated_index.search_entities(limit=2, offset=0)
        first_ids = {r.entity_id for r in first_page}
        offset_ids = {r.entity_id for r in results}
        assert first_ids.isdisjoint(offset_ids)

    def test_entity_offset_beyond_total(self, populated_index):
        """Offset beyond total returns empty results."""
        results, total = populated_index.search_entities(offset=100)
        assert results == []
        assert total == 5

    def test_extraction_pagination(self, populated_index):
        """Extraction search pagination."""
        results, total = populated_index.search_extractions(limit=2, offset=0)
        assert len(results) == 2
        assert total == 4

        results2, total2 = populated_index.search_extractions(limit=2, offset=2)
        assert len(results2) == 2
        assert total2 == 4


# ---------------------------------------------------------------------------
# Remove / cleanup tests
# ---------------------------------------------------------------------------


class TestRemoveJob:
    def test_remove_job(self, populated_index):
        """Remove all entries for a job."""
        removed = populated_index.remove_job("job_aaa111bbb222")
        assert removed == 6  # 3 entities + 3 extractions

        # Verify they are gone
        results, total = populated_index.search_entities(job_id="job_aaa111bbb222")
        assert total == 0

        results, total = populated_index.search_extractions(job_id="job_aaa111bbb222")
        assert total == 0

    def test_remove_job_nonexistent(self, populated_index):
        """Removing a non-existent job returns 0."""
        removed = populated_index.remove_job("job_does_not_exist")
        assert removed == 0

    def test_remove_job_does_not_affect_others(self, populated_index):
        """Removing one job leaves other jobs intact."""
        populated_index.remove_job("job_aaa111bbb222")

        results, total = populated_index.search_entities(job_id="job_ccc333ddd444")
        assert total == 2

        results, total = populated_index.search_extractions(job_id="job_ccc333ddd444")
        assert total == 1


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_populated(self, populated_index):
        """Stats returns correct counts for populated index."""
        stats = populated_index.stats()
        assert stats["total_entities"] == 5
        assert stats["total_extractions"] == 4
        assert stats["unique_entity_types"] == 3  # PERSON, DATE, ORG
        assert stats["unique_field_names"] == 3  # invoice_number, date, amount
        assert stats["jobs_indexed"] == 2

    def test_stats_after_remove(self, populated_index):
        """Stats update after job removal."""
        populated_index.remove_job("job_aaa111bbb222")
        stats = populated_index.stats()
        assert stats["total_entities"] == 2
        assert stats["total_extractions"] == 1
        assert stats["jobs_indexed"] == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_entity_type_from_entity_type_key(self, index):
        """Entities with 'entity_type' key instead of 'type'."""
        count = index.index_entities("job_001", "doc.pdf", [
            {"entity_type": "AMOUNT", "text": "$50", "confidence": 0.8},
        ])
        assert count == 1
        results, total = index.search_entities(entity_type="AMOUNT")
        assert total == 1
        assert results[0].text == "$50"

    def test_extraction_field_name_key(self, index):
        """Extractions with 'field_name'/'field_value' keys."""
        count = index.index_extractions("job_001", "doc.pdf", [
            {"field_name": "reference", "field_value": "REF-001", "confidence": 0.7},
        ])
        assert count == 1
        results, total = index.search_extractions(field_name="reference")
        assert total == 1
        assert results[0].field_value == "REF-001"

    def test_missing_confidence_defaults_to_zero(self, index):
        """Missing confidence defaults to 0.0."""
        index.index_entities("job_001", "doc.pdf", [
            {"type": "PERSON", "text": "Nobody"},
        ])
        results, _ = index.search_entities()
        assert results[0].confidence == 0.0

    def test_indexed_at_is_populated(self, index):
        """Indexed entities have a non-empty indexed_at timestamp."""
        index.index_entities("job_001", "doc.pdf", [
            {"type": "DATE", "text": "today", "confidence": 0.5},
        ])
        results, _ = index.search_entities()
        assert results[0].indexed_at != ""
        assert "T" in results[0].indexed_at  # ISO format

    def test_document_name_stored(self, index):
        """Document name is stored and returned."""
        index.index_entities("job_001", "my_report.pdf", [
            {"type": "ORG", "text": "TestCorp", "confidence": 0.9},
        ])
        results, _ = index.search_entities()
        assert results[0].document_name == "my_report.pdf"
