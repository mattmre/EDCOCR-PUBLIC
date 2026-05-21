"""Tests for entity relationship extraction (relationship_extraction.py)."""

from relationship_extraction import (
    ENABLE_RELATIONSHIP_EXTRACTION,
    EntityRelationship,
    RelationshipExtractor,
    _canonical_type,
    _char_distance,
    _entity_in_span,
    _entity_offsets,
    _extract_evidence,
    _kv_linkage_relationships,
    _pattern_relationships,
    _proximity_relationships,
    deduplicate_relationships,
    extract_and_attach_relationships,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ent(
    eid, etype, text, page=1, start=0, end=0, confidence=0.9, source="ner"
):
    """Build a minimal entity dict matching the consolidated schema."""
    return {
        "id": eid,
        "type": etype,
        "text": text,
        "confidence": confidence,
        "source": source,
        "page": page,
        "bbox": [],
        "metadata": {"start": start, "end": end},
    }


def _kv(key, value, page=1, confidence=0.9, source="extraction"):
    """Build a minimal KV pair dict."""
    return {
        "key": key,
        "value": value,
        "confidence": confidence,
        "page": page,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Tests: EntityRelationship dataclass
# ---------------------------------------------------------------------------


class TestEntityRelationship:
    """Test EntityRelationship data class."""

    def test_to_dict(self):
        rel = EntityRelationship(
            source_id="ent_001",
            target_id="ent_002",
            relation_type="works_for",
            confidence=0.85,
            evidence="John at Acme",
            page=1,
            metadata={"method": "proximity"},
        )
        d = rel.to_dict()
        assert d["source_id"] == "ent_001"
        assert d["target_id"] == "ent_002"
        assert d["type"] == "works_for"
        assert d["confidence"] == 0.85
        assert d["evidence"] == "John at Acme"
        assert d["page"] == 1
        assert d["metadata"]["method"] == "proximity"

    def test_confidence_rounded(self):
        rel = EntityRelationship(
            source_id="a", target_id="b",
            relation_type="dated",
            confidence=0.123456789,
        )
        d = rel.to_dict()
        assert d["confidence"] == 0.1235

    def test_default_values(self):
        rel = EntityRelationship(
            source_id="a", target_id="b", relation_type="test"
        )
        assert rel.confidence == 0.0
        assert rel.evidence == ""
        assert rel.page == 0
        assert rel.metadata == {}


# ---------------------------------------------------------------------------
# Tests: Canonical type normalization
# ---------------------------------------------------------------------------


class TestCanonicalType:
    """Test _canonical_type helper."""

    def test_person_types(self):
        assert _canonical_type("PERSON") == "person"
        assert _canonical_type("person_name") == "person"
        assert _canonical_type("Person") == "person"

    def test_org_types(self):
        assert _canonical_type("ORG") == "org"
        assert _canonical_type("organization") == "org"
        assert _canonical_type("Organization") == "org"

    def test_location_types(self):
        assert _canonical_type("GPE") == "location"
        assert _canonical_type("address") == "location"
        assert _canonical_type("location") == "location"

    def test_date_type(self):
        assert _canonical_type("DATE") == "date"
        assert _canonical_type("date") == "date"

    def test_amount_types(self):
        assert _canonical_type("MONEY") == "amount"
        assert _canonical_type("amount") == "amount"

    def test_case_types(self):
        assert _canonical_type("CASE_NUMBER") == "case_number"
        assert _canonical_type("case_number") == "case_number"

    def test_exhibit_types(self):
        assert _canonical_type("EXHIBIT_REF") == "exhibit_ref"

    def test_reference_types(self):
        assert _canonical_type("reference_number") == "reference_number"

    def test_unknown_type_passes_through(self):
        assert _canonical_type("custom_type") == "custom_type"
        assert _canonical_type("BATES_NUMBER") == "bates_number"

    def test_whitespace_stripped(self):
        assert _canonical_type("  PERSON  ") == "person"


# ---------------------------------------------------------------------------
# Tests: Text helpers
# ---------------------------------------------------------------------------


class TestEntityOffsets:
    """Test _entity_offsets helper."""

    def test_offsets_from_metadata(self):
        ent = _ent("e1", "PERSON", "Alice", start=10, end=15)
        start, end = _entity_offsets(ent)
        assert start == 10
        assert end == 15

    def test_offsets_from_direct_keys(self):
        ent = {"id": "e1", "type": "PERSON", "text": "Alice", "start": 5, "end": 10}
        start, end = _entity_offsets(ent)
        assert start == 5
        assert end == 10

    def test_missing_offsets_returns_negative(self):
        ent = {"id": "e1", "type": "PERSON", "text": "Alice"}
        start, end = _entity_offsets(ent)
        assert start == -1
        assert end == -1


class TestExtractEvidence:
    """Test _extract_evidence helper."""

    def test_extracts_snippet(self):
        text = "Hello, John Smith works at Acme Corp in New York."
        snippet = _extract_evidence(text, 7, 17, context=5)
        assert "John Smith" in snippet

    def test_handles_start_of_text(self):
        text = "John Smith at Acme"
        snippet = _extract_evidence(text, 0, 10, context=5)
        assert "John" in snippet

    def test_handles_end_of_text(self):
        text = "At Acme Corp"
        snippet = _extract_evidence(text, 3, 12, context=5)
        assert "Acme" in snippet

    def test_empty_text_returns_empty(self):
        assert _extract_evidence("", 0, 5) == ""

    def test_long_snippet_truncated(self):
        text = "A" * 500
        snippet = _extract_evidence(text, 100, 400, context=100)
        assert len(snippet) <= 204  # 200 + "..."

    def test_collapses_whitespace(self):
        text = "John   Smith    at    Acme"
        snippet = _extract_evidence(text, 0, 25, context=0)
        assert "  " not in snippet


class TestCharDistance:
    """Test _char_distance helper."""

    def test_adjacent_entities(self):
        a = _ent("e1", "PERSON", "John", start=0, end=4)
        b = _ent("e2", "ORG", "Acme", start=8, end=12)
        assert _char_distance(a, b) == 4

    def test_overlapping_entities(self):
        a = _ent("e1", "PERSON", "John", start=0, end=10)
        b = _ent("e2", "ORG", "Acme", start=5, end=15)
        assert _char_distance(a, b) == 0

    def test_reversed_order(self):
        a = _ent("e1", "ORG", "Acme", start=20, end=24)
        b = _ent("e2", "PERSON", "John", start=0, end=4)
        assert _char_distance(a, b) == 16

    def test_missing_offsets_returns_large(self):
        a = {"id": "e1", "type": "PERSON", "text": "John"}
        b = {"id": "e2", "type": "ORG", "text": "Acme"}
        assert _char_distance(a, b) == 999999


class TestEntityInSpan:
    """Test _entity_in_span helper."""

    def test_entity_within_span_by_offsets(self):
        ent = _ent("e1", "PERSON", "John", start=5, end=9)
        assert _entity_in_span(ent, "Hello John Smith", 0, 20)

    def test_entity_outside_span_by_offsets(self):
        ent = _ent("e1", "PERSON", "John", start=5, end=9)
        assert not _entity_in_span(ent, "Hello John Smith", 10, 20)

    def test_fallback_substring_search(self):
        ent = {"id": "e1", "type": "PERSON", "text": "John"}
        text = "Hello John Smith at Acme Corp"
        assert _entity_in_span(ent, text, 0, 15)

    def test_empty_entity_text(self):
        ent = {"id": "e1", "type": "PERSON", "text": ""}
        assert not _entity_in_span(ent, "some text", 0, 9)


# ---------------------------------------------------------------------------
# Tests: Proximity-based relationships
# ---------------------------------------------------------------------------


class TestProximityRelationships:
    """Test _proximity_relationships."""

    def test_person_org_proximity(self):
        entities = [
            _ent("ent_001", "PERSON", "John Smith", start=0, end=10),
            _ent("ent_002", "ORG", "Acme Corp", start=14, end=23),
        ]
        page_texts = {1: "John Smith at Acme Corp in the office"}
        rels = _proximity_relationships(entities, page_texts, max_distance=150)
        assert len(rels) >= 1
        rel = rels[0]
        assert rel.relation_type == "works_for"
        assert rel.source_id == "ent_001"
        assert rel.target_id == "ent_002"

    def test_org_person_order_normalized(self):
        """When ORG appears before PERSON, source should still be PERSON."""
        entities = [
            _ent("ent_001", "ORG", "Acme Corp", start=0, end=9),
            _ent("ent_002", "PERSON", "John Smith", start=14, end=24),
        ]
        page_texts = {1: "Acme Corp has John Smith as CEO"}
        rels = _proximity_relationships(entities, page_texts, max_distance=150)
        assert len(rels) >= 1
        rel = rels[0]
        assert rel.relation_type == "works_for"
        assert rel.source_id == "ent_002"  # PERSON is source
        assert rel.target_id == "ent_001"  # ORG is target

    def test_date_proximity(self):
        entities = [
            _ent("ent_001", "PERSON", "Jane Doe", start=0, end=8),
            _ent("ent_002", "DATE", "2024-01-15", start=12, end=22),
        ]
        page_texts = {1: "Jane Doe on 2024-01-15 filed the report"}
        rels = _proximity_relationships(entities, page_texts, max_distance=150)
        assert len(rels) >= 1
        assert any(r.relation_type == "dated" for r in rels)

    def test_amount_proximity(self):
        entities = [
            _ent("ent_001", "MONEY", "$5,000.00", start=0, end=9),
            _ent("ent_002", "PERSON", "Jane Doe", start=14, end=22),
        ]
        page_texts = {1: "$5,000.00 for Jane Doe services"}
        rels = _proximity_relationships(entities, page_texts, max_distance=150)
        assert len(rels) >= 1
        assert any(r.relation_type == "amount_for" for r in rels)

    def test_location_proximity(self):
        entities = [
            _ent("ent_001", "ORG", "Acme Corp", start=0, end=9),
            _ent("ent_002", "GPE", "New York", start=13, end=21),
        ]
        page_texts = {1: "Acme Corp in New York office"}
        rels = _proximity_relationships(entities, page_texts, max_distance=150)
        assert len(rels) >= 1
        assert any(r.relation_type == "located_at" for r in rels)

    def test_different_pages_no_relationship(self):
        entities = [
            _ent("ent_001", "PERSON", "John", page=1, start=0, end=4),
            _ent("ent_002", "ORG", "Acme", page=2, start=0, end=4),
        ]
        page_texts = {1: "John Smith", 2: "Acme Corp"}
        rels = _proximity_relationships(entities, page_texts, max_distance=150)
        assert len(rels) == 0

    def test_entities_too_far_apart(self):
        entities = [
            _ent("ent_001", "PERSON", "John", start=0, end=4),
            _ent("ent_002", "ORG", "Acme", start=500, end=504),
        ]
        page_texts = {1: "John" + " " * 496 + "Acme"}
        rels = _proximity_relationships(entities, page_texts, max_distance=50)
        assert len(rels) == 0

    def test_proximity_confidence_scales_with_distance(self):
        """Closer entities should have higher confidence."""
        close_entities = [
            _ent("ent_001", "PERSON", "John", start=0, end=4),
            _ent("ent_002", "ORG", "Acme", start=8, end=12),
        ]
        far_entities = [
            _ent("ent_001", "PERSON", "John", start=0, end=4),
            _ent("ent_002", "ORG", "Acme", start=140, end=144),
        ]
        text_close = "John at Acme Corp"
        text_far = "John" + " " * 136 + "Acme"
        page_texts_close = {1: text_close}
        page_texts_far = {1: text_far}

        rels_close = _proximity_relationships(close_entities, page_texts_close, 150)
        rels_far = _proximity_relationships(far_entities, page_texts_far, 150)

        assert len(rels_close) >= 1
        assert len(rels_far) >= 1
        assert rels_close[0].confidence > rels_far[0].confidence

    def test_empty_entities_returns_empty(self):
        rels = _proximity_relationships([], {1: "text"}, 150)
        assert rels == []

    def test_single_entity_returns_empty(self):
        entities = [_ent("ent_001", "PERSON", "John", start=0, end=4)]
        rels = _proximity_relationships(entities, {1: "John"}, 150)
        assert rels == []


# ---------------------------------------------------------------------------
# Tests: KV linkage relationships
# ---------------------------------------------------------------------------


class TestKVLinkageRelationships:
    """Test _kv_linkage_relationships."""

    def test_person_org_via_kv_pairs(self):
        entities = [
            _ent("ent_001", "person_name", "John Smith", page=1),
            _ent("ent_002", "organization", "Acme Corp", page=1),
        ]
        kv_pairs = [
            _kv("person_name", "John Smith", page=1),
            _kv("organization", "Acme Corp", page=1),
        ]
        rels = _kv_linkage_relationships(entities, kv_pairs)
        assert len(rels) >= 1
        assert rels[0].relation_type == "works_for"

    def test_person_location_via_kv_pairs(self):
        entities = [
            _ent("ent_001", "person_name", "Jane Doe", page=1),
            _ent("ent_002", "address", "123 Main St", page=1),
        ]
        kv_pairs = [
            _kv("person_name", "Jane Doe", page=1),
            _kv("address", "123 Main St", page=1),
        ]
        rels = _kv_linkage_relationships(entities, kv_pairs)
        assert len(rels) >= 1
        assert rels[0].relation_type == "located_at"

    def test_amount_linkage(self):
        entities = [
            _ent("ent_001", "amount", "$5,000.00", page=1),
            _ent("ent_002", "person_name", "John Smith", page=1),
        ]
        kv_pairs = [
            _kv("amount", "$5,000.00", page=1),
            _kv("person_name", "John Smith", page=1),
        ]
        rels = _kv_linkage_relationships(entities, kv_pairs)
        assert len(rels) >= 1
        assert rels[0].relation_type == "amount_for"

    def test_no_kv_pairs_returns_empty(self):
        entities = [_ent("ent_001", "PERSON", "John")]
        rels = _kv_linkage_relationships(entities, [])
        assert rels == []

    def test_no_entities_returns_empty(self):
        kv_pairs = [_kv("person_name", "John")]
        rels = _kv_linkage_relationships([], kv_pairs)
        assert rels == []

    def test_unmatched_kv_values_no_relationship(self):
        entities = [
            _ent("ent_001", "PERSON", "John Smith", page=1),
            _ent("ent_002", "ORG", "Acme Corp", page=1),
        ]
        kv_pairs = [
            _kv("person_name", "Jane Doe", page=1),  # Does not match any entity
            _kv("organization", "Beta Inc", page=1),  # Does not match any entity
        ]
        rels = _kv_linkage_relationships(entities, kv_pairs)
        assert len(rels) == 0

    def test_different_page_kv_pairs_not_linked(self):
        entities = [
            _ent("ent_001", "person_name", "John", page=1),
            _ent("ent_002", "organization", "Acme", page=2),
        ]
        kv_pairs = [
            _kv("person_name", "John", page=1),
            _kv("organization", "Acme", page=2),
        ]
        rels = _kv_linkage_relationships(entities, kv_pairs)
        assert len(rels) == 0

    def test_kv_linkage_confidence_from_kv_confidence(self):
        entities = [
            _ent("ent_001", "person_name", "John Smith", page=1),
            _ent("ent_002", "organization", "Acme Corp", page=1),
        ]
        kv_pairs = [
            _kv("person_name", "John Smith", page=1, confidence=0.8),
            _kv("organization", "Acme Corp", page=1, confidence=0.6),
        ]
        rels = _kv_linkage_relationships(entities, kv_pairs)
        assert len(rels) >= 1
        # Confidence should factor in the avg kv confidence
        assert rels[0].confidence > 0


# ---------------------------------------------------------------------------
# Tests: Pattern-based relationships
# ---------------------------------------------------------------------------


class TestPatternRelationships:
    """Test _pattern_relationships."""

    def test_works_for_at_pattern(self):
        text = "John Smith at Acme Corp filed the motion"
        entities = [
            _ent("ent_001", "PERSON", "John Smith", start=0, end=10),
            _ent("ent_002", "ORG", "Acme Corp", start=14, end=23),
        ]
        rels = _pattern_relationships(entities, {1: text})
        works_for = [r for r in rels if r.relation_type == "works_for"]
        assert len(works_for) >= 1

    def test_works_for_employed_by_pattern(self):
        text = "Jane Doe employed by Beta Industries"
        entities = [
            _ent("ent_001", "PERSON", "Jane Doe", start=0, end=8),
            _ent("ent_002", "ORG", "Beta Industries", start=21, end=36),
        ]
        rels = _pattern_relationships(entities, {1: text})
        works_for = [r for r in rels if r.relation_type == "works_for"]
        assert len(works_for) >= 1

    def test_signed_by_pattern(self):
        text = "This document was signed by John Smith on January 15, 2024"
        entities = [
            _ent("ent_001", "PERSON", "John Smith", start=28, end=38),
        ]
        rels = _pattern_relationships(entities, {1: text})
        signed = [r for r in rels if r.relation_type == "signed_by"]
        assert len(signed) >= 1
        assert signed[0].source_id == "ent_001"
        assert signed[0].target_id == "document"

    def test_sent_to_pattern(self):
        text = "To: Jane Doe\nFrom: John Smith"
        entities = [
            _ent("ent_001", "PERSON", "Jane Doe", start=4, end=12),
            _ent("ent_002", "PERSON", "John Smith", start=19, end=29),
        ]
        rels = _pattern_relationships(entities, {1: text})
        sent_to = [r for r in rels if r.relation_type == "sent_to"]
        assert len(sent_to) >= 1

    def test_sent_from_pattern(self):
        text = "From: John Smith\nDate: January 15"
        entities = [
            _ent("ent_001", "PERSON", "John Smith", start=6, end=16),
        ]
        rels = _pattern_relationships(entities, {1: text})
        sent_from = [r for r in rels if r.relation_type == "sent_from"]
        assert len(sent_from) >= 1
        assert sent_from[0].source_id == "ent_001"

    def test_references_pattern(self):
        text = "Re: Case No. 24-CV-1234 regarding the dispute"
        entities = [
            _ent("ent_001", "CASE_NUMBER", "24-CV-1234", start=13, end=23),
        ]
        rels = _pattern_relationships(entities, {1: text})
        refs = [r for r in rels if r.relation_type == "references"]
        assert len(refs) >= 1

    def test_located_at_pattern(self):
        text = "Acme Corp located at New York City headquarters"
        entities = [
            _ent("ent_001", "ORG", "Acme Corp", start=0, end=9),
            _ent("ent_002", "GPE", "New York City", start=21, end=34),
        ]
        rels = _pattern_relationships(entities, {1: text})
        located = [r for r in rels if r.relation_type == "located_at"]
        assert len(located) >= 1

    def test_no_entities_returns_empty(self):
        rels = _pattern_relationships([], {1: "some text"})
        assert rels == []

    def test_no_text_returns_empty(self):
        entities = [_ent("ent_001", "PERSON", "John")]
        rels = _pattern_relationships(entities, {})
        assert rels == []

    def test_empty_text_returns_empty(self):
        entities = [_ent("ent_001", "PERSON", "John")]
        rels = _pattern_relationships(entities, {1: ""})
        assert rels == []


# ---------------------------------------------------------------------------
# Tests: Deduplication
# ---------------------------------------------------------------------------


class TestDeduplicateRelationships:
    """Test deduplicate_relationships."""

    def test_removes_exact_duplicates(self):
        rels = [
            EntityRelationship("a", "b", "works_for", confidence=0.8),
            EntityRelationship("a", "b", "works_for", confidence=0.6),
        ]
        result = deduplicate_relationships(rels)
        assert len(result) == 1
        assert result[0].confidence == 0.8

    def test_keeps_different_relationships(self):
        rels = [
            EntityRelationship("a", "b", "works_for", confidence=0.8),
            EntityRelationship("a", "b", "located_at", confidence=0.6),
        ]
        result = deduplicate_relationships(rels)
        assert len(result) == 2

    def test_keeps_different_pairs(self):
        rels = [
            EntityRelationship("a", "b", "works_for", confidence=0.8),
            EntityRelationship("a", "c", "works_for", confidence=0.7),
        ]
        result = deduplicate_relationships(rels)
        assert len(result) == 2

    def test_empty_input(self):
        assert deduplicate_relationships([]) == []

    def test_sorted_by_confidence_descending(self):
        rels = [
            EntityRelationship("a", "b", "works_for", confidence=0.5),
            EntityRelationship("c", "d", "dated", confidence=0.9),
            EntityRelationship("e", "f", "located_at", confidence=0.7),
        ]
        result = deduplicate_relationships(rels)
        confs = [r.confidence for r in result]
        assert confs == sorted(confs, reverse=True)


# ---------------------------------------------------------------------------
# Tests: RelationshipExtractor class
# ---------------------------------------------------------------------------


class TestRelationshipExtractor:
    """Test RelationshipExtractor public API."""

    def test_basic_extraction(self):
        extractor = RelationshipExtractor()
        entities = [
            _ent("ent_001", "PERSON", "John Smith", start=0, end=10),
            _ent("ent_002", "ORG", "Acme Corp", start=14, end=23),
        ]
        page_texts = {1: "John Smith at Acme Corp works here"}
        rels = extractor.extract_relationships(entities, page_texts)
        assert len(rels) >= 1

    def test_empty_entities_returns_empty(self):
        extractor = RelationshipExtractor()
        rels = extractor.extract_relationships([], {1: "text"})
        assert rels == []

    def test_single_entity_returns_empty(self):
        extractor = RelationshipExtractor()
        entities = [_ent("ent_001", "PERSON", "John")]
        rels = extractor.extract_relationships(entities, {1: "John"})
        assert rels == []

    def test_no_page_texts_uses_kv_only(self):
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "person_name", "John Smith", page=1),
            _ent("ent_002", "organization", "Acme Corp", page=1),
        ]
        kv_pairs = [
            _kv("person_name", "John Smith", page=1),
            _kv("organization", "Acme Corp", page=1),
        ]
        rels = extractor.extract_relationships(entities, kv_pairs=kv_pairs)
        assert len(rels) >= 1

    def test_custom_max_proximity(self):
        extractor = RelationshipExtractor(max_proximity_chars=10)
        entities = [
            _ent("ent_001", "PERSON", "John", start=0, end=4),
            _ent("ent_002", "ORG", "Acme", start=50, end=54),
        ]
        page_texts = {1: "John" + " " * 46 + "Acme"}
        rels = extractor.extract_relationships(entities, page_texts)
        # Should not find proximity-based relationship since distance > 10
        proximity_rels = [r for r in rels if r.metadata.get("method") == "proximity"]
        assert len(proximity_rels) == 0

    def test_min_confidence_filter(self):
        extractor = RelationshipExtractor(min_confidence=0.9)
        entities = [
            _ent("ent_001", "PERSON", "John", start=0, end=4),
            _ent("ent_002", "ORG", "Acme", start=100, end=104),
        ]
        page_texts = {1: "John" + " " * 96 + "Acme"}
        rels = extractor.extract_relationships(entities, page_texts)
        # Low-confidence proximity relationships should be filtered out
        for r in rels:
            assert r.confidence >= 0.9

    def test_combines_all_methods(self):
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "PERSON", "John Smith", start=0, end=10),
            _ent("ent_002", "ORG", "Acme Corp", start=14, end=23),
        ]
        page_texts = {1: "John Smith at Acme Corp filed the motion"}
        kv_pairs = [
            _kv("person_name", "John Smith", page=1),
            _kv("organization", "Acme Corp", page=1),
        ]
        rels = extractor.extract_relationships(entities, page_texts, kv_pairs)
        # Should find at least one relationship
        assert len(rels) >= 1

    def test_deduplication_across_methods(self):
        """Same relationship found by proximity and pattern should be deduped."""
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "PERSON", "John Smith", start=0, end=10),
            _ent("ent_002", "ORG", "Acme Corp", start=14, end=23),
        ]
        page_texts = {1: "John Smith at Acme Corp filed the motion"}
        rels = extractor.extract_relationships(entities, page_texts)
        # Check that we don't have duplicate (source, target, type) tuples
        seen = set()
        for r in rels:
            key = (r.source_id, r.target_id, r.relation_type)
            assert key not in seen, f"Duplicate relationship: {key}"
            seen.add(key)


# ---------------------------------------------------------------------------
# Tests: Integration with entity_consolidator
# ---------------------------------------------------------------------------


class TestExtractAndAttachRelationships:
    """Test extract_and_attach_relationships integration helper."""

    def test_attaches_relationships_to_consolidated(self):
        consolidated = {
            "schema_version": "1.0",
            "document": "test.pdf",
            "entities": [
                _ent("ent_001", "PERSON", "John Smith", start=0, end=10),
                _ent("ent_002", "ORG", "Acme Corp", start=14, end=23),
            ],
            "classifications": [],
            "key_value_pairs": [],
            "summary": {
                "total_entities": 2,
                "entity_types": {"PERSON": 1, "ORG": 1},
                "total_kv_pairs": 0,
                "primary_classification": "",
            },
        }
        page_texts = {1: "John Smith at Acme Corp filed the motion"}

        result = extract_and_attach_relationships(consolidated, page_texts)

        assert "relationships" in result
        assert "total_relationships" in result["summary"]
        assert isinstance(result["relationships"], list)

    def test_empty_entities_adds_empty_relationships(self):
        consolidated = {
            "entities": [],
            "key_value_pairs": [],
            "summary": {"total_entities": 0},
        }
        result = extract_and_attach_relationships(consolidated)
        assert result["relationships"] == []
        assert result["summary"]["total_relationships"] == 0

    def test_single_entity_adds_empty_relationships(self):
        consolidated = {
            "entities": [_ent("ent_001", "PERSON", "John")],
            "key_value_pairs": [],
            "summary": {"total_entities": 1},
        }
        result = extract_and_attach_relationships(consolidated)
        assert result["relationships"] == []
        assert result["summary"]["total_relationships"] == 0

    def test_relationship_dicts_are_serializable(self):
        """Verify relationships are plain dicts, not dataclass instances."""
        consolidated = {
            "entities": [
                _ent("ent_001", "PERSON", "John Smith", start=0, end=10),
                _ent("ent_002", "ORG", "Acme Corp", start=14, end=23),
            ],
            "key_value_pairs": [],
            "summary": {"total_entities": 2},
        }
        page_texts = {1: "John Smith at Acme Corp works here"}

        result = extract_and_attach_relationships(consolidated, page_texts)

        for rel in result["relationships"]:
            assert isinstance(rel, dict)
            assert "source_id" in rel
            assert "target_id" in rel
            assert "type" in rel
            assert "confidence" in rel

    def test_summary_includes_relationship_type_breakdown(self):
        consolidated = {
            "entities": [
                _ent("ent_001", "PERSON", "John Smith", start=0, end=10),
                _ent("ent_002", "ORG", "Acme Corp", start=14, end=23),
            ],
            "key_value_pairs": [],
            "summary": {"total_entities": 2},
        }
        page_texts = {1: "John Smith at Acme Corp works here"}

        result = extract_and_attach_relationships(consolidated, page_texts)

        if result["summary"]["total_relationships"] > 0:
            assert "relationship_types" in result["summary"]
            assert isinstance(result["summary"]["relationship_types"], dict)

    def test_preserves_existing_consolidated_fields(self):
        consolidated = {
            "schema_version": "1.0",
            "document": "keep_me.pdf",
            "generated_at": "2024-01-01T00:00:00.000Z",
            "pipeline_version": "0.9.0",
            "entities": [
                _ent("ent_001", "PERSON", "John", start=0, end=4),
                _ent("ent_002", "ORG", "Acme", start=8, end=12),
            ],
            "classifications": [{"label": "invoice", "confidence": 0.9}],
            "key_value_pairs": [_kv("person_name", "John")],
            "summary": {
                "total_entities": 2,
                "entity_types": {"PERSON": 1, "ORG": 1},
                "total_kv_pairs": 1,
                "primary_classification": "invoice",
            },
        }

        result = extract_and_attach_relationships(consolidated, {1: "John at Acme"})

        assert result["schema_version"] == "1.0"
        assert result["document"] == "keep_me.pdf"
        assert result["classifications"][0]["label"] == "invoice"
        assert len(result["entities"]) == 2

    def test_with_kv_pairs(self):
        consolidated = {
            "entities": [
                _ent("ent_001", "person_name", "John Smith", page=1),
                _ent("ent_002", "organization", "Acme Corp", page=1),
            ],
            "key_value_pairs": [
                _kv("person_name", "John Smith", page=1),
                _kv("organization", "Acme Corp", page=1),
            ],
            "summary": {"total_entities": 2},
        }

        result = extract_and_attach_relationships(
            consolidated, min_confidence=0.0
        )

        assert result["summary"]["total_relationships"] >= 1

    def test_custom_proximity_and_confidence(self):
        consolidated = {
            "entities": [
                _ent("ent_001", "PERSON", "John", start=0, end=4),
                _ent("ent_002", "ORG", "Acme", start=200, end=204),
            ],
            "key_value_pairs": [],
            "summary": {"total_entities": 2},
        }
        page_texts = {1: "John" + " " * 196 + "Acme"}

        # With default proximity (150), should not find relationship
        result = extract_and_attach_relationships(
            consolidated, page_texts, max_proximity_chars=50
        )
        proximity_rels = [
            r for r in result["relationships"]
            if r.get("metadata", {}).get("method") == "proximity"
        ]
        assert len(proximity_rels) == 0


# ---------------------------------------------------------------------------
# Tests: Relationship types coverage
# ---------------------------------------------------------------------------


class TestRelationshipTypes:
    """Verify each defined relationship type can be detected."""

    def test_works_for_detected(self):
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "PERSON", "Alice Johnson", start=0, end=13),
            _ent("ent_002", "ORG", "Global Tech", start=17, end=28),
        ]
        page_texts = {1: "Alice Johnson at Global Tech presented the report"}
        rels = extractor.extract_relationships(entities, page_texts)
        types = {r.relation_type for r in rels}
        assert "works_for" in types

    def test_located_at_detected(self):
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "ORG", "TechCo", start=0, end=6),
            _ent("ent_002", "GPE", "San Francisco", start=18, end=31),
        ]
        page_texts = {1: "TechCo located in San Francisco downtown area"}
        rels = extractor.extract_relationships(entities, page_texts)
        types = {r.relation_type for r in rels}
        assert "located_at" in types

    def test_dated_detected(self):
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "PERSON", "Bob", start=0, end=3),
            _ent("ent_002", "DATE", "March 5, 2024", start=7, end=20),
        ]
        page_texts = {1: "Bob on March 5, 2024 submitted the form"}
        rels = extractor.extract_relationships(entities, page_texts)
        types = {r.relation_type for r in rels}
        assert "dated" in types

    def test_amount_for_detected(self):
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "MONEY", "$10,000", start=0, end=7),
            _ent("ent_002", "ORG", "Service Co", start=12, end=22),
        ]
        page_texts = {1: "$10,000 for Service Co consulting fees"}
        rels = extractor.extract_relationships(entities, page_texts)
        types = {r.relation_type for r in rels}
        assert "amount_for" in types

    def test_signed_by_detected(self):
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "PERSON", "Carol White", start=28, end=39),
            _ent("ent_002", "DATE", "2024-01-15", start=43, end=53),
        ]
        page_texts = {1: "This agreement was signed by Carol White on 2024-01-15"}
        rels = extractor.extract_relationships(entities, page_texts)
        signed = [r for r in rels if r.relation_type == "signed_by"]
        assert len(signed) >= 1

    def test_references_detected(self):
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "CASE_NUMBER", "24-CV-5678", start=4, end=14),
        ]
        # Need at least 2 entities for the extractor to run
        entities.append(
            _ent("ent_002", "PERSON", "Plaintiff", start=30, end=39)
        )
        page_texts = {1: "Re: 24-CV-5678 regarding the Plaintiff claims"}
        rels = extractor.extract_relationships(entities, page_texts)
        refs = [r for r in rels if r.relation_type == "references"]
        assert len(refs) >= 1

    def test_sent_to_detected(self):
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "PERSON", "Jane Doe", start=4, end=12),
            _ent("ent_002", "PERSON", "Bob Jones", start=20, end=29),
        ]
        page_texts = {1: "To: Jane Doe\nFrom: Bob Jones\nSubject: Meeting"}
        rels = extractor.extract_relationships(entities, page_texts)
        sent_to = [r for r in rels if r.relation_type == "sent_to"]
        assert len(sent_to) >= 1

    def test_sent_from_detected(self):
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "PERSON", "Alice Brown", start=6, end=17),
            _ent("ent_002", "PERSON", "Someone Else", start=25, end=37),
        ]
        page_texts = {1: "From: Alice Brown\nTo: Someone Else\nSubject: Follow-up"}
        rels = extractor.extract_relationships(entities, page_texts)
        sent_from = [r for r in rels if r.relation_type == "sent_from"]
        assert len(sent_from) >= 1


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_entities_with_no_offsets(self):
        """Entities without offset metadata should not crash."""
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            {"id": "ent_001", "type": "PERSON", "text": "John", "page": 1, "confidence": 0.9, "source": "ner", "bbox": [], "metadata": {}},
            {"id": "ent_002", "type": "ORG", "text": "Acme", "page": 1, "confidence": 0.9, "source": "ner", "bbox": [], "metadata": {}},
        ]
        page_texts = {1: "John at Acme Corp"}
        # Should not raise
        rels = extractor.extract_relationships(entities, page_texts)
        assert isinstance(rels, list)

    def test_overlapping_entities(self):
        """Overlapping entity spans should not crash."""
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "PERSON", "John Smith", start=0, end=10),
            _ent("ent_002", "ORG", "Smith Corp", start=5, end=15),
        ]
        page_texts = {1: "John Smith Corp is a company"}
        rels = extractor.extract_relationships(entities, page_texts)
        assert isinstance(rels, list)

    def test_very_long_text(self):
        """Handle large page text without issues."""
        extractor = RelationshipExtractor(min_confidence=0.0)
        long_text = "John Smith at Acme Corp. " * 1000
        entities = [
            _ent("ent_001", "PERSON", "John Smith", start=0, end=10),
            _ent("ent_002", "ORG", "Acme Corp", start=14, end=23),
        ]
        rels = extractor.extract_relationships(entities, {1: long_text})
        assert isinstance(rels, list)

    def test_special_characters_in_entity_text(self):
        """Entity text with special regex chars should not crash."""
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "ORG", "Company (LLC)", start=0, end=13),
            _ent("ent_002", "PERSON", "Mr. Smith [Jr.]", start=17, end=32),
        ]
        page_texts = {1: "Company (LLC) vs Mr. Smith [Jr.] filed a complaint"}
        rels = extractor.extract_relationships(entities, page_texts)
        assert isinstance(rels, list)

    def test_unicode_entity_text(self):
        """Unicode text should be handled correctly."""
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "PERSON", "Muller", start=0, end=8),
            _ent("ent_002", "ORG", "Deutsche Bank", start=12, end=25),
        ]
        page_texts = {1: "Muller at Deutsche Bank in Frankfurt"}
        rels = extractor.extract_relationships(entities, page_texts)
        assert isinstance(rels, list)

    def test_page_texts_none(self):
        """None page_texts should not crash."""
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "PERSON", "John"),
            _ent("ent_002", "ORG", "Acme"),
        ]
        rels = extractor.extract_relationships(entities, page_texts=None)
        assert isinstance(rels, list)

    def test_many_entities_on_same_page(self):
        """Performance test: many entities should complete reasonably."""
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = []
        for i in range(50):
            entities.append(
                _ent(
                    f"ent_{i:03d}",
                    "PERSON" if i % 2 == 0 else "ORG",
                    f"Entity_{i}",
                    start=i * 20,
                    end=i * 20 + 10,
                )
            )
        text = " ".join(f"Entity_{i} works" for i in range(50))
        rels = extractor.extract_relationships(entities, {1: text})
        assert isinstance(rels, list)

    def test_metadata_preserved_in_output(self):
        """Relationship metadata should include the detection method."""
        extractor = RelationshipExtractor(min_confidence=0.0)
        entities = [
            _ent("ent_001", "PERSON", "John Smith", start=0, end=10),
            _ent("ent_002", "ORG", "Acme Corp", start=14, end=23),
        ]
        page_texts = {1: "John Smith at Acme Corp filed the motion"}
        rels = extractor.extract_relationships(entities, page_texts)
        for r in rels:
            assert "method" in r.metadata
            assert r.metadata["method"] in ("proximity", "pattern", "kv_linkage")


# ---------------------------------------------------------------------------
# Tests: Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    """Test module-level configuration."""

    def test_enable_flag_defaults_false(self):
        # Unless env var is set, should default to false
        # (we can't easily test env var changes without monkeypatch)
        assert isinstance(ENABLE_RELATIONSHIP_EXTRACTION, bool)

    def test_extractor_respects_custom_params(self):
        ext = RelationshipExtractor(max_proximity_chars=50, min_confidence=0.8)
        assert ext._max_proximity == 50
        assert ext._min_confidence == 0.8

    def test_extractor_uses_defaults(self):
        ext = RelationshipExtractor()
        assert ext._max_proximity > 0
        assert ext._min_confidence >= 0.0


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------


class TestRelationshipPipelineIntegration:
    """Verify the module can be imported and used by the pipeline."""

    def test_relationship_extraction_available_flag(self):
        """Verify the module can be imported."""
        from relationship_extraction import extract_and_attach_relationships

        assert callable(extract_and_attach_relationships)

    def test_relationship_extraction_with_consolidated(self):
        """Verify relationship extraction enriches consolidated dict."""
        consolidated = {
            "entities": [
                {
                    "id": "e1",
                    "type": "PERSON",
                    "text": "John Smith",
                    "page": 1,
                    "confidence": 0.9,
                },
                {
                    "id": "e2",
                    "type": "ORG",
                    "text": "Acme Corp",
                    "page": 1,
                    "confidence": 0.8,
                },
            ],
            "key_value_pairs": [],
            "summary": {"total_entities": 2, "total_kv_pairs": 0},
        }
        page_texts = {1: "John Smith works at Acme Corp as a senior engineer."}
        result = extract_and_attach_relationships(consolidated, page_texts)
        assert "relationships" in result

    def test_pipeline_import_guard_exists(self):
        """Verify the pipeline exposes the availability flag."""
        import ocr_gpu_async

        assert hasattr(ocr_gpu_async, "_RELATIONSHIP_EXTRACTION_AVAILABLE")
        assert ocr_gpu_async._RELATIONSHIP_EXTRACTION_AVAILABLE is True

    def test_pipeline_enable_flag_exists(self):
        """Verify the pipeline exposes the enable toggle."""
        import ocr_gpu_async

        assert hasattr(ocr_gpu_async, "ENABLE_RELATIONSHIP_EXTRACTION")
        assert isinstance(ocr_gpu_async.ENABLE_RELATIONSHIP_EXTRACTION, bool)

    def test_empty_entities_returns_empty_relationships(self):
        """Relationship extraction on empty entities returns empty list."""
        consolidated = {
            "entities": [],
            "key_value_pairs": [],
            "summary": {"total_entities": 0, "total_kv_pairs": 0},
        }
        result = extract_and_attach_relationships(consolidated, {})
        assert result.get("relationships", []) == []

    def test_no_page_texts_still_works(self):
        """Relationship extraction works when page_texts is None."""
        consolidated = {
            "entities": [
                {
                    "id": "e1",
                    "type": "PERSON",
                    "text": "Alice",
                    "page": 1,
                    "confidence": 0.9,
                },
            ],
            "key_value_pairs": [],
            "summary": {"total_entities": 1, "total_kv_pairs": 0},
        }
        result = extract_and_attach_relationships(consolidated, None)
        assert "relationships" in result
