"""Tests for LayoutLMv3 structure.json integration module.

Covers the integrator class, ensemble merging, weighted confidence
computation, and structure.json output format. No torch/transformers
dependencies required.

Run with: python -m pytest tests/test_layoutlm_structure.py -v
"""


import pytest

# Add project root to path
from layoutlm_structure import (
    LAYOUTLM_ENSEMBLE_WEIGHT,
    IntegratedPageResult,
    LayoutLMv3StructureIntegrator,
    SemanticEntityRecord,
    _boxes_overlap,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def integrator():
    """Default integrator with 0.7 ensemble weight."""
    return LayoutLMv3StructureIntegrator(ensemble_weight=0.7)


@pytest.fixture
def sample_layoutlm_entities():
    """Sample LayoutLMv3 entity dicts (as produced by Celery task or extractor)."""
    return [
        {
            "text": "Invoice #12345",
            "label": "INVOICE_NUMBER",
            "confidence": 0.95,
            "bbox": [100, 50, 300, 80],
            "page_num": 1,
        },
        {
            "text": "2026-03-14",
            "label": "DATE",
            "confidence": 0.92,
            "bbox": [400, 50, 550, 80],
            "page_num": 1,
        },
        {
            "text": "$5,000.00",
            "label": "AMOUNT",
            "confidence": 0.88,
            "bbox": [100, 200, 250, 230],
            "page_num": 1,
        },
    ]


@pytest.fixture
def sample_ppstructure_results():
    """Sample PP-StructureV3 output dict (from parse_structure_result)."""
    return {
        "layout_regions": [
            {
                "type": "title",
                "bbox": [50, 30, 600, 90],
                "confidence": 0.85,
                "text": "INVOICE",
            },
            {
                "type": "text",
                "bbox": [50, 100, 600, 250],
                "confidence": 0.78,
                "text": "Payment details follow",
            },
            {
                "type": "table",
                "bbox": [50, 300, 600, 500],
                "confidence": 0.91,
                "table_index": 0,
            },
        ],
        "tables": [
            {
                "html": "<table><tr><td>Item</td><td>Amount</td></tr></table>",
                "cell_bbox": [[50, 300, 300, 350], [300, 300, 600, 350]],
            },
        ],
        "key_value_pairs": [
            {"key": "Invoice", "value": "#12345", "confidence": 0.80},
        ],
        "form_fields": [
            {"field_type": "text_input", "bbox": [100, 600, 400, 630]},
        ],
    }


# ---------------------------------------------------------------------------
# SemanticEntityRecord tests
# ---------------------------------------------------------------------------


class TestSemanticEntityRecord:
    def test_default_values(self):
        record = SemanticEntityRecord(text="hello", label="DATE")
        assert record.text == "hello"
        assert record.label == "DATE"
        assert record.bbox == []
        assert record.confidence == 0.0
        assert record.source == "layoutlmv3"
        assert record.page_num == 0

    def test_custom_values(self):
        record = SemanticEntityRecord(
            text="$100",
            label="AMOUNT",
            bbox=[10, 20, 30, 40],
            confidence=0.95,
            source="layoutlmv3",
            page_num=3,
        )
        assert record.text == "$100"
        assert record.label == "AMOUNT"
        assert record.bbox == [10, 20, 30, 40]
        assert record.confidence == 0.95
        assert record.page_num == 3


# ---------------------------------------------------------------------------
# IntegratedPageResult tests
# ---------------------------------------------------------------------------


class TestIntegratedPageResult:
    def test_default_values(self):
        result = IntegratedPageResult()
        assert result.page_number == 0
        assert result.layout_regions == []
        assert result.tables == []
        assert result.key_value_pairs == []
        assert result.form_fields == []
        assert result.semantic_entities == []
        assert result.ensemble_source == ""


# ---------------------------------------------------------------------------
# _boxes_overlap tests
# ---------------------------------------------------------------------------


class TestBoxesOverlap:
    def test_overlapping_boxes(self):
        assert _boxes_overlap([0, 0, 100, 100], [50, 50, 150, 150]) is True

    def test_non_overlapping_left_right(self):
        assert _boxes_overlap([0, 0, 50, 50], [60, 0, 110, 50]) is False

    def test_non_overlapping_top_bottom(self):
        assert _boxes_overlap([0, 0, 50, 50], [0, 60, 50, 110]) is False

    def test_adjacent_boxes_no_overlap(self):
        # Touching edges are not considered overlapping (strict inequality)
        assert _boxes_overlap([0, 0, 50, 50], [50, 0, 100, 50]) is False

    def test_contained_box(self):
        assert _boxes_overlap([0, 0, 200, 200], [50, 50, 100, 100]) is True

    def test_invalid_box_a(self):
        assert _boxes_overlap([0, 0], [0, 0, 50, 50]) is False

    def test_invalid_box_b(self):
        assert _boxes_overlap([0, 0, 50, 50], []) is False

    def test_both_invalid(self):
        assert _boxes_overlap([], []) is False

    def test_partial_overlap(self):
        assert _boxes_overlap([0, 0, 100, 100], [90, 90, 200, 200]) is True


# ---------------------------------------------------------------------------
# Integrator: LayoutLMv3-only tests
# ---------------------------------------------------------------------------


class TestIntegrateLayoutLMOnly:
    def test_entities_become_semantic_entities(
        self, integrator, sample_layoutlm_entities
    ):
        result = integrator.integrate(
            layoutlm_results=sample_layoutlm_entities,
        )
        assert result.ensemble_source == "layoutlmv3"
        assert len(result.semantic_entities) == 3
        assert result.layout_regions == []
        assert result.tables == []

    def test_entity_fields_preserved(
        self, integrator, sample_layoutlm_entities
    ):
        result = integrator.integrate(
            layoutlm_results=sample_layoutlm_entities,
        )
        first = result.semantic_entities[0]
        assert isinstance(first, SemanticEntityRecord)
        assert first.text == "Invoice #12345"
        assert first.label == "INVOICE_NUMBER"
        assert first.confidence == 0.95
        assert first.bbox == [100, 50, 300, 80]
        assert first.source == "layoutlmv3"

    def test_empty_layoutlm_results(self, integrator):
        result = integrator.integrate(layoutlm_results=[])
        assert result.ensemble_source == "none"
        assert result.semantic_entities == []


# ---------------------------------------------------------------------------
# Integrator: PP-StructureV3-only tests
# ---------------------------------------------------------------------------


class TestIntegratePPStructureOnly:
    def test_standard_structure_preserved(
        self, integrator, sample_ppstructure_results
    ):
        result = integrator.integrate(
            ppstructure_results=sample_ppstructure_results,
        )
        assert result.ensemble_source == "ppstructure"
        assert len(result.layout_regions) == 3
        assert len(result.tables) == 1
        assert len(result.key_value_pairs) == 1
        assert len(result.form_fields) == 1
        assert result.semantic_entities == []

    def test_empty_ppstructure_results(self, integrator):
        result = integrator.integrate(ppstructure_results={})
        assert result.ensemble_source == "none"
        assert result.layout_regions == []


# ---------------------------------------------------------------------------
# Integrator: ensemble merging tests
# ---------------------------------------------------------------------------


class TestEnsembleMerging:
    def test_ensemble_source_label(
        self,
        integrator,
        sample_layoutlm_entities,
        sample_ppstructure_results,
    ):
        result = integrator.integrate(
            layoutlm_results=sample_layoutlm_entities,
            ppstructure_results=sample_ppstructure_results,
        )
        assert result.ensemble_source == "layoutlmv3+ppstructure"

    def test_semantic_entities_present_in_ensemble(
        self,
        integrator,
        sample_layoutlm_entities,
        sample_ppstructure_results,
    ):
        result = integrator.integrate(
            layoutlm_results=sample_layoutlm_entities,
            ppstructure_results=sample_ppstructure_results,
        )
        assert len(result.semantic_entities) == 3

    def test_tables_preserved_in_ensemble(
        self,
        integrator,
        sample_layoutlm_entities,
        sample_ppstructure_results,
    ):
        result = integrator.integrate(
            layoutlm_results=sample_layoutlm_entities,
            ppstructure_results=sample_ppstructure_results,
        )
        assert len(result.tables) == 1
        assert "html" in result.tables[0]

    def test_weighted_confidence_merge(self, integrator):
        """When a LayoutLMv3 entity overlaps a PP-StructureV3 region,
        confidence should be the weighted average."""
        layoutlm = [
            {
                "text": "header text",
                "label": "ORGANIZATION",
                "confidence": 0.90,
                "bbox": [60, 40, 200, 85],
                "page_num": 1,
            },
        ]
        ppstructure = {
            "layout_regions": [
                {
                    "type": "title",
                    "bbox": [50, 30, 600, 90],
                    "confidence": 0.80,
                },
            ],
            "tables": [],
            "key_value_pairs": [],
            "form_fields": [],
        }
        result = integrator.integrate(
            layoutlm_results=layoutlm,
            ppstructure_results=ppstructure,
        )
        region = result.layout_regions[0]
        # Expected: 0.7 * 0.90 + 0.3 * 0.80 = 0.63 + 0.24 = 0.87
        expected = round(0.7 * 0.90 + 0.3 * 0.80, 4)
        assert region["confidence"] == expected
        assert region["ensemble_source"] == "layoutlmv3+ppstructure"

    def test_non_overlapping_region_unchanged(self, integrator):
        """Regions with no overlapping entities keep original confidence."""
        layoutlm = [
            {
                "text": "far away",
                "label": "ADDRESS",
                "confidence": 0.95,
                "bbox": [700, 700, 800, 800],
                "page_num": 1,
            },
        ]
        ppstructure = {
            "layout_regions": [
                {
                    "type": "text",
                    "bbox": [0, 0, 100, 100],
                    "confidence": 0.75,
                },
            ],
            "tables": [],
            "key_value_pairs": [],
            "form_fields": [],
        }
        result = integrator.integrate(
            layoutlm_results=layoutlm,
            ppstructure_results=ppstructure,
        )
        region = result.layout_regions[0]
        assert region["confidence"] == 0.75
        assert region["ensemble_source"] == "ppstructure"
        assert "layoutlm_entities" not in region

    def test_multiple_entities_overlap_one_region(self, integrator):
        """When multiple entities overlap a region, highest entity confidence wins."""
        layoutlm = [
            {
                "text": "Invoice #12345",
                "label": "INVOICE_NUMBER",
                "confidence": 0.80,
                "bbox": [60, 40, 200, 85],
                "page_num": 1,
            },
            {
                "text": "2026-03-14",
                "label": "DATE",
                "confidence": 0.95,
                "bbox": [300, 40, 500, 85],
                "page_num": 1,
            },
        ]
        ppstructure = {
            "layout_regions": [
                {
                    "type": "title",
                    "bbox": [50, 30, 600, 90],
                    "confidence": 0.70,
                },
            ],
            "tables": [],
            "key_value_pairs": [],
            "form_fields": [],
        }
        result = integrator.integrate(
            layoutlm_results=layoutlm,
            ppstructure_results=ppstructure,
        )
        region = result.layout_regions[0]
        # max entity conf = 0.95
        # Expected: 0.7 * 0.95 + 0.3 * 0.70 = 0.665 + 0.21 = 0.875
        expected = round(0.7 * 0.95 + 0.3 * 0.70, 4)
        assert region["confidence"] == expected
        assert len(region["layoutlm_entities"]) == 2


# ---------------------------------------------------------------------------
# Configurable ensemble weight tests
# ---------------------------------------------------------------------------


class TestConfigurableEnsembleWeight:
    def test_custom_weight(self):
        integrator = LayoutLMv3StructureIntegrator(ensemble_weight=0.5)
        assert integrator.ensemble_weight == 0.5

    def test_weight_clamp_above_one(self):
        integrator = LayoutLMv3StructureIntegrator(ensemble_weight=1.5)
        assert integrator.ensemble_weight == 1.0

    def test_weight_clamp_below_zero(self):
        integrator = LayoutLMv3StructureIntegrator(ensemble_weight=-0.3)
        assert integrator.ensemble_weight == 0.0

    def test_zero_weight_uses_ppstructure_only(self):
        integrator = LayoutLMv3StructureIntegrator(ensemble_weight=0.0)
        layoutlm = [
            {
                "text": "test",
                "label": "DATE",
                "confidence": 0.99,
                "bbox": [60, 40, 200, 85],
                "page_num": 1,
            },
        ]
        ppstructure = {
            "layout_regions": [
                {
                    "type": "title",
                    "bbox": [50, 30, 600, 90],
                    "confidence": 0.60,
                },
            ],
            "tables": [],
            "key_value_pairs": [],
            "form_fields": [],
        }
        result = integrator.integrate(
            layoutlm_results=layoutlm,
            ppstructure_results=ppstructure,
        )
        region = result.layout_regions[0]
        # 0.0 * 0.99 + 1.0 * 0.60 = 0.60
        assert region["confidence"] == 0.6

    def test_one_weight_uses_layoutlm_only(self):
        integrator = LayoutLMv3StructureIntegrator(ensemble_weight=1.0)
        layoutlm = [
            {
                "text": "test",
                "label": "DATE",
                "confidence": 0.99,
                "bbox": [60, 40, 200, 85],
                "page_num": 1,
            },
        ]
        ppstructure = {
            "layout_regions": [
                {
                    "type": "title",
                    "bbox": [50, 30, 600, 90],
                    "confidence": 0.60,
                },
            ],
            "tables": [],
            "key_value_pairs": [],
            "form_fields": [],
        }
        result = integrator.integrate(
            layoutlm_results=layoutlm,
            ppstructure_results=ppstructure,
        )
        region = result.layout_regions[0]
        # 1.0 * 0.99 + 0.0 * 0.60 = 0.99
        assert region["confidence"] == 0.99

    def test_default_weight_from_env(self):
        assert LAYOUTLM_ENSEMBLE_WEIGHT == 0.7


# ---------------------------------------------------------------------------
# to_structure_json tests
# ---------------------------------------------------------------------------


class TestToStructureJson:
    def test_basic_page_output(self, integrator, sample_layoutlm_entities):
        result = integrator.integrate(
            layoutlm_results=sample_layoutlm_entities,
        )
        page_dict = integrator.to_structure_json(result, page_number=1)
        assert page_dict["page_num"] == 1
        assert "semantic_entities" in page_dict
        assert len(page_dict["semantic_entities"]) == 3
        assert page_dict["ensemble_source"] == "layoutlmv3"

    def test_semantic_entity_schema(self, integrator, sample_layoutlm_entities):
        result = integrator.integrate(
            layoutlm_results=sample_layoutlm_entities,
        )
        page_dict = integrator.to_structure_json(result, page_number=1)
        entity = page_dict["semantic_entities"][0]
        assert "text" in entity
        assert "label" in entity
        assert "bbox" in entity
        assert "confidence" in entity
        assert "source" in entity
        assert entity["source"] == "layoutlmv3"

    def test_ppstructure_only_no_semantic_entities(
        self, integrator, sample_ppstructure_results
    ):
        result = integrator.integrate(
            ppstructure_results=sample_ppstructure_results,
        )
        page_dict = integrator.to_structure_json(result, page_number=1)
        assert "semantic_entities" not in page_dict
        assert page_dict["ensemble_source"] == "ppstructure"
        assert len(page_dict["layout_regions"]) == 3

    def test_ensemble_output_includes_both(
        self,
        integrator,
        sample_layoutlm_entities,
        sample_ppstructure_results,
    ):
        result = integrator.integrate(
            layoutlm_results=sample_layoutlm_entities,
            ppstructure_results=sample_ppstructure_results,
        )
        page_dict = integrator.to_structure_json(result, page_number=1)
        assert "semantic_entities" in page_dict
        assert len(page_dict["layout_regions"]) == 3
        assert page_dict["ensemble_source"] == "layoutlmv3+ppstructure"

    def test_form_fields_and_kv_pairs_preserved(
        self,
        integrator,
        sample_ppstructure_results,
    ):
        result = integrator.integrate(
            ppstructure_results=sample_ppstructure_results,
        )
        page_dict = integrator.to_structure_json(result, page_number=2)
        assert page_dict["page_num"] == 2
        assert len(page_dict["key_value_pairs"]) == 1
        assert len(page_dict["form_fields"]) == 1

    def test_empty_result_page(self, integrator):
        result = integrator.integrate()
        page_dict = integrator.to_structure_json(result, page_number=5)
        assert page_dict["page_num"] == 5
        assert page_dict["layout_regions"] == []
        assert page_dict["tables"] == []
        assert "semantic_entities" not in page_dict


# ---------------------------------------------------------------------------
# Empty / missing input edge cases
# ---------------------------------------------------------------------------


class TestEmptyAndMissingInputs:
    def test_all_none_inputs(self, integrator):
        result = integrator.integrate(
            ocr_results=None,
            layoutlm_results=None,
            ppstructure_results=None,
        )
        assert result.ensemble_source == "none"
        assert result.layout_regions == []
        assert result.semantic_entities == []
        assert result.tables == []

    def test_empty_lists(self, integrator):
        result = integrator.integrate(
            layoutlm_results=[],
            ppstructure_results={},
        )
        assert result.ensemble_source == "none"

    def test_ppstructure_with_missing_keys(self, integrator):
        result = integrator.integrate(
            ppstructure_results={"layout_regions": [{"type": "text", "bbox": [0, 0, 100, 100]}]},
        )
        assert len(result.layout_regions) == 1
        assert result.tables == []
        assert result.key_value_pairs == []
        assert result.form_fields == []

    def test_layoutlm_entity_missing_fields(self, integrator):
        """Entities with missing keys should still be processed."""
        entities = [
            {"text": "partial", "label": "DATE"},
        ]
        result = integrator.integrate(layoutlm_results=entities)
        assert len(result.semantic_entities) == 1
        assert result.semantic_entities[0].confidence == 0.0
        assert result.semantic_entities[0].bbox == []

    def test_ensemble_with_empty_regions(self, integrator):
        layoutlm = [
            {
                "text": "test",
                "label": "DATE",
                "confidence": 0.9,
                "bbox": [0, 0, 100, 100],
            },
        ]
        ppstructure = {
            "layout_regions": [],
            "tables": [],
            "key_value_pairs": [],
            "form_fields": [],
        }
        result = integrator.integrate(
            layoutlm_results=layoutlm,
            ppstructure_results=ppstructure,
        )
        # Both sources present but regions list is empty
        assert result.ensemble_source == "layoutlmv3+ppstructure"
        assert result.layout_regions == []
        assert len(result.semantic_entities) == 1


# ---------------------------------------------------------------------------
# Direct ensemble() method tests
# ---------------------------------------------------------------------------


class TestEnsembleMethod:
    def test_empty_regions_returns_empty(self, integrator):
        result = integrator.ensemble(
            layoutlm_entities=[{"text": "x", "label": "DATE", "confidence": 0.9, "bbox": [0, 0, 50, 50]}],
            ppstructure_regions=[],
        )
        assert result == []

    def test_empty_entities_returns_original(self, integrator):
        regions = [{"type": "text", "bbox": [0, 0, 100, 100], "confidence": 0.8}]
        result = integrator.ensemble(
            layoutlm_entities=[],
            ppstructure_regions=regions,
        )
        assert len(result) == 1
        assert result[0]["confidence"] == 0.8

    def test_regions_not_mutated_in_place(self, integrator):
        """Original region list should not be modified."""
        original_region = {"type": "text", "bbox": [0, 0, 200, 200], "confidence": 0.5}
        regions = [original_region]
        entities = [
            {"text": "x", "label": "DATE", "confidence": 0.9, "bbox": [10, 10, 100, 100]},
        ]
        integrator.ensemble(entities, regions)
        # Original region should be untouched
        assert original_region["confidence"] == 0.5
        assert "ensemble_source" not in original_region


# ---------------------------------------------------------------------------
# SemanticEntity dataclass compatibility tests
# ---------------------------------------------------------------------------


class TestDataclassEntityInput:
    def test_dataclass_entity_input(self, integrator):
        """Integrator should handle SemanticEntity-like objects, not just dicts."""

        class MockSemanticEntity:
            def __init__(self):
                self.text = "John Doe"
                self.label = "PERSON_NAME"
                self.confidence = 0.91
                self.bbox = [200, 300, 400, 330]
                self.page_num = 2

        entities = [MockSemanticEntity()]
        result = integrator.integrate(layoutlm_results=entities)
        assert len(result.semantic_entities) == 1
        record = result.semantic_entities[0]
        assert record.text == "John Doe"
        assert record.label == "PERSON_NAME"
        assert record.confidence == 0.91


# ---------------------------------------------------------------------------
# Confidence rounding tests
# ---------------------------------------------------------------------------


class TestConfidenceRounding:
    def test_confidence_rounded_to_four_decimals(self, integrator):
        layoutlm = [
            {
                "text": "test",
                "label": "DATE",
                "confidence": 0.123456789,
                "bbox": [60, 40, 200, 85],
                "page_num": 1,
            },
        ]
        result = integrator.integrate(layoutlm_results=layoutlm)
        page_dict = integrator.to_structure_json(result, page_number=1)
        entity = page_dict["semantic_entities"][0]
        # Confidence should be rounded to 4 decimal places
        assert entity["confidence"] == round(0.123456789, 4)

    def test_ensemble_confidence_rounded(self, integrator):
        layoutlm = [
            {
                "text": "test",
                "label": "DATE",
                "confidence": 0.876543,
                "bbox": [60, 40, 200, 85],
            },
        ]
        ppstructure = {
            "layout_regions": [
                {"type": "title", "bbox": [50, 30, 600, 90], "confidence": 0.654321},
            ],
            "tables": [],
            "key_value_pairs": [],
            "form_fields": [],
        }
        result = integrator.integrate(
            layoutlm_results=layoutlm,
            ppstructure_results=ppstructure,
        )
        region = result.layout_regions[0]
        # Should be rounded to 4 decimal places
        raw = 0.7 * 0.876543 + 0.3 * 0.654321
        assert region["confidence"] == round(raw, 4)
