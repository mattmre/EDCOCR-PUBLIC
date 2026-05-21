"""Tests for durable .entities.json sidecar output."""

import json
import os

from extraction import DocumentExtraction, PageExtraction, finalize_extraction
from semantic_extraction import finalize_entity_output, write_entities_json
from version import __version__

PIPELINE_VERSION = f"{__version__}-test"


def _make_finalized_doc(source_file: str = "test_doc.pdf") -> DocumentExtraction:
    """Create a finalized extraction document with deterministic field layout."""
    doc = DocumentExtraction(document_id="doc-entities", source_file=source_file)
    doc.pages = [
        PageExtraction(
            page_num=1,
            fields=[
                {
                    "field_type": "reference_number",
                    "text": "INV-2024-001",
                    "confidence": 0.98,
                    "page_num": 1,
                    "start": 0,
                    "end": 12,
                    "extraction_method": "uie",
                    "normalized_value": "",
                },
                {
                    "field_type": "date",
                    "text": "2024-01-15",
                    "confidence": 0.97,
                    "page_num": 1,
                    "start": 20,
                    "end": 30,
                    "extraction_method": "regex",
                    "normalized_value": "2024-01-15",
                },
                {
                    "field_type": "amount",
                    "text": "$500.00",
                    "confidence": 0.96,
                    "page_num": 1,
                    "start": 40,
                    "end": 47,
                    "extraction_method": "regex",
                    "normalized_value": "",
                },
                {
                    "field_type": "person_name",
                    "text": "Jane Doe",
                    "confidence": 0.93,
                    "page_num": 1,
                    "start": 60,
                    "end": 68,
                    "extraction_method": "uie",
                    "normalized_value": "",
                },
                {
                    "field_type": "organization",
                    "text": "Acme Corp",
                    "confidence": 0.91,
                    "page_num": 1,
                    "start": 74,
                    "end": 83,
                    "extraction_method": "semantic",
                    "normalized_value": "",
                    "bbox": [10, 10, 100, 30],
                },
                {
                    "field_type": "email_address",
                    "text": "billing@acme.test",
                    "confidence": 0.9,
                    "page_num": 1,
                    "start": 90,
                    "end": 107,
                    "extraction_method": "regex",
                    "normalized_value": "",
                },
                {
                    "field_type": "phone_number",
                    "text": "555-010-9999",
                    "confidence": 0.89,
                    "page_num": 1,
                    "start": 112,
                    "end": 124,
                    "extraction_method": "regex",
                    "normalized_value": "",
                },
            ],
        ),
    ]
    return finalize_extraction(doc)


class TestFinalizeEntityOutput:
    """Test rule-based entity, relationship, and KV derivation."""

    def test_derives_relationships_and_key_value_pairs(self):
        doc = _make_finalized_doc()

        result = finalize_entity_output(doc)

        assert result.total_entities == 7
        assert result.entity_type_counts["reference_number"] == 1
        assert result.entity_type_counts["organization"] == 1
        assert result.total_relationships == 5
        assert result.total_key_value_pairs == 5

        page = result.pages[0]
        relationship_types = {
            item["relationship_type"] for item in page["relationships"]
        }
        assert relationship_types == {
            "contact_value",
            "document_amount",
            "document_date",
            "party_association",
        }

        pair_types = {item["pair_type"] for item in page["key_value_pairs"]}
        assert pair_types == {
            "organization_to_email_address",
            "organization_to_phone_number",
            "person_name_to_organization",
            "reference_number_to_amount",
            "reference_number_to_date",
        }

        org_entity = next(
            entity for entity in page["entities"] if entity["field_type"] == "organization"
        )
        assert org_entity["bbox"] == [10, 10, 100, 30]

    def test_unique_entity_summary_tracks_occurrences(self):
        doc = _make_finalized_doc()
        doc.pages.append(
            PageExtraction(
                page_num=2,
                fields=[
                    {
                        "field_type": "organization",
                        "text": "Acme Corp",
                        "confidence": 0.88,
                        "page_num": 2,
                        "start": 10,
                        "end": 19,
                        "extraction_method": "semantic",
                        "normalized_value": "",
                    },
                ],
            )
        )
        finalize_extraction(doc)

        result = finalize_entity_output(doc)

        org_summary = next(
            item
            for item in result.unique_entities
            if item["field_type"] == "organization" and item["text"] == "Acme Corp"
        )
        assert org_summary["occurrences"] == 2
        assert org_summary["pages"] == [1, 2]


class TestWriteEntitiesJson:
    """Test durable .entities.json writer behavior."""

    def test_json_file_created_with_expected_schema(self, tmp_path):
        doc = _make_finalized_doc()
        entities_doc = finalize_entity_output(doc)

        result_path = write_entities_json(
            entities_doc,
            str(tmp_path),
            "batch_001",
            PIPELINE_VERSION,
        )

        assert result_path is not None
        assert os.path.exists(result_path)
        assert result_path.endswith(".entities.json")

        with open(result_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        assert data["schema_version"] == "1.0"
        assert data["document_id"] == "doc-entities"
        assert data["processing"]["relationship_engine"] == "rule_based"
        assert data["entities_summary"]["total_entities"] == 7
        assert data["entities_summary"]["total_relationships"] == 5
        assert data["entities_summary"]["total_key_value_pairs"] == 5
        assert len(data["document_entities"]) == 7
        assert len(data["pages"]) == 1
        assert len(data["pages"][0]["relationships"]) == 5
        assert len(data["pages"][0]["key_value_pairs"]) == 5

    def test_non_pdf_uses_ext_token(self, tmp_path):
        doc = _make_finalized_doc(source_file="images/photo.jpg")
        entities_doc = finalize_entity_output(doc)

        result_path = write_entities_json(
            entities_doc,
            str(tmp_path),
            "",
            PIPELINE_VERSION,
        )

        assert result_path is not None
        assert os.path.basename(result_path) == "photo__jpg.entities.json"

    def test_path_traversal_is_contained(self, tmp_path):
        doc = _make_finalized_doc()
        entities_doc = finalize_entity_output(doc)

        result_path = write_entities_json(
            entities_doc,
            str(tmp_path),
            "../../etc",
            PIPELINE_VERSION,
        )

        assert result_path is not None
        entities_root = os.path.join(str(tmp_path), "EXPORT", "ENTITIES")
        resolved = os.path.realpath(result_path)
        assert resolved.startswith(os.path.realpath(entities_root))
