"""Unit tests for specialist routing sidecar generation."""

import json

from classification import DocumentClassification
from routing import derive_document_routing, write_routing_json
from semantic_extraction import DocumentEntityOutput
from version import __version__


def test_profile_route_wins_priority():
    doc_cls = DocumentClassification(document_id="doc1", source_file="invoice.pdf")
    doc_cls.document_type = "invoice"
    doc_cls.document_confidence = 0.82
    doc_cls.document_labels = [
        {"label": "invoice", "confidence": 0.82, "source": "classification"},
        {"label": "receipt", "confidence": 0.41, "source": "classification"},
    ]
    doc_cls.custom_profile_matches = [
        {
            "name": "invoice_packet",
            "base_type": "invoice",
            "route": "finance_invoice_review",
            "confidence": 0.91,
            "occurrences": 1,
        }
    ]
    doc_entities = DocumentEntityOutput(document_id="doc1", source_file="invoice.pdf")
    doc_entities.entity_type_counts = {"amount": 2, "reference_number": 1}

    result = derive_document_routing(doc_cls, doc_entities)

    assert result.primary_route == "finance_invoice_review"
    assert result.routing_mode == "classification+entities"
    assert result.recommended_routes[0]["source"] == "profile+builtin"
    assert len(result.recommended_routes) == 1


def test_general_review_when_no_specialist_match():
    doc_cls = DocumentClassification(document_id="doc2", source_file="note.pdf")
    doc_cls.document_type = "other"
    doc_cls.document_confidence = 0.33
    doc_cls.document_labels = [
        {"label": "other", "confidence": 0.33, "source": "classification"}
    ]

    result = derive_document_routing(doc_cls)

    assert result.primary_route == "general_review"
    assert result.review_required is True
    assert "no_specialist_match" in result.review_reasons
    assert "entity_output_unavailable" in result.review_reasons


def test_write_routing_json(tmp_path):
    doc_cls = DocumentClassification(document_id="doc3", source_file="contract.pdf")
    doc_cls.document_type = "contract"
    doc_cls.document_confidence = 0.77
    doc_cls.document_labels = [
        {"label": "contract", "confidence": 0.77, "source": "classification"},
        {"label": "letter", "confidence": 0.7, "source": "classification"},
    ]
    doc_entities = DocumentEntityOutput(document_id="doc3", source_file="contract.pdf")
    doc_entities.entity_type_counts = {"person_name": 2, "date": 1}

    routing_doc = derive_document_routing(doc_cls, doc_entities)
    output_path = write_routing_json(routing_doc, str(tmp_path), ".", __version__)

    assert output_path is not None
    with open(output_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["routing_summary"]["primary_route"] == routing_doc.primary_route
    assert payload["recommended_routes"]
