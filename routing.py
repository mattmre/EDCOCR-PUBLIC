"""Specialist routing derived from classification and entity output.

Produces durable `.routing.json` sidecars that recommend downstream specialist
lanes without modifying the OCR pipeline's core processing path.
"""

import datetime
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from ocr_distributed.ocr_utils import (
    build_sidecar_base_name,
    sanitize_path_segment,
)

logger = logging.getLogger(__name__)


@dataclass
class RouteRecommendation:
    """A single route recommendation for one document."""

    route: str
    confidence: float
    priority: int
    rationale: str
    matched_signals: list = field(default_factory=list)
    source: str = "builtin"


@dataclass
class DocumentRouting:
    """Document-level routing summary."""

    document_id: str
    source_file: str
    routing_mode: str = "classification_only"
    primary_route: str = "general_review"
    recommended_routes: list = field(default_factory=list)
    matched_labels: list = field(default_factory=list)
    review_required: bool = False
    review_reasons: list = field(default_factory=list)


_BUILTIN_ROUTE_RULES = [
    {
        "name": "invoice_capture",
        "route": "finance_invoice_review",
        "priority": 90,
        "document_types": {"invoice", "receipt"},
        "entity_types": {"amount", "reference_number"},
        "rationale": "Financial document with monetary and reference signals",
    },
    {
        "name": "contract_analysis",
        "route": "contract_clause_review",
        "priority": 85,
        "document_types": {"contract", "legal_filing"},
        "entity_types": {"person_name", "date"},
        "rationale": "Contract-like document with named parties and dates",
    },
    {
        "name": "form_capture",
        "route": "form_field_capture",
        "priority": 80,
        "document_types": {"form", "government_form", "medical_record"},
        "entity_types": set(),
        "rationale": "Form-centric document suited for field capture workflows",
    },
    {
        "name": "contact_enrichment",
        "route": "contact_enrichment",
        "priority": 70,
        "document_types": {"letter", "memo", "report"},
        "entity_types": {"email_address", "phone_number"},
        "rationale": "Contact-bearing correspondence or reporting document",
    },
]


def _collect_classification_labels(doc_cls) -> set:
    labels = {getattr(doc_cls, "document_type", "other")}
    for item in getattr(doc_cls, "document_labels", []) or []:
        label = item.get("label")
        if label:
            labels.add(label)
    for item in getattr(doc_cls, "custom_profile_matches", []) or []:
        label = item.get("name")
        base_type = item.get("base_type")
        if label:
            labels.add(label)
        if base_type:
            labels.add(base_type)
    return {label for label in labels if label}


def _collect_entity_types(doc_entities) -> set:
    if doc_entities is None:
        return set()
    return {
        entity_type
        for entity_type in getattr(doc_entities, "entity_type_counts", {}).keys()
        if entity_type
    }


def _profile_recommendations(doc_cls) -> list:
    recommendations = []
    for match in getattr(doc_cls, "custom_profile_matches", []) or []:
        route_name = match.get("route", "").strip()
        if not route_name:
            continue
        recommendations.append(
            RouteRecommendation(
                route=route_name,
                confidence=round(float(match.get("confidence", 0.0) or 0.0), 4),
                priority=95,
                rationale=f"Matched customer profile {match.get('name', '')}",
                matched_signals=[
                    f"profile:{match.get('name', '')}",
                    f"base_type:{match.get('base_type', '')}",
                ],
                source="profile",
            )
        )
    return recommendations


def _dedupe_recommendations(recommendations: list) -> list:
    """Collapse duplicate routes while preserving the strongest recommendation."""
    deduped = {}
    for item in recommendations:
        existing = deduped.get(item.route)
        if existing is None:
            deduped[item.route] = item
            continue

        if (item.priority, item.confidence) > (existing.priority, existing.confidence):
            winner, loser = item, existing
        else:
            winner, loser = existing, item

        winner.matched_signals = sorted(
            set(winner.matched_signals) | set(loser.matched_signals)
        )
        if winner.source != loser.source:
            winner.source = "profile+builtin"
        deduped[item.route] = winner

    return list(deduped.values())


def derive_document_routing(doc_cls, doc_entities=None) -> DocumentRouting:
    """Derive specialist routing from classification and entity output."""
    routing = DocumentRouting(
        document_id=getattr(doc_cls, "document_id", ""),
        source_file=getattr(doc_cls, "source_file", ""),
        routing_mode="classification+entities" if doc_entities is not None else "classification_only",
    )

    matched_labels = sorted(_collect_classification_labels(doc_cls))
    entity_types = _collect_entity_types(doc_entities)
    recommendations = _profile_recommendations(doc_cls)

    for rule in _BUILTIN_ROUTE_RULES:
        if not (rule["document_types"] & set(matched_labels)):
            continue
        if rule["entity_types"] and not rule["entity_types"].issubset(entity_types):
            continue

        confidence = round(
            min(
                float(getattr(doc_cls, "document_confidence", 0.0) or 0.0) + 0.15,
                1.0,
            ),
            4,
        )
        matched_signals = sorted(
            [f"label:{label}" for label in matched_labels if label in rule["document_types"]]
            + [f"entity:{entity}" for entity in entity_types if entity in rule["entity_types"]]
        )
        recommendations.append(
            RouteRecommendation(
                route=rule["route"],
                confidence=confidence,
                priority=rule["priority"],
                rationale=rule["rationale"],
                matched_signals=matched_signals,
            )
        )

    if not recommendations:
        recommendations.append(
            RouteRecommendation(
                route="general_review",
                confidence=round(float(getattr(doc_cls, "document_confidence", 0.0) or 0.0), 4),
                priority=10,
                rationale="No specialist route matched; keep in general review",
                matched_signals=[f"label:{getattr(doc_cls, 'document_type', 'other')}"],
            )
        )
        routing.review_required = True
        routing.review_reasons.append("no_specialist_match")

    recommendations = _dedupe_recommendations(recommendations)
    recommendations.sort(key=lambda item: (-item.priority, -item.confidence, item.route))
    if len(getattr(doc_cls, "document_labels", []) or []) > 1:
        top_labels = getattr(doc_cls, "document_labels", [])
        if len(top_labels) > 1:
            delta = abs(top_labels[0]["confidence"] - top_labels[1]["confidence"])
            if delta < 0.15:
                routing.review_required = True
                routing.review_reasons.append("ambiguous_multilabel_document")

    primary = recommendations[0]
    routing.primary_route = primary.route
    routing.recommended_routes = [
        {
            "route": item.route,
            "confidence": item.confidence,
            "priority": item.priority,
            "rationale": item.rationale,
            "matched_signals": item.matched_signals,
            "source": item.source,
        }
        for item in recommendations
    ]
    routing.matched_labels = matched_labels
    if doc_entities is None:
        routing.review_reasons.append("entity_output_unavailable")
    routing.review_reasons = sorted(set(routing.review_reasons))
    return routing


def write_routing_json(
    doc_routing: DocumentRouting,
    output_folder: str,
    subfolder: str,
    pipeline_version: str,
) -> Optional[str]:
    """Write `.routing.json` sidecar output."""
    try:
        routing_dir = os.path.join(output_folder, "EXPORT", "ROUTING")
        if subfolder and subfolder != ".":
            safe_parts = [
                sanitize_path_segment(part)
                for part in subfolder.replace("\\", "/").split("/")
                if part
            ]
            target_dir = (
                os.path.join(routing_dir, *safe_parts)
                if safe_parts
                else routing_dir
            )
        else:
            target_dir = routing_dir

        resolved = os.path.realpath(target_dir)
        if not resolved.startswith(os.path.realpath(routing_dir)):
            logger.error("Path traversal blocked in routing output: %s", subfolder)
            return None

        os.makedirs(target_dir, exist_ok=True)

        base_name = build_sidecar_base_name(doc_routing.source_file)
        json_path = os.path.join(target_dir, f"{base_name}.routing.json")
        report = {
            "schema_version": "1.0",
            "document_id": doc_routing.document_id,
            "source_file": doc_routing.source_file,
            "processing": {
                "routing_mode": doc_routing.routing_mode,
                "pipeline_version": pipeline_version,
                "timestamp": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(timespec="milliseconds"),
            },
            "routing_summary": {
                "primary_route": doc_routing.primary_route,
                "review_required": doc_routing.review_required,
                "review_reasons": doc_routing.review_reasons,
                "matched_labels": doc_routing.matched_labels,
            },
            "recommended_routes": doc_routing.recommended_routes,
        }

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, default=str)

        return json_path
    except Exception as exc:
        logger.error(
            "Failed to write routing JSON for %s: %s",
            doc_routing.document_id,
            exc,
        )
        return None
