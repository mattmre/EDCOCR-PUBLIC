"""Unified entity consolidation for forensic OCR pipeline.

Merges outputs from NER (ner.py), structured extraction (extraction.py),
and document classification (classification.py) into a single durable
``.entities.json`` sidecar file per document.

Output: EXPORT/ENTITIES/<subfolder>/<document_name>.entities.json

The consolidated file supplements -- not replaces -- the individual
sidecar outputs (*.ner.json, *.extraction.json, *.classification.json).
Each source module continues to write its own output for backward
compatibility.

Graceful degradation: if any source (NER, extraction, classification)
is disabled or produces no output, the consolidator works with whatever
data is available.  If all sources are empty the consolidator still
writes a valid (but empty) sidecar.
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENABLE_ENTITY_CONSOLIDATION = os.environ.get(
    "ENABLE_ENTITY_CONSOLIDATION", "false"
).lower() in ("1", "true", "yes")

# Schema version for the consolidated output
_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConsolidatedEntity:
    """A single entity from any source (NER, extraction, or classification)."""

    entity_id: str  # e.g. "ent_001"
    entity_type: str  # PERSON, DATE, amount, reference_number, etc.
    text: str
    confidence: float = 0.0
    source: str = ""  # "ner", "extraction", or "classification"
    page: int = 0
    bbox: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ConsolidatedClassification:
    """A document classification label."""

    label: str
    confidence: float = 0.0
    method: str = ""


@dataclass
class ConsolidatedKeyValuePair:
    """A key-value pair from extraction or NER."""

    key: str
    value: str
    confidence: float = 0.0
    page: int = 0
    source: str = ""


@dataclass
class ConsolidatedOutput:
    """Full consolidated entity output for a document."""

    document: str = ""
    entities: list = field(default_factory=list)
    classifications: list = field(default_factory=list)
    key_value_pairs: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# NER result adapters
# ---------------------------------------------------------------------------


def _entities_from_ner(doc_ner) -> list:
    """Extract entity dicts from a finalized DocumentNER.

    Accepts both dataclass instances and plain dicts for page/entity data.

    Returns:
        List of dicts with keys matching ConsolidatedEntity fields.
    """
    entities = []
    ordinal = 0

    pages = getattr(doc_ner, "pages", [])
    for page_data in pages:
        if isinstance(page_data, dict):
            page_num = page_data.get("page_num", 0)
            page_entities = page_data.get("entities", [])
        else:
            page_num = getattr(page_data, "page_num", 0)
            page_entities = getattr(page_data, "entities", [])

        for ent in page_entities:
            ordinal += 1
            if isinstance(ent, dict):
                ent_type = ent.get("type", ent.get("entity_type", "UNKNOWN"))
                ent_text = ent.get("text", "")
                ent_conf = float(ent.get("confidence", 0.0))
                ent_page = int(ent.get("page_num", page_num))
                ent_start = int(ent.get("start", 0))
                ent_end = int(ent.get("end", 0))
            else:
                ent_type = getattr(ent, "entity_type", "UNKNOWN")
                ent_text = getattr(ent, "text", "")
                ent_conf = float(getattr(ent, "confidence", 0.0))
                ent_page = int(getattr(ent, "page_num", page_num))
                ent_start = int(getattr(ent, "start", 0))
                ent_end = int(getattr(ent, "end", 0))

            if not ent_text:
                continue

            entities.append({
                "id": f"ner_{ordinal:03d}",
                "type": ent_type,
                "text": ent_text.strip(),
                "confidence": round(ent_conf, 4),
                "source": "ner",
                "page": ent_page,
                "bbox": [],
                "metadata": {
                    "start": ent_start,
                    "end": ent_end,
                },
            })

    return entities


# ---------------------------------------------------------------------------
# Extraction result adapters
# ---------------------------------------------------------------------------


def _entities_from_extraction(doc_ext) -> tuple:
    """Extract entity and key-value pair dicts from a finalized DocumentExtraction.

    Returns:
        Tuple of (entities_list, kv_pairs_list).
    """
    entities = []
    kv_pairs = []
    ordinal = 0

    pages = getattr(doc_ext, "pages", [])
    for page_data in pages:
        if isinstance(page_data, dict):
            page_num = int(page_data.get("page_num", 0))
            page_fields = page_data.get("fields", [])
        else:
            page_num = int(getattr(page_data, "page_num", 0))
            page_fields = getattr(page_data, "fields", [])

        for f in page_fields:
            ordinal += 1
            if isinstance(f, dict):
                f_type = f.get("field_type", "")
                f_text = f.get("text", "")
                f_conf = float(f.get("confidence", 0.0))
                f_page = int(f.get("page_num", page_num))
                f_method = f.get("extraction_method", "")
                f_norm = f.get("normalized_value", "")
                f_bbox = f.get("bbox", [])
                f_start = int(f.get("start", 0))
                f_end = int(f.get("end", 0))
            else:
                f_type = getattr(f, "field_type", "")
                f_text = getattr(f, "text", "")
                f_conf = float(getattr(f, "confidence", 0.0))
                f_page = int(getattr(f, "page_num", page_num))
                f_method = getattr(f, "extraction_method", "")
                f_norm = getattr(f, "normalized_value", "")
                f_bbox = getattr(f, "bbox", [])
                f_start = int(getattr(f, "start", 0))
                f_end = int(getattr(f, "end", 0))

            if not f_text:
                continue

            bbox = f_bbox if isinstance(f_bbox, list) else []

            metadata = {
                "extraction_method": f_method,
                "start": f_start,
                "end": f_end,
            }
            if f_norm:
                metadata["normalized_value"] = f_norm

            entities.append({
                "id": f"ext_{ordinal:03d}",
                "type": f_type,
                "text": f_text.strip(),
                "confidence": round(f_conf, 4),
                "source": "extraction",
                "page": f_page,
                "bbox": bbox,
                "metadata": metadata,
            })

            # Build a key-value pair for typed fields
            if f_type and f_text:
                kv_pairs.append({
                    "key": f_type,
                    "value": f_text.strip(),
                    "confidence": round(f_conf, 4),
                    "page": f_page,
                    "source": "extraction",
                })

    return entities, kv_pairs


# ---------------------------------------------------------------------------
# Classification result adapters
# ---------------------------------------------------------------------------


def _classifications_from_result(doc_cls) -> list:
    """Extract classification labels from a finalized DocumentClassification.

    Returns:
        List of dicts with keys matching ConsolidatedClassification fields.
    """
    classifications = []

    # Primary document-level classification
    doc_type = getattr(doc_cls, "document_type", "other")
    doc_conf = float(getattr(doc_cls, "document_confidence", 0.0))

    if doc_type and doc_type != "other":
        classifications.append({
            "label": doc_type,
            "confidence": round(doc_conf, 4),
            "method": "ensemble",
        })

    # Additional labels from document_labels if present
    doc_labels = getattr(doc_cls, "document_labels", [])
    seen = {doc_type} if doc_type else set()
    for label_item in doc_labels:
        if isinstance(label_item, dict):
            label = label_item.get("label", "")
            conf = float(label_item.get("confidence", 0.0))
            method = label_item.get("source", "classification")
        else:
            continue

        if label and label not in seen:
            seen.add(label)
            classifications.append({
                "label": label,
                "confidence": round(conf, 4),
                "method": method,
            })

    return classifications


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def merge_duplicate_entities(entities: list) -> list:
    """Deduplicate entities from different sources.

    When entities from different sources have matching type and text (case-
    insensitive), the one with higher confidence is kept.  Ties are broken by
    source priority: extraction > ner > classification.

    Args:
        entities: List of entity dicts.

    Returns:
        Deduplicated list with renumbered IDs.
    """
    if not entities:
        return []

    source_priority = {"extraction": 0, "ner": 1, "classification": 2}

    # Group by (type, normalized_text)
    groups = {}
    for ent in entities:
        key = (
            ent.get("type", "").lower(),
            ent.get("text", "").strip().lower(),
        )
        if key not in groups:
            groups[key] = []
        groups[key].append(ent)

    deduped = []
    for _key, group in groups.items():
        # Pick the best entity by confidence, then source priority
        best = max(
            group,
            key=lambda e: (
                e.get("confidence", 0.0),
                -source_priority.get(e.get("source", ""), 99),
            ),
        )
        deduped.append(best)

    # Sort by page then original order, and renumber IDs
    deduped.sort(key=lambda e: (e.get("page", 0), e.get("id", "")))
    for i, ent in enumerate(deduped, start=1):
        ent["id"] = f"ent_{i:03d}"

    return deduped


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def consolidate_entities(
    ner_results=None,
    extraction_results=None,
    classification_results=None,
    doc_name: str = "",
    pipeline_version: str = "",
) -> dict:
    """Consolidate entities from NER, extraction, and classification sources.

    Args:
        ner_results: Finalized DocumentNER instance (or None).
        extraction_results: Finalized DocumentExtraction instance (or None).
        classification_results: Finalized DocumentClassification (or None).
        doc_name: Document filename for output metadata.
        pipeline_version: Pipeline version string.

    Returns:
        Dict matching the consolidated .entities.json schema.
    """
    all_entities = []
    kv_pairs = []
    classifications = []

    # Collect entities from NER
    if ner_results is not None:
        ner_entities = _entities_from_ner(ner_results)
        all_entities.extend(ner_entities)

    # Collect entities and KV pairs from extraction
    if extraction_results is not None:
        ext_entities, ext_kv = _entities_from_extraction(extraction_results)
        all_entities.extend(ext_entities)
        kv_pairs.extend(ext_kv)

    # Collect classification labels
    if classification_results is not None:
        classifications = _classifications_from_result(classification_results)

    # Deduplicate entities
    deduped_entities = merge_duplicate_entities(all_entities)

    # Build summary
    entity_type_counts = {}
    for ent in deduped_entities:
        etype = ent.get("type", "UNKNOWN")
        entity_type_counts[etype] = entity_type_counts.get(etype, 0) + 1

    primary_classification = ""
    if classifications:
        primary_classification = classifications[0].get("label", "")

    summary = {
        "total_entities": len(deduped_entities),
        "entity_types": dict(sorted(entity_type_counts.items())),
        "total_kv_pairs": len(kv_pairs),
        "primary_classification": primary_classification,
    }

    return {
        "schema_version": _SCHEMA_VERSION,
        "document": doc_name,
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="milliseconds"),
        "pipeline_version": pipeline_version,
        "entities": deduped_entities,
        "classifications": classifications,
        "key_value_pairs": kv_pairs,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def write_consolidated_entities_json(
    consolidated: dict,
    output_folder: str,
    subfolder: str,
    doc_name: str,
) -> Optional[str]:
    """Write a unified .entities.json sidecar file.

    Output to EXPORT/ENTITIES/<subfolder>/<doc_name>.entities.json

    This supplements the existing per-source sidecar files. The individual
    *.ner.json, *.extraction.json, and *.classification.json files continue
    to be written by their respective modules.

    Args:
        consolidated: Dict from consolidate_entities().
        output_folder: Root output directory (e.g. /app/ocr_output).
        subfolder: Relative subfolder path mirroring source structure.
        doc_name: Source document filename (used for sidecar naming).

    Returns:
        Path to the written JSON file, or None on failure.
    """
    try:
        entities_dir = os.path.join(output_folder, "EXPORT", "ENTITIES")
        if subfolder and subfolder != ".":
            safe_parts = [
                sanitize_path_segment(p)
                for p in subfolder.replace("\\", "/").split("/")
                if p
            ]
            target_dir = (
                os.path.join(entities_dir, *safe_parts)
                if safe_parts
                else entities_dir
            )
        else:
            target_dir = entities_dir

        # Path traversal protection
        resolved = os.path.realpath(target_dir)
        if not resolved.startswith(os.path.realpath(entities_dir)):
            logger.error(
                "Path traversal blocked in consolidated entities output: %s",
                subfolder,
            )
            return None

        os.makedirs(target_dir, exist_ok=True)

        base_name = build_sidecar_base_name(doc_name)
        json_path = os.path.join(target_dir, f"{base_name}.entities.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False, default=str)

        return json_path
    except Exception as e:
        logger.error(
            "Failed to write consolidated entities JSON for %s: %s",
            doc_name,
            e,
        )
        return None
