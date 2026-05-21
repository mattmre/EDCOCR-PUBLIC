"""Named Entity Recognition (NER) for forensic OCR pipeline.

Extracts named entities (persons, organizations, locations, dates, monetary
values) from OCR text using spaCy, plus custom regex patterns for legal
entities (case numbers, Bates numbers, exhibit references).

Output: EXPORT/NER/<subfolder>/<document_name>.ner.json

Graceful degradation: if spaCy is not installed, spaCy-based extraction
returns empty results while custom regex extraction continues to work.
"""

import datetime
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from ocr_distributed.ocr_utils import (
    build_sidecar_base_name,
    sanitize_path_segment,
)

__all__ = [
    "Entity",
    "PageNER",
    "DocumentNER",
    "extract_entities",
    "extract_custom_entities",
    "finalize_ner",
    "write_ner_json",
]

logger = logging.getLogger(__name__)

# --- Guarded spaCy import ---
try:
    import spacy

    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

# Default spaCy model name
_SPACY_MODEL = "en_core_web_sm"

# Cached spaCy NLP pipeline (loaded lazily on first use)
_nlp_cache = None

# spaCy entity types we extract
_SPACY_ENTITY_TYPES = frozenset({"PERSON", "ORG", "GPE", "DATE", "MONEY"})

# ---------------------------------------------------------------------------
# Custom regex patterns for legal/forensic entities
# ---------------------------------------------------------------------------

# Case numbers: "Case No. 2024-CV-1234", "Case No. 24-1234", "Docket No. 2024-12345"
_CASE_NUMBER_PATTERN = re.compile(
    r"\b(?:Case|Docket|Matter)\s+(?:No\.?|Number)\s*[:.]?\s*"
    r"(\d{2,4}[-/]\w{1,5}[-/]\d{2,10}|\d{2,4}[-/]\d{2,10})",
    re.IGNORECASE,
)

# Bates numbers: "ABC001234", "XYZ 000001", 3+ alpha prefix + 4+ digits
_BATES_NUMBER_PATTERN = re.compile(
    r"\b([A-Z]{3,10}\s?\d{4,10})\b",
    re.IGNORECASE,
)

# Exhibit references: "Exhibit A", "Exhibit 1", "Ex. A-1", "Exhibit No. 42"
_EXHIBIT_REF_PATTERN = re.compile(
    r"\b(?:Exhibit|Ex\.?)\s+(?:No\.?\s*)?([A-Z0-9]{1,4}(?:[-][A-Z0-9]{1,4})?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """A single extracted entity."""

    entity_type: str  # PERSON, ORG, GPE, DATE, MONEY, CASE_NUMBER, BATES_NUMBER, EXHIBIT_REF
    text: str
    confidence: float = 0.0
    page_num: int = 0
    start: int = 0  # Start character offset in page text
    end: int = 0  # End character offset in page text


@dataclass
class PageNER:
    """NER results for a single page."""

    page_num: int
    entities: list = field(default_factory=list)  # List[Entity as dict]


@dataclass
class DocumentNER:
    """NER results for an entire document."""

    document_id: str
    source_file: str
    pages: list = field(default_factory=list)  # List[PageNER as dict]

    # Summary (computed by finalize_ner)
    total_entities: int = 0
    entity_type_counts: dict = field(default_factory=dict)
    unique_entities: list = field(default_factory=list)  # Deduplicated entity texts


# ---------------------------------------------------------------------------
# spaCy NLP loader
# ---------------------------------------------------------------------------


def _load_nlp():
    """Load and cache the spaCy NLP pipeline."""
    global _nlp_cache
    if _nlp_cache is not None:
        return _nlp_cache
    if not _SPACY_AVAILABLE:
        return None
    try:
        _nlp_cache = spacy.load(_SPACY_MODEL)
        logger.info("spaCy model '%s' loaded successfully", _SPACY_MODEL)
        return _nlp_cache
    except OSError:
        logger.warning(
            "spaCy model '%s' not found. Run: python -m spacy download %s",
            _SPACY_MODEL,
            _SPACY_MODEL,
        )
        return None


# ---------------------------------------------------------------------------
# Entity extraction functions
# ---------------------------------------------------------------------------


def extract_entities(text: str, page_num: int) -> list:
    """Extract named entities from text using spaCy NER.

    Args:
        text: Page text to analyze.
        page_num: Page number for entity attribution.

    Returns:
        List of Entity objects. Empty list if spaCy is unavailable.
    """
    if not _SPACY_AVAILABLE:
        return []

    nlp = _load_nlp()
    if nlp is None:
        return []

    entities = []
    try:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in _SPACY_ENTITY_TYPES:
                entities.append(Entity(
                    entity_type=ent.label_,
                    text=ent.text.strip(),
                    confidence=round(1.0, 4),  # spaCy NER does not expose per-entity confidence
                    page_num=page_num,
                    start=ent.start_char,
                    end=ent.end_char,
                ))
    except Exception as e:
        logger.warning("spaCy NER failed for page %d: %s", page_num, e)

    return entities


def extract_custom_entities(text: str, page_num: int) -> list:
    """Extract custom legal/forensic entities using regex patterns.

    Works independently of spaCy -- always available.

    Args:
        text: Page text to analyze.
        page_num: Page number for entity attribution.

    Returns:
        List of Entity objects for CASE_NUMBER, BATES_NUMBER, EXHIBIT_REF.
    """
    entities = []

    # Case numbers — group(1) captures the identifier without "Case No." prefix
    for match in _CASE_NUMBER_PATTERN.finditer(text):
        entities.append(Entity(
            entity_type="CASE_NUMBER",
            text=match.group(1).strip(),
            confidence=1.0,
            page_num=page_num,
            start=match.start(1),
            end=match.end(1),
        ))

    # Bates numbers — group(1) captures the full Bates stamp
    for match in _BATES_NUMBER_PATTERN.finditer(text):
        entities.append(Entity(
            entity_type="BATES_NUMBER",
            text=match.group(1).strip(),
            confidence=1.0,
            page_num=page_num,
            start=match.start(1),
            end=match.end(1),
        ))

    # Exhibit references — group(1) captures the identifier without "Exhibit" prefix
    for match in _EXHIBIT_REF_PATTERN.finditer(text):
        entities.append(Entity(
            entity_type="EXHIBIT_REF",
            text=match.group(1).strip(),
            confidence=1.0,
            page_num=page_num,
            start=match.start(1),
            end=match.end(1),
        ))

    return entities


# ---------------------------------------------------------------------------
# Finalization and deduplication
# ---------------------------------------------------------------------------


def _entity_to_dict(entity):
    """Convert an Entity (dataclass or dict) to a plain dict."""
    if isinstance(entity, Entity):
        return {
            "type": entity.entity_type,
            "text": entity.text,
            "confidence": entity.confidence,
            "page_num": entity.page_num,
            "start": entity.start,
            "end": entity.end,
        }
    if isinstance(entity, dict):
        return entity
    return {}


def finalize_ner(doc_ner: DocumentNER) -> DocumentNER:
    """Compute summary statistics, deduplicate entities, and fill counts.

    Args:
        doc_ner: DocumentNER with pages already populated.

    Returns:
        The same DocumentNER instance with summary fields filled in.
    """
    type_counts = {}
    seen_entities = set()  # (type, normalized_text) for dedup
    unique_list = []
    total = 0

    for page_data in doc_ner.pages:
        # Accept both PageNER/dict forms
        if isinstance(page_data, PageNER):
            page_entities = page_data.entities
        elif isinstance(page_data, dict):
            page_entities = page_data.get("entities", [])
        else:
            continue

        for ent in page_entities:
            ent_dict = _entity_to_dict(ent)
            ent_type = ent_dict.get("type", "UNKNOWN")
            ent_text = ent_dict.get("text", "")
            total += 1
            type_counts[ent_type] = type_counts.get(ent_type, 0) + 1

            # Dedup by (type, normalized text)
            key = (ent_type, ent_text.strip().lower())
            if key not in seen_entities:
                seen_entities.add(key)
                unique_list.append({"type": ent_type, "text": ent_text})

    doc_ner.total_entities = total
    doc_ner.entity_type_counts = dict(sorted(type_counts.items()))
    doc_ner.unique_entities = unique_list

    return doc_ner


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def write_ner_json(
    doc_ner: DocumentNER,
    output_folder: str,
    subfolder: str,
    pipeline_version: str,
) -> Optional[str]:
    """Write .ner.json sidecar file.

    Output to EXPORT/NER/<subfolder>/<name>.ner.json

    Args:
        doc_ner: Finalized DocumentNER dataclass.
        output_folder: Root NER output directory (e.g. /app/ocr_output/EXPORT/NER
                       or the base output dir -- caller decides).
        subfolder: Relative subfolder path mirroring source structure.
        pipeline_version: Pipeline version string for metadata.

    Returns:
        Path to the written JSON file, or None on failure.
    """
    try:
        # Sanitize subfolder path segments for filesystem safety
        if subfolder and subfolder != ".":
            safe_parts = [sanitize_path_segment(p) for p in subfolder.replace("\\", "/").split("/") if p]
            target_dir = os.path.join(output_folder, *safe_parts) if safe_parts else output_folder
        else:
            target_dir = output_folder
        os.makedirs(target_dir, exist_ok=True)

        base_name = build_sidecar_base_name(doc_ner.source_file)
        json_path = os.path.join(target_dir, f"{base_name}.ner.json")

        # Determine NER engine info
        ner_engine = "spacy" if _SPACY_AVAILABLE else "regex_only"
        ner_model = _SPACY_MODEL if _SPACY_AVAILABLE else "n/a"

        # Build page-level output with entity dicts
        pages_output = []
        for page_data in doc_ner.pages:
            if isinstance(page_data, PageNER):
                p_num = page_data.page_num
                p_entities = [_entity_to_dict(e) for e in page_data.entities]
            elif isinstance(page_data, dict):
                p_num = page_data.get("page_num", 0)
                raw_entities = page_data.get("entities", [])
                p_entities = [_entity_to_dict(e) for e in raw_entities]
            else:
                continue
            pages_output.append({
                "page_num": p_num,
                "entities": p_entities,
            })

        report = {
            "schema_version": "1.0",
            "document_id": doc_ner.document_id,
            "source_file": doc_ner.source_file,
            "processing": {
                "ner_engine": ner_engine,
                "ner_model": ner_model,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
                "pipeline_version": pipeline_version,
            },
            "entities_summary": {
                "total_entities": doc_ner.total_entities,
                "entity_types": doc_ner.entity_type_counts,
                "unique_entities": len(doc_ner.unique_entities),
            },
            "pages": pages_output,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return json_path
    except Exception as e:
        logger.error("Failed to write NER JSON for %s: %s", doc_ner.document_id, e)
        return None
