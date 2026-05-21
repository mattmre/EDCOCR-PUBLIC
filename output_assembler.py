"""Unified retrieval output assembler for EDCOCR.

Merges OCR text, entities, classification, extraction, structure, and
metadata into a single retrieval-ready document-level JSON output.

Output: EXPORT/RETRIEVAL/<subfolder>/<document_name>.retrieval.json

Also supports optional Markdown-friendly output for human consumption.

Opt-in via ENABLE_RETRIEVAL_OUTPUT=true or --enable-retrieval-output CLI flag.
"""

import datetime
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from entity_consolidator import merge_duplicate_entities
from ocr_distributed.ocr_utils import build_sidecar_base_name, sanitize_path_segment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENABLE_RETRIEVAL_OUTPUT = os.environ.get(
    "ENABLE_RETRIEVAL_OUTPUT", "false"
).lower() in ("1", "true", "yes")

_SCHEMA_VERSION = "1.0"


def _escape_md_table_cell(value: str) -> str:
    """Escape characters that break Markdown table formatting.

    Pipe characters (``|``) are the most critical because they delimit
    table columns.  Newlines are replaced with spaces to keep cell
    content on a single line.
    """
    return value.replace("|", "\\|").replace("\n", " ").replace("\r", "")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RetrievalDocument:
    """Unified document representation for retrieval."""

    document_id: str
    source_file: str
    schema_version: str = _SCHEMA_VERSION

    # Text
    text: str = ""
    text_by_page: list = field(default_factory=list)  # [{page: int, text: str}]

    # Classification
    classification: dict = field(default_factory=dict)  # {label, confidence, method}

    # Entities
    entities: list = field(default_factory=list)  # Consolidated entity dicts

    # Key-value pairs
    key_value_pairs: list = field(default_factory=list)  # From extraction

    # Tables
    tables: list = field(default_factory=list)  # From structure.json

    # Metadata
    metadata: dict = field(default_factory=dict)
    # {pages, quality, languages, ocr_methods, overall_confidence, ...}

    # Relationships (if available)
    relationships: list = field(default_factory=list)

    # Handwriting
    handwriting: dict = field(default_factory=dict)
    # {detected, regions_count, coverage, is_primarily_handwritten}

    # Pipeline version (populated at assembly time)
    pipeline_version: str = ""

    def to_dict(self) -> dict:
        """Serialize to dict matching retrieval.schema.json."""
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "generated_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="milliseconds"),
            "pipeline_version": self.pipeline_version,
            "text": self.text,
            "text_by_page": self.text_by_page,
            "classification": self.classification,
            "entities": self.entities,
            "key_value_pairs": self.key_value_pairs,
            "tables": self.tables,
            "handwriting": self.handwriting,
            "relationships": self.relationships,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """Generate Markdown-friendly representation for human consumption."""
        lines = []

        # Header
        lines.append(f"# Document: {self.source_file}")
        lines.append("")

        # Classification + metadata summary
        cls_label = self.classification.get("label", "unknown")
        cls_conf = self.classification.get("confidence", 0.0)
        quality = self.metadata.get("quality", "unknown")
        pages = self.metadata.get("pages", 0)
        languages = self.metadata.get("languages", [])
        lang_str = ", ".join(languages) if languages else "unknown"

        lines.append(
            f"**Classification:** {cls_label} ({cls_conf:.0%})"
        )
        lines.append(
            f"**Quality:** {quality} | **Pages:** {pages} | **Language:** {lang_str}"
        )
        lines.append("")

        # Text content
        lines.append("## Text Content")
        lines.append("")
        if self.text:
            lines.append(self.text)
        else:
            lines.append("_No text extracted._")
        lines.append("")

        # Entities table
        if self.entities:
            lines.append("## Entities")
            lines.append("")
            lines.append("| Type | Text | Confidence | Page |")
            lines.append("|------|------|------------|------|")
            for ent in self.entities:
                etype = _escape_md_table_cell(str(ent.get("type", "")))
                etext = _escape_md_table_cell(str(ent.get("text", "")))
                econf = ent.get("confidence", 0.0)
                epage = ent.get("page", 0)
                lines.append(f"| {etype} | {etext} | {econf:.2f} | {epage} |")
            lines.append("")

        # Key-value pairs table
        if self.key_value_pairs:
            lines.append("## Key-Value Pairs")
            lines.append("")
            lines.append("| Key | Value | Confidence |")
            lines.append("|-----|-------|------------|")
            for kv in self.key_value_pairs:
                kv_key = _escape_md_table_cell(str(kv.get("key", "")))
                kv_val = _escape_md_table_cell(str(kv.get("value", "")))
                kv_conf = kv.get("confidence", 0.0)
                lines.append(f"| {kv_key} | {kv_val} | {kv_conf:.2f} |")
            lines.append("")

        # Tables
        if self.tables:
            lines.append("## Tables")
            lines.append("")
            for i, table in enumerate(self.tables, start=1):
                page = table.get("page", 0)
                html = table.get("html", "")
                lines.append(f"### Table {i} (Page {page})")
                lines.append("")
                if html:
                    lines.append(html)
                else:
                    lines.append("_No table HTML available._")
                lines.append("")

        # Handwriting
        if self.handwriting.get("detected"):
            lines.append("## Handwriting")
            lines.append("")
            hw_regions = self.handwriting.get("regions_count", 0)
            hw_coverage = self.handwriting.get("coverage", 0.0)
            hw_primary = self.handwriting.get("is_primarily_handwritten", False)
            lines.append(
                f"- **Regions detected:** {hw_regions}"
            )
            lines.append(f"- **Coverage:** {hw_coverage:.0%}")
            lines.append(
                f"- **Primarily handwritten:** {'Yes' if hw_primary else 'No'}"
            )
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidecar data extraction helpers
# ---------------------------------------------------------------------------


def _extract_entities_from_ner_data(ner_data: dict) -> list:
    """Extract entity dicts from a parsed .ner.json sidecar.

    Expected keys: pages[].entities[] with type, text, confidence, page_num.
    """
    entities = []
    ordinal = 0
    for page in ner_data.get("pages", []):
        page_num = page.get("page_num", 0)
        for ent in page.get("entities", []):
            ordinal += 1
            text = ent.get("text", "")
            if not text:
                continue
            entities.append({
                "id": f"ner_{ordinal:03d}",
                "type": ent.get("entity_type", ent.get("type", "UNKNOWN")),
                "text": text.strip(),
                "confidence": round(float(ent.get("confidence", 0.0)), 4),
                "source": "ner",
                "page": int(ent.get("page_num", page_num)),
                "bbox": [],
                "metadata": {},
            })
    return entities


def _extract_from_extraction_data(extraction_data: dict) -> tuple:
    """Extract entity and key-value pair dicts from a parsed .extraction.json.

    Expected keys: pages[].fields[] with field_type, text, confidence, page_num.

    Returns:
        Tuple of (entities_list, kv_pairs_list).
    """
    entities = []
    kv_pairs = []
    ordinal = 0
    for page in extraction_data.get("pages", []):
        page_num = int(page.get("page_num", 0))
        for f in page.get("fields", []):
            ordinal += 1
            text = f.get("text", "")
            if not text:
                continue
            f_type = f.get("field_type", "")
            f_conf = round(float(f.get("confidence", 0.0)), 4)
            f_page = int(f.get("page_num", page_num))
            bbox = f.get("bbox", [])
            if not isinstance(bbox, list):
                bbox = []

            entities.append({
                "id": f"ext_{ordinal:03d}",
                "type": f_type,
                "text": text.strip(),
                "confidence": f_conf,
                "source": "extraction",
                "page": f_page,
                "bbox": bbox,
                "metadata": {},
            })

            if f_type and text:
                kv_pairs.append({
                    "key": f_type,
                    "value": text.strip(),
                    "confidence": f_conf,
                    "page": f_page,
                    "source": "extraction",
                })
    return entities, kv_pairs


def _extract_classification(classification_data: dict) -> dict:
    """Extract primary classification from a parsed .classification.json."""
    doc_type = classification_data.get("document_type", "")
    confidence = float(classification_data.get("confidence", 0.0))
    method = classification_data.get("classification_method", "")

    # Some formats use nested structures
    if not doc_type:
        doc_type = classification_data.get("label", "")
    if not method:
        method = classification_data.get("method", "")

    if doc_type:
        return {
            "label": doc_type,
            "confidence": round(confidence, 4),
            "method": method,
        }
    return {}


def _extract_tables(structure_data: dict) -> list:
    """Extract table data from a parsed .structure.json."""
    tables = []
    for page in structure_data.get("pages", []):
        page_num = page.get("page_num", 0)
        for table in page.get("tables", []):
            tables.append({
                "page": page_num,
                "html": table.get("html", ""),
                "cell_bbox": table.get("cell_bbox", []),
            })
    return tables


def _extract_metadata(validation_data: dict) -> dict:
    """Extract metadata from a parsed .validation.json."""
    quality_block = validation_data.get("quality", {})
    page_count = validation_data.get("page_count", {})

    return {
        "pages": page_count.get("source", 0),
        "quality": quality_block.get("classification", "unknown"),
        "overall_confidence": round(
            float(quality_block.get("overall_confidence", 0.0)), 4
        ),
        "languages": [],  # Populated from validation pages if available
        "ocr_methods": quality_block.get("ocr_methods_used", []),
        "text_extraction_rate": round(
            float(quality_block.get("text_extraction_rate", 0.0)), 4
        ),
        "total_text_length": int(quality_block.get("total_text_length", 0)),
        "output_hash": validation_data.get("output_hash", ""),
    }


def _extract_handwriting(handwriting_data: dict) -> dict:
    """Extract handwriting summary from a parsed .handwriting.json."""
    total_hw_pages = int(handwriting_data.get("total_handwritten_pages", 0))
    overall_coverage = float(
        handwriting_data.get("overall_handwriting_coverage", 0.0)
    )
    is_primary = handwriting_data.get("is_primarily_handwritten", False)

    # Count total regions across all pages
    regions_count = 0
    for page in handwriting_data.get("pages", []):
        regions_count += len(page.get("handwriting_regions", []))

    return {
        "detected": total_hw_pages > 0,
        "regions_count": regions_count,
        "coverage": round(overall_coverage, 4),
        "is_primarily_handwritten": bool(is_primary),
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_retrieval_output(
    document_id: str,
    source_file: str,
    ocr_text: str = "",
    text_by_page: Optional[list] = None,
    entities_data: Optional[dict] = None,
    classification_data: Optional[dict] = None,
    extraction_data: Optional[dict] = None,
    structure_data: Optional[dict] = None,
    validation_data: Optional[dict] = None,
    handwriting_data: Optional[dict] = None,
    relationship_data: Optional[dict] = None,
    pipeline_version: str = "",
) -> RetrievalDocument:
    """Assemble a unified retrieval document from individual sidecar data.

    Each parameter is the parsed content of the respective sidecar JSON.
    Pass None for any unavailable source. The assembler degrades gracefully:
    missing sources produce empty but valid sections.

    Args:
        document_id: Unique document identifier.
        source_file: Source document filename.
        ocr_text: Full concatenated OCR text.
        text_by_page: List of dicts with ``page`` and ``text`` keys.
        entities_data: Parsed .entities.json or .ner.json sidecar.
        classification_data: Parsed .classification.json sidecar.
        extraction_data: Parsed .extraction.json sidecar.
        structure_data: Parsed .structure.json sidecar.
        validation_data: Parsed .validation.json sidecar.
        handwriting_data: Parsed .handwriting.json sidecar.
        relationship_data: Parsed relationship data dict.
        pipeline_version: Pipeline version string.

    Returns:
        A populated RetrievalDocument instance.
    """
    doc = RetrievalDocument(
        document_id=document_id,
        source_file=source_file,
        pipeline_version=pipeline_version,
    )

    # --- Text ---
    doc.text = ocr_text or ""
    doc.text_by_page = text_by_page or []

    # --- Classification ---
    if classification_data:
        doc.classification = _extract_classification(classification_data)

    # --- Entities (merge NER + extraction, deduplicate) ---
    all_entities = []

    if entities_data:
        # entities_data may be a consolidated .entities.json (already merged)
        # or a raw .ner.json
        if "entities" in entities_data and isinstance(entities_data["entities"], list):
            # Check if entities are pre-consolidated (have id/type/text/source)
            sample = entities_data["entities"][0] if entities_data["entities"] else {}
            if "source" in sample:
                # Pre-consolidated: use directly
                all_entities.extend(entities_data["entities"])
            else:
                # Raw NER format: adapt
                all_entities.extend(_extract_entities_from_ner_data(entities_data))
        elif "pages" in entities_data:
            # Raw per-page NER format
            all_entities.extend(_extract_entities_from_ner_data(entities_data))

    # --- Extraction ---
    kv_pairs = []
    if extraction_data:
        ext_entities, ext_kv = _extract_from_extraction_data(extraction_data)
        all_entities.extend(ext_entities)
        kv_pairs.extend(ext_kv)

        # Also pull top-level key_value_pairs if present
        if "key_value_pairs" in extraction_data:
            for kv in extraction_data["key_value_pairs"]:
                if isinstance(kv, dict) and kv.get("key") and kv.get("value"):
                    kv_pairs.append({
                        "key": kv["key"],
                        "value": kv["value"],
                        "confidence": round(float(kv.get("confidence", 0.0)), 4),
                        "page": int(kv.get("page", 0)),
                        "source": kv.get("source", "extraction"),
                    })

    # Deduplicate entities
    doc.entities = merge_duplicate_entities(all_entities)
    doc.key_value_pairs = kv_pairs

    # --- Tables from structure data ---
    if structure_data:
        doc.tables = _extract_tables(structure_data)

    # --- Metadata from validation ---
    if validation_data:
        doc.metadata = _extract_metadata(validation_data)
    else:
        doc.metadata = {
            "pages": len(doc.text_by_page) if doc.text_by_page else 0,
            "quality": "unknown",
            "overall_confidence": 0.0,
            "languages": [],
            "ocr_methods": [],
            "text_extraction_rate": 0.0,
            "total_text_length": len(doc.text),
            "output_hash": "",
        }

    # --- Handwriting ---
    if handwriting_data:
        doc.handwriting = _extract_handwriting(handwriting_data)
    else:
        doc.handwriting = {
            "detected": False,
            "regions_count": 0,
            "coverage": 0.0,
            "is_primarily_handwritten": False,
        }

    # --- Relationships ---
    if relationship_data:
        doc.relationships = relationship_data.get("relationships", [])

    return doc


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def write_retrieval_json(
    doc: RetrievalDocument,
    output_dir: str,
    subfolder: str = "",
) -> Optional[str]:
    """Write retrieval JSON to EXPORT/RETRIEVAL/ directory.

    Args:
        doc: Populated RetrievalDocument instance.
        output_dir: Root output directory (e.g. /app/ocr_output).
        subfolder: Relative subfolder path mirroring source structure.

    Returns:
        Path to the written JSON file, or None on failure.
    """
    try:
        retrieval_dir = os.path.join(output_dir, "EXPORT", "RETRIEVAL")
        if subfolder and subfolder != ".":
            safe_parts = [
                sanitize_path_segment(p)
                for p in subfolder.replace("\\", "/").split("/")
                if p
            ]
            target_dir = (
                os.path.join(retrieval_dir, *safe_parts)
                if safe_parts
                else retrieval_dir
            )
        else:
            target_dir = retrieval_dir

        # Path traversal protection
        resolved = os.path.realpath(target_dir)
        if not resolved.startswith(os.path.realpath(retrieval_dir)):
            logger.error(
                "Path traversal blocked in retrieval output: %s",
                subfolder,
            )
            return None

        os.makedirs(target_dir, exist_ok=True)

        base_name = build_sidecar_base_name(doc.source_file)
        json_path = os.path.join(target_dir, f"{base_name}.retrieval.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        logger.info("Wrote retrieval JSON: %s", json_path)
        return json_path
    except Exception as e:
        logger.error(
            "Failed to write retrieval JSON for %s: %s",
            doc.source_file,
            e,
        )
        return None


def write_retrieval_markdown(
    doc: RetrievalDocument,
    output_dir: str,
    subfolder: str = "",
) -> Optional[str]:
    """Write Markdown representation to EXPORT/RETRIEVAL/ directory.

    Args:
        doc: Populated RetrievalDocument instance.
        output_dir: Root output directory (e.g. /app/ocr_output).
        subfolder: Relative subfolder path mirroring source structure.

    Returns:
        Path to the written Markdown file, or None on failure.
    """
    try:
        retrieval_dir = os.path.join(output_dir, "EXPORT", "RETRIEVAL")
        if subfolder and subfolder != ".":
            safe_parts = [
                sanitize_path_segment(p)
                for p in subfolder.replace("\\", "/").split("/")
                if p
            ]
            target_dir = (
                os.path.join(retrieval_dir, *safe_parts)
                if safe_parts
                else retrieval_dir
            )
        else:
            target_dir = retrieval_dir

        # Path traversal protection
        resolved = os.path.realpath(target_dir)
        if not resolved.startswith(os.path.realpath(retrieval_dir)):
            logger.error(
                "Path traversal blocked in retrieval markdown output: %s",
                subfolder,
            )
            return None

        os.makedirs(target_dir, exist_ok=True)

        base_name = build_sidecar_base_name(doc.source_file)
        md_path = os.path.join(target_dir, f"{base_name}.retrieval.md")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(doc.to_markdown())

        logger.info("Wrote retrieval Markdown: %s", md_path)
        return md_path
    except Exception as e:
        logger.error(
            "Failed to write retrieval Markdown for %s: %s",
            doc.source_file,
            e,
        )
        return None
