"""Enhanced structured extraction for forensic OCR pipeline.

Extracts structured fields (dates, amounts, names, addresses, reference numbers)
from OCR text using PaddleNLP UIE (Universal Information Extraction) with regex
fallback. Optionally integrates LayoutLMv3 semantic extraction for token-level
Key Information Extraction (KIE) with spatial layout awareness.

Output: EXPORT/EXTRACTION/<subfolder>/<document_name>.extraction.json

Graceful degradation: if PaddleNLP is not installed, UIE-based extraction is
skipped while regex pattern extraction continues to work. If transformers/torch
are not installed, semantic extraction is skipped.

Extraction modes (``EXTRACTION_MODE`` env var):
- ``regex``    -- regex patterns only
- ``uie``      -- PaddleNLP UIE + regex fallback (default, existing behavior)
- ``semantic`` -- LayoutLMv3 + regex fallback (no UIE)
- ``all``      -- UIE + LayoutLMv3 + regex (all three tiers)
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
    "EXTRACTION_MODE",
    "DEFAULT_SCHEMA",
    "ExtractedField",
    "PageExtraction",
    "DocumentExtraction",
    "extract_fields_uie",
    "extract_fields_regex",
    "extract_page_fields",
    "finalize_extraction",
    "write_extraction_json",
]

logger = logging.getLogger(__name__)

# --- Extraction mode configuration ---
EXTRACTION_MODE = os.environ.get("EXTRACTION_MODE", "uie").lower()

# --- Guarded PaddleNLP import ---
try:
    from paddlenlp import Taskflow

    _PADDLENLP_AVAILABLE = True
except ImportError:
    _PADDLENLP_AVAILABLE = False

# --- UIE Schema ---
DEFAULT_SCHEMA = [
    "Date",
    "Amount",
    "Person Name",
    "Organization",
    "Address",
    "Reference Number",
    "Phone Number",
    "Email Address",
]

# --- UIE Engine (lazy-loaded, cached) ---
_uie_engine = None
_uie_init_failed = False  # Prevent repeated init attempts after failure

# --- UIE label to field type mapping ---
_UIE_TYPE_MAP = {
    "Date": "date",
    "Amount": "amount",
    "Person Name": "person_name",
    "Organization": "organization",
    "Address": "address",
    "Reference Number": "reference_number",
    "Phone Number": "phone_number",
    "Email Address": "email_address",
}

# ---------------------------------------------------------------------------
# Regex patterns for fallback
# ---------------------------------------------------------------------------

# ISO dates: 2024-01-15, 2024/01/15
_DATE_ISO = re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b")
# US dates: 01/15/2024, 1-15-2024
_DATE_US = re.compile(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b")
# Written dates: January 15, 2024; Jan 15, 2024; 15 January 2024
_DATE_WRITTEN = re.compile(
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}|"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+"
    r"\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
# Currency amounts: $1,234.56, EUR 100.00, USD 1234, etc.
_AMOUNT = re.compile(
    r"(?:\$|USD|EUR|GBP|CAD|AUD)\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b|"
    r"\b\d{1,3}(?:,\d{3})*\.\d{2}\s*(?:USD|EUR|GBP|CAD|AUD)\b",
    re.IGNORECASE,
)
# Phone numbers: (555) 123-4567, 555-123-4567, +1-555-123-4567
_PHONE = re.compile(
    r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
# Email addresses
_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)
# Reference numbers: INV-2024-001, PO#12345, REF: 12345, Ref No. ABC-123
_REFERENCE = re.compile(
    r"\b(?:INV|PO|REF|SO|WO|RFQ|RFP|DOC|ID)[-#:\s]*\d{3,10}(?:[-]\d{1,6})?\b",
    re.IGNORECASE,
)

# Month name lookup for date normalization
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExtractedField:
    """A single extracted field."""

    field_type: str  # date, amount, person_name, organization, address,
    #                  reference_number, phone_number, email_address
    text: str  # Raw extracted value
    confidence: float = 0.0
    page_num: int = 0
    start: int = 0  # Character offset
    end: int = 0
    extraction_method: str = ""  # "uie" or "regex"
    normalized_value: str = ""  # Optional normalized form (e.g., ISO date)


@dataclass
class PageExtraction:
    """Extraction results for a single page."""

    page_num: int
    fields: list = field(default_factory=list)  # List[ExtractedField as dict]


@dataclass
class DocumentExtraction:
    """Extraction results for an entire document."""

    document_id: str
    source_file: str
    pages: list = field(default_factory=list)  # List[PageExtraction as dict]
    total_fields: int = 0
    field_type_counts: dict = field(default_factory=dict)
    extraction_engine: str = ""  # "uie", "regex", or "hybrid"


# ---------------------------------------------------------------------------
# UIE engine loader
# ---------------------------------------------------------------------------


def _get_uie_engine():
    """Lazy-load and cache the PaddleNLP UIE engine.

    Returns the engine instance, or None if PaddleNLP is unavailable or
    initialization fails.
    """
    global _uie_engine, _uie_init_failed

    if _uie_engine is not None:
        return _uie_engine
    if _uie_init_failed or not _PADDLENLP_AVAILABLE:
        return None

    try:
        model_name = os.environ.get("UIE_MODEL", "uie-base")
        schema_env = os.environ.get("UIE_SCHEMA", "")
        schema = (
            [s.strip() for s in schema_env.split(",") if s.strip()]
            if schema_env
            else DEFAULT_SCHEMA
        )
        _uie_engine = Taskflow("information_extraction", model=model_name, schema=schema)
        logger.info("PaddleNLP UIE engine loaded: model=%s, schema=%s", model_name, schema)
        return _uie_engine
    except Exception as e:
        _uie_init_failed = True
        logger.warning("PaddleNLP UIE init failed (will use regex fallback): %s", e)
        return None


# ---------------------------------------------------------------------------
# UIE label normalization
# ---------------------------------------------------------------------------


def _normalize_uie_type(uie_label: str) -> str:
    """Map a UIE schema label to a normalized field type string."""
    return _UIE_TYPE_MAP.get(uie_label, uie_label.lower().replace(" ", "_"))


# ---------------------------------------------------------------------------
# Date normalization helper
# ---------------------------------------------------------------------------


def _normalize_date(text: str) -> str:
    """Best-effort normalization of a date string to ISO 8601 (YYYY-MM-DD).

    Returns the normalized string or empty string if parsing fails.
    """
    try:
        # Try ISO: 2024-01-15 or 2024/01/15
        m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", text)
        if m:
            return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

        # Try US: 01/15/2024 or 1-15-2024
        m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", text)
        if m:
            return f"{int(m.group(3)):04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"

        # Try written: "January 15, 2024" or "15 January 2024" or "Jan. 15, 2024"
        # Pattern: Month Day, Year
        m = re.match(
            r"^([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$", text.strip()
        )
        if m:
            month = _MONTH_NAMES.get(m.group(1).lower().rstrip("."))
            if month:
                return f"{int(m.group(3)):04d}-{month:02d}-{int(m.group(2)):02d}"

        # Pattern: Day Month Year
        m = re.match(
            r"^(\d{1,2})\s+([A-Za-z]+)\.?\s+(\d{4})$", text.strip()
        )
        if m:
            month = _MONTH_NAMES.get(m.group(2).lower().rstrip("."))
            if month:
                return f"{int(m.group(3)):04d}-{month:02d}-{int(m.group(1)):02d}"

    except (ValueError, AttributeError):
        pass
    return ""


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------


def extract_fields_uie(text: str, page_num: int) -> list:
    """Extract structured fields using PaddleNLP UIE.

    Args:
        text: Page text to analyze.
        page_num: Page number for field attribution.

    Returns:
        List of ExtractedField objects. Empty list if UIE is unavailable.
    """
    engine = _get_uie_engine()
    if engine is None:
        return []

    fields = []
    try:
        results = engine(text)
        # UIE returns a list of dicts; each dict maps schema labels to lists of
        # extraction results: [{"Date": [{"text": "...", "start": N, ...}], ...}]
        for result_dict in results:
            for label, extractions in result_dict.items():
                field_type = _normalize_uie_type(label)
                for item in extractions:
                    fields.append(ExtractedField(
                        field_type=field_type,
                        text=item.get("text", "").strip(),
                        confidence=round(item.get("probability", 0.0), 4),
                        page_num=page_num,
                        start=item.get("start", 0),
                        end=item.get("end", 0),
                        extraction_method="uie",
                    ))
    except Exception as e:
        logger.warning("UIE extraction failed for page %d: %s", page_num, e)

    return fields


def extract_fields_regex(text: str, page_num: int) -> list:
    """Extract structured fields using regex patterns.

    Works independently of PaddleNLP -- always available.

    Args:
        text: Page text to analyze.
        page_num: Page number for field attribution.

    Returns:
        List of ExtractedField objects.
    """
    fields = []

    # --- Dates ---
    for pattern, label in [
        (_DATE_ISO, "date"),
        (_DATE_US, "date"),
        (_DATE_WRITTEN, "date"),
    ]:
        for match in pattern.finditer(text):
            raw = match.group(1)
            fields.append(ExtractedField(
                field_type=label,
                text=raw,
                confidence=1.0,
                page_num=page_num,
                start=match.start(1),
                end=match.end(1),
                extraction_method="regex",
                normalized_value=_normalize_date(raw),
            ))

    # --- Amounts ---
    for match in _AMOUNT.finditer(text):
        fields.append(ExtractedField(
            field_type="amount",
            text=match.group(0),
            confidence=1.0,
            page_num=page_num,
            start=match.start(),
            end=match.end(),
            extraction_method="regex",
        ))

    # --- Phone numbers ---
    for match in _PHONE.finditer(text):
        fields.append(ExtractedField(
            field_type="phone_number",
            text=match.group(0),
            confidence=1.0,
            page_num=page_num,
            start=match.start(),
            end=match.end(),
            extraction_method="regex",
        ))

    # --- Email addresses ---
    for match in _EMAIL.finditer(text):
        fields.append(ExtractedField(
            field_type="email_address",
            text=match.group(0),
            confidence=1.0,
            page_num=page_num,
            start=match.start(),
            end=match.end(),
            extraction_method="regex",
        ))

    # --- Reference numbers ---
    for match in _REFERENCE.finditer(text):
        fields.append(ExtractedField(
            field_type="reference_number",
            text=match.group(0),
            confidence=1.0,
            page_num=page_num,
            start=match.start(),
            end=match.end(),
            extraction_method="regex",
        ))

    return fields


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _deduplicate_fields(fields: list) -> list:
    """Remove duplicate fields where UIE and regex found the same value.

    When two fields have the same field_type and overlapping character ranges,
    the UIE result is kept (higher quality) and the regex result is dropped.

    Args:
        fields: List of ExtractedField objects (mixed UIE + regex).

    Returns:
        Deduplicated list of ExtractedField objects.
    """
    if not fields:
        return []

    # Separate UIE and regex results
    uie_fields = [f for f in fields if f.extraction_method == "uie"]
    regex_fields = [f for f in fields if f.extraction_method != "uie"]

    # Keep all UIE fields; only keep regex fields that do not overlap with UIE
    kept = list(uie_fields)
    for rf in regex_fields:
        overlaps = False
        for uf in uie_fields:
            if rf.field_type == uf.field_type:
                # Check character range overlap
                if rf.start < uf.end and rf.end > uf.start:
                    overlaps = True
                    break
        if not overlaps:
            kept.append(rf)

    return kept


# ---------------------------------------------------------------------------
# Page-level extraction
# ---------------------------------------------------------------------------


def extract_page_fields(
    text: str,
    page_num: int,
    use_uie: bool = True,
    paddle_lines: Optional[list] = None,
    page_image=None,
) -> PageExtraction:
    """Extract all structured fields from a single page.

    Supports three extraction tiers controlled by the ``EXTRACTION_MODE``
    env var:
    - ``regex``    -- regex only
    - ``uie``      -- UIE + regex (default, existing behavior)
    - ``semantic`` -- LayoutLMv3 + regex
    - ``all``      -- UIE + LayoutLMv3 + regex

    Args:
        text: Page text to analyze.
        page_num: Page number for field attribution.
        use_uie: Whether to attempt UIE extraction (default True).
        paddle_lines: Optional PaddleOCR line tuples for semantic extraction.
        page_image: Optional PIL.Image for semantic extraction.

    Returns:
        PageExtraction with fields populated.
    """
    mode = EXTRACTION_MODE

    # Determine which tiers to run
    run_uie = use_uie and mode in ("uie", "all")
    run_semantic = mode in ("semantic", "all") and paddle_lines and page_image
    # Tier 1: UIE extraction
    uie_fields = []
    if run_uie:
        uie_fields = extract_fields_uie(text, page_num)

    # Tier 2: Regex extraction (always runs)
    regex_fields = extract_fields_regex(text, page_num)

    # Combine UIE + regex with existing deduplication
    if uie_fields:
        all_fields = _deduplicate_fields(uie_fields + regex_fields)
    else:
        all_fields = regex_fields

    field_dicts = [_field_to_dict(f) for f in all_fields]

    # Tier 3: Semantic (LayoutLMv3) extraction
    if run_semantic:
        try:
            from semantic_extraction import (
                extract_semantic_fields,
                merge_with_existing_extraction,
            )

            semantic_result = extract_semantic_fields(
                paddle_lines, page_image, page_num
            )
            if semantic_result and semantic_result.entities:
                field_dicts = merge_with_existing_extraction(
                    semantic_result, field_dicts
                )
        except ImportError:
            logger.debug(
                "semantic_extraction module not available; skipping semantic tier"
            )
        except Exception as exc:
            logger.warning(
                "Semantic extraction failed for page %d: %s", page_num, exc
            )

    return PageExtraction(
        page_num=page_num,
        fields=field_dicts,
    )


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


def _field_to_dict(f) -> dict:
    """Convert an ExtractedField (dataclass or dict) to a plain dict."""
    if isinstance(f, ExtractedField):
        return {
            "field_type": f.field_type,
            "text": f.text,
            "confidence": f.confidence,
            "page_num": f.page_num,
            "start": f.start,
            "end": f.end,
            "extraction_method": f.extraction_method,
            "normalized_value": f.normalized_value,
        }
    if isinstance(f, dict):
        return f
    return {}


def finalize_extraction(doc_ext: DocumentExtraction) -> DocumentExtraction:
    """Compute summary statistics from page-level extraction data.

    Args:
        doc_ext: DocumentExtraction with pages already populated.

    Returns:
        The same DocumentExtraction instance with summary fields filled in.
    """
    type_counts = {}
    total = 0
    methods_seen = set()

    for page_data in doc_ext.pages:
        if isinstance(page_data, PageExtraction):
            page_fields = page_data.fields
        elif isinstance(page_data, dict):
            page_fields = page_data.get("fields", [])
        else:
            continue

        for f in page_fields:
            f_dict = _field_to_dict(f)
            f_type = f_dict.get("field_type", "unknown")
            f_method = f_dict.get("extraction_method", "")
            total += 1
            type_counts[f_type] = type_counts.get(f_type, 0) + 1
            if f_method:
                methods_seen.add(f_method)

    doc_ext.total_fields = total
    doc_ext.field_type_counts = dict(sorted(type_counts.items()))

    # Determine extraction engine label
    has_uie = "uie" in methods_seen
    has_semantic = "semantic" in methods_seen
    has_regex = "regex" in methods_seen
    non_regex_count = sum([has_uie, has_semantic])

    if non_regex_count >= 2 or (non_regex_count >= 1 and has_regex):
        doc_ext.extraction_engine = "hybrid"
    elif has_uie:
        doc_ext.extraction_engine = "uie"
    elif has_semantic:
        doc_ext.extraction_engine = "semantic"
    elif has_regex:
        doc_ext.extraction_engine = "regex"
    else:
        doc_ext.extraction_engine = ""

    return doc_ext


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def write_extraction_json(
    doc_ext: DocumentExtraction,
    output_folder: str,
    subfolder: str,
    pipeline_version: str,
) -> Optional[str]:
    """Write .extraction.json sidecar file.

    Output to EXPORT/EXTRACTION/<subfolder>/<name>.extraction.json

    Args:
        doc_ext: Finalized DocumentExtraction dataclass.
        output_folder: Root output directory (e.g. /app/ocr_output).
        subfolder: Relative subfolder path mirroring source structure.
        pipeline_version: Pipeline version string for metadata.

    Returns:
        Path to the written JSON file, or None on failure.
    """
    try:
        extraction_dir = os.path.join(output_folder, "EXPORT", "EXTRACTION")
        if subfolder and subfolder != ".":
            safe_parts = [
                sanitize_path_segment(p)
                for p in subfolder.replace("\\", "/").split("/")
                if p
            ]
            target_dir = (
                os.path.join(extraction_dir, *safe_parts)
                if safe_parts
                else extraction_dir
            )
        else:
            target_dir = extraction_dir

        # Path traversal protection
        resolved = os.path.realpath(target_dir)
        if not resolved.startswith(os.path.realpath(extraction_dir)):
            logger.error(
                "Path traversal blocked in extraction output: %s", subfolder
            )
            return None

        os.makedirs(target_dir, exist_ok=True)

        base_name = build_sidecar_base_name(doc_ext.source_file)
        json_path = os.path.join(target_dir, f"{base_name}.extraction.json")

        # Determine UIE model name for metadata
        uie_model = os.environ.get("UIE_MODEL", "uie-base") if _PADDLENLP_AVAILABLE else "n/a"

        # Build page-level output
        pages_output = []
        for page_data in doc_ext.pages:
            if isinstance(page_data, PageExtraction):
                p_num = page_data.page_num
                p_fields = [_field_to_dict(f) for f in page_data.fields]
            elif isinstance(page_data, dict):
                p_num = page_data.get("page_num", 0)
                raw_fields = page_data.get("fields", [])
                p_fields = [_field_to_dict(f) for f in raw_fields]
            else:
                continue
            pages_output.append({
                "page_num": p_num,
                "fields": p_fields,
            })

        report = {
            "schema_version": "1.0",
            "document_id": doc_ext.document_id,
            "source_file": doc_ext.source_file,
            "processing": {
                "extraction_engine": doc_ext.extraction_engine,
                "uie_model": uie_model,
                "pipeline_version": pipeline_version,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
            },
            "extraction_summary": {
                "total_fields": doc_ext.total_fields,
                "field_type_counts": doc_ext.field_type_counts,
                "extraction_engine": doc_ext.extraction_engine,
            },
            "pages": pages_output,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return json_path
    except Exception as e:
        logger.error(
            "Failed to write extraction JSON for %s: %s", doc_ext.document_id, e
        )
        return None
