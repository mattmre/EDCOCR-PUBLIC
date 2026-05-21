"""Specialist routing from document classification output.

Maps classification labels to specialized extraction configurations, enabling
document-type-specific field extraction with custom regex patterns.  When
classification identifies a document as an invoice, contract, medical record,
etc., the specialist router selects a tailored extraction config that targets
the fields most relevant to that document type.

The module is opt-in and augments (never replaces) the generic extraction
pipeline.  When no specialist matches -- or confidence is below threshold --
the generic extraction runs unchanged.

Configuration:
- ``ENABLE_SPECIALIST_ROUTING``: master toggle (default False, set via
  ``--enable-specialist-routing`` or env var)
- ``SPECIALIST_CONFIG_PATH``: optional path to a JSON file with custom
  specialist definitions that augment the built-in specialists

Output is written alongside extraction output to
``EXPORT/EXTRACTION/<subfolder>/<name>.specialist.json``.
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPECIALIST_CONFIG_PATH = os.environ.get("SPECIALIST_CONFIG_PATH", "").strip()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SpecialistConfig:
    """Configuration for a document type specialist."""

    doc_type: str  # e.g., "invoice", "contract", "medical_record"
    extraction_fields: list  # field names to extract for this type
    confidence_threshold: float = 0.7  # minimum classification confidence
    custom_patterns: dict = field(default_factory=dict)  # regex patterns per field
    source: str = "builtin"  # "builtin" or "custom"


@dataclass
class SpecialistField:
    """A single field extracted by a specialist."""

    field_name: str
    text: str
    confidence: float = 1.0
    page_num: int = 0
    start: int = 0
    end: int = 0
    extraction_method: str = "specialist_regex"


@dataclass
class SpecialistResult:
    """Result of specialist extraction for a document."""

    document_id: str
    source_file: str
    doc_type: str  # the matched specialist document type
    specialist_source: str = "builtin"
    confidence: float = 0.0
    fields: list = field(default_factory=list)  # List[dict] from SpecialistField
    total_fields: int = 0
    field_counts: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in specialist regex patterns
# ---------------------------------------------------------------------------

# Invoice patterns
_INVOICE_NUMBER = re.compile(
    r"(?:invoice|inv)[\s#:.-]*(\w{2,}(?:[-/]\w+)*)", re.IGNORECASE
)
_DUE_DATE = re.compile(
    r"(?:due\s+date|payment\s+due|due\s+by)[\s:]*"
    r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_VENDOR = re.compile(
    r"(?:from|vendor|supplier|sold\s+by|billed?\s+from)[\s:]*([A-Z][A-Za-z0-9 &.,'-]{2,60})",
    re.IGNORECASE,
)
_TOTAL_AMOUNT = re.compile(
    r"(?:total|amount\s+due|grand\s+total|balance\s+due)[\s:]*"
    r"(\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
    re.IGNORECASE,
)
_TAX_AMOUNT = re.compile(
    r"(?:tax|vat|gst|hst|sales\s+tax)[\s:]*"
    r"(\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
    re.IGNORECASE,
)
_LINE_ITEM = re.compile(
    r"(\d+)\s+(.{5,60}?)\s+(\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*$",
    re.MULTILINE,
)

# Contract patterns
_PARTIES = re.compile(
    r"(?:between|party|parties|hereinafter)[\s:]*"
    r"([A-Z][A-Za-z0-9 &.,'-]{2,80})",
    re.IGNORECASE,
)
_EFFECTIVE_DATE = re.compile(
    r"(?:effective\s+date|commencement\s+date|dated?\s+as\s+of)[\s:]*"
    r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_TERM = re.compile(
    r"(?:term|duration|period)(?:\s+(?:of|is|shall\s+be|:)[\s\w]*?)?\s+(\d+\s*(?:year|month|day|week)s?)",
    re.IGNORECASE,
)
_GOVERNING_LAW = re.compile(
    r"(?:governing\s+law|governed\s+by|laws?\s+of(?:\s+the\s+(?:State|Commonwealth)\s+of)?)"
    r"[\s:]*([A-Z][A-Za-z ]{2,40})",
    re.IGNORECASE,
)

# Medical record patterns
_PATIENT_NAME = re.compile(
    r"(?:patient(?:\s+name)?|name\s+of\s+patient)[\s:]*"
    r"([A-Z][A-Za-z '-]{2,50})",
    re.IGNORECASE,
)
_DOB = re.compile(
    r"(?:d\.?o\.?b\.?|date\s+of\s+birth|birth\s*date)[\s:]*"
    r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})",
    re.IGNORECASE,
)
_MRN = re.compile(
    r"(?:mrn|medical\s+record\s+(?:number|no\.?|#)|patient\s+id)[\s#:]*"
    r"(\d{4,12})",
    re.IGNORECASE,
)
_DIAGNOSIS = re.compile(
    r"(?:diagnosis|dx|assessment|impression)[\s:]+(.{5,120})",
    re.IGNORECASE,
)
_MEDICATION = re.compile(
    r"(?:medication|rx|prescribed?|drug)[\s:]+([A-Za-z][A-Za-z0-9 /-]{2,60})",
    re.IGNORECASE,
)
_PROVIDER = re.compile(
    r"(?:provider|physician|doctor|attending|dr\.?)[\s:]*"
    r"([A-Z][A-Za-z '-]{2,50})",
    re.IGNORECASE,
)

# Legal filing patterns
_CASE_NUMBER = re.compile(
    r"(?:case\s+(?:no\.?|number|#)|docket\s+(?:no\.?|#))[\s:]*"
    r"([A-Za-z0-9][\w-]{3,20})",
    re.IGNORECASE,
)
_COURT = re.compile(
    r"(?:(?:in\s+the\s+)?(?:united\s+states\s+)?(?:district|superior|circuit|"
    r"supreme|county|municipal|bankruptcy)\s+court)"
    r"(?:\s+(?:of|for)\s+(?:the\s+)?)?([A-Za-z ]{2,60})?",
    re.IGNORECASE,
)
_FILING_DATE = re.compile(
    r"(?:filed?(?:\s+date)?|date\s+filed)[\s:]*"
    r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_JUDGE = re.compile(
    r"(?:judge|hon\.?|honorable|justice|magistrate)[\s:]*"
    r"([A-Z][A-Za-z '-]{2,50})",
    re.IGNORECASE,
)

# Correspondence patterns
_SENDER = re.compile(
    r"(?:from|sender)[\s:]*([A-Z][A-Za-z0-9 &.,'-]{2,60})",
    re.IGNORECASE,
)
_RECIPIENT = re.compile(
    r"(?:to|recipient|dear)[\s:]*([A-Z][A-Za-z0-9 &.,'-]{2,60})",
    re.IGNORECASE,
)
_SUBJECT = re.compile(
    r"(?:subject|re|regarding)[\s:]+(.{3,120})",
    re.IGNORECASE,
)
_REFERENCE_NUMBER = re.compile(
    r"(?:ref(?:erence)?[\s.#:]*(?:no\.?|number)?|your\s+ref)[\s#:]*"
    r"([A-Za-z0-9][\w-]{2,20})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Built-in pattern registry (field_name -> compiled regex)
# ---------------------------------------------------------------------------

_BUILTIN_PATTERNS = {
    "invoice": {
        "invoice_number": _INVOICE_NUMBER,
        "due_date": _DUE_DATE,
        "vendor": _VENDOR,
        "total_amount": _TOTAL_AMOUNT,
        "tax": _TAX_AMOUNT,
        "line_item": _LINE_ITEM,
    },
    "contract": {
        "parties": _PARTIES,
        "effective_date": _EFFECTIVE_DATE,
        "term": _TERM,
        "governing_law": _GOVERNING_LAW,
    },
    "medical_record": {
        "patient_name": _PATIENT_NAME,
        "dob": _DOB,
        "mrn": _MRN,
        "diagnosis": _DIAGNOSIS,
        "medications": _MEDICATION,
        "provider": _PROVIDER,
    },
    "legal_filing": {
        "case_number": _CASE_NUMBER,
        "court": _COURT,
        "filing_date": _FILING_DATE,
        "judge": _JUDGE,
        "parties": _PARTIES,
    },
    "letter": {
        "sender": _SENDER,
        "recipient": _RECIPIENT,
        "subject": _SUBJECT,
        "reference_number": _REFERENCE_NUMBER,
    },
    "memo": {
        "sender": _SENDER,
        "recipient": _RECIPIENT,
        "subject": _SUBJECT,
        "reference_number": _REFERENCE_NUMBER,
    },
    "receipt": {
        "total_amount": _TOTAL_AMOUNT,
        "tax": _TAX_AMOUNT,
        "vendor": _VENDOR,
    },
}


# ---------------------------------------------------------------------------
# SpecialistRouter
# ---------------------------------------------------------------------------


class SpecialistRouter:
    """Routes documents to specialized extraction configurations.

    The router maintains a registry of specialist configs, each mapping a
    document type to the fields and regex patterns used for extraction.

    Built-in specialists are loaded automatically.  Custom specialists can
    be added via :meth:`add_specialist` or loaded from a JSON config file.
    """

    def __init__(self, config_path: str = ""):
        self._specialists: dict[str, SpecialistConfig] = {}
        self._compiled_custom: dict[str, dict[str, re.Pattern]] = {}
        self._load_defaults()
        if config_path:
            self._load_custom(config_path)

    # -- loading --

    def _load_defaults(self):
        """Register built-in specialist configs."""
        for doc_type, patterns in _BUILTIN_PATTERNS.items():
            self._specialists[doc_type] = SpecialistConfig(
                doc_type=doc_type,
                extraction_fields=list(patterns.keys()),
                confidence_threshold=0.7,
                source="builtin",
            )

    def _load_custom(self, config_path: str):
        """Load custom specialist configs from a JSON file.

        Custom configs augment built-in specialists.  If a custom config
        defines a doc_type that already exists, it merges new fields and
        patterns into the existing specialist.
        """
        try:
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.warning("Specialist config file not found: %s", config_path)
            return
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON in specialist config %s: %s", config_path, exc)
            return
        except Exception as exc:
            logger.warning("Failed to read specialist config %s: %s", config_path, exc)
            return

        specialists = data.get("specialists", {})
        if not isinstance(specialists, dict):
            logger.warning("specialists key in config must be a dict; skipping")
            return

        for doc_type, spec in specialists.items():
            if not isinstance(spec, dict):
                logger.warning("Skipping invalid specialist entry: %s", doc_type)
                continue

            extraction_fields = spec.get("extraction_fields", [])
            if not isinstance(extraction_fields, list):
                extraction_fields = []

            confidence_threshold = spec.get("confidence_threshold", 0.7)
            try:
                confidence_threshold = float(confidence_threshold)
            except (ValueError, TypeError):
                confidence_threshold = 0.7

            custom_patterns = spec.get("custom_patterns", {})
            if not isinstance(custom_patterns, dict):
                custom_patterns = {}

            # Compile custom regex patterns
            compiled = {}
            for field_name, pattern_str in custom_patterns.items():
                try:
                    compiled[field_name] = re.compile(pattern_str, re.IGNORECASE)
                except re.error as regex_err:
                    logger.warning(
                        "Invalid regex for %s/%s: %s", doc_type, field_name, regex_err
                    )

            # Merge with existing specialist if present
            existing = self._specialists.get(doc_type)
            if existing is not None:
                merged_fields = list(
                    dict.fromkeys(existing.extraction_fields + extraction_fields)
                )
                existing.extraction_fields = merged_fields
                existing.custom_patterns.update(custom_patterns)
                existing.source = "builtin+custom"
                # Merge compiled patterns
                existing_compiled = self._compiled_custom.get(doc_type, {})
                existing_compiled.update(compiled)
                self._compiled_custom[doc_type] = existing_compiled
            else:
                self._specialists[doc_type] = SpecialistConfig(
                    doc_type=doc_type,
                    extraction_fields=extraction_fields,
                    confidence_threshold=confidence_threshold,
                    custom_patterns=custom_patterns,
                    source="custom",
                )
                self._compiled_custom[doc_type] = compiled

    # -- public API --

    def add_specialist(self, config: SpecialistConfig):
        """Register a specialist config programmatically."""
        self._specialists[config.doc_type] = config

    @property
    def specialists(self) -> dict:
        """Return the internal specialist registry (read-only copy)."""
        return dict(self._specialists)

    def route(self, classification_result) -> Optional[SpecialistConfig]:
        """Select a specialist based on classification output.

        Accepts either a ``DocumentClassification`` dataclass or a plain dict
        with ``document_type`` and ``document_confidence`` keys.

        Returns the matched ``SpecialistConfig``, or ``None`` if no specialist
        matches or confidence is below the specialist's threshold.
        """
        if classification_result is None:
            return None

        # Extract type and confidence from either dataclass or dict
        if isinstance(classification_result, dict):
            doc_type = classification_result.get("document_type", "other")
            confidence = float(classification_result.get("document_confidence", 0.0))
        else:
            doc_type = getattr(classification_result, "document_type", "other")
            confidence = float(
                getattr(classification_result, "document_confidence", 0.0)
            )

        specialist = self._specialists.get(doc_type)
        if specialist is None:
            logger.debug("No specialist registered for doc_type=%s", doc_type)
            return None

        if confidence < specialist.confidence_threshold:
            logger.debug(
                "Classification confidence %.3f below threshold %.3f for %s",
                confidence,
                specialist.confidence_threshold,
                doc_type,
            )
            return None

        return specialist

    def extract_specialized(
        self,
        doc_text: str,
        config: SpecialistConfig,
        page_num: int = 0,
    ) -> list:
        """Run specialized extraction for the matched document type.

        Uses the built-in patterns for the specialist's doc_type, overlaid
        with any custom patterns.  Returns a list of ``SpecialistField``
        objects.

        Args:
            doc_text: Full document text (or page text) to extract from.
            config: The specialist config to use.
            page_num: Page number for field attribution (0 = whole document).

        Returns:
            List of SpecialistField objects.
        """
        if not doc_text or not config:
            return []

        fields = []

        # Get built-in patterns for this doc_type
        builtin = _BUILTIN_PATTERNS.get(config.doc_type, {})
        # Get custom compiled patterns
        custom = self._compiled_custom.get(config.doc_type, {})

        # Merge: custom patterns override built-in for same field name
        all_patterns = dict(builtin)
        all_patterns.update(custom)

        for field_name in config.extraction_fields:
            pattern = all_patterns.get(field_name)
            if pattern is None:
                continue

            for match in pattern.finditer(doc_text):
                # Use group(1) if available, otherwise group(0)
                try:
                    text = match.group(1).strip()
                except (IndexError, AttributeError):
                    text = match.group(0).strip()

                if not text:
                    continue

                fields.append(SpecialistField(
                    field_name=field_name,
                    text=text,
                    confidence=1.0,
                    page_num=page_num,
                    start=match.start(),
                    end=match.end(),
                    extraction_method="specialist_regex",
                ))

        return fields

    def process_document(
        self,
        classification_result,
        doc_text: str,
        document_id: str = "",
        source_file: str = "",
    ) -> Optional[SpecialistResult]:
        """End-to-end specialist processing: route + extract.

        Args:
            classification_result: DocumentClassification or dict.
            doc_text: Full document text to extract from.
            document_id: Document identifier for output.
            source_file: Source file path for output.

        Returns:
            SpecialistResult if a specialist matched, None otherwise.
        """
        config = self.route(classification_result)
        if config is None:
            return None

        fields = self.extract_specialized(doc_text, config)

        # Build field dicts and counts
        field_dicts = [_specialist_field_to_dict(f) for f in fields]
        field_counts = {}
        for f in fields:
            field_counts[f.field_name] = field_counts.get(f.field_name, 0) + 1

        if isinstance(classification_result, dict):
            confidence = float(classification_result.get("document_confidence", 0.0))
        else:
            confidence = float(
                getattr(classification_result, "document_confidence", 0.0)
            )

        return SpecialistResult(
            document_id=document_id,
            source_file=source_file,
            doc_type=config.doc_type,
            specialist_source=config.source,
            confidence=round(confidence, 4),
            fields=field_dicts,
            total_fields=len(field_dicts),
            field_counts=dict(sorted(field_counts.items())),
        )


# ---------------------------------------------------------------------------
# Field serialization
# ---------------------------------------------------------------------------


def _specialist_field_to_dict(f: SpecialistField) -> dict:
    """Convert a SpecialistField to a plain dict."""
    return {
        "field_name": f.field_name,
        "text": f.text,
        "confidence": f.confidence,
        "page_num": f.page_num,
        "start": f.start,
        "end": f.end,
        "extraction_method": f.extraction_method,
    }


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def write_specialist_json(
    result: SpecialistResult,
    output_folder: str,
    subfolder: str,
    pipeline_version: str,
) -> Optional[str]:
    """Write ``.specialist.json`` sidecar file.

    Output to ``EXPORT/EXTRACTION/<subfolder>/<name>.specialist.json``.

    Args:
        result: SpecialistResult from process_document.
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
                "Path traversal blocked in specialist output: %s", subfolder
            )
            return None

        os.makedirs(target_dir, exist_ok=True)

        base_name = build_sidecar_base_name(result.source_file)
        json_path = os.path.join(target_dir, f"{base_name}.specialist.json")

        report = {
            "schema_version": "1.0",
            "document_id": result.document_id,
            "source_file": result.source_file,
            "processing": {
                "specialist_doc_type": result.doc_type,
                "specialist_source": result.specialist_source,
                "classification_confidence": result.confidence,
                "pipeline_version": pipeline_version,
                "timestamp": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(timespec="milliseconds"),
            },
            "specialist_summary": {
                "doc_type": result.doc_type,
                "total_fields": result.total_fields,
                "field_counts": result.field_counts,
            },
            "fields": result.fields,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return json_path
    except Exception as exc:
        logger.error(
            "Failed to write specialist JSON for %s: %s",
            result.document_id,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

# Singleton router instance, lazily loaded on first use
_router: Optional[SpecialistRouter] = None


def get_router() -> SpecialistRouter:
    """Get or create the module-level SpecialistRouter singleton."""
    global _router
    if _router is None:
        _router = SpecialistRouter(config_path=SPECIALIST_CONFIG_PATH)
    return _router


def reset_router():
    """Reset the singleton router (useful for testing)."""
    global _router
    _router = None
