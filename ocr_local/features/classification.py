"""Document classification for forensic OCR pipeline.

Categorizes documents by type (invoice, contract, form, letter, etc.) using
text pattern matching, optional layout feature analysis, and optional ML-based
classification via LayoutLMv3.

Output: EXPORT/CLASSIFICATION/<subfolder>/<document_name>.classification.json

Three-tier approach:
- Tier 1: Text pattern matching (always available, uses OCR text)
- Tier 2: Layout feature analysis (requires DocIntel output)
- Tier 3: ML classification via LayoutLMv3 (optional, requires torch + transformers)

Classification mode is controlled by the CLASSIFICATION_MODE env var:
- "heuristic" (default): Tier 1 + Tier 2 ensemble (existing behavior)
- "ml": ML-only classification via LayoutLMv3
- "ensemble": Weighted hybrid of heuristic + ML (ML=0.7, heuristic=0.3)
"""

import datetime
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Optional

from ocr_distributed.ocr_utils import (
    build_sidecar_base_name,
    sanitize_path_segment,
)

__all__ = [
    "CLASSIFICATION_MODE",
    "ML_CLASSIFICATION_MODEL",
    "ML_CLASSIFICATION_CONFIDENCE_THRESHOLD",
    "CLASSIFICATION_MULTI_LABEL_THRESHOLD",
    "CLASSIFICATION_MULTI_LABEL_MAX_LABELS",
    "CLASSIFICATION_PROFILE_PATH",
    "DOCUMENT_TYPES",
    "DOCUMENT_TYPES_EXTENDED",
    "PageClassification",
    "DocumentClassification",
    "MLDocumentClassifier",
    "classify_page_by_text",
    "classify_page_by_layout",
    "classify_page_ensemble",
    "classify_page_ml",
    "classify_page_hybrid",
    "get_ml_classifier",
    "finalize_classification",
    "write_classification_json",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ML classification configuration (env-driven, opt-in)
# ---------------------------------------------------------------------------


# Canonical env-parsing helpers (DRY consolidation)
from ocr_distributed.ocr_utils import get_env_float as _get_env_float
from ocr_distributed.ocr_utils import get_env_int as _get_env_int

CLASSIFICATION_MODE = os.environ.get("CLASSIFICATION_MODE", "heuristic").lower().strip()
ML_CLASSIFICATION_MODEL = os.environ.get(
    "ML_CLASSIFICATION_MODEL", "microsoft/layoutlmv3-base"
)
ML_CLASSIFICATION_CONFIDENCE_THRESHOLD = _get_env_float(
    "ML_CLASSIFICATION_CONFIDENCE_THRESHOLD", 0.5
)
CLASSIFICATION_MULTI_LABEL_THRESHOLD = _get_env_float(
    "CLASSIFICATION_MULTI_LABEL_THRESHOLD", 0.3
)
CLASSIFICATION_MULTI_LABEL_MAX_LABELS = _get_env_int(
    "CLASSIFICATION_MULTI_LABEL_MAX_LABELS", 3, max_val=10
)
CLASSIFICATION_PROFILE_PATH = os.environ.get(
    "CLASSIFICATION_PROFILE_PATH", ""
).strip()
_CLASSIFICATION_PROFILE_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Document type taxonomy
# ---------------------------------------------------------------------------

DOCUMENT_TYPES = [
    "invoice", "contract", "letter", "form", "report",
    "memo", "receipt", "handwritten", "photograph", "other",
]

# Extended taxonomy for ML classification (superset of DOCUMENT_TYPES)
DOCUMENT_TYPES_EXTENDED = [
    "invoice", "contract", "letter", "form", "report",
    "memo", "receipt", "handwritten_note", "photograph", "other",
    "scientific_paper", "legal_filing", "email_printout",
    "spreadsheet", "presentation", "specification",
    "resume", "medical_record", "government_form",
]

# Mapping from extended ML types back to base types for backward compatibility
_ML_TO_BASE_TYPE = {
    "handwritten_note": "handwritten",
    "scientific_paper": "report",
    "legal_filing": "contract",
    "email_printout": "letter",
    "spreadsheet": "report",
    "presentation": "report",
    "specification": "report",
    "resume": "form",
    "medical_record": "form",
    "government_form": "form",
}

# ---------------------------------------------------------------------------
# Tier 1: Text pattern rules (keyword matching)
# ---------------------------------------------------------------------------

_TEXT_RULES = {
    "invoice": {
        "keywords": [
            r"\binvoice\b", r"\bbill\s+to\b", r"\bamount\s+due\b",
            r"\bpayment\s+terms\b", r"\btotal\s*:", r"\bsubtotal\b",
            r"\binv[#\-]\s*\d+", r"\bdue\s+date\b",
        ],
        "weight": 1.0,
    },
    "contract": {
        "keywords": [
            r"\bagreement\b", r"\bhereby\b", r"\bparties\b",
            r"\bwhereas\b", r"\bwitnesseth\b", r"\bexecuted\b",
            r"\bterms\s+and\s+conditions\b", r"\bindemnif",
        ],
        "weight": 1.0,
    },
    "letter": {
        "keywords": [
            r"\bdear\b", r"\bsincerely\b", r"\bregards\b",
            r"\bto\s+whom\s+it\s+may\s+concern\b", r"\byours\s+truly\b",
        ],
        "weight": 1.0,
    },
    "form": {
        "keywords": [
            r"\bfill\s+in\b", r"\bplease\s+complete\b", r"\bapplicant\b",
            r"\bcheckbox\b", r"\bsignature\s+(?:line|block|below)\b",
            r"\b(?:first|last)\s+name\s*:", r"\bdate\s+of\s+birth\b",
        ],
        "weight": 1.0,
    },
    "report": {
        "keywords": [
            r"\bexecutive\s+summary\b", r"\bfindings\b", r"\bmethodology\b",
            r"\bconclusion\b", r"\bappendix\b", r"\babstract\b",
        ],
        "weight": 1.0,
    },
    "memo": {
        "keywords": [
            r"\bmemorandum\b", r"\bmemo\b",
        ],
        "weight": 1.0,
        "co_locate": [r"\bto\s*:", r"\bfrom\s*:", r"\bsubject\s*:"],
        "co_locate_min": 2,
    },
    "receipt": {
        "keywords": [
            r"\breceipt\b", r"\btransaction\b", r"\bpaid\b",
            r"\bchange\s+due\b", r"\bcard\s+ending\b",
        ],
        "weight": 1.0,
    },
}

# Pre-compile all regex patterns at module level for efficiency
_COMPILED_RULES = {}
for _doc_type, _rule in _TEXT_RULES.items():
    _COMPILED_RULES[_doc_type] = {
        "patterns": [re.compile(p, re.IGNORECASE) for p in _rule["keywords"]],
        "weight": _rule["weight"],
    }
    if "co_locate" in _rule:
        _COMPILED_RULES[_doc_type]["co_locate"] = [
            re.compile(p, re.IGNORECASE) for p in _rule["co_locate"]
        ]
        _COMPILED_RULES[_doc_type]["co_locate_min"] = _rule["co_locate_min"]

# Pattern for detecting monetary values in layout text (used by Tier 2)
_MONEY_PATTERN = re.compile(
    r"(?:\$|USD|EUR|GBP)\s*\d|total\s*:|subtotal|amount\s+due", re.IGNORECASE
)

# Ensemble weights
_TEXT_WEIGHT = 0.6
_LAYOUT_WEIGHT = 0.4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PageClassification:
    """Classification result for a single page."""

    page_num: int
    predicted_type: str = "other"
    confidence: float = 0.0
    method: str = ""
    type_scores: dict = field(default_factory=dict)
    is_handwritten: bool = False
    profile_matches: list = field(default_factory=list)


@dataclass
class DocumentClassification:
    """Classification result for an entire document."""

    document_id: str
    source_file: str
    pages: list = field(default_factory=list)
    document_type: str = "other"
    document_confidence: float = 0.0
    type_distribution: dict = field(default_factory=dict)
    document_type_scores: dict = field(default_factory=dict)
    document_labels: list = field(default_factory=list)
    custom_profile_matches: list = field(default_factory=list)


def _is_within_root(path: str, root: str) -> bool:
    """Return True when path resolves within the provided root directory."""
    normalized_path = os.path.normcase(os.path.realpath(path))
    normalized_root = os.path.normcase(os.path.realpath(root))
    return normalized_path == normalized_root or normalized_path.startswith(
        normalized_root + os.sep
    )


def _resolve_classification_profile_path(path: str) -> Optional[str]:
    """Resolve a profile path and reject traversal outside the repo/app root."""
    candidate = path
    if not os.path.isabs(candidate):
        candidate = os.path.join(_CLASSIFICATION_PROFILE_ROOT, candidate)

    try:
        resolved_path = os.path.realpath(candidate)
    except OSError as exc:
        logger.warning(
            "Unable to resolve classification profile path %s: %s", path, exc
        )
        return None

    if not _is_within_root(resolved_path, _CLASSIFICATION_PROFILE_ROOT):
        logger.warning(
            "Classification profile path outside allowed root blocked: %s", path
        )
        return None

    return resolved_path


def _read_classification_profile_payload(path: str):
    """Read and parse the classification profile payload from disk."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        logger.warning("Classification profile file not found: %s", path)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid classification profile JSON %s: %s", path, exc)
    except OSError as exc:
        logger.warning("Unable to read classification profile file %s: %s", path, exc)
    return None


def _parse_classification_profile(profile: dict, index: int) -> Optional[dict]:
    """Parse and validate a single profile entry."""
    if not isinstance(profile, dict):
        logger.warning(
            "Skipping invalid classification profile at index %d: expected object",
            index,
        )
        return None

    name = str(profile.get("name", "")).strip()
    if not name:
        logger.warning("Skipping unnamed classification profile at index %d", index)
        return None

    base_type = str(profile.get("base_type", "other")).strip().lower() or "other"
    keywords = [
        str(keyword).strip()
        for keyword in profile.get("keywords", [])
        if str(keyword).strip()
    ]
    if not keywords:
        logger.warning("Skipping classification profile %s without keywords", name)
        return None

    try:
        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in keywords]
    except re.error as exc:
        logger.warning(
            "Skipping classification profile %s due to invalid regex: %s",
            name,
            exc,
        )
        return None

    return {
        "name": name,
        "base_type": base_type,
        "route": str(profile.get("route", "")).strip(),
        "weight": max(float(profile.get("weight", 1.0)), 0.1),
        "patterns": compiled,
    }


def _load_classification_profiles(path: str) -> list:
    """Load optional customer classification profiles from JSON."""
    if not path:
        return []

    resolved_path = _resolve_classification_profile_path(path)
    if not resolved_path:
        return []

    payload = _read_classification_profile_payload(resolved_path)
    if payload is None:
        return []

    if not isinstance(payload, dict):
        logger.warning(
            "Invalid classification profile payload in %s: expected object",
            resolved_path,
        )
        return []

    raw_profiles = payload.get("profiles", [])
    if not isinstance(raw_profiles, list):
        logger.warning(
            "Invalid classification profile payload in %s: profiles must be a list",
            resolved_path,
        )
        return []

    profiles = []
    for index, profile in enumerate(raw_profiles, start=1):
        parsed_profile = _parse_classification_profile(profile, index)
        if parsed_profile is not None:
            profiles.append(parsed_profile)

    return profiles


_CLASSIFICATION_PROFILES = _load_classification_profiles(CLASSIFICATION_PROFILE_PATH)


def _match_classification_profiles(text: str) -> list:
    """Return matching customer profile labels for the provided text."""
    if not text or not text.strip():
        return []

    matches = []
    for profile in _CLASSIFICATION_PROFILES:
        patterns = profile["patterns"]
        hit_count = sum(1 for pattern in patterns if pattern.search(text))
        if hit_count <= 0:
            continue

        confidence = round(min((hit_count / len(patterns)) * profile["weight"], 1.0), 4)
        matches.append(
            {
                "name": profile["name"],
                "base_type": profile["base_type"],
                "route": profile["route"],
                "confidence": confidence,
            }
        )

    matches.sort(key=lambda item: (-item["confidence"], item["name"]))
    return matches


def _build_document_labels(
    document_type_scores: dict,
    primary_type: str,
) -> list:
    """Build a deterministic multi-label summary from document-level scores."""
    labels = []
    score_items = sorted(
        document_type_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    max_labels = max(1, CLASSIFICATION_MULTI_LABEL_MAX_LABELS)

    for label, confidence in score_items:
        if confidence < CLASSIFICATION_MULTI_LABEL_THRESHOLD and label != primary_type:
            continue
        labels.append(
            {
                "label": label,
                "confidence": round(confidence, 4),
                "source": "classification",
            }
        )
        if len(labels) >= max_labels:
            break

    if primary_type and not any(item["label"] == primary_type for item in labels):
        labels.insert(
            0,
            {
                "label": primary_type,
                "confidence": round(document_type_scores.get(primary_type, 0.0), 4),
                "source": "classification",
            },
        )

    deduped = []
    seen = set()
    for item in labels:
        key = item["label"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:max_labels]


# ---------------------------------------------------------------------------
# Tier 1: Text pattern classification
# ---------------------------------------------------------------------------


def classify_page_by_text(text: str, page_num: int) -> PageClassification:
    """Classify a page by matching OCR text against keyword patterns.

    Args:
        text: OCR text content of the page.
        page_num: Page number (1-based).

    Returns:
        PageClassification with method="text_rules".
    """
    result = PageClassification(page_num=page_num, method="text_rules")

    if not text or not text.strip():
        return result

    type_scores = {}

    for doc_type, rule in _COMPILED_RULES.items():
        patterns = rule["patterns"]
        if not patterns:
            continue

        match_count = sum(1 for p in patterns if p.search(text))

        # Special handling for memo: check co-located fields
        if doc_type == "memo" and "co_locate" in rule:
            co_matches = sum(1 for p in rule["co_locate"] if p.search(text))
            if co_matches >= rule["co_locate_min"]:
                match_count += co_matches

        if match_count > 0:
            total_patterns = len(patterns)
            if "co_locate" in rule:
                total_patterns += len(rule["co_locate"])
            score = match_count / total_patterns
            type_scores[doc_type] = round(score, 4)

    result.type_scores = type_scores

    if type_scores:
        best_type = max(type_scores, key=type_scores.get)
        result.predicted_type = best_type
        result.confidence = round(type_scores[best_type], 4)
    else:
        result.predicted_type = "other"
        result.confidence = 0.0

    profile_matches = _match_classification_profiles(text)
    if profile_matches:
        result.profile_matches = profile_matches
        for profile_match in profile_matches:
            base_type = profile_match["base_type"]
            profile_confidence = profile_match["confidence"]
            existing = type_scores.get(base_type, 0.0)
            if profile_confidence > existing:
                type_scores[base_type] = profile_confidence
        result.type_scores = dict(sorted(type_scores.items()))
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            result.predicted_type = best_type
            result.confidence = round(type_scores[best_type], 4)

    return result


# ---------------------------------------------------------------------------
# Tier 2: Layout feature classification
# ---------------------------------------------------------------------------


def classify_page_by_layout(
    layout_regions: list,
    tables: list,
    form_fields: list,
    page_num: int,
) -> PageClassification:
    """Classify a page using structural layout features from DocIntel.

    Args:
        layout_regions: List of region dicts with "type" key.
        tables: List of table dicts.
        form_fields: List of form field dicts.
        page_num: Page number (1-based).

    Returns:
        PageClassification with method="layout_features".
    """
    result = PageClassification(page_num=page_num, method="layout_features")

    # If no layout data available, return zero-confidence
    if not layout_regions and not tables and not form_fields:
        return result

    # Count region types
    region_counts = {}
    region_texts = []
    for region in (layout_regions or []):
        rtype = region.get("type", "unknown")
        region_counts[rtype] = region_counts.get(rtype, 0) + 1
        # Collect any text content from regions for money detection
        rtext = region.get("text", "")
        if rtext:
            region_texts.append(rtext)

    text_count = region_counts.get("text", 0)
    figure_count = region_counts.get("figure", 0)
    title_count = region_counts.get("title", 0)
    table_count = len(tables or [])
    field_count = len(form_fields or [])
    combined_text = " ".join(region_texts)

    type_scores = {}

    # Invoice / Receipt: tables with monetary content
    has_money = bool(_MONEY_PATTERN.search(combined_text))
    if table_count >= 1 and has_money:
        type_scores["invoice"] = 0.7
        type_scores["receipt"] = 0.5

    # Form: multiple form fields
    if field_count >= 2:
        type_scores["form"] = min(0.3 + field_count * 0.1, 0.9)

    # Photograph: more figures than text regions
    if figure_count > 0 and figure_count > text_count:
        type_scores["photograph"] = round(
            figure_count / max(figure_count + text_count, 1), 4
        )

    # Report: structured with titles and text blocks
    if title_count >= 2 and text_count >= 3:
        type_scores["report"] = 0.6

    # Tables without money context suggest report or form
    if table_count >= 1 and not has_money:
        existing_report = type_scores.get("report", 0.0)
        type_scores["report"] = max(existing_report, 0.4)

    result.type_scores = type_scores

    if type_scores:
        best_type = max(type_scores, key=type_scores.get)
        result.predicted_type = best_type
        result.confidence = round(type_scores[best_type], 4)
    else:
        result.predicted_type = "other"
        result.confidence = 0.0

    return result


# ---------------------------------------------------------------------------
# Ensemble classification
# ---------------------------------------------------------------------------


def _heuristic_ensemble(
    text_result: Optional[PageClassification],
    layout_result: Optional[PageClassification],
    page_num: int,
) -> PageClassification:
    """Original heuristic ensemble logic (text + layout weighted merge).

    Extracted as a helper so classify_page_ensemble can route by mode.

    Text weight: 0.6, Layout weight: 0.4.
    Falls back to text-only if layout has zero confidence.
    """
    if text_result is None:
        text_result = PageClassification(page_num=page_num, method="text_rules")

    # If layout is missing or has no signal, fall back to text-only
    if layout_result is None or layout_result.confidence <= 0.0:
        return PageClassification(
            page_num=page_num,
            predicted_type=text_result.predicted_type,
            confidence=text_result.confidence,
            method="ensemble",
            type_scores=dict(text_result.type_scores),
        )

    # Merge type_scores with weights
    combined = {}
    for doc_type, score in text_result.type_scores.items():
        combined[doc_type] = combined.get(doc_type, 0.0) + score * _TEXT_WEIGHT
    for doc_type, score in layout_result.type_scores.items():
        combined[doc_type] = combined.get(doc_type, 0.0) + score * _LAYOUT_WEIGHT

    # Round scores
    combined = {k: round(v, 4) for k, v in combined.items()}

    result = PageClassification(
        page_num=page_num,
        method="ensemble",
        type_scores=combined,
    )

    if combined:
        best_type = max(combined, key=combined.get)
        result.predicted_type = best_type
        result.confidence = round(combined[best_type], 4)
    else:
        result.predicted_type = "other"
        result.confidence = 0.0

    return result


def classify_page_ensemble(
    text_result: Optional[PageClassification],
    layout_result: Optional[PageClassification],
    page_num: int,
    *,
    text: str = "",
    bbox_list: Optional[list] = None,
    page_image=None,
) -> PageClassification:
    """Combine text and layout classifications with weighted scoring.

    Behavior is controlled by CLASSIFICATION_MODE:
    - "heuristic": Original text+layout weighted ensemble (default).
    - "ml": ML-only classification via LayoutLMv3. Falls back to heuristic
      on any ML error.
    - "ensemble": Weighted hybrid of heuristic + ML results.

    The text, bbox_list, and page_image keyword arguments are only used
    when CLASSIFICATION_MODE is "ml" or "ensemble".

    Args:
        text_result: Result from classify_page_by_text.
        layout_result: Result from classify_page_by_layout.
        page_num: Page number (1-based).
        text: OCR text for ML classification (keyword-only).
        bbox_list: List of bounding boxes for ML classification (keyword-only).
        page_image: PIL Image for ML classification (keyword-only).

    Returns:
        PageClassification with method="ensemble", "ml", or "hybrid".
    """
    mode = CLASSIFICATION_MODE

    if mode == "heuristic":
        return _heuristic_ensemble(text_result, layout_result, page_num)

    if mode == "ml":
        ml_result = classify_page_ml(text, bbox_list, page_image, page_num)
        if ml_result.confidence > 0.0:
            return ml_result
        # ML failed or returned zero confidence -- fall back to heuristic
        logger.debug(
            "ML classification returned zero confidence for page %d, "
            "falling back to heuristic",
            page_num,
        )
        return _heuristic_ensemble(text_result, layout_result, page_num)

    if mode == "ensemble":
        heuristic_result = _heuristic_ensemble(
            text_result, layout_result, page_num
        )
        ml_result = classify_page_ml(text, bbox_list, page_image, page_num)
        return classify_page_hybrid(heuristic_result, ml_result, page_num)

    # Unknown mode -- warn and fall back to heuristic
    logger.warning(
        "Unknown CLASSIFICATION_MODE=%r, falling back to heuristic", mode
    )
    return _heuristic_ensemble(text_result, layout_result, page_num)


# ---------------------------------------------------------------------------
# Tier 3: ML classification (LayoutLMv3)
# ---------------------------------------------------------------------------

# Hybrid ensemble weights (ML vs heuristic)
_ML_WEIGHT = 0.7
_HEURISTIC_WEIGHT = 0.3


class MLDocumentClassifier:
    """LayoutLMv3-based document classifier with lazy model loading.

    Thread-safe: model is loaded once on first call and cached.
    All torch/transformers imports are guarded with try/except ImportError
    so the module can be imported without ML dependencies installed.
    """

    _MAX_MODEL_LENGTH = 512

    def __init__(
        self,
        model_path: str = ML_CLASSIFICATION_MODEL,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._processor = None
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()
        # Label mapping: index -> document type
        self._id2label = {
            i: dtype for i, dtype in enumerate(DOCUMENT_TYPES_EXTENDED)
        }
        self._label2id = {
            dtype: i for i, dtype in enumerate(DOCUMENT_TYPES_EXTENDED)
        }

    def _load_model(self):
        """Load LayoutLMv3ForSequenceClassification from HuggingFace.

        Called once on first classify() invocation. Thread-safe via lock.
        Raises ImportError if torch/transformers are not available.
        """
        if self._loaded or self._load_failed:
            return

        with self._lock:
            # Double-check after acquiring lock
            if self._loaded or self._load_failed:
                return

            try:
                import torch  # noqa: F401
                from transformers import (
                    LayoutLMv3ForSequenceClassification,
                    LayoutLMv3Processor,
                )
            except ImportError as e:
                self._load_failed = True
                logger.warning(
                    "ML classification unavailable: %s. "
                    "Install torch and transformers for LayoutLMv3 support.",
                    e,
                )
                raise

            try:
                self._processor = LayoutLMv3Processor.from_pretrained(
                    self.model_path, apply_ocr=False
                )
                self._model = LayoutLMv3ForSequenceClassification.from_pretrained(
                    self.model_path,
                    num_labels=len(DOCUMENT_TYPES_EXTENDED),
                    id2label=self._id2label,
                    label2id=self._label2id,
                )
                self._model.eval()
                logger.info(
                    "Loaded ML classification model: %s on %s",
                    self.model_path,
                    self.device,
                )
                self._loaded = True
            except Exception as e:
                self._load_failed = True
                logger.error("Failed to load ML classification model: %s", e)
                raise

    def _preprocess(self, text, bbox_list, page_image):
        """Encode inputs for LayoutLMv3.

        Args:
            text: OCR text string.
            bbox_list: List of [x0, y0, x1, y1] bounding boxes per word.
            page_image: PIL Image of the page.

        Returns:
            Dict of model input tensors.
        """
        # Tokenize text into words (simple whitespace split)
        words = text.split() if text else []

        # Ensure bbox_list matches word count
        if not bbox_list or len(bbox_list) != len(words):
            # Generate dummy bboxes (0,0,0,0) when not available
            bbox_list = [[0, 0, 0, 0]] * len(words)

        # Ensure we have a page image (create blank if needed)
        if page_image is None:
            try:
                from PIL import Image
                page_image = Image.new("RGB", (224, 224), (255, 255, 255))
            except ImportError:
                raise ImportError(
                    "Pillow is required for ML classification"
                )

        # Normalize bboxes to 0-1000 range expected by LayoutLMv3
        normalized_boxes = []
        for box in bbox_list:
            if len(box) >= 4:
                normalized_boxes.append([
                    max(0, min(1000, int(box[0]))),
                    max(0, min(1000, int(box[1]))),
                    max(0, min(1000, int(box[2]))),
                    max(0, min(1000, int(box[3]))),
                ])
            else:
                normalized_boxes.append([0, 0, 0, 0])

        # Truncate to processor max length
        max_words = self._MAX_MODEL_LENGTH
        words = words[:max_words]
        normalized_boxes = normalized_boxes[:max_words]

        encoding = self._processor(
            page_image,
            words,
            boxes=normalized_boxes,
            return_tensors="pt",
            truncation=True,
            max_length=self._MAX_MODEL_LENGTH,
            padding="max_length",
        )

        return {k: v.to(self.device) for k, v in encoding.items()}

    def classify(self, text, bbox_list=None, page_image=None):
        """Run ML inference to classify document page.

        Args:
            text: OCR text string.
            bbox_list: Optional list of bounding boxes per word.
            page_image: Optional PIL Image of the page.

        Returns:
            Tuple of (doc_type: str, confidence: float).
            Returns ("other", 0.0) on any error.
        """
        try:
            self._load_model()
        except (ImportError, Exception):
            return ("other", 0.0)

        try:
            import torch

            inputs = self._preprocess(text, bbox_list, page_image)

            with torch.no_grad():
                outputs = self._model(**inputs)

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted_idx = torch.max(probs, dim=-1)

            doc_type = self._id2label.get(
                predicted_idx.item(), "other"
            )
            conf_value = round(confidence.item(), 4)

            return (doc_type, conf_value)
        except Exception as e:
            logger.error("ML classification inference error: %s", e)
            return ("other", 0.0)


# ---------------------------------------------------------------------------
# ML classifier singleton (thread-safe, lazy)
# ---------------------------------------------------------------------------

_ml_classifier_instance = None
_ml_classifier_lock = threading.Lock()
_ml_classifier_init_failed = False


def get_ml_classifier(
    model_path: str = ML_CLASSIFICATION_MODEL,
    device: str = "cpu",
) -> Optional[MLDocumentClassifier]:
    """Get or create the singleton MLDocumentClassifier instance.

    Thread-safe via lock. Returns None if ML dependencies are missing
    or a prior initialization attempt failed.

    Args:
        model_path: HuggingFace model name or local path.
        device: Device string ("cpu" or "cuda").

    Returns:
        MLDocumentClassifier instance, or None on failure.
    """
    global _ml_classifier_instance, _ml_classifier_init_failed

    if _ml_classifier_init_failed:
        return None

    if _ml_classifier_instance is not None:
        return _ml_classifier_instance

    with _ml_classifier_lock:
        # Double-check after acquiring lock
        if _ml_classifier_init_failed:
            return None
        if _ml_classifier_instance is not None:
            return _ml_classifier_instance

        try:
            instance = MLDocumentClassifier(
                model_path=model_path, device=device
            )
            _ml_classifier_instance = instance
            return instance
        except Exception as e:
            _ml_classifier_init_failed = True
            logger.error("Failed to create ML classifier: %s", e)
            return None


def classify_page_ml(
    text: str,
    bbox_list: Optional[list],
    page_image,
    page_num: int,
) -> PageClassification:
    """Classify a page using ML (LayoutLMv3).

    Guards against missing torch/transformers and model load failures.
    Falls back to zero-confidence "other" on any error.

    Args:
        text: OCR text content of the page.
        bbox_list: List of bounding boxes per word.
        page_image: PIL Image of the page.
        page_num: Page number (1-based).

    Returns:
        PageClassification with method="ml".
    """
    result = PageClassification(page_num=page_num, method="ml")

    classifier = get_ml_classifier()
    if classifier is None:
        logger.debug(
            "ML classifier unavailable for page %d, returning default", page_num
        )
        return result

    doc_type, confidence = classifier.classify(text, bbox_list, page_image)

    result.predicted_type = doc_type
    result.confidence = confidence
    result.type_scores = {doc_type: confidence} if confidence > 0.0 else {}

    return result


def classify_page_hybrid(
    heuristic_result: PageClassification,
    ml_result: PageClassification,
    page_num: int,
) -> PageClassification:
    """Combine heuristic and ML classification results.

    ML weight: 0.7, Heuristic weight: 0.3.
    Falls back to heuristic-only when ML confidence is below threshold.

    Args:
        heuristic_result: Result from heuristic ensemble (text+layout).
        ml_result: Result from classify_page_ml.
        page_num: Page number (1-based).

    Returns:
        PageClassification with method="hybrid".
    """
    result = PageClassification(page_num=page_num, method="hybrid")

    # If ML result is below threshold, fall back to heuristic only
    if ml_result.confidence < ML_CLASSIFICATION_CONFIDENCE_THRESHOLD:
        result.predicted_type = heuristic_result.predicted_type
        result.confidence = heuristic_result.confidence
        result.type_scores = dict(heuristic_result.type_scores)
        return result

    # Map extended ML types to base types for score merging
    ml_scores = {}
    for dtype, score in ml_result.type_scores.items():
        base_type = _ML_TO_BASE_TYPE.get(dtype, dtype)
        ml_scores[base_type] = max(ml_scores.get(base_type, 0.0), score)

    # Merge scores with weights
    combined = {}
    for doc_type, score in heuristic_result.type_scores.items():
        combined[doc_type] = combined.get(doc_type, 0.0) + score * _HEURISTIC_WEIGHT
    for doc_type, score in ml_scores.items():
        combined[doc_type] = combined.get(doc_type, 0.0) + score * _ML_WEIGHT

    # Round scores
    combined = {k: round(v, 4) for k, v in combined.items()}

    result.type_scores = combined

    if combined:
        best_type = max(combined, key=combined.get)
        result.predicted_type = best_type
        result.confidence = round(combined[best_type], 4)
    else:
        result.predicted_type = "other"
        result.confidence = 0.0

    return result


# ---------------------------------------------------------------------------
# Finalization: document-level aggregation
# ---------------------------------------------------------------------------


def finalize_classification(
    doc_cls: DocumentClassification,
) -> DocumentClassification:
    """Compute document-level classification from page results.

    Uses majority vote across pages. Tie-breaking prefers the type with
    higher average confidence.

    Args:
        doc_cls: DocumentClassification with pages already populated.

    Returns:
        The same DocumentClassification with summary fields filled in.
    """
    if not doc_cls.pages:
        doc_cls.document_type = "other"
        doc_cls.document_confidence = 0.0
        doc_cls.type_distribution = {}
        doc_cls.document_type_scores = {}
        doc_cls.document_labels = []
        doc_cls.custom_profile_matches = []
        return doc_cls

    # Count pages per type and accumulate confidence
    type_page_count = {}
    type_conf_sum = {}
    type_score_sum = {}
    profile_summary = {}

    for page_data in doc_cls.pages:
        if isinstance(page_data, PageClassification):
            ptype = page_data.predicted_type
            pconf = page_data.confidence
            page_scores = page_data.type_scores
            profile_matches = page_data.profile_matches
        elif isinstance(page_data, dict):
            ptype = page_data.get("predicted_type", "other")
            pconf = page_data.get("confidence", 0.0)
            page_scores = page_data.get("type_scores", {})
            profile_matches = page_data.get("profile_matches", [])
        else:
            continue

        type_page_count[ptype] = type_page_count.get(ptype, 0) + 1
        type_conf_sum[ptype] = type_conf_sum.get(ptype, 0.0) + pconf
        for dtype, score in (page_scores or {}).items():
            type_score_sum[dtype] = type_score_sum.get(dtype, 0.0) + score

        for profile_match in profile_matches or []:
            key = profile_match.get("name", "")
            if not key:
                continue
            summary = profile_summary.setdefault(
                key,
                {
                    "name": key,
                    "base_type": profile_match.get("base_type", "other"),
                    "route": profile_match.get("route", ""),
                    "confidence": 0.0,
                    "occurrences": 0,
                },
            )
            summary["occurrences"] += 1
            summary["confidence"] = max(
                summary["confidence"],
                float(profile_match.get("confidence", 0.0) or 0.0),
            )

    doc_cls.type_distribution = dict(sorted(type_page_count.items()))

    # Majority vote with confidence tie-breaking
    best_type = "other"
    best_count = 0
    best_avg_conf = 0.0

    for dtype, count in type_page_count.items():
        avg_conf = type_conf_sum[dtype] / count if count > 0 else 0.0
        if count > best_count or (count == best_count and avg_conf > best_avg_conf):
            best_type = dtype
            best_count = count
            best_avg_conf = avg_conf

    doc_cls.document_type = best_type
    doc_cls.document_confidence = round(best_avg_conf, 4)
    page_count = max(len(doc_cls.pages), 1)
    doc_cls.document_type_scores = {
        dtype: round(score / page_count, 4)
        for dtype, score in sorted(
            type_score_sum.items(),
            key=lambda item: (-item[1], item[0]),
        )
    }
    doc_cls.document_labels = _build_document_labels(
        doc_cls.document_type_scores,
        doc_cls.document_type,
    )
    doc_cls.custom_profile_matches = sorted(
        profile_summary.values(),
        key=lambda item: (-item["confidence"], item["name"]),
    )

    return doc_cls


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def _page_cls_to_dict(page_cls):
    """Convert a PageClassification (dataclass or dict) to a plain dict."""
    if isinstance(page_cls, PageClassification):
        return {
            "page_num": page_cls.page_num,
            "predicted_type": page_cls.predicted_type,
            "confidence": page_cls.confidence,
            "method": page_cls.method,
            "type_scores": page_cls.type_scores,
            "is_handwritten": page_cls.is_handwritten,
            "profile_matches": page_cls.profile_matches,
        }
    if isinstance(page_cls, dict):
        return page_cls
    return {}


def write_classification_json(
    doc_cls: DocumentClassification,
    output_folder: str,
    subfolder: str,
    pipeline_version: str,
) -> Optional[str]:
    """Write .classification.json sidecar file.

    Output to EXPORT/CLASSIFICATION/<subfolder>/<name>.classification.json

    Args:
        doc_cls: Finalized DocumentClassification dataclass.
        output_folder: Root output directory (e.g. /app/ocr_output).
        subfolder: Relative subfolder path mirroring source structure.
        pipeline_version: Pipeline version string for metadata.

    Returns:
        Path to the written JSON file, or None on failure.
    """
    try:
        classification_dir = os.path.join(output_folder, "EXPORT", "CLASSIFICATION")
        if subfolder and subfolder != ".":
            safe_parts = [
                sanitize_path_segment(p)
                for p in subfolder.replace("\\", "/").split("/")
                if p
            ]
            target_dir = (
                os.path.join(classification_dir, *safe_parts)
                if safe_parts
                else classification_dir
            )
        else:
            target_dir = classification_dir

        # Path traversal protection
        resolved = os.path.realpath(target_dir)
        if not resolved.startswith(os.path.realpath(classification_dir)):
            logger.error(
                "Path traversal blocked in classification output: %s", subfolder
            )
            return None

        os.makedirs(target_dir, exist_ok=True)

        base_name = build_sidecar_base_name(doc_cls.source_file)
        json_path = os.path.join(target_dir, f"{base_name}.classification.json")

        # Determine which tiers were used
        methods_seen = set()
        for page_data in doc_cls.pages:
            if isinstance(page_data, PageClassification):
                methods_seen.add(page_data.method)
            elif isinstance(page_data, dict):
                methods_seen.add(page_data.get("method", ""))

        if "hybrid" in methods_seen:
            engine_label = "text_rules+layout_features+ml"
        elif "ml" in methods_seen:
            engine_label = "ml"
        elif "ensemble" in methods_seen:
            engine_label = "text_rules+layout_features"
        elif "layout_features" in methods_seen:
            engine_label = "layout_features"
        else:
            engine_label = "text_rules"

        # Build page output
        pages_output = [_page_cls_to_dict(p) for p in doc_cls.pages]

        report = {
            "schema_version": "1.0",
            "document_id": doc_cls.document_id,
            "source_file": doc_cls.source_file,
            "processing": {
                "classification_engine": engine_label,
                "pipeline_version": pipeline_version,
                "timestamp": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(timespec="milliseconds"),
            },
            "document_summary": {
                "document_type": doc_cls.document_type,
                "document_confidence": doc_cls.document_confidence,
                "type_distribution": doc_cls.type_distribution,
                "document_type_scores": doc_cls.document_type_scores,
                "document_labels": doc_cls.document_labels,
                "custom_profile_matches": doc_cls.custom_profile_matches,
            },
            "pages": pages_output,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return json_path
    except Exception as e:
        logger.error(
            "Failed to write classification JSON for %s: %s",
            doc_cls.document_id,
            e,
        )
        return None
