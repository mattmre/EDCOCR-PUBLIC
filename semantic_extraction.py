"""LayoutLMv3 semantic extraction for forensic OCR pipeline.

Token-level Key Information Extraction (KIE) using LayoutLMv3 with BIO
tagging. Combines spatial layout (bounding boxes), visual features (page
image), and text tokens to extract structured entities such as dates,
amounts, names, organizations, addresses, and reference numbers.

Output:
- entities are merged into the existing extraction pipeline via
  ``merge_with_existing_extraction``
- finalized extraction results can be emitted as durable
  ``.entities.json`` sidecars with first-pass relationship and
  key-value derivation

Graceful degradation: if ``transformers`` or ``torch`` are not installed,
all public functions return empty results and the pipeline continues to
operate with UIE + regex extraction.
"""

import datetime
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from ocr_distributed.ocr_utils import (
    build_sidecar_base_name,
    sanitize_path_segment,
)
from ocr_local.ml.layoutlm_model_registry import resolve_active_model_selection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env vars)
# ---------------------------------------------------------------------------

ENABLE_SEMANTIC_EXTRACTION = os.environ.get(
    "ENABLE_SEMANTIC_EXTRACTION", ""
).lower() in ("1", "true", "yes")

SEMANTIC_MODEL_PATH = os.environ.get(
    "SEMANTIC_MODEL_PATH", "microsoft/layoutlmv3-base"
)

SEMANTIC_CONFIDENCE_THRESHOLD = float(
    os.environ.get("SEMANTIC_CONFIDENCE_THRESHOLD", "0.5")
)

# ---------------------------------------------------------------------------
# Entity labels for KIE (BIO tagging scheme)
# ---------------------------------------------------------------------------

SEMANTIC_ENTITY_LABELS = [
    "O",  # Outside any entity
    "B-INVOICE_NUMBER", "I-INVOICE_NUMBER",
    "B-DATE", "I-DATE",
    "B-AMOUNT", "I-AMOUNT",
    "B-PERSON_NAME", "I-PERSON_NAME",
    "B-ORGANIZATION", "I-ORGANIZATION",
    "B-ADDRESS", "I-ADDRESS",
    "B-REFERENCE_NUMBER", "I-REFERENCE_NUMBER",
    "B-PHONE_NUMBER", "I-PHONE_NUMBER",
    "B-EMAIL", "I-EMAIL",
]

# Label-to-index and index-to-label mappings
_LABEL2ID = {label: idx for idx, label in enumerate(SEMANTIC_ENTITY_LABELS)}
_ID2LABEL = {idx: label for idx, label in enumerate(SEMANTIC_ENTITY_LABELS)}

# Map semantic entity labels to extraction field_type strings (matching
# the vocabulary in extraction.py for merge compatibility).
_SEMANTIC_TYPE_MAP = {
    "INVOICE_NUMBER": "reference_number",
    "DATE": "date",
    "AMOUNT": "amount",
    "PERSON_NAME": "person_name",
    "ORGANIZATION": "organization",
    "ADDRESS": "address",
    "REFERENCE_NUMBER": "reference_number",
    "PHONE_NUMBER": "phone_number",
    "EMAIL": "email_address",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SemanticEntity:
    """A single entity extracted by LayoutLMv3."""

    text: str
    label: str  # Raw BIO label base (e.g. "DATE", "AMOUNT")
    confidence: float = 0.0
    bbox: list = field(default_factory=list)  # [x1, y1, x2, y2] in original coords
    page_num: int = 0

    @property
    def field_type(self) -> str:
        """Map label to extraction.py field_type vocabulary."""
        return _SEMANTIC_TYPE_MAP.get(self.label, self.label.lower())


@dataclass
class SemanticExtractionResult:
    """Aggregated results from LayoutLMv3 extraction for a document."""

    entities: list = field(default_factory=list)  # List[SemanticEntity]
    model_name: str = ""
    processing_time: float = 0.0  # seconds
    page_count: int = 0


@dataclass
class DocumentEntityOutput:
    """Durable entity/relationship output derived from extraction results."""

    document_id: str
    source_file: str
    pages: list = field(default_factory=list)
    total_entities: int = 0
    entity_type_counts: dict = field(default_factory=dict)
    total_relationships: int = 0
    relationship_type_counts: dict = field(default_factory=dict)
    total_key_value_pairs: int = 0
    unique_entities: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# LayoutLMv3 extractor class
# ---------------------------------------------------------------------------


class LayoutLMv3Extractor:
    """Wrapper for LayoutLMv3 token classification inference.

    Lazily loads the model and processor on first use. All imports of
    ``transformers`` and ``torch`` are done inside methods so that the
    module can be imported even when those packages are absent.
    """

    def __init__(
        self,
        model_path: str = SEMANTIC_MODEL_PATH,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self._device_hint = device
        self._model = None
        self._processor = None
        self._device = None
        self._load_failed = False
        self.model_source = "fallback"
        self.active_model_spec = ""
        self.adapter_path = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        """Load LayoutLMv3 model and processor.

        Returns True on success, False on failure. Failures are permanent
        for the lifetime of the extractor instance -- repeated calls will
        not re-attempt.
        """
        if self._model is not None:
            return True
        if self._load_failed:
            return False

        with _extractor_lock:
            if self._model is not None:
                return True
            if self._load_failed:
                return False

            try:
                import torch
                from transformers import (
                    AutoModelForTokenClassification,
                    AutoProcessor,
                )
            except ImportError as exc:
                self._load_failed = True
                logger.warning(
                    "LayoutLMv3 dependencies not installed (transformers/torch): %s. "
                    "Semantic extraction disabled.",
                    exc,
                )
                return False

            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.model_path, apply_ocr=False
                )
                self._model = AutoModelForTokenClassification.from_pretrained(
                    self.model_path,
                    num_labels=len(SEMANTIC_ENTITY_LABELS),
                    id2label=_ID2LABEL,
                    label2id=_LABEL2ID,
                )

                # Resolve device
                if self._device_hint:
                    self._device = torch.device(self._device_hint)
                elif torch.cuda.is_available():
                    self._device = torch.device("cuda")
                else:
                    self._device = torch.device("cpu")

                self._model.to(self._device)
                self._model.eval()
                logger.info(
                    "LayoutLMv3 loaded: model=%s, device=%s",
                    self.model_path,
                    self._device,
                )
                return True

            except Exception as exc:
                self._load_failed = True
                logger.warning(
                    "LayoutLMv3 model load failed: %s. Semantic extraction disabled.",
                    exc,
                )
                return False

    @property
    def is_available(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Box normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_boxes(
        boxes: list,
        width: int,
        height: int,
    ) -> list:
        """Normalize bounding boxes to the 0-1000 range expected by LayoutLMv3.

        Args:
            boxes: List of [x1, y1, x2, y2] in pixel coordinates.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            List of [x1, y1, x2, y2] scaled to 0-1000.
        """
        if width <= 0 or height <= 0:
            return [[0, 0, 0, 0]] * len(boxes)

        normalized = []
        for box in boxes:
            if len(box) < 4:
                normalized.append([0, 0, 0, 0])
                continue
            x1 = max(0, min(1000, int(box[0] * 1000 / width)))
            y1 = max(0, min(1000, int(box[1] * 1000 / height)))
            x2 = max(0, min(1000, int(box[2] * 1000 / width)))
            y2 = max(0, min(1000, int(box[3] * 1000 / height)))
            normalized.append([x1, y1, x2, y2])

        return normalized

    # ------------------------------------------------------------------
    # BIO postprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess_predictions(
        tokens: list,
        predictions: list,
        confidences: list,
        boxes: Optional[list] = None,
    ) -> list:
        """Merge BIO-tagged token predictions into contiguous entities.

        Args:
            tokens: List of word strings.
            predictions: List of predicted label indices (int).
            confidences: List of confidence scores per token (float).
            boxes: Optional list of [x1, y1, x2, y2] per token.

        Returns:
            List of SemanticEntity objects.
        """
        entities = []
        current_entity = None  # dict with keys: tokens, label, confidences, boxes

        for i, (token, pred_id, conf) in enumerate(
            zip(tokens, predictions, confidences)
        ):
            label = _ID2LABEL.get(pred_id, "O")
            box = boxes[i] if boxes and i < len(boxes) else []

            if label.startswith("B-"):
                # Flush previous entity
                if current_entity is not None:
                    entities.append(current_entity)
                # Start new entity
                entity_type = label[2:]
                current_entity = {
                    "tokens": [token],
                    "label": entity_type,
                    "confidences": [conf],
                    "boxes": [box] if box else [],
                }
            elif label.startswith("I-") and current_entity is not None:
                entity_type = label[2:]
                if entity_type == current_entity["label"]:
                    # Continue current entity
                    current_entity["tokens"].append(token)
                    current_entity["confidences"].append(conf)
                    if box:
                        current_entity["boxes"].append(box)
                else:
                    # Type mismatch -- flush and start new
                    entities.append(current_entity)
                    current_entity = {
                        "tokens": [token],
                        "label": entity_type,
                        "confidences": [conf],
                        "boxes": [box] if box else [],
                    }
            elif label.startswith("I-") and current_entity is None:
                # I- tag without preceding B- -- treat as B-
                entity_type = label[2:]
                current_entity = {
                    "tokens": [token],
                    "label": entity_type,
                    "confidences": [conf],
                    "boxes": [box] if box else [],
                }
            else:
                # O tag -- flush current entity
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None

        # Flush final entity
        if current_entity is not None:
            entities.append(current_entity)

        # Convert to SemanticEntity objects
        result = []
        for ent in entities:
            text = " ".join(ent["tokens"])
            avg_conf = (
                sum(ent["confidences"]) / len(ent["confidences"])
                if ent["confidences"]
                else 0.0
            )
            # Merge bounding boxes into enclosing box
            merged_box = _merge_boxes(ent["boxes"])
            result.append(SemanticEntity(
                text=text,
                label=ent["label"],
                confidence=round(avg_conf, 4),
                bbox=merged_box,
            ))

        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def extract_entities(
        self,
        words: list,
        boxes: list,
        image,
        page_num: int = 0,
    ) -> list:
        """Run LayoutLMv3 inference on a single page.

        Args:
            words: List of word strings from OCR.
            boxes: List of [x1, y1, x2, y2] bounding boxes (pixel coords).
            image: PIL.Image of the page.
            page_num: Page number for attribution.

        Returns:
            List of SemanticEntity objects.
        """
        if not words or not boxes:
            return []

        if not self._load_model():
            return []

        try:
            import torch

            # Get image dimensions for box normalization
            img_width, img_height = image.size
            norm_boxes = self._normalize_boxes(boxes, img_width, img_height)

            # Prepare inputs
            encoding = self._processor(
                image,
                words,
                boxes=norm_boxes,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            encoding = {k: v.to(self._device) for k, v in encoding.items()}

            # Inference
            with torch.no_grad():
                outputs = self._model(**encoding)

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).squeeze().tolist()
            max_probs = torch.max(probs, dim=-1).values.squeeze().tolist()

            # Handle single-token edge case
            if isinstance(pred_ids, int):
                pred_ids = [pred_ids]
                max_probs = [max_probs]

            # Map back to word-level predictions (skip special tokens).
            # LayoutLMv3 processor may split words into subword tokens;
            # we take the prediction of the first subword for each word.
            word_ids = encoding.get("word_ids", None)
            if word_ids is not None and hasattr(word_ids, "__call__"):
                # BatchEncoding.word_ids() is a method for batch index 0
                word_id_list = [
                    encoding.word_ids(batch_index=0)[i]
                    for i in range(len(pred_ids))
                ]
            else:
                # Fallback: assume 1:1 mapping
                word_id_list = list(range(len(pred_ids)))

            # Collect first-subword predictions per word
            word_preds = []
            word_confs = []
            seen_word_ids = set()
            for idx, wid in enumerate(word_id_list):
                if wid is None:
                    continue  # special token
                if wid in seen_word_ids:
                    continue  # skip subsequent subwords
                seen_word_ids.add(wid)
                word_preds.append(pred_ids[idx])
                word_confs.append(max_probs[idx])

            # Truncation safety: ensure we don't exceed the original word count
            word_preds = word_preds[: len(words)]
            word_confs = word_confs[: len(words)]

            # Postprocess BIO tags into entities
            entities = self._postprocess_predictions(
                tokens=words,
                predictions=word_preds,
                confidences=word_confs,
                boxes=boxes,
            )

            # Filter by confidence threshold and set page number
            filtered = []
            for ent in entities:
                if ent.confidence >= SEMANTIC_CONFIDENCE_THRESHOLD:
                    ent.page_num = page_num
                    filtered.append(ent)

            return filtered

        except Exception as exc:
            logger.warning(
                "LayoutLMv3 inference failed for page %d: %s",
                page_num,
                exc,
            )
            return []


# ---------------------------------------------------------------------------
# Helper: merge bounding boxes
# ---------------------------------------------------------------------------


def _merge_boxes(boxes: list) -> list:
    """Merge a list of bounding boxes into their enclosing rectangle.

    Args:
        boxes: List of [x1, y1, x2, y2].

    Returns:
        Single [x1, y1, x2, y2] enclosing all input boxes, or empty list.
    """
    if not boxes:
        return []
    valid = [b for b in boxes if len(b) >= 4]
    if not valid:
        return []
    x1 = min(b[0] for b in valid)
    y1 = min(b[1] for b in valid)
    x2 = max(b[2] for b in valid)
    y2 = max(b[3] for b in valid)
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Singleton factory (thread-safe, lazy init)
# ---------------------------------------------------------------------------

_extractor_instance: Optional[LayoutLMv3Extractor] = None
_extractor_lock = threading.Lock()


def get_semantic_extractor() -> Optional[LayoutLMv3Extractor]:
    """Return a singleton LayoutLMv3Extractor instance.

    Thread-safe. Returns None if ENABLE_SEMANTIC_EXTRACTION is False.
    """
    global _extractor_instance

    if not ENABLE_SEMANTIC_EXTRACTION:
        return None

    if _extractor_instance is not None:
        return _extractor_instance

    with _extractor_lock:
        # Double-checked locking
        if _extractor_instance is not None:
            return _extractor_instance
        model_selection = resolve_active_model_selection(SEMANTIC_MODEL_PATH)
        _extractor_instance = LayoutLMv3Extractor(
            model_path=model_selection.model_path,
        )
        _extractor_instance.model_source = model_selection.source
        _extractor_instance.active_model_spec = (
            model_selection.active_model_spec
        )
        _extractor_instance.adapter_path = model_selection.adapter_path
        if model_selection.source == "registry":
            logger.info(
                "LayoutLMv3 using active registry model %s -> %s",
                model_selection.active_model_spec,
                model_selection.model_path,
            )
            if model_selection.adapter_path:
                logger.info(
                    "LayoutLMv3 registry entry includes adapter_path=%s; "
                    "live inference uses model_path only in this pass.",
                    model_selection.adapter_path,
                )
        return _extractor_instance


# ---------------------------------------------------------------------------
# High-level extraction function
# ---------------------------------------------------------------------------


def extract_semantic_fields(
    paddle_lines: list,
    page_image,
    page_num: int = 0,
) -> SemanticExtractionResult:
    """Extract semantic entities from PaddleOCR output using LayoutLMv3.

    This is the primary entry point for pipeline integration. It converts
    PaddleOCR line-level output into the word/box format required by
    LayoutLMv3 and returns a SemanticExtractionResult.

    Args:
        paddle_lines: List of (text, confidence, [x1, y1, x2, y2]) tuples
            from PaddleOCR output.
        page_image: PIL.Image of the page.
        page_num: Page number for entity attribution.

    Returns:
        SemanticExtractionResult with extracted entities.
    """
    start_time = time.monotonic()
    result = SemanticExtractionResult(
        model_name=SEMANTIC_MODEL_PATH,
        page_count=1,
    )

    extractor = get_semantic_extractor()
    if extractor is None:
        result.processing_time = time.monotonic() - start_time
        return result
    result.model_name = extractor.model_path

    # Convert PaddleOCR lines to word-level tokens and boxes.
    # Each line may contain multiple words; we split on whitespace.
    words = []
    boxes = []
    for line_data in paddle_lines:
        if len(line_data) < 3:
            continue
        text, _conf, bbox = line_data[0], line_data[1], line_data[2]
        if not text or not bbox or len(bbox) < 4:
            continue
        line_words = text.split()
        if not line_words:
            continue

        # Distribute the line bounding box evenly across words
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        word_width = (x2 - x1) / max(len(line_words), 1)
        for j, word in enumerate(line_words):
            wx1 = x1 + j * word_width
            wx2 = wx1 + word_width
            words.append(word)
            boxes.append([int(wx1), int(y1), int(wx2), int(y2)])

    if not words:
        result.processing_time = time.monotonic() - start_time
        return result

    entities = extractor.extract_entities(words, boxes, page_image, page_num)
    result.entities = entities
    result.processing_time = round(time.monotonic() - start_time, 4)

    return result


# ---------------------------------------------------------------------------
# Merge with existing extraction results
# ---------------------------------------------------------------------------


def merge_with_existing_extraction(
    semantic_result: SemanticExtractionResult,
    existing_fields: list,
) -> list:
    """Merge LayoutLMv3 semantic entities with existing UIE/regex fields.

    Priority order:
    1. UIE fields are kept as-is (highest priority).
    2. Semantic (LayoutLMv3) fields are added if they do not overlap with
       UIE fields on the same field_type and character-range/text span.
    3. Regex fields are kept if they do not overlap with either UIE or
       semantic fields.

    Args:
        semantic_result: SemanticExtractionResult from LayoutLMv3.
        existing_fields: List of dicts from extraction.py (with keys
            field_type, text, confidence, extraction_method, etc.).

    Returns:
        Merged list of field dicts, deduplicated.
    """
    if not semantic_result or not semantic_result.entities:
        return existing_fields

    # Convert semantic entities to the same dict format as extraction.py
    semantic_dicts = []
    for ent in semantic_result.entities:
        semantic_dicts.append({
            "field_type": ent.field_type,
            "text": ent.text,
            "confidence": ent.confidence,
            "page_num": ent.page_num,
            "start": 0,  # No character offset from LayoutLMv3 (box-based)
            "end": 0,
            "extraction_method": "semantic",
            "normalized_value": "",
            "bbox": ent.bbox,
        })

    # Separate existing fields by method
    uie_fields = [
        f for f in existing_fields if f.get("extraction_method") == "uie"
    ]
    regex_fields = [
        f for f in existing_fields if f.get("extraction_method") == "regex"
    ]
    other_fields = [
        f for f in existing_fields
        if f.get("extraction_method") not in ("uie", "regex")
    ]

    # Keep all UIE fields (highest priority)
    kept = list(uie_fields)

    # Add semantic fields that do not overlap with UIE fields
    for sf in semantic_dicts:
        if not _text_overlaps_any(sf, uie_fields):
            kept.append(sf)

    # Add regex fields that do not overlap with UIE or semantic fields
    for rf in regex_fields:
        if not _text_overlaps_any(rf, kept):
            kept.append(rf)

    # Preserve any other extraction methods
    kept.extend(other_fields)

    return kept


def _text_overlaps_any(field_dict: dict, existing: list) -> bool:
    """Check if a field's text overlaps with any field in existing list.

    Two fields overlap if they share the same field_type and have
    matching text content (case-insensitive, stripped).
    """
    f_type = field_dict.get("field_type", "")
    f_text = field_dict.get("text", "").strip().lower()

    for ef in existing:
        if ef.get("field_type") == f_type:
            ef_text = ef.get("text", "").strip().lower()
            # Exact text match only. Substring checks drop richer values.
            if f_text == ef_text:
                return True
    return False


# ---------------------------------------------------------------------------
# Durable entity output from extraction results
# ---------------------------------------------------------------------------


def _field_to_entity_dict(field_obj, page_num: int, ordinal: int) -> dict:
    """Normalize an extracted field into an entity-like dictionary."""
    if isinstance(field_obj, dict):
        raw = dict(field_obj)
    else:
        raw = {
            "field_type": getattr(field_obj, "field_type", ""),
            "text": getattr(field_obj, "text", ""),
            "confidence": getattr(field_obj, "confidence", 0.0),
            "page_num": getattr(field_obj, "page_num", page_num),
            "start": getattr(field_obj, "start", 0),
            "end": getattr(field_obj, "end", 0),
            "extraction_method": getattr(field_obj, "extraction_method", ""),
            "normalized_value": getattr(field_obj, "normalized_value", ""),
            "bbox": getattr(field_obj, "bbox", []),
        }

    text = str(raw.get("text", "") or "").strip()
    field_type = str(raw.get("field_type", "") or "").strip()
    bbox = raw.get("bbox", [])
    if not isinstance(bbox, list):
        bbox = []

    return {
        "entity_id": f"p{page_num}-e{ordinal}",
        "field_type": field_type,
        "text": text,
        "normalized_value": str(raw.get("normalized_value", "") or ""),
        "confidence": round(float(raw.get("confidence", 0.0) or 0.0), 4),
        "page_num": int(raw.get("page_num", page_num) or page_num),
        "start": int(raw.get("start", 0) or 0),
        "end": int(raw.get("end", 0) or 0),
        "extraction_method": str(raw.get("extraction_method", "") or ""),
        "bbox": bbox,
        "_ordinal": ordinal,
    }


def _page_to_fields(page_obj) -> tuple[int, list]:
    """Return page number and raw field list from extraction page data."""
    if isinstance(page_obj, dict):
        return int(page_obj.get("page_num", 0) or 0), page_obj.get("fields", [])
    return int(getattr(page_obj, "page_num", 0) or 0), getattr(page_obj, "fields", [])


def _entity_anchor(entity: dict) -> tuple[int, int]:
    """Return a deterministic position anchor for nearest-neighbor linking."""
    start = int(entity.get("start", 0) or 0)
    end = int(entity.get("end", 0) or 0)
    if start > 0:
        return (start, end if end > 0 else start)
    ordinal = int(entity.get("_ordinal", 0) or 0)
    return (ordinal * 100, ordinal * 100)


def _entity_distance(left: dict, right: dict) -> tuple[int, int, int]:
    """Sort key for selecting the nearest related entity on a page."""
    left_anchor = _entity_anchor(left)
    right_anchor = _entity_anchor(right)
    return (
        abs(left_anchor[0] - right_anchor[0]),
        abs(int(left.get("_ordinal", 0) or 0) - int(right.get("_ordinal", 0) or 0)),
        int(right.get("_ordinal", 0) or 0),
    )


def _pick_nearest_entity(source: dict, candidates: list) -> Optional[dict]:
    """Pick the nearest candidate entity using text offsets or field order."""
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: _entity_distance(source, candidate))


def _relationship_dict(
    relationship_type: str,
    source: dict,
    target: dict,
) -> dict:
    """Build a deterministic relationship record."""
    return {
        "relationship_type": relationship_type,
        "source_entity_id": source["entity_id"],
        "target_entity_id": target["entity_id"],
        "source_field_type": source["field_type"],
        "target_field_type": target["field_type"],
        "source_text": source["text"],
        "target_text": target["text"],
        "confidence": round(
            (float(source.get("confidence", 0.0)) + float(target.get("confidence", 0.0))) / 2.0,
            4,
        ),
    }


def _kv_pair_dict(
    pair_type: str,
    key_entity: dict,
    value_entity: dict,
) -> dict:
    """Build a deterministic semantic key-value pair record."""
    return {
        "pair_type": pair_type,
        "key_entity_id": key_entity["entity_id"],
        "value_entity_id": value_entity["entity_id"],
        "key_field_type": key_entity["field_type"],
        "value_field_type": value_entity["field_type"],
        "key_text": key_entity["text"],
        "value_text": value_entity["text"],
        "confidence": round(
            (float(key_entity.get("confidence", 0.0)) + float(value_entity.get("confidence", 0.0)))
            / 2.0,
            4,
        ),
    }


def _add_page_relationship(
    relationships: list,
    seen_keys: set,
    relationship_type: str,
    source: Optional[dict],
    target: Optional[dict],
) -> None:
    """Append a page relationship if both ends exist and it is new."""
    if not source or not target:
        return
    dedup_key = (relationship_type, source["entity_id"], target["entity_id"])
    if dedup_key in seen_keys:
        return
    seen_keys.add(dedup_key)
    relationships.append(_relationship_dict(relationship_type, source, target))


def _add_page_kv_pair(
    kv_pairs: list,
    seen_keys: set,
    pair_type: str,
    key_entity: Optional[dict],
    value_entity: Optional[dict],
) -> None:
    """Append a page key-value pair if both ends exist and it is new."""
    if not key_entity or not value_entity:
        return
    dedup_key = (pair_type, key_entity["entity_id"], value_entity["entity_id"])
    if dedup_key in seen_keys:
        return
    seen_keys.add(dedup_key)
    kv_pairs.append(_kv_pair_dict(pair_type, key_entity, value_entity))


def _derive_page_relationships(page_entities: list) -> tuple[list, list]:
    """Derive first-pass relationships and semantic KV pairs for one page."""
    relationships = []
    kv_pairs = []
    relationship_keys = set()
    kv_keys = set()

    entities_by_type = {}
    for entity in page_entities:
        entities_by_type.setdefault(entity["field_type"], []).append(entity)

    reference_numbers = entities_by_type.get("reference_number", [])
    dates = entities_by_type.get("date", [])
    amounts = entities_by_type.get("amount", [])
    people = entities_by_type.get("person_name", [])
    organizations = entities_by_type.get("organization", [])
    contacts = entities_by_type.get("email_address", []) + entities_by_type.get("phone_number", [])

    for reference in reference_numbers:
        nearest_date = _pick_nearest_entity(reference, dates)
        _add_page_relationship(
            relationships,
            relationship_keys,
            "document_date",
            reference,
            nearest_date,
        )
        _add_page_kv_pair(
            kv_pairs,
            kv_keys,
            "reference_number_to_date",
            reference,
            nearest_date,
        )

        nearest_amount = _pick_nearest_entity(reference, amounts)
        _add_page_relationship(
            relationships,
            relationship_keys,
            "document_amount",
            reference,
            nearest_amount,
        )
        _add_page_kv_pair(
            kv_pairs,
            kv_keys,
            "reference_number_to_amount",
            reference,
            nearest_amount,
        )

    for person in people:
        nearest_org = _pick_nearest_entity(person, organizations)
        _add_page_relationship(
            relationships,
            relationship_keys,
            "party_association",
            person,
            nearest_org,
        )
        _add_page_kv_pair(
            kv_pairs,
            kv_keys,
            "person_name_to_organization",
            person,
            nearest_org,
        )

    contact_anchors = people + organizations
    for contact in contacts:
        nearest_anchor = _pick_nearest_entity(contact, contact_anchors)
        _add_page_relationship(
            relationships,
            relationship_keys,
            "contact_value",
            nearest_anchor,
            contact,
        )
        if nearest_anchor:
            pair_type = f"{nearest_anchor['field_type']}_to_{contact['field_type']}"
            _add_page_kv_pair(
                kv_pairs,
                kv_keys,
                pair_type,
                nearest_anchor,
                contact,
            )

    return relationships, kv_pairs


def _public_entity_dict(entity: dict) -> dict:
    """Drop internal-only fields before serializing to JSON."""
    return {
        "entity_id": entity["entity_id"],
        "field_type": entity["field_type"],
        "text": entity["text"],
        "normalized_value": entity["normalized_value"],
        "confidence": entity["confidence"],
        "page_num": entity["page_num"],
        "start": entity["start"],
        "end": entity["end"],
        "extraction_method": entity["extraction_method"],
        "bbox": entity["bbox"],
    }


def finalize_entity_output(doc_ext) -> DocumentEntityOutput:
    """Derive durable entity, relationship, and KV output from extraction data."""
    doc_entities = DocumentEntityOutput(
        document_id=getattr(doc_ext, "document_id", ""),
        source_file=getattr(doc_ext, "source_file", ""),
    )

    entity_type_counts = {}
    relationship_type_counts = {}
    unique_entities = {}

    for page_obj in getattr(doc_ext, "pages", []):
        page_num, page_fields = _page_to_fields(page_obj)
        page_entities = []

        for ordinal, field_obj in enumerate(page_fields, start=1):
            entity = _field_to_entity_dict(field_obj, page_num, ordinal)
            if not entity["field_type"] or not entity["text"]:
                continue

            page_entities.append(entity)
            field_type = entity["field_type"]
            entity_type_counts[field_type] = entity_type_counts.get(field_type, 0) + 1

            unique_key = (
                field_type,
                (entity["normalized_value"] or entity["text"]).strip().lower(),
            )
            unique_record = unique_entities.setdefault(
                unique_key,
                {
                    "field_type": field_type,
                    "text": entity["text"],
                    "normalized_value": entity["normalized_value"],
                    "occurrences": 0,
                    "pages": set(),
                },
            )
            unique_record["occurrences"] += 1
            unique_record["pages"].add(page_num)

        relationships, kv_pairs = _derive_page_relationships(page_entities)
        for relationship in relationships:
            rel_type = relationship["relationship_type"]
            relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1

        doc_entities.pages.append(
            {
                "page_num": page_num,
                "entities": [_public_entity_dict(entity) for entity in page_entities],
                "relationships": relationships,
                "key_value_pairs": kv_pairs,
            }
        )

        doc_entities.total_entities += len(page_entities)
        doc_entities.total_relationships += len(relationships)
        doc_entities.total_key_value_pairs += len(kv_pairs)

    doc_entities.entity_type_counts = dict(sorted(entity_type_counts.items()))
    doc_entities.relationship_type_counts = dict(sorted(relationship_type_counts.items()))
    doc_entities.unique_entities = sorted(
        [
            {
                "field_type": record["field_type"],
                "text": record["text"],
                "normalized_value": record["normalized_value"],
                "occurrences": record["occurrences"],
                "pages": sorted(record["pages"]),
            }
            for record in unique_entities.values()
        ],
        key=lambda record: (record["field_type"], record["text"].lower()),
    )

    return doc_entities


def write_entities_json(
    doc_entities: DocumentEntityOutput,
    output_folder: str,
    subfolder: str,
    pipeline_version: str,
) -> Optional[str]:
    """Write a durable .entities.json sidecar file."""
    try:
        entities_dir = os.path.join(output_folder, "EXPORT", "ENTITIES")
        if subfolder and subfolder != ".":
            safe_parts = [
                sanitize_path_segment(part)
                for part in subfolder.replace("\\", "/").split("/")
                if part
            ]
            target_dir = (
                os.path.join(entities_dir, *safe_parts)
                if safe_parts
                else entities_dir
            )
        else:
            target_dir = entities_dir

        resolved = os.path.realpath(target_dir)
        if not resolved.startswith(os.path.realpath(entities_dir)):
            logger.error(
                "Path traversal blocked in entities output: %s", subfolder
            )
            return None

        os.makedirs(target_dir, exist_ok=True)

        base_name = build_sidecar_base_name(doc_entities.source_file)
        json_path = os.path.join(target_dir, f"{base_name}.entities.json")
        report = {
            "schema_version": "1.0",
            "document_id": doc_entities.document_id,
            "source_file": doc_entities.source_file,
            "processing": {
                "entity_source": "extraction",
                "relationship_engine": "rule_based",
                "kv_engine": "rule_based",
                "pipeline_version": pipeline_version,
                "timestamp": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(timespec="milliseconds"),
            },
            "entities_summary": {
                "total_entities": doc_entities.total_entities,
                "entity_type_counts": doc_entities.entity_type_counts,
                "total_relationships": doc_entities.total_relationships,
                "relationship_type_counts": doc_entities.relationship_type_counts,
                "total_key_value_pairs": doc_entities.total_key_value_pairs,
                "unique_entities": len(doc_entities.unique_entities),
            },
            "document_entities": doc_entities.unique_entities,
            "pages": doc_entities.pages,
        }

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, default=str)

        return json_path
    except Exception as exc:
        logger.error(
            "Failed to write entities JSON for %s: %s",
            doc_entities.document_id,
            exc,
        )
        return None
