"""LayoutLMv3 structure.json integration with ensemble merging.

Integrates LayoutLMv3 semantic entity predictions into the existing
structure.json output format (schema v1.0) produced by the Document
Intelligence pipeline (PP-StructureV3).  When both models are available,
an ensemble strategy merges their predictions using configurable
confidence weighting.

Design principles:
- No torch/transformers imports (pure data integration module).
- Graceful degradation: missing inputs produce valid but empty results.
- Additive: LayoutLMv3 entities are appended under a new
  ``semantic_entities`` key alongside existing ``layout_regions``.
- Ensemble: when both PP-StructureV3 regions and LayoutLMv3 entities
  are present, the integrator merges them with weighted confidence.

Environment Variables:
    LAYOUTLM_ENSEMBLE_WEIGHT (float):
        Weight for LayoutLMv3 confidence in ensemble merging (0.0-1.0).
        PP-StructureV3 receives ``1 - weight``.  Default: ``0.7``.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LAYOUTLM_ENSEMBLE_WEIGHT: float = max(
    0.0,
    min(
        1.0,
        float(os.environ.get("LAYOUTLM_ENSEMBLE_WEIGHT", "0.7")),
    ),
)

# Region type aliases: map PP-StructureV3 region types to a common
# vocabulary shared with LayoutLMv3 entity labels for overlap detection.
_REGION_TYPE_TO_ENTITY_LABEL = {
    "text": None,  # generic text regions have no entity equivalent
    "title": None,
    "figure": None,
    "table": None,
    "list": None,
    "header": None,
    "footer": None,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SemanticEntityRecord:
    """A single semantic entity for structure.json output."""

    text: str
    label: str
    bbox: list = field(default_factory=list)
    confidence: float = 0.0
    source: str = "layoutlmv3"
    page_num: int = 0


@dataclass
class IntegratedPageResult:
    """Merged structure result for a single page."""

    page_number: int = 0
    layout_regions: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    key_value_pairs: list = field(default_factory=list)
    form_fields: list = field(default_factory=list)
    semantic_entities: list = field(default_factory=list)
    ensemble_source: str = ""


# ---------------------------------------------------------------------------
# Integrator class
# ---------------------------------------------------------------------------


class LayoutLMv3StructureIntegrator:
    """Integrates LayoutLMv3 predictions into the structure.json schema.

    Supports three modes of operation:
    1. LayoutLMv3-only: entities become ``semantic_entities``; layout
       regions are empty.
    2. PP-StructureV3-only: standard structure.json with no
       ``semantic_entities`` (backward compatible).
    3. Ensemble: both sources merged with weighted confidence.
    """

    def __init__(self, ensemble_weight: Optional[float] = None):
        """Initialize the integrator.

        Args:
            ensemble_weight: Weight for LayoutLMv3 confidence (0.0-1.0).
                If None, uses the ``LAYOUTLM_ENSEMBLE_WEIGHT`` env var.
        """
        if ensemble_weight is not None:
            self.ensemble_weight = max(0.0, min(1.0, float(ensemble_weight)))
        else:
            self.ensemble_weight = LAYOUTLM_ENSEMBLE_WEIGHT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def integrate(
        self,
        ocr_results: Optional[list] = None,
        layoutlm_results: Optional[list] = None,
        ppstructure_results: Optional[dict] = None,
    ) -> IntegratedPageResult:
        """Merge LayoutLMv3 and PP-StructureV3 predictions for one page.

        Args:
            ocr_results: OCR line-level output (unused in current
                integration -- reserved for future word-level alignment).
            layoutlm_results: List of LayoutLMv3 entity dicts, each with
                keys ``text``, ``label``, ``confidence``, ``bbox``, and
                optionally ``page_num`` and ``field_type``.
            ppstructure_results: Dict from ``parse_structure_result()`` with
                keys ``layout_regions``, ``tables``, ``key_value_pairs``,
                ``form_fields``.

        Returns:
            IntegratedPageResult with merged data.
        """
        result = IntegratedPageResult()

        has_layoutlm = bool(layoutlm_results)
        has_ppstructure = bool(ppstructure_results)

        # Determine ensemble source label
        if has_layoutlm and has_ppstructure:
            result.ensemble_source = "layoutlmv3+ppstructure"
        elif has_layoutlm:
            result.ensemble_source = "layoutlmv3"
        elif has_ppstructure:
            result.ensemble_source = "ppstructure"
        else:
            result.ensemble_source = "none"

        # PP-StructureV3 regions (tables, layout, form fields, KV pairs)
        if has_ppstructure:
            result.layout_regions = list(
                ppstructure_results.get("layout_regions", [])
            )
            result.tables = list(ppstructure_results.get("tables", []))
            result.key_value_pairs = list(
                ppstructure_results.get("key_value_pairs", [])
            )
            result.form_fields = list(
                ppstructure_results.get("form_fields", [])
            )

        # LayoutLMv3 semantic entities
        if has_layoutlm:
            semantic_entities = self._normalize_layoutlm_entities(
                layoutlm_results
            )
            result.semantic_entities = semantic_entities

        # Ensemble: enrich layout regions with LayoutLMv3 confidence
        if has_layoutlm and has_ppstructure:
            result.layout_regions = self.ensemble(
                layoutlm_results, result.layout_regions
            )

        return result

    def ensemble(
        self,
        layoutlm_entities: list,
        ppstructure_regions: list,
    ) -> list:
        """Combine LayoutLMv3 entities with PP-StructureV3 regions.

        For each PP-StructureV3 region that spatially overlaps a
        LayoutLMv3 entity, the region confidence is updated using
        weighted merging.  Regions without overlapping entities are
        kept unchanged.

        Args:
            layoutlm_entities: List of LayoutLMv3 entity dicts.
            ppstructure_regions: List of PP-StructureV3 region dicts.

        Returns:
            Updated list of region dicts (copies, not mutated in place).
        """
        if not ppstructure_regions:
            return []
        if not layoutlm_entities:
            return list(ppstructure_regions)

        merged_regions = []
        for region in ppstructure_regions:
            region_copy = dict(region)
            region_bbox = region_copy.get("bbox", [])

            # Find overlapping LayoutLMv3 entities
            overlapping = []
            for entity in layoutlm_entities:
                entity_bbox = self._get_entity_bbox(entity)
                if entity_bbox and region_bbox and _boxes_overlap(
                    region_bbox, entity_bbox
                ):
                    overlapping.append(entity)

            if overlapping:
                # Weighted confidence merge
                pp_conf = float(region_copy.get("confidence", 0.0))
                lm_conf = max(
                    float(self._get_entity_confidence(e)) for e in overlapping
                )
                merged_conf = (
                    self.ensemble_weight * lm_conf
                    + (1.0 - self.ensemble_weight) * pp_conf
                )
                region_copy["confidence"] = round(merged_conf, 4)
                region_copy["ensemble_source"] = "layoutlmv3+ppstructure"
                region_copy["layoutlm_entities"] = [
                    self._entity_to_dict(e) for e in overlapping
                ]
            else:
                region_copy["ensemble_source"] = "ppstructure"

            merged_regions.append(region_copy)

        return merged_regions

    def to_structure_json(
        self,
        integrated_result: IntegratedPageResult,
        page_number: int,
    ) -> dict:
        """Format an IntegratedPageResult as a structure.json page entry.

        Args:
            integrated_result: Merged result from ``integrate()``.
            page_number: 1-based page number.

        Returns:
            Dict matching the structure.json page schema (v1.0) with
            the additional ``semantic_entities`` and ``ensemble_source``
            keys.
        """
        page_dict = {
            "page_num": page_number,
            "layout_regions": integrated_result.layout_regions,
            "tables": integrated_result.tables,
            "key_value_pairs": integrated_result.key_value_pairs,
            "form_fields": integrated_result.form_fields,
        }

        # Add semantic entities only if present (backward compatible)
        if integrated_result.semantic_entities:
            page_dict["semantic_entities"] = [
                self._semantic_entity_to_dict(e)
                for e in integrated_result.semantic_entities
            ]

        # Always include ensemble source when integration happened
        if integrated_result.ensemble_source:
            page_dict["ensemble_source"] = integrated_result.ensemble_source

        return page_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_layoutlm_entities(self, entities: list) -> list:
        """Convert raw LayoutLMv3 entity dicts to SemanticEntityRecord objects.

        Accepts both SemanticEntity dataclass instances (from
        semantic_extraction.py) and plain dicts (from Celery task output).
        """
        records = []
        for entity in entities:
            record = SemanticEntityRecord(
                text=self._get_entity_text(entity),
                label=self._get_entity_label(entity),
                bbox=self._get_entity_bbox(entity),
                confidence=self._get_entity_confidence(entity),
                source="layoutlmv3",
                page_num=self._get_entity_page_num(entity),
            )
            records.append(record)
        return records

    @staticmethod
    def _get_entity_text(entity) -> str:
        if isinstance(entity, dict):
            return str(entity.get("text", ""))
        return str(getattr(entity, "text", ""))

    @staticmethod
    def _get_entity_label(entity) -> str:
        if isinstance(entity, dict):
            return str(entity.get("label", ""))
        return str(getattr(entity, "label", ""))

    @staticmethod
    def _get_entity_bbox(entity) -> list:
        if isinstance(entity, dict):
            bbox = entity.get("bbox", [])
        else:
            bbox = getattr(entity, "bbox", [])
        return list(bbox) if bbox else []

    @staticmethod
    def _get_entity_confidence(entity) -> float:
        if isinstance(entity, dict):
            return float(entity.get("confidence", 0.0))
        return float(getattr(entity, "confidence", 0.0))

    @staticmethod
    def _get_entity_page_num(entity) -> int:
        if isinstance(entity, dict):
            return int(entity.get("page_num", 0))
        return int(getattr(entity, "page_num", 0))

    @staticmethod
    def _entity_to_dict(entity) -> dict:
        """Convert an entity (dict or dataclass) to a plain dict."""
        if isinstance(entity, dict):
            return {
                "text": str(entity.get("text", "")),
                "label": str(entity.get("label", "")),
                "confidence": round(float(entity.get("confidence", 0.0)), 4),
                "bbox": list(entity.get("bbox", [])),
            }
        return {
            "text": str(getattr(entity, "text", "")),
            "label": str(getattr(entity, "label", "")),
            "confidence": round(float(getattr(entity, "confidence", 0.0)), 4),
            "bbox": list(getattr(entity, "bbox", [])),
        }

    @staticmethod
    def _semantic_entity_to_dict(entity) -> dict:
        """Convert a SemanticEntityRecord to a serializable dict."""
        if isinstance(entity, SemanticEntityRecord):
            return {
                "text": entity.text,
                "label": entity.label,
                "bbox": entity.bbox,
                "confidence": round(entity.confidence, 4),
                "source": entity.source,
            }
        if isinstance(entity, dict):
            return {
                "text": str(entity.get("text", "")),
                "label": str(entity.get("label", "")),
                "bbox": list(entity.get("bbox", [])),
                "confidence": round(float(entity.get("confidence", 0.0)), 4),
                "source": str(entity.get("source", "layoutlmv3")),
            }
        return {
            "text": str(getattr(entity, "text", "")),
            "label": str(getattr(entity, "label", "")),
            "bbox": list(getattr(entity, "bbox", [])),
            "confidence": round(float(getattr(entity, "confidence", 0.0)), 4),
            "source": str(getattr(entity, "source", "layoutlmv3")),
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _boxes_overlap(box_a: list, box_b: list) -> bool:
    """Check whether two [x1, y1, x2, y2] bounding boxes overlap.

    Returns False if either box is invalid (fewer than 4 coordinates).
    """
    if len(box_a) < 4 or len(box_b) < 4:
        return False

    ax1, ay1, ax2, ay2 = (
        float(box_a[0]),
        float(box_a[1]),
        float(box_a[2]),
        float(box_a[3]),
    )
    bx1, by1, bx2, by2 = (
        float(box_b[0]),
        float(box_b[1]),
        float(box_b[2]),
        float(box_b[3]),
    )

    # No overlap if one box is entirely to the left/right/above/below
    if ax2 <= bx1 or bx2 <= ax1:
        return False
    if ay2 <= by1 or by2 <= ay1:
        return False

    return True
