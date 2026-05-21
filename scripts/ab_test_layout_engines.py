"""A/B testing framework comparing LayoutLMv3 vs PP-StructureV3 engines.

Accepts a directory of test documents, runs both layout-analysis engines
on each, and compares results using IoU (bounding box overlap), entity
extraction F1, and inference latency.  Statistical significance is
assessed via the Mann-Whitney U test.

Generates a comparison report in both JSON and markdown formats.

All heavy ML imports are lazy -- the module is importable and testable
without GPU dependencies.

Usage::

    python scripts/ab_test_layout_engines.py \\
        --test-dir ./test_docs \\
        --engines both \\
        --output-dir ./ab_results

Environment Variables:
    AB_TEST_CONFIDENCE (float):
        Statistical significance threshold.  Default: ``0.05``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = float(os.environ.get("AB_TEST_CONFIDENCE", "0.05"))
_PPSTRUCTURE_ENGINE = None
_PPSTRUCTURE_INIT_FAILED = False
_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
)
_AMOUNT_PATTERN = re.compile(
    r"\b(?:\$|USD|EUR|GBP)\s?\d[\d,]*(?:\.\d{2})?\b|\b\d[\d,]*(?:\.\d{2})\b"
)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?=(?:\D*\d){10,15}\b)(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,4}\d{2,4}\b"
)
_REFERENCE_PATTERN = re.compile(r"\b(?:ref|invoice|case|account)[\s#:.-]*([A-Z0-9-]{3,})\b", re.IGNORECASE)
_REGEX_ENTITY_CONFIDENCE = 0.95

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BoundingBox:
    """Axis-aligned bounding box as [x0, y0, x1, y1]."""

    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0


@dataclass
class EngineResult:
    """Result from a single engine run on one document."""

    engine: str = ""
    document: str = ""
    latency_ms: float = 0.0
    regions: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""


@dataclass
class ComparisonResult:
    """Aggregated comparison between two engines."""

    engine_a: str = ""
    engine_b: str = ""
    num_documents: int = 0
    iou_scores: List[float] = field(default_factory=list)
    avg_iou: float = 0.0
    entity_f1: float = 0.0
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    latency_a_ms: List[float] = field(default_factory=list)
    latency_b_ms: List[float] = field(default_factory=list)
    avg_latency_a_ms: float = 0.0
    avg_latency_b_ms: float = 0.0
    mann_whitney_u: float = 0.0
    mann_whitney_p: float = 1.0
    significant: bool = False
    timestamp: str = ""


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Compute Intersection over Union for two bounding boxes.

    Args:
        box_a: First bounding box.
        box_b: Second bounding box.

    Returns:
        IoU score in [0.0, 1.0].
    """
    x_left = max(box_a.x0, box_b.x0)
    y_top = max(box_a.y0, box_b.y0)
    x_right = min(box_a.x1, box_b.x1)
    y_bottom = min(box_a.y1, box_b.y1)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, (box_a.x1 - box_a.x0) * (box_a.y1 - box_a.y0))
    area_b = max(0.0, (box_b.x1 - box_b.x0) * (box_b.y1 - box_b.y0))
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_mean_iou(
    boxes_a: List[BoundingBox],
    boxes_b: List[BoundingBox],
) -> float:
    """Compute mean IoU between two sets of bounding boxes.

    Uses greedy matching: for each box in A, find the best matching
    box in B (highest IoU) and average across all pairs.

    Args:
        boxes_a: Bounding boxes from engine A.
        boxes_b: Bounding boxes from engine B.

    Returns:
        Mean IoU score.
    """
    if not boxes_a or not boxes_b:
        return 0.0

    total_iou = 0.0
    used_b = set()

    for a in boxes_a:
        best_iou = 0.0
        best_idx = -1
        for idx, b in enumerate(boxes_b):
            if idx in used_b:
                continue
            iou = compute_iou(a, b)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            used_b.add(best_idx)
        total_iou += best_iou

    return total_iou / len(boxes_a)


# ---------------------------------------------------------------------------
# Entity comparison
# ---------------------------------------------------------------------------


def compute_entity_f1(
    entities_a: List[str],
    entities_b: List[str],
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 between two entity lists.

    Treats entities_a as the reference (ground truth) and
    entities_b as predictions.

    Args:
        entities_a: Reference entity labels.
        entities_b: Predicted entity labels.

    Returns:
        Tuple of (precision, recall, f1).
    """
    if not entities_a and not entities_b:
        return 1.0, 1.0, 1.0
    if not entities_a or not entities_b:
        return 0.0, 0.0, 0.0

    set_a = set(entities_a)
    set_b = set(entities_b)

    tp = len(set_a & set_b)
    fp = len(set_b - set_a)
    fn = len(set_a - set_b)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


# ---------------------------------------------------------------------------
# Mann-Whitney U test (pure Python)
# ---------------------------------------------------------------------------


def mann_whitney_u_test(
    sample_a: List[float],
    sample_b: List[float],
) -> Tuple[float, float]:
    """Compute Mann-Whitney U statistic and approximate p-value.

    Uses the normal approximation for p-value computation, which
    is valid when both samples have at least 20 observations.

    Args:
        sample_a: First sample of measurements.
        sample_b: Second sample of measurements.

    Returns:
        Tuple of (U statistic, approximate p-value).
    """
    import math

    if not sample_a or not sample_b:
        return 0.0, 1.0

    n1 = len(sample_a)
    n2 = len(sample_b)

    # Combine and rank
    combined = [(v, 0, i) for i, v in enumerate(sample_a)]
    combined += [(v, 1, i) for i, v in enumerate(sample_b)]
    combined.sort(key=lambda x: x[0])

    # Assign ranks (handle ties by average rank)
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Sum ranks for sample A
    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)

    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    # Normal approximation for p-value
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)

    if sigma == 0:
        return u, 1.0

    z = (u - mu) / sigma
    # Two-tailed p-value using error function approximation
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))

    return u, max(0.0, min(1.0, p))


def _normal_cdf(x: float) -> float:
    """Approximate the standard normal CDF using the error function."""
    import math

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_comparison_report(
    result: ComparisonResult,
    output_dir: str,
) -> Dict[str, str]:
    """Generate JSON and markdown comparison reports.

    Args:
        result: Comparison result to format.
        output_dir: Directory to write reports.

    Returns:
        Dict with paths to generated files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_path = out / "ab_test_report.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(result), fh, indent=2, default=str)

    # Markdown report
    md_path = out / "ab_test_report.md"
    md_lines = [
        f"# A/B Test: {result.engine_a} vs {result.engine_b}",
        "",
        f"**Date**: {result.timestamp}",
        f"**Documents tested**: {result.num_documents}",
        "",
        "## Bounding Box IoU",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mean IoU | {result.avg_iou:.4f} |",
        f"| Min IoU | {min(result.iou_scores):.4f}" if result.iou_scores else "| Min IoU | N/A |",
        f"| Max IoU | {max(result.iou_scores):.4f}" if result.iou_scores else "| Max IoU | N/A |",
        "",
        "## Entity Extraction",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Precision | {result.entity_precision:.4f} |",
        f"| Recall | {result.entity_recall:.4f} |",
        f"| F1 | {result.entity_f1:.4f} |",
        "",
        "## Latency",
        "",
        "| Engine | Avg (ms) | Median (ms) |",
        "|--------|----------|-------------|",
    ]

    if result.latency_a_ms:
        median_a = statistics.median(result.latency_a_ms)
        md_lines.append(
            f"| {result.engine_a} | {result.avg_latency_a_ms:.2f} | {median_a:.2f} |"
        )
    if result.latency_b_ms:
        median_b = statistics.median(result.latency_b_ms)
        md_lines.append(
            f"| {result.engine_b} | {result.avg_latency_b_ms:.2f} | {median_b:.2f} |"
        )

    md_lines += [
        "",
        "## Statistical Significance",
        "",
        "| Test | Value |",
        "|------|-------|",
        f"| Mann-Whitney U | {result.mann_whitney_u:.2f} |",
        f"| p-value | {result.mann_whitney_p:.6f} |",
        f"| Significant (p < {CONFIDENCE_THRESHOLD}) | {'Yes' if result.significant else 'No'} |",
        "",
    ]

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines))

    logger.info("Reports saved to %s", out)
    return {"json": str(json_path), "markdown": str(md_path)}


def _load_document_images(doc_path: Path) -> List[Any]:
    """Load a document into one or more RGB PIL images."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for A/B layout testing") from exc

    suffix = doc_path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required for PDF A/B layout testing") from exc

        pages = []
        with fitz.open(str(doc_path)) as pdf:
            for page in pdf:
                pix = page.get_pixmap(alpha=False)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                pages.append(image)
        return pages

    with Image.open(str(doc_path)) as image:
        return [image.convert("RGB")]


def _coerce_bbox(raw_bbox: Any, fallback: Optional[List[float]] = None) -> List[float]:
    """Normalize a bbox candidate to [x0, y0, x1, y1]."""
    if (
        isinstance(raw_bbox, (list, tuple))
        and len(raw_bbox) == 4
        and all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in raw_bbox)
    ):
        xs = [float(point[0]) for point in raw_bbox]
        ys = [float(point[1]) for point in raw_bbox]
        return [min(xs), min(ys), max(xs), max(ys)]
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        try:
            return [
                float(raw_bbox[0]),
                float(raw_bbox[1]),
                float(raw_bbox[2]),
                float(raw_bbox[3]),
            ]
        except (TypeError, ValueError):
            logger.debug("Unable to coerce bbox %r", raw_bbox, exc_info=True)
    if fallback is not None:
        return list(fallback)
    return [0.0, 0.0, 0.0, 0.0]


def _get_ppstructure_engine():
    """Return a cached PP-Structure engine when available."""
    global _PPSTRUCTURE_ENGINE, _PPSTRUCTURE_INIT_FAILED
    if _PPSTRUCTURE_ENGINE is not None:
        return _PPSTRUCTURE_ENGINE
    if _PPSTRUCTURE_INIT_FAILED:
        return None

    try:
        from paddleocr import PPStructure
    except ImportError:
        _PPSTRUCTURE_INIT_FAILED = True
        return None

    try:
        _PPSTRUCTURE_ENGINE = PPStructure(
            show_log=False,
            recovery=False,
            layout=True,
            table=False,
        )
    except Exception as exc:
        logger.warning("PP-StructureV3 initialization failed: %s", exc)
        _PPSTRUCTURE_INIT_FAILED = True
        return None

    return _PPSTRUCTURE_ENGINE


def _run_ppstructure_page(page_image) -> List[Dict[str, Any]]:
    """Run PP-StructureV3 on a page image and return the raw result list."""
    engine = _get_ppstructure_engine()
    if engine is None:
        return []

    try:
        import numpy as np
    except ImportError:
        logger.warning("numpy is required for PP-StructureV3 A/B testing")
        return []

    try:
        return engine(np.array(page_image)) or []
    except Exception as exc:
        logger.warning("PP-StructureV3 inference failed: %s", exc)
        return []


def _extract_entities_from_text(
    text: str,
    bbox: List[float],
    *,
    source: str,
    page_num: int,
) -> List[Dict[str, Any]]:
    """Derive entity records from actual extracted text content."""
    entities = []
    candidates = [
        ("DATE", _DATE_PATTERN),
        ("AMOUNT", _AMOUNT_PATTERN),
        ("EMAIL", _EMAIL_PATTERN),
        ("PHONE_NUMBER", _PHONE_PATTERN),
        ("REFERENCE_NUMBER", _REFERENCE_PATTERN),
    ]
    for label, pattern in candidates:
        if pattern.search(text):
            entities.append(
                {
                    "type": label,
                    "text": text,
                    "bbox": list(bbox),
                    "confidence": _REGEX_ENTITY_CONFIDENCE,
                    "page_num": page_num,
                    "source": source,
                }
            )
    return entities


def _normalize_ppstructure_output(
    raw_result: List[Dict[str, Any]],
    *,
    page_num: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, float, List[float]]]]:
    """Normalize PP-StructureV3 output into benchmark regions/entities/ocr lines."""
    regions: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    paddle_lines: List[Tuple[str, float, List[float]]] = []
    type_map = {
        "title": "title",
        "text": "paragraph",
        "table": "table",
        "figure": "figure",
        "list": "list",
        "header": "header",
        "footer": "footer",
        "reference": "paragraph",
        "equation": "figure",
    }

    for item in raw_result:
        if not isinstance(item, dict):
            continue

        item_bbox = _coerce_bbox(item.get("bbox"))
        item_type = str(item.get("type", "")).lower()
        regions.append(
            {
                "type": type_map.get(item_type, item_type or "unknown"),
                "bbox": item_bbox,
                "confidence": float(item.get("score", 0.0) or 0.0),
                "page_num": page_num,
            }
        )

        for line in item.get("res", []) if isinstance(item.get("res"), list) else []:
            if not isinstance(line, dict):
                continue
            text = str(line.get("text", "")).strip()
            if not text:
                continue
            line_bbox = _coerce_bbox(line.get("bbox"), item_bbox)
            confidence = float(line.get("confidence", item.get("score", 0.0)) or 0.0)
            paddle_lines.append((text, confidence, line_bbox))
            entities.extend(
                _extract_entities_from_text(
                    text,
                    line_bbox,
                    source="ppstructure",
                    page_num=page_num,
                )
            )

    return regions, entities, paddle_lines


def _paddle_lines_to_word_boxes(
    paddle_lines: List[Tuple[str, float, List[float]]]
) -> Tuple[List[str], List[List[int]]]:
    """Expand OCR lines into word tokens and approximate word boxes."""
    words: List[str] = []
    boxes: List[List[int]] = []
    for text, _confidence, bbox in paddle_lines:
        line_words = text.split()
        if not line_words:
            continue
        x0, y0, x1, y1 = bbox
        width = (x1 - x0) / max(len(line_words), 1)
        for index, word in enumerate(line_words):
            wx0 = x0 + index * width
            wx1 = x0 + (index + 1) * width if width > 0 else x1
            words.append(word)
            boxes.append([int(wx0), int(y0), int(wx1), int(y1)])
    return words, boxes


def _run_layoutlm_regions(
    page_image,
    paddle_lines: List[Tuple[str, float, List[float]]],
    *,
    page_num: int,
) -> List[Dict[str, Any]]:
    """Run the repo's LayoutLMv3 region adapter and normalize the result."""
    try:
        from benchmark_doclaynet import run_layoutlmv3
    except ImportError:
        return []

    words, boxes = _paddle_lines_to_word_boxes(paddle_lines)
    try:
        regions = run_layoutlmv3(page_image, words, boxes)
    except Exception as exc:
        logger.warning("LayoutLMv3 layout inference failed: %s", exc)
        return []

    normalized = []
    for region in regions:
        bbox = getattr(region, "bbox", None)
        normalized.append(
            {
                "type": getattr(region, "region_type", "unknown"),
                "bbox": [
                    float(getattr(bbox, "x1", 0.0)),
                    float(getattr(bbox, "y1", 0.0)),
                    float(getattr(bbox, "x2", 0.0)),
                    float(getattr(bbox, "y2", 0.0)),
                ],
                "confidence": float(getattr(region, "confidence", 0.0) or 0.0),
                "page_num": page_num,
            }
        )
    return normalized


def _run_layoutlm_entities(
    paddle_lines: List[Tuple[str, float, List[float]]],
    page_image,
    *,
    page_num: int,
) -> List[Dict[str, Any]]:
    """Run the repo's LayoutLMv3 semantic extractor and normalize entities."""
    try:
        from semantic_extraction import extract_semantic_fields
    except ImportError:
        return []

    try:
        result = extract_semantic_fields(paddle_lines, page_image, page_num)
    except Exception as exc:
        logger.warning("LayoutLMv3 semantic extraction failed: %s", exc)
        return []

    entities = []
    for entity in getattr(result, "entities", []):
        entities.append(
            {
                "type": entity.label,
                "text": entity.text,
                "bbox": list(entity.bbox),
                "confidence": float(entity.confidence or 0.0),
                "page_num": entity.page_num,
                "source": "layoutlm",
            }
        )
    return entities


# ---------------------------------------------------------------------------
# A/B test runner
# ---------------------------------------------------------------------------


def run_ab_test(
    test_dir: str,
    engines: str = "both",
    output_dir: str = "./ab_results",
) -> ComparisonResult:
    """Run the A/B test comparison.

    Scans ``test_dir`` for supported document files, runs the
    specified engines on each, and computes comparative metrics.

    Args:
        test_dir: Directory of test documents.
        engines: Engine selector (``"layoutlm"``, ``"ppstructure"``,
                 or ``"both"``).
        output_dir: Directory for output reports.

    Returns:
        ComparisonResult with aggregate comparison metrics.
    """
    test_path = Path(test_dir)
    if not test_path.is_dir():
        logger.error("Test directory does not exist: %s", test_dir)
        return ComparisonResult(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    docs = [
        f for f in sorted(test_path.iterdir())
        if f.is_file() and f.suffix.lower() in supported
    ]

    if not docs:
        logger.warning("No supported documents found in %s", test_dir)
        return ComparisonResult(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    logger.info("Found %d test documents in %s", len(docs), test_dir)

    iou_scores: List[float] = []
    latency_a: List[float] = []
    latency_b: List[float] = []
    all_entities_a: List[str] = []
    all_entities_b: List[str] = []

    for doc in docs:
        result_a = _run_engine("layoutlm", doc)
        result_b = _run_engine("ppstructure", doc)

        latency_a.append(result_a.latency_ms)
        latency_b.append(result_b.latency_ms)

        # Compute IoU from region bounding boxes
        boxes_a = [
            BoundingBox(*r.get("bbox", [0, 0, 0, 0])[:4])
            for r in result_a.regions
            if "bbox" in r
        ]
        boxes_b = [
            BoundingBox(*r.get("bbox", [0, 0, 0, 0])[:4])
            for r in result_b.regions
            if "bbox" in r
        ]
        if boxes_a and boxes_b:
            iou_scores.append(compute_mean_iou(boxes_a, boxes_b))

        # Collect entities
        all_entities_a.extend(e.get("type", "") for e in result_a.entities)
        all_entities_b.extend(e.get("type", "") for e in result_b.entities)

    # Entity F1
    precision, recall, f1 = compute_entity_f1(all_entities_a, all_entities_b)

    # Statistical test on latency
    u_stat, p_val = mann_whitney_u_test(latency_a, latency_b)

    result = ComparisonResult(
        engine_a="layoutlm",
        engine_b="ppstructure",
        num_documents=len(docs),
        iou_scores=iou_scores,
        avg_iou=statistics.mean(iou_scores) if iou_scores else 0.0,
        entity_f1=round(f1, 4),
        entity_precision=round(precision, 4),
        entity_recall=round(recall, 4),
        latency_a_ms=latency_a,
        latency_b_ms=latency_b,
        avg_latency_a_ms=statistics.mean(latency_a) if latency_a else 0.0,
        avg_latency_b_ms=statistics.mean(latency_b) if latency_b else 0.0,
        mann_whitney_u=u_stat,
        mann_whitney_p=p_val,
        significant=p_val < CONFIDENCE_THRESHOLD,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    generate_comparison_report(result, output_dir)
    return result


def _run_engine(engine_name: str, doc_path: Path) -> EngineResult:
    """Run a single engine on a document using the repo's real engine adapters."""
    start = time.perf_counter()
    regions: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    errors: List[str] = []

    try:
        page_images = _load_document_images(doc_path)
    except Exception as exc:
        errors.append(str(exc))
        page_images = []

    for page_num, page_image in enumerate(page_images, start=1):
        paddle_lines: List[Tuple[str, float, List[float]]] = []
        raw_ppstructure = _run_ppstructure_page(page_image)
        pp_regions, pp_entities, paddle_lines = _normalize_ppstructure_output(
            raw_ppstructure,
            page_num=page_num,
        )

        if engine_name == "ppstructure":
            regions.extend(pp_regions)
            entities.extend(pp_entities)
            continue

        if engine_name == "layoutlm":
            regions.extend(
                _run_layoutlm_regions(
                    page_image,
                    paddle_lines,
                    page_num=page_num,
                )
            )
            entities.extend(
                _run_layoutlm_entities(
                    paddle_lines,
                    page_image,
                    page_num=page_num,
                )
            )
            continue

        errors.append(f"unknown engine: {engine_name}")
        break

    elapsed_ms = (time.perf_counter() - start) * 1000

    return EngineResult(
        engine=engine_name,
        document=doc_path.name,
        latency_ms=elapsed_ms,
        regions=regions,
        entities=entities,
        error="; ".join(error for error in errors if error),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="A/B test LayoutLMv3 vs PP-StructureV3 layout engines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test-dir", required=True,
        help="Directory of test documents.",
    )
    parser.add_argument(
        "--engines", default="both",
        choices=["layoutlm", "ppstructure", "both"],
        help="Which engines to run.",
    )
    parser.add_argument(
        "--metrics", default="iou,f1,latency",
        help="Comma-separated metrics to compute.",
    )
    parser.add_argument(
        "--output-dir", default="./ab_results",
        help="Output directory for reports.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for A/B testing."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = run_ab_test(
        test_dir=args.test_dir,
        engines=args.engines,
        output_dir=args.output_dir,
    )

    print("\n=== A/B Test Complete ===")
    print(f"  Documents: {result.num_documents}")
    print(f"  Mean IoU:  {result.avg_iou:.4f}")
    print(f"  Entity F1: {result.entity_f1:.4f}")
    print(f"  Latency A: {result.avg_latency_a_ms:.2f} ms")
    print(f"  Latency B: {result.avg_latency_b_ms:.2f} ms")
    print(f"  Significant: {result.significant} (p={result.mann_whitney_p:.6f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
