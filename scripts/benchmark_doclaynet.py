"""DocLayNet layout analysis benchmark comparing PP-StructureV3 vs LayoutLMv3.

Generates synthetic document pages with known layout regions, runs layout
detection models against them, and computes per-region IoU, mAP, and F1
metrics with inference timing comparison.

Usage:
    python scripts/benchmark_doclaynet.py generate --count 50 --output-dir benchmark_data/doclaynet
    python scripts/benchmark_doclaynet.py run --corpus-dir benchmark_data/doclaynet --models all
    python scripts/benchmark_doclaynet.py report --results results.json

Requires: Pillow (for synthetic generation), numpy (optional, for noise)
Graceful degradation: if torch/paddleocr/ppstructure are not available,
those model runners return empty results and are reported as unavailable.
"""

import argparse
import datetime
import json
import logging
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ensure project root is on sys.path so classification can be imported
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout region types (DocLayNet-aligned)
# ---------------------------------------------------------------------------

LAYOUT_REGION_TYPES = [
    "title",
    "paragraph",
    "table",
    "figure",
    "list",
    "header",
    "footer",
    "caption",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BBox:
    """Bounding box with integer pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def area(self) -> int:
        """Return the area of the bounding box."""
        w = max(0, self.x2 - self.x1)
        h = max(0, self.y2 - self.y1)
        return w * h

    def to_list(self) -> list:
        """Convert to [x1, y1, x2, y2] list."""
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class LayoutRegion:
    """A single layout region annotation or prediction."""

    region_type: str
    bbox: BBox
    confidence: float = 1.0
    text: str = ""


@dataclass
class AnnotatedPage:
    """A synthetic document page with ground truth layout regions."""

    image_path: str = ""
    width: int = 0
    height: int = 0
    regions: list = field(default_factory=list)  # List[LayoutRegion]


@dataclass
class ModelResult:
    """Results from a single model on the entire corpus."""

    model_name: str = ""
    available: bool = False
    total_pages: int = 0
    per_type_iou: dict = field(default_factory=dict)
    mean_iou: float = 0.0
    map_50: float = 0.0
    map_75: float = 0.0
    per_type_f1: dict = field(default_factory=dict)
    mean_f1: float = 0.0
    avg_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    min_inference_ms: float = 0.0
    max_inference_ms: float = 0.0
    timestamp: str = ""


@dataclass
class BenchmarkReport:
    """Complete benchmark comparison report."""

    corpus_dir: str = ""
    total_pages: int = 0
    region_types: list = field(default_factory=list)
    models: list = field(default_factory=list)  # List[ModelResult]
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_iou(box_a: BBox, box_b: BBox) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Parameters
    ----------
    box_a : BBox
        First bounding box.
    box_b : BBox
        Second bounding box.

    Returns
    -------
    float
        IoU value between 0.0 and 1.0.
    """
    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = box_a.area()
    area_b = box_b.area()
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_per_type_iou(
    predictions: list,
    ground_truth: list,
    region_types: list,
) -> dict:
    """Compute average IoU per region type using greedy matching.

    For each ground truth region, finds the best-matching prediction of the
    same type (highest IoU) and averages across all GT regions of that type.

    Parameters
    ----------
    predictions : list[LayoutRegion]
        Predicted layout regions.
    ground_truth : list[LayoutRegion]
        Ground truth layout regions.
    region_types : list[str]
        Region types to evaluate.

    Returns
    -------
    dict[str, float]
        Per-type average IoU.
    """
    result = {}
    for rtype in region_types:
        gt_regions = [r for r in ground_truth if r.region_type == rtype]
        pred_regions = [r for r in predictions if r.region_type == rtype]

        if not gt_regions:
            continue

        ious = []
        used_preds = set()
        for gt_r in gt_regions:
            best_iou = 0.0
            best_idx = -1
            for idx, pred_r in enumerate(pred_regions):
                if idx in used_preds:
                    continue
                iou = compute_iou(gt_r.bbox, pred_r.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0:
                used_preds.add(best_idx)
            ious.append(best_iou)

        result[rtype] = round(statistics.mean(ious), 4) if ious else 0.0

    return result


def compute_ap_at_iou(
    predictions: list,
    ground_truth: list,
    iou_threshold: float,
    region_type: str,
) -> float:
    """Compute Average Precision at a given IoU threshold for one region type.

    Predictions are sorted by confidence. For each prediction, we check if it
    matches a ground truth region (IoU >= threshold). Matched GT regions are
    consumed (no double-counting).

    Parameters
    ----------
    predictions : list[LayoutRegion]
        Predicted regions, will be sorted by confidence.
    ground_truth : list[LayoutRegion]
        Ground truth regions.
    iou_threshold : float
        IoU threshold for a match (e.g. 0.5 or 0.75).
    region_type : str
        Region type to evaluate.

    Returns
    -------
    float
        Average Precision value between 0.0 and 1.0.
    """
    gt_regions = [r for r in ground_truth if r.region_type == region_type]
    pred_regions = sorted(
        [r for r in predictions if r.region_type == region_type],
        key=lambda r: r.confidence,
        reverse=True,
    )

    if not gt_regions:
        return 1.0 if not pred_regions else 0.0

    if not pred_regions:
        return 0.0

    tp_list = []
    matched_gt = set()

    for pred in pred_regions:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_r in enumerate(gt_regions):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred.bbox, gt_r.bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_list.append(1)
            matched_gt.add(best_gt_idx)
        else:
            tp_list.append(0)

    # Compute precision-recall curve and AP
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []
    total_gt = len(gt_regions)

    for is_tp in tp_list:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt
        precisions.append(precision)
        recalls.append(recall)

    # Monotonically decreasing precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # AP = area under PR curve (11-point interpolation simplified)
    ap = 0.0
    prev_recall = 0.0
    for prec, rec in zip(precisions, recalls):
        ap += prec * (rec - prev_recall)
        prev_recall = rec

    return round(ap, 4)


def compute_map(
    predictions: list,
    ground_truth: list,
    region_types: list,
    iou_threshold: float,
) -> float:
    """Compute mean Average Precision across all region types.

    Parameters
    ----------
    predictions : list[LayoutRegion]
        All predicted regions.
    ground_truth : list[LayoutRegion]
        All ground truth regions.
    region_types : list[str]
        Region types to evaluate.
    iou_threshold : float
        IoU threshold (0.5 or 0.75).

    Returns
    -------
    float
        mAP value between 0.0 and 1.0.
    """
    aps = []
    for rtype in region_types:
        gt_of_type = [r for r in ground_truth if r.region_type == rtype]
        if gt_of_type or any(r.region_type == rtype for r in predictions):
            ap = compute_ap_at_iou(predictions, ground_truth, iou_threshold, rtype)
            aps.append(ap)

    if not aps:
        return 0.0

    return round(statistics.mean(aps), 4)


def compute_per_type_f1(
    predictions: list,
    ground_truth: list,
    region_types: list,
    iou_threshold: float = 0.5,
) -> dict:
    """Compute per-type F1 score based on detection (IoU matching).

    A prediction is a true positive if it matches a ground truth region of the
    same type with IoU >= threshold.

    Parameters
    ----------
    predictions : list[LayoutRegion]
        Predicted regions.
    ground_truth : list[LayoutRegion]
        Ground truth regions.
    region_types : list[str]
        Region types to evaluate.
    iou_threshold : float
        IoU threshold for a match.

    Returns
    -------
    dict[str, float]
        Per-type F1 scores.
    """
    result = {}
    for rtype in region_types:
        gt_regions = [r for r in ground_truth if r.region_type == rtype]
        pred_regions = [r for r in predictions if r.region_type == rtype]

        if not gt_regions and not pred_regions:
            continue

        # Greedy matching: for each prediction, find best GT match
        matched_gt = set()
        tp = 0
        for pred in pred_regions:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt_r in enumerate(gt_regions):
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(pred.bbox, gt_r.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)

        fp = len(pred_regions) - tp
        fn = len(gt_regions) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        result[rtype] = round(f1, 4)

    return result


# ---------------------------------------------------------------------------
# Synthetic document generation
# ---------------------------------------------------------------------------


def _draw_filled_rect(pixels, x1, y1, x2, y2, color, width, height):
    """Draw a filled rectangle on a flat pixel buffer.

    Parameters
    ----------
    pixels : list[int]
        Flat list of RGB byte values (length = width * height * 3).
    x1, y1, x2, y2 : int
        Rectangle coordinates (clamped to image bounds).
    color : tuple[int, int, int]
        RGB fill color.
    width, height : int
        Image dimensions.
    """
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    for y in range(y1, y2):
        row_offset = y * width * 3
        for x in range(x1, x2):
            idx = row_offset + x * 3
            pixels[idx] = color[0]
            pixels[idx + 1] = color[1]
            pixels[idx + 2] = color[2]


def _draw_text_lines(pixels, x1, y1, x2, y2, rng, width, height):
    """Draw simulated text lines (dark horizontal bands) in a region.

    Parameters
    ----------
    pixels : list[int]
        Flat pixel buffer.
    x1, y1, x2, y2 : int
        Region bounds.
    rng : random.Random
        Random number generator.
    width, height : int
        Image dimensions.
    """
    region_w = x2 - x1
    region_h = y2 - y1
    if region_w < 30 or region_h < 20:
        return
    line_y = y1 + 4
    line_height = min(rng.randint(8, 12), region_h // 3)
    while line_y + line_height < y2 - 2:
        indent = rng.randint(0, min(10, region_w // 4))
        lx1 = min(x1 + indent, x2 - 1)
        min_end = min(lx1 + 20, x2)
        mid_end = max(min_end, (x2 - x1) // 2 + x1)
        if mid_end >= x2:
            mid_end = max(lx1 + 1, x2 - 1)
        line_end = rng.randint(mid_end, x2) if mid_end < x2 else x2
        _draw_filled_rect(
            pixels, lx1, line_y, line_end, line_y + line_height,
            (40, 40, 40), width, height,
        )
        line_y += line_height + rng.randint(4, 8)
        line_height = min(rng.randint(8, 12), max(1, (y2 - line_y) // 2))


def generate_synthetic_page(
    page_width: int = 2550,
    page_height: int = 3300,
    seed: int = 0,
    region_types: list = None,
) -> AnnotatedPage:
    """Generate a synthetic document page with known layout regions.

    Creates a white-background page image with colored region placeholders
    and returns the ground truth annotations.

    Parameters
    ----------
    page_width : int
        Page width in pixels.
    page_height : int
        Page height in pixels.
    seed : int
        Random seed for reproducibility.
    region_types : list[str], optional
        Region types to include. Defaults to all LAYOUT_REGION_TYPES.

    Returns
    -------
    AnnotatedPage
        Page with image data and ground truth region annotations.
    """
    from PIL import Image

    if region_types is None:
        region_types = list(LAYOUT_REGION_TYPES)

    rng = random.Random(seed)

    # White background
    pixels = [245] * (page_width * page_height * 3)

    # Region color hints (light tints for visual distinction)
    region_colors = {
        "title": (220, 220, 240),
        "paragraph": (240, 240, 240),
        "table": (220, 240, 220),
        "figure": (240, 220, 220),
        "list": (240, 240, 220),
        "header": (230, 230, 245),
        "footer": (230, 230, 245),
        "caption": (245, 235, 225),
    }

    margin_x = 150
    margin_top = 100
    margin_bottom = 100
    content_width = page_width - 2 * margin_x
    regions = []
    y_cursor = margin_top

    # Determine how many regions to place
    num_regions = rng.randint(3, min(7, len(region_types)))
    chosen_types = []

    # Always include a title at the top
    if "title" in region_types:
        chosen_types.append("title")
        remaining = [t for t in region_types if t != "title"]
    else:
        remaining = list(region_types)

    # Fill remaining slots
    while len(chosen_types) < num_regions and remaining:
        pick = rng.choice(remaining)
        chosen_types.append(pick)

    # Header region (if selected)
    if "header" in chosen_types:
        h = rng.randint(30, 60)
        region = LayoutRegion(
            region_type="header",
            bbox=BBox(margin_x, y_cursor, margin_x + content_width, y_cursor + h),
            confidence=1.0,
        )
        color = region_colors.get("header", (230, 230, 230))
        _draw_filled_rect(
            pixels, region.bbox.x1, region.bbox.y1,
            region.bbox.x2, region.bbox.y2, color, page_width, page_height,
        )
        _draw_text_lines(
            pixels, region.bbox.x1 + 4, region.bbox.y1 + 2,
            region.bbox.x2 - 4, region.bbox.y2 - 2, rng, page_width, page_height,
        )
        regions.append(region)
        y_cursor += h + rng.randint(10, 20)
        chosen_types.remove("header")

    # Place remaining regions top-to-bottom
    for rtype in chosen_types:
        if y_cursor >= page_height - margin_bottom - 40:
            break

        remaining_height = page_height - margin_bottom - y_cursor
        if remaining_height < 40:
            break

        if rtype == "title":
            h = rng.randint(40, 80)
            w = rng.randint(content_width // 2, content_width)
            x_offset = (content_width - w) // 2
            region = LayoutRegion(
                region_type="title",
                bbox=BBox(
                    margin_x + x_offset, y_cursor,
                    margin_x + x_offset + w, y_cursor + h,
                ),
                confidence=1.0,
            )
        elif rtype == "paragraph":
            h = rng.randint(80, min(250, remaining_height))
            region = LayoutRegion(
                region_type="paragraph",
                bbox=BBox(
                    margin_x, y_cursor,
                    margin_x + content_width, y_cursor + h,
                ),
                confidence=1.0,
            )
        elif rtype == "table":
            h = rng.randint(100, min(300, remaining_height))
            region = LayoutRegion(
                region_type="table",
                bbox=BBox(
                    margin_x, y_cursor,
                    margin_x + content_width, y_cursor + h,
                ),
                confidence=1.0,
            )
        elif rtype == "figure":
            h = rng.randint(120, min(350, remaining_height))
            w = rng.randint(content_width // 3, content_width)
            x_offset = rng.randint(0, content_width - w)
            region = LayoutRegion(
                region_type="figure",
                bbox=BBox(
                    margin_x + x_offset, y_cursor,
                    margin_x + x_offset + w, y_cursor + h,
                ),
                confidence=1.0,
            )
        elif rtype == "list":
            h = rng.randint(60, min(200, remaining_height))
            region = LayoutRegion(
                region_type="list",
                bbox=BBox(
                    margin_x + 30, y_cursor,
                    margin_x + content_width, y_cursor + h,
                ),
                confidence=1.0,
            )
        elif rtype == "caption":
            h = rng.randint(20, 50)
            region = LayoutRegion(
                region_type="caption",
                bbox=BBox(
                    margin_x, y_cursor,
                    margin_x + content_width, y_cursor + h,
                ),
                confidence=1.0,
            )
        elif rtype == "footer":
            # Footer goes at the bottom -- defer placement
            continue
        else:
            h = rng.randint(40, min(150, remaining_height))
            region = LayoutRegion(
                region_type=rtype,
                bbox=BBox(
                    margin_x, y_cursor,
                    margin_x + content_width, y_cursor + h,
                ),
                confidence=1.0,
            )

        color = region_colors.get(rtype, (235, 235, 235))
        _draw_filled_rect(
            pixels, region.bbox.x1, region.bbox.y1,
            region.bbox.x2, region.bbox.y2, color, page_width, page_height,
        )

        # Draw simulated text content
        if rtype in ("title", "paragraph", "list", "caption"):
            _draw_text_lines(
                pixels, region.bbox.x1 + 4, region.bbox.y1 + 2,
                region.bbox.x2 - 4, region.bbox.y2 - 2, rng,
                page_width, page_height,
            )
        elif rtype == "table":
            # Draw grid lines for tables
            num_rows = rng.randint(3, 6)
            num_cols = rng.randint(2, 5)
            row_h = (region.bbox.y2 - region.bbox.y1) // num_rows
            col_w = (region.bbox.x2 - region.bbox.x1) // num_cols
            for r_idx in range(num_rows + 1):
                gy = region.bbox.y1 + r_idx * row_h
                _draw_filled_rect(
                    pixels, region.bbox.x1, gy, region.bbox.x2,
                    min(gy + 2, region.bbox.y2), (100, 100, 100),
                    page_width, page_height,
                )
            for c_idx in range(num_cols + 1):
                gx = region.bbox.x1 + c_idx * col_w
                _draw_filled_rect(
                    pixels, gx, region.bbox.y1, min(gx + 2, region.bbox.x2),
                    region.bbox.y2, (100, 100, 100), page_width, page_height,
                )

        regions.append(region)
        y_cursor = region.bbox.y2 + rng.randint(15, 40)

    # Footer at the bottom
    if "footer" in chosen_types:
        footer_h = rng.randint(30, 50)
        footer_y = page_height - margin_bottom - footer_h
        if footer_y > y_cursor:
            region = LayoutRegion(
                region_type="footer",
                bbox=BBox(margin_x, footer_y, margin_x + content_width, footer_y + footer_h),
                confidence=1.0,
            )
            color = region_colors.get("footer", (230, 230, 230))
            _draw_filled_rect(
                pixels, region.bbox.x1, region.bbox.y1,
                region.bbox.x2, region.bbox.y2, color, page_width, page_height,
            )
            _draw_text_lines(
                pixels, region.bbox.x1 + 4, region.bbox.y1 + 2,
                region.bbox.x2 - 4, region.bbox.y2 - 2, rng,
                page_width, page_height,
            )
            regions.append(region)

    img = Image.frombytes("RGB", (page_width, page_height), bytes(pixels))

    return AnnotatedPage(
        width=page_width,
        height=page_height,
        regions=regions,
    ), img


def generate_corpus(
    output_dir: str,
    count: int = 50,
    page_width: int = 2550,
    page_height: int = 3300,
) -> list:
    """Generate a synthetic corpus of annotated document pages.

    Parameters
    ----------
    output_dir : str
        Directory to write images and annotation JSON files.
    count : int
        Number of pages to generate.
    page_width, page_height : int
        Page dimensions.

    Returns
    -------
    list[str]
        List of generated annotation JSON paths.
    """
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    annotations_dir = out_path / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    annotation_paths = []

    for i in range(count):
        page, img = generate_synthetic_page(
            page_width=page_width,
            page_height=page_height,
            seed=i,
        )

        img_filename = f"page_{i:04d}.png"
        img_path = images_dir / img_filename
        img.save(str(img_path))

        page.image_path = str(img_path)

        annotation = {
            "image_path": str(img_path),
            "width": page.width,
            "height": page.height,
            "regions": [
                {
                    "region_type": r.region_type,
                    "bbox": r.bbox.to_list(),
                    "confidence": r.confidence,
                }
                for r in page.regions
            ],
        }
        ann_filename = f"page_{i:04d}.json"
        ann_path = annotations_dir / ann_filename
        with open(str(ann_path), "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2)

        annotation_paths.append(str(ann_path))
        logger.debug("Generated page %d with %d regions", i, len(page.regions))

    logger.info("Generated %d pages in %s", count, output_dir)
    return annotation_paths


def load_corpus(corpus_dir: str) -> list:
    """Load an annotated corpus from disk.

    Parameters
    ----------
    corpus_dir : str
        Root directory containing images/ and annotations/ subdirectories.

    Returns
    -------
    list[tuple[AnnotatedPage, PIL.Image.Image]]
        List of (page_annotation, image) tuples.
    """
    from PIL import Image

    corpus_path = Path(corpus_dir)
    annotations_dir = corpus_path / "annotations"
    if not annotations_dir.is_dir():
        logger.error("No annotations/ directory found in %s", corpus_dir)
        return []

    pages = []
    for ann_file in sorted(annotations_dir.glob("*.json")):
        with open(str(ann_file), "r", encoding="utf-8") as f:
            ann = json.load(f)

        regions = []
        for r in ann.get("regions", []):
            bbox_list = r.get("bbox", [0, 0, 0, 0])
            regions.append(LayoutRegion(
                region_type=r["region_type"],
                bbox=BBox(*bbox_list),
                confidence=r.get("confidence", 1.0),
            ))

        page = AnnotatedPage(
            image_path=ann.get("image_path", ""),
            width=ann.get("width", 0),
            height=ann.get("height", 0),
            regions=regions,
        )

        # Load image
        img_path = Path(page.image_path)
        if not img_path.is_file():
            # Try relative to corpus dir
            alt_path = corpus_path / "images" / img_path.name
            if alt_path.is_file():
                img_path = alt_path
            else:
                logger.warning("Image not found: %s", page.image_path)
                continue

        img = Image.open(str(img_path)).convert("RGB")
        pages.append((page, img))

    logger.info("Loaded %d pages from %s", len(pages), corpus_dir)
    return pages


# ---------------------------------------------------------------------------
# Model runners (all with graceful degradation)
# ---------------------------------------------------------------------------


def run_ppstructure(image) -> list:
    """Run PP-StructureV3 layout detection on a page image.

    Parameters
    ----------
    image : PIL.Image.Image
        Page image.

    Returns
    -------
    list[LayoutRegion]
        Detected layout regions with bounding boxes and types.
    """
    try:
        import numpy as np
        from paddleocr import PPStructure
    except ImportError:
        return []

    try:
        engine = PPStructure(show_log=False, recovery=False, layout=True, table=False)
        img_np = np.array(image)
        result = engine(img_np)
    except Exception as exc:
        logger.warning("PP-StructureV3 inference failed: %s", exc)
        return []

    regions = []
    if result:
        for item in result:
            if isinstance(item, dict):
                bbox_raw = item.get("bbox", [])
                rtype = item.get("type", "").lower()
                conf = float(item.get("score", 0.8))

                # Map PPStructure types to our canonical types
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
                mapped_type = type_map.get(rtype, rtype)

                if len(bbox_raw) >= 4:
                    regions.append(LayoutRegion(
                        region_type=mapped_type,
                        bbox=BBox(
                            int(bbox_raw[0]), int(bbox_raw[1]),
                            int(bbox_raw[2]), int(bbox_raw[3]),
                        ),
                        confidence=conf,
                    ))

    return regions


def run_layoutlmv3(image, ocr_words: list = None, ocr_boxes: list = None) -> list:
    """Run LayoutLMv3 layout classification on a page image.

    Uses the existing classification.py MLDocumentClassifier to get
    document-level classification. For layout-level predictions, this
    simulates region-level classification by running the model on
    the full page and producing a single region.

    Parameters
    ----------
    image : PIL.Image.Image
        Page image.
    ocr_words : list[str], optional
        OCR word tokens.
    ocr_boxes : list[list[int]], optional
        Bounding boxes per word [x1, y1, x2, y2].

    Returns
    -------
    list[LayoutRegion]
        Detected layout regions.
    """
    try:
        from classification import MLDocumentClassifier
    except ImportError:
        return []

    try:
        classifier = MLDocumentClassifier()
        text = " ".join(ocr_words) if ocr_words else ""
        doc_type, confidence = classifier.classify(
            text=text,
            bbox_list=ocr_boxes,
            page_image=image,
        )

        if doc_type == "other" and confidence == 0.0:
            # Model not available
            return []

        # Map document type to layout region
        w, h = image.size
        regions = [
            LayoutRegion(
                region_type="paragraph",
                bbox=BBox(0, 0, w, h),
                confidence=confidence,
            )
        ]
        return regions

    except Exception as exc:
        logger.warning("LayoutLMv3 inference failed: %s", exc)
        return []


def run_heuristic(text: str, page_width: int = 2550, page_height: int = 3300) -> list:
    """Run heuristic text-based classification as a baseline.

    Uses classification.py's text pattern matching to predict the document
    type and produces a single full-page region as the prediction.

    Parameters
    ----------
    text : str
        OCR text content.
    page_width, page_height : int
        Page dimensions for the full-page bounding box.

    Returns
    -------
    list[LayoutRegion]
        Baseline layout predictions.
    """
    try:
        from classification import classify_page_by_text
    except ImportError:
        return []

    try:
        result = classify_page_by_text(text, page_num=0)

        if isinstance(result, dict):
            doc_type = result.get("predicted_type", "other")
            conf = result.get("confidence", 0.5)
        elif isinstance(result, str):
            doc_type = result
            conf = 0.5
        elif hasattr(result, "predicted_type"):
            doc_type = result.predicted_type
            conf = getattr(result, "confidence", 0.5)
        else:
            doc_type = "other"
            conf = 0.5

        # Map to layout region type
        type_map = {
            "invoice": "table",
            "contract": "paragraph",
            "letter": "paragraph",
            "form": "table",
            "report": "paragraph",
            "memo": "paragraph",
            "receipt": "list",
        }
        region_type = type_map.get(doc_type, "paragraph")

        return [
            LayoutRegion(
                region_type=region_type,
                bbox=BBox(0, 0, page_width, page_height),
                confidence=conf,
            )
        ]

    except Exception as exc:
        logger.warning("Heuristic classification failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _check_model_available(model_name: str) -> bool:
    """Check if a model backend is available for benchmarking.

    Parameters
    ----------
    model_name : str
        Model name: "ppstructure", "layoutlmv3", or "heuristic".

    Returns
    -------
    bool
        True if the model dependencies are importable.
    """
    if model_name == "ppstructure":
        try:
            from paddleocr import PPStructure  # noqa: F401
            return True
        except ImportError:
            return False
    elif model_name == "layoutlmv3":
        try:
            import torch  # noqa: F401
            from transformers import LayoutLMv3ForSequenceClassification  # noqa: F401
            return True
        except ImportError:
            return False
    elif model_name == "heuristic":
        try:
            from classification import classify_page_by_text  # noqa: F401
            return True
        except ImportError:
            return False
    return False


def run_benchmark(
    corpus_dir: str,
    models: list = None,
) -> BenchmarkReport:
    """Run the benchmark across specified models on a corpus.

    Parameters
    ----------
    corpus_dir : str
        Path to the corpus directory (with images/ and annotations/).
    models : list[str], optional
        Model names to benchmark. Default: all available.
        Options: "ppstructure", "layoutlmv3", "heuristic"

    Returns
    -------
    BenchmarkReport
        Complete comparison results.
    """
    if models is None:
        models = ["ppstructure", "layoutlmv3", "heuristic"]

    pages = load_corpus(corpus_dir)
    if not pages:
        logger.error("No pages loaded from corpus. Run 'generate' first.")
        return BenchmarkReport(
            corpus_dir=corpus_dir,
            timestamp=datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="milliseconds"),
        )

    # Collect all GT region types
    all_gt_regions = []
    for page, _img in pages:
        all_gt_regions.extend(page.regions)
    gt_types = sorted(set(r.region_type for r in all_gt_regions))

    report = BenchmarkReport(
        corpus_dir=corpus_dir,
        total_pages=len(pages),
        region_types=gt_types,
        timestamp=datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="milliseconds"),
    )

    for model_name in models:
        logger.info("Benchmarking model: %s", model_name)
        available = _check_model_available(model_name)

        model_result = ModelResult(
            model_name=model_name,
            available=available,
            total_pages=len(pages),
            timestamp=datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="milliseconds"),
        )

        if not available:
            logger.warning(
                "Model %s not available (missing dependencies). Skipping.",
                model_name,
            )
            report.models.append(model_result)
            continue

        all_predictions = []
        all_ground_truth = []
        timings = []

        for page, img in pages:
            start = time.perf_counter()

            if model_name == "ppstructure":
                preds = run_ppstructure(img)
            elif model_name == "layoutlmv3":
                preds = run_layoutlmv3(img)
            elif model_name == "heuristic":
                preds = run_heuristic("", page.width, page.height)
            else:
                preds = []

            elapsed_ms = (time.perf_counter() - start) * 1000
            timings.append(elapsed_ms)

            all_predictions.extend(preds)
            all_ground_truth.extend(page.regions)

        # Compute metrics
        if timings:
            model_result.avg_inference_ms = round(statistics.mean(timings), 2)
            model_result.min_inference_ms = round(min(timings), 2)
            model_result.max_inference_ms = round(max(timings), 2)
            sorted_timings = sorted(timings)
            p95_idx = min(int(len(sorted_timings) * 0.95), len(sorted_timings) - 1)
            model_result.p95_inference_ms = round(sorted_timings[p95_idx], 2)

        model_result.per_type_iou = compute_per_type_iou(
            all_predictions, all_ground_truth, gt_types,
        )
        iou_values = [v for v in model_result.per_type_iou.values() if v > 0]
        model_result.mean_iou = (
            round(statistics.mean(iou_values), 4) if iou_values else 0.0
        )

        model_result.map_50 = compute_map(
            all_predictions, all_ground_truth, gt_types, 0.5,
        )
        model_result.map_75 = compute_map(
            all_predictions, all_ground_truth, gt_types, 0.75,
        )

        model_result.per_type_f1 = compute_per_type_f1(
            all_predictions, all_ground_truth, gt_types, 0.5,
        )
        f1_values = [v for v in model_result.per_type_f1.values() if v > 0]
        model_result.mean_f1 = (
            round(statistics.mean(f1_values), 4) if f1_values else 0.0
        )

        report.models.append(model_result)
        logger.info(
            "  %s: mIoU=%.4f, mAP@50=%.4f, mAP@75=%.4f, mF1=%.4f, "
            "avg_ms=%.1f",
            model_name,
            model_result.mean_iou,
            model_result.map_50,
            model_result.map_75,
            model_result.mean_f1,
            model_result.avg_inference_ms,
        )

    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_markdown_report(report: BenchmarkReport) -> str:
    """Format benchmark report as a Markdown comparison table.

    Parameters
    ----------
    report : BenchmarkReport
        Benchmark results.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = []
    lines.append("# DocLayNet Layout Analysis Benchmark")
    lines.append("")
    lines.append(f"**Corpus:** {report.corpus_dir}")
    lines.append(f"**Pages:** {report.total_pages}")
    lines.append(f"**Region types:** {', '.join(report.region_types)}")
    lines.append(f"**Timestamp:** {report.timestamp}")
    lines.append("")

    # Overall comparison table
    lines.append("## Overall Comparison")
    lines.append("")
    lines.append(
        "| Model | Available | mIoU | mAP@50 | mAP@75 | mF1 | "
        "Avg (ms) | P95 (ms) |"
    )
    lines.append(
        "|-------|-----------|------|--------|--------|-----|"
        "----------|----------|"
    )

    for m in report.models:
        avail = "Yes" if m.available else "No"
        lines.append(
            f"| {m.model_name} | {avail} | "
            f"{m.mean_iou:.4f} | {m.map_50:.4f} | {m.map_75:.4f} | "
            f"{m.mean_f1:.4f} | {m.avg_inference_ms:.1f} | "
            f"{m.p95_inference_ms:.1f} |"
        )

    lines.append("")

    # Per-type IoU table
    if any(m.per_type_iou for m in report.models):
        lines.append("## Per-Type IoU")
        lines.append("")
        header = "| Region Type |"
        sep = "|-------------|"
        for m in report.models:
            header += f" {m.model_name} |"
            sep += "--------|"
        lines.append(header)
        lines.append(sep)

        for rtype in report.region_types:
            row = f"| {rtype} |"
            for m in report.models:
                val = m.per_type_iou.get(rtype, 0.0)
                row += f" {val:.4f} |"
            lines.append(row)

        lines.append("")

    # Per-type F1 table
    if any(m.per_type_f1 for m in report.models):
        lines.append("## Per-Type F1 Score")
        lines.append("")
        header = "| Region Type |"
        sep = "|-------------|"
        for m in report.models:
            header += f" {m.model_name} |"
            sep += "--------|"
        lines.append(header)
        lines.append(sep)

        for rtype in report.region_types:
            row = f"| {rtype} |"
            for m in report.models:
                val = m.per_type_f1.get(rtype, 0.0)
                row += f" {val:.4f} |"
            lines.append(row)

        lines.append("")

    # Inference timing
    lines.append("## Inference Timing")
    lines.append("")
    lines.append("| Model | Avg (ms) | P95 (ms) | Min (ms) | Max (ms) |")
    lines.append("|-------|----------|----------|----------|----------|")
    for m in report.models:
        if m.available:
            lines.append(
                f"| {m.model_name} | {m.avg_inference_ms:.1f} | "
                f"{m.p95_inference_ms:.1f} | {m.min_inference_ms:.1f} | "
                f"{m.max_inference_ms:.1f} |"
            )

    lines.append("")
    return "\n".join(lines)


def format_text_report(report: BenchmarkReport) -> str:
    """Format benchmark report as a plain-text report for console display.

    Parameters
    ----------
    report : BenchmarkReport
        Benchmark results.

    Returns
    -------
    str
        Plain-text formatted report.
    """
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("DOCLAYNET LAYOUT ANALYSIS BENCHMARK")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"  Corpus:        {report.corpus_dir}")
    lines.append(f"  Pages:         {report.total_pages}")
    lines.append(f"  Region types:  {', '.join(report.region_types)}")
    lines.append(f"  Timestamp:     {report.timestamp}")
    lines.append("")

    # Overall comparison
    lines.append("OVERALL COMPARISON")
    lines.append("-" * 90)
    lines.append(
        f"{'Model':<20} {'Avail':>6} {'mIoU':>8} {'mAP@50':>8} "
        f"{'mAP@75':>8} {'mF1':>8} {'Avg(ms)':>10} {'P95(ms)':>10}"
    )
    lines.append("-" * 90)
    for m in report.models:
        avail = "Yes" if m.available else "No"
        lines.append(
            f"{m.model_name:<20} {avail:>6} {m.mean_iou:>8.4f} "
            f"{m.map_50:>8.4f} {m.map_75:>8.4f} {m.mean_f1:>8.4f} "
            f"{m.avg_inference_ms:>10.1f} {m.p95_inference_ms:>10.1f}"
        )
    lines.append("-" * 90)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for the DocLayNet benchmark tool."""
    parser = argparse.ArgumentParser(
        description=(
            "DocLayNet layout analysis benchmark: "
            "PP-StructureV3 vs LayoutLMv3 comparison"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_doclaynet.py generate --count 50 --output-dir benchmark_data/doclaynet
  python scripts/benchmark_doclaynet.py run --corpus-dir benchmark_data/doclaynet --models all
  python scripts/benchmark_doclaynet.py run --corpus-dir benchmark_data/doclaynet --models ppstructure layoutlmv3
  python scripts/benchmark_doclaynet.py report --results results.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Benchmark command")

    # Generate subcommand
    gen_parser = subparsers.add_parser(
        "generate", help="Generate synthetic corpus with ground truth"
    )
    gen_parser.add_argument(
        "--count", type=int, default=50,
        help="Number of pages to generate (default: 50)",
    )
    gen_parser.add_argument(
        "--output-dir", type=str, default="benchmark_data/doclaynet",
        help="Output directory for generated corpus",
    )
    gen_parser.add_argument(
        "--width", type=int, default=2550,
        help="Page width in pixels (default: 2550)",
    )
    gen_parser.add_argument(
        "--height", type=int, default=3300,
        help="Page height in pixels (default: 3300)",
    )

    # Run subcommand
    run_parser = subparsers.add_parser(
        "run", help="Run benchmark on a corpus"
    )
    run_parser.add_argument(
        "--corpus-dir", type=str, required=True,
        help="Path to corpus directory",
    )
    run_parser.add_argument(
        "--models", nargs="+",
        default=["all"],
        help="Models to benchmark: ppstructure, layoutlmv3, heuristic, all",
    )
    run_parser.add_argument(
        "--output", type=str,
        help="Output JSON path for structured results",
    )

    # Report subcommand
    report_parser = subparsers.add_parser(
        "report", help="Generate report from saved results"
    )
    report_parser.add_argument(
        "--results", type=str, required=True,
        help="Path to results JSON file",
    )
    report_parser.add_argument(
        "--format", choices=["text", "markdown"], default="text",
        help="Report format (default: text)",
    )

    # Global args
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "generate":
        generate_corpus(
            output_dir=args.output_dir,
            count=args.count,
            page_width=args.width,
            page_height=args.height,
        )
        logger.info("Corpus generated at %s", args.output_dir)
        return 0

    elif args.command == "run":
        models = args.models
        if "all" in models:
            models = ["ppstructure", "layoutlmv3", "heuristic"]

        report = run_benchmark(
            corpus_dir=args.corpus_dir,
            models=models,
        )

        # Display text report
        text_report = format_text_report(report)
        print(text_report)

        # Save JSON if requested
        if args.output:
            report_dict = asdict(report)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2)
            logger.info("Results saved to %s", args.output)

        return 0

    elif args.command == "report":
        results_path = Path(args.results)
        if not results_path.is_file():
            logger.error("Results file not found: %s", args.results)
            return 1

        with open(str(results_path), "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct report from JSON
        report = BenchmarkReport(
            corpus_dir=data.get("corpus_dir", ""),
            total_pages=data.get("total_pages", 0),
            region_types=data.get("region_types", []),
            timestamp=data.get("timestamp", ""),
        )
        for m_data in data.get("models", []):
            report.models.append(ModelResult(
                model_name=m_data.get("model_name", ""),
                available=m_data.get("available", False),
                total_pages=m_data.get("total_pages", 0),
                per_type_iou=m_data.get("per_type_iou", {}),
                mean_iou=m_data.get("mean_iou", 0.0),
                map_50=m_data.get("map_50", 0.0),
                map_75=m_data.get("map_75", 0.0),
                per_type_f1=m_data.get("per_type_f1", {}),
                mean_f1=m_data.get("mean_f1", 0.0),
                avg_inference_ms=m_data.get("avg_inference_ms", 0.0),
                p95_inference_ms=m_data.get("p95_inference_ms", 0.0),
                min_inference_ms=m_data.get("min_inference_ms", 0.0),
                max_inference_ms=m_data.get("max_inference_ms", 0.0),
                timestamp=m_data.get("timestamp", ""),
            ))

        if args.format == "markdown":
            print(format_markdown_report(report))
        else:
            print(format_text_report(report))

        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
