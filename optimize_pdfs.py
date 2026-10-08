import argparse
import logging
import math
import os
import stat
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import fitz
import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageStat

# --- Configuration ---
INPUT_DIR = "/app/ocr_output"
# Ghostscript PDF Settings:
# /screen   (72 dpi, smallest)
# /ebook    (150 dpi, medium - Recommended)
# /printer  (300 dpi, high)
# /prepress (300 dpi, max quality, preserves color)
DEFAULT_QUALITY = "/prepress"
VALID_QUALITIES = {"/screen", "/ebook", "/printer", "/prepress"}
GHOSTSCRIPT_TIMEOUT = 300  # seconds; prevents hung threads on malformed PDFs
# Page MediaBox drift tolerance in PDF points (1 pt = 1/72 in).
# Ghostscript can introduce sub-point float noise; 1.0 pt still catches
# meaningful crop/scale changes that would alter layout/evidence geometry.
PAGE_DIMENSION_TOLERANCE_PT = 1.0
# Searchable OCR text is often rendered invisibly, so raster comparison cannot
# detect a displaced text layer.  Compare every extracted word rectangle in
# PDF points as a separate forensic invariant.  Ghostscript may introduce
# small floating-point/font-metric drift, but movement beyond one half point
# is not benign compression noise.
TEXT_GEOMETRY_TOLERANCE_PT = 0.5
# PDF writers legitimately encode different font ascent/descent extents around
# the same unchanged baseline.  Ghostscript /prepress shifts the word-box top
# by ~2.75pt for a 9pt Helvetica span while preserving the exact span origin.
# Permit that metric-only shape change, but independently pin span origins to
# TEXT_GEOMETRY_TOLERANCE_PT below.
TEXT_WORD_VERTICAL_METRIC_TOLERANCE_PT = 4.0
TEXT_WORD_VERTICAL_METRIC_TOLERANCE_RATIO = 0.35

# --- Rendered page equivalence (visual forensic gate) ---
# Source and candidate pages are rasterized at RENDER_NOMINAL_DPI, then the
# scale is reduced so width*height stays at or under RENDER_MAX_PIXELS
# (~1.25M).  One page pair is processed at a time (metric-only errors).
#
# Minimum effective comparison DPI (RENDER_MIN_EFFECTIVE_DPI = 36):
#   If honoring the pixel cap would force effective DPI below this floor,
#   optimization is skipped (SKIPPED_VISUAL_BUDGET) before Ghostscript runs.
#   This is a *compression* validation budget, not an OCR input rejection:
#   the original PDF is left byte-identical.  At 36 DPI + 1.25M pixels,
#   normal letter/A4 through roughly A2 stay comparable; larger canvases
#   (A1+, multi-meter engineering sheets) refuse rather than dilute sparse
#   signature/field detection.
#
# Comparison is intentionally noise-tolerant for ordinary Ghostscript
# /prepress re-encoding while remaining fail-closed for wholesale
# color/image/background substitution and field/signature-sized (including
# sparse/light) localized changes:
#   * global: downsample + mild blur MAE + changed-pixel fraction + coarse
#     per-tile MAE (tolerates mild recompression noise)
#   * local MAE: full-resolution (bounded) non-overlapping 16x16 raw MAE
#     (no per-tile PIL crops; NumPy aggregation)
#   * local blur-overlap changed fraction: BoxBlur(2) each plane *separately*,
#     then max-channel delta; full 16x16 windows with stride 8 plus final
#     right/bottom anchors (no partial edge tiles)
#   * local energy: BoxBlur(3), max-channel delta, full 8x8 windows at stride
#     4, with both MAE and RMS caps.  The smaller scale catches 2pt x 1pt
#     black marks and 0.5pt x 40pt light-gray strokes at page edges/phases.
#
# Threshold notes:
#   RENDER_MAE_MAX / RENDER_CHANGED_FRAC_MAX / RENDER_TILE_MAE_MAX —
#     global noise gate on the downsampled plane.
#   RENDER_LOCAL_TILE_MAE_MAX / RENDER_LOCAL_BLUR_CHANGED_FRAC_MAX —
#     full-res 16x16 local gates.
#   RENDER_LOCAL_ENERGY_MAE_MAX / RENDER_LOCAL_ENERGY_RMS_MAX — bounded 8x8
#     local energy gates.  A planted dense 0.01pt vector displacement is a
#     calibration reject; a JPEG q95→q90 proxy remains accepted.  A dense
#     0.25pt displacement is also deliberately rejected and is not described
#     as tolerated jitter.
RENDER_NOMINAL_DPI = 120.0
RENDER_MAX_PIXELS = 1_250_000
RENDER_MIN_EFFECTIVE_DPI = 36.0
RENDER_DOWNSAMPLE = 4
RENDER_BLUR_RADIUS = 1
RENDER_PIXEL_DIFF_THRESHOLD = 18  # 0-255 after blur/downsample
RENDER_MAE_MAX = 10.0
RENDER_CHANGED_FRAC_MAX = 0.035
RENDER_TILE_SIZE = 32  # tiles on the downsampled global plane
RENDER_TILE_MAE_MAX = 28.0
RENDER_LOCAL_TILE_SIZE = 16  # full-res local window edge (no downsample)
# Conservative fail-closed local MAE: sparse 1pt lines ~25, light gray fills
# higher; mild recompression noise stays well under 5.
RENDER_LOCAL_TILE_MAE_MAX = 16.0
# Blur-stabilized local changed-fraction gate (phase-tolerant).
RENDER_LOCAL_BLUR_RADIUS = 2
RENDER_LOCAL_BLUR_STRIDE = 8  # overlapping 16x16 windows
RENDER_LOCAL_BLUR_CHANGED_FRAC_MAX = 0.06
RENDER_LOCAL_ENERGY_BLUR_RADIUS = 3
RENDER_LOCAL_ENERGY_WINDOW = 8
RENDER_LOCAL_ENERGY_STRIDE = 4
RENDER_LOCAL_ENERGY_MAE_MAX = 1.7
RENDER_LOCAL_ENERGY_RMS_MAX = 2.1
# Backward-compatible alias used by existing tests/imports.
RENDER_LOCAL_TILE_CHANGED_FRAC_MAX = RENDER_LOCAL_BLUR_CHANGED_FRAC_MAX


class OptimizationDisposition(str, Enum):
    """Detailed optimizer outcomes for forensic callers."""

    APPLIED = "APPLIED"
    SKIPPED_VISUAL_BUDGET = "SKIPPED_VISUAL_BUDGET"
    SKIPPED_CANDIDATE_REJECTED = "SKIPPED_CANDIDATE_REJECTED"
    FAILED_PRESERVED = "FAILED_PRESERVED"
    FAILED_INDETERMINATE = "FAILED_INDETERMINATE"


@dataclass(frozen=True)
class OptimizeResult:
    """Structured optimize_pdf outcome.

    ``applied`` is True only when Ghostscript replacement was committed.
    Public ``optimize_pdf(...)`` returns that boolean for compatibility.
    """

    disposition: OptimizationDisposition
    reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def applied(self) -> bool:
        return self.disposition is OptimizationDisposition.APPLIED

    @property
    def skipped_visual_budget(self) -> bool:
        return self.disposition is OptimizationDisposition.SKIPPED_VISUAL_BUDGET

    @property
    def skipped_candidate_rejected(self) -> bool:
        return self.disposition is OptimizationDisposition.SKIPPED_CANDIDATE_REJECTED


# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def _normalize_extracted_text(text):
    """Normalize searchable text for per-page forensic equivalence checks."""
    if not text:
        return ""
    return " ".join(str(text).split())


def _page_word_geometry(page):
    """Return reading-order ``(text, bbox)`` tuples for searchable words.

    PyMuPDF's rendered pixmap intentionally omits invisible OCR text, so this
    geometry is validated independently from the RGB comparison.  Malformed
    or non-finite word rectangles fail closed instead of being ignored.
    """
    words = page.get_text("words", sort=True)
    geometry = []
    for word in words:
        if len(word) < 5:
            raise ValueError("searchable word record is missing its bounding box")
        token = _normalize_extracted_text(word[4])
        if not token:
            continue
        bbox = tuple(float(value) for value in word[:4])
        if not all(math.isfinite(value) for value in bbox):
            raise ValueError("searchable word bounding box is non-finite")
        x0, y0, x1, y1 = bbox
        if x1 < x0 or y1 < y0:
            raise ValueError("searchable word bounding box is inverted")
        geometry.append((token, bbox))
    return geometry


def _page_text_span_geometry(page):
    """Return non-empty span text, baseline origin, and font size in order."""
    spans = []
    text_dict = page.get_text("dict", sort=True)
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = _normalize_extracted_text(span.get("text", ""))
                if not text:
                    continue
                origin = span.get("origin")
                if not isinstance(origin, (list, tuple)) or len(origin) != 2:
                    raise ValueError("searchable text span is missing its origin")
                origin = tuple(float(value) for value in origin)
                size = float(span.get("size", 0.0))
                if not all(math.isfinite(value) for value in (*origin, size)):
                    raise ValueError("searchable text span geometry is non-finite")
                if size <= 0:
                    raise ValueError("searchable text span font size is invalid")
                spans.append((text, origin, size))
    return spans


def _word_geometry_equivalent(
    original_page,
    candidate_page,
    *,
    tolerance=TEXT_GEOMETRY_TOLERANCE_PT,
):
    """Compare word identity and placement for one searchable PDF page."""
    original_spans = _page_text_span_geometry(original_page)
    candidate_spans = _page_text_span_geometry(candidate_page)
    if len(candidate_spans) != len(original_spans):
        return False, (
            "searchable text span geometry count changed during optimization: "
            f"source={len(original_spans)}, output={len(candidate_spans)}"
        )
    max_delta = 0.0
    for span_index, (original_span, candidate_span) in enumerate(
        zip(original_spans, candidate_spans),
        start=1,
    ):
        original_text, original_origin, original_size = original_span
        candidate_text, candidate_origin, candidate_size = candidate_span
        if candidate_text != original_text:
            return False, (
                "searchable text span sequence changed during optimization at "
                f"span {span_index}"
            )
        origin_delta = max(
            abs(candidate_value - original_value)
            for original_value, candidate_value in zip(original_origin, candidate_origin)
        )
        size_delta = abs(candidate_size - original_size)
        max_delta = max(max_delta, origin_delta, size_delta)
        if origin_delta > tolerance or size_delta > tolerance:
            return False, (
                "searchable word geometry moved during optimization at "
                f"span {span_index}: origin_delta={origin_delta:.3f}pt, "
                f"size_delta={size_delta:.3f}pt (tolerance={tolerance:.3f}pt)"
            )

    original_words = _page_word_geometry(original_page)
    candidate_words = _page_word_geometry(candidate_page)
    if len(candidate_words) != len(original_words):
        return False, (
            "searchable word geometry count changed during optimization: "
            f"source={len(original_words)}, output={len(candidate_words)}"
        )

    for word_index, (original_word, candidate_word) in enumerate(
        zip(original_words, candidate_words),
        start=1,
    ):
        original_text, original_bbox = original_word
        candidate_text, candidate_bbox = candidate_word
        if candidate_text != original_text:
            return False, (
                "searchable word sequence changed during optimization at "
                f"word {word_index}"
            )
        horizontal_delta = max(
            abs(candidate_bbox[index] - original_bbox[index])
            for index in (0, 2)
        )
        vertical_metric_delta = max(
            abs(candidate_bbox[index] - original_bbox[index])
            for index in (1, 3)
        )
        original_height = original_bbox[3] - original_bbox[1]
        vertical_metric_tolerance = max(
            TEXT_WORD_VERTICAL_METRIC_TOLERANCE_PT,
            original_height * TEXT_WORD_VERTICAL_METRIC_TOLERANCE_RATIO,
        )
        max_delta = max(max_delta, horizontal_delta, vertical_metric_delta)
        if (
            horizontal_delta > tolerance
            or vertical_metric_delta > vertical_metric_tolerance
        ):
            return False, (
                "searchable word geometry moved during optimization at "
                f"word {word_index}: horizontal_delta={horizontal_delta:.3f}pt, "
                f"vertical_metric_delta={vertical_metric_delta:.3f}pt "
                f"(position_tolerance={tolerance:.3f}pt, "
                f"vertical_metric_tolerance={vertical_metric_tolerance:.3f}pt)"
            )

    return True, max_delta


def _page_dimensions(page):
    """Return (width, height) in PDF points for *page*."""
    rect = page.rect
    return float(rect.width), float(rect.height)


def _dimensions_match(original_size, candidate_size, tolerance=PAGE_DIMENSION_TOLERANCE_PT):
    """Return True when width/height stay within *tolerance* PDF points."""
    original_width, original_height = original_size
    candidate_width, candidate_height = candidate_size
    return (
        abs(candidate_width - original_width) <= tolerance
        and abs(candidate_height - original_height) <= tolerance
    )


def _render_scale_for_page(
    width_pt,
    height_pt,
    *,
    nominal_dpi=RENDER_NOMINAL_DPI,
    max_pixels=RENDER_MAX_PIXELS,
    min_effective_dpi=RENDER_MIN_EFFECTIVE_DPI,
):
    """Return ``(scale, metrics)`` for one page under the pixel / DPI budget.

    At *nominal_dpi*, PDF points map as ``pixels = points * (dpi / 72)``.
    If the resulting area would exceed *max_pixels*, the scale is reduced by
    ``sqrt(max_pixels / area)`` so each rendered page stays bounded.

    When that reduced scale would fall below *min_effective_dpi*, returns
    ``(None, metrics)`` so callers fail closed instead of comparing a diluted
    raster.  *metrics* always includes numeric fields only (no page content):
    ``width_pt``, ``height_pt``, ``scale``, ``effective_dpi``, ``max_pixels``,
    ``min_effective_dpi``, ``capped``.
    """
    width_pt = max(float(width_pt), 1e-6)
    height_pt = max(float(height_pt), 1e-6)
    scale = float(nominal_dpi) / 72.0
    area = (width_pt * scale) * (height_pt * scale)
    capped = False
    if area > max_pixels:
        scale *= math.sqrt(max_pixels / area)
        capped = True
    effective_dpi = scale * 72.0
    metrics = {
        "width_pt": width_pt,
        "height_pt": height_pt,
        "scale": scale,
        "effective_dpi": effective_dpi,
        "max_pixels": float(max_pixels),
        "min_effective_dpi": float(min_effective_dpi),
        "capped": 1.0 if capped else 0.0,
    }
    if effective_dpi + 1e-9 < float(min_effective_dpi):
        return None, metrics
    return scale, metrics


def _render_page_rgb(page, *, scale=None):
    """Render *page* to an RGB :class:`PIL.Image.Image` under the pixel cap.

    Returns ``(image, scale_used)``.  Caller must close/discard the image after
    the per-page comparison so only one page pair is live at a time.

    *scale* must already be a positive finite zoom accepted by
    :func:`_render_scale_for_page` (callers that hit the min-DPI floor must not
    render).
    """
    if scale is None:
        width_pt, height_pt = _page_dimensions(page)
        scale, scale_metrics = _render_scale_for_page(width_pt, height_pt)
        if scale is None:
            raise ValueError(
                "page exceeds compression visual-validation budget: "
                f"effective_dpi={scale_metrics['effective_dpi']:.2f}, "
                f"min_effective_dpi={scale_metrics['min_effective_dpi']:.2f}, "
                f"max_pixels={int(scale_metrics['max_pixels'])}"
            )
    matrix = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
    try:
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    finally:
        pixmap = None
    return image, scale


def _downsample_blur(image, *, downsample=RENDER_DOWNSAMPLE, blur_radius=RENDER_BLUR_RADIUS):
    """Return a small blurred RGB image for noise-tolerant comparison."""
    width, height = image.size
    factor = max(int(downsample), 1)
    small_size = (max(1, width // factor), max(1, height // factor))
    reduced = image.resize(small_size, Image.Resampling.BILINEAR)
    if blur_radius and blur_radius > 0:
        reduced = reduced.filter(ImageFilter.BoxBlur(blur_radius))
    return reduced


def _mean_absolute_error(image_a, image_b):
    """Mean absolute per-channel error across two same-sized RGB images."""
    diff = ImageChops.difference(image_a, image_b)
    stats = ImageStat.Stat(diff)
    # stats.mean is per-band; average across RGB.
    return sum(stats.mean) / max(len(stats.mean), 1)


def _changed_pixel_fraction(image_a, image_b, *, threshold=RENDER_PIXEL_DIFF_THRESHOLD):
    """Fraction of pixels whose max per-channel delta exceeds *threshold*."""
    diff = ImageChops.difference(image_a, image_b)
    # Convert max-channel difference to a single-plane "L" image via max of RGB.
    extrema = diff.split()
    max_plane = extrema[0]
    for plane in extrema[1:]:
        max_plane = ImageChops.lighter(max_plane, plane)
    # Pixels with value > threshold count as changed.
    # point() keeps 0 for <= threshold and 255 for > threshold.
    mask = max_plane.point(lambda value: 255 if value > threshold else 0)
    changed = ImageStat.Stat(mask).sum[0] / 255.0
    total = max_plane.size[0] * max_plane.size[1]
    if total <= 0:
        return 0.0
    return changed / total


def _max_tile_mae(image_a, image_b, *, tile_size=RENDER_TILE_SIZE):
    """Maximum MAE across non-overlapping tiles (global/downsampled plane).

    Last edge tiles may be smaller.  Used only on the already-small global
    plane; full-res local MAE uses :func:`_max_local_mae_numpy`.
    """
    width, height = image_a.size
    tile = max(int(tile_size), 1)
    worst = 0.0
    for top in range(0, height, tile):
        for left in range(0, width, tile):
            box = (left, top, min(left + tile, width), min(top + tile, height))
            mae = _mean_absolute_error(image_a.crop(box), image_b.crop(box))
            if mae > worst:
                worst = mae
    return worst


def _box_blur_rgb_array(arr: np.ndarray, radius: int) -> np.ndarray:
    """Box-blur an HxWx3 uint8 array; returns float32 RGB (same shape)."""
    if radius <= 0:
        return arr.astype(np.float32, copy=False)
    image = Image.fromarray(arr, mode="RGB")
    return np.asarray(
        image.filter(ImageFilter.BoxBlur(int(radius))),
        dtype=np.float32,
    )


def _full_window_origins(length: int, window: int, stride: int):
    """Origins for full *window* slices with *stride*, plus final edge anchor.

    Never emits partial edge windows.  When *length* < *window*, returns a
    single origin at 0 so callers can fall back to the whole axis.
    """
    window = max(int(window), 1)
    stride = max(int(stride), 1)
    length = max(int(length), 0)
    if length <= 0:
        return []
    if length < window:
        return [0]
    last = length - window
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


def _max_local_mae_numpy(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    *,
    tile_size: int = RENDER_LOCAL_TILE_SIZE,
) -> float:
    """Max per-tile raw MAE on full-res RGB arrays (no PIL crop loop).

    Non-overlapping tiles; trailing edge tiles may be smaller (legacy MAE
    behavior retained intentionally — only the changed-fraction gate uses
    full anchored windows).

    Aggregation is fully vectorized via a summed-area table and advanced
    indexing over all tile origin pairs (no per-tile Python loops).
    """
    if arr_a.shape != arr_b.shape:
        raise ValueError("local MAE arrays must share shape")
    # int16 abs-diff is exact for uint8 channels; mean promotes to float64.
    per_pixel = np.mean(
        np.abs(arr_a.astype(np.int16, copy=False) - arr_b.astype(np.int16, copy=False)),
        axis=2,
        dtype=np.float64,
    )
    height, width = per_pixel.shape
    if height <= 0 or width <= 0:
        return 0.0
    tile = max(int(tile_size), 1)

    # Integral image: tile sum via four-corner lookup (correct partial edges).
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    integral[1:, 1:] = per_pixel.cumsum(axis=0).cumsum(axis=1)

    tops = np.arange(0, height, tile, dtype=np.intp)
    lefts = np.arange(0, width, tile, dtype=np.intp)
    bottoms = np.minimum(tops + tile, height)
    rights = np.minimum(lefts + tile, width)

    # Broadcast all (top,left) pairs: shape (n_rows, n_cols).
    totals = (
        integral[bottoms[:, None], rights[None, :]]
        - integral[tops[:, None], rights[None, :]]
        - integral[bottoms[:, None], lefts[None, :]]
        + integral[tops[:, None], lefts[None, :]]
    )
    areas = (bottoms[:, None] - tops[:, None]) * (rights[None, :] - lefts[None, :])
    # areas is always > 0 for non-empty images with the origin construction above.
    maes = totals / areas
    return float(maes.max()) if maes.size else 0.0


def _max_blur_overlap_changed_fraction(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    *,
    blur_radius: int = RENDER_LOCAL_BLUR_RADIUS,
    window: int = RENDER_LOCAL_TILE_SIZE,
    stride: int = RENDER_LOCAL_BLUR_STRIDE,
    threshold: int = RENDER_PIXEL_DIFF_THRESHOLD,
) -> float:
    """Max changed fraction over full blurred overlapping local windows.

    Box-blurs source and candidate *separately*, subtracts (never blurs the
    absolute-diff plane), marks pixels whose max RGB channel delta exceeds
    *threshold*, then aggregates with a summed-area table over full
    *window*×*window* tiles at *stride* plus final right/bottom anchors.

    Window aggregation is fully vectorized (origin arrays + integral
    advanced indexing); no per-window Python loops.
    """
    if arr_a.shape != arr_b.shape:
        raise ValueError("blur-overlap arrays must share shape")
    blur_a = _box_blur_rgb_array(arr_a, blur_radius)
    blur_b = _box_blur_rgb_array(arr_b, blur_radius)
    max_delta = np.max(np.abs(blur_a - blur_b), axis=2)
    changed = (max_delta > float(threshold)).astype(np.float64)
    return _max_overlap_window_mean(changed, window=window, stride=stride)


def _max_overlap_window_mean(
    values: np.ndarray,
    *,
    window: int,
    stride: int,
) -> float:
    """Return the maximum full overlapping-window mean for a 2-D plane.

    Uses one summed-area table and vectorized origin indexing.  Final edge
    anchors are included, partial windows are not, and no per-window Python
    loop is used.  Images smaller than the window use their whole plane.
    """
    if values.ndim != 2:
        raise ValueError("overlap values must be a 2-D plane")
    height, width = values.shape
    win = max(int(window), 1)
    if height < win or width < win:
        return float(values.mean()) if values.size else 0.0

    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    integral[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
    area = float(win * win)

    tops = np.asarray(
        _full_window_origins(height, win, stride),
        dtype=np.intp,
    )
    lefts = np.asarray(
        _full_window_origins(width, win, stride),
        dtype=np.intp,
    )
    if tops.size == 0 or lefts.size == 0:
        return 0.0
    bottoms = tops + win
    rights = lefts + win

    totals = (
        integral[bottoms[:, None], rights[None, :]]
        - integral[tops[:, None], rights[None, :]]
        - integral[bottoms[:, None], lefts[None, :]]
        + integral[tops[:, None], lefts[None, :]]
    )
    means = totals / area
    return float(means.max())


def _max_blur_overlap_mae_energy(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    *,
    blur_radius: int = RENDER_LOCAL_ENERGY_BLUR_RADIUS,
    window: int = RENDER_LOCAL_ENERGY_WINDOW,
    stride: int = RENDER_LOCAL_ENERGY_STRIDE,
) -> Tuple[float, float]:
    """Return max 8x8 blurred max-channel MAE and RMS energy.

    Blurring each render separately suppresses antialias/recompression phase
    noise while preserving the integrated energy of short or light marks.
    Window aggregation is vectorized and bounded by the render pixel cap.
    """
    if arr_a.shape != arr_b.shape:
        raise ValueError("local energy arrays must share shape")
    blur_a = _box_blur_rgb_array(arr_a, blur_radius)
    blur_b = _box_blur_rgb_array(arr_b, blur_radius)
    max_delta = np.max(np.abs(blur_a - blur_b), axis=2).astype(
        np.float64,
        copy=False,
    )
    max_mae = _max_overlap_window_mean(
        max_delta,
        window=window,
        stride=stride,
    )
    max_mean_square = _max_overlap_window_mean(
        max_delta * max_delta,
        window=window,
        stride=stride,
    )
    return max_mae, math.sqrt(max_mean_square)


def _images_perceptually_equivalent(image_a, image_b):
    """Compare two RGB images with documented noise-tolerant metrics.

    Global metrics use downsample + mild blur so ordinary recompression
    noise is accepted.  Local gates use the full bounded render:

    * raw local MAE (non-overlapping 16×16, NumPy) ≤
      :data:`RENDER_LOCAL_TILE_MAE_MAX`
    * blur-overlap changed fraction (BoxBlur each plane, max-channel >
      threshold, full 16×16 / stride 8 + edge anchors) ≤
      :data:`RENDER_LOCAL_BLUR_CHANGED_FRAC_MAX`
    * blur-overlap max-channel MAE / RMS energy (8×8 / stride 4 + edge
      anchors) ≤ their respective local energy caps

    Returns ``(ok, metrics)`` where *metrics* contains only numeric fields
    suitable for fail-closed log messages (never page content).
    """
    if image_a.size != image_b.size:
        # Ghostscript may alter the raster grid by a pixel; align to source.
        image_b = image_b.resize(image_a.size, Image.Resampling.BILINEAR)

    prepared_a = _downsample_blur(image_a)
    prepared_b = _downsample_blur(image_b)
    mae = _mean_absolute_error(prepared_a, prepared_b)
    changed_frac = _changed_pixel_fraction(prepared_a, prepared_b)
    tile_mae = _max_tile_mae(prepared_a, prepared_b)

    arr_a = np.asarray(image_a, dtype=np.uint8)
    arr_b = np.asarray(image_b, dtype=np.uint8)
    local_tile_mae = _max_local_mae_numpy(
        arr_a,
        arr_b,
        tile_size=RENDER_LOCAL_TILE_SIZE,
    )
    local_blur_overlap_changed_frac = _max_blur_overlap_changed_fraction(
        arr_a,
        arr_b,
        blur_radius=RENDER_LOCAL_BLUR_RADIUS,
        window=RENDER_LOCAL_TILE_SIZE,
        stride=RENDER_LOCAL_BLUR_STRIDE,
        threshold=RENDER_PIXEL_DIFF_THRESHOLD,
    )
    local_energy_mae, local_energy_rms = _max_blur_overlap_mae_energy(
        arr_a,
        arr_b,
    )
    metrics = {
        "mae": mae,
        "changed_frac": changed_frac,
        "tile_mae": tile_mae,
        "local_tile_mae": local_tile_mae,
        "local_blur_overlap_changed_frac": local_blur_overlap_changed_frac,
        "local_energy_mae": local_energy_mae,
        "local_energy_rms": local_energy_rms,
        # Compat alias for callers still reading the old metric key name.
        "local_tile_changed_frac": local_blur_overlap_changed_frac,
        "width": image_a.size[0],
        "height": image_a.size[1],
    }
    ok = (
        mae <= RENDER_MAE_MAX
        and changed_frac <= RENDER_CHANGED_FRAC_MAX
        and tile_mae <= RENDER_TILE_MAE_MAX
        and local_tile_mae <= RENDER_LOCAL_TILE_MAE_MAX
        and local_blur_overlap_changed_frac <= RENDER_LOCAL_BLUR_CHANGED_FRAC_MAX
        and local_energy_mae <= RENDER_LOCAL_ENERGY_MAE_MAX
        and local_energy_rms <= RENDER_LOCAL_ENERGY_RMS_MAX
    )
    return ok, metrics


def _rendered_pages_equivalent(original_page, candidate_page, page_label):
    """Fail closed when rendered RGB content diverges beyond noise tolerances.

    Error strings include page number and metrics only — never page text or
    image payloads.  Pages that would require effective DPI below
    :data:`RENDER_MIN_EFFECTIVE_DPI` under the pixel cap are refused without
    downscaled comparison.
    """
    original_image = None
    candidate_image = None
    try:
        width_pt, height_pt = _page_dimensions(original_page)
        scale, scale_metrics = _render_scale_for_page(width_pt, height_pt)
        if scale is None:
            return False, (
                f"page {page_label} exceeds compression visual-validation "
                f"budget (refuse optimization rather than dilute comparison): "
                f"width_pt={scale_metrics['width_pt']:.2f}, "
                f"height_pt={scale_metrics['height_pt']:.2f}, "
                f"effective_dpi={scale_metrics['effective_dpi']:.2f}, "
                f"min_effective_dpi={scale_metrics['min_effective_dpi']:.2f}, "
                f"max_pixels={int(scale_metrics['max_pixels'])}"
            ), dict(scale_metrics)
        original_image, _ = _render_page_rgb(original_page, scale=scale)
        candidate_image, _ = _render_page_rgb(candidate_page, scale=scale)
        ok, metrics = _images_perceptually_equivalent(original_image, candidate_image)
        if not ok:
            return False, (
                f"page {page_label} rendered content changed during optimization: "
                f"mae={metrics['mae']:.3f}, "
                f"changed_frac={metrics['changed_frac']:.4f}, "
                f"tile_mae={metrics['tile_mae']:.3f}, "
                f"local_tile_mae={metrics['local_tile_mae']:.3f}, "
                f"local_blur_overlap_changed_frac="
                f"{metrics['local_blur_overlap_changed_frac']:.4f}, "
                f"local_energy_mae={metrics['local_energy_mae']:.3f}, "
                f"local_energy_rms={metrics['local_energy_rms']:.3f}"
            ), metrics
        return True, "", metrics
    finally:
        if original_image is not None:
            original_image.close()
        if candidate_image is not None:
            candidate_image.close()


def _preflight_visual_budget(
    input_path: str,
) -> Tuple[Optional[OptimizationDisposition], str, Dict[str, Any]]:
    """Check every original page's render scale before temp/GS work.

    Returns:
      ``(None, "", metrics)`` when all pages stay at or above
      :data:`RENDER_MIN_EFFECTIVE_DPI` under the pixel cap.
      ``(SKIPPED_VISUAL_BUDGET, reason, metrics)`` when any page would
      dilute below the floor — caller must not create temp, invoke GS, or
      mutate the input.
      ``(FAILED_PRESERVED, reason, metrics)`` when the source cannot be
      opened/inspected (not a safe budget skip).
    """
    try:
        with fitz.open(input_path) as document:
            if document.is_encrypted or document.needs_pass:
                return (
                    OptimizationDisposition.FAILED_PRESERVED,
                    "source PDF is encrypted and cannot be validated",
                    {},
                )
            page_count = document.page_count
            if page_count <= 0:
                return (
                    OptimizationDisposition.FAILED_PRESERVED,
                    "source PDF must contain at least one page",
                    {},
                )
            for page_number in range(page_count):
                page = document.load_page(page_number)
                width_pt, height_pt = _page_dimensions(page)
                scale, scale_metrics = _render_scale_for_page(width_pt, height_pt)
                if scale is None:
                    page_label = page_number + 1
                    reason = (
                        f"page {page_label} exceeds compression visual-validation "
                        f"budget (refuse optimization rather than dilute comparison): "
                        f"width_pt={scale_metrics['width_pt']:.2f}, "
                        f"height_pt={scale_metrics['height_pt']:.2f}, "
                        f"effective_dpi={scale_metrics['effective_dpi']:.2f}, "
                        f"min_effective_dpi={scale_metrics['min_effective_dpi']:.2f}, "
                        f"max_pixels={int(scale_metrics['max_pixels'])}"
                    )
                    return (
                        OptimizationDisposition.SKIPPED_VISUAL_BUDGET,
                        reason,
                        dict(scale_metrics),
                    )
    except Exception as exc:
        return (
            OptimizationDisposition.FAILED_PRESERVED,
            f"visual-budget preflight failed: {exc}",
            {},
        )
    return None, "", {}


def _unique_temp_path(input_path):
    """Build a per-call temp path on the same directory/filesystem as *input_path*.

    Prefer ``tempfile.mkstemp`` (O_EXCL) so concurrent compressors cannot share a
    name.  If the parent directory is missing or unusable (unit tests with fake
    paths), fall back to a UUID-suffixed name in the same logical directory so
    callers still get collision-resistant ``-sOutputFile`` values.
    """
    directory = os.path.dirname(os.path.abspath(input_path)) or os.getcwd()
    prefix = f".{os.path.basename(input_path)}."
    try:
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp.pdf",
            prefix=prefix,
            dir=directory,
        )
        os.close(fd)
        return temp_path
    except OSError:
        return os.path.join(directory, f"{prefix}{uuid.uuid4().hex}.tmp.pdf")


def _cleanup_temp(temp_path):
    """Best-effort removal of a temp PDF; never raises."""
    if not temp_path:
        return
    try:
        if os.path.exists(temp_path):
            if os.name == "nt":
                # Windows refuses to delete read-only files; candidate may
                # already carry the preserved final mode (e.g. 0444).
                try:
                    mode = stat.S_IMODE(os.stat(temp_path).st_mode)
                    if not (mode & stat.S_IWRITE):
                        os.chmod(temp_path, mode | stat.S_IWRITE)
                except OSError:
                    pass
            os.remove(temp_path)
    except OSError:
        logger.warning("Failed to clean temp PDF: %s", temp_path, exc_info=True)


def _apply_mode(path, mode):
    """Apply regular-file *mode* bits to *path*.

    Returns True on success or when *mode* is None.  Returns False on OSError so
    callers can treat mode application failure as a pre-commit failure.
    """
    if mode is None:
        return True
    try:
        os.chmod(path, mode)
        return True
    except OSError:
        logger.error("Could not apply mode %s on %s", oct(mode), path, exc_info=True)
        return False


def _fsync_path(path):
    """fsync an existing regular file so content reaches durable storage.

    Prefer :func:`_apply_mode_and_fsync` when the final mode may be read-only:
    opening ``O_RDWR`` after ``chmod 0444`` fails on Windows.  This helper
    remains for callers that fsync before any read-only mode is applied.

    Uses ``os.open`` (not builtins.open) so durability is independent of
    Python file-object wrappers and test patches of ``open``.  On Windows,
    open read/write so ``FlushFileBuffers`` accepts the handle.
    """
    flags = os.O_RDWR if os.name == "nt" else os.O_RDONLY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _apply_mode_and_fsync(path, mode):
    """Apply *mode* and fsync content+mode before atomic replace.

    Sequence (cross-platform, including Windows read-only final modes):

    1. Open a writable candidate descriptor (``O_RDWR``).
    2. Apply the final mode while that descriptor remains open.
    3. ``fsync`` through the already-open descriptor (content + metadata).
    4. Close the descriptor.  Caller then ``os.replace``.

    Opening before ``chmod`` is required on Windows: applying ``0444`` first
    then reopening ``O_RDWR`` fails, so content would never be flushed.

    Returns True on success (or when *mode* is None and fsync succeeds).
    Returns False on any open/chmod/fsync failure so the caller keeps the
    original file (pre-replace).
    """
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        logger.error(
            "Could not open candidate for durable mode/fsync: %s",
            path,
            exc_info=True,
        )
        return False
    try:
        if mode is not None:
            try:
                # Prefer fchmod on the open fd when available (POSIX).
                fchmod = getattr(os, "fchmod", None)
                if fchmod is not None:
                    try:
                        fchmod(fd, mode)
                    except (NotImplementedError, OSError):
                        os.chmod(path, mode)
                else:
                    os.chmod(path, mode)
            except OSError:
                logger.error(
                    "Could not apply mode %s on %s",
                    oct(mode),
                    path,
                    exc_info=True,
                )
                return False
        try:
            os.fsync(fd)
        except OSError:
            logger.error(
                "Could not fsync candidate before commit: %s",
                path,
                exc_info=True,
            )
            return False
        return True
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _fsync_parent_dir(path):
    """fsync the parent directory on POSIX; no-op on Windows.

    Directory fsync makes the rename durable.  Windows does not support
    directory descriptors the same way, so this is intentionally a no-op there
    (Windows durable replace uses ``MoveFileExW`` write-through instead).

    Raises OSError when the directory open/fsync fails on POSIX so callers
    can refuse to claim durable success after replace.
    """
    if os.name == "nt":
        return
    directory = os.path.dirname(os.path.abspath(path)) or os.getcwd()
    flags = getattr(os, "O_RDONLY", 0)
    # O_DIRECTORY is preferred when available so we never open a non-dir.
    flags |= getattr(os, "O_DIRECTORY", 0)
    dir_fd = os.open(directory, flags)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


# MoveFileExW flags (winbase.h): replace existing + flush through to storage.
_MOVEFILE_REPLACE_EXISTING = 0x1
_MOVEFILE_WRITE_THROUGH = 0x8
_MOVEFILE_REPLACE_WRITE_THROUGH = (
    _MOVEFILE_REPLACE_EXISTING | _MOVEFILE_WRITE_THROUGH
)


def _windows_extended_path(path):
    """Return an extended-length Unicode path suitable for Win32 APIs."""
    absolute = os.path.abspath(path)
    if absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        # UNC: \\server\share\... -> \\?\UNC\server\share\...
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def _windows_destination_is_readonly(path):
    """Return True when *path* exists and lacks the user write bit."""
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        return False
    return not bool(mode & stat.S_IWRITE)


def _windows_movefile_ex(src, dst, *, flags=_MOVEFILE_REPLACE_WRITE_THROUGH):
    """Call ``MoveFileExW`` with *flags*; raise OSError on failure.

    Thin ctypes wrapper around the official Win32 API.  *src* and *dst* are
    converted to extended-length Unicode paths.  Does not chmod either path.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    move_file_ex_w = kernel32.MoveFileExW
    move_file_ex_w.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    move_file_ex_w.restype = wintypes.BOOL

    src_w = _windows_extended_path(src)
    dst_w = _windows_extended_path(dst)
    if not move_file_ex_w(src_w, dst_w, int(flags)):
        err = ctypes.get_last_error()
        raise ctypes.WinError(err)


def _durable_replace(src, dst):
    """Replace *dst* with *src* using platform durable-commit semantics.

    Preconditions (caller responsibility):
      * *src* is a same-volume temp candidate that has already been
        mode-preserved and fsynced (``_apply_mode_and_fsync``).
      * Success has not yet been claimed.

    Windows (native ``os.name == "nt"``):
      * If *dst* exists and is read-only, fail *before* any mutation with a
        clear unsupported durable-replace message.  Never clear the
        destination write-protect bit (avoids a crash window where the
        original is writable and directory fsync is a no-op).
      * Otherwise call ``MoveFileExW`` with
        ``MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH`` (0x1|0x8).
      * On failure leave *dst* untouched, do not claim success; caller cleans
        the temp.

    POSIX:
      * ``os.replace(src, dst)`` then mandatory parent-directory fsync.
      * Directory fsync failure raises OSError so the caller refuses to claim
        durable success (replacement may already exist).

    Returns None on success.  Raises OSError / RuntimeError on failure.
    """
    if os.name == "nt":
        if _windows_destination_is_readonly(dst):
            raise OSError(
                "unsupported durable replace on Windows for read-only "
                f"destination: {dst} (refusing to clear write-protect; "
                "original left unchanged)"
            )
        _windows_movefile_ex(
            src,
            dst,
            flags=_MOVEFILE_REPLACE_WRITE_THROUGH,
        )
        return

    os.replace(src, dst)
    try:
        _fsync_parent_dir(dst)
    except OSError as exc:
        # Replace already succeeded — original bytes cannot be restored.
        raise OSError(
            f"post-replace parent directory fsync failed for {dst} "
            f"({exc}); replacement may exist but durability is unproven "
            f"(not claiming success)"
        ) from exc


def _validate_pdf_structure(input_path, candidate_path, *, metrics_out=None):
    """Validate that *candidate_path* preserves readable structure and content.

    Ghostscript returning zero is not sufficient proof that its output is a
    usable, forensically equivalent PDF.  Before the original is replaced,
    require:

    * matching page counts (and at least one page)
    * every candidate page loads
    * per-page normalized extracted-text equivalence
    * per-word searchable-text geometry within TEXT_GEOMETRY_TOLERANCE_PT
    * per-page dimensions stable within PAGE_DIMENSION_TOLERANCE_PT
    * per-page rendered RGB equivalence under the documented noise tolerances
    """
    try:
        with fitz.open(input_path) as original, fitz.open(candidate_path) as candidate:
            if original.is_encrypted or original.needs_pass:
                return False, "source PDF is encrypted and cannot be validated"
            if candidate.is_encrypted or candidate.needs_pass:
                return False, "Ghostscript output is encrypted"

            original_page_count = original.page_count
            candidate_page_count = candidate.page_count
            if original_page_count <= 0 or candidate_page_count <= 0:
                return False, "source and Ghostscript output must contain at least one page"
            if candidate_page_count != original_page_count:
                return False, (
                    "page count changed during optimization: "
                    f"source={original_page_count}, output={candidate_page_count}"
                )

            # Process one page pair at a time (text/dimensions then render).
            for page_number in range(original_page_count):
                original_page = original.load_page(page_number)
                candidate_page = candidate.load_page(page_number)
                page_label = page_number + 1

                original_size = _page_dimensions(original_page)
                candidate_size = _page_dimensions(candidate_page)
                if not _dimensions_match(original_size, candidate_size):
                    original_width, original_height = original_size
                    candidate_width, candidate_height = candidate_size
                    return False, (
                        f"page {page_label} dimensions changed during optimization: "
                        f"source={original_width:.2f}x{original_height:.2f}, "
                        f"output={candidate_width:.2f}x{candidate_height:.2f} "
                        f"(tolerance={PAGE_DIMENSION_TOLERANCE_PT}pt)"
                    )

                original_text = _normalize_extracted_text(original_page.get_text())
                candidate_text = _normalize_extracted_text(candidate_page.get_text())
                if candidate_text != original_text:
                    if not candidate_text and original_text:
                        return False, (
                            f"page {page_label} searchable text removed during optimization"
                        )
                    return False, (
                        f"page {page_label} searchable text changed during optimization"
                    )

                geometry_ok, geometry_detail = _word_geometry_equivalent(
                    original_page,
                    candidate_page,
                )
                if metrics_out is not None and isinstance(geometry_detail, (int, float)):
                    metrics_out["text_geometry_max_delta_pt"] = max(
                        float(geometry_detail),
                        float(metrics_out.get("text_geometry_max_delta_pt", 0.0)),
                    )
                if not geometry_ok:
                    return False, f"page {page_label} {geometry_detail}"

                render_ok, render_error, render_metrics = _rendered_pages_equivalent(
                    original_page,
                    candidate_page,
                    page_label,
                )
                if metrics_out is not None:
                    for key, value in render_metrics.items():
                        if isinstance(value, (int, float)):
                            metrics_out[key] = max(
                                float(value),
                                float(metrics_out.get(key, value)),
                            )
                if not render_ok:
                    return False, render_error
    except Exception as exc:
        return False, f"PDF structure validation failed: {exc}"

    return True, ""


def optimize_pdf_detailed(input_path, quality=DEFAULT_QUALITY) -> OptimizeResult:
    """Optimize *input_path* in place; return a typed :class:`OptimizeResult`.

    Dispositions:
      * ``APPLIED`` — validated Ghostscript replacement committed.
      * ``SKIPPED_VISUAL_BUDGET`` — page(s) below min effective DPI under the
        pixel cap; no temp, no Ghostscript, input bytes unchanged.
      * ``SKIPPED_CANDIDATE_REJECTED`` — Ghostscript produced a candidate,
        but structural/visual/text equivalence rejected it before replacement;
        the original remains authoritative.
      * ``FAILED_PRESERVED`` — another pre-replace failure; original intact.
      * ``FAILED_INDETERMINATE`` — post-replace durability uncertainty;
        original may already be replaced; never treated as a safe skip.
    """
    if quality not in VALID_QUALITIES:
        logger.error("Invalid Ghostscript quality setting: %s", quality)
        return OptimizeResult(
            OptimizationDisposition.FAILED_PRESERVED,
            reason=f"invalid Ghostscript quality setting: {quality}",
        )

    # Preflight every original page *before* temp creation or Ghostscript.
    preflight_disp, preflight_reason, preflight_metrics = _preflight_visual_budget(
        input_path
    )
    if preflight_disp is OptimizationDisposition.SKIPPED_VISUAL_BUDGET:
        logger.info(
            "Skipping optimization (visual budget): %s — %s",
            os.path.basename(input_path),
            preflight_reason,
        )
        return OptimizeResult(
            OptimizationDisposition.SKIPPED_VISUAL_BUDGET,
            reason=preflight_reason,
            metrics=preflight_metrics,
        )
    if preflight_disp is OptimizationDisposition.FAILED_PRESERVED:
        logger.error("  > Failed preflight: %s", preflight_reason)
        return OptimizeResult(
            OptimizationDisposition.FAILED_PRESERVED,
            reason=preflight_reason,
            metrics=preflight_metrics,
        )

    # Basename suffix never controls success.  The standalone batch scanner
    # (main) may skip names ending in _optimized.pdf to avoid reprocessing
    # already-renamed deliverables; direct callers must still run Ghostscript
    # and only return APPLIED after validation + atomic replacement.

    file_name = os.path.basename(input_path)
    # Per-call unique temp on the same filesystem as the input so os.replace
    # is atomic and concurrent compressors never share .tmp.pdf / .bak names.
    temp_path = _unique_temp_path(input_path)
    original_mode = None

    logger.info(f"Optimizing: {file_name} [Quality: {quality}]...")

    # Ghostscript command
    cmd = [
        "gs",
        "-dSAFER",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        f"-dPDFSETTINGS={quality}",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={temp_path}",
        input_path
    ]

    try:
        try:
            original_mode = stat.S_IMODE(os.stat(input_path).st_mode)
        except OSError:
            original_mode = None

        # Run Ghostscript into the unique temp; original stays untouched.
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=GHOSTSCRIPT_TIMEOUT,
        )

        # Check sizes
        original_size = os.path.getsize(input_path)
        if original_size == 0:
            logger.warning("Skipping zero-byte file: %s", input_path)
            _cleanup_temp(temp_path)
            return OptimizeResult(
                OptimizationDisposition.FAILED_PRESERVED,
                reason="zero-byte source PDF",
            )

        if not os.path.exists(temp_path):
            logger.error("  > Failed: Output file not created.")
            return OptimizeResult(
                OptimizationDisposition.FAILED_PRESERVED,
                reason="Ghostscript output file not created",
            )

        new_size = os.path.getsize(temp_path)

        # Integrity check: verify output is a valid PDF before replacing (TD-011)
        if new_size < 100:
            logger.error(
                "  > Failed: Output too small (%s bytes), keeping original",
                new_size,
            )
            _cleanup_temp(temp_path)
            return OptimizeResult(
                OptimizationDisposition.FAILED_PRESERVED,
                reason=f"Ghostscript output too small ({new_size} bytes)",
            )

        with open(temp_path, "rb") as check_f:
            header = check_f.read(5)
        if header != b"%PDF-":
            logger.error("  > Failed: Output is not a valid PDF, keeping original")
            _cleanup_temp(temp_path)
            return OptimizeResult(
                OptimizationDisposition.FAILED_PRESERVED,
                reason="Ghostscript output is not a valid PDF",
            )

        visual_metrics = {}
        structure_valid, validation_error = _validate_pdf_structure(
            input_path,
            temp_path,
            metrics_out=visual_metrics,
        )
        if not structure_valid:
            logger.error(
                "  > Failed: %s, keeping original",
                validation_error,
            )
            _cleanup_temp(temp_path)
            # The optimizer candidate failed equivalence and was discarded
            # before replacement. Distinguish this measured safe preservation
            # from source/preflight, Ghostscript, and durability failures so
            # callers can retain the original without failing an otherwise
            # complete OCR job.
            return OptimizeResult(
                OptimizationDisposition.SKIPPED_CANDIDATE_REJECTED,
                reason=validation_error,
                metrics={
                    f"visual_{key}": value
                    for key, value in visual_metrics.items()
                },
            )

        reduction = (1 - (new_size / original_size)) * 100

        # Durable mode-preserving commit:
        # 1) open writable candidate FD, apply original mode, fsync via that FD
        #    (mode+content flushed before replace; works with final 0444 modes)
        # 2) platform durable replace via _durable_replace:
        #      Windows: MoveFileExW REPLACE_EXISTING|WRITE_THROUGH; refuse
        #               read-only destinations without clearing write-protect
        #      POSIX:   os.replace then mandatory parent-directory fsync
        # Pre-replace failures keep the original.  Post-replace POSIX
        # directory fsync failure cannot restore the old bytes but MUST NOT
        # claim durable success (final-hash/custody must not treat this as Done).
        if not _apply_mode_and_fsync(temp_path, original_mode):
            logger.error(
                "  > Failed: could not preserve mode and fsync candidate before "
                "commit, keeping original"
            )
            _cleanup_temp(temp_path)
            return OptimizeResult(
                OptimizationDisposition.FAILED_PRESERVED,
                reason="could not preserve mode and fsync candidate before commit",
            )

        try:
            _durable_replace(temp_path, input_path)
        except OSError as exc:
            message = str(exc)
            if "durability is unproven" in message:
                # POSIX: replace already succeeded; do not delete dst/temp.
                temp_path = None
                logger.error("  > Failed: %s", message)
                return OptimizeResult(
                    OptimizationDisposition.FAILED_INDETERMINATE,
                    reason=message,
                )
            # Pre-replace failure (including Windows read-only refusal):
            # original remains; clean candidate temp.
            logger.error(
                "  > Failed: durable replace failed (%s), keeping original",
                exc,
            )
            _cleanup_temp(temp_path)
            return OptimizeResult(
                OptimizationDisposition.FAILED_PRESERVED,
                reason=f"durable replace failed: {exc}",
            )
        temp_path = None  # ownership transferred; do not delete

        logger.info(
            f"  > Done! Size: {original_size/1024/1024:.2f}MB -> "
            f"{new_size/1024/1024:.2f}MB (-{reduction:.1f}%)"
        )
        return OptimizeResult(
            OptimizationDisposition.APPLIED,
            reason="",
            metrics={
                "original_size": float(original_size),
                "new_size": float(new_size),
                "reduction_pct": float(reduction),
                **{
                    f"visual_{key}": value
                    for key, value in visual_metrics.items()
                },
            },
        )

    except subprocess.TimeoutExpired:
        logger.warning(
            "Ghostscript timed out after %ds for %s", GHOSTSCRIPT_TIMEOUT, input_path
        )
        _cleanup_temp(temp_path)
        return OptimizeResult(
            OptimizationDisposition.FAILED_PRESERVED,
            reason=f"Ghostscript timed out after {GHOSTSCRIPT_TIMEOUT}s",
        )
    except subprocess.CalledProcessError as e:
        stderr = ""
        if e.stderr:
            try:
                stderr = e.stderr.decode()
            except Exception:
                stderr = str(e.stderr)
        logger.error("  > Failed to optimize %s: %s", file_name, stderr)
        _cleanup_temp(temp_path)
        return OptimizeResult(
            OptimizationDisposition.FAILED_PRESERVED,
            reason=f"Ghostscript failed: {stderr or e}",
        )
    except Exception as e:
        logger.error(f"  > Error: {e}")
        _cleanup_temp(temp_path)
        return OptimizeResult(
            OptimizationDisposition.FAILED_PRESERVED,
            reason=str(e),
        )


def optimize_pdf(input_path, quality=DEFAULT_QUALITY) -> bool:
    """Public bool API: True only when a verified replacement was applied."""
    return optimize_pdf_detailed(input_path, quality=quality).applied

def main():
    parser = argparse.ArgumentParser(description="Optimize PDFs using Ghostscript")
    parser.add_argument("--quality", default=DEFAULT_QUALITY,
                        choices=sorted(VALID_QUALITIES),
                        help="Ghostscript PDF settings")
    args = parser.parse_args()

    logger.info(f"Starting PDF Optimization Scan (Target Quality: {args.quality})")
    
    pdf_count = 0
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            # Standalone batch scanner may skip already-suffixed names; direct
            # optimize_pdf() never uses the basename as a success signal.
            if file.lower().endswith(".pdf") and not file.lower().endswith("_optimized.pdf"):
                full_path = os.path.join(root, file)
                optimize_pdf(full_path, args.quality)
                pdf_count += 1

    logger.info(f"Scan complete. Processed {pdf_count} potential files.")

if __name__ == "__main__":
    main()
