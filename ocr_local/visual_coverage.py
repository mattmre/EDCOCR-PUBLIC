"""Bounded raster evidence for OCR completeness; never authorizes skipping OCR.

Coordinates in this module are pixels in the supplied original image. Candidate
boxes may instead declare their own raster size and are scaled into that space.
The detector is intentionally conservative: ambiguous graphics, photos, and
resource limits produce unknown coverage, which callers must route to review.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - optional in stripped deployments
    cv2 = None

Box = tuple[float, float, float, float]
RecognizedSpan = tuple[str, Sequence[float] | Sequence[Sequence[float]], float | None]


@dataclass(frozen=True)
class CoverageLimits:
    max_input_pixels: int = 35_000_000
    max_analysis_pixels: int = 4_000_000
    max_components: int = 12_000
    max_regions: int = 256
    max_seconds: float = 1.0


@dataclass(frozen=True)
class RasterTextEvidence:
    status: str  # supported | unknown
    reason: str
    image_size: tuple[int, int]
    regions: tuple[Box, ...]
    component_boxes: tuple[Box, ...]
    component_count: int


@dataclass(frozen=True)
class CoverageAssessment:
    status: str  # measured | unknown
    coverage: float | None
    region_coverage: float | None
    component_coverage: float | None
    character_support: float | None
    missing_regions: tuple[Box, ...]
    needs_retry: bool
    needs_review: bool
    reason: str


def _unknown(size: tuple[int, int], reason: str) -> RasterTextEvidence:
    return RasterTextEvidence("unknown", reason, size, (), (), 0)


def _white_composited_rgb(image: Image.Image | np.ndarray, target: tuple[int, int]) -> np.ndarray:
    if isinstance(image, Image.Image):
        if image.size != target:
            image = image.resize(target, Image.Resampling.BOX)
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.alpha_composite(rgba)
        return np.asarray(white.convert("RGB"))
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4) or array.dtype != np.uint8:
        raise ValueError("raster must be a PIL image or uint8 RGB/RGBA array")
    if (array.shape[1], array.shape[0]) != target:
        array = cv2.resize(array, target, interpolation=cv2.INTER_AREA)
    if array.shape[2] == 3:
        return array
    alpha = array[:, :, 3:4].astype(np.float32) / 255.0
    return np.rint(array[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)


def assess_raster_text_regions(
    image: Image.Image | np.ndarray, *, limits: CoverageLimits = CoverageLimits()
) -> RasterTextEvidence:
    """Measure text-like ink without reading any OCR result or old PDF layer."""
    if cv2 is None:
        size = image.size if isinstance(image, Image.Image) else (int(image.shape[1]), int(image.shape[0]))
        return _unknown(size, "opencv_unavailable")
    start = time.monotonic()
    width, height = image.size if isinstance(image, Image.Image) else (int(image.shape[1]), int(image.shape[0]))
    size = (width, height)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in (limits.max_input_pixels, limits.max_analysis_pixels, limits.max_components, limits.max_regions, limits.max_seconds)):
        return _unknown(size, "invalid_limits")
    if width <= 0 or height <= 0 or width * height > limits.max_input_pixels:
        return _unknown(size, "input_pixel_limit")
    scale = min(1.0, math.sqrt(limits.max_analysis_pixels / (width * height)))
    rgb = _white_composited_rgb(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if float(np.mean((gray > 35) & (gray < 220))) > 0.45:
        return _unknown(size, "continuous_tone_or_photographic_page")
    # Adaptive threshold tolerates uneven scans but rules, seals and photos are
    # excluded only when their shape/density is clearly unlike printed glyphs.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    ink = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    ink_fraction = float(np.count_nonzero(ink) / ink.size)
    if ink_fraction < 0.00005:
        return _unknown(size, "no_text_like_ink")
    if ink_fraction > 0.28:
        return _unknown(size, "dense_or_photographic_page")
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if count - 1 > limits.max_components:
        return _unknown(size, "component_limit")
    sx = width / gray.shape[1]
    sy = height / gray.shape[0]
    components: list[Box] = []
    heights: list[int] = []
    for x, y, w, h, area in stats[1:]:
        if area < 2 or h < 2 or w < 1 or h > gray.shape[0] * 0.08 or w > gray.shape[1] * 0.2:
            continue
        if w / max(h, 1) > 9 or h / max(w, 1) > 15:
            continue
        components.append((float(x * sx), float(y * sy), float((x + w) * sx), float((y + h) * sy)))
        heights.append(int(h))
    if time.monotonic() - start > limits.max_seconds:
        return _unknown(size, "analysis_time_limit")
    if len(components) < 8:
        return _unknown(size, "too_few_text_like_components")
    median_height = float(np.median(heights))
    if median_height < 2 or float(np.percentile(heights, 90)) > median_height * 4:
        return _unknown(size, "ambiguous_component_shapes")
    # Group independently observed glyph-like components into horizontal
    # bands. The groups are evidence regions, not OCR-reported line boxes.
    ordered = sorted(components, key=lambda box: (box[1] + box[3]) / 2)
    bands: list[list[Box]] = []
    band_centers: list[float] = []
    band_counts: list[int] = []
    for component in ordered:
        center = (component[1] + component[3]) / 2
        if bands and abs(center - band_centers[-1]) <= median_height * sy * 0.9:
            bands[-1].append(component)
            band_centers[-1] = (band_centers[-1] * band_counts[-1] + center) / (band_counts[-1] + 1)
            band_counts[-1] += 1
        else:
            bands.append([component])
            band_centers.append(center)
            band_counts.append(1)
        if len(bands) > limits.max_regions:
            return _unknown(size, "region_limit")
        if time.monotonic() - start > limits.max_seconds:
            return _unknown(size, "analysis_time_limit")
    bands = [band for band in bands if len(band) >= 3]
    if not bands:
        return _unknown(size, "no_supported_text_regions")
    regions = tuple((min(box[0] for box in band), min(box[1] for box in band), max(box[2] for box in band), max(box[3] for box in band)) for band in bands)
    supported = tuple(box for band in bands for box in band)
    return RasterTextEvidence("supported", "text_like_components", size, regions, supported, len(supported))


def _box(value: Sequence[float] | Sequence[Sequence[float]]) -> Box:
    if len(value) == 4 and all(isinstance(number, (int, float, np.integer, np.floating)) for number in value):
        x0, y0, x1, y1 = value  # type: ignore[misc]
        return float(x0), float(y0), float(x1), float(y1)
    points = value  # type: ignore[assignment]
    if len(points) != 4 or any(len(point) != 2 for point in points):
        raise ValueError("candidate box must be xyxy or four-corner quad")
    return min(float(point[0]) for point in points), min(float(point[1]) for point in points), max(float(point[0]) for point in points), max(float(point[1]) for point in points)


def assess_candidate_coverage(
    evidence: RasterTextEvidence,
    spans: Sequence[RecognizedSpan],
    *,
    candidate_raster_size: tuple[int, int] | None = None,
) -> CoverageAssessment:
    """Estimate independent coverage; low/unknown means retry or review, never skip."""
    if evidence.status != "supported":
        return CoverageAssessment("unknown", None, None, None, None, (), True, True, evidence.reason)
    if len(spans) > 2048:
        return CoverageAssessment("unknown", None, None, None, None, (), True, True, "candidate_span_limit")
    if sum(len(str(span[0])) for span in spans) > 128_000:
        return CoverageAssessment("unknown", None, None, None, None, (), True, True, "candidate_character_limit")
    started = time.monotonic()
    source_w, source_h = evidence.image_size
    candidate_w, candidate_h = candidate_raster_size or evidence.image_size
    if candidate_w <= 0 or candidate_h <= 0:
        return CoverageAssessment("unknown", None, None, None, None, (), True, True, "invalid_candidate_raster_size")
    xscale, yscale = source_w / candidate_w, source_h / candidate_h
    boxes = []
    chars = 0
    try:
        for text, geometry, _confidence in spans:
            if not text.strip():
                continue
            box = _box(geometry)
            if not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
                continue
            boxes.append((box[0] * xscale, box[1] * yscale, box[2] * xscale, box[3] * yscale))
            chars += sum(not char.isspace() for char in text)
    except (TypeError, ValueError, IndexError):
        return CoverageAssessment("unknown", None, None, None, None, (), True, True, "invalid_candidate_boxes")
    components = np.asarray(evidence.component_boxes, dtype=np.float32)
    centers = (components[:, :2] + components[:, 2:]) / 2
    hits = np.zeros(len(components), dtype=bool)
    for index, box in enumerate(boxes):
        hits |= (centers[:, 0] >= box[0]) & (centers[:, 0] <= box[2]) & (centers[:, 1] >= box[1]) & (centers[:, 1] <= box[3])
        if index % 32 == 0 and time.monotonic() - started > 1.0:
            return CoverageAssessment("unknown", None, None, None, None, (), True, True, "candidate_analysis_time_limit")
    component_fraction = float(np.mean(hits))
    region_hits = []
    missing = []
    for index, region in enumerate(evidence.regions):
        member_mask = (centers[:, 0] >= region[0]) & (centers[:, 0] <= region[2]) & (centers[:, 1] >= region[1]) & (centers[:, 1] <= region[3])
        found = bool(np.any(member_mask)) and float(np.mean(hits[member_mask])) >= 0.5
        region_hits.append(found)
        if not found:
            missing.append(region)
        if index % 32 == 0 and time.monotonic() - started > 1.0:
            return CoverageAssessment("unknown", None, None, None, None, (), True, True, "candidate_analysis_time_limit")
    region_fraction = sum(region_hits) / len(region_hits)
    # A giant OCR box around a full page with one recognized word must not
    # masquerade as complete. This is a conservative plausibility floor, not
    # an estimated exact word count for any script.
    character_support = min(1.0, chars / max(1.0, evidence.component_count * 0.25))
    coverage = min(component_fraction, region_fraction, character_support)
    # Even one missing paragraph is material. A high component/character score
    # is only supporting evidence, never proof that recognition is complete.
    low = bool(missing) or component_fraction < 0.9 or character_support < 0.9
    return CoverageAssessment("measured", round(coverage, 6), round(region_fraction, 6), round(component_fraction, 6), round(character_support, 6), tuple(missing), low, low, "insufficient_raster_support" if low else "raster_regions_supported")
