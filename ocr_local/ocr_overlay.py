"""Pure OCR overlay normalization for per-word searchable text placement."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite
from threading import local
from typing import Any, Mapping

import fitz

from ocr_local.language_routing import CandidateDecision
from ocr_local.pdf_geometry import PageGeometry, RasterTransform

_font_metrics_local = local()


@dataclass(frozen=True)
class PageReadingDirectionEvidence:
    orientation: int | None
    reference_text: str
    score: float
    original_score: float
    corrected_score: float
    reason: str
    opposite_score: float | None = None


def probe_page_reading_direction(
    image,
    source_rotation: int,
    *,
    traineddata: str,
    script: str,
    timeout_seconds: float = 5.0,
    max_pixels: int = 2_000_000,
) -> PageReadingDirectionEvidence:
    """Qualify visible glyph direction independently of PDF /Rotate metadata.

    This CPU-only probe compares the rendered source with both possible
    quarter turns (or one half turn). Neither a failed nor tied result
    authorizes a text direction. The caller retains the original source raster.
    """
    if (
        source_rotation not in (90, 180, 270)
        or not isinstance(traineddata, str) or not traineddata.strip()
        or not 0 < timeout_seconds <= 10
        or not 1 <= max_pixels <= 8_000_000
    ):
        raise ValueError("half-turn probe limits or traineddata are invalid")
    import pytesseract

    width, height = image.size
    if width <= 0 or height <= 0:
        return PageReadingDirectionEvidence(None, "", 0.0, 0.0, 0.0, "invalid_raster")
    if script.casefold() not in {"latin", "greek", "cyrillic"}:
        return PageReadingDirectionEvidence(None, "", 0.0, 0.0, 0.0, "script_not_supported")
    if width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    scores: list[float] = []
    readings: list[str] = []
    angles = (
        (0, 180) if source_rotation == 180
        else (0, source_rotation, 360 - source_rotation)
    )
    for angle in angles:
        candidate = image if angle == 0 else image.rotate(angle, expand=True)
        try:
            data = pytesseract.image_to_data(
                candidate, lang=traineddata, config="--psm 11",
                output_type=pytesseract.Output.DICT, timeout=timeout_seconds,
            )
        except (OSError, RuntimeError, TypeError, ValueError, AttributeError):
            return PageReadingDirectionEvidence(None, "", 0.0, 0.0, 0.0, "probe_unavailable")
        if not isinstance(data, dict):
            return PageReadingDirectionEvidence(None, "", 0.0, 0.0, 0.0, "malformed_probe")
        weighted_sum = 0.0
        letters = 0
        words = []
        for raw_text, raw_confidence in zip(data.get("text", ()), data.get("conf", ())):
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                continue
            if not isfinite(confidence) or confidence < 0 or confidence > 100:
                continue
            word = str(raw_text or "")
            count = sum(char.isalpha() for char in word)
            if count:
                words.append(word)
                weighted_sum += confidence * count
                letters += count
        reading = " ".join(words)
        readings.append(reading)
        if letters < 8:
            scores.append(0.0)
        else:
            from ocr_local.language_routing import text_supports_script

            scores.append(
                weighted_sum / (100 * letters)
                if text_supports_script(reading, script) else 0.0
            )
    original, corrected = scores[:2]
    opposite = scores[2] if len(scores) == 3 else None
    if max(scores) < 0.75:
        return PageReadingDirectionEvidence(None, "", max(scores), original, corrected, "weak_reading", opposite)
    ranked = sorted(scores, reverse=True)
    if ranked[0] - ranked[1] < 0.20:
        return PageReadingDirectionEvidence(None, "", max(scores), original, corrected, "ambiguous_direction", opposite)
    winner = scores.index(ranked[0])
    return PageReadingDirectionEvidence(
        angles[winner],
        readings[winner], scores[winner], original, corrected,
        "qualified_raster_direction", opposite,
    )


def probe_half_turn_direction(
    image,
    *,
    traineddata: str,
    script: str,
    timeout_seconds: float = 5.0,
    max_pixels: int = 2_000_000,
) -> PageReadingDirectionEvidence:
    """Compatibility form of the shared page direction probe for /Rotate 180."""
    return probe_page_reading_direction(
        image, 180, traineddata=traineddata, script=script,
        timeout_seconds=timeout_seconds, max_pixels=max_pixels,
    )


def _font_metrics(text: str, fontname: str, fontfile: str | None) -> tuple[float, float, float]:
    """Bounded per-thread font reuse; PyMuPDF font objects are not shared."""
    fonts = getattr(_font_metrics_local, "fonts", None)
    widths = getattr(_font_metrics_local, "widths", None)
    if fonts is None:
        fonts = _font_metrics_local.fonts = OrderedDict()
        widths = _font_metrics_local.widths = OrderedDict()
    key = (fontname, fontfile)
    font = fonts.get(key)
    if font is None:
        font = fitz.Font(fontfile=fontfile) if fontfile else fitz.Font(fontname=fontname)
        fonts[key] = font
        if len(fonts) > 8:
            fonts.popitem(last=False)
    else:
        fonts.move_to_end(key)
    width_key = (key, text)
    unit_width = widths.get(width_key)
    if unit_width is None:
        unit_width = font.text_length(text, fontsize=1.0)
        widths[width_key] = unit_width
        if len(widths) > 128:
            widths.popitem(last=False)
    else:
        widths.move_to_end(width_key)
    return unit_width, float(font.ascender), float(font.descender)


def tesseract_data_to_lines_with_review(
    data: Mapping[str, Any], *, transform: RasterTransform | None = None
) -> tuple[list[tuple[str, list[float], float]], list[dict[str, Any]]]:
    """Return valid Tesseract words in original image_to_data row order.

    Each word retains its own geometry.  Confidence is normalized to [0, 1].
    Bad/empty rows are discarded without affecting the position of later rows.
    A transform maps OCR raster pixels into output PDF points when supplied.
    """
    lines = []
    issues = []
    texts = data.get("text") or []
    for index, raw_text in enumerate(texts):
        text = str(raw_text or "").strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
            if confidence < 0:
                continue  # Tesseract layout/nonword row.
            if not isfinite(confidence):
                raise ValueError("non-finite word confidence")
            left = float(data["left"][index])
            top = float(data["top"][index])
            width = float(data["width"][index])
            height = float(data["height"][index])
            if not all(isfinite(value) for value in (left, top, width, height)):
                raise ValueError("non-finite word geometry")
            if confidence > 100 or width <= 0 or height <= 0:
                raise ValueError("invalid word confidence or extent")
            box = [left, top, left + width, top + height]
            if transform is not None:
                box = transform.map_box(box)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            issues.append({"index": index, "text": text, "reason": str(exc)})
            continue
        lines.append((text, box, confidence / 100.0))
    return lines, issues


def tesseract_data_to_lines(
    data: Mapping[str, Any], *, transform: RasterTransform | None = None
) -> list[tuple[str, list[float], float]]:
    """Return placeable words; use review-aware variant to retain rejected spans."""
    return tesseract_data_to_lines_with_review(data, transform=transform)[0]


def bounded_source_raster_size(image_size, max_pixels: int = 2_000_000) -> tuple[int, int]:
    """Return the exact integer raster size used by a bounded source probe."""
    width, height = image_size
    if width <= 0 or height <= 0 or not 1 <= max_pixels <= 8_000_000:
        raise ValueError("invalid source raster limits")
    if width * height <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / (width * height))
    return max(1, int(width * scale)), max(1, int(height * scale))


def read_source_tesseract_words(
    image, *, traineddata: str, rotation: int = 0,
    timeout_seconds: float = 5.0, max_pixels: int = 2_000_000,
) -> tuple[list[tuple[str, Any, float]], list[dict[str, Any]]]:
    """One bounded CPU reading, mapped back to original source-image pixels."""
    if rotation not in (0, 90, 180, 270) or not traineddata.strip():
        raise ValueError("invalid source OCR route")
    if not 0 < timeout_seconds <= 10 or not 1 <= max_pixels <= 8_000_000:
        raise ValueError("invalid source OCR limits")
    import pytesseract

    width, height = image.size
    probe_size = bounded_source_raster_size(image.size, max_pixels)
    raster = image
    if probe_size != image.size:
        raster = image.resize(probe_size)
    ocr_copy = raster if rotation == 0 else raster.rotate(rotation, expand=True)
    data = pytesseract.image_to_data(
        ocr_copy, lang=traineddata, config="--psm 3",
        output_type=pytesseract.Output.DICT, timeout=timeout_seconds,
    )
    words, issues = tesseract_data_to_lines_with_review(data)
    transform = PageGeometry.from_image_pixels(width, height, dpi=72).for_raster(
        *raster.size
    )
    if rotation:
        mapped, mapping_issues = map_rotated_words_to_source(words, transform, rotation)
    else:
        mapped, mapping_issues = transform.map_lines_with_review(words)
    return mapped, issues + mapping_issues


def _revision_box(box) -> tuple[float, float, float, float] | None:
    try:
        if len(box) != 4:
            return None
        if hasattr(box[0], "__iter__"):
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            values = (min(xs), min(ys), max(xs), max(ys))
        else:
            values = tuple(float(value) for value in box)
        if not all(isfinite(value) for value in values):
            return None
        x0, y0, x1, y1 = values
        return values if x1 > x0 and y1 > y0 else None
    except (TypeError, ValueError, IndexError):
        return None


def _revision_coverage(value) -> float | None:
    if isinstance(value, Mapping):
        if value.get("status") != "measured":
            return None
        value = value.get("coverage")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) and 0 <= number <= 1 else None


def compare_source_grounded_text_revision(
    prior_lines, candidate_lines, *, prior_coverage, candidate_coverage,
    candidate_confidence_floor: float = 0.85, script_supported: bool,
) -> CandidateDecision:
    """Accept a full alternate reading only for grounded punctuation/spacing.

    Cross-engine confidence is never ranked. Every candidate word must have
    its own valid source box, and both readings must cover the same locations.
    The caller retains both raw readings and records review on selection.
    """
    def reject(reason):
        return CandidateDecision(False, reason, True)

    if not prior_lines or not candidate_lines or script_supported is not True:
        return reject("source_revision_missing_or_unsupported")
    try:
        prior_text = " ".join(str(line[0]) for line in prior_lines)
        candidate_text = " ".join(str(line[0]) for line in candidate_lines)
        def normalize(value):
            return unicodedata.normalize("NFKC", value).casefold()
        token_pattern = r"[^\W_]+"
        normalized_prior = normalize(prior_text)
        normalized_candidate = normalize(candidate_text)
        prior_matches = list(re.finditer(token_pattern, normalized_prior))
        candidate_matches = list(re.finditer(token_pattern, normalized_candidate))
        prior_tokens = tuple(match.group() for match in prior_matches)
        candidate_tokens = tuple(match.group() for match in candidate_matches)
        if not prior_tokens or prior_tokens != candidate_tokens:
            return reject("source_revision_alphanumeric_disagreement")
        if re.sub(r"\s+", " ", normalized_prior).strip() == re.sub(
            r"\s+", " ", normalized_candidate
        ).strip():
            return reject("source_revision_no_text_change")
        confidences = [float(line[2]) for line in candidate_lines]
        if not confidences or any(
            not isfinite(value) or value < 0 or value > 1
            for value in confidences
        ):
            return reject("source_revision_weak_candidate_word")
        if (
            not isfinite(candidate_confidence_floor)
            or not 0 <= candidate_confidence_floor <= 1
        ):
            return reject("source_revision_invalid_confidence_floor")
        # A full Tesseract reading can contain a weak *unchanged* word. It is
        # not evidence against a separate, strongly read punctuation change.
        # The whole candidate still needs a credible weighted confidence, and
        # every word touching the changed separator needs its own support.
        candidate_word_for_token = []
        weighted_total = 0.0
        total_weight = 0
        for word_index, (word, _box, confidence) in enumerate(candidate_lines):
            token_count = len(re.findall(token_pattern, normalize(str(word))))
            candidate_word_for_token.extend([word_index] * token_count)
            weight = max(1, sum(char.isalnum() for char in normalize(str(word))))
            weighted_total += float(confidence) * weight
            total_weight += weight
            if token_count == 0 and confidence < candidate_confidence_floor:
                return reject("source_revision_weak_changed_word")
        if (
            len(candidate_word_for_token) != len(candidate_tokens)
            or weighted_total / total_weight < candidate_confidence_floor
        ):
            return reject("source_revision_weak_candidate_reading")

        def separators(value, matches):
            return [value[:matches[0].start()]] + [
                value[first.end():second.start()]
                for first, second in zip(matches, matches[1:])
            ] + [value[matches[-1].end():]]

        prior_separators = separators(normalized_prior, prior_matches)
        candidate_separators = separators(normalized_candidate, candidate_matches)
        touched_tokens = set()
        for index, (before, after) in enumerate(zip(
            prior_separators, candidate_separators
        )):
            if re.sub(r"\s+", " ", before) == re.sub(r"\s+", " ", after):
                continue
            if index > 0:
                touched_tokens.add(index - 1)
            if index < len(candidate_tokens):
                touched_tokens.add(index)
        if not touched_tokens or any(
            confidences[candidate_word_for_token[index]] < candidate_confidence_floor
            for index in touched_tokens
        ):
            return reject("source_revision_weak_changed_word")
        prior_boxes = [_revision_box(line[1]) for line in prior_lines]
        candidate_boxes = [_revision_box(line[1]) for line in candidate_lines]
        if any(box is None for box in prior_boxes + candidate_boxes):
            return reject("source_revision_unplaced_span")
        old_coverage = _revision_coverage(prior_coverage)
        new_coverage = _revision_coverage(candidate_coverage)
        if old_coverage is None or new_coverage is None or new_coverage + 1e-9 < old_coverage:
            return reject("source_revision_coverage_unproven")
        def overlaps(first, second):
            x0 = max(first[0], second[0])
            y0 = max(first[1], second[1])
            x1 = min(first[2], second[2])
            y1 = min(first[3], second[3])
            intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            area = min(
                (first[2] - first[0]) * (first[3] - first[1]),
                (second[2] - second[0]) * (second[3] - second[1]),
            )
            return intersection >= 0.25 * area
        if not all(any(overlaps(box, other) for other in prior_boxes)
                   for box in candidate_boxes) or not all(
                       any(overlaps(box, other) for other in candidate_boxes)
                       for box in prior_boxes
                   ):
            return reject("source_revision_geometry_disagreement")
    except (IndexError, TypeError, ValueError):
        return reject("source_revision_malformed_reading")
    return CandidateDecision(True, "source_grounded_punctuation_or_spacing_revision", True)


def map_rotated_words_to_source(
    lines: list[tuple[str, list[float], float]],
    transform: RasterTransform,
    rotation: int,
) -> tuple[list[tuple[str, list[list[float]], float]], list[dict[str, Any]]]:
    """Place words from a right-angle OCR copy over the original raster.

    The PDF background stays in source orientation. A quad also preserves the
    text advance direction, so hidden search text follows rotated glyphs.
    """
    if rotation not in (90, 180, 270):
        raise ValueError("OCR rotation must be 90, 180, or 270 degrees")
    width, height = transform.width_px, transform.height_px
    mapped = []
    issues = []
    for index, (text, box, confidence) in enumerate(lines):
        try:
            x0, y0, x1, y1 = box
            corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            if rotation == 90:
                source_corners = tuple((width - y, x) for x, y in corners)
            elif rotation == 180:
                source_corners = tuple((width - x, height - y) for x, y in corners)
            else:
                source_corners = tuple((y, height - x) for x, y in corners)
            quad = transform.map_quad(source_corners)
        except (TypeError, ValueError) as exc:
            issues.append({"index": index, "text": text, "reason": str(exc)})
            continue
        mapped.append((text, quad, confidence))
    return mapped, issues


def map_half_turn_words_to_source(
    lines: list[tuple[str, list[float], float]], transform: RasterTransform
) -> tuple[list[tuple[str, list[list[float]], float]], list[dict[str, Any]]]:
    """Compatibility name for the half-turn OCR path."""
    return map_rotated_words_to_source(lines, transform, 180)


def map_rotated_detector_lines_to_source(
    lines: list[tuple[str, Any, float]],
    transform: RasterTransform,
    rotation: int,
) -> tuple[list[tuple[str, list[list[float]], float]], list[tuple[str, list[list[float]], float]], list[dict[str, Any]]]:
    """Inverse-map Paddle quads from an OCR-only right-angle copy.

    Return both original-raster pixel quads and PDF-point quads. Detector
    reading order is preserved by rotating all four vertices, never replacing
    the source visual background with the OCR copy.
    """
    if rotation not in (90, 180, 270):
        raise ValueError("OCR rotation must be 90, 180, or 270 degrees")
    width, height = transform.width_px, transform.height_px
    pixel_lines = []
    pdf_lines = []
    issues = []
    for index, (text, box, confidence) in enumerate(lines):
        try:
            if len(box) != 4:
                raise ValueError("detector box must have four corners or extents")
            if hasattr(box[0], "__iter__"):
                corners = box
            else:
                x0, y0, x1, y1 = box
                corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            source_corners = []
            for point in corners:
                x, y = float(point[0]), float(point[1])
                if rotation == 90:
                    source_corners.append([width - y, x])
                elif rotation == 180:
                    source_corners.append([width - x, height - y])
                else:
                    source_corners.append([y, height - x])
            pdf_quad = transform.map_quad(source_corners)
        except (TypeError, ValueError, IndexError) as exc:
            issues.append({"index": index, "text": text, "reason": str(exc)})
            continue
        pixel_lines.append((text, source_corners, confidence))
        pdf_lines.append((text, pdf_quad, confidence))
    return pixel_lines, pdf_lines, issues


def map_half_turn_detector_lines_to_source(
    lines: list[tuple[str, Any, float]], transform: RasterTransform
) -> tuple[list[tuple[str, list[list[float]], float]], list[tuple[str, list[list[float]], float]], list[dict[str, Any]]]:
    """Compatibility form of the right-angle detector mapper for 180 degrees."""
    return map_rotated_detector_lines_to_source(lines, transform, 180)


def insert_hidden_text_in_box(
    page: Any,
    text: str,
    box: Any,
    *,
    fontname: str,
    fontfile: str | None = None,
    source_rotation: int = 0,
) -> bool:
    """Fit invisible text inside a grounded PDF-point OCR rectangle.

    Search boxes use the same font's ascender/descender and glyph advance as
    insertion.  An empty, malformed, or impossibly small detector box fails
    without inventing a page anchor or forced minimum font size.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    # Base-14 fonts use a limited PDF encoding even when fitz.Font.has_glyph()
    # reports a Unicode outline.  Without an embedded font, non-ASCII insert
    # can silently turn Greek/Cyrillic/CJK into middle dots in extracted text.
    if fontfile is None and any(ord(char) > 127 for char in text):
        return False
    try:
        points = list(box)
        if len(points) != 4:
            return False
        quad = hasattr(points[0], "__iter__")
        if quad:
            corners = [tuple(float(value) for value in point) for point in points]
            if any(len(point) != 2 or not all(isfinite(value) for value in point) for point in corners):
                return False
            p0, p1, p2, p3 = corners
            # Paddle's angle classifier can read a line on a quarter-turned
            # PDF while returning an axis-aligned detector rectangle. Its
            # short top edge is then glyph height, not reading advance.
            # PDF rotation supplies the direction; aspect alone cannot.
            short_edge = hypot(p1[0] - p0[0], p1[1] - p0[1])
            long_edge = hypot(p3[0] - p0[0], p3[1] - p0[1])
            if (
                source_rotation in (90, 270)
                and len(text.strip()) >= 4
                and long_edge > 3 * short_edge
            ):
                p0, p1, p2, p3 = (
                    (p1, p2, p3, p0)
                    if source_rotation == 90
                    else (p3, p0, p1, p2)
                )
            elif (
                source_rotation == 180
                and len(text.strip()) >= 4
                and short_edge > 3 * long_edge
            ):
                p0, p1, p2, p3 = p2, p3, p0, p1
            top_x, top_y = p1[0] - p0[0], p1[1] - p0[1]
            side_x, side_y = p3[0] - p0[0], p3[1] - p0[1]
            advance_space = hypot(top_x, top_y)
            normal_space = hypot(side_x, side_y)
            if advance_space <= 0 or normal_space <= 0:
                return False
            ux, uy = top_x / advance_space, top_y / advance_space
            vx, vy = side_x / normal_space, side_y / normal_space
            # An affine text transform can fit a rotated rectangle, but it
            # cannot truthfully fit a degenerate or projective trapezoid.
            closure_error = hypot(
                p2[0] - (p1[0] + side_x),
                p2[1] - (p1[1] + side_y),
            )
            if (
                ux * vy - uy * vx <= 0
                or abs(ux * vx + uy * vy) > 0.1
                or closure_error > max(0.5, 0.1 * min(advance_space, normal_space))
            ):
                return False
            xs, ys = zip(*corners)
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        else:
            left, top, right, bottom = (float(value) for value in points)
            p0 = (left, top)
            ux, uy, vx, vy = 1.0, 0.0, 0.0, 1.0
            advance_space, normal_space = right - left, bottom - top
        if not all(isfinite(value) for value in (left, top, right, bottom)):
            return False
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            return False
        bounds = page.rect
        if left < 0 or top < 0 or right > bounds.width or bottom > bounds.height:
            return False
        unit_width, ascent, descent = _font_metrics(text, fontname, fontfile)
        if fontfile:
            font = _font_metrics_local.fonts[(fontname, fontfile)]
            if any(font.has_glyph(ord(char)) == 0 for char in text if not char.isspace()):
                # Script-specific Noto fonts may omit Latin glyphs. An ASCII
                # line on an Arabic/Indic page still has exact Base14 coverage;
                # recompute its geometry with that font instead of dropping it.
                # Mixed or unsupported Unicode remains fail-closed.
                if not text.isascii():
                    return False
                fontname, fontfile = "helv", None
                unit_width, ascent, descent = _font_metrics(text, fontname, fontfile)
        line_span = ascent - descent
        if not all(isfinite(value) for value in (unit_width, ascent, descent, line_span)):
            return False
        if unit_width <= 0 or line_span <= 0:
            return False
        # The detector width and height are independent constraints. A single
        # font size chosen from min(width, height) makes short text inside a
        # wide line box collapse toward its center and yields false highlights.
        # Fit height with font size, then scale glyph advance along the quad.
        fontsize = normal_space / line_span * 0.96
        if not isfinite(fontsize) or fontsize <= 0.01:
            return False
        horizontal_scale = advance_space / (fontsize * unit_width) * 0.98
        if not isfinite(horizontal_scale) or horizontal_scale <= 0:
            return False
        along_margin = advance_space * 0.01
        normal_margin = (normal_space - fontsize * line_span) / 2
        baseline = (
            p0[0] + ux * along_margin + vx * (normal_margin + fontsize * ascent),
            p0[1] + uy * along_margin + vy * (normal_margin + fontsize * ascent),
        )
        font_kwargs = {"fontname": fontname}
        if fontfile:
            font_kwargs["fontfile"] = fontfile
        angle = -degrees(atan2(uy, ux))
        morph = fitz.Matrix(horizontal_scale, 0, 0, 1, 0, 0) * fitz.Matrix(angle)
        page.insert_text(
            baseline, text, fontsize=fontsize, render_mode=3,
            morph=(fitz.Point(*baseline), morph),
            **font_kwargs,
        )
    except (TypeError, ValueError, RuntimeError, AttributeError):
        return False
    return True
