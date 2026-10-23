"""Immutable page-space contract for raster OCR and searchable PDF output.

The caller owns source snapshot authorization and supplies an already-open PDF
page or image metadata.  No source path is opened here.  Raster dimensions are
measured *after* rendering/preprocessing, so a retry at another DPI has the
same physical output size and independently scaled OCR coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable


def _positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _rect_tuple(rect: Any) -> tuple[float, float, float, float]:
    values = (rect.x0, rect.y0, rect.x1, rect.y1)
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("source rectangle must be finite")
    return tuple(float(value) for value in values)


def _dpi_pair(dpi: Any, fallback_dpi: float) -> tuple[float, float]:
    fallback = _positive(fallback_dpi, "fallback_dpi")
    if isinstance(dpi, (tuple, list)) and len(dpi) == 2:
        candidates = dpi
    else:
        candidates = (dpi, dpi)
    result = []
    for candidate in candidates:
        try:
            value = _positive(candidate, "dpi")
        except ValueError:
            value = fallback
        # Image metadata frequently contains placeholder or corrupt values.
        result.append(value if 10 <= value <= 2400 else fallback)
    return result[0], result[1]


@dataclass(frozen=True)
class PageGeometry:
    """Physical size of the visible page, in PDF points (1/72 inch).

    PDF ``page.rect`` already includes CropBox and page rotation in the same
    orientation used by normal raster renderers.  Its origin is deliberately
    discarded: new output pages always have local origin (0, 0).  The source
    boxes and rotation are retained for provenance and diagnostics.
    """

    width_pt: float
    height_pt: float
    source_kind: str
    source_cropbox: tuple[float, float, float, float] | None = None
    source_mediabox: tuple[float, float, float, float] | None = None
    source_rotation: int = 0
    source_dpi: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _positive(self.width_pt, "width_pt")
        _positive(self.height_pt, "height_pt")
        if self.source_kind not in ("pdf", "image", "checkpoint-local", "raster-dpi-fallback"):
            raise ValueError("invalid page geometry source kind")
        if self.source_rotation not in (0, 90, 180, 270):
            raise ValueError("invalid source page rotation")
        for name in ("source_cropbox", "source_mediabox"):
            box = getattr(self, name)
            if box is not None:
                try:
                    x0, y0, x1, y1 = (float(value) for value in box)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} must be a finite rectangle") from exc
                if not all(isfinite(value) for value in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
                    raise ValueError(f"{name} must be a finite rectangle")
        if self.source_dpi is not None:
            try:
                x_dpi, y_dpi = self.source_dpi
            except (TypeError, ValueError) as exc:
                raise ValueError("source_dpi must have two valid values") from exc
            _positive(x_dpi, "source X DPI")
            _positive(y_dpi, "source Y DPI")

    @classmethod
    def from_pdf_page(cls, page: Any) -> PageGeometry:
        rect = page.rect
        rotation = int(page.rotation)
        if rotation not in (0, 90, 180, 270):
            raise ValueError("PDF page rotation must be 0, 90, 180, or 270")
        return cls(
            width_pt=_positive(rect.width, "visible page width"),
            height_pt=_positive(rect.height, "visible page height"),
            source_kind="pdf",
            source_cropbox=_rect_tuple(page.cropbox),
            source_mediabox=_rect_tuple(page.mediabox),
            source_rotation=rotation,
        )

    @classmethod
    def from_image_pixels(
        cls,
        width: int,
        height: int,
        *,
        dpi: float | tuple[float, float] | None = None,
        fallback_dpi: float = 300.0,
    ) -> PageGeometry:
        width_px = _positive(width, "source image width")
        height_px = _positive(height, "source image height")
        dpi_x, dpi_y = _dpi_pair(dpi, fallback_dpi)
        return cls(
            width_pt=width_px * 72.0 / dpi_x,
            height_pt=height_px * 72.0 / dpi_y,
            source_kind="image",
            source_dpi=(dpi_x, dpi_y),
        )

    def for_raster(self, width: int, height: int) -> RasterTransform:
        return RasterTransform(self, _positive(width, "raster width"), _positive(height, "raster height"))


@dataclass(frozen=True)
class RasterTransform:
    """Convert coordinates from the *actual OCR raster* into output points."""

    page: PageGeometry
    width_px: float
    height_px: float

    def __post_init__(self) -> None:
        _positive(self.width_px, "raster width")
        _positive(self.height_px, "raster height")

    def map_point(self, x: float, y: float) -> list[float]:
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError) as exc:
            raise ValueError("point coordinates must be finite") from exc
        if not isfinite(x) or not isfinite(y):
            raise ValueError("point coordinates must be finite")
        return [x * self.page.width_pt / self.width_px, y * self.page.height_pt / self.height_px]

    def map_box(self, box: Iterable[float]) -> list[float]:
        try:
            x0, y0, x1, y1 = box
        except (TypeError, ValueError) as exc:
            raise ValueError("box must contain four coordinates") from exc
        # Tesseract can report boxes a pixel beyond an image edge.  Clip the
        # axis-aligned rectangle before scaling; reject boxes wholly outside.
        left, top = self.map_point(x0, y0)
        right, bottom = self.map_point(x1, y1)
        left = max(0.0, left)
        top = max(0.0, top)
        right = min(self.page.width_pt, right)
        bottom = min(self.page.height_pt, bottom)
        if right <= left or bottom <= top:
            raise ValueError("box must have positive width and height")
        return [left, top, right, bottom]

    def map_quad(self, quad: Iterable[Iterable[float]]) -> list[list[float]]:
        try:
            corners = [self.map_point(*corner) for corner in quad]
        except (TypeError, ValueError) as exc:
            raise ValueError("quad must contain four 2D points") from exc
        if len(corners) != 4:
            raise ValueError("quad must contain four 2D points")
        # Detectors can overrun by a pixel or two at a page edge.  Clamp only
        # that bounded overhang; larger errors require explicit review.
        tolerance_x = 2 * self.page.width_pt / self.width_px
        tolerance_y = 2 * self.page.height_pt / self.height_px
        if any(
            x < -tolerance_x or x > self.page.width_pt + tolerance_x
            or y < -tolerance_y or y > self.page.height_pt + tolerance_y
            for x, y in corners
        ):
            raise ValueError("quad lies outside the raster tolerance")
        corners = [
            [min(max(x, 0.0), self.page.width_pt), min(max(y, 0.0), self.page.height_pt)]
            for x, y in corners
        ]
        xs, ys = zip(*corners)
        if max(xs) <= min(xs) or max(ys) <= min(ys):
            raise ValueError("quad must have positive extent")
        return corners

    def map_lines_with_review(
        self, lines: Iterable[tuple[str, Any, float]]
    ) -> tuple[list[tuple[str, Any, float]], list[dict[str, Any]]]:
        """Map placeable lines and return rejected raw spans for review.

        No fabricated location is assigned to an unplaceable OCR span.
        Consumers must persist the issues and treat the page as needing review.
        """
        mapped = []
        issues = []
        for index, (text, box, confidence) in enumerate(lines):
            try:
                # Paddle may supply numpy arrays; use shape, not concrete
                # collection type, to distinguish a polygon from a rectangle.
                if len(box) == 4 and hasattr(box[0], "__iter__"):
                    out_box = self.map_quad(box)
                else:
                    out_box = self.map_box(box)
                conf = float(confidence)
                if not isfinite(conf):
                    conf = 0.0
                    issues.append({"index": index, "text": text, "reason": "invalid confidence"})
            except (TypeError, ValueError) as exc:
                issues.append({"index": index, "text": text, "reason": str(exc)})
                continue
            mapped.append((text, out_box, conf))
        return mapped, issues

    def map_lines(self, lines: Iterable[tuple[str, Any, float]]) -> list[tuple[str, Any, float]]:
        """Map placeable OCR text; use ``map_lines_with_review`` for issues."""
        return self.map_lines_with_review(lines)[0]

