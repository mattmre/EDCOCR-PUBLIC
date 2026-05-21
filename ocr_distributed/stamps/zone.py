"""Zone and placement helpers for stamp operations.

Provides utilities for detecting text block zones, computing placement coordinates,
and detecting overlaps between stamps and existing content.
"""

from dataclasses import dataclass
from typing import Optional

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False
    fitz = None  # type: ignore

from .base import StampPlacement


@dataclass
class Rect:
    """Rectangle coordinates."""
    x0: float
    y0: float
    x1: float
    y1: float

    def overlaps(self, other: "Rect") -> bool:
        """Check if this rectangle overlaps with another."""
        return not (
            self.x1 <= other.x0
            or self.x0 >= other.x1
            or self.y1 <= other.y0
            or self.y0 >= other.y1
        )

    def area(self) -> float:
        """Compute rectangle area."""
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)

    def intersection_area(self, other: "Rect") -> float:
        """Compute intersection area with another rectangle."""
        if not self.overlaps(other):
            return 0.0
        
        x0 = max(self.x0, other.x0)
        x1 = min(self.x1, other.x1)
        y0 = max(self.y0, other.y0)
        y1 = min(self.y1, other.y1)
        
        return max(0, x1 - x0) * max(0, y1 - y0)


@dataclass
class Zone:
    """Text or stamp zone on a page."""
    rect: Rect
    zone_type: str  # "text", "stamp", "image"
    confidence: float = 1.0  # 0.0-1.0

    def overlaps(self, other: "Zone") -> bool:
        """Check if this zone overlaps with another."""
        return self.rect.overlaps(other.rect)


class ZoneDetector:
    """Detects text zones and computes placement with overlap avoidance."""

    def __init__(self, margin: float = 10.0, min_clearance: float = 5.0):
        """Initialize zone detector.
        
        Args:
            margin: Margin from page edges (points)
            min_clearance: Minimum clearance from text blocks (points)
        """
        self.margin = margin
        self.min_clearance = min_clearance

    def detect_text_zones(self, page: "fitz.Page") -> list[Zone]:
        """Detect text zones on a page using PyMuPDF.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            List of detected text zones
        """
        if not _HAS_FITZ:
            return []
        
        zones = []
        blocks = page.get_text("blocks")
        
        for block in blocks:
            # block is (x0, y0, x1, y1, text, block_no, block_type)
            if len(block) >= 7:
                x0, y0, x1, y1 = block[0:4]
                zones.append(Zone(
                    rect=Rect(x0, y0, x1, y1),
                    zone_type="text",
                    confidence=1.0
                ))
        
        return zones

    def compute_placement_rect(
        self,
        page: "fitz.Page",
        placement: StampPlacement,
        stamp_width: float,
        stamp_height: float,
        existing_zones: Optional[list[Zone]] = None
    ) -> Rect:
        """Compute stamp placement rectangle with overlap avoidance.
        
        Args:
            page: PyMuPDF page object
            placement: Desired stamp placement
            stamp_width: Stamp width (points)
            stamp_height: Stamp height (points)
            existing_zones: Pre-detected zones (auto-detect if None)
            
        Returns:
            Rectangle for stamp placement
        """
        if existing_zones is None:
            existing_zones = self.detect_text_zones(page)
        
        page_rect = page.rect
        width = page_rect.width
        height = page_rect.height
        
        # Compute primary placement coordinates
        x0, y0 = self._compute_base_placement(
            placement, width, height, stamp_width, stamp_height
        )
        
        stamp_rect = Rect(x0, y0, x0 + stamp_width, y0 + stamp_height)
        
        # Check for overlaps
        if not self._has_overlap(stamp_rect, existing_zones):
            return stamp_rect
        
        # Try fallback placements if primary has overlap
        fallback_order = self._get_fallback_placements(placement)
        for fallback in fallback_order:
            x0, y0 = self._compute_base_placement(
                fallback, width, height, stamp_width, stamp_height
            )
            stamp_rect = Rect(x0, y0, x0 + stamp_width, y0 + stamp_height)
            
            if not self._has_overlap(stamp_rect, existing_zones):
                return stamp_rect
        
        # If all placements overlap, return primary with warning flag
        x0, y0 = self._compute_base_placement(
            placement, width, height, stamp_width, stamp_height
        )
        return Rect(x0, y0, x0 + stamp_width, y0 + stamp_height)

    def _compute_base_placement(
        self,
        placement: StampPlacement,
        page_width: float,
        page_height: float,
        stamp_width: float,
        stamp_height: float
    ) -> tuple[float, float]:
        """Compute base placement coordinates (x0, y0)."""
        m = self.margin
        
        if placement == StampPlacement.TOP_LEFT:
            return (m, m)
        elif placement == StampPlacement.TOP_CENTER:
            return ((page_width - stamp_width) / 2, m)
        elif placement == StampPlacement.TOP_RIGHT:
            return (page_width - stamp_width - m, m)
        elif placement == StampPlacement.BOTTOM_LEFT:
            return (m, page_height - stamp_height - m)
        elif placement == StampPlacement.BOTTOM_CENTER:
            return ((page_width - stamp_width) / 2, page_height - stamp_height - m)
        elif placement == StampPlacement.BOTTOM_RIGHT:
            return (page_width - stamp_width - m, page_height - stamp_height - m)
        elif placement == StampPlacement.CENTER:
            return ((page_width - stamp_width) / 2, (page_height - stamp_height) / 2)
        else:
            # Default to bottom right
            return (page_width - stamp_width - m, page_height - stamp_height - m)

    def _has_overlap(self, stamp_rect: Rect, zones: list[Zone]) -> bool:
        """Check if stamp rect overlaps with any zone beyond clearance threshold."""
        for zone in zones:
            # Expand zone by min_clearance for buffer
            expanded = Rect(
                zone.rect.x0 - self.min_clearance,
                zone.rect.y0 - self.min_clearance,
                zone.rect.x1 + self.min_clearance,
                zone.rect.y1 + self.min_clearance
            )
            # Inclusive boundary check for clearance: touching the clearance edge
            # still counts as overlap to preserve the requested spacing buffer.
            if not (
                stamp_rect.x1 < expanded.x0
                or stamp_rect.x0 > expanded.x1
                or stamp_rect.y1 < expanded.y0
                or stamp_rect.y0 > expanded.y1
            ):
                return True
        return False

    def _get_fallback_placements(self, primary: StampPlacement) -> list[StampPlacement]:
        """Get deterministic fallback placement order for given primary placement."""
        # Preference: opposite corner, then adjacent corners, then center
        fallback_map = {
            StampPlacement.TOP_LEFT: [
                StampPlacement.BOTTOM_RIGHT,
                StampPlacement.TOP_RIGHT,
                StampPlacement.BOTTOM_LEFT,
                StampPlacement.CENTER,
            ],
            StampPlacement.TOP_CENTER: [
                StampPlacement.BOTTOM_CENTER,
                StampPlacement.TOP_RIGHT,
                StampPlacement.TOP_LEFT,
                StampPlacement.CENTER,
            ],
            StampPlacement.TOP_RIGHT: [
                StampPlacement.BOTTOM_LEFT,
                StampPlacement.TOP_LEFT,
                StampPlacement.BOTTOM_RIGHT,
                StampPlacement.CENTER,
            ],
            StampPlacement.BOTTOM_LEFT: [
                StampPlacement.TOP_RIGHT,
                StampPlacement.BOTTOM_RIGHT,
                StampPlacement.TOP_LEFT,
                StampPlacement.CENTER,
            ],
            StampPlacement.BOTTOM_CENTER: [
                StampPlacement.TOP_CENTER,
                StampPlacement.BOTTOM_RIGHT,
                StampPlacement.BOTTOM_LEFT,
                StampPlacement.CENTER,
            ],
            StampPlacement.BOTTOM_RIGHT: [
                StampPlacement.TOP_LEFT,
                StampPlacement.BOTTOM_LEFT,
                StampPlacement.TOP_RIGHT,
                StampPlacement.CENTER,
            ],
            StampPlacement.CENTER: [
                StampPlacement.BOTTOM_RIGHT,
                StampPlacement.TOP_LEFT,
                StampPlacement.BOTTOM_LEFT,
                StampPlacement.TOP_RIGHT,
            ],
        }
        return fallback_map.get(primary, [])

    def detect_overlap_warnings(
        self,
        stamp_rect: Rect,
        zones: list[Zone],
        threshold: float = 0.1
    ) -> list[str]:
        """Detect overlaps and return warning messages.
        
        Args:
            stamp_rect: Stamp rectangle
            zones: Existing zones to check
            threshold: Overlap area threshold (fraction of stamp area)
            
        Returns:
            List of warning messages
        """
        warnings = []
        stamp_area = stamp_rect.area()
        
        if stamp_area == 0:
            return warnings
        
        for zone in zones:
            overlap_area = stamp_rect.intersection_area(zone.rect)
            if overlap_area > 0:
                overlap_pct = (overlap_area / stamp_area) * 100
                if overlap_pct >= threshold * 100:
                    warnings.append(
                        f"Stamp overlaps {zone.zone_type} zone by {overlap_pct:.1f}%"
                    )
        
        return warnings
