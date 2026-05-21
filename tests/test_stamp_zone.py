"""Unit tests for zone detection and placement helpers.

Tests zone detection, overlap detection, placement computation, and
fallback logic for stamp operations.

Run with: python -m pytest tests/test_stamp_zone.py -v
"""

import pytest

try:
    import fitz
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

from ocr_distributed.stamps import StampPlacement

if _HAS_FITZ:
    from ocr_distributed.stamps import Rect, Zone, ZoneDetector


# --- Rect Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
class TestRect:
    """Tests for Rect dataclass."""

    def test_rect_creation(self):
        rect = Rect(10, 20, 100, 200)
        assert rect.x0 == 10
        assert rect.y0 == 20
        assert rect.x1 == 100
        assert rect.y1 == 200

    def test_rect_overlaps_true(self):
        rect1 = Rect(0, 0, 100, 100)
        rect2 = Rect(50, 50, 150, 150)
        assert rect1.overlaps(rect2)
        assert rect2.overlaps(rect1)

    def test_rect_overlaps_false(self):
        rect1 = Rect(0, 0, 100, 100)
        rect2 = Rect(150, 150, 200, 200)
        assert not rect1.overlaps(rect2)
        assert not rect2.overlaps(rect1)

    def test_rect_overlaps_edge_touching(self):
        rect1 = Rect(0, 0, 100, 100)
        rect2 = Rect(100, 100, 200, 200)  # Touching at corner
        assert not rect1.overlaps(rect2)

    def test_rect_area(self):
        rect = Rect(0, 0, 100, 50)
        assert rect.area() == 5000

    def test_rect_area_negative_dimensions(self):
        rect = Rect(100, 100, 50, 50)  # Inverted
        assert rect.area() == 0

    def test_rect_intersection_area_overlap(self):
        rect1 = Rect(0, 0, 100, 100)
        rect2 = Rect(50, 50, 150, 150)
        intersection = rect1.intersection_area(rect2)
        assert intersection == 2500  # 50x50 overlap

    def test_rect_intersection_area_no_overlap(self):
        rect1 = Rect(0, 0, 100, 100)
        rect2 = Rect(150, 150, 200, 200)
        assert rect1.intersection_area(rect2) == 0


# --- Zone Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
class TestZone:
    """Tests for Zone dataclass."""

    def test_zone_creation(self):
        rect = Rect(10, 20, 100, 200)
        zone = Zone(rect=rect, zone_type="text", confidence=0.95)
        assert zone.rect == rect
        assert zone.zone_type == "text"
        assert zone.confidence == 0.95

    def test_zone_default_confidence(self):
        rect = Rect(10, 20, 100, 200)
        zone = Zone(rect=rect, zone_type="stamp")
        assert zone.confidence == 1.0

    def test_zone_overlaps(self):
        zone1 = Zone(rect=Rect(0, 0, 100, 100), zone_type="text")
        zone2 = Zone(rect=Rect(50, 50, 150, 150), zone_type="stamp")
        assert zone1.overlaps(zone2)
        assert zone2.overlaps(zone1)


# --- ZoneDetector Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
class TestZoneDetector:
    """Tests for ZoneDetector."""

    def setup_method(self):
        """Create detector for each test."""
        self.detector = ZoneDetector(margin=10.0, min_clearance=5.0)

    def test_detector_initialization(self):
        assert self.detector.margin == 10.0
        assert self.detector.min_clearance == 5.0

    def test_detect_text_zones_empty_page(self):
        """Test zone detection on empty page."""
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        zones = self.detector.detect_text_zones(page)
        assert zones == []
        doc.close()

    def test_detect_text_zones_with_text(self):
        """Test zone detection on page with text."""
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 50), "Test text", fontsize=12)
        zones = self.detector.detect_text_zones(page)
        assert len(zones) > 0
        assert zones[0].zone_type == "text"
        doc.close()

    def test_compute_base_placement_top_left(self):
        """Test top-left placement computation."""
        x0, y0 = self.detector._compute_base_placement(
            StampPlacement.TOP_LEFT, 595, 842, 100, 20
        )
        assert x0 == self.detector.margin
        assert y0 == self.detector.margin

    def test_compute_base_placement_bottom_right(self):
        """Test bottom-right placement computation."""
        x0, y0 = self.detector._compute_base_placement(
            StampPlacement.BOTTOM_RIGHT, 595, 842, 100, 20
        )
        assert x0 == 595 - 100 - self.detector.margin
        assert y0 == 842 - 20 - self.detector.margin

    def test_compute_base_placement_center(self):
        """Test center placement computation."""
        x0, y0 = self.detector._compute_base_placement(
            StampPlacement.CENTER, 595, 842, 100, 20
        )
        assert x0 == (595 - 100) / 2
        assert y0 == (842 - 20) / 2

    def test_compute_base_placement_top_center(self):
        """Test top-center placement computation."""
        x0, y0 = self.detector._compute_base_placement(
            StampPlacement.TOP_CENTER, 595, 842, 100, 20
        )
        assert x0 == (595 - 100) / 2
        assert y0 == self.detector.margin

    def test_has_overlap_true(self):
        """Test overlap detection returns True when zones overlap."""
        stamp_rect = Rect(50, 50, 150, 70)
        zones = [Zone(rect=Rect(40, 40, 160, 80), zone_type="text")]
        assert self.detector._has_overlap(stamp_rect, zones)

    def test_has_overlap_false(self):
        """Test overlap detection returns False when no overlap."""
        stamp_rect = Rect(200, 200, 300, 220)
        zones = [Zone(rect=Rect(40, 40, 160, 80), zone_type="text")]
        assert not self.detector._has_overlap(stamp_rect, zones)

    def test_has_overlap_with_clearance(self):
        """Test overlap detection respects clearance buffer."""
        # Stamp just outside text zone, but within clearance
        stamp_rect = Rect(165, 40, 265, 60)  # 5 points from zone
        zones = [Zone(rect=Rect(40, 40, 160, 80), zone_type="text")]
        # Should detect overlap due to clearance
        assert self.detector._has_overlap(stamp_rect, zones)

    def test_get_fallback_placements_bottom_right(self):
        """Test fallback placement order for bottom-right."""
        fallbacks = self.detector._get_fallback_placements(StampPlacement.BOTTOM_RIGHT)
        assert len(fallbacks) > 0
        assert StampPlacement.TOP_LEFT in fallbacks

    def test_get_fallback_placements_top_left(self):
        """Test fallback placement order for top-left."""
        fallbacks = self.detector._get_fallback_placements(StampPlacement.TOP_LEFT)
        assert len(fallbacks) > 0
        assert StampPlacement.BOTTOM_RIGHT in fallbacks

    def test_compute_placement_rect_no_overlap(self):
        """Test placement computation with no existing zones."""
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        
        rect = self.detector.compute_placement_rect(
            page, StampPlacement.BOTTOM_RIGHT, 100, 20, []
        )
        
        # Should use primary placement
        expected_x = 595 - 100 - self.detector.margin
        expected_y = 842 - 20 - self.detector.margin
        assert rect.x0 == expected_x
        assert rect.y0 == expected_y
        doc.close()

    def test_compute_placement_rect_with_overlap_fallback(self):
        """Test placement computation with overlap triggers fallback."""
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        
        # Create zone at bottom-right (primary placement)
        bottom_right_zone = Zone(
            rect=Rect(450, 800, 580, 830),
            zone_type="text"
        )
        
        rect = self.detector.compute_placement_rect(
            page, StampPlacement.BOTTOM_RIGHT, 100, 20, [bottom_right_zone]
        )
        
        # Should fall back to different placement (not bottom-right)
        expected_x = 595 - 100 - self.detector.margin
        expected_y = 842 - 20 - self.detector.margin
        assert not (rect.x0 == expected_x and rect.y0 == expected_y)
        doc.close()

    def test_detect_overlap_warnings_no_overlap(self):
        """Test overlap warning detection with no overlap."""
        stamp_rect = Rect(200, 200, 300, 220)
        zones = [Zone(rect=Rect(40, 40, 160, 80), zone_type="text")]
        warnings = self.detector.detect_overlap_warnings(stamp_rect, zones)
        assert warnings == []

    def test_detect_overlap_warnings_with_overlap(self):
        """Test overlap warning detection with overlap."""
        stamp_rect = Rect(50, 50, 150, 70)  # 100x20
        zones = [Zone(rect=Rect(100, 50, 160, 70), zone_type="text")]  # 50x20 overlap
        warnings = self.detector.detect_overlap_warnings(stamp_rect, zones, threshold=0.1)
        assert len(warnings) > 0
        assert "overlap" in warnings[0].lower()

    def test_detect_overlap_warnings_below_threshold(self):
        """Test overlap warning only triggers above threshold."""
        stamp_rect = Rect(50, 50, 150, 70)  # 100x20 = 2000 area
        # Small overlap (100 area = 5% of stamp)
        zones = [Zone(rect=Rect(145, 65, 160, 80), zone_type="text")]
        warnings = self.detector.detect_overlap_warnings(stamp_rect, zones, threshold=0.1)
        # Should not warn for <10% overlap
        assert len(warnings) == 0

    def test_detector_with_custom_margin(self):
        """Test detector with custom margin."""
        detector = ZoneDetector(margin=20.0)
        x0, y0 = detector._compute_base_placement(
            StampPlacement.TOP_LEFT, 595, 842, 100, 20
        )
        assert x0 == 20.0
        assert y0 == 20.0

    def test_detector_with_custom_clearance(self):
        """Test detector with custom clearance."""
        detector = ZoneDetector(min_clearance=15.0)
        stamp_rect = Rect(175, 40, 275, 60)  # 15 points from zone
        zones = [Zone(rect=Rect(40, 40, 160, 80), zone_type="text")]
        # Should detect overlap due to 15pt clearance
        assert detector._has_overlap(stamp_rect, zones)
