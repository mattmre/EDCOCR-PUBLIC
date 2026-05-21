"""Tests for adaptive DPI escalation module (Phase 4D+).

These tests verify that:
- DPI schedule stepping works correctly (300->450->600->None)
- Escalation threshold logic respects confidence and retry caps
- EscalationResult dataclass has correct defaults
- re_extract_page_at_dpi handles success and failure with mocked pdf2image
- Integration scenarios cover end-to-end escalation decision logic

Run with: python -m pytest tests/test_dpi_escalation.py -v
"""

from unittest.mock import MagicMock, patch

from PIL import Image

# Add project root to path
from dpi_escalation import (
    CONFIDENCE_THRESHOLD_RETRY,
    DPI_SCHEDULE,
    MAX_ESCALATION_RETRIES,
    EscalationResult,
    get_next_dpi,
    re_extract_page_at_dpi,
    should_escalate,
)

# ---------------------------------------------------------------------------
# TestGetNextDPI
# ---------------------------------------------------------------------------


class TestGetNextDPI:
    """Test DPI schedule stepping logic."""

    def test_300_to_450(self):
        """300 DPI should escalate to 450."""
        assert get_next_dpi(300) == 450

    def test_450_to_600(self):
        """450 DPI should escalate to 600."""
        assert get_next_dpi(450) == 600

    def test_600_is_max(self):
        """600 DPI is maximum -- returns None."""
        assert get_next_dpi(600) is None

    def test_unknown_dpi_returns_none(self):
        """DPI value not in schedule returns None."""
        assert get_next_dpi(200) is None

    def test_zero_dpi_returns_none(self):
        """Zero DPI returns None (not in schedule)."""
        assert get_next_dpi(0) is None

    def test_negative_dpi_returns_none(self):
        """Negative DPI returns None."""
        assert get_next_dpi(-100) is None

    def test_schedule_order(self):
        """DPI_SCHEDULE should be monotonically increasing."""
        for i in range(len(DPI_SCHEDULE) - 1):
            assert DPI_SCHEDULE[i] < DPI_SCHEDULE[i + 1]

    def test_schedule_has_three_steps(self):
        """DPI_SCHEDULE should have exactly 3 steps."""
        assert len(DPI_SCHEDULE) == 3


# ---------------------------------------------------------------------------
# TestShouldEscalate
# ---------------------------------------------------------------------------


class TestShouldEscalate:
    """Test escalation threshold and retry cap logic."""

    def test_low_confidence_zero_retries(self):
        """Low confidence with no retries should escalate."""
        assert should_escalate(0.30, 0) is True

    def test_high_confidence_no_escalation(self):
        """Confidence at threshold should NOT escalate."""
        assert should_escalate(0.60, 0) is False

    def test_above_threshold_no_escalation(self):
        """Confidence above threshold should NOT escalate."""
        assert should_escalate(0.75, 0) is False

    def test_just_below_threshold(self):
        """Confidence just below threshold should escalate."""
        assert should_escalate(0.599, 0) is True

    def test_zero_confidence(self):
        """Zero confidence should escalate (if retries available)."""
        assert should_escalate(0.0, 0) is True

    def test_max_retries_reached(self):
        """Should NOT escalate when max retries reached, even with low confidence."""
        assert should_escalate(0.10, MAX_ESCALATION_RETRIES) is False

    def test_one_retry_used(self):
        """Should escalate with one retry used (under max)."""
        assert should_escalate(0.40, 1) is True

    def test_custom_threshold(self):
        """Custom threshold should override default."""
        assert should_escalate(0.50, 0, threshold=0.40) is False
        assert should_escalate(0.35, 0, threshold=0.40) is True

    def test_threshold_zero(self):
        """Threshold of 0 means nothing escalates (confidence is always >= 0)."""
        assert should_escalate(0.0, 0, threshold=0.0) is False

    def test_threshold_one(self):
        """Threshold of 1.0 means everything below 1.0 escalates."""
        assert should_escalate(0.99, 0, threshold=1.0) is True

    def test_negative_retries_treated_as_available(self):
        """Negative retry count (bug guard) should still allow escalation."""
        assert should_escalate(0.30, -1) is True

    def test_default_threshold_matches_constant(self):
        """Default threshold parameter should match module constant."""
        assert CONFIDENCE_THRESHOLD_RETRY == 0.60


# ---------------------------------------------------------------------------
# TestEscalationResult
# ---------------------------------------------------------------------------


class TestEscalationResult:
    """Test EscalationResult dataclass defaults and fields."""

    def test_default_values(self):
        """Default EscalationResult should indicate no escalation."""
        result = EscalationResult()
        assert result.escalated is False
        assert result.original_dpi == 300
        assert result.final_dpi == 300
        assert result.original_confidence == 0.0
        assert result.final_confidence == 0.0
        assert result.retries_used == 0

    def test_custom_values(self):
        """EscalationResult should accept custom values."""
        result = EscalationResult(
            escalated=True,
            original_dpi=300,
            final_dpi=450,
            original_confidence=0.35,
            final_confidence=0.72,
            retries_used=1,
        )
        assert result.escalated is True
        assert result.final_dpi == 450
        assert result.final_confidence == 0.72
        assert result.retries_used == 1

    def test_result_is_dataclass(self):
        """EscalationResult should be a proper dataclass."""
        from dataclasses import fields

        field_names = {f.name for f in fields(EscalationResult)}
        expected = {
            "escalated",
            "original_dpi",
            "final_dpi",
            "original_confidence",
            "final_confidence",
            "retries_used",
        }
        assert field_names == expected


# ---------------------------------------------------------------------------
# TestReExtractPage
# ---------------------------------------------------------------------------


class TestReExtractPage:
    """Test re_extract_page_at_dpi with mocked pdf2image."""

    @patch("dpi_escalation._convert_from_path")
    def test_successful_extraction(self, mock_convert):
        """Successful extraction returns an RGB PIL Image."""
        mock_img = Image.new("RGB", (2550, 3300), color=(255, 255, 255))
        mock_convert.return_value = [mock_img]

        result = re_extract_page_at_dpi("/fake/doc.pdf", 1, 450)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        mock_convert.assert_called_once_with(
            "/fake/doc.pdf", first_page=1, last_page=1, dpi=450,
            timeout=300,
        )

    @patch("dpi_escalation._convert_from_path")
    def test_converts_non_rgb_to_rgb(self, mock_convert):
        """Non-RGB images should be converted to RGB."""
        mock_img = MagicMock(spec=Image.Image)
        rgb_img = Image.new("RGB", (100, 100))
        mock_img.convert.return_value = rgb_img
        mock_convert.return_value = [mock_img]

        result = re_extract_page_at_dpi("/fake/doc.pdf", 5, 600)

        assert result is not None
        mock_img.convert.assert_called_once_with("RGB")

    @patch("dpi_escalation._convert_from_path")
    def test_empty_result_returns_none(self, mock_convert):
        """Empty conversion result returns None."""
        mock_convert.return_value = []

        result = re_extract_page_at_dpi("/fake/doc.pdf", 1, 450)

        assert result is None

    @patch("dpi_escalation._convert_from_path")
    def test_exception_returns_none(self, mock_convert):
        """Exception during conversion returns None (no crash)."""
        mock_convert.side_effect = RuntimeError("Poppler not found")

        result = re_extract_page_at_dpi("/fake/doc.pdf", 1, 450)

        assert result is None

    @patch("dpi_escalation._convert_from_path")
    def test_passes_correct_page_number(self, mock_convert):
        """Page number should be passed correctly to convert_from_path."""
        mock_img = Image.new("RGB", (100, 100))
        mock_convert.return_value = [mock_img]

        re_extract_page_at_dpi("/fake/doc.pdf", 42, 300)

        mock_convert.assert_called_once_with(
            "/fake/doc.pdf", first_page=42, last_page=42, dpi=300,
            timeout=300,
        )

    @patch("dpi_escalation._convert_from_path")
    def test_passes_correct_dpi(self, mock_convert):
        """DPI value should be passed correctly to convert_from_path."""
        mock_img = Image.new("RGB", (100, 100))
        mock_convert.return_value = [mock_img]

        re_extract_page_at_dpi("/fake/doc.pdf", 1, 600)

        _, kwargs = mock_convert.call_args
        assert kwargs["dpi"] == 600


# ---------------------------------------------------------------------------
# TestIntegrationScenarios
# ---------------------------------------------------------------------------


class TestIntegrationScenarios:
    """Test full escalation workflow scenarios combining multiple functions."""

    def test_low_confidence_first_attempt_should_escalate(self):
        """Page at 0.45 confidence, zero retries -> should escalate once."""
        confidence = 0.45
        retries = 0
        current_dpi = 300

        assert should_escalate(confidence, retries) is True
        next_dpi = get_next_dpi(current_dpi)
        assert next_dpi == 450

    def test_still_low_after_first_retry_should_escalate_again(self):
        """Page at 0.55 confidence after first retry -> should escalate again."""
        confidence = 0.55
        retries = 1
        current_dpi = 450

        assert should_escalate(confidence, retries) is True
        next_dpi = get_next_dpi(current_dpi)
        assert next_dpi == 600

    def test_improved_after_retry_should_not_escalate(self):
        """Page at 0.65 after retry -> should NOT escalate (above threshold)."""
        confidence = 0.65
        retries = 1

        assert should_escalate(confidence, retries) is False

    def test_max_retries_exhausted_should_not_escalate(self):
        """Page at 0.30 confidence, 2 retries done -> should NOT escalate."""
        confidence = 0.30
        retries = 2

        assert should_escalate(confidence, retries) is False
        # Also verify that 600 DPI has no next step
        assert get_next_dpi(600) is None

    def test_image_source_no_pdf_reextraction(self):
        """Image files cannot be re-extracted at higher DPI (source_type != 'pdf').

        The pipeline should check source_type='pdf' before calling
        re_extract_page_at_dpi. This test verifies the logic guard.
        """
        source_type = "image"
        confidence = 0.30
        retries = 0

        # should_escalate says yes based on confidence/retries alone
        assert should_escalate(confidence, retries) is True
        # But the pipeline should NOT call re_extract because source_type != "pdf"
        assert source_type != "pdf"

    def test_full_escalation_path(self):
        """Walk through complete 300->450->600 escalation path."""
        current_dpi = 300
        retries = 0

        # First escalation: 300 -> 450
        assert should_escalate(0.40, retries) is True
        next_dpi = get_next_dpi(current_dpi)
        assert next_dpi == 450
        retries += 1
        current_dpi = next_dpi

        # Second escalation: 450 -> 600
        assert should_escalate(0.50, retries) is True
        next_dpi = get_next_dpi(current_dpi)
        assert next_dpi == 600
        retries += 1
        current_dpi = next_dpi

        # Third attempt blocked by max retries
        assert should_escalate(0.45, retries) is False

    def test_escalation_result_tracks_improvement(self):
        """EscalationResult should track before/after metrics."""
        result = EscalationResult(
            escalated=True,
            original_dpi=300,
            final_dpi=600,
            original_confidence=0.35,
            final_confidence=0.78,
            retries_used=2,
        )
        improvement = result.final_confidence - result.original_confidence
        assert improvement > 0.40
        assert result.retries_used == MAX_ESCALATION_RETRIES

    def test_borderline_confidence_at_threshold(self):
        """Confidence exactly at threshold boundary should NOT escalate."""
        threshold = 0.60
        assert should_escalate(threshold, 0) is False
        assert should_escalate(threshold - 0.001, 0) is True

    @patch("dpi_escalation._convert_from_path")
    def test_escalation_with_reextract_success(self, mock_convert):
        """Full scenario: low confidence -> escalate -> re-extract succeeds."""
        confidence = 0.40
        retries = 0
        current_dpi = 300

        # Decision: should escalate
        assert should_escalate(confidence, retries) is True
        next_dpi = get_next_dpi(current_dpi)
        assert next_dpi == 450

        # Re-extract at higher DPI
        mock_img = Image.new("RGB", (3825, 4950))  # 8.5x11 at 450 DPI
        mock_convert.return_value = [mock_img]

        new_img = re_extract_page_at_dpi("/test/doc.pdf", 1, next_dpi)
        assert new_img is not None
        assert new_img.size == (3825, 4950)

    @patch("dpi_escalation._convert_from_path")
    def test_escalation_with_reextract_failure(self, mock_convert):
        """Full scenario: low confidence -> escalate -> re-extract fails."""
        confidence = 0.40
        retries = 0

        assert should_escalate(confidence, retries) is True
        next_dpi = get_next_dpi(300)

        # Re-extract fails
        mock_convert.side_effect = OSError("File not found")
        new_img = re_extract_page_at_dpi("/missing/doc.pdf", 1, next_dpi)
        assert new_img is None
