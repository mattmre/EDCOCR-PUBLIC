"""Unit tests for Bates numbering stamp operations.

Tests BatesAssigner, BatesConfig, and BatesStampOperation with synthetic PDFs.

Run with: python -m pytest tests/test_stamp_bates.py -v
"""

import tempfile
from pathlib import Path

import pytest

try:
    import fitz
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

from ocr_distributed.stamps import (
    StampConfig,
    StampPlacement,
    StampValidationError,
)

if _HAS_FITZ:
    from ocr_distributed.stamps import (
        BatesAssigner,
        BatesConfig,
        BatesStampOperation,
    )


# --- Test Fixtures ---


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_3pages(temp_dir):
    """Create a sample 3-page PDF for testing."""
    if not _HAS_FITZ:
        pytest.skip("PyMuPDF not available")
    
    pdf_path = temp_dir / "sample_3pages.pdf"
    doc = fitz.open()
    
    for i in range(3):
        page = doc.new_page(width=595, height=842)  # A4 size
        page.insert_text((50, 50), f"Page {i+1}", fontsize=24)
    
    doc.save(str(pdf_path))
    doc.close()
    
    return pdf_path


# --- BatesConfig Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
class TestBatesConfig:
    """Tests for BatesConfig dataclass."""

    def test_default_config(self):
        config = BatesConfig()
        assert config.prefix == ""
        assert config.suffix == ""
        assert config.start == 1
        assert config.width == 6
        assert config.separator == ""

    def test_custom_config(self):
        config = BatesConfig(
            prefix="PROD",
            suffix="CONF",
            start=100,
            width=8,
            separator="-"
        )
        assert config.prefix == "PROD"
        assert config.suffix == "CONF"
        assert config.start == 100
        assert config.width == 8
        assert config.separator == "-"

    def test_invalid_start_raises(self):
        with pytest.raises(StampValidationError, match="start must be >= 1"):
            BatesConfig(start=0)

    def test_invalid_width_too_small_raises(self):
        with pytest.raises(StampValidationError, match="width must be 1-12"):
            BatesConfig(width=0)

    def test_invalid_width_too_large_raises(self):
        with pytest.raises(StampValidationError, match="width must be 1-12"):
            BatesConfig(width=13)


# --- BatesAssigner Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
class TestBatesAssigner:
    """Tests for BatesAssigner."""

    def test_simple_numbering(self):
        config = BatesConfig(start=1, width=4)
        assigner = BatesAssigner(config)
        
        assert assigner.next() == "0001"
        assert assigner.next() == "0002"
        assert assigner.next() == "0003"

    def test_numbering_with_prefix(self):
        config = BatesConfig(prefix="PROD", start=1, width=4)
        assigner = BatesAssigner(config)
        
        assert assigner.next() == "PROD0001"
        assert assigner.next() == "PROD0002"

    def test_numbering_with_suffix(self):
        config = BatesConfig(suffix="CONF", start=1, width=4)
        assigner = BatesAssigner(config)
        
        assert assigner.next() == "0001CONF"
        assert assigner.next() == "0002CONF"

    def test_numbering_with_prefix_and_suffix(self):
        config = BatesConfig(prefix="PROD", suffix="CONF", start=1, width=4)
        assigner = BatesAssigner(config)
        
        assert assigner.next() == "PROD0001CONF"
        assert assigner.next() == "PROD0002CONF"

    def test_numbering_with_separator(self):
        config = BatesConfig(prefix="PROD", suffix="CONF", start=1, width=4, separator="-")
        assigner = BatesAssigner(config)
        
        assert assigner.next() == "PROD-0001-CONF"
        assert assigner.next() == "PROD-0002-CONF"

    def test_numbering_custom_start(self):
        config = BatesConfig(start=100, width=6)
        assigner = BatesAssigner(config)
        
        assert assigner.next() == "000100"
        assert assigner.next() == "000101"

    def test_numbering_custom_width(self):
        config = BatesConfig(start=1, width=8)
        assigner = BatesAssigner(config)
        
        assert assigner.next() == "00000001"

    def test_peek_does_not_increment(self):
        config = BatesConfig(start=1, width=4)
        assigner = BatesAssigner(config)
        
        assert assigner.peek() == "0001"
        assert assigner.peek() == "0001"  # Should not increment
        assert assigner.next() == "0001"
        assert assigner.peek() == "0002"

    def test_reset_to_start(self):
        config = BatesConfig(start=10, width=4)
        assigner = BatesAssigner(config)
        
        assigner.next()  # 0010
        assigner.next()  # 0011
        assigner.reset()
        assert assigner.next() == "0010"

    def test_reset_to_custom_value(self):
        config = BatesConfig(start=10, width=4)
        assigner = BatesAssigner(config)
        
        assigner.next()  # 0010
        assigner.reset(start=50)
        assert assigner.next() == "0050"


# --- BatesStampOperation Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
class TestBatesStampOperation:
    """Tests for BatesStampOperation."""

    def test_metadata(self):
        op = BatesStampOperation()
        metadata = op.get_metadata()
        
        assert metadata["name"] == "bates"
        assert metadata["version"] == "1.0.0"
        assert ".pdf" in metadata["supported_formats"]
        assert "start" in metadata["parameters"]

    def test_validate_config_success(self):
        op = BatesStampOperation()
        config = StampConfig(
            operation_name="bates",
            params={"start": 1, "width": 6}
        )
        errors = op.validate_config(config)
        assert errors == []

    def test_validate_config_invalid_start(self):
        op = BatesStampOperation()
        config = StampConfig(
            operation_name="bates",
            params={"start": 0}
        )
        errors = op.validate_config(config)
        assert len(errors) > 0
        assert "start must be >= 1" in errors[0]

    def test_validate_config_invalid_width(self):
        op = BatesStampOperation()
        config = StampConfig(
            operation_name="bates",
            params={"start": 1, "width": 15}
        )
        errors = op.validate_config(config)
        assert len(errors) > 0
        assert "width must be 1-12" in errors[0]

    def test_validate_config_invalid_font_size(self):
        op = BatesStampOperation()
        config = StampConfig(
            operation_name="bates",
            params={"start": 1, "font_size": -5}
        )
        errors = op.validate_config(config)
        assert len(errors) > 0
        assert "font_size must be positive" in errors[0]

    def test_validate_config_invalid_font_color(self):
        op = BatesStampOperation()
        config = StampConfig(
            operation_name="bates",
            params={"start": 1, "font_color": (1.5, 0.0, 0.0)}
        )
        errors = op.validate_config(config)
        assert len(errors) > 0
        assert "0.0-1.0" in errors[0]

    def test_execute_basic_stamping(self, sample_pdf_3pages, temp_dir):
        op = BatesStampOperation()
        output_path = temp_dir / "stamped.pdf"
        
        config = StampConfig(
            operation_name="bates",
            params={"start": 1, "width": 6}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert result.output_path == str(output_path)
        assert result.pages_stamped == 3
        assert len(result.stamp_values) == 3
        assert result.stamp_values[0] == "000001"
        assert result.stamp_values[1] == "000002"
        assert result.stamp_values[2] == "000003"
        assert output_path.exists()

    def test_execute_with_prefix_suffix(self, sample_pdf_3pages, temp_dir):
        op = BatesStampOperation()
        output_path = temp_dir / "stamped_prefix_suffix.pdf"
        
        config = StampConfig(
            operation_name="bates",
            params={
                "prefix": "PROD",
                "suffix": "CONF",
                "start": 100,
                "width": 4
            }
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert result.stamp_values[0] == "PROD0100CONF"
        assert result.stamp_values[1] == "PROD0101CONF"
        assert result.stamp_values[2] == "PROD0102CONF"

    def test_execute_with_separator(self, sample_pdf_3pages, temp_dir):
        op = BatesStampOperation()
        output_path = temp_dir / "stamped_separator.pdf"
        
        config = StampConfig(
            operation_name="bates",
            params={
                "prefix": "PROD",
                "start": 1,
                "width": 4,
                "separator": "-"
            }
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert result.stamp_values[0] == "PROD-0001"

    def test_execute_custom_placement(self, sample_pdf_3pages, temp_dir):
        op = BatesStampOperation()
        output_path = temp_dir / "stamped_top_right.pdf"
        
        config = StampConfig(
            operation_name="bates",
            placement=StampPlacement.TOP_RIGHT,
            params={"start": 1, "width": 6}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert output_path.exists()

    def test_execute_page_range_all(self, sample_pdf_3pages, temp_dir):
        op = BatesStampOperation()
        output_path = temp_dir / "stamped_all.pdf"
        
        config = StampConfig(
            operation_name="bates",
            params={"start": 1, "width": 6, "page_range": "all"}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert result.pages_stamped == 3

    def test_execute_page_range_subset(self, sample_pdf_3pages, temp_dir):
        op = BatesStampOperation()
        output_path = temp_dir / "stamped_subset.pdf"
        
        config = StampConfig(
            operation_name="bates",
            params={"start": 1, "width": 6, "page_range": "1,3"}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert result.pages_stamped == 2
        assert result.stamp_values[0] == "000001"
        assert result.stamp_values[1] == "000002"  # Sequential numbering

    def test_execute_page_range_range(self, sample_pdf_3pages, temp_dir):
        op = BatesStampOperation()
        output_path = temp_dir / "stamped_range.pdf"
        
        config = StampConfig(
            operation_name="bates",
            params={"start": 1, "width": 6, "page_range": "1-2"}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert result.pages_stamped == 2

    def test_execute_input_not_found(self, temp_dir):
        op = BatesStampOperation()
        output_path = temp_dir / "output.pdf"
        config = StampConfig(operation_name="bates", params={"start": 1})
        
        with pytest.raises(StampValidationError, match="Input file not found"):
            op.execute("/nonexistent/file.pdf", str(output_path), config)

    def test_execute_unsupported_format(self, temp_dir):
        op = BatesStampOperation()
        input_path = temp_dir / "test.txt"
        input_path.write_text("test")
        output_path = temp_dir / "output.pdf"
        config = StampConfig(operation_name="bates", params={"start": 1})
        
        result = op.execute(str(input_path), str(output_path), config)
        
        assert not result.success
        assert "Unsupported format" in result.error_message

    def test_execute_with_overlap_checking(self, sample_pdf_3pages, temp_dir):
        op = BatesStampOperation()
        output_path = temp_dir / "stamped_overlap_check.pdf"
        
        config = StampConfig(
            operation_name="bates",
            params={"start": 1, "width": 6},
            check_overlap=True
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        # Warnings may or may not be present depending on content
        assert isinstance(result.warnings, list)

    def test_supports_format(self):
        op = BatesStampOperation()
        assert op.supports_format(".pdf")
        assert op.supports_format(".PDF")
        assert not op.supports_format(".txt")

    def test_parse_page_range_all(self):
        op = BatesStampOperation()
        pages = op._parse_page_range("all", 5)
        assert pages == [0, 1, 2, 3, 4]

    def test_parse_page_range_single(self):
        op = BatesStampOperation()
        pages = op._parse_page_range("3", 5)
        assert pages == [2]  # 0-based

    def test_parse_page_range_multiple(self):
        op = BatesStampOperation()
        pages = op._parse_page_range("1,3,5", 5)
        assert pages == [0, 2, 4]

    def test_parse_page_range_range(self):
        op = BatesStampOperation()
        pages = op._parse_page_range("2-4", 5)
        assert pages == [1, 2, 3]

    def test_parse_page_range_mixed(self):
        op = BatesStampOperation()
        pages = op._parse_page_range("1,3-4,5", 5)
        assert pages == [0, 2, 3, 4]
