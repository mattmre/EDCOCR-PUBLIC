"""Unit tests for confidentiality designation stamp operations.

Tests DesignationStampOperation with synthetic PDFs.

Run with: python -m pytest tests/test_stamp_designation.py -v
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
        STANDARD_DESIGNATIONS,
        DesignationStampOperation,
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


# --- Standard Designations Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
class TestStandardDesignations:
    """Tests for standard designation constants."""

    def test_standard_designations_defined(self):
        assert "CONFIDENTIAL" in STANDARD_DESIGNATIONS
        assert "HIGHLY CONFIDENTIAL" in STANDARD_DESIGNATIONS
        assert "ATTORNEYS' EYES ONLY" in STANDARD_DESIGNATIONS

    def test_standard_designations_count(self):
        assert len(STANDARD_DESIGNATIONS) == 3


# --- DesignationStampOperation Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
class TestDesignationStampOperation:
    """Tests for DesignationStampOperation."""

    def test_metadata(self):
        op = DesignationStampOperation()
        metadata = op.get_metadata()
        
        assert metadata["name"] == "designation"
        assert metadata["version"] == "1.0.0"
        assert ".pdf" in metadata["supported_formats"]
        assert "text" in metadata["parameters"]

    def test_validate_config_success_standard(self):
        op = DesignationStampOperation()
        config = StampConfig(
            operation_name="designation",
            params={"text": "CONFIDENTIAL"}
        )
        errors = op.validate_config(config)
        assert errors == []

    def test_validate_config_success_all_standard(self):
        op = DesignationStampOperation()
        for designation in STANDARD_DESIGNATIONS:
            config = StampConfig(
                operation_name="designation",
                params={"text": designation}
            )
            errors = op.validate_config(config)
            assert errors == [], f"Failed for {designation}"

    def test_validate_config_missing_text(self):
        op = DesignationStampOperation()
        config = StampConfig(
            operation_name="designation",
            params={}
        )
        errors = op.validate_config(config)
        assert len(errors) > 0
        assert "text parameter is required" in errors[0]

    def test_validate_config_invalid_designation_without_custom(self):
        op = DesignationStampOperation()
        config = StampConfig(
            operation_name="designation",
            params={"text": "SECRET"}
        )
        errors = op.validate_config(config)
        assert len(errors) > 0
        assert "Invalid designation" in errors[0]

    def test_validate_config_custom_allowed(self):
        op = DesignationStampOperation()
        config = StampConfig(
            operation_name="designation",
            params={"text": "SECRET", "allow_custom": True}
        )
        errors = op.validate_config(config)
        assert errors == []

    def test_validate_config_invalid_font_size(self):
        op = DesignationStampOperation()
        config = StampConfig(
            operation_name="designation",
            params={"text": "CONFIDENTIAL", "font_size": -10}
        )
        errors = op.validate_config(config)
        assert len(errors) > 0
        assert "font_size must be positive" in errors[0]

    def test_validate_config_invalid_font_color(self):
        op = DesignationStampOperation()
        config = StampConfig(
            operation_name="designation",
            params={"text": "CONFIDENTIAL", "font_color": (1.5, 0.0, 0.0)}
        )
        errors = op.validate_config(config)
        assert len(errors) > 0
        assert "0.0-1.0" in errors[0]

    def test_validate_config_invalid_page_range_type(self):
        op = DesignationStampOperation()
        config = StampConfig(
            operation_name="designation",
            params={"text": "CONFIDENTIAL", "page_range": 123}
        )
        errors = op.validate_config(config)
        assert len(errors) > 0
        assert "page_range must be string" in errors[0]

    def test_execute_basic_stamping(self, sample_pdf_3pages, temp_dir):
        op = DesignationStampOperation()
        output_path = temp_dir / "stamped_confidential.pdf"
        
        config = StampConfig(
            operation_name="designation",
            params={"text": "CONFIDENTIAL"}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert result.output_path == str(output_path)
        assert result.pages_stamped == 3
        assert len(result.stamp_values) == 3
        assert all(v == "CONFIDENTIAL" for v in result.stamp_values)
        assert output_path.exists()

    def test_execute_highly_confidential(self, sample_pdf_3pages, temp_dir):
        op = DesignationStampOperation()
        output_path = temp_dir / "stamped_highly_conf.pdf"
        
        config = StampConfig(
            operation_name="designation",
            params={"text": "HIGHLY CONFIDENTIAL"}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert all(v == "HIGHLY CONFIDENTIAL" for v in result.stamp_values)

    def test_execute_attorneys_eyes_only(self, sample_pdf_3pages, temp_dir):
        op = DesignationStampOperation()
        output_path = temp_dir / "stamped_aeo.pdf"
        
        config = StampConfig(
            operation_name="designation",
            params={"text": "ATTORNEYS' EYES ONLY"}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert all(v == "ATTORNEYS' EYES ONLY" for v in result.stamp_values)

    def test_execute_custom_designation(self, sample_pdf_3pages, temp_dir):
        op = DesignationStampOperation()
        output_path = temp_dir / "stamped_custom.pdf"
        
        config = StampConfig(
            operation_name="designation",
            params={"text": "INTERNAL USE ONLY", "allow_custom": True}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert all(v == "INTERNAL USE ONLY" for v in result.stamp_values)

    def test_execute_custom_font_size(self, sample_pdf_3pages, temp_dir):
        op = DesignationStampOperation()
        output_path = temp_dir / "stamped_large.pdf"
        
        config = StampConfig(
            operation_name="designation",
            params={"text": "CONFIDENTIAL", "font_size": 18.0}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert result.metadata["font_size"] == 18.0

    def test_execute_custom_font_color(self, sample_pdf_3pages, temp_dir):
        op = DesignationStampOperation()
        output_path = temp_dir / "stamped_blue.pdf"
        
        config = StampConfig(
            operation_name="designation",
            params={
                "text": "CONFIDENTIAL",
                "font_color": (0.0, 0.0, 1.0)  # Blue
            }
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert result.metadata["font_color"] == (0.0, 0.0, 1.0)

    def test_execute_custom_placement(self, sample_pdf_3pages, temp_dir):
        op = DesignationStampOperation()
        output_path = temp_dir / "stamped_top_center.pdf"
        
        config = StampConfig(
            operation_name="designation",
            placement=StampPlacement.TOP_CENTER,
            params={"text": "CONFIDENTIAL"}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert output_path.exists()

    def test_execute_page_range_all(self, sample_pdf_3pages, temp_dir):
        op = DesignationStampOperation()
        output_path = temp_dir / "stamped_all.pdf"
        
        config = StampConfig(
            operation_name="designation",
            params={"text": "CONFIDENTIAL", "page_range": "all"}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert result.pages_stamped == 3

    def test_execute_page_range_subset(self, sample_pdf_3pages, temp_dir):
        op = DesignationStampOperation()
        output_path = temp_dir / "stamped_subset.pdf"
        
        config = StampConfig(
            operation_name="designation",
            params={"text": "CONFIDENTIAL", "page_range": "1,3"}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert result.pages_stamped == 2

    def test_execute_page_range_range(self, sample_pdf_3pages, temp_dir):
        op = DesignationStampOperation()
        output_path = temp_dir / "stamped_range.pdf"
        
        config = StampConfig(
            operation_name="designation",
            params={"text": "CONFIDENTIAL", "page_range": "1-2"}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert result.pages_stamped == 2

    def test_execute_input_not_found(self, temp_dir):
        op = DesignationStampOperation()
        output_path = temp_dir / "output.pdf"
        config = StampConfig(
            operation_name="designation",
            params={"text": "CONFIDENTIAL"}
        )
        
        with pytest.raises(StampValidationError, match="Input file not found"):
            op.execute("/nonexistent/file.pdf", str(output_path), config)

    def test_execute_unsupported_format(self, temp_dir):
        op = DesignationStampOperation()
        input_path = temp_dir / "test.txt"
        input_path.write_text("test")
        output_path = temp_dir / "output.pdf"
        config = StampConfig(
            operation_name="designation",
            params={"text": "CONFIDENTIAL"}
        )
        
        result = op.execute(str(input_path), str(output_path), config)
        
        assert not result.success
        assert "Unsupported format" in result.error_message

    def test_execute_with_overlap_checking(self, sample_pdf_3pages, temp_dir):
        op = DesignationStampOperation()
        output_path = temp_dir / "stamped_overlap_check.pdf"
        
        config = StampConfig(
            operation_name="designation",
            params={"text": "CONFIDENTIAL"},
            check_overlap=True
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        # Warnings may or may not be present depending on content
        assert isinstance(result.warnings, list)

    def test_supports_format(self):
        op = DesignationStampOperation()
        assert op.supports_format(".pdf")
        assert op.supports_format(".PDF")
        assert not op.supports_format(".txt")

    def test_parse_page_range_all(self):
        op = DesignationStampOperation()
        pages = op._parse_page_range("all", 5)
        assert pages == [0, 1, 2, 3, 4]

    def test_parse_page_range_single(self):
        op = DesignationStampOperation()
        pages = op._parse_page_range("3", 5)
        assert pages == [2]  # 0-based

    def test_parse_page_range_multiple(self):
        op = DesignationStampOperation()
        pages = op._parse_page_range("1,3,5", 5)
        assert pages == [0, 2, 4]

    def test_parse_page_range_range(self):
        op = DesignationStampOperation()
        pages = op._parse_page_range("2-4", 5)
        assert pages == [1, 2, 3]

    def test_parse_page_range_mixed(self):
        op = DesignationStampOperation()
        pages = op._parse_page_range("1,3-4,5", 5)
        assert pages == [0, 2, 3, 4]

    def test_execute_custom_font_name(self, sample_pdf_3pages, temp_dir):
        op = DesignationStampOperation()
        output_path = temp_dir / "stamped_times.pdf"
        
        config = StampConfig(
            operation_name="designation",
            params={"text": "CONFIDENTIAL", "font_name": "times-bold"}
        )
        
        result = op.execute(str(sample_pdf_3pages), str(output_path), config)
        
        assert result.success
        assert result.metadata["font_name"] == "times-bold"
