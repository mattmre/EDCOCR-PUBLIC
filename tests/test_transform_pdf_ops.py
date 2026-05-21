"""Unit tests for PDF transform operations.

Tests PDF operations (extract, delete, rotate, reorder, split, merge, insert)
using synthetic PDFs created with PyMuPDF. Keeps tests fast with tiny documents.

Run with: python -m pytest tests/test_transform_pdf_ops.py -v
"""

import tempfile
from pathlib import Path

import pytest

try:
    import fitz
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

from ocr_distributed.transforms import (
    TransformConfig,
    TransformValidationError,
)

if _HAS_FITZ:
    from ocr_distributed.transforms.pdf_ops import (
        PDFDeleteOperation,
        PDFExtractOperation,
        PDFInsertOperation,
        PDFMergeOperation,
        PDFReorderOperation,
        PDFRotateOperation,
        PDFSplitOperation,
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
    
    # Add 3 pages with text
    for i in range(3):
        page = doc.new_page(width=595, height=842)  # A4 size
        page.insert_text((50, 50), f"Page {i+1}", fontsize=24)
    
    doc.save(str(pdf_path))
    doc.close()
    
    return pdf_path


@pytest.fixture
def sample_pdf_5pages(temp_dir):
    """Create a sample 5-page PDF for testing."""
    if not _HAS_FITZ:
        pytest.skip("PyMuPDF not available")
    
    pdf_path = temp_dir / "sample_5pages.pdf"
    doc = fitz.open()
    
    for i in range(5):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 50), f"Page {i+1}", fontsize=24)
    
    doc.save(str(pdf_path))
    doc.close()
    
    return pdf_path


# --- PDF Extract Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_extract_single_page(sample_pdf_3pages, temp_dir):
    """Test extracting a single page."""
    op = PDFExtractOperation()
    config = TransformConfig(operation_name="pdf_extract", params={"pages": 0})
    output_path = temp_dir / "extracted.pdf"
    
    result = op.execute(str(sample_pdf_3pages), str(output_path), config)
    
    assert result.success
    assert result.output_path == str(output_path)
    assert result.pages_processed == 1
    assert output_path.exists()
    
    # Verify output has 1 page
    doc = fitz.open(str(output_path))
    assert len(doc) == 1
    doc.close()


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_extract_range(sample_pdf_5pages, temp_dir):
    """Test extracting a range of pages."""
    op = PDFExtractOperation()
    config = TransformConfig(operation_name="pdf_extract", params={"pages": "1-3"})
    output_path = temp_dir / "extracted.pdf"
    
    result = op.execute(str(sample_pdf_5pages), str(output_path), config)
    
    assert result.success
    assert result.pages_processed == 3
    
    doc = fitz.open(str(output_path))
    assert len(doc) == 3
    doc.close()


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_extract_list(sample_pdf_5pages, temp_dir):
    """Test extracting specific pages by list."""
    op = PDFExtractOperation()
    config = TransformConfig(operation_name="pdf_extract", params={"pages": [0, 2, 4]})
    output_path = temp_dir / "extracted.pdf"
    
    result = op.execute(str(sample_pdf_5pages), str(output_path), config)
    
    assert result.success
    assert result.pages_processed == 3
    
    doc = fitz.open(str(output_path))
    assert len(doc) == 3
    doc.close()


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_extract_validation_error(sample_pdf_3pages, temp_dir):
    """Test validation error for missing pages parameter."""
    op = PDFExtractOperation()
    config = TransformConfig(operation_name="pdf_extract", params={})
    
    errors = op.validate_config(config)
    assert len(errors) > 0
    assert "pages" in errors[0]


# --- PDF Delete Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_delete_single_page(sample_pdf_3pages, temp_dir):
    """Test deleting a single page."""
    op = PDFDeleteOperation()
    config = TransformConfig(operation_name="pdf_delete", params={"pages": 1})
    output_path = temp_dir / "deleted.pdf"
    
    result = op.execute(str(sample_pdf_3pages), str(output_path), config)
    
    assert result.success
    assert result.pages_processed == 2  # 3 - 1 = 2 remaining
    
    doc = fitz.open(str(output_path))
    assert len(doc) == 2
    doc.close()


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_delete_cannot_delete_all(sample_pdf_3pages, temp_dir):
    """Test that deleting all pages raises validation error."""
    op = PDFDeleteOperation()
    config = TransformConfig(operation_name="pdf_delete", params={"pages": "all"})
    output_path = temp_dir / "deleted.pdf"

    with pytest.raises(TransformValidationError, match="Cannot delete all pages"):
        op.execute(str(sample_pdf_3pages), str(output_path), config)


# --- PDF Rotate Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_rotate_all_pages(sample_pdf_3pages, temp_dir):
    """Test rotating all pages."""
    op = PDFRotateOperation()
    config = TransformConfig(operation_name="pdf_rotate", params={"angle": 90, "pages": "all"})
    output_path = temp_dir / "rotated.pdf"
    
    result = op.execute(str(sample_pdf_3pages), str(output_path), config)
    
    assert result.success
    assert result.pages_processed == 3
    assert output_path.exists()


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_rotate_specific_pages(sample_pdf_5pages, temp_dir):
    """Test rotating specific pages."""
    op = PDFRotateOperation()
    config = TransformConfig(operation_name="pdf_rotate", params={"angle": 180, "pages": [0, 2]})
    output_path = temp_dir / "rotated.pdf"
    
    result = op.execute(str(sample_pdf_5pages), str(output_path), config)
    
    assert result.success
    assert result.pages_processed == 2


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_rotate_invalid_angle(sample_pdf_3pages, temp_dir):
    """Test validation error for invalid angle."""
    op = PDFRotateOperation()
    config = TransformConfig(operation_name="pdf_rotate", params={"angle": 45})
    
    errors = op.validate_config(config)
    assert len(errors) > 0
    assert "90/180/270" in errors[0]


# --- PDF Reorder Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_reorder_pages(sample_pdf_3pages, temp_dir):
    """Test reordering pages."""
    op = PDFReorderOperation()
    config = TransformConfig(operation_name="pdf_reorder", params={"order": [2, 0, 1]})
    output_path = temp_dir / "reordered.pdf"
    
    result = op.execute(str(sample_pdf_3pages), str(output_path), config)
    
    assert result.success
    assert result.pages_processed == 3
    
    doc = fitz.open(str(output_path))
    assert len(doc) == 3
    doc.close()


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_reorder_validation_error(sample_pdf_3pages, temp_dir):
    """Test validation error for invalid order."""
    op = PDFReorderOperation()
    config = TransformConfig(operation_name="pdf_reorder", params={"order": []})
    
    errors = op.validate_config(config)
    assert len(errors) > 0


# --- PDF Split Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_split_single_pages(sample_pdf_3pages, temp_dir):
    """Test splitting into single-page files."""
    op = PDFSplitOperation()
    output_dir = temp_dir / "split_output"
    output_dir.mkdir()
    
    config = TransformConfig(operation_name="pdf_split", params={"output_dir": str(output_dir)})
    output_path = temp_dir / "dummy.pdf"
    
    result = op.execute(str(sample_pdf_3pages), str(output_path), config)
    
    assert result.success
    assert result.pages_processed == 3
    assert len(result.metadata["output_files"]) == 3


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_split_ranges(sample_pdf_5pages, temp_dir):
    """Test splitting with custom ranges."""
    op = PDFSplitOperation()
    output_dir = temp_dir / "split_output"
    output_dir.mkdir()
    
    config = TransformConfig(
        operation_name="pdf_split",
        params={"output_dir": str(output_dir), "ranges": [[0, 1], [2, 4]]}
    )
    output_path = temp_dir / "dummy.pdf"
    
    result = op.execute(str(sample_pdf_5pages), str(output_path), config)
    
    assert result.success
    assert len(result.metadata["output_files"]) == 2


# --- PDF Merge Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_merge_two_pdfs(sample_pdf_3pages, sample_pdf_5pages, temp_dir):
    """Test merging two PDFs."""
    op = PDFMergeOperation()
    config = TransformConfig(
        operation_name="pdf_merge",
        params={"input_files": [str(sample_pdf_3pages), str(sample_pdf_5pages)]}
    )
    output_path = temp_dir / "merged.pdf"
    
    result = op.execute("", str(output_path), config)
    
    assert result.success
    assert result.pages_processed == 8  # 3 + 5
    
    doc = fitz.open(str(output_path))
    assert len(doc) == 8
    doc.close()


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_merge_validation_error():
    """Test validation error for insufficient input files."""
    op = PDFMergeOperation()
    config = TransformConfig(operation_name="pdf_merge", params={"input_files": ["one.pdf"]})
    
    errors = op.validate_config(config)
    assert len(errors) > 0
    assert "at least 2" in errors[0]


# --- PDF Insert Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_insert_at_position(sample_pdf_3pages, sample_pdf_5pages, temp_dir):
    """Test inserting pages at specific position."""
    op = PDFInsertOperation()
    config = TransformConfig(
        operation_name="pdf_insert",
        params={"source_pdf": str(sample_pdf_5pages), "position": 1, "pages": 0}
    )
    output_path = temp_dir / "inserted.pdf"
    
    result = op.execute(str(sample_pdf_3pages), str(output_path), config)
    
    assert result.success
    assert result.pages_processed == 4  # 3 + 1 inserted
    
    doc = fitz.open(str(output_path))
    assert len(doc) == 4
    doc.close()


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_insert_at_end(sample_pdf_3pages, sample_pdf_5pages, temp_dir):
    """Test inserting pages at end (-1 position)."""
    op = PDFInsertOperation()
    config = TransformConfig(
        operation_name="pdf_insert",
        params={"source_pdf": str(sample_pdf_5pages), "position": -1, "pages": "0-1"}
    )
    output_path = temp_dir / "inserted.pdf"
    
    result = op.execute(str(sample_pdf_3pages), str(output_path), config)
    
    assert result.success
    assert result.pages_processed == 5  # 3 + 2 inserted
    
    doc = fitz.open(str(output_path))
    assert len(doc) == 5
    doc.close()


# --- Metadata Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_all_operations_have_metadata():
    """Test that all operations have valid metadata."""
    operations = [
        PDFExtractOperation(),
        PDFDeleteOperation(),
        PDFRotateOperation(),
        PDFReorderOperation(),
        PDFSplitOperation(),
        PDFMergeOperation(),
        PDFInsertOperation(),
    ]
    
    for op in operations:
        metadata = op.get_metadata()
        assert "name" in metadata
        assert "description" in metadata
        assert "version" in metadata
        assert "supported_formats" in metadata
        assert "output_format" in metadata
        assert "parameters" in metadata
