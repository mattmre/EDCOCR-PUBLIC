"""Unit tests for image transform operations.

Tests image operations (pdf_to_images, image_convert, images_to_pdf)
using synthetic images/PDFs created with PIL and PyMuPDF.

Run with: python -m pytest tests/test_transform_image_ops.py -v
"""

import tempfile
from pathlib import Path

import pytest

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import fitz
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

from ocr_distributed.transforms import (
    TransformConfig,
)

if _HAS_PIL and _HAS_FITZ:
    from ocr_distributed.transforms.image_ops import (
        ImageConvertOperation,
        ImagesToPDFOperation,
        PDFToImagesOperation,
    )


# --- Test Fixtures ---


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_png_image(temp_dir):
    """Create a sample PNG image."""
    if not _HAS_PIL:
        pytest.skip("Pillow not available")
    
    img_path = temp_dir / "sample.png"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(str(img_path))
    
    return img_path


@pytest.fixture
def sample_jpeg_image(temp_dir):
    """Create a sample JPEG image."""
    if not _HAS_PIL:
        pytest.skip("Pillow not available")
    
    img_path = temp_dir / "sample.jpg"
    img = Image.new("RGB", (100, 100), color=(0, 255, 0))
    img.save(str(img_path), quality=95)
    
    return img_path


@pytest.fixture
def sample_pdf_2pages(temp_dir):
    """Create a sample 2-page PDF."""
    if not _HAS_FITZ:
        pytest.skip("PyMuPDF not available")
    
    pdf_path = temp_dir / "sample_2pages.pdf"
    doc = fitz.open()
    
    for i in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 50), f"Page {i+1}", fontsize=24)
    
    doc.save(str(pdf_path))
    doc.close()
    
    return pdf_path


# --- PDF to Images Tests ---


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_to_images_all_pages(sample_pdf_2pages, temp_dir):
    """Test converting all PDF pages to images."""
    op = PDFToImagesOperation()
    output_dir = temp_dir / "images"
    output_dir.mkdir()
    
    config = TransformConfig(
        operation_name="pdf_to_images",
        params={"output_dir": str(output_dir), "format": "png"}
    )
    output_path = temp_dir / "dummy.png"
    
    result = op.execute(str(sample_pdf_2pages), str(output_path), config)
    
    assert result.success
    assert result.pages_processed == 2
    assert len(result.metadata["output_files"]) == 2
    
    # Verify images exist
    for img_path in result.metadata["output_files"]:
        assert Path(img_path).exists()
        assert Path(img_path).suffix == ".png"


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_to_images_specific_pages(sample_pdf_2pages, temp_dir):
    """Test converting specific pages."""
    op = PDFToImagesOperation()
    output_dir = temp_dir / "images"
    output_dir.mkdir()
    
    config = TransformConfig(
        operation_name="pdf_to_images",
        params={"output_dir": str(output_dir), "format": "jpg", "pages": 0}
    )
    output_path = temp_dir / "dummy.jpg"
    
    result = op.execute(str(sample_pdf_2pages), str(output_path), config)
    
    assert result.success
    assert result.pages_processed == 1
    assert len(result.metadata["output_files"]) == 1


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_to_images_custom_dpi(sample_pdf_2pages, temp_dir):
    """Test converting with custom DPI."""
    op = PDFToImagesOperation()
    output_dir = temp_dir / "images"
    output_dir.mkdir()
    
    config = TransformConfig(
        operation_name="pdf_to_images",
        params={"output_dir": str(output_dir), "format": "png", "dpi": 300}
    )
    output_path = temp_dir / "dummy.png"
    
    result = op.execute(str(sample_pdf_2pages), str(output_path), config)
    
    assert result.success
    assert result.metadata["dpi"] == 300


@pytest.mark.skipif(not _HAS_FITZ, reason="PyMuPDF not available")
def test_pdf_to_images_validation_error():
    """Test validation error for missing output_dir."""
    op = PDFToImagesOperation()
    config = TransformConfig(operation_name="pdf_to_images", params={})
    
    errors = op.validate_config(config)
    assert len(errors) > 0
    assert "output_dir" in errors[0]


# --- Image Convert Tests ---


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_image_convert_png_to_jpeg(sample_png_image, temp_dir):
    """Test converting PNG to JPEG."""
    op = ImageConvertOperation()
    config = TransformConfig(
        operation_name="image_convert",
        params={"format": "jpg", "quality": 90}
    )
    output_path = temp_dir / "converted.jpg"
    
    result = op.execute(str(sample_png_image), str(output_path), config)
    
    assert result.success
    assert result.output_path == str(output_path)
    assert output_path.exists()
    assert result.metadata["target_format"] == ".jpg"


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_image_convert_jpeg_to_png(sample_jpeg_image, temp_dir):
    """Test converting JPEG to PNG."""
    op = ImageConvertOperation()
    config = TransformConfig(
        operation_name="image_convert",
        params={"format": "png"}
    )
    output_path = temp_dir / "converted.png"
    
    result = op.execute(str(sample_jpeg_image), str(output_path), config)
    
    assert result.success
    assert result.output_path == str(output_path)
    assert output_path.exists()


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_image_convert_validation_error():
    """Test validation error for missing format."""
    op = ImageConvertOperation()
    config = TransformConfig(operation_name="image_convert", params={})
    
    errors = op.validate_config(config)
    assert len(errors) > 0
    assert "format" in errors[0]


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_image_convert_invalid_quality():
    """Test validation error for invalid quality."""
    op = ImageConvertOperation()
    config = TransformConfig(
        operation_name="image_convert",
        params={"format": "jpg", "quality": 150}
    )
    
    errors = op.validate_config(config)
    assert len(errors) > 0
    assert "quality" in errors[0]


# --- Images to PDF Tests ---


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_images_to_pdf_single_image(sample_png_image, temp_dir):
    """Test converting single image to PDF."""
    op = ImagesToPDFOperation()
    config = TransformConfig(
        operation_name="images_to_pdf",
        params={"input_files": [str(sample_png_image)]}
    )
    output_path = temp_dir / "output.pdf"
    
    result = op.execute("", str(output_path), config)
    
    assert result.success
    assert result.output_path == str(output_path)
    assert output_path.exists()
    assert result.pages_processed == 1


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_images_to_pdf_multiple_images(sample_png_image, sample_jpeg_image, temp_dir):
    """Test converting multiple images to PDF."""
    op = ImagesToPDFOperation()
    config = TransformConfig(
        operation_name="images_to_pdf",
        params={"input_files": [str(sample_png_image), str(sample_jpeg_image)]}
    )
    output_path = temp_dir / "output.pdf"
    
    result = op.execute("", str(output_path), config)
    
    assert result.success
    assert result.pages_processed == 2
    
    # Verify PDF has 2 pages if PyMuPDF available
    if _HAS_FITZ:
        doc = fitz.open(str(output_path))
        assert len(doc) == 2
        doc.close()


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_images_to_pdf_validation_error():
    """Test validation error for missing input_files."""
    op = ImagesToPDFOperation()
    config = TransformConfig(operation_name="images_to_pdf", params={})
    
    errors = op.validate_config(config)
    assert len(errors) > 0
    assert "input_files" in errors[0]


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_images_to_pdf_empty_list():
    """Test validation error for empty input list."""
    op = ImagesToPDFOperation()
    config = TransformConfig(
        operation_name="images_to_pdf",
        params={"input_files": []}
    )
    
    errors = op.validate_config(config)
    assert len(errors) > 0
    assert "at least 1" in errors[0]


# --- Metadata Tests ---


@pytest.mark.skipif(not (_HAS_PIL and _HAS_FITZ), reason="Pillow and PyMuPDF not available")
def test_all_operations_have_metadata():
    """Test that all operations have valid metadata."""
    operations = [
        PDFToImagesOperation(),
        ImageConvertOperation(),
        ImagesToPDFOperation(),
    ]
    
    for op in operations:
        metadata = op.get_metadata()
        assert "name" in metadata
        assert "description" in metadata
        assert "version" in metadata
        assert "supported_formats" in metadata
        assert "output_format" in metadata
        assert "parameters" in metadata


# --- Format Support Tests ---


@pytest.mark.skipif(not (_HAS_PIL and _HAS_FITZ), reason="Pillow and PyMuPDF not available")
def test_pdf_to_images_supports_pdf():
    """Test that pdf_to_images supports PDF format."""
    op = PDFToImagesOperation()
    assert op.supports_format(".pdf")
    assert not op.supports_format(".png")


@pytest.mark.skipif(not (_HAS_PIL and _HAS_FITZ), reason="Pillow and PyMuPDF not available")
def test_image_convert_supports_multiple_formats():
    """Test that image_convert supports multiple image formats."""
    op = ImageConvertOperation()
    assert op.supports_format(".png")
    assert op.supports_format(".jpg")
    assert op.supports_format(".jpeg")
    assert op.supports_format(".tiff")
    assert not op.supports_format(".pdf")


@pytest.mark.skipif(not (_HAS_PIL and _HAS_FITZ), reason="Pillow and PyMuPDF not available")
def test_images_to_pdf_supports_image_formats():
    """Test that images_to_pdf supports image formats."""
    op = ImagesToPDFOperation()
    assert op.supports_format(".png")
    assert op.supports_format(".jpg")
    assert not op.supports_format(".pdf")
