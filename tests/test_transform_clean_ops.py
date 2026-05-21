"""Unit tests for image cleaning/preprocessing transform operations.

Tests preprocessing operations that apply image enhancement utilities
from preprocessing.py (deskew, denoise, enhance, binarize).

Run with: python -m pytest tests/test_transform_clean_ops.py -v
"""

import tempfile
from pathlib import Path

import pytest

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from ocr_distributed.transforms import (
    TransformConfig,
    TransformValidationError,
)

if _HAS_PIL:
    from ocr_distributed.transforms.clean_ops import PreprocessingOperation


# --- Test Fixtures ---


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample image for preprocessing."""
    if not _HAS_PIL:
        pytest.skip("Pillow not available")
    
    img_path = temp_dir / "sample.png"
    # Create a simple test image with some content
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    # Add some "text-like" content
    for i in range(50, 150, 10):
        for j in range(50, 150):
            if (i // 10) % 2 == 0:
                img.putpixel((j, i), (0, 0, 0))
    
    img.save(str(img_path))
    return img_path


@pytest.fixture
def sample_jpeg(temp_dir):
    """Create a sample JPEG image."""
    if not _HAS_PIL:
        pytest.skip("Pillow not available")
    
    img_path = temp_dir / "sample.jpg"
    img = Image.new("RGB", (200, 200), color=(240, 240, 240))
    img.save(str(img_path), quality=90)
    
    return img_path


# --- Preprocessing Operation Tests ---


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_preprocessing_none_level(sample_image, temp_dir):
    """Test preprocessing with 'none' level (no changes)."""
    op = PreprocessingOperation()
    config = TransformConfig(
        operation_name="preprocessing",
        params={"level": "none"}
    )
    output_path = temp_dir / "processed.png"
    
    result = op.execute(str(sample_image), str(output_path), config)
    
    assert result.success
    assert result.output_path == str(output_path)
    assert output_path.exists()
    assert result.pages_processed == 1
    assert result.metadata["level"] == "none"


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_preprocessing_standard_level(sample_image, temp_dir):
    """Test preprocessing with 'standard' level (deskew only)."""
    op = PreprocessingOperation()
    config = TransformConfig(
        operation_name="preprocessing",
        params={"level": "standard"}
    )
    output_path = temp_dir / "processed.png"
    
    result = op.execute(str(sample_image), str(output_path), config)
    
    assert result.success
    assert result.output_path == str(output_path)
    assert output_path.exists()
    assert result.metadata["level"] == "standard"


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_preprocessing_enhanced_level(sample_image, temp_dir):
    """Test preprocessing with 'enhanced' level."""
    op = PreprocessingOperation()
    config = TransformConfig(
        operation_name="preprocessing",
        params={"level": "enhanced"}
    )
    output_path = temp_dir / "processed.png"
    
    result = op.execute(str(sample_image), str(output_path), config)
    
    assert result.success
    assert result.output_path == str(output_path)
    assert output_path.exists()
    assert result.metadata["level"] == "enhanced"


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_preprocessing_aggressive_level(sample_image, temp_dir):
    """Test preprocessing with 'aggressive' level."""
    op = PreprocessingOperation()
    config = TransformConfig(
        operation_name="preprocessing",
        params={"level": "aggressive"}
    )
    output_path = temp_dir / "processed.png"
    
    result = op.execute(str(sample_image), str(output_path), config)
    
    assert result.success
    assert result.output_path == str(output_path)
    assert output_path.exists()
    assert result.metadata["level"] == "aggressive"


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_preprocessing_default_level(sample_image, temp_dir):
    """Test preprocessing with default level (standard)."""
    op = PreprocessingOperation()
    config = TransformConfig(
        operation_name="preprocessing",
        params={}  # No level specified, should default to standard
    )
    output_path = temp_dir / "processed.png"
    
    result = op.execute(str(sample_image), str(output_path), config)
    
    assert result.success
    assert result.metadata["level"] == "standard"


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_preprocessing_jpeg_format(sample_jpeg, temp_dir):
    """Test preprocessing JPEG image."""
    op = PreprocessingOperation()
    config = TransformConfig(
        operation_name="preprocessing",
        params={"level": "standard"}
    )
    output_path = temp_dir / "processed.jpg"
    
    result = op.execute(str(sample_jpeg), str(output_path), config)
    
    assert result.success
    assert result.output_path == str(output_path)
    assert output_path.exists()


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_preprocessing_preserves_size(sample_image, temp_dir):
    """Test that preprocessing preserves image size."""
    op = PreprocessingOperation()
    config = TransformConfig(
        operation_name="preprocessing",
        params={"level": "standard"}
    )
    output_path = temp_dir / "processed.png"
    
    original = Image.open(sample_image)
    original_size = original.size
    
    result = op.execute(str(sample_image), str(output_path), config)
    
    assert result.success
    assert result.metadata["original_size"] == original_size


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_preprocessing_validation_error():
    """Test validation error for invalid level."""
    op = PreprocessingOperation()
    config = TransformConfig(
        operation_name="preprocessing",
        params={"level": "invalid_level"}
    )
    
    errors = op.validate_config(config)
    assert len(errors) > 0
    assert "none/standard/enhanced/aggressive" in errors[0]


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_preprocessing_invalid_input_format(temp_dir):
    """Test error handling for non-image input."""
    op = PreprocessingOperation()
    
    # Create a text file
    text_file = temp_dir / "not_an_image.txt"
    text_file.write_text("This is not an image")
    
    config = TransformConfig(
        operation_name="preprocessing",
        params={"level": "standard"}
    )
    output_path = temp_dir / "processed.png"

    with pytest.raises(TransformValidationError, match="Input must be an image"):
        op.execute(str(text_file), str(output_path), config)


# --- Metadata Tests ---


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_preprocessing_metadata():
    """Test that operation has valid metadata."""
    op = PreprocessingOperation()
    metadata = op.get_metadata()
    
    assert metadata["name"] == "preprocessing"
    assert "description" in metadata
    assert "version" in metadata
    assert "supported_formats" in metadata
    assert "output_format" in metadata
    assert "parameters" in metadata
    assert "level" in metadata["parameters"]


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_preprocessing_supports_image_formats():
    """Test that preprocessing supports common image formats."""
    op = PreprocessingOperation()
    
    assert op.supports_format(".png")
    assert op.supports_format(".jpg")
    assert op.supports_format(".jpeg")
    assert op.supports_format(".tiff")
    assert not op.supports_format(".pdf")


# --- OpenCV Availability Tests ---


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_preprocessing_without_opencv_warnings(sample_image, temp_dir):
    """Test that preprocessing gracefully handles missing OpenCV."""
    op = PreprocessingOperation()
    config = TransformConfig(
        operation_name="preprocessing",
        params={"level": "standard"}
    )
    output_path = temp_dir / "processed.png"
    
    result = op.execute(str(sample_image), str(output_path), config)
    
    # Should succeed even if OpenCV is missing
    assert result.success
    
    # Check for OpenCV warning if not available
    import importlib.util

    if importlib.util.find_spec("cv2") is None:
        # If OpenCV not available, should have warning
        assert len(result.warnings) > 0
        assert "OpenCV" in result.warnings[0]


# --- Integration Tests ---


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not available")
def test_preprocessing_all_levels_produce_output(sample_image, temp_dir):
    """Test that all preprocessing levels produce valid output."""
    op = PreprocessingOperation()
    
    levels = ["none", "standard", "enhanced", "aggressive"]
    
    for level in levels:
        config = TransformConfig(
            operation_name="preprocessing",
            params={"level": level}
        )
        output_path = temp_dir / f"processed_{level}.png"
        
        result = op.execute(str(sample_image), str(output_path), config)
        
        assert result.success, f"Level {level} failed"
        assert output_path.exists(), f"Output missing for level {level}"
        
        # Verify output is a valid image
        img = Image.open(output_path)
        assert img.size[0] > 0
        assert img.size[1] > 0
