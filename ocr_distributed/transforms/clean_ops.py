"""Image cleaning/preprocessing transform operations.

Provides transform operations that apply image preprocessing utilities
from preprocessing.py to improve image quality for OCR or archival.

- preprocessing: Apply preprocessing pipeline (deskew, denoise, enhance, binarize)

Reuses existing preprocessing.py utilities for consistency with OCR pipeline.
"""

import logging
from typing import Any

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    Image = None  # type: ignore

from .base import (
    TransformConfig,
    TransformError,
    TransformOperation,
    TransformResult,
    TransformValidationError,
)
from .utils import (
    atomic_write,
    get_file_extension,
    validate_file_exists,
    validate_output_path,
)

# Import preprocessing utilities
try:
    from preprocessing import _CV2_AVAILABLE, preprocess_for_ocr
except ImportError:
    preprocess_for_ocr = None
    _CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


# Supported image formats
IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}


# --- Preprocessing Operation ---


class PreprocessingOperation(TransformOperation):
    """Apply image preprocessing pipeline for OCR enhancement.
    
    Reuses preprocessing.py utilities to apply:
    - "none": No preprocessing
    - "standard": Deskew only (safe default)
    - "enhanced": Deskew + denoise + contrast enhancement
    - "aggressive": All steps including adaptive binarization
    """
    
    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "preprocessing",
            "description": "Apply image preprocessing pipeline (deskew, denoise, enhance, binarize)",
            "version": "1.0.0",
            "supported_formats": list(IMAGE_FORMATS),
            "output_format": ".png",  # Default, preserves input format
            "parameters": {
                "level": {
                    "type": "string",
                    "description": "Preprocessing level: none, standard, enhanced, aggressive (default: standard)",
                    "required": False,
                },
            },
        }
    
    def validate_config(self, config: TransformConfig) -> list[str]:
        errors = []
        level = config.params.get("level", "standard")
        if level not in ["none", "standard", "enhanced", "aggressive"]:
            errors.append(f"'level' must be none/standard/enhanced/aggressive, got {level}")
        return errors
    
    def execute(self, input_path: str, output_path: str, config: TransformConfig) -> TransformResult:
        if not _HAS_PIL:
            raise TransformError("Pillow (PIL) is required for image preprocessing")
        
        if preprocess_for_ocr is None:
            raise TransformError("preprocessing module not available")
        
        try:
            validate_file_exists(input_path)
            validate_output_path(output_path)
            
            if get_file_extension(input_path) not in IMAGE_FORMATS:
                raise TransformValidationError(f"Input must be an image, got {input_path}")
            
            level = config.params.get("level", "standard")
            
            # Open image
            img = Image.open(input_path)
            original_mode = img.mode
            original_size = img.size
            
            # Apply preprocessing
            if not _CV2_AVAILABLE:
                # OpenCV not available, return original
                processed_img = img
                warnings = ["OpenCV not available, preprocessing skipped"]
            else:
                processed_img = preprocess_for_ocr(img, level=level)
                warnings = []
            
            # Preserve original format if possible
            save_kwargs = {}
            ext = get_file_extension(input_path)
            if ext in {".jpg", ".jpeg"}:
                save_kwargs["quality"] = 95
                save_kwargs["optimize"] = True
            elif ext == ".png":
                save_kwargs["optimize"] = True
            
            # Atomic write
            def writer(path):
                processed_img.save(path, **save_kwargs)
            
            atomic_write(writer, output_path)
            
            return TransformResult(
                success=True,
                output_path=output_path,
                pages_processed=1,
                warnings=warnings,
                metadata={
                    "level": level,
                    "original_size": original_size,
                    "original_mode": original_mode,
                    "processed_size": processed_img.size,
                    "processed_mode": processed_img.mode,
                },
            )
                
        except TransformValidationError:
            raise
        except Exception as exc:
            logger.error(f"Image preprocessing failed: {exc}")
            return TransformResult(
                success=False,
                error_message=f"Image preprocessing failed: {exc}",
            )
