"""Image transform operations using PyMuPDF and PIL.

Provides concrete transform operations for image/PDF conversion:
- pdf_to_images: Convert PDF pages to individual images
- image_convert: Convert image format (PNG, JPEG, TIFF, etc.)
- images_to_pdf: Combine multiple images into a single PDF

All operations use atomic writes and preserve quality where possible.
"""

import logging
from pathlib import Path
from typing import Any

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    Image = None  # type: ignore

from .base import (
    _HAS_FITZ,
    TransformConfig,
    TransformError,
    TransformOperation,
    TransformResult,
    TransformValidationError,
    fitz,
)
from .utils import (
    atomic_write,
    get_file_extension,
    normalize_extension,
    parse_page_spec,
    validate_file_exists,
    validate_output_path,
)

logger = logging.getLogger(__name__)


# Supported image formats
IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}


# --- PDF to Images Operation ---


class PDFToImagesOperation(TransformOperation):
    """Convert PDF pages to individual images."""
    
    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "pdf_to_images",
            "description": "Convert PDF pages to individual images",
            "version": "1.0.0",
            "supported_formats": [".pdf"],
            "output_format": ".png",  # Default, can be overridden
            "parameters": {
                "output_dir": {
                    "type": "string",
                    "description": "Directory for output image files",
                    "required": True,
                },
                "format": {
                    "type": "string",
                    "description": "Output image format (png, jpg, tiff, default: png)",
                    "required": False,
                },
                "dpi": {
                    "type": "int",
                    "description": "DPI for rendering (default: 150)",
                    "required": False,
                },
                "pages": {
                    "type": "string|list|int",
                    "description": "Pages to convert (default: 'all')",
                    "required": False,
                },
            },
        }
    
    def validate_config(self, config: TransformConfig) -> list[str]:
        errors = []
        if "output_dir" not in config.params:
            errors.append("'output_dir' parameter is required")
        
        fmt = config.params.get("format", "png")
        if normalize_extension(fmt) not in IMAGE_FORMATS:
            errors.append(f"Unsupported image format: {fmt}")
        
        dpi = config.params.get("dpi", 150)
        if not isinstance(dpi, int) or dpi <= 0:
            errors.append(f"'dpi' must be positive integer, got {dpi}")
        
        return errors
    
    def execute(self, input_path: str, output_path: str, config: TransformConfig) -> TransformResult:
        if not _HAS_FITZ:
            raise TransformError("PyMuPDF (fitz) is required for PDF to image conversion")
        
        try:
            validate_file_exists(input_path)
            
            if get_file_extension(input_path) != ".pdf":
                raise TransformValidationError(f"Input must be PDF, got {input_path}")
            
            output_dir = Path(config.params["output_dir"])
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            
            fmt = config.params.get("format", "png")
            dpi = config.params.get("dpi", 150)
            ext = normalize_extension(fmt)
            
            doc = fitz.open(input_path)
            try:
                total_pages = len(doc)
                pages_to_convert = parse_page_spec(
                    config.params.get("pages", "all"),
                    total_pages
                )
                
                output_files = []
                input_name = Path(input_path).stem
                
                # Convert each page to image
                for page_idx in pages_to_convert:
                    page = doc[page_idx]
                    
                    # Render page to image at specified DPI
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Save image
                    output_file = output_dir / f"{input_name}_page{page_idx+1:03d}{ext}"
                    
                    def writer(path):
                        pix.save(path)
                    
                    atomic_write(writer, str(output_file))
                    output_files.append(str(output_file))
                
                # Use first output file as primary result
                return TransformResult(
                    success=True,
                    output_path=output_files[0] if output_files else str(output_dir),
                    pages_processed=len(pages_to_convert),
                    metadata={
                        "output_files": output_files,
                        "format": fmt,
                        "dpi": dpi,
                    },
                )
            finally:
                doc.close()
                
        except TransformValidationError:
            raise
        except Exception as exc:
            logger.error(f"PDF to images conversion failed: {exc}")
            return TransformResult(
                success=False,
                error_message=f"PDF to images conversion failed: {exc}",
            )


# --- Image Convert Operation ---


class ImageConvertOperation(TransformOperation):
    """Convert image from one format to another."""
    
    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "image_convert",
            "description": "Convert image format (PNG, JPEG, TIFF, etc.)",
            "version": "1.0.0",
            "supported_formats": list(IMAGE_FORMATS),
            "output_format": ".png",  # Default, can be overridden
            "parameters": {
                "format": {
                    "type": "string",
                    "description": "Target image format (png, jpg, tiff, etc.)",
                    "required": True,
                },
                "quality": {
                    "type": "int",
                    "description": "JPEG quality 1-100 (default: 95)",
                    "required": False,
                },
            },
        }
    
    def validate_config(self, config: TransformConfig) -> list[str]:
        errors = []
        fmt = config.params.get("format")
        if fmt is None:
            errors.append("'format' parameter is required")
        elif normalize_extension(fmt) not in IMAGE_FORMATS:
            errors.append(f"Unsupported image format: {fmt}")
        
        quality = config.params.get("quality", 95)
        if not isinstance(quality, int) or not (1 <= quality <= 100):
            errors.append(f"'quality' must be integer 1-100, got {quality}")
        
        return errors
    
    def execute(self, input_path: str, output_path: str, config: TransformConfig) -> TransformResult:
        if not _HAS_PIL:
            raise TransformError("Pillow (PIL) is required for image conversion")
        
        try:
            validate_file_exists(input_path)
            validate_output_path(output_path)
            
            if get_file_extension(input_path) not in IMAGE_FORMATS:
                raise TransformValidationError(f"Input must be an image, got {input_path}")
            
            fmt = config.params["format"]
            quality = config.params.get("quality", 95)
            
            # Open and convert image
            img = Image.open(input_path)
            
            # Convert format
            target_ext = normalize_extension(fmt)
            
            # Handle transparency for JPEG
            if target_ext in {".jpg", ".jpeg"} and img.mode in ("RGBA", "LA", "P"):
                # Convert to RGB for JPEG
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = background
            
            # Determine save parameters
            save_kwargs = {}
            if target_ext in {".jpg", ".jpeg"}:
                save_kwargs["quality"] = quality
                save_kwargs["optimize"] = True
            elif target_ext == ".png":
                save_kwargs["optimize"] = True
            
            # Atomic write
            def writer(path):
                img.save(path, **save_kwargs)
            
            atomic_write(writer, output_path)
            
            return TransformResult(
                success=True,
                output_path=output_path,
                pages_processed=1,
                metadata={
                    "source_format": get_file_extension(input_path),
                    "target_format": target_ext,
                    "quality": quality,
                    "size": img.size,
                },
            )
                
        except TransformValidationError:
            raise
        except Exception as exc:
            logger.error(f"Image conversion failed: {exc}")
            return TransformResult(
                success=False,
                error_message=f"Image conversion failed: {exc}",
            )


# --- Images to PDF Operation ---


class ImagesToPDFOperation(TransformOperation):
    """Combine multiple images into a single PDF."""
    
    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "images_to_pdf",
            "description": "Combine multiple images into a single PDF",
            "version": "1.0.0",
            "supported_formats": list(IMAGE_FORMATS),
            "output_format": ".pdf",
            "parameters": {
                "input_files": {
                    "type": "list",
                    "description": "List of image file paths to combine",
                    "required": True,
                },
            },
        }
    
    def validate_config(self, config: TransformConfig) -> list[str]:
        errors = []
        input_files = config.params.get("input_files")
        if input_files is None:
            errors.append("'input_files' parameter is required")
        elif not isinstance(input_files, list):
            errors.append(f"'input_files' must be a list, got {type(input_files)}")
        elif len(input_files) < 1:
            errors.append("'input_files' must contain at least 1 file")
        return errors
    
    def execute(self, input_path: str, output_path: str, config: TransformConfig) -> TransformResult:
        if not _HAS_PIL:
            raise TransformError("Pillow (PIL) is required for images to PDF conversion")
        
        try:
            validate_output_path(output_path)
            
            input_files = config.params["input_files"]
            
            # Validate all input files exist and are images
            images = []
            for file_path in input_files:
                validate_file_exists(file_path, f"input_files[{file_path}]")
                if get_file_extension(file_path) not in IMAGE_FORMATS:
                    raise TransformValidationError(f"All input files must be images, got {file_path}")
                
                # Open and convert to RGB if needed
                img = Image.open(file_path)
                if img.mode not in ("RGB", "L"):
                    # Convert RGBA, P, etc. to RGB
                    if img.mode == "RGBA" or img.mode == "LA":
                        # Create white background for transparency
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    else:
                        img = img.convert("RGB")
                images.append(img)
            
            if not images:
                raise TransformValidationError("No valid images to convert")
            
            # Save as PDF
            def writer(path):
                # First image is main, rest are appended
                first_img = images[0]
                if len(images) > 1:
                    first_img.save(
                        path,
                        save_all=True,
                        append_images=images[1:],
                        resolution=100.0,
                    )
                else:
                    first_img.save(path, resolution=100.0)
            
            atomic_write(writer, output_path)
            
            return TransformResult(
                success=True,
                output_path=output_path,
                pages_processed=len(images),
                metadata={
                    "input_files": input_files,
                    "total_images": len(images),
                },
            )
                
        except TransformValidationError:
            raise
        except Exception as exc:
            logger.error(f"Images to PDF conversion failed: {exc}")
            return TransformResult(
                success=False,
                error_message=f"Images to PDF conversion failed: {exc}",
            )
