"""Built-in transform operations registration.

Provides a convenience function to register all built-in transform operations
with the transform registry. This includes PDF operations, image operations,
and preprocessing operations.
"""

import logging
from typing import Optional

from .registry import TransformRegistry, get_transform_registry

logger = logging.getLogger(__name__)


def register_builtin_transforms(registry: Optional[TransformRegistry] = None) -> TransformRegistry:
    """Register all built-in transform operations.
    
    Registers the following operations:
    
    PDF Operations (pdf_ops.py):
    - pdf_extract: Extract specific pages from PDF
    - pdf_delete: Delete specific pages from PDF
    - pdf_rotate: Rotate PDF pages
    - pdf_reorder: Reorder PDF pages
    - pdf_split: Split PDF into multiple files
    - pdf_merge: Merge multiple PDFs
    - pdf_insert: Insert pages from another PDF
    
    Image Operations (image_ops.py):
    - pdf_to_images: Convert PDF pages to images
    - image_convert: Convert image format
    - images_to_pdf: Combine images into PDF
    
    Cleaning Operations (clean_ops.py):
    - preprocessing: Apply image preprocessing pipeline
    
    Args:
        registry: Transform registry to use (defaults to global registry)
        
    Returns:
        The registry with built-in operations registered
        
    Note:
        Operations are only registered if their dependencies are available.
        Missing dependencies (PyMuPDF, Pillow, OpenCV) will result in warnings
        but not errors.
    """
    if registry is None:
        registry = get_transform_registry()
    
    # Track registration success/failure
    registered = []
    skipped = []
    
    # --- PDF Operations ---
    try:
        from .pdf_ops import (
            PDFDeleteOperation,
            PDFExtractOperation,
            PDFInsertOperation,
            PDFMergeOperation,
            PDFReorderOperation,
            PDFRotateOperation,
            PDFSplitOperation,
        )
        
        pdf_ops = [
            PDFExtractOperation(),
            PDFDeleteOperation(),
            PDFRotateOperation(),
            PDFReorderOperation(),
            PDFSplitOperation(),
            PDFMergeOperation(),
            PDFInsertOperation(),
        ]
        
        for op in pdf_ops:
            try:
                registry.register(op)
                registered.append(op.get_metadata()["name"])
            except Exception as exc:
                skipped.append((op.get_metadata()["name"], str(exc)))
                logger.warning(f"Failed to register {op.get_metadata()['name']}: {exc}")
    
    except ImportError as exc:
        logger.warning(f"PDF operations not available (PyMuPDF missing): {exc}")
    
    # --- Image Operations ---
    try:
        from .image_ops import (
            ImageConvertOperation,
            ImagesToPDFOperation,
            PDFToImagesOperation,
        )
        
        image_ops = [
            PDFToImagesOperation(),
            ImageConvertOperation(),
            ImagesToPDFOperation(),
        ]
        
        for op in image_ops:
            try:
                registry.register(op)
                registered.append(op.get_metadata()["name"])
            except Exception as exc:
                skipped.append((op.get_metadata()["name"], str(exc)))
                logger.warning(f"Failed to register {op.get_metadata()['name']}: {exc}")
    
    except ImportError as exc:
        logger.warning(f"Image operations not available (Pillow/PyMuPDF missing): {exc}")
    
    # --- Cleaning/Preprocessing Operations ---
    try:
        from .clean_ops import PreprocessingOperation
        
        clean_ops = [
            PreprocessingOperation(),
        ]
        
        for op in clean_ops:
            try:
                registry.register(op)
                registered.append(op.get_metadata()["name"])
            except Exception as exc:
                skipped.append((op.get_metadata()["name"], str(exc)))
                logger.warning(f"Failed to register {op.get_metadata()['name']}: {exc}")
    
    except ImportError as exc:
        logger.warning(f"Preprocessing operations not available (Pillow missing): {exc}")
    
    # Log summary
    if registered:
        logger.info(f"Registered {len(registered)} built-in transform operations: {', '.join(registered)}")
    if skipped:
        logger.warning(f"Skipped {len(skipped)} operations due to errors: {[name for name, _ in skipped]}")
    
    return registry
