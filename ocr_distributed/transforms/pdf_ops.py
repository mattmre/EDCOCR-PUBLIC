"""PDF transform operations using PyMuPDF (fitz).

Provides concrete transform operations for PDF manipulation:
- extract: Extract specific pages
- delete: Remove specific pages
- rotate: Rotate pages by angle
- reorder: Reorder pages by custom sequence
- split: Split PDF into multiple files
- merge: Merge multiple PDFs
- insert: Insert pages from another PDF

All operations preserve metadata by default and use atomic writes.
"""

import logging
from pathlib import Path
from typing import Any

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
    parse_page_spec,
    validate_file_exists,
    validate_output_path,
)

logger = logging.getLogger(__name__)


# --- PDF Extract Operation ---


class PDFExtractOperation(TransformOperation):
    """Extract specific pages from a PDF."""
    
    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "pdf_extract",
            "description": "Extract specific pages from a PDF",
            "version": "1.0.0",
            "supported_formats": [".pdf"],
            "output_format": ".pdf",
            "parameters": {
                "pages": {
                    "type": "string|list|int",
                    "description": "Pages to extract (e.g., 'all', '0-2', [0,2,4])",
                    "required": True,
                }
            },
        }
    
    def validate_config(self, config: TransformConfig) -> list[str]:
        errors = []
        if "pages" not in config.params:
            errors.append("'pages' parameter is required")
        return errors
    
    def execute(self, input_path: str, output_path: str, config: TransformConfig) -> TransformResult:
        if not _HAS_FITZ:
            raise TransformError("PyMuPDF (fitz) is required for PDF operations")
        
        try:
            validate_file_exists(input_path)
            validate_output_path(output_path)
            
            if get_file_extension(input_path) != ".pdf":
                raise TransformValidationError(f"Input must be PDF, got {input_path}")
            
            doc = fitz.open(input_path)
            try:
                total_pages = len(doc)
                pages_to_extract = parse_page_spec(config.params["pages"], total_pages)
                
                if not pages_to_extract:
                    raise TransformValidationError("No pages selected for extraction")
                
                # Create new PDF with selected pages
                output_doc = fitz.open()
                for page_idx in pages_to_extract:
                    output_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
                
                # Preserve metadata if requested
                if config.preserve_metadata:
                    output_doc.set_metadata(doc.metadata)
                
                # Atomic write
                def writer(path):
                    output_doc.save(path)
                atomic_write(writer, output_path)
                
                output_doc.close()
                
                return TransformResult(
                    success=True,
                    output_path=output_path,
                    pages_processed=len(pages_to_extract),
                    metadata={"extracted_pages": pages_to_extract, "source_pages": total_pages},
                )
            finally:
                doc.close()
                
        except TransformValidationError:
            raise
        except Exception as exc:
            logger.error(f"PDF extract failed: {exc}")
            return TransformResult(
                success=False,
                error_message=f"PDF extraction failed: {exc}",
            )


# --- PDF Delete Operation ---


class PDFDeleteOperation(TransformOperation):
    """Delete specific pages from a PDF."""
    
    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "pdf_delete",
            "description": "Delete specific pages from a PDF",
            "version": "1.0.0",
            "supported_formats": [".pdf"],
            "output_format": ".pdf",
            "parameters": {
                "pages": {
                    "type": "string|list|int",
                    "description": "Pages to delete (e.g., '0-2', [0,2,4])",
                    "required": True,
                }
            },
        }
    
    def validate_config(self, config: TransformConfig) -> list[str]:
        errors = []
        if "pages" not in config.params:
            errors.append("'pages' parameter is required")
        return errors
    
    def execute(self, input_path: str, output_path: str, config: TransformConfig) -> TransformResult:
        if not _HAS_FITZ:
            raise TransformError("PyMuPDF (fitz) is required for PDF operations")
        
        try:
            validate_file_exists(input_path)
            validate_output_path(output_path)
            
            if get_file_extension(input_path) != ".pdf":
                raise TransformValidationError(f"Input must be PDF, got {input_path}")
            
            doc = fitz.open(input_path)
            try:
                total_pages = len(doc)
                pages_to_delete = parse_page_spec(config.params["pages"], total_pages)
                
                if not pages_to_delete:
                    raise TransformValidationError("No pages selected for deletion")
                
                if len(pages_to_delete) >= total_pages:
                    raise TransformValidationError("Cannot delete all pages from PDF")
                
                # Delete pages in reverse order to maintain indices
                for page_idx in sorted(pages_to_delete, reverse=True):
                    doc.delete_page(page_idx)
                
                # Atomic write
                def writer(path):
                    doc.save(path)
                atomic_write(writer, output_path)
                
                return TransformResult(
                    success=True,
                    output_path=output_path,
                    pages_processed=total_pages - len(pages_to_delete),
                    metadata={"deleted_pages": pages_to_delete, "remaining_pages": total_pages - len(pages_to_delete)},
                )
            finally:
                doc.close()
                
        except TransformValidationError:
            raise
        except Exception as exc:
            logger.error(f"PDF delete failed: {exc}")
            return TransformResult(
                success=False,
                error_message=f"PDF deletion failed: {exc}",
            )


# --- PDF Rotate Operation ---


class PDFRotateOperation(TransformOperation):
    """Rotate PDF pages by specified angle."""
    
    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "pdf_rotate",
            "description": "Rotate PDF pages by angle (90, 180, 270)",
            "version": "1.0.0",
            "supported_formats": [".pdf"],
            "output_format": ".pdf",
            "parameters": {
                "angle": {
                    "type": "int",
                    "description": "Rotation angle (90, 180, 270, -90, -180, -270)",
                    "required": True,
                },
                "pages": {
                    "type": "string|list|int",
                    "description": "Pages to rotate (default: 'all')",
                    "required": False,
                },
            },
        }
    
    def validate_config(self, config: TransformConfig) -> list[str]:
        errors = []
        angle = config.params.get("angle")
        if angle is None:
            errors.append("'angle' parameter is required")
        elif angle not in [90, 180, 270, -90, -180, -270]:
            errors.append(f"'angle' must be 90/180/270/-90/-180/-270, got {angle}")
        return errors
    
    def execute(self, input_path: str, output_path: str, config: TransformConfig) -> TransformResult:
        if not _HAS_FITZ:
            raise TransformError("PyMuPDF (fitz) is required for PDF operations")
        
        try:
            validate_file_exists(input_path)
            validate_output_path(output_path)
            
            if get_file_extension(input_path) != ".pdf":
                raise TransformValidationError(f"Input must be PDF, got {input_path}")
            
            doc = fitz.open(input_path)
            try:
                total_pages = len(doc)
                pages_to_rotate = parse_page_spec(
                    config.params.get("pages", "all"), 
                    total_pages
                )
                
                angle = config.params["angle"]
                
                # Rotate selected pages
                for page_idx in pages_to_rotate:
                    page = doc[page_idx]
                    page.set_rotation(angle)
                
                # Atomic write
                def writer(path):
                    doc.save(path)
                atomic_write(writer, output_path)
                
                return TransformResult(
                    success=True,
                    output_path=output_path,
                    pages_processed=len(pages_to_rotate),
                    metadata={"angle": angle, "rotated_pages": pages_to_rotate},
                )
            finally:
                doc.close()
                
        except TransformValidationError:
            raise
        except Exception as exc:
            logger.error(f"PDF rotate failed: {exc}")
            return TransformResult(
                success=False,
                error_message=f"PDF rotation failed: {exc}",
            )


# --- PDF Reorder Operation ---


class PDFReorderOperation(TransformOperation):
    """Reorder PDF pages by custom sequence."""
    
    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "pdf_reorder",
            "description": "Reorder PDF pages by custom sequence",
            "version": "1.0.0",
            "supported_formats": [".pdf"],
            "output_format": ".pdf",
            "parameters": {
                "order": {
                    "type": "list",
                    "description": "New page order as list of indices (0-indexed)",
                    "required": True,
                }
            },
        }
    
    def validate_config(self, config: TransformConfig) -> list[str]:
        errors = []
        order = config.params.get("order")
        if order is None:
            errors.append("'order' parameter is required")
        elif not isinstance(order, list):
            errors.append(f"'order' must be a list, got {type(order)}")
        elif not order:
            errors.append("'order' cannot be empty")
        return errors
    
    def execute(self, input_path: str, output_path: str, config: TransformConfig) -> TransformResult:
        if not _HAS_FITZ:
            raise TransformError("PyMuPDF (fitz) is required for PDF operations")
        
        try:
            validate_file_exists(input_path)
            validate_output_path(output_path)
            
            if get_file_extension(input_path) != ".pdf":
                raise TransformValidationError(f"Input must be PDF, got {input_path}")
            
            doc = fitz.open(input_path)
            try:
                total_pages = len(doc)
                order = config.params["order"]
                
                # Validate order
                if not all(isinstance(idx, int) for idx in order):
                    raise TransformValidationError("'order' must contain only integers")
                
                if not all(0 <= idx < total_pages for idx in order):
                    raise TransformValidationError(f"'order' indices must be in range [0, {total_pages-1}]")
                
                # Create new PDF with reordered pages
                output_doc = fitz.open()
                for page_idx in order:
                    output_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
                
                # Preserve metadata if requested
                if config.preserve_metadata:
                    output_doc.set_metadata(doc.metadata)
                
                # Atomic write
                def writer(path):
                    output_doc.save(path)
                atomic_write(writer, output_path)
                
                output_doc.close()
                
                return TransformResult(
                    success=True,
                    output_path=output_path,
                    pages_processed=len(order),
                    metadata={"order": order, "source_pages": total_pages},
                )
            finally:
                doc.close()
                
        except TransformValidationError:
            raise
        except Exception as exc:
            logger.error(f"PDF reorder failed: {exc}")
            return TransformResult(
                success=False,
                error_message=f"PDF reorder failed: {exc}",
            )


# --- PDF Split Operation ---


class PDFSplitOperation(TransformOperation):
    """Split PDF into multiple files."""
    
    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "pdf_split",
            "description": "Split PDF into multiple files (one page per file or by ranges)",
            "version": "1.0.0",
            "supported_formats": [".pdf"],
            "output_format": ".pdf",
            "parameters": {
                "output_dir": {
                    "type": "string",
                    "description": "Directory for output files",
                    "required": True,
                },
                "ranges": {
                    "type": "list",
                    "description": "List of page ranges [[0,2], [3,5]] or None for single pages",
                    "required": False,
                },
            },
        }
    
    def validate_config(self, config: TransformConfig) -> list[str]:
        errors = []
        if "output_dir" not in config.params:
            errors.append("'output_dir' parameter is required")
        return errors
    
    def execute(self, input_path: str, output_path: str, config: TransformConfig) -> TransformResult:
        if not _HAS_FITZ:
            raise TransformError("PyMuPDF (fitz) is required for PDF operations")
        
        try:
            validate_file_exists(input_path)
            
            if get_file_extension(input_path) != ".pdf":
                raise TransformValidationError(f"Input must be PDF, got {input_path}")
            
            output_dir = Path(config.params["output_dir"])
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            
            doc = fitz.open(input_path)
            try:
                total_pages = len(doc)
                ranges = config.params.get("ranges")
                
                # Generate ranges if not specified (one page per file)
                if ranges is None:
                    ranges = [[i, i] for i in range(total_pages)]
                
                # Validate ranges
                for r in ranges:
                    if not isinstance(r, list) or len(r) != 2:
                        raise TransformValidationError(f"Each range must be [start, end], got {r}")
                    start, end = r
                    if not (0 <= start <= end < total_pages):
                        raise TransformValidationError(f"Invalid range {r} for document with {total_pages} pages")
                
                output_files = []
                input_name = Path(input_path).stem
                
                for idx, (start, end) in enumerate(ranges):
                    output_file = output_dir / f"{input_name}_part{idx+1:03d}.pdf"
                    
                    # Create output doc with pages from range
                    output_doc = fitz.open()
                    for page_idx in range(start, end + 1):
                        output_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
                    
                    # Preserve metadata if requested
                    if config.preserve_metadata:
                        output_doc.set_metadata(doc.metadata)
                    
                    output_doc.save(str(output_file))
                    output_doc.close()
                    output_files.append(str(output_file))
                
                # Use first output file as primary result
                return TransformResult(
                    success=True,
                    output_path=output_files[0] if output_files else str(output_dir),
                    pages_processed=total_pages,
                    metadata={"output_files": output_files, "ranges": ranges},
                )
            finally:
                doc.close()
                
        except TransformValidationError:
            raise
        except Exception as exc:
            logger.error(f"PDF split failed: {exc}")
            return TransformResult(
                success=False,
                error_message=f"PDF split failed: {exc}",
            )


# --- PDF Merge Operation ---


class PDFMergeOperation(TransformOperation):
    """Merge multiple PDFs into one."""
    
    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "pdf_merge",
            "description": "Merge multiple PDFs into one",
            "version": "1.0.0",
            "supported_formats": [".pdf"],
            "output_format": ".pdf",
            "parameters": {
                "input_files": {
                    "type": "list",
                    "description": "List of PDF file paths to merge",
                    "required": True,
                }
            },
        }
    
    def validate_config(self, config: TransformConfig) -> list[str]:
        errors = []
        input_files = config.params.get("input_files")
        if input_files is None:
            errors.append("'input_files' parameter is required")
        elif not isinstance(input_files, list):
            errors.append(f"'input_files' must be a list, got {type(input_files)}")
        elif len(input_files) < 2:
            errors.append("'input_files' must contain at least 2 files")
        return errors
    
    def execute(self, input_path: str, output_path: str, config: TransformConfig) -> TransformResult:
        if not _HAS_FITZ:
            raise TransformError("PyMuPDF (fitz) is required for PDF operations")
        
        try:
            validate_output_path(output_path)
            
            input_files = config.params["input_files"]
            
            # Validate all input files exist
            for file_path in input_files:
                validate_file_exists(file_path, f"input_files[{file_path}]")
                if get_file_extension(file_path) != ".pdf":
                    raise TransformValidationError(f"All input files must be PDF, got {file_path}")
            
            # Create merged PDF
            output_doc = fitz.open()
            total_pages = 0
            
            for file_path in input_files:
                doc = fitz.open(file_path)
                try:
                    output_doc.insert_pdf(doc)
                    total_pages += len(doc)
                finally:
                    doc.close()
            
            # Atomic write
            def writer(path):
                output_doc.save(path)
            atomic_write(writer, output_path)
            
            output_doc.close()
            
            return TransformResult(
                success=True,
                output_path=output_path,
                pages_processed=total_pages,
                metadata={"input_files": input_files, "merged_pages": total_pages},
            )
                
        except TransformValidationError:
            raise
        except Exception as exc:
            logger.error(f"PDF merge failed: {exc}")
            return TransformResult(
                success=False,
                error_message=f"PDF merge failed: {exc}",
            )


# --- PDF Insert Operation ---


class PDFInsertOperation(TransformOperation):
    """Insert pages from another PDF at specified position."""
    
    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "pdf_insert",
            "description": "Insert pages from another PDF at specified position",
            "version": "1.0.0",
            "supported_formats": [".pdf"],
            "output_format": ".pdf",
            "parameters": {
                "source_pdf": {
                    "type": "string",
                    "description": "Path to PDF to insert from",
                    "required": True,
                },
                "position": {
                    "type": "int",
                    "description": "Position to insert at (0-indexed, -1 for end)",
                    "required": True,
                },
                "pages": {
                    "type": "string|list|int",
                    "description": "Pages from source to insert (default: 'all')",
                    "required": False,
                },
            },
        }
    
    def validate_config(self, config: TransformConfig) -> list[str]:
        errors = []
        if "source_pdf" not in config.params:
            errors.append("'source_pdf' parameter is required")
        if "position" not in config.params:
            errors.append("'position' parameter is required")
        return errors
    
    def execute(self, input_path: str, output_path: str, config: TransformConfig) -> TransformResult:
        if not _HAS_FITZ:
            raise TransformError("PyMuPDF (fitz) is required for PDF operations")
        
        try:
            validate_file_exists(input_path)
            validate_output_path(output_path)
            
            if get_file_extension(input_path) != ".pdf":
                raise TransformValidationError(f"Input must be PDF, got {input_path}")
            
            source_pdf = config.params["source_pdf"]
            validate_file_exists(source_pdf, "source_pdf")
            if get_file_extension(source_pdf) != ".pdf":
                raise TransformValidationError(f"source_pdf must be PDF, got {source_pdf}")
            
            position = config.params["position"]
            
            # Open both documents
            doc = fitz.open(input_path)
            source_doc = fitz.open(source_pdf)
            
            try:
                total_pages = len(doc)
                source_total = len(source_doc)
                
                # Parse pages to insert
                pages_to_insert = parse_page_spec(
                    config.params.get("pages", "all"),
                    source_total
                )
                
                # Validate position
                if position < 0:
                    position = total_pages  # -1 means append at end
                if position > total_pages:
                    raise TransformValidationError(f"position {position} out of range [0, {total_pages}]")
                
                # Insert pages at position
                for offset, page_idx in enumerate(pages_to_insert):
                    doc.insert_pdf(source_doc, from_page=page_idx, to_page=page_idx, start_at=position + offset)
                
                # Atomic write
                def writer(path):
                    doc.save(path)
                atomic_write(writer, output_path)
                
                return TransformResult(
                    success=True,
                    output_path=output_path,
                    pages_processed=total_pages + len(pages_to_insert),
                    metadata={
                        "inserted_pages": pages_to_insert,
                        "position": position,
                        "source_pdf": source_pdf,
                    },
                )
            finally:
                doc.close()
                source_doc.close()
                
        except TransformValidationError:
            raise
        except Exception as exc:
            logger.error(f"PDF insert failed: {exc}")
            return TransformResult(
                success=False,
                error_message=f"PDF insert failed: {exc}",
            )
