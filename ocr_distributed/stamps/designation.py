"""Confidentiality designation stamp operations.

Provides stamp operations for applying confidentiality designations to PDFs
(e.g., CONFIDENTIAL, HIGHLY CONFIDENTIAL, ATTORNEYS' EYES ONLY).
"""

import os
from typing import Any

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False
    fitz = None  # type: ignore

from .base import (
    StampConfig,
    StampOperation,
    StampResult,
    StampValidationError,
)
from .zone import ZoneDetector

# Standard confidentiality designations
STANDARD_DESIGNATIONS = [
    "CONFIDENTIAL",
    "HIGHLY CONFIDENTIAL",
    "ATTORNEYS' EYES ONLY",
]


class DesignationStampOperation(StampOperation):
    """Stamp operation for applying confidentiality designations to PDFs."""

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "designation",
            "description": "Apply confidentiality designation to PDF documents",
            "version": "1.0.0",
            "supported_formats": [".pdf"],
            "parameters": {
                "text": {
                    "type": "str",
                    "description": "Designation text (standard or custom)",
                    "required": True,
                },
                "allow_custom": {
                    "type": "bool",
                    "description": "Allow custom designation text beyond standard set",
                    "required": False,
                    "default": False,
                },
                "font_size": {
                    "type": "float",
                    "description": "Font size in points",
                    "required": False,
                    "default": 12.0,
                },
                "font_color": {
                    "type": "tuple[float, float, float]",
                    "description": "RGB color (0.0-1.0)",
                    "required": False,
                    "default": (1.0, 0.0, 0.0),
                },
                "font_name": {
                    "type": "str",
                    "description": "Font name (helv, times, cour)",
                    "required": False,
                    "default": "helv",
                },
                "page_range": {
                    "type": "str",
                    "description": "Page range (e.g., '1-5,7,9-12' or 'all')",
                    "required": False,
                    "default": "all",
                },
            },
        }

    def validate_config(self, config: StampConfig) -> list[str]:
        errors = []
        
        # Validate designation text
        text = config.params.get("text")
        if not text:
            errors.append("text parameter is required")
        elif not isinstance(text, str):
            errors.append(f"text must be string, got {type(text)}")
        else:
            allow_custom = config.params.get("allow_custom", False)
            if not allow_custom and text not in STANDARD_DESIGNATIONS:
                errors.append(
                    f"Invalid designation: {text}. "
                    f"Must be one of {STANDARD_DESIGNATIONS} or set allow_custom=True"
                )
        
        # Validate font size
        font_size = config.params.get("font_size", 12.0)
        if not isinstance(font_size, (int, float)) or font_size <= 0:
            errors.append(f"font_size must be positive number, got {font_size}")
        
        # Validate font color
        font_color = config.params.get("font_color", (1.0, 0.0, 0.0))
        if not isinstance(font_color, (list, tuple)) or len(font_color) != 3:
            errors.append(f"font_color must be RGB tuple, got {font_color}")
        else:
            if not all(isinstance(c, (int, float)) and 0.0 <= c <= 1.0 for c in font_color):
                errors.append("font_color values must be 0.0-1.0")
        
        # Validate page range
        page_range = config.params.get("page_range", "all")
        if not isinstance(page_range, str):
            errors.append(f"page_range must be string, got {type(page_range)}")
        
        return errors

    def execute(self, input_path: str, output_path: str, config: StampConfig) -> StampResult:
        if not _HAS_FITZ:
            return StampResult(
                success=False,
                error_message="PyMuPDF not available - cannot apply designation stamps"
            )
        
        # Validate input
        if not os.path.exists(input_path):
            raise StampValidationError(f"Input file not found: {input_path}")
        
        if not input_path.lower().endswith(".pdf"):
            return StampResult(
                success=False,
                error_message=f"Unsupported format: {input_path} (only PDF supported)"
            )
        
        # Validate configuration
        errors = self.validate_config(config)
        if errors:
            return StampResult(
                success=False,
                error_message=f"Configuration errors: {'; '.join(errors)}"
            )
        
        # Extract parameters
        text = config.params.get("text", "")
        font_size = config.params.get("font_size", 12.0)
        font_color = config.params.get("font_color", (1.0, 0.0, 0.0))
        font_name = config.params.get("font_name", "helv")
        page_range = config.params.get("page_range", "all")
        
        try:
            # Open PDF
            doc = fitz.open(input_path)
            
            # Parse page range
            pages_to_stamp = self._parse_page_range(page_range, len(doc))
            
            # Initialize zone detector
            detector = ZoneDetector()
            
            stamp_values = []
            warnings = []
            
            # Stamp each page
            for page_num in pages_to_stamp:
                page = doc[page_num]
                stamp_values.append(text)
                
                # Detect zones if overlap checking enabled
                zones = None
                if config.check_overlap:
                    zones = detector.detect_text_zones(page)
                
                # Compute stamp size
                text_width = fitz.get_text_length(text, fontname=font_name, fontsize=font_size)
                text_height = font_size * 1.2  # Approximate line height
                
                # Compute placement
                stamp_rect = detector.compute_placement_rect(
                    page, config.placement, text_width, text_height, zones
                )
                
                # Check for overlap warnings
                if config.check_overlap and zones:
                    page_warnings = detector.detect_overlap_warnings(stamp_rect, zones)
                    for warning in page_warnings:
                        warnings.append(f"Page {page_num + 1}: {warning}")
                
                # Insert text
                page.insert_text(
                    (stamp_rect.x0, stamp_rect.y0 + font_size),  # Baseline adjustment
                    text,
                    fontname=font_name,
                    fontsize=font_size,
                    color=font_color,
                )
            
            # Save output
            doc.save(output_path)
            doc.close()
            
            return StampResult(
                success=True,
                output_path=output_path,
                pages_stamped=len(stamp_values),
                stamp_values=stamp_values,
                warnings=warnings,
                metadata={
                    "text": text,
                    "font_size": font_size,
                    "font_color": font_color,
                    "font_name": font_name,
                },
            )
        
        except Exception as e:
            return StampResult(
                success=False,
                error_message=f"Designation stamping failed: {str(e)}"
            )

    def _parse_page_range(self, page_range: str, total_pages: int) -> list[int]:
        """Parse page range string into list of page indices (0-based).
        
        Args:
            page_range: Page range string (e.g., '1-5,7,9-12' or 'all')
            total_pages: Total number of pages in document
            
        Returns:
            List of page indices (0-based)
        """
        if page_range.lower() == "all":
            return list(range(total_pages))
        
        pages = set()
        parts = page_range.split(",")
        
        for part in parts:
            part = part.strip()
            if "-" in part:
                # Range
                start_str, end_str = part.split("-", 1)
                start = int(start_str.strip()) - 1  # Convert to 0-based
                end = int(end_str.strip()) - 1
                for i in range(start, end + 1):
                    if 0 <= i < total_pages:
                        pages.add(i)
            else:
                # Single page
                page_num = int(part) - 1  # Convert to 0-based
                if 0 <= page_num < total_pages:
                    pages.add(page_num)
        
        return sorted(pages)
