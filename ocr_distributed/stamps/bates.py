"""Bates numbering stamp operations.

Provides BatesAssigner for sequential number generation and BatesStampOperation
for applying Bates numbers to PDF documents.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

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


@dataclass
class BatesConfig:
    """Configuration for Bates numbering.
    
    Attributes:
        prefix: Optional prefix (e.g., "PROD")
        suffix: Optional suffix (e.g., "CONF")
        start: Starting number (must be >= 1)
        width: Zero-padding width (1-12)
        separator: Separator between prefix/number/suffix (default: no separator)
    """
    prefix: str = ""
    suffix: str = ""
    start: int = 1
    width: int = 6
    separator: str = ""

    def __post_init__(self):
        if self.start < 1:
            raise StampValidationError(f"start must be >= 1, got {self.start}")
        if self.width < 1 or self.width > 12:
            raise StampValidationError(f"width must be 1-12, got {self.width}")


class BatesAssigner:
    """Generates sequential Bates numbers with prefix/suffix support."""

    def __init__(self, config: BatesConfig):
        """Initialize Bates assigner.
        
        Args:
            config: Bates configuration
        """
        self.config = config
        self._current = config.start

    def next(self) -> str:
        """Generate next Bates number.
        
        Returns:
            Formatted Bates number
        """
        number_str = str(self._current).zfill(self.config.width)
        self._current += 1
        
        parts = []
        if self.config.prefix:
            parts.append(self.config.prefix)
        parts.append(number_str)
        if self.config.suffix:
            parts.append(self.config.suffix)
        
        return self.config.separator.join(parts)

    def peek(self) -> str:
        """Peek at next Bates number without incrementing.
        
        Returns:
            Next Bates number
        """
        number_str = str(self._current).zfill(self.config.width)
        
        parts = []
        if self.config.prefix:
            parts.append(self.config.prefix)
        parts.append(number_str)
        if self.config.suffix:
            parts.append(self.config.suffix)
        
        return self.config.separator.join(parts)

    def reset(self, start: Optional[int] = None) -> None:
        """Reset counter to start value.
        
        Args:
            start: New start value (uses config.start if None)
        """
        self._current = start if start is not None else self.config.start


class BatesStampOperation(StampOperation):
    """Stamp operation for applying Bates numbers to PDFs."""

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "bates",
            "description": "Apply Bates numbering to PDF documents",
            "version": "1.0.0",
            "supported_formats": [".pdf"],
            "parameters": {
                "prefix": {
                    "type": "str",
                    "description": "Bates prefix (e.g., 'PROD')",
                    "required": False,
                    "default": "",
                },
                "suffix": {
                    "type": "str",
                    "description": "Bates suffix (e.g., 'CONF')",
                    "required": False,
                    "default": "",
                },
                "start": {
                    "type": "int",
                    "description": "Starting Bates number",
                    "required": False,
                    "default": 1,
                },
                "width": {
                    "type": "int",
                    "description": "Zero-padding width (1-12)",
                    "required": False,
                    "default": 6,
                },
                "separator": {
                    "type": "str",
                    "description": "Separator between prefix/number/suffix",
                    "required": False,
                    "default": "",
                },
                "font_size": {
                    "type": "float",
                    "description": "Font size in points",
                    "required": False,
                    "default": 10.0,
                },
                "font_color": {
                    "type": "tuple[float, float, float]",
                    "description": "RGB color (0.0-1.0)",
                    "required": False,
                    "default": (0.0, 0.0, 0.0),
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
        
        # Validate Bates config
        try:
            self._extract_bates_config(config)
        except StampValidationError as e:
            errors.append(str(e))
            return errors
        
        # Validate font size
        font_size = config.params.get("font_size", 10.0)
        if not isinstance(font_size, (int, float)) or font_size <= 0:
            errors.append(f"font_size must be positive number, got {font_size}")
        
        # Validate font color
        font_color = config.params.get("font_color", (0.0, 0.0, 0.0))
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
                error_message="PyMuPDF not available - cannot apply Bates stamps"
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
        bates_config = self._extract_bates_config(config)
        font_size = config.params.get("font_size", 10.0)
        font_color = config.params.get("font_color", (0.0, 0.0, 0.0))
        page_range = config.params.get("page_range", "all")
        
        try:
            # Open PDF
            doc = fitz.open(input_path)
            
            # Parse page range
            pages_to_stamp = self._parse_page_range(page_range, len(doc))
            
            # Initialize Bates assigner and zone detector
            assigner = BatesAssigner(bates_config)
            detector = ZoneDetector()
            
            stamp_values = []
            warnings = []
            
            # Stamp each page
            for page_num in pages_to_stamp:
                page = doc[page_num]
                bates_number = assigner.next()
                stamp_values.append(bates_number)
                
                # Detect zones if overlap checking enabled
                zones = None
                if config.check_overlap:
                    zones = detector.detect_text_zones(page)
                
                # Compute stamp size
                text_width = fitz.get_text_length(bates_number, fontsize=font_size)
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
                    bates_number,
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
                    "prefix": bates_config.prefix,
                    "suffix": bates_config.suffix,
                    "start": bates_config.start,
                    "width": bates_config.width,
                    "separator": bates_config.separator,
                },
            )
        
        except Exception as e:
            return StampResult(
                success=False,
                error_message=f"Bates stamping failed: {str(e)}"
            )

    def _extract_bates_config(self, config: StampConfig) -> BatesConfig:
        """Extract BatesConfig from StampConfig."""
        return BatesConfig(
            prefix=config.params.get("prefix", ""),
            suffix=config.params.get("suffix", ""),
            start=config.params.get("start", 1),
            width=config.params.get("width", 6),
            separator=config.params.get("separator", ""),
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
