"""Base contracts for stamp operations.

Defines the abstract interface and data models for Bates and confidentiality
stamping operations. All stamp implementations must inherit from StampOperation
and implement the required methods.
"""

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False
    fitz = None  # type: ignore


# --- Enums ---


class StampPlacement(str, Enum):
    """Stamp placement location on page."""
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"


# --- Exceptions ---


class StampError(Exception):
    """Base exception for stamp operation failures."""
    pass


class StampValidationError(StampError):
    """Raised when stamp configuration or input validation fails."""
    pass


# --- Data Models ---


@dataclass
class StampConfig:
    """Configuration for a stamp operation.
    
    Attributes:
        operation_name: Registered stamp operation name
        placement: Where to place the stamp on each page
        params: Operation-specific parameters (e.g., Bates prefix, designation text)
        validate_input: Whether to validate input before stamping
        check_overlap: Whether to detect and warn about stamp overlaps
    """
    operation_name: str
    placement: StampPlacement = StampPlacement.BOTTOM_RIGHT
    params: dict[str, Any] = field(default_factory=dict)
    validate_input: bool = True
    check_overlap: bool = True

    def __post_init__(self):
        if not self.operation_name:
            raise StampValidationError("operation_name is required")
        if not isinstance(self.placement, StampPlacement):
            raise StampValidationError(f"placement must be StampPlacement enum, got {type(self.placement)}")


@dataclass
class StampResult:
    """Result of a stamp operation.
    
    Attributes:
        success: Whether the stamping completed successfully
        output_path: Path to the stamped output file (if success)
        error_message: Error description (if not success)
        metadata: Operation-specific result metadata
        pages_stamped: Number of pages stamped
        stamp_values: List of stamp values applied (e.g., Bates numbers)
        warnings: List of non-fatal warnings (e.g., overlap detected)
    """
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pages_stamped: int = 0
    stamp_values: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.success and not self.output_path:
            raise StampValidationError("output_path is required when success=True")
        if not self.success and not self.error_message:
            raise StampValidationError("error_message is required when success=False")


# --- Abstract Base Class ---


class StampOperation(abc.ABC):
    """Abstract base class for all stamp operations.
    
    Stamp operations are deterministic overlay operations that add visual
    elements (Bates numbers, confidentiality designations) to documents.
    
    Implementations must:
    1. Provide metadata via get_metadata()
    2. Validate configuration via validate_config()
    3. Execute stamping via execute()
    """

    @abc.abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return operation metadata for registry introspection.
        
        Returns:
            Dictionary with keys:
                - name (str): Unique operation name
                - description (str): Human-readable description
                - version (str): Implementation version
                - supported_formats (list[str]): Supported input extensions
                - parameters (dict): Expected parameter schema
        """
        pass

    @abc.abstractmethod
    def validate_config(self, config: StampConfig) -> list[str]:
        """Validate stamp configuration.
        
        Args:
            config: Stamp configuration to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        pass

    @abc.abstractmethod
    def execute(self, input_path: str, output_path: str, config: StampConfig) -> StampResult:
        """Execute the stamp operation.
        
        Args:
            input_path: Path to input file (must exist)
            output_path: Path where output should be written
            config: Stamp configuration
            
        Returns:
            StampResult with operation outcome
            
        Raises:
            StampError: On fatal execution errors
            StampValidationError: On invalid input
        """
        pass

    def supports_format(self, file_extension: str) -> bool:
        """Check if this operation supports the given file extension.
        
        Args:
            file_extension: File extension (e.g., '.pdf')
            
        Returns:
            True if format is supported
        """
        metadata = self.get_metadata()
        supported = metadata.get("supported_formats", [])
        return file_extension.lower() in [fmt.lower() for fmt in supported]
