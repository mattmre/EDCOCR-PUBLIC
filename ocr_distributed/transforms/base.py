"""Base contracts for transform operations.

Defines the abstract interface and data models for PDF/image transform operations.
All transform implementations must inherit from TransformOperation and implement
the required methods.
"""

import abc
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False
    fitz = None  # type: ignore


# --- Exceptions ---


class TransformError(Exception):
    """Base exception for transform operation failures."""
    pass


class TransformValidationError(TransformError):
    """Raised when transform configuration or input validation fails."""
    pass


# --- Data Models ---


@dataclass
class TransformConfig:
    """Configuration for a transform operation.
    
    Attributes:
        operation_name: Registered transform operation name
        params: Operation-specific parameters
        validate_input: Whether to validate input before transform
        preserve_metadata: Whether to preserve PDF/image metadata
    """
    operation_name: str
    params: dict[str, Any] = field(default_factory=dict)
    validate_input: bool = True
    preserve_metadata: bool = True

    def __post_init__(self):
        if not self.operation_name:
            raise TransformValidationError("operation_name is required")


@dataclass
class TransformResult:
    """Result of a transform operation.
    
    Attributes:
        success: Whether the transform completed successfully
        output_path: Path to the transformed output file (if success)
        error_message: Error description (if not success)
        metadata: Operation-specific result metadata
        pages_processed: Number of pages/frames processed
        warnings: List of non-fatal warnings
    """
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pages_processed: int = 0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.success and not self.output_path:
            raise TransformValidationError("output_path is required when success=True")
        if not self.success and not self.error_message:
            raise TransformValidationError("error_message is required when success=False")


# --- Abstract Base Class ---


class TransformOperation(abc.ABC):
    """Abstract base class for all transform operations.
    
    Transform operations are deterministic, side-effect-free transformations
    that take an input file and configuration and produce an output file.
    
    Implementations must:
    1. Provide metadata via get_metadata()
    2. Validate configuration via validate_config()
    3. Execute transformation via execute()
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
                - output_format (str): Output file extension
                - parameters (dict): Expected parameter schema
        """
        pass

    @abc.abstractmethod
    def validate_config(self, config: TransformConfig) -> list[str]:
        """Validate transform configuration.
        
        Args:
            config: Transform configuration to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        pass

    @abc.abstractmethod
    def execute(self, input_path: str, output_path: str, config: TransformConfig) -> TransformResult:
        """Execute the transform operation.
        
        Args:
            input_path: Path to input file (must exist)
            output_path: Path where output should be written
            config: Transform configuration
            
        Returns:
            TransformResult with operation outcome
            
        Raises:
            TransformError: On fatal execution errors
            TransformValidationError: On invalid input
        """
        pass

    def supports_format(self, file_extension: str) -> bool:
        """Check if this operation supports the given file extension.
        
        Args:
            file_extension: File extension (e.g., '.pdf', '.png')
            
        Returns:
            True if format is supported
        """
        metadata = self.get_metadata()
        supported = metadata.get("supported_formats", [])
        return file_extension.lower() in [fmt.lower() for fmt in supported]
