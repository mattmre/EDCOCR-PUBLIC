"""Transform operation registry.

Central registry for transform operations with discovery, validation,
and duplicate prevention.
"""

import logging
import threading
from typing import Optional

from .base import TransformOperation, TransformValidationError

logger = logging.getLogger(__name__)


class TransformRegistry:
    """Registry for transform operations with discovery and introspection.
    
    The registry:
    - Prevents duplicate operation names
    - Provides metadata listing for API discovery
    - Validates operations at registration time
    - Thread-safe for concurrent access
    """

    def __init__(self):
        """Initialize an empty transform registry."""
        self._operations: dict[str, TransformOperation] = {}
        self._lock = threading.Lock()

    def register(self, operation: TransformOperation) -> None:
        """Register a transform operation.
        
        Args:
            operation: Transform operation instance to register
            
        Raises:
            TransformValidationError: If operation invalid or name already registered
        """
        if not isinstance(operation, TransformOperation):
            raise TransformValidationError(
                f"Operation must inherit from TransformOperation, got {type(operation)}"
            )

        metadata = operation.get_metadata()
        if "name" not in metadata:
            raise TransformValidationError("Operation metadata must include 'name'")
        name = metadata.get("name")
        if not isinstance(name, str) or not name.strip():
            raise TransformValidationError(f"Operation name must be non-empty string, got {name!r}")

        with self._lock:
            if name in self._operations:
                raise TransformValidationError(
                    f"Transform operation '{name}' is already registered"
                )
            self._operations[name] = operation
            logger.info(f"Registered transform operation: {name}")

    def get(self, name: str) -> Optional[TransformOperation]:
        """Get a registered transform operation by name.
        
        Args:
            name: Operation name
            
        Returns:
            Transform operation instance or None if not found
        """
        with self._lock:
            return self._operations.get(name)

    def list_operations(self) -> list[str]:
        """List all registered operation names.
        
        Returns:
            Sorted list of operation names
        """
        with self._lock:
            return sorted(self._operations.keys())

    def get_metadata(self, name: str) -> Optional[dict]:
        """Get metadata for a registered operation.
        
        Args:
            name: Operation name
            
        Returns:
            Operation metadata dictionary or None if not found
        """
        operation = self.get(name)
        if operation:
            return operation.get_metadata()
        return None

    def list_all_metadata(self) -> list[dict]:
        """Get metadata for all registered operations.
        
        Returns:
            List of operation metadata dictionaries, sorted by name
        """
        with self._lock:
            operations = list(self._operations.items())
        
        metadata_list = []
        for _name, operation in sorted(operations, key=lambda x: x[0]):
            metadata_list.append(operation.get_metadata())
        
        return metadata_list

    def clear(self) -> None:
        """Clear all registered operations (primarily for testing)."""
        with self._lock:
            self._operations.clear()
            logger.debug("Cleared all transform operations from registry")


# --- Global Registry ---

_global_registry: Optional[TransformRegistry] = None
_registry_lock = threading.Lock()


def get_transform_registry() -> TransformRegistry:
    """Get the global transform registry singleton.
    
    Returns:
        Global TransformRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = TransformRegistry()
    return _global_registry
