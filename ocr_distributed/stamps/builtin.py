"""Built-in stamp operation registration.

Provides a helper function to register all built-in stamp operations
(Bates, designation) with a registry.
"""

from typing import Optional

from .bates import BatesStampOperation
from .designation import DesignationStampOperation
from .registry import StampRegistry, get_stamp_registry


def register_builtin_stamps(registry: Optional[StampRegistry] = None) -> None:
    """Register all built-in stamp operations.
    
    Args:
        registry: StampRegistry to register operations with (uses global if None)
    """
    if registry is None:
        registry = get_stamp_registry()
    
    # Register Bates stamp
    registry.register(BatesStampOperation())
    
    # Register designation stamp
    registry.register(DesignationStampOperation())
