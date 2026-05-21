"""Stamp operations contract and implementation exports."""

from .base import (
    StampConfig,
    StampError,
    StampOperation,
    StampPlacement,
    StampResult,
    StampValidationError,
)
from .bates import BatesAssigner, BatesConfig, BatesStampOperation
from .builtin import register_builtin_stamps
from .designation import STANDARD_DESIGNATIONS, DesignationStampOperation
from .registry import StampRegistry, get_stamp_registry
from .zone import Rect, Zone, ZoneDetector

__all__ = [
    "StampConfig",
    "StampError",
    "StampOperation",
    "StampPlacement",
    "StampResult",
    "StampValidationError",
    "StampRegistry",
    "get_stamp_registry",
    "BatesAssigner",
    "BatesConfig",
    "BatesStampOperation",
    "DesignationStampOperation",
    "STANDARD_DESIGNATIONS",
    "Rect",
    "Zone",
    "ZoneDetector",
    "register_builtin_stamps",
]
