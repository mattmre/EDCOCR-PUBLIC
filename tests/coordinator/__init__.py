"""Test package shim that coexists with the real coordinator namespace."""

from pathlib import Path

_REAL_COORDINATOR = Path(__file__).resolve().parents[2] / "coordinator"
if _REAL_COORDINATOR.exists():
    __path__.append(str(_REAL_COORDINATOR))  # type: ignore[name-defined]
