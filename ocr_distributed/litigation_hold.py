"""Process-independent litigation-hold policy shared by both data planes."""

from __future__ import annotations

import os

TRUTHY_HOLD_VALUES = frozenset({"1", "true", "yes"})


def is_litigation_hold_active() -> bool:
    """Return whether the deployment-wide deletion hold is active."""
    return os.environ.get("LITIGATION_HOLD", "").strip().lower() in TRUTHY_HOLD_VALUES
