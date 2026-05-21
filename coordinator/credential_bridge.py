"""Bridge module to import credential_manager from the repo root.

The ``credential_manager.py`` module lives at the repository root (``/app/``
in Docker).  This bridge provides a clean import path for the coordinator
Django project without requiring PYTHONPATH adjustments at every call site.

When ``credential_manager`` is not importable (e.g. running coordinator
tests in isolation), the bridge falls back to plain ``os.environ`` lookups
so that existing behaviour is preserved.
"""

from __future__ import annotations

import logging
import os
import sys

# Ensure repo root is on sys.path (handles both Docker /app and local dev)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logger = logging.getLogger(__name__)

try:
    from credential_manager import (
        CredentialManager,
        get_credential,
        validate_credentials,
    )
except ImportError:
    logger.warning(
        "credential_manager not available, falling back to os.environ"
    )

    def get_credential(key, default=None):  # type: ignore[misc]
        """Fallback: read from os.environ when credential_manager is absent."""
        return os.environ.get(key, default)

    def validate_credentials(**kwargs):  # type: ignore[misc]
        """No-op validation stub when credential_manager is absent."""
        return type(
            "ValidationReport",
            (),
            {
                "passed": True,
                "errors": [],
                "summary": lambda self: "PASS: credential_manager not available",
            },
        )()

    CredentialManager = None  # type: ignore[assignment,misc]

__all__ = ["get_credential", "validate_credentials", "CredentialManager"]
