"""Convenience re-exports for API audit middleware components.

The core implementation lives in :mod:`api.audit`.  This module provides
a shorter import path for the most commonly used symbols so that
callers can write::

    from api.audit_middleware import ApiAuditMiddleware, mask_api_key

Configuration is controlled by the following environment variables:

- ``API_AUDIT_LOG_ENABLED`` (default ``1``) -- master toggle
- ``API_AUDIT_LOG_PATH`` -- override output file location
- ``API_AUDIT_EXCLUDE_HEALTH`` -- skip ``/api/v1/health`` from audit
"""

from __future__ import annotations

from api.audit import (  # noqa: F401 — re-exports
    ApiAuditMiddleware,
    build_api_audit_event,
    mask_api_key,
    record_api_audit_event,
    reset_chain_state,
    verify_audit_chain,
)

__all__ = [
    "ApiAuditMiddleware",
    "build_api_audit_event",
    "mask_api_key",
    "record_api_audit_event",
    "reset_chain_state",
    "verify_audit_chain",
]
