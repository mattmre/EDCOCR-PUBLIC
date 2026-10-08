"""Shared tenant-boundary helpers for translation API routers."""

from __future__ import annotations

from fastapi import HTTPException, Request


def _authenticated_tenant_id(request: Request) -> str | None:
    return getattr(request.state, "tenant_id", None)


def _is_platform_admin(request: Request) -> bool:
    """True when server-attached tenant_permissions includes platform_admin."""
    from api.authz import is_platform_admin

    return is_platform_admin(request)


def has_tenant_scope(request: Request) -> bool:
    """Return whether this request must be constrained to one tenant."""
    return _authenticated_tenant_id(request) is not None and not _is_platform_admin(
        request
    )


def bind_requested_tenant(
    request: Request,
    requested_tenant_id: str | None,
    *,
    allow_default_alias: bool = False,
) -> str | None:
    """Bind a caller-controlled tenant value to the authenticated tenant.

    Legacy API-key, OAuth, and explicit unauthenticated single-tenant modes do
    not set ``request.state.tenant_id`` and retain their existing semantics.
    Tenant API keys may only select their own tenant.  ``platform_admin`` keys
    retain cross-tenant access.
    """
    from api.authz import require_tenant_scope_or_platform

    # Resolve the shared scope first.  This preserves legacy single-tenant and
    # platform-admin behavior while rejecting OAuth/anonymous callers that have
    # no tenant binding under multitenancy.
    authenticated = require_tenant_scope_or_platform(request)
    if authenticated is None:
        return requested_tenant_id

    if allow_default_alias and requested_tenant_id in (None, "", "default"):
        return authenticated

    if requested_tenant_id != authenticated:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "tenant_mismatch",
                "message": "Cannot access translation resources for another tenant.",
            },
        )
    return authenticated


def require_owned_tenant(
    request: Request,
    resource_tenant_id: str,
    *,
    not_found_detail: str = "translation resource not found",
) -> None:
    """Hide a foreign translation resource from a tenant-scoped caller."""
    from api.authz import require_tenant_scope_or_platform

    authenticated = require_tenant_scope_or_platform(request)
    if authenticated is None:
        return
    if resource_tenant_id != authenticated:
        raise HTTPException(status_code=404, detail=not_found_detail)
