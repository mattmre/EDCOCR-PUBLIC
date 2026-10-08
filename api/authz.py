"""Shared multi-tenant authorization helpers (fail-closed under multitenancy).

Privilege source of truth for platform_admin is ``request.state.tenant_permissions``
only — never free-form JWT / identity ``claims["permissions"]`` (R4-RC-11).

Under ``ENABLE_MULTITENANCY``, data-plane access requires a tenant-scoped API
key or an explicit ``platform_admin`` permission (RC-01 / R4-RC-13).
"""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Request, status

PLATFORM_ADMIN_PERMISSION = "platform_admin"

# Stored tenant-key permissions are a closed capability vocabulary.  Keeping
# this in the authorization module gives request validation and persistence a
# shared source of truth.
VALID_TENANT_PERMISSIONS = frozenset(
    {"submit", "read", "admin", PLATFORM_ADMIN_PERMISSION}
)

# Granted to the legacy OCR_API_KEY path when multitenancy is enabled so
# break-glass single-key operators remain platform-capable without the
# ``permissions is None`` fail-open that also elevated OAuth/anonymous.
LEGACY_PLATFORM_PERMISSIONS: list[str] = [
    "platform_admin",
    "admin",
    "submit",
    "read",
]


def authenticated_tenant_id(request: Request) -> Optional[str]:
    """Return the authenticated tenant id, if any."""
    return getattr(request.state, "tenant_id", None)


def tenant_permissions(request: Request) -> Optional[list]:
    """Return server-attached tenant permissions (or None if unset)."""
    permissions = getattr(request.state, "tenant_permissions", None)
    if not isinstance(permissions, (list, tuple, set, frozenset)):
        return None
    return list(permissions)


def is_platform_admin(request: Request) -> bool:
    """True only when server-attached tenant_permissions includes platform_admin.

    Does **not** trust identity.claims (OAuth JWT free-form claim injection).
    """
    permissions = tenant_permissions(request) or []
    return PLATFORM_ADMIN_PERMISSION in permissions


def require_tenant_scope_or_platform(request: Request) -> Optional[str]:
    """Return tenant filter id, or None when platform-global access is allowed.

    * ``platform_admin``: ``None`` (cross-tenant allowed).
    * Authenticated tenant key (``request.state.tenant_id`` set): that tenant id
      (always pinned when present).
    * Multitenancy **off** + no tenant: ``None`` (single-tenant / legacy).
    * Multitenancy **on** + no tenant and not platform: **403** fail-closed.
    """
    # Read at call time so tests / runtime can flip api.config.ENABLE_MULTITENANCY.
    from api.config import ENABLE_MULTITENANCY

    if is_platform_admin(request):
        return None
    tid = authenticated_tenant_id(request)
    if tid is not None:
        return tid
    if not ENABLE_MULTITENANCY:
        return None
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": "tenant_scope_required",
            "message": (
                "Multi-tenancy requires a tenant-scoped API key or "
                "platform_admin permission. Principals without a tenant "
                "binding (including OAuth without tenant mapping) cannot "
                "access tenant data planes."
            ),
        },
    )


def enforce_tenant_capability(request: Request) -> None:
    """Enforce stored tenant-key capabilities in addition to coarse RBAC.

    Roles remain useful for OAuth and single-tenant deployments, but a tenant
    API key's closed capability list is the authority for its data-plane
    operations.  Read-only keys cannot mutate; submit-only keys cannot browse
    evidence or decide reviews.
    """
    permissions = tenant_permissions(request)
    if permissions is None or is_platform_admin(request):
        return
    granted = set(permissions)
    method = request.method.upper()
    path = request.url.path.rstrip("/")
    if method in {"GET", "HEAD", "OPTIONS"}:
        required = "read"
    elif method == "POST" and path in {
        "/api/v1/jobs",
        "/api/v1/jobs/batch",
        "/api/v1/translation/jobs",
        "/api/v1/translation/batches",
    }:
        required = "submit"
    else:
        required = "admin"
    if required in granted or "admin" in granted:
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": "tenant_capability_required",
            "message": f"This operation requires tenant capability '{required}'.",
        },
    )


def require_platform_admin_for_shared_paths(request: Request) -> None:
    """Guard path-based operations that have no tenant storage binding.

    Transform and stamp execution accept server-side paths beneath the shared
    SOURCE/OUTPUT roots.  Until those operations grow tenant-derived storage
    roots or a job ownership binding, a tenant-scoped operator must not be able
    to select another tenant's file by path.  Single-tenant deployments retain
    their existing role-gated behavior.
    """
    from api.config import ENABLE_MULTITENANCY

    if ENABLE_MULTITENANCY and not is_platform_admin(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "shared_path_operation_requires_platform_admin",
                "message": (
                    "Path-based transform and stamp execution is platform-admin "
                    "only while multitenancy uses shared storage roots."
                ),
            },
        )


def require_platform_admin_for_global_control_plane(request: Request) -> None:
    """Reject tenant-scoped principals from global operational surfaces.

    Fleet state, queue alerts/thresholds, and Prometheus output are process-
    or platform-wide rather than tenant-owned resources. Under multitenancy,
    an ordinary tenant ``admin`` role must not be treated as a platform
    operator merely because it passes role-based RBAC.
    Single-tenant deployments retain the legacy role-gated behavior.
    """
    from api.config import ENABLE_MULTITENANCY

    if ENABLE_MULTITENANCY and not is_platform_admin(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "platform_admin_required",
                "message": (
                    "This global control-plane surface requires explicit "
                    "platform_admin permission under multitenancy."
                ),
            },
        )
