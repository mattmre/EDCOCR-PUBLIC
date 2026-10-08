"""Authentication identity model and role-based access control.

R4-RC-14: never invent a platform-admin identity for missing auth. Anonymous
access is only allowed when ``ALLOW_UNAUTHENTICATED`` is explicitly enabled,
and under multi-tenancy anonymous roles are demoted to viewer (never
admin/operator).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """Supported authorization roles."""

    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


# Valid role names (ordered by privilege level, highest first)
VALID_ROLES = tuple(role.value for role in UserRole)

# Roles that must never be granted to unauthenticated callers under MT.
_PRIVILEGED_ANON_ROLES = frozenset({"admin", "operator", "platform_admin"})


@dataclass(frozen=True)
class AuthIdentity:
    """Represents an authenticated identity attached to a request.

    Attributes:
        subject: Unique identifier for the identity (e.g. "apikey", user ID, email).
        role: Authorization role — one of "admin", "operator", "viewer".
        auth_method: How the identity was authenticated — "apikey", "oauth2", or "none".
        email: Optional email from JWT claims.
        name: Optional display name from JWT claims.
        claims: Raw JWT claims dict (empty for API key auth).
    """

    subject: str
    role: str
    auth_method: str
    email: Optional[str] = None
    name: Optional[str] = None
    claims: dict[str, Any] = field(default_factory=dict)


def _anonymous_role() -> str:
    """Resolve least-privilege anonymous role (R4-RC-14)."""
    from api import config

    role = (getattr(config, "ANONYMOUS_ROLE", None) or "viewer").strip() or "viewer"
    mt = bool(getattr(config, "ENABLE_MULTITENANCY", False))
    if mt and role in _PRIVILEGED_ANON_ROLES:
        logger.warning(
            "ANONYMOUS_ROLE=%r demoted to viewer under multi-tenancy (R4-RC-14)",
            role,
        )
        return UserRole.VIEWER.value
    if role not in VALID_ROLES:
        return UserRole.VIEWER.value
    return role


def get_identity(request: Request) -> AuthIdentity:
    """Extract the AuthIdentity from request.state.

    Missing identity:
    - If ``ALLOW_UNAUTHENTICATED`` is false → HTTP 401 (no invented admin).
    - If true → anonymous with ``ANONYMOUS_ROLE`` (viewer under MT).
    """
    identity = getattr(request.state, "identity", None)
    if identity is not None:
        return identity

    from api import config

    if not getattr(config, "ALLOW_UNAUTHENTICATED", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "unauthorized",
                "message": "Authentication required",
            },
        )

    return AuthIdentity(
        subject="anonymous",
        role=_anonymous_role(),
        auth_method="none",
    )


def require_role(*allowed_roles: str):
    """FastAPI dependency factory that enforces role-based access.

    Usage::

        @router.post("/endpoint")
        async def my_endpoint(
            request: Request,
            _auth: None = Depends(require_role("admin", "operator")),
        ):
            ...

    Admin role always passes (super-user).  If the caller's role is not
    in *allowed_roles* and is not "admin", a 403 is raised.
    """

    async def _check_role(request: Request) -> None:
        identity = get_identity(request)

        from api.authz import enforce_tenant_capability

        enforce_tenant_capability(request)

        # Admin always passes
        if identity.role == "admin":
            return

        if identity.role not in allowed_roles:
            logger.warning(
                "RBAC denied: subject=%s role=%s required=%s path=%s",
                identity.subject,
                identity.role,
                allowed_roles,
                request.url.path,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "forbidden",
                    "message": (
                        f"Role '{identity.role}' is not authorized for this endpoint. "
                        f"Required: {', '.join(allowed_roles)}"
                    ),
                },
            )

    return _check_role
