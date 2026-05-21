"""Authentication identity model and role-based access control."""

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


@dataclass(frozen=True)
class AuthIdentity:
    """Represents an authenticated identity attached to a request.

    Attributes:
        subject: Unique identifier for the identity (e.g. "apikey", user ID, email).
        role: Authorization role — one of "admin", "operator", "viewer".
        auth_method: How the identity was authenticated — "apikey" or "oauth2".
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


def get_identity(request: Request) -> AuthIdentity:
    """Extract the AuthIdentity from request.state.

    Returns a default admin identity if no identity was set (backward compat
    for unauthenticated mode).
    """
    identity = getattr(request.state, "identity", None)
    if identity is None:
        # Unauthenticated mode — treat as admin for backward compatibility
        return AuthIdentity(
            subject="anonymous",
            role="admin",
            auth_method="none",
        )
    return identity


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
