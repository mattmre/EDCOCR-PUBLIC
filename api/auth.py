"""API authentication middleware — API key + optional OAuth2 JWT."""

import json
import logging
import os
import secrets

from fastapi import Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from api.config import (
    ALLOW_UNAUTHENTICATED,
    ANONYMOUS_ROLE,
    API_ALLOWED_IPS,
    ENABLE_MULTITENANCY,
    EXPOSE_API_DOCS,
    OCR_API_KEY,
)
from api.identity import AuthIdentity

logger = logging.getLogger(__name__)

# Paths exempt from authentication.  Health probes are always exempt so that
# orchestrators (Kubernetes, Docker HEALTHCHECK, etc.) can poll without
# credentials.  OpenAPI docs are only exempt when EXPOSE_API_DOCS=true
#; by default they are not mounted at all so unauthenticated
# callers cannot enumerate the API surface.
_ALWAYS_EXEMPT_PATHS = {
    "/api/v1/health",
    "/api/v1/health/detailed",
    "/api/v1/ready",
    "/api/v1/readiness",
    "/api/v1/translation/readiness",
}
_DOCS_EXEMPT_PATHS = {"/docs", "/openapi.json", "/redoc"}
# Plan C Phase 1, item C6 -- the federation custody ingest endpoint
# carries its own ``Authorization: Bearer`` auth (against
# ``OCR_FEDERATION_CUSTODY_AUTH_TOKEN``) and is only mounted when
# ``OCR_FEDERATION_CUSTODY_ENABLED=true``.  Exempt it from the global
# API-key middleware so peer clusters can post events without also
# being provisioned with a tenant API key.  The exemption is gated on
# the same env flag so it is a no-op when the feature is off.
_FEDERATION_CUSTODY_PATHS = {"/api/v1/federation/custody/ingest"}


def _is_exempt_path(path: str) -> bool:
    """Return True when ``path`` should skip authentication."""
    if path in _ALWAYS_EXEMPT_PATHS:
        return True
    if EXPOSE_API_DOCS and path in _DOCS_EXEMPT_PATHS:
        return True
    if path in _FEDERATION_CUSTODY_PATHS and (
        os.environ.get("OCR_FEDERATION_CUSTODY_ENABLED", "").strip().lower()
        in ("1", "true", "yes", "on")
    ):
        return True
    return False


def _safe_compare_api_key(provided: str, expected: str) -> bool:
    """Constant-time API-key comparison that tolerates arbitrary input.

    ``secrets.compare_digest`` raises ``TypeError`` when a ``str`` operand
    contains characters with ordinal > 127 (AUTH-BUG).  An unauthenticated
    attacker sending a non-ASCII ``X-API-Key`` header would therefore crash
    the auth middleware.  We normalise both sides to UTF-8 bytes before the
    comparison and swallow the narrow set of errors that indicate an
    obviously invalid key.
    """
    if not provided or not expected:
        return False
    try:
        provided_bytes = provided.encode("utf-8")
        expected_bytes = expected.encode("utf-8")
        return secrets.compare_digest(provided_bytes, expected_bytes)
    except (UnicodeEncodeError, TypeError, AttributeError):
        return False

# Log auth configuration at import time
if not OCR_API_KEY:
    if ALLOW_UNAUTHENTICATED:
        logger.warning(
            "OCR_API_KEY is not set and ALLOW_UNAUTHENTICATED=true "
            "— API authentication is disabled"
        )
    else:
        logger.error(
            "OCR_API_KEY is not set and ALLOW_UNAUTHENTICATED=false "
            "— API startup must fail"
        )


def _set_audit_outcome(request: Request, outcome: str) -> None:
    """Attach a structured auth outcome for request audit logging."""
    request.state.audit_auth_outcome = outcome


def get_api_key() -> str:
    """Return configured API key (empty string = disabled)."""
    return OCR_API_KEY


def get_allowed_ips() -> tuple[str, ...]:
    """Return optional API ingress allowlist (empty tuple = disabled)."""
    return API_ALLOWED_IPS


def allow_unauthenticated() -> bool:
    """Return True when explicit insecure override is enabled."""
    return ALLOW_UNAUTHENTICATED


def validate_auth_configuration() -> None:
    """Fail startup unless auth is configured or explicitly overridden."""
    if get_api_key():
        return
    if allow_unauthenticated():
        logger.warning(
            "Starting with API auth disabled because ALLOW_UNAUTHENTICATED=true"
        )
        return
    if ENABLE_MULTITENANCY:
        logger.info(
            "Multi-tenancy enabled without legacy OCR_API_KEY; "
            "tenant keys will be used for authentication"
        )
        return

    from api.oauth2_config import OAUTH2_ENABLED

    if OAUTH2_ENABLED:
        logger.info(
            "OAuth2 enabled without legacy OCR_API_KEY; "
            "bearer-token auth will be used"
        )
        return

    raise RuntimeError(
        "OCR_API_KEY is required unless ALLOW_UNAUTHENTICATED=true is set."
    )


def _try_bearer_auth(request: Request) -> AuthIdentity | None:
    """Attempt OAuth2 Bearer token authentication."""
    from api.oauth2_config import OAUTH2_ENABLED

    if not OAUTH2_ENABLED:
        return None

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return None

    token = auth_header[7:].strip()
    if not token:
        return None

    try:
        from api.oauth2 import extract_role, validate_jwt

        claims = validate_jwt(token)
        role = extract_role(claims)
        return AuthIdentity(
            subject=claims.get("sub", "unknown"),
            role=role,
            auth_method="oauth2",
            email=claims.get("email"),
            name=claims.get("name"),
            claims=claims,
        )
    except (ValueError, ImportError, RuntimeError) as exc:
        logger.warning("OAuth2 Bearer auth failed: %s", exc)
        return None


def _identity_role_from_permissions(permissions: list[str]) -> str:
    """Map tenant API-key permissions onto the RBAC role model."""
    if "platform_admin" in permissions or "admin" in permissions:
        return "admin"
    if "submit" in permissions:
        return "operator"
    return "viewer"


def _try_multitenant_auth(request: Request, provided_key: str) -> bool | None:
    """Attempt multi-tenant API key resolution."""
    if not ENABLE_MULTITENANCY:
        return None

    from api.database import get_session_factory
    from api.tenant_manager import hash_api_key, resolve_tenant_by_key

    key_hash = hash_api_key(provided_key)
    session = get_session_factory()()
    try:
        result = resolve_tenant_by_key(key_hash, session=session)
        if result is None:
            return False

        tenant, key_record = result
        tenant_id = tenant.tenant_id
        key_id = key_record.key_id
        raw_permissions = key_record.permissions

        try:
            permissions = json.loads(raw_permissions)
        except (json.JSONDecodeError, TypeError):
            permissions = ["submit", "read"]
    except SQLAlchemyError:
        logger.exception("Error during multi-tenant key resolution")
        return False
    finally:
        session.close()

    request.state.tenant_id = tenant_id
    request.state.tenant_permissions = permissions
    request.state.key_id = key_id
    request.state.identity = AuthIdentity(
        subject=tenant_id,
        role=_identity_role_from_permissions(permissions),
        auth_method="apikey",
        claims={
            "tenant_id": tenant_id,
            "key_id": key_id,
            "permissions": permissions,
        },
    )
    return True


async def api_key_middleware(request: Request, call_next):
    """Validate authentication: tenant key, legacy API key, or OAuth2 bearer."""
    if _is_exempt_path(request.url.path):
        _set_audit_outcome(request, "exempt")
        return await call_next(request)

    allowed_ips = get_allowed_ips()
    if allowed_ips:
        client_host = request.client.host if request.client else ""
        if client_host not in allowed_ips:
            _set_audit_outcome(request, "ip_allowlist_denied")
            logger.warning(
                "Forbidden API request from non-allowlisted client %s",
                client_host or "unknown",
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "forbidden", "message": "Client is not allowlisted"},
            )

    api_key = get_api_key()
    provided_key = request.headers.get("X-API-Key", "")

    # Explicit API keys take priority over Bearer tokens.
    if provided_key:
        if ENABLE_MULTITENANCY:
            mt_result = _try_multitenant_auth(request, provided_key)
            if mt_result is True:
                tenant_id = getattr(request.state, "tenant_id", None)
                _set_audit_outcome(request, "authorized")
                response = await call_next(request)
                if tenant_id and response.status_code < status.HTTP_400_BAD_REQUEST:
                    try:
                        from api.usage import record_api_call

                        record_api_call(tenant_id)
                    except Exception:
                        logger.exception(
                            "Failed to record tenant API call usage for %s",
                            tenant_id,
                        )
                return response

        if api_key and _safe_compare_api_key(provided_key, api_key):
            from api.oauth2_config import APIKEY_ROLE

            request.state.identity = AuthIdentity(
                subject="apikey",
                role=APIKEY_ROLE,
                auth_method="apikey",
            )
            _set_audit_outcome(request, "authorized")
            return await call_next(request)

        _set_audit_outcome(request, "invalid_api_key")
        logger.warning(
            "Unauthorized API request (bad key) from %s",
            request.client.host if request.client else "unknown",
        )
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "unauthorized", "message": "Invalid or missing API key"},
        )

    bearer_identity = _try_bearer_auth(request)
    if bearer_identity is not None:
        request.state.identity = bearer_identity
        _set_audit_outcome(request, "authorized")
        return await call_next(request)

    if not api_key:
        from api.oauth2_config import OAUTH2_ENABLED

        if not allow_unauthenticated():
            if ENABLE_MULTITENANCY and OAUTH2_ENABLED:
                _set_audit_outcome(request, "missing_credentials")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "error": "unauthorized",
                        "message": "API key or bearer token required",
                    },
                )
            if ENABLE_MULTITENANCY:
                _set_audit_outcome(request, "missing_credentials")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"error": "unauthorized", "message": "API key required"},
                )
            if OAUTH2_ENABLED:
                _set_audit_outcome(request, "missing_credentials")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "error": "unauthorized",
                        "message": "Invalid or missing bearer token",
                    },
                )

            logger.error(
                "Rejecting request due to missing OCR_API_KEY without insecure override"
            )
            _set_audit_outcome(request, "server_misconfigured")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "server_misconfigured",
                    "message": (
                        "API key authentication is required but OCR_API_KEY is not "
                        "configured."
                    ),
                },
            )

        request.state.identity = AuthIdentity(
            subject="anonymous",
            role=ANONYMOUS_ROLE,
            auth_method="none",
        )
        _set_audit_outcome(request, "anonymous_override")
        return await call_next(request)

    _set_audit_outcome(request, "missing_api_key")
    logger.warning(
        "Unauthorized API request from %s",
        request.client.host if request.client else "unknown",
    )
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"error": "unauthorized", "message": "Invalid or missing API key"},
    )
