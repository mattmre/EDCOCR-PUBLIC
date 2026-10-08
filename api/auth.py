"""API authentication middleware — API key + optional OAuth2 JWT."""

import json
import logging
import os
import secrets
from urllib.parse import urlsplit

from fastapi import Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from api.config import (
    ALLOW_UNAUTHENTICATED,  # noqa: F401 - kept as ``api.auth.ALLOW_UNAUTHENTICATED`` patch target
    ANONYMOUS_ROLE,  # noqa: F401 - kept as ``api.auth.ANONYMOUS_ROLE`` patch target
    API_ALLOWED_IPS,
    ENABLE_MULTITENANCY,
    EXPOSE_API_DOCS,
    OCR_API_KEY,
    is_local_dev_environment,
)
from api.identity import AuthIdentity

logger = logging.getLogger(__name__)

# Paths exempt from authentication.  Health probes are always exempt so that
# orchestrators (Kubernetes, Docker HEALTHCHECK, etc.) can poll without
# credentials.  OpenAPI docs are only exempt when EXPOSE_API_DOCS=true
# (maintenance review); by default they are not mounted at all so unauthenticated
# callers cannot enumerate the API surface.
# Liveness/readiness only — orchestrators poll without credentials.
# `/api/v1/health/detailed` returns job-count recon and is NOT exempt under
# production (or REQUIRE_DETAILED_HEALTH_AUTH) — post-fix HIGH residual.
_ALWAYS_EXEMPT_PATHS = {
    "/api/v1/health",
    "/api/v1/ready",
    "/api/v1/readiness",
    "/api/v1/translation/readiness",
    # The VLM probe is explicitly documented as a public monitoring surface.
    # Its response contains no tenant/job data and must be reachable by the
    # orchestrator without a tenant credential.
    "/api/v1/search/vlm/health",
    # LT-10: bootstrap HttpOnly session without prior X-API-Key header
    "/api/v1/auth/session",
}
_DETAILED_HEALTH_PATH = "/api/v1/health/detailed"
_DOCS_EXEMPT_PATHS = {"/docs", "/openapi.json", "/redoc"}
# Plan C Phase 1, item C6 -- the federation custody ingest endpoint
# carries its own ``Authorization: Bearer`` auth (against
# ``OCR_FEDERATION_CUSTODY_AUTH_TOKEN``) and is only mounted when
# ``OCR_FEDERATION_CUSTODY_ENABLED=true``.  Exempt it from the global
# API-key middleware so peer clusters can post events without also
# being provisioned with a tenant API key.  The exemption is gated on
# the same env flag so it is a no-op when the feature is off.
_FEDERATION_CUSTODY_PATHS = {"/api/v1/federation/custody/ingest"}
_MAX_BEARER_TOKEN_LENGTH = 64 * 1024
_SESSION_COOKIE_NAME = "ocr_api_key"
_SESSION_PATH = "/api/v1/auth/session"
_SAFE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})


def _canonical_http_origin(value: object) -> tuple[str, str, int] | None:
    """Normalize an HTTP(S) origin; reject opaque origins and URL components."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except (AttributeError, TypeError, ValueError):
        return None
    if (
        scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return None
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, hostname, port


def _request_http_origin(request: Request) -> tuple[str, str, int] | None:
    """Return the request origin from the ASGI URL after proxy normalization."""
    try:
        scheme = request.url.scheme.lower()
        hostname = (request.url.hostname or "").lower().rstrip(".")
        port = request.url.port
    except (AttributeError, TypeError, ValueError):
        return None
    if scheme not in {"http", "https"} or not hostname:
        return None
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, hostname, port


def _cookie_origin_is_trusted(request: Request) -> bool:
    """Trust only same-origin or an exact configured CORS origin, never `*`."""
    origin = _canonical_http_origin(request.headers.get("Origin", ""))
    if origin is None:
        return False
    if origin == _request_http_origin(request):
        return True

    # Read the current config each request so cookie CSRF checks follow the
    # same exact-origin policy as the CORS middleware.
    from api import config as api_config

    for configured in api_config.CORS_ALLOWED_ORIGINS:
        value = str(configured).strip()
        if value != "*" and origin == _canonical_http_origin(value):
            return True
    return False


def _uses_cookie_for_unsafe_auth(
    request: Request, *, explicit_api_key: str, cookie_api_key: str
) -> bool:
    """Detect requests whose unsafe authority comes only from an ambient cookie."""
    if explicit_api_key or not cookie_api_key:
        return False
    if request.method.upper() in _SAFE_HTTP_METHODS:
        return False
    if request.url.path == _SESSION_PATH:
        # Session creation validates a body credential; logout mutates the
        # cookie and must pass the Origin check.
        return request.method.upper() == "DELETE"
    return not _is_exempt_path(request.url.path)


def _csrf_origin_denied(request: Request) -> JSONResponse:
    """Reject an unsafe cookie-authenticated request without exposing secrets."""
    _set_audit_outcome(request, "csrf_origin_denied")
    logger.warning(
        "Rejected unsafe cookie-authenticated request with untrusted Origin: "
        "method=%s path=%s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={
            "error": "csrf_origin_denied",
            "message": "Cookie-authenticated unsafe requests require a trusted Origin",
        },
    )


def _detailed_health_requires_auth() -> bool:
    """True when detailed health must not be unauthenticated."""
    if os.environ.get("REQUIRE_DETAILED_HEALTH_AUTH", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    dep = os.environ.get("DEPLOYMENT_ENV", "").strip().lower()
    return dep in ("production", "prod")


def _is_exempt_path(path: str) -> bool:
    """Return True when ``path`` should skip authentication."""
    if path in _ALWAYS_EXEMPT_PATHS:
        return True
    # Lab may keep detailed health open for operator convenience; production
    # and REQUIRE_DETAILED_HEALTH_AUTH force API key / OAuth.
    if path == _DETAILED_HEALTH_PATH and not _detailed_health_requires_auth():
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


# BE001: single fail-closed auth-enforcement helper shared by HTTP + WS.
# Auth is enforced unless an explicit insecure override applies in an
# explicit local-dev environment.  Default-deny everywhere else.
_UNAUTH_OPEN_WARNED = False


def is_auth_enforced() -> bool:
    """Return True when requests must present credentials (default-deny).

    Returns False only when no API key is configured AND
    ``ALLOW_UNAUTHENTICATED`` is set AND the deployment is an explicit
    local-dev environment (``is_local_dev_environment()``).  Both the HTTP
    middleware and the WebSocket handshake must use this helper so the two
    surfaces always agree (BE001 AC3).
    """
    global _UNAUTH_OPEN_WARNED  # noqa: PLW0603
    if get_api_key():
        return True
    if allow_unauthenticated() and is_local_dev_environment():
        if not _UNAUTH_OPEN_WARNED:
            _UNAUTH_OPEN_WARNED = True
            logger.warning(
                "ALLOW_UNAUTHENTICATED=true in a local-dev environment "
                "— API authentication is disabled; never use outside the lab"
            )
        return False
    return True


# Auth-configuration import-time log runs after the accessors below.


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
    from api import config as api_config

    return bool(api_config.ALLOW_UNAUTHENTICATED)


# Log auth configuration at import time (after accessors are defined).
if not OCR_API_KEY:
    if not is_auth_enforced():
        logger.warning(
            "OCR_API_KEY is not set and ALLOW_UNAUTHENTICATED=true "
            "in local-dev — API authentication is disabled"
        )
    else:
        logger.error(
            "OCR_API_KEY is not set and auth is enforced — API startup must fail"
        )


def validate_auth_configuration() -> None:
    """Fail startup unless auth is configured or explicitly overridden.

    R4-RC-14: multi-tenancy refuses privileged anonymous ladders.
    Read live ``api.config`` values so a prior importer cannot freeze a
    privileged anonymous role into this module's globals.
    """
    from api import config as api_config

    deployment = os.environ.get("DEPLOYMENT_ENV", "").strip().lower()
    if allow_unauthenticated() and deployment in ("production", "prod"):
        raise RuntimeError(
            "ALLOW_UNAUTHENTICATED=true is forbidden in production; configure "
            "OCR_API_KEY, tenant API keys, or OAuth2 instead."
        )

    # Multi-tenant production must never elevate anonymous to admin/operator.
    if api_config.ENABLE_MULTITENANCY and allow_unauthenticated():
        role = (api_config.ANONYMOUS_ROLE or "viewer").strip().lower()
        if role in ("admin", "operator", "platform_admin"):
            raise RuntimeError(
                "ALLOW_UNAUTHENTICATED with ANONYMOUS_ROLE="
                f"{api_config.ANONYMOUS_ROLE!r} is forbidden under "
                "ENABLE_MULTITENANCY (R4-RC-14). Use tenant API keys or set "
                "ANONYMOUS_ROLE=viewer."
            )
        logger.warning(
            "ALLOW_UNAUTHENTICATED=true under multi-tenancy: anonymous role "
            "forced to least privilege (viewer) at identity resolution"
        )

    if get_api_key():
        return
    if not is_auth_enforced():
        logger.warning(
            "Starting with API auth disabled because ALLOW_UNAUTHENTICATED=true "
            "in an explicit local-dev environment (never for production)"
        )
        return
    if api_config.ENABLE_MULTITENANCY:
        logger.info(
            "Multi-tenancy enabled without legacy OCR_API_KEY; "
            "tenant keys will be used for authentication"
        )
        return

    from api.oauth2_config import OAUTH2_ENABLED

    if OAUTH2_ENABLED:
        logger.info(
            "OAuth2 enabled without legacy OCR_API_KEY; bearer-token auth will be used"
        )
        return

    raise RuntimeError(
        "OCR_API_KEY is required unless ALLOW_UNAUTHENTICATED=true is set "
        "with an explicit local-dev DEPLOYMENT_ENV "
        "(local/dev/development/lab/test)."
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
    if len(token.encode("utf-8")) > _MAX_BEARER_TOKEN_LENGTH:
        logger.warning("Rejecting oversized OAuth2 bearer token")
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


def apply_oauth_tenant_binding(state, identity: AuthIdentity) -> None:
    """Bind a verified OAuth tenant claim onto request or websocket state.

    A signed tenant string is not proof that the subject belongs to that
    tenant. Operators bind OAuth subjects to tenant capabilities in
    ``OAUTH2_TENANT_MEMBERSHIP_MAP``; absence or corruption fails closed.
    HTTP and WebSocket must share this helper so a JWT claim cannot pin
    an arbitrary tenant on one surface.
    """
    if not ENABLE_MULTITENANCY:
        return
    from api.oauth2_config import OAUTH2_TENANT_CLAIM

    if not OAUTH2_TENANT_CLAIM:
        return
    raw_tid = identity.claims.get(OAUTH2_TENANT_CLAIM)
    if isinstance(raw_tid, list):
        raw_tid = raw_tid[0] if len(raw_tid) == 1 else None
    if not isinstance(raw_tid, str):
        logger.warning("OAuth token has no usable configured tenant claim")
        return
    tenant_id = raw_tid.strip()
    if not tenant_id or len(tenant_id) > 64 or "\x00" in tenant_id:
        logger.warning("OAuth token has an invalid configured tenant claim")
        return

    try:
        membership_map = json.loads(
            os.environ.get("OAUTH2_TENANT_MEMBERSHIP_MAP", "") or "{}"
        )
    except (json.JSONDecodeError, TypeError):
        logger.error("OAUTH2_TENANT_MEMBERSHIP_MAP is invalid JSON")
        return
    subject_memberships = (
        membership_map.get(identity.subject)
        if isinstance(membership_map, dict)
        else None
    )
    permissions = None
    if isinstance(subject_memberships, dict):
        permissions = subject_memberships.get(tenant_id)
    if not isinstance(permissions, list):
        logger.warning("OAuth subject is not configured for claimed tenant")
        return
    parsed_permissions = _parse_tenant_permissions(json.dumps(permissions))
    if parsed_permissions is None:
        logger.warning("OAuth tenant membership has invalid permissions")
        return
    try:
        from api.database import Tenant, get_session_factory

        session = get_session_factory()()
        try:
            exists = (
                session.query(Tenant.tenant_id)
                .filter(Tenant.tenant_id == tenant_id)
                .first()
            )
        finally:
            session.close()
    except Exception:
        logger.exception("OAuth tenant existence lookup failed")
        return
    if not exists:
        logger.warning("OAuth token claims an unknown tenant")
        return
    state.tenant_id = tenant_id
    state.tenant_permissions = parsed_permissions


def _bind_oauth_tenant(request: Request, identity: AuthIdentity) -> None:
    """Bind the verified OAuth tenant claim to HTTP request scope."""
    apply_oauth_tenant_binding(request.state, identity)


def _identity_role_from_permissions(permissions: list[str]) -> str:
    """Map tenant API-key permissions onto the RBAC role model."""
    if "platform_admin" in permissions or "admin" in permissions:
        return "admin"
    if "submit" in permissions:
        return "operator"
    return "viewer"


from api.authz import VALID_TENANT_PERMISSIONS as _VALID_TENANT_PERMISSIONS


def _parse_tenant_permissions(raw_permissions: object) -> list[str] | None:
    """Parse a stored permission document, failing closed on corruption."""
    if not isinstance(raw_permissions, str):
        return None
    try:
        decoded = json.loads(raw_permissions)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(decoded, list):
        return None
    if not decoded or len(decoded) > len(_VALID_TENANT_PERMISSIONS):
        return None
    permissions = []
    for value in decoded:
        if not isinstance(value, str) or value not in _VALID_TENANT_PERMISSIONS:
            return None
        if value in permissions:
            return None
        permissions.append(value)
    return permissions


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

        permissions = _parse_tenant_permissions(raw_permissions)
        if permissions is None:
            logger.warning(
                "Rejecting tenant API key %s with malformed permissions",
                key_id,
            )
            return False
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
    explicit_api_key = request.headers.get("X-API-Key", "")
    cookie_api_key = request.cookies.get(_SESSION_COOKIE_NAME, "") or ""
    if _uses_cookie_for_unsafe_auth(
        request,
        explicit_api_key=explicit_api_key,
        cookie_api_key=cookie_api_key,
    ) and not _cookie_origin_is_trusted(request):
        return _csrf_origin_denied(request)

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
    # LT-10: optional HttpOnly session cookie (set by POST /api/v1/auth/session).
    if not provided_key:
        provided_key = request.cookies.get("ocr_api_key", "") or ""

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

            # Under multitenancy, legacy OCR_API_KEY is an explicit break-glass
            # platform principal. Attach server-built permissions so admin/data
            # planes never treat bare ``permissions is None`` as platform
            # (that fail-open also elevated OAuth/anonymous — R4-RC-12 / RC-04).
            legacy_perms: list[str] = []
            if ENABLE_MULTITENANCY:
                from api.authz import LEGACY_PLATFORM_PERMISSIONS

                legacy_perms = list(LEGACY_PLATFORM_PERMISSIONS)
                request.state.tenant_permissions = legacy_perms
                request.state.tenant_id = None
            request.state.identity = AuthIdentity(
                subject="apikey",
                role=APIKEY_ROLE,
                auth_method="apikey",
                claims={
                    "legacy_ocr_api_key": True,
                    "permissions": legacy_perms,
                },
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
        _bind_oauth_tenant(request, bearer_identity)
        _set_audit_outcome(request, "authorized")
        return await call_next(request)

    if not api_key:
        from api.oauth2_config import OAUTH2_ENABLED

        if is_auth_enforced():
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
            # BE001 AC1: fail closed with 401 (not 200, and without leaking
            # server configuration state in the response body).
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "unauthorized",
                    "message": "Invalid or missing API key",
                },
            )

        # R4-RC-14: same demotion rules as get_identity (never raw admin under MT).
        from api.identity import _anonymous_role

        request.state.identity = AuthIdentity(
            subject="anonymous",
            role=_anonymous_role(),
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
