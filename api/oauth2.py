"""OAuth2 JWT validation with JWKS caching.

Handles:
- JWKS fetching from OIDC issuer with 1-hour cache TTL
- JWT signature verification, expiry, audience, and issuer checks
- Role extraction from configurable JWT claim path
- Graceful degradation when JWKS endpoint is unreachable
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy imports — these are optional dependencies
_pyjwt = None
_pyjwt_exceptions = None
_httpx = None

_JWKS_CACHE_TTL = 3600  # 1 hour


def _ensure_jwt():
    """Lazy-import PyJWT so the module can be imported even without it."""
    global _pyjwt, _pyjwt_exceptions
    if _pyjwt is None:
        try:
            import jwt as pyjwt_mod
            import jwt.exceptions as pyjwt_exc

            _pyjwt = pyjwt_mod
            _pyjwt_exceptions = pyjwt_exc
        except ImportError:
            raise ImportError(
                "PyJWT is required for OAuth2 support. "
                "Install with: pip install PyJWT[crypto]"
            )
    return _pyjwt, _pyjwt_exceptions


def _ensure_httpx():
    """Lazy-import httpx for JWKS fetching."""
    global _httpx
    if _httpx is None:
        try:
            import httpx as httpx_mod

            _httpx = httpx_mod
        except ImportError:
            raise ImportError(
                "httpx is required for OAuth2 JWKS fetching. "
                "Install with: pip install httpx"
            )
    return _httpx


class JWKSCache:
    """Thread-safe JWKS key cache with configurable TTL.

    Fetches the JSON Web Key Set from the issuer's JWKS URI and caches
    the keys.  Automatically refreshes when the cache expires.
    """

    def __init__(self, jwks_uri: str, ttl: int = _JWKS_CACHE_TTL):
        self._jwks_uri = jwks_uri
        self._ttl = ttl
        self._keys: list[dict[str, Any]] = []
        self._fetched_at: float = 0.0
        self._lock = threading.Lock()

    @property
    def jwks_uri(self) -> str:
        return self._jwks_uri

    def get_keys(self) -> list[dict[str, Any]]:
        """Return cached JWKS keys, refreshing if expired."""
        now = time.time()
        if now - self._fetched_at < self._ttl and self._keys:
            return self._keys

        with self._lock:
            # Double-check after acquiring lock
            if now - self._fetched_at < self._ttl and self._keys:
                return self._keys
            return self._refresh()

    def _refresh(self) -> list[dict[str, Any]]:
        """Fetch JWKS from the remote endpoint."""
        httpx = _ensure_httpx()
        try:
            resp = httpx.get(self._jwks_uri, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            self._keys = data.get("keys", [])
            self._fetched_at = time.time()
            logger.info(
                "JWKS cache refreshed from %s (%d keys)",
                self._jwks_uri,
                len(self._keys),
            )
        except Exception:
            logger.warning(
                "Failed to fetch JWKS from %s — using stale cache (%d keys)",
                self._jwks_uri,
                len(self._keys),
                exc_info=True,
            )
        return self._keys

    def invalidate(self) -> None:
        """Force cache expiry so the next call to get_keys() refreshes."""
        with self._lock:
            self._fetched_at = 0.0


# Module-level singleton (initialized lazily by validate_jwt)
_jwks_cache: Optional[JWKSCache] = None
_jwks_cache_lock = threading.Lock()


def _get_jwks_cache() -> JWKSCache:
    """Return the module-level JWKSCache singleton, creating it if needed."""
    global _jwks_cache
    if _jwks_cache is not None:
        return _jwks_cache

    with _jwks_cache_lock:
        if _jwks_cache is not None:
            return _jwks_cache

        from api.oauth2_config import OAUTH2_ISSUER, OAUTH2_JWKS_URI

        jwks_uri = OAUTH2_JWKS_URI
        if not jwks_uri and OAUTH2_ISSUER:
            jwks_uri = OAUTH2_ISSUER.rstrip("/") + "/.well-known/jwks.json"

        if not jwks_uri:
            raise RuntimeError(
                "OAUTH2_JWKS_URI or OAUTH2_ISSUER must be configured "
                "when OAUTH2_ENABLED=true"
            )

        _jwks_cache = JWKSCache(jwks_uri)
        return _jwks_cache


def reset_jwks_cache() -> None:
    """Reset the module-level JWKS cache (for testing)."""
    global _jwks_cache
    with _jwks_cache_lock:
        _jwks_cache = None


def set_jwks_cache(cache: JWKSCache) -> None:
    """Replace the module-level JWKS cache (for testing)."""
    global _jwks_cache
    with _jwks_cache_lock:
        _jwks_cache = cache


def validate_jwt(token: str) -> dict[str, Any]:
    """Decode and validate a JWT token using the cached JWKS.

    Returns the decoded payload dict on success.
    Raises ``ValueError`` on any validation failure.
    """
    pyjwt, pyjwt_exc = _ensure_jwt()

    from api.oauth2_config import (
        OAUTH2_ALGORITHMS,
        OAUTH2_AUDIENCE,
        OAUTH2_ISSUER,
    )

    cache = _get_jwks_cache()
    keys = cache.get_keys()

    if not keys:
        raise ValueError("No JWKS keys available — cannot validate JWT")

    # Build options
    options: dict[str, Any] = {}
    if not OAUTH2_AUDIENCE:
        options["verify_aud"] = False

    # Build a PyJWKSet from the cached keys and find the signing key
    try:
        jwk_set = pyjwt.PyJWKSet.from_dict({"keys": keys})
    except (pyjwt_exc.PyJWKSetError, pyjwt_exc.PyJWKError) as exc:
        raise ValueError(f"Invalid JWKS keys: {exc}")

    # Try to match the signing key by kid from the token header
    try:
        unverified_header = pyjwt.get_unverified_header(token)
    except pyjwt_exc.DecodeError as exc:
        raise ValueError(f"JWT validation failed: {exc}")

    token_kid = unverified_header.get("kid")
    signing_key = None

    if token_kid:
        for key in jwk_set.keys:
            if key.key_id == token_kid:
                signing_key = key
                break
    if signing_key is None:
        # No kid match or no kid in token — use first key
        signing_key = jwk_set.keys[0]

    try:
        payload = pyjwt.decode(
            token,
            signing_key.key,
            algorithms=OAUTH2_ALGORITHMS,
            audience=OAUTH2_AUDIENCE or None,
            issuer=OAUTH2_ISSUER or None,
            options=options,
        )
        return payload
    except pyjwt_exc.ExpiredSignatureError:
        raise ValueError("JWT token has expired")
    except (
        pyjwt_exc.InvalidIssuerError,
        pyjwt_exc.InvalidAudienceError,
        pyjwt_exc.MissingRequiredClaimError,
    ) as exc:
        raise ValueError(f"JWT claims validation failed: {exc}")
    except pyjwt_exc.PyJWTError as exc:
        raise ValueError(f"JWT validation failed: {exc}")


def extract_role(claims: dict[str, Any]) -> str:
    """Extract the user's role from JWT claims.

    Looks at the configured OAUTH2_ROLE_CLAIM path.  If the claim is a
    list, returns the highest-privilege role found.  Falls back to
    OAUTH2_DEFAULT_ROLE if no matching role is found.
    """
    from api.oauth2_config import (
        OAUTH2_ADMIN_ROLE,
        OAUTH2_DEFAULT_ROLE,
        OAUTH2_OPERATOR_ROLE,
        OAUTH2_ROLE_CLAIM,
        OAUTH2_VIEWER_ROLE,
    )

    role_value = claims.get(OAUTH2_ROLE_CLAIM)
    if role_value is None:
        return OAUTH2_DEFAULT_ROLE

    # Map external role names to internal roles (priority order)
    role_map = {
        OAUTH2_ADMIN_ROLE: "admin",
        OAUTH2_OPERATOR_ROLE: "operator",
        OAUTH2_VIEWER_ROLE: "viewer",
    }

    # If the claim is a list of roles, pick the highest-privilege match
    if isinstance(role_value, list):
        for external_role, internal_role in role_map.items():
            if external_role in role_value:
                return internal_role
        return OAUTH2_DEFAULT_ROLE

    # Single string value
    if isinstance(role_value, str):
        return role_map.get(role_value, OAUTH2_DEFAULT_ROLE)

    return OAUTH2_DEFAULT_ROLE
