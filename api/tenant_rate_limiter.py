"""Per-tenant rate limiting with admin-configurable overrides.

Provides a dynamic rate limit resolver that extracts tenant_id from
the authenticated request and looks up tenant-specific rate limits
from the database.  Falls back to the global default when no
tenant-specific limit is configured.

Rate limit strings follow slowapi/limits format: "100/minute",
"500/hour", "10/second", etc.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

from starlette.requests import Request

from api.config import ENABLE_MULTITENANCY

logger = logging.getLogger(__name__)

# Validation regex for rate limit strings (e.g. "100/minute", "10/second")
_RATE_LIMIT_RE = re.compile(
    r"^\d+/(second|minute|hour|day|month|year)$"
)

# In-memory cache of tenant_id -> rate_limit_string.
# Refreshed periodically from the database.
_tenant_rate_limits: dict[str, str] = {}
_cache_lock = threading.RLock()
_cache_last_refreshed: float = 0.0

# Cache TTL in seconds (how often to re-read from DB)
_CACHE_TTL_SECONDS = 60


def validate_rate_limit_string(rate_limit: str) -> bool:
    """Return True if the rate limit string matches the expected format."""
    return bool(_RATE_LIMIT_RE.match(rate_limit))


def _refresh_cache_if_needed() -> None:
    """Reload tenant rate limits from the database if the cache is stale."""
    global _cache_last_refreshed

    now = time.monotonic()
    if now - _cache_last_refreshed < _CACHE_TTL_SECONDS:
        return

    with _cache_lock:
        # Double-check after acquiring lock
        if time.monotonic() - _cache_last_refreshed < _CACHE_TTL_SECONDS:
            return
        try:
            _load_tenant_rate_limits()
            _cache_last_refreshed = time.monotonic()
        except Exception:
            logger.exception("Failed to refresh tenant rate limit cache")


def _load_tenant_rate_limits() -> None:
    """Read all TenantRateLimit rows from the database into the cache."""
    from api.database import TenantRateLimit, get_session_factory

    session = get_session_factory()()
    try:
        rows = session.query(TenantRateLimit).all()
        new_cache: dict[str, str] = {}
        for row in rows:
            if row.rate_limit and validate_rate_limit_string(row.rate_limit):
                new_cache[row.tenant_id] = row.rate_limit
        with _cache_lock:
            _tenant_rate_limits.clear()
            _tenant_rate_limits.update(new_cache)
        logger.debug(
            "Refreshed tenant rate limit cache: %d entries", len(new_cache)
        )
    finally:
        session.close()


def set_tenant_rate_limit(
    tenant_id: str,
    rate_limit: str,
) -> None:
    """Persist a tenant-specific rate limit and update the in-memory cache.

    Args:
        tenant_id: The tenant identifier (e.g. "tenant_abc123def456").
        rate_limit: Rate limit string (e.g. "100/minute").

    Raises:
        ValueError: If rate_limit does not match the expected format.
    """
    if not validate_rate_limit_string(rate_limit):
        raise ValueError(
            f"Invalid rate limit format: {rate_limit!r}. "
            "Expected format: '<number>/<period>' "
            "(e.g. '100/minute', '500/hour')."
        )

    from api.database import TenantRateLimit, get_session_factory

    session = get_session_factory()()
    try:
        existing = session.get(TenantRateLimit, tenant_id)
        if existing:
            existing.rate_limit = rate_limit
        else:
            row = TenantRateLimit(tenant_id=tenant_id, rate_limit=rate_limit)
            session.add(row)
        session.commit()
        logger.info(
            "Set rate limit for tenant %s: %s", tenant_id, rate_limit
        )
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # Immediately update the in-memory cache so the change takes effect
    with _cache_lock:
        _tenant_rate_limits[tenant_id] = rate_limit


def get_tenant_rate_limit(tenant_id: str) -> Optional[str]:
    """Return the configured rate limit for a tenant, or None if unset."""
    _refresh_cache_if_needed()
    with _cache_lock:
        return _tenant_rate_limits.get(tenant_id)


def delete_tenant_rate_limit(tenant_id: str) -> bool:
    """Remove a tenant-specific rate limit. Returns True if a row was deleted."""
    from api.database import TenantRateLimit, get_session_factory

    session = get_session_factory()()
    try:
        existing = session.get(TenantRateLimit, tenant_id)
        if not existing:
            return False
        session.delete(existing)
        session.commit()
        logger.info("Deleted rate limit for tenant %s", tenant_id)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    with _cache_lock:
        _tenant_rate_limits.pop(tenant_id, None)
    return True


def invalidate_cache() -> None:
    """Force cache refresh on next access. Useful for tests."""
    global _cache_last_refreshed
    with _cache_lock:
        _tenant_rate_limits.clear()
        _cache_last_refreshed = 0.0


# ---------------------------------------------------------------------------
# slowapi integration: key function and dynamic limit resolvers
# ---------------------------------------------------------------------------


def tenant_key_func(request: Request) -> str:
    """Extract a rate-limit key from the request.

    For multi-tenant requests, the key is the tenant_id so that each
    tenant has an independent rate limit bucket.  For non-tenant
    requests, falls back to the remote IP address.
    """
    if ENABLE_MULTITENANCY:
        tenant_id = getattr(request.state, "tenant_id", None)
        if tenant_id:
            return tenant_id

    # Fallback: use remote address (same as slowapi default)
    from slowapi.util import get_remote_address

    return get_remote_address(request)


def _make_dynamic_limit_resolver(fallback_rate_func):
    """Create a callable that returns a per-tenant or default rate limit.

    Args:
        fallback_rate_func: A zero-arg callable returning the default
            rate limit string (e.g. ``get_default_rate``).

    Returns:
        A callable ``(request: str) -> str`` suitable for use as the
        ``limit_value`` argument to ``@limiter.limit(...)``.

    Note: slowapi passes the key_func result (a string) as the first
    argument to callable limit values, not the Request object. So this
    resolver receives the key string.
    """

    def _resolve(key: str) -> str:
        if ENABLE_MULTITENANCY and key.startswith("tenant_"):
            tenant_limit = get_tenant_rate_limit(key)
            if tenant_limit:
                return tenant_limit
        return fallback_rate_func()

    return _resolve


def get_dynamic_default_rate():
    """Dynamic limit resolver for read/default endpoints.

    Returns a callable suitable for ``@limiter.limit(get_dynamic_default_rate())``.
    """
    from api.limits import get_default_rate

    return _make_dynamic_limit_resolver(get_default_rate)


def get_dynamic_submit_rate():
    """Dynamic limit resolver for submit endpoints.

    Returns a callable suitable for ``@limiter.limit(get_dynamic_submit_rate())``.
    """
    from api.limits import get_submit_rate

    return _make_dynamic_limit_resolver(get_submit_rate)
