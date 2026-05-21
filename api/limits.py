"""Rate limiting configuration using slowapi."""

import os

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from api.config import ENABLE_MULTITENANCY

_DEFAULT_RATE = os.environ.get("OCR_RATE_LIMIT", "60/minute")
_SUBMIT_RATE = os.environ.get("OCR_SUBMIT_RATE_LIMIT", "10/minute")


def _get_real_client_ip(request: Request) -> str:
    """Extract real client IP, respecting proxy headers.

    Checks ``X-Forwarded-For`` (first/leftmost entry = original client),
    then ``X-Real-IP``, then falls back to ``request.client.host``.

    .. warning::
        These headers can be spoofed by clients.  In production behind a
        trusted reverse proxy, configure ``TRUSTED_PROXIES`` to restrict
        which upstream addresses are allowed to set forwarding headers.

    # TODO: Add TRUSTED_PROXIES env var to only honour forwarding headers
    # when ``request.client.host`` is in the trusted set.  Without this,
    # an attacker can spoof X-Forwarded-For to bypass per-IP rate limits.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        # Leftmost IP is the original client; rightmost is the last proxy
        client_ip = xff.split(",")[0].strip()
        if client_ip:
            return client_ip

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        client_ip = real_ip.strip()
        if client_ip:
            return client_ip

    return get_remote_address(request)


def _tenant_aware_key_func(request: Request) -> str:
    """Extract rate-limit key: tenant_id when multi-tenancy is active, else IP.

    This is the default key_func for the global limiter. When multi-tenancy
    is enabled and a request has been authenticated with a tenant API key,
    the tenant_id is used as the rate-limit bucket key so each tenant gets
    an independent rate limit. Otherwise, falls back to real client IP
    (proxy-aware via ``_get_real_client_ip``).
    """
    if ENABLE_MULTITENANCY:
        tenant_id = getattr(request.state, "tenant_id", None)
        if tenant_id:
            return tenant_id
    return _get_real_client_ip(request)


limiter = Limiter(key_func=_tenant_aware_key_func)


def get_default_rate() -> str:
    """Default rate limit string for read endpoints."""
    return _DEFAULT_RATE


def get_submit_rate() -> str:
    """Rate limit string for submit endpoint."""
    return _SUBMIT_RATE
