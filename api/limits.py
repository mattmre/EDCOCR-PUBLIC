"""Rate limiting configuration using slowapi."""

from __future__ import annotations

import ipaddress
import os

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from api.config import ENABLE_MULTITENANCY

_DEFAULT_RATE = os.environ.get("OCR_RATE_LIMIT", "60/minute")
_SUBMIT_RATE = os.environ.get("OCR_SUBMIT_RATE_LIMIT", "10/minute")

# PH3-11: multi-worker rate-limit theater residual.
# When RATE_LIMIT_STORAGE_URI is set (e.g. redis://...), slowapi uses shared
# storage. Production without it logs a one-shot warning — counters remain
# per-process (honest, not claimed multi-worker-correct).
_RATE_LIMIT_STORAGE_URI = os.environ.get("RATE_LIMIT_STORAGE_URI", "").strip()
_rate_limit_multiworker_warned = False


def _trusted_proxies() -> set[str]:
    """Comma-separated TRUSTED_PROXIES host/IP list (empty = never trust XFF)."""
    raw = os.environ.get("TRUSTED_PROXIES", "").strip()
    if not raw:
        return set()
    return {p.strip() for p in raw.split(",") if p.strip()}


def _proxy_is_trusted(request: Request) -> bool:
    """Only honor XFF/X-Real-IP when peer is in TRUSTED_PROXIES (B04)."""
    trusted = _trusted_proxies()
    if not trusted:
        return False
    host = getattr(getattr(request, "client", None), "host", None) or ""
    return host in trusted


def _sanitize_ip_token(token: str) -> str | None:
    """Return IP string if parseable; reject tenant-shaped spoof values (B04)."""
    t = (token or "").strip()
    if not t:
        return None
    # Never accept tenant_* as an "IP" (bucket collision with tenant keys).
    if t.lower().startswith("tenant_") or t.lower().startswith("tenant:"):
        return None
    try:
        ipaddress.ip_address(t)
        return t
    except ValueError:
        # Allow hostnames only if they do not look like tenant ids
        if ":" in t or t.replace(".", "").isdigit():
            return None
        if t.lower().startswith("tenant"):
            return None
        return t


def _get_real_client_ip(request: Request) -> str:
    """Extract real client IP, respecting proxy headers only when trusted.

    B04: Without ``TRUSTED_PROXIES``, X-Forwarded-For / X-Real-IP are ignored
    (peer address only). Spoofed ``tenant_*`` values never become keys.
    """
    if _proxy_is_trusted(request):
        xff = request.headers.get("x-forwarded-for")
        if xff:
            client_ip = _sanitize_ip_token(xff.split(",")[0])
            if client_ip:
                return client_ip

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            client_ip = _sanitize_ip_token(real_ip)
            if client_ip:
                return client_ip

    return get_remote_address(request)


def _tenant_aware_key_func(request: Request) -> str:
    """Rate-limit key with explicit namespaces (B04).

    - Authenticated tenant → ``tenant:{id}``
    - Otherwise → ``ip:{addr}``
    Never collides tenant buckets with spoofable IP/header material.
    """
    if ENABLE_MULTITENANCY:
        tenant_id = getattr(request.state, "tenant_id", None)
        if tenant_id:
            return f"tenant:{tenant_id}"
    return f"ip:{_get_real_client_ip(request)}"


def assert_rate_limit_storage_policy() -> None:
    """PH3-11 production fail-closed for multi-worker rate-limit storage.

    Call from app startup (``api.main``). Raises RuntimeError when production
    lacks ``RATE_LIMIT_STORAGE_URI`` unless ``RATE_LIMIT_ALLOW_MEMORY`` is set.
    """
    if _RATE_LIMIT_STORAGE_URI:
        return
    dep = os.environ.get("DEPLOYMENT_ENV", "").strip().lower()
    allow_mem = os.environ.get("RATE_LIMIT_ALLOW_MEMORY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if dep in ("production", "prod") and not allow_mem:
        raise RuntimeError(
            "PH3-11: RATE_LIMIT_STORAGE_URI is required under "
            "DEPLOYMENT_ENV=production (shared multi-worker rate-limit "
            "buckets). Set RATE_LIMIT_STORAGE_URI=redis://... or set "
            "RATE_LIMIT_ALLOW_MEMORY=true only for non-shared lab theater."
        )


def _build_limiter() -> Limiter:
    """Construct slowapi Limiter with optional shared Redis storage (PH3-11)."""
    global _rate_limit_multiworker_warned
    kwargs: dict = {"key_func": _tenant_aware_key_func}
    if _RATE_LIMIT_STORAGE_URI:
        kwargs["storage_uri"] = _RATE_LIMIT_STORAGE_URI
    else:
        dep = os.environ.get("DEPLOYMENT_ENV", "").strip().lower()
        allow_mem = os.environ.get("RATE_LIMIT_ALLOW_MEMORY", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if (
            dep in ("production", "prod")
            and allow_mem
            and not _rate_limit_multiworker_warned
        ):
            import logging

            logging.getLogger(__name__).warning(
                "PH3-11: RATE_LIMIT_ALLOW_MEMORY under production — "
                "slowapi counters are per-process (uneven multi-worker)."
            )
            _rate_limit_multiworker_warned = True
    return Limiter(**kwargs)


limiter = _build_limiter()


def get_default_rate() -> str:
    """Default rate limit string for read endpoints."""
    return _DEFAULT_RATE


def get_submit_rate() -> str:
    """Rate limit string for submit endpoint."""
    return _SUBMIT_RATE
