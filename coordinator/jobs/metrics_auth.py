"""Shared authentication helpers for metrics and coordinator search endpoints.

RC-06 / R3-RC-02 / R4-RC-02: empty ``METRICS_API_KEY`` must not open PII
search surfaces (entities / forms / semantic chunk text). Prometheus-style
scrape endpoints may still use :func:`has_valid_metrics_key` with the
historical empty-key open-lab behaviour; PII surfaces use
:func:`require_metrics_api_key` (fail-closed when unset).
"""

import hashlib
import hmac
import json
import os
import secrets
import time

_MAX_TENANT_ID_LENGTH = 128
_MAX_SCOPE_HEADER_LENGTH = 512
_TENANT_SCOPE_TTL_SECONDS = 300

METRICS_API_KEY = os.environ.get("METRICS_API_KEY", "")


def _extract_bearer_token(request) -> str:
    """Extract Bearer token from Authorization header."""
    auth = request.META.get("HTTP_AUTHORIZATION", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def _safe_compare_metrics_key(provided: str, expected: str) -> bool:
    """Compare arbitrary request input without letting ``compare_digest`` raise.

    Django exposes HTTP headers as strings.  ``secrets.compare_digest`` raises
    ``TypeError`` for non-ASCII string operands, so an unauthenticated caller
    could turn a malformed API key into a coordinator 500.  UTF-8 byte
    comparison preserves constant-time behaviour for valid input and treats
    malformed/non-string values as a failed authentication attempt.
    """
    if not isinstance(provided, str) or not isinstance(expected, str):
        return False
    try:
        return secrets.compare_digest(
            provided.encode("utf-8"),
            expected.encode("utf-8"),
        )
    except (UnicodeEncodeError, TypeError, AttributeError):
        return False


def _key_matches(request) -> bool:
    """True when a presented key matches a configured metrics credential."""
    return _key_matches_tenant(request, None)


def _presented_metrics_keys(request) -> tuple[str, ...]:
    """Return all presented credentials so either supported header can match."""
    values: list[str] = []
    api_key = request.META.get("HTTP_X_API_KEY", "")
    if isinstance(api_key, str) and api_key:
        values.append(api_key)
    bearer = _extract_bearer_token(request)
    if bearer:
        values.append(bearer)
    return tuple(values)


def _tenant_key_map() -> dict[str, str]:
    """Decode optional tenant-to-key bindings without exposing key material."""
    raw = os.environ.get("METRICS_TENANT_KEYS_JSON", "").strip()
    if not raw or len(raw.encode("utf-8")) > 64 * 1024:
        return {}
    try:
        decoded = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {
        str(tenant_id).strip(): str(key)
        for tenant_id, key in decoded.items()
        if isinstance(tenant_id, str)
        and tenant_id.strip()
        and isinstance(key, str)
        and key
        and len(tenant_id.strip()) <= _MAX_TENANT_ID_LENGTH
    }


def _signed_tenant_scope(request) -> str | None:
    """Verify an optional short-lived tenant scope assertion.

    Format: ``tenant_id|unix_timestamp|hex_hmac`` where the HMAC covers the
    first two fields and uses ``METRICS_TENANT_SCOPE_SECRET``.  A global
    metrics key may authenticate the request, but it never supplies tenant
    identity by itself.
    """
    secret = os.environ.get("METRICS_TENANT_SCOPE_SECRET", "").strip()
    raw = request.META.get("HTTP_X_TENANT_SCOPE", "")
    if (
        not secret
        or not isinstance(raw, str)
        or not raw
        or len(raw) > _MAX_SCOPE_HEADER_LENGTH
    ):
        return None
    parts = raw.split("|")
    if len(parts) != 3:
        return None
    tenant_id, timestamp_text, signature = (part.strip() for part in parts)
    if not tenant_id or len(tenant_id) > _MAX_TENANT_ID_LENGTH:
        return None
    try:
        timestamp = int(timestamp_text)
    except (TypeError, ValueError):
        return None
    if abs(int(time.time()) - timestamp) > _TENANT_SCOPE_TTL_SECONDS:
        return None
    if len(signature) != hashlib.sha256().digest_size * 2:
        return None
    message = f"{tenant_id}|{timestamp}".encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature.lower(), expected):
        return None
    return tenant_id


def _multitenancy_enabled() -> bool:
    return os.environ.get("ENABLE_MULTITENANCY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _tenant_binding_required() -> bool:
    """Require an authenticated tenant binding on shared PII surfaces."""
    return bool(
        _multitenancy_enabled()
        or _tenant_key_map()
        or os.environ.get("METRICS_TENANT_SCOPE_SECRET", "").strip()
    )


def _key_matches_tenant(request, tenant_id: str | None) -> bool:
    provided_values = _presented_metrics_keys(request)
    if not provided_values:
        return False
    expected_values: list[str] = []
    if METRICS_API_KEY:
        expected_values.append(METRICS_API_KEY)
    if tenant_id:
        expected = _tenant_key_map().get(tenant_id)
        if expected:
            expected_values.append(expected)
    return any(
        _safe_compare_metrics_key(provided, expected)
        for provided in provided_values
        for expected in expected_values
    )


def has_valid_metrics_key(request) -> bool:
    """Auth for scrape-style metrics endpoints.

    When ``METRICS_API_KEY`` is empty, returns True (lab open scrape). When
    set, requires a matching X-Api-Key or Bearer token.
    """
    if not METRICS_API_KEY:
        # An open scrape is a deliberate lab convenience only.  Production
        # must configure a dedicated metrics credential rather than silently
        # exposing tenant/job aggregates because an environment variable was
        # omitted.
        deployment = os.environ.get("DEPLOYMENT_ENV", "").strip().lower()
        return deployment not in {"production", "prod"}
    return _key_matches(request)


def require_metrics_api_key(request) -> bool:
    """Fail-closed auth for PII-bearing coordinator search APIs.

    Empty ``METRICS_API_KEY`` never authorizes entity/form/semantic dump
    surfaces (RC-06 amp). Callers must set a non-empty key in production.
    """
    requested = _raw_tenant_scope(request)
    if _tenant_binding_required():
        bound = _signed_tenant_scope(request)
        if bound is None:
            # A tenant-to-key map can bind identity without a signed header.
            bound = requested if requested in _tenant_key_map() else None
        if bound is None or (requested and requested != bound):
            return False
        return _key_matches_tenant(request, bound)
    if not METRICS_API_KEY:
        return False
    return _key_matches(request)


def extract_tenant_scope(request, body: dict | None = None) -> str | None:
    """Return a caller-bound tenant scope, never a free-form tenant selector."""
    requested = _raw_tenant_scope(request, body)
    if not _tenant_binding_required():
        return requested

    bound = _signed_tenant_scope(request)
    if bound is None:
        tenant_keys = _tenant_key_map()
        if requested in tenant_keys and _key_matches_tenant(request, requested):
            bound = requested
    if bound is None or (requested and requested != bound):
        return None
    return bound


def _raw_tenant_scope(request, body: dict | None = None) -> str | None:
    """Read the requested tenant selector before authorization binding."""
    header = request.META.get("HTTP_X_TENANT_ID", "").strip()
    if header:
        return header[:_MAX_TENANT_ID_LENGTH]
    q = request.GET.get("tenant_id", "").strip()
    if q:
        return q[:_MAX_TENANT_ID_LENGTH]
    if isinstance(body, dict):
        tid = body.get("tenant_id")
        if isinstance(tid, str) and tid.strip():
            return tid.strip()[:_MAX_TENANT_ID_LENGTH]
    return None
