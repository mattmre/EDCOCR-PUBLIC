"""Shared authentication helpers for metrics endpoints."""

import os
import secrets

METRICS_API_KEY = os.environ.get("METRICS_API_KEY", "")


def _extract_bearer_token(request) -> str:
    """Extract Bearer token from Authorization header."""
    auth = request.META.get("HTTP_AUTHORIZATION", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def has_valid_metrics_key(request) -> bool:
    """Check if request has a valid metrics API key via X-Api-Key or Bearer."""
    if not METRICS_API_KEY:
        return True
    api_key = request.META.get("HTTP_X_API_KEY", "")
    if api_key and secrets.compare_digest(api_key, METRICS_API_KEY):
        return True
    bearer = _extract_bearer_token(request)
    if bearer and secrets.compare_digest(bearer, METRICS_API_KEY):
        return True
    return False
