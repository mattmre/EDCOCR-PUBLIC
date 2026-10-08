"""Fail-closed adapter for external review strong-auth verification."""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

_MAX_VERIFIER_RESPONSE_BYTES = 64 * 1024
_ALLOWED_METHODS = frozenset({"piv_cac", "oidc_mfa", "hardware_token"})


class StrongAuthUnavailable(RuntimeError):
    """Raised when no trustworthy verifier can establish the proof."""


@dataclass(frozen=True)
class StrongAuthProof:
    subject: str
    auth_method: str
    proof_id: str


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


def verify_review_strong_auth(
    *,
    token: str,
    auth_method: str,
    review_id: str,
    job_id: str,
    tenant_id: str,
    expected_subject: str,
) -> StrongAuthProof:
    """Verify a one-time proof through the operator-configured HTTPS service."""
    verifier_url = os.environ.get("REVIEW_STRONG_AUTH_VERIFIER_URL", "").strip()
    if not verifier_url:
        raise StrongAuthUnavailable("strong-auth verifier is not configured")
    parsed = urllib.parse.urlsplit(verifier_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username:
        raise StrongAuthUnavailable("strong-auth verifier must be an HTTPS URL")
    if auth_method not in _ALLOWED_METHODS:
        raise StrongAuthUnavailable("unsupported strong-auth method")

    payload = json.dumps(
        {
            "auth_method": auth_method,
            "review_id": review_id,
            "job_id": job_id,
            "tenant_id": tenant_id,
            "expected_subject": expected_subject,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        verifier_url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    ca_file = os.environ.get("REVIEW_STRONG_AUTH_CA_FILE", "").strip() or None
    ssl_context = ssl.create_default_context(cafile=ca_file)
    opener = urllib.request.build_opener(
        _NoRedirect(), urllib.request.HTTPSHandler(context=ssl_context)
    )
    try:
        with opener.open(request, timeout=5.0) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > _MAX_VERIFIER_RESPONSE_BYTES:
                raise StrongAuthUnavailable("strong-auth response exceeds size limit")
            raw = response.read(_MAX_VERIFIER_RESPONSE_BYTES + 1)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise StrongAuthUnavailable("strong-auth verifier request failed") from exc
    if len(raw) > _MAX_VERIFIER_RESPONSE_BYTES:
        raise StrongAuthUnavailable("strong-auth response exceeds size limit")
    try:
        result = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise StrongAuthUnavailable(
            "strong-auth verifier returned invalid JSON"
        ) from exc
    if not isinstance(result, dict) or result.get("active") is not True:
        raise StrongAuthUnavailable("strong-auth proof is inactive")
    subject = str(result.get("subject") or "").strip()
    verified_method = str(result.get("auth_method") or "").strip()
    proof_id = str(result.get("proof_id") or "").strip()
    if not subject or subject != expected_subject:
        raise StrongAuthUnavailable("strong-auth subject does not match requester")
    if verified_method != auth_method or verified_method not in _ALLOWED_METHODS:
        raise StrongAuthUnavailable("strong-auth method does not match request")
    if not proof_id or len(proof_id) > 256:
        raise StrongAuthUnavailable("strong-auth verifier omitted bounded proof id")
    return StrongAuthProof(
        subject=subject, auth_method=verified_method, proof_id=proof_id
    )
