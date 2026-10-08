"""Bounded, redirect-free HTTP helpers for federation control-plane probes."""

from __future__ import annotations

import os
import ssl
import urllib.parse
import urllib.request
from typing import Any

from ocr_distributed.ssrf import NoRedirectHandler

DEFAULT_MAX_RESPONSE_BYTES = 256 * 1024
_READ_CHUNK_BYTES = 64 * 1024


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _federation_opener() -> urllib.request.OpenerDirector:
    """Build a no-proxy, no-redirect opener with optional federation mTLS."""
    handlers: list[Any] = [
        urllib.request.ProxyHandler({}),
        NoRedirectHandler(),
    ]
    if _truthy_env("OCR_FEDERATION_MTLS_ENABLED"):
        ca_path = os.environ.get("OCR_FEDERATION_MTLS_CA_PATH", "").strip()
        cert_path = os.environ.get("OCR_FEDERATION_MTLS_CERT_PATH", "").strip()
        key_path = os.environ.get("OCR_FEDERATION_MTLS_KEY_PATH", "").strip()
        if not ca_path or not cert_path or not key_path:
            raise ValueError(
                "federation mTLS requires CA, client certificate, and key paths"
            )
        context = ssl.create_default_context(cafile=ca_path)
        context.load_cert_chain(certfile=cert_path, keyfile=key_path)
        handlers.append(urllib.request.HTTPSHandler(context=context))
    return urllib.request.build_opener(*handlers)


class ResponseTooLarge(ValueError):
    """Raised when a peer response exceeds the federation safety budget."""


def open_federation_request(request, *, timeout: float):
    """Open a configured peer request without following redirects."""
    if _truthy_env("OCR_FEDERATION_MTLS_ENABLED"):
        scheme = urllib.parse.urlsplit(request.full_url).scheme.lower()
        if scheme != "https":
            raise ValueError("federation mTLS requires an HTTPS peer URL")
    return _federation_opener().open(request, timeout=timeout)


def read_response_bounded(
    response,
    *,
    max_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> bytes:
    """Read a peer response with a hard memory bound.

    ``urllib`` response objects accept a size argument on ``read``.  The
    helper deliberately requests at most ``max_bytes + 1`` bytes so an
    oversized body is rejected before it can become an unbounded allocation.
    """
    if not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")

    headers = getattr(response, "headers", None)
    declared = headers.get("Content-Length") if headers is not None else None
    if declared is not None:
        try:
            declared_bytes = int(declared)
        except (TypeError, ValueError) as exc:
            raise ValueError("peer response has an invalid Content-Length") from exc
        if declared_bytes < 0:
            raise ValueError("peer response has a negative Content-Length")
        if declared_bytes > max_bytes:
            raise ResponseTooLarge(
                f"peer response exceeds the {max_bytes}-byte safety limit"
            )

    chunks: list[bytes] = []
    total = 0
    while total <= max_bytes:
        chunk = response.read(min(_READ_CHUNK_BYTES, max_bytes + 1 - total))
        if not chunk:
            break
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        if not isinstance(chunk, bytes):
            raise ValueError("peer response returned a non-byte body")
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise ResponseTooLarge(
                f"peer response exceeds the {max_bytes}-byte safety limit"
            )
    return b"".join(chunks)


__all__ = [
    "DEFAULT_MAX_RESPONSE_BYTES",
    "ResponseTooLarge",
    "open_federation_request",
    "read_response_bounded",
]
