"""Client seam for an external EDC_TRANSLATION service.

This module lets OCR-side code call a standalone translation service using
``DocumentBundle v1`` and ``TranslationBundle v1``.  It deliberately does
not replace the existing in-repo translation facade; callers must opt in by
constructing this client.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ocr_distributed.ssrf import NoRedirectHandler
from ocr_local.contracts import validate_contract_payload

DEFAULT_TRANSLATION_SERVICE_URL = "http://localhost:8080"
DEFAULT_TRANSLATION_READINESS_PATH = "/health"
DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
DEFAULT_TRANSLATION_TIMEOUT_SECONDS = 30.0
MIN_TRANSLATION_TIMEOUT_SECONDS = 0.1
MAX_TRANSLATION_TIMEOUT_SECONDS = 600.0
_RESPONSE_READ_CHUNK_BYTES = 64 * 1024

_NO_REDIRECT_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    NoRedirectHandler(),
)
_DEFAULT_URLOPEN = urllib.request.urlopen


def _open_external_request(request, *, timeout: float):
    """Open through the safe opener; preserve the module's test seam."""
    # Existing callers/tests patch urllib.request.urlopen as their transport
    # seam. Production keeps the original function untouched and uses the
    # redirect/proxy-free opener.
    if urllib.request.urlopen is not _DEFAULT_URLOPEN:
        return urllib.request.urlopen(request, timeout=timeout)
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


class TranslationServiceError(RuntimeError):
    """Raised when the external translation service call fails."""


def validate_translation_timeout(value: object) -> float:
    """Return a finite, bounded external-service timeout."""
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("translation timeout must be numeric") from exc
    if not math.isfinite(timeout) or not (
        MIN_TRANSLATION_TIMEOUT_SECONDS <= timeout <= MAX_TRANSLATION_TIMEOUT_SECONDS
    ):
        raise ValueError(
            "translation timeout must be between "
            f"{MIN_TRANSLATION_TIMEOUT_SECONDS:g} and "
            f"{MAX_TRANSLATION_TIMEOUT_SECONDS:g} seconds"
        )
    return timeout


def translation_timeout_from_env(
    name: str = "EDC_TRANSLATION_TIMEOUT_SECONDS",
) -> float:
    """Read an external-service timeout, falling back on malformed env data."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return DEFAULT_TRANSLATION_TIMEOUT_SECONDS
    try:
        return validate_translation_timeout(raw)
    except ValueError:
        return DEFAULT_TRANSLATION_TIMEOUT_SECONDS


@dataclass(frozen=True)
class TranslationReadinessStatus:
    """OCR-side view of external EDC_TRANSLATION readiness."""

    status: str
    ready: bool
    enabled: bool
    url: str | None
    message: str
    latency_ms: float | None = None
    detail: dict[str, Any] | None = None


class TranslationServiceClient:
    """Small stdlib HTTP client for EDC_TRANSLATION bundle endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_TRANSLATION_SERVICE_URL,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        if not isinstance(max_response_bytes, int) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = validate_translation_timeout(timeout)
        self.max_response_bytes = max_response_bytes

    @classmethod
    def from_env(cls) -> "TranslationServiceClient":
        """Build a client from environment variables."""

        return cls(
            os.environ.get(
                "EDC_TRANSLATION_URL",
                DEFAULT_TRANSLATION_SERVICE_URL,
            ),
            api_key=os.environ.get("EDC_TRANSLATION_API_KEY"),
            timeout=translation_timeout_from_env(),
            max_response_bytes=_max_response_bytes_from_env(),
        )

    def list_engines(self) -> list[dict[str, Any]]:
        """Return available provider engines from the external service."""

        payload = self._request_json("GET", "/api/v1/translation/engines")
        engines = payload.get("engines")
        if not isinstance(engines, list):
            raise TranslationServiceError(
                "translation service response missing engines"
            )
        return engines

    def check_readiness(
        self,
        *,
        path: str = DEFAULT_TRANSLATION_READINESS_PATH,
        source_language: str = "en",
        target_language: str = "en",
    ) -> TranslationReadinessStatus:
        """Probe the configured EDC_TRANSLATION readiness endpoint."""

        readiness_path = _readiness_path_with_query(
            path,
            source_language=source_language,
            target_language=target_language,
        )
        try:
            payload, latency_ms = self._request_json_with_latency(
                "GET",
                readiness_path,
                wrap_errors=False,
            )
        except urllib.error.HTTPError as exc:
            detail = _decode_error_detail(exc, self.max_response_bytes)
            return TranslationReadinessStatus(
                status="unavailable",
                ready=False,
                enabled=True,
                url=f"{self.base_url}{readiness_path}",
                message=f"readiness endpoint returned HTTP {exc.code}",
                detail=detail,
            )
        except urllib.error.URLError as exc:
            return TranslationReadinessStatus(
                status="unreachable",
                ready=False,
                enabled=True,
                url=f"{self.base_url}{readiness_path}",
                message=f"readiness endpoint unreachable: {exc.reason}",
            )
        except TranslationServiceError as exc:
            return TranslationReadinessStatus(
                status="unavailable",
                ready=False,
                enabled=True,
                url=f"{self.base_url}{readiness_path}",
                message=str(exc),
            )

        service_status = str(payload.get("status", "")).lower()
        ready = service_status in {"healthy", "ready", "ok"}
        return TranslationReadinessStatus(
            status="ready" if ready else "unavailable",
            ready=ready,
            enabled=True,
            url=f"{self.base_url}{readiness_path}",
            message=(
                f"readiness endpoint reported {service_status or 'unknown status'}"
            ),
            latency_ms=latency_ms,
            detail=payload,
        )

    def translate_bundle(
        self,
        document_bundle: dict[str, Any],
        *,
        target_language: str,
        provider_id: str = "passthrough",
        certified: bool = False,
    ) -> dict[str, Any]:
        """Submit a ``DocumentBundle v1`` and return ``TranslationBundle v1``."""

        validate_contract_payload(document_bundle, "document-bundle-v1")
        response = self._request_json(
            "POST",
            "/api/v1/translation/bundles",
            {
                "document_bundle": document_bundle,
                "target_language": target_language,
                "provider_id": provider_id,
                "certified": certified,
            },
        )
        translation_bundle = response.get("translation_bundle", response)
        if not isinstance(translation_bundle, dict):
            raise TranslationServiceError(
                "translation service response missing translation_bundle"
            )
        validate_contract_payload(translation_bundle, "translation-bundle-v1")
        return translation_bundle

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        decoded, _latency_ms = self._request_json_with_latency(method, path, payload)
        return decoded

    def _request_json_with_latency(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        wrap_errors: bool = True,
    ) -> tuple[dict[str, Any], float]:
        import time

        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            started = time.monotonic()
            with _open_external_request(request, timeout=self.timeout) as response:
                raw = _read_limited(response, self.max_response_bytes)
            latency_ms = (time.monotonic() - started) * 1000
        except urllib.error.HTTPError as exc:
            if not wrap_errors:
                raise
            # Upstream error bodies may contain credentials, filesystem paths,
            # or arbitrary provider text. Keep the public exception generic.
            _read_limited(exc, self.max_response_bytes)
            raise TranslationServiceError(
                f"translation service HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            if not wrap_errors:
                raise
            raise TranslationServiceError(
                f"translation service unavailable: {exc.reason}"
            ) from exc

        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TranslationServiceError(
                "translation service returned invalid JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise TranslationServiceError(
                "translation service returned non-object JSON"
            )
        return decoded, round(latency_ms, 1)


def _readiness_path_with_query(
    path: str,
    *,
    source_language: str,
    target_language: str,
) -> str:
    if "?" in path:
        return path
    if path.rstrip("/") != "/api/v1/translation/readiness/auto-route":
        return path
    query = urllib.parse.urlencode(
        {
            "source_language": source_language,
            "target_language": target_language,
        }
    )
    return f"{path}?{query}"


def _max_response_bytes_from_env() -> int:
    """Read the external-service response cap with a safe default."""
    raw = os.environ.get("EDC_TRANSLATION_MAX_RESPONSE_BYTES", "")
    if not raw:
        return DEFAULT_MAX_RESPONSE_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_RESPONSE_BYTES
    return value if value > 0 else DEFAULT_MAX_RESPONSE_BYTES


def _read_limited(stream, max_bytes: int) -> bytes:
    """Read a response without allowing an unbounded remote allocation."""
    chunks: list[bytes] = []
    total = 0
    while True:
        remaining = max_bytes - total
        if remaining < 0:
            raise TranslationServiceError(
                f"translation service response exceeds {max_bytes} bytes"
            )
        chunk = stream.read(min(_RESPONSE_READ_CHUNK_BYTES, remaining + 1))
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            chunk = bytes(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise TranslationServiceError(
                f"translation service response exceeds {max_bytes} bytes"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_error_detail(
    exc: urllib.error.HTTPError,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> dict[str, Any] | None:
    try:
        raw = _read_limited(exc, max_response_bytes).decode("utf-8", errors="replace")
    except TranslationServiceError:
        return {"error": "error response exceeds configured size limit"}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return decoded if isinstance(decoded, dict) else {"raw": decoded}
