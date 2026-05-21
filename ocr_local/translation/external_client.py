"""Client seam for an external EDC_TRANSLATION service.

This module lets OCR-side code call a standalone translation service using
``DocumentBundle v1`` and ``TranslationBundle v1``.  It deliberately does
not replace the existing in-repo translation facade; callers must opt in by
constructing this client.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ocr_local.contracts import validate_contract_payload

DEFAULT_TRANSLATION_SERVICE_URL = "http://localhost:8080"
DEFAULT_TRANSLATION_READINESS_PATH = "/health"


class TranslationServiceError(RuntimeError):
    """Raised when the external translation service call fails."""


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "TranslationServiceClient":
        """Build a client from environment variables."""

        return cls(
            os.environ.get(
                "EDC_TRANSLATION_URL",
                DEFAULT_TRANSLATION_SERVICE_URL,
            ),
            api_key=os.environ.get("EDC_TRANSLATION_API_KEY"),
            timeout=float(os.environ.get("EDC_TRANSLATION_TIMEOUT_SECONDS", "30")),
        )

    def list_engines(self) -> list[dict[str, Any]]:
        """Return available provider engines from the external service."""

        payload = self._request_json("GET", "/api/v1/translation/engines")
        engines = payload.get("engines")
        if not isinstance(engines, list):
            raise TranslationServiceError("translation service response missing engines")
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
            detail = _decode_error_detail(exc)
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
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
            latency_ms = (time.monotonic() - started) * 1000
        except urllib.error.HTTPError as exc:
            if not wrap_errors:
                raise
            detail = exc.read().decode("utf-8", errors="replace")
            raise TranslationServiceError(
                f"translation service HTTP {exc.code}: {detail}"
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
            raise TranslationServiceError("translation service returned non-object JSON")
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


def _decode_error_detail(exc: urllib.error.HTTPError) -> dict[str, Any] | None:
    raw = exc.read().decode("utf-8", errors="replace")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return decoded if isinstance(decoded, dict) else {"raw": decoded}
