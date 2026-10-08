"""VLM inference gateway client.

Abstract client for communicating with deployed VLM inference servers
(vLLM, TensorRT-LLM, or any OpenAI-compatible endpoint). This module
is a *client* -- it does not run a model; it sends requests to an
external inference server configured via VLM_ENDPOINT_URL.

Usage::

    from vlm_config import load_vlm_config
    from vlm_gateway import VLMGateway

    cfg = load_vlm_config()
    gw = VLMGateway(cfg)
    if gw.health_check():
        result = gw.analyze_document(pages=[...])
"""

from __future__ import annotations

import logging
import socket
import time
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from ocr_distributed.ssrf import (
    is_private_ip,
    pin_url_for_ssrf,
    register_host_pin,
    unregister_host_pin,
)
from vlm_config import VLMConfig, _allowed_endpoint_matches, _normalize_allowed_host

logger = logging.getLogger(__name__)

# Sentinel for unavailable gateway
_DISABLED_MSG = (
    "VLM gateway is disabled. Set VLM_ENABLED=true and configure VLM_ENDPOINT_URL."
)
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class VLMGatewayError(Exception):
    """Base exception for VLM gateway errors."""


class VLMConnectionError(VLMGatewayError):
    """Raised when the VLM endpoint is unreachable."""


class VLMAuthError(VLMGatewayError):
    """Raised when VLM endpoint returns an authentication error."""


class VLMTimeoutError(VLMGatewayError):
    """Raised when a VLM request times out."""


class VLMServerError(VLMGatewayError):
    """Raised when the VLM endpoint returns a 5xx error."""


_URL_HANDOFF_KEYS = {
    "document_url",
    "document_uri",
    "file_url",
    "file_uri",
    "href",
    "image_url",
    "image_uri",
    "link",
    "presigned_url",
    "presigned_uri",
    "source_url",
    "source_uri",
    "src",
    "uri",
    "url",
}

_ALLOWED_ANALYSIS_PAGE_KEYS = {"page_number", "text", "image_b64"}
_REQUIRED_ANALYSIS_PAGE_KEYS = {"page_number", "text"}
_MAX_ANALYSIS_PAGES = 100
_MAX_ANALYSIS_TEXT_CHARS = 200_000
_MAX_INLINE_IMAGE_B64_CHARS = 16 * 1024 * 1024


def _contains_url_handoff_payload(value: Any) -> bool:
    """Return True when a nested page payload contains remote URL handoff fields."""
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key).lower()
            if normalized_key in _URL_HANDOFF_KEYS and nested:
                return True
            if _contains_url_handoff_payload(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_url_handoff_payload(item) for item in value)
    return False


def _normalize_analysis_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and copy VLM page payloads before handing them to the endpoint."""
    if len(pages) > _MAX_ANALYSIS_PAGES:
        raise VLMGatewayError(
            f"VLM analysis accepts at most {_MAX_ANALYSIS_PAGES} pages"
        )
    normalized_pages: list[dict[str, Any]] = []
    for index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            raise VLMGatewayError("VLM page payloads must be JSON objects")

        if _contains_url_handoff_payload(page):
            raise VLMGatewayError(
                "VLM URL handoff payloads are disabled until a validated "
                "presigned URL schema and egress policy are configured."
            )

        unknown_keys = set(page) - _ALLOWED_ANALYSIS_PAGE_KEYS
        if unknown_keys:
            unknown = ", ".join(sorted(str(key) for key in unknown_keys))
            raise VLMGatewayError(
                f"VLM page {index} contains unsupported fields: {unknown}"
            )

        missing_keys = _REQUIRED_ANALYSIS_PAGE_KEYS - set(page)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise VLMGatewayError(
                f"VLM page {index} is missing required fields: {missing}"
            )

        page_number = page["page_number"]
        text = page["text"]
        if type(page_number) is not int or page_number < 1:
            raise VLMGatewayError("VLM page_number must be an integer >= 1")
        if type(text) is not str:
            raise VLMGatewayError("VLM page text must be a string")
        if len(text) > _MAX_ANALYSIS_TEXT_CHARS:
            raise VLMGatewayError(
                f"VLM page text exceeds {_MAX_ANALYSIS_TEXT_CHARS} characters"
            )

        normalized_page: dict[str, Any] = {
            "page_number": page_number,
            "text": text,
        }
        if "image_b64" in page:
            image_b64 = page["image_b64"]
            if image_b64 is not None and type(image_b64) is not str:
                raise VLMGatewayError("VLM image_b64 must be a string when provided")
            if (
                isinstance(image_b64, str)
                and len(image_b64) > _MAX_INLINE_IMAGE_B64_CHARS
            ):
                raise VLMGatewayError(
                    "VLM inline image payload exceeds the configured size limit"
                )
            normalized_page["image_b64"] = image_b64
        normalized_pages.append(normalized_page)

    return normalized_pages


def _validate_vlm_endpoint(config: VLMConfig) -> tuple[str, str, str]:
    """Validate VLM endpoint egress and pin DNS for the request lifetime.

    Returns:
        ``(pinned_base_url, original_hostname, pinned_ip)``

        *pinned_ip* may be empty only for unresolvable private/dev names when
        ``allow_private_egress`` is set; public endpoints always pin or raise.
    """
    endpoint = (config.endpoint_url or "").strip().rstrip("/")
    if not endpoint:
        raise VLMGatewayError(_DISABLED_MSG)

    parsed = urlparse(endpoint)
    host = (parsed.hostname or "").lower()
    allowed_hosts = tuple(
        normalized_host
        for configured_host in config.allowed_hosts
        if (normalized_host := _normalize_allowed_host(configured_host))
    )
    if not allowed_hosts:
        raise VLMGatewayError(
            "VLM_ALLOWED_HOSTS must include the configured endpoint host "
            "before VLM egress is allowed"
        )
    if not any(
        _allowed_endpoint_matches(configured_host, endpoint)
        for configured_host in allowed_hosts
    ):
        raise VLMGatewayError("VLM endpoint host/port is not in VLM_ALLOWED_HOSTS")
    if (
        parsed.scheme == "http"
        and config.allow_insecure_local
        and not (config.allow_private_egress and is_private_ip(host))
    ):
        raise VLMGatewayError(
            "VLM_ALLOW_INSECURE_LOCAL only permits HTTP for private/local "
            "allowlisted endpoints when VLM_ALLOW_PRIVATE_EGRESS=true"
        )

    try:
        pinned_url, original_host, pinned_ip = pin_url_for_ssrf(
            endpoint,
            allow_http=config.allow_insecure_local,
            allow_private=config.allow_private_egress,
            # A long-lived VLM pin must be based on current DNS, not an old
            # process-wide delivery/client pin that happens to share the host.
            bypass_pins=True,
        )
    except ValueError as exc:
        raise VLMGatewayError(f"VLM endpoint rejected by SSRF policy: {exc}") from exc
    return pinned_url.rstrip("/"), original_host, pinned_ip


def _contains_inline_image_payload(value: Any) -> bool:
    """Return True when a nested payload contains inline base64 image fields."""
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"image_b64", "image_base64", "image_bytes_b64"} and nested:
                return True
            if _contains_inline_image_payload(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_inline_image_payload(item) for item in value)
    return False


class VLMGateway:
    """Client for custom VLM inference endpoints.

    Connects to vLLM, TensorRT-LLM, or any OpenAI-compatible server.
    All methods are synchronous (suitable for threading-based pipeline).
    """

    def __init__(self, config: VLMConfig) -> None:
        self._config = config
        self._client: Optional[httpx.Client] = None
        self._pinned_ip: str = ""
        self._original_host: str = ""
        self._original_port: Optional[int] = None
        self._pin_registered = False

    @property
    def enabled(self) -> bool:
        """Whether the VLM gateway is enabled and configured."""
        return self._config.enabled and bool(self._config.endpoint_url)

    def _get_client(self) -> httpx.Client:
        """Return a lazily-initialized httpx client with DNS-pinned connect target.

        base_url keeps the original hostname so HTTPS SNI/cert verify work.
        ``register_host_pin`` forces socket connect to the validated IP for the
        request lifetime (anti-rebind).
        """
        if self._client is None:
            endpoint_url, original_host, pinned_ip = _validate_vlm_endpoint(
                self._config
            )
            self._original_host = original_host
            self._pinned_ip = pinned_ip
            parsed = urlparse(self._config.endpoint_url or "")
            self._original_port = parsed.port
            headers: dict[str, str] = {
                "Content-Type": "application/json",
                "User-Agent": "OCR-LOCAL-VLMGateway/1.0",
            }
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"
            if pinned_ip and original_host and original_host != pinned_ip:
                try:
                    register_host_pin(original_host, pinned_ip)
                    self._pin_registered = True
                except ValueError as exc:
                    self._pinned_ip = ""
                    self._original_host = ""
                    self._original_port = None
                    raise VLMGatewayError(
                        f"VLM endpoint has a conflicting active DNS pin: {exc}"
                    ) from exc

            try:
                self._client = httpx.Client(
                    base_url=endpoint_url,
                    headers=headers,
                    timeout=httpx.Timeout(self._config.timeout_seconds),
                    follow_redirects=False,
                    trust_env=False,
                )
            except Exception:
                if self._pin_registered:
                    unregister_host_pin(self._original_host)
                    self._pin_registered = False
                self._pinned_ip = ""
                self._original_host = ""
                self._original_port = None
                raise
        return self._client

    def _assert_pin_still_valid(self) -> None:
        """Fail closed if DNS no longer includes the pinned IP (rebind detect).

        Connect uses the registered pin (getaddrinfo hook). This check refuses
        to continue if the original name no longer answers with the pin.
        """
        if not self._pinned_ip or not self._original_host:
            return
        if self._original_host == self._pinned_ip:
            return
        # Bypass pin hook: resolve via original getaddrinfo for rebind detect.
        from ocr_distributed import ssrf as _ssrf

        try:
            resolved = _ssrf._ORIG_GETADDRINFO(
                self._original_host, None, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
            current = []
            for _f, _t, _p, _c, sockaddr in resolved:
                ip = sockaddr[0]
                if ip not in current:
                    current.append(ip)
        except OSError:
            current = []
        if not current:
            # Transient DNS failure: keep using pin (already validated).
            return
        if self._pinned_ip not in current:
            raise VLMConnectionError(
                "VLM endpoint DNS rebind detected: pinned IP no longer in "
                f"resolution for {self._original_host!r}"
            )

    def close(self) -> None:
        """Close the underlying HTTP client and drop DNS pin."""
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._pin_registered and self._original_host and self._pinned_ip:
            unregister_host_pin(self._original_host)
            self._pin_registered = False
        self._pinned_ip = ""
        self._original_host = ""
        self._original_port = None

    def __enter__(self) -> VLMGateway:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Check VLM endpoint availability.

        Returns True if the server responds with a 2xx status on the
        health/models endpoint.
        """
        if not self.enabled:
            logger.debug("VLM gateway disabled; health_check returns False")
            return False

        try:
            self._get_client()
            self._assert_pin_still_valid()
            resp = self._get_client().get("/health")
            if resp.status_code < 300:
                return True
            # Fallback: try OpenAI-style /v1/models
            resp = self._get_client().get("/v1/models")
            return resp.status_code < 300
        except httpx.ConnectError:
            logger.warning("VLM health check failed: connection refused")
            return False
        except httpx.TimeoutException:
            logger.warning("VLM health check failed: timeout")
            return False
        except Exception:
            logger.exception("VLM health check failed unexpectedly")
            return False

    def analyze_document(
        self,
        pages: list[dict[str, Any]] | None,
        prompt: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send document pages to the VLM for analysis.

        Args:
            pages: List of page dicts, each containing at minimum:
                - ``page_number`` (int)
                - ``text`` (str) -- extracted OCR text
                - ``image_b64`` (str, optional) -- base64-encoded page image
            prompt: Optional analysis prompt / instruction.
            document_id: Optional document identifier for logging.

        Returns:
            dict with keys: ``entities``, ``summary``, ``relationships``,
            ``confidence``, ``model``, ``processing_time_ms``.

        Raises:
            VLMGatewayError: On any communication or server error.
        """
        if not self.enabled:
            raise VLMGatewayError(_DISABLED_MSG)

        if not pages:
            raise VLMGatewayError("VLM analysis requires at least one page")

        normalized_pages = _normalize_analysis_pages(pages)

        if not self._config.allow_inline_images and _contains_inline_image_payload(
            normalized_pages
        ):
            raise VLMGatewayError(
                "Inline VLM image payloads are disabled. Use presigned image URL "
                "handoff or set VLM_ALLOW_INLINE_IMAGES=true only for an explicit "
                "risk-accepted local/dev path."
            )

        # Enforce max context pages
        if len(normalized_pages) > self._config.max_context_pages:
            normalized_pages = normalized_pages[: self._config.max_context_pages]
            logger.info(
                "Truncated page list to %d pages (VLM_MAX_CONTEXT_PAGES)",
                self._config.max_context_pages,
            )

        payload: dict[str, Any] = {
            "model": self._config.model_name,
            "pages": normalized_pages,
        }
        if prompt:
            payload["prompt"] = prompt
        if document_id:
            payload["document_id"] = document_id

        return self._request_with_retry("POST", "/v1/analyze", json_body=payload)

    def semantic_search(
        self,
        query: str,
        document_id: Optional[str] = None,
        max_results: int = 10,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search documents using semantic understanding.

        Args:
            query: Natural-language search query.
            document_id: Optional scope to a single document.
            max_results: Maximum results to return.
            min_score: Minimum relevance score threshold (0.0-1.0).

        Returns:
            List of result dicts with ``text``, ``score``, ``page``,
            ``document_id``, ``bbox``.

        Raises:
            VLMGatewayError: On any communication or server error.
        """
        if not self.enabled:
            raise VLMGatewayError(_DISABLED_MSG)

        payload: dict[str, Any] = {
            "model": self._config.model_name,
            "query": query,
            "max_results": max_results,
            "min_score": min_score,
        }
        if document_id:
            payload["document_id"] = document_id

        response = self._request_with_retry("POST", "/v1/search", json_body=payload)
        results = response.get("results")
        if not isinstance(results, list):
            raise VLMGatewayError("VLM search response must contain a results list")
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request_with_retry(
        self,
        method: str,
        path: str,
        json_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic for transient failures.

        Args:
            method: HTTP method ("GET", "POST", etc.)
            path: URL path relative to the base URL.
            json_body: Optional JSON body for POST/PUT.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            VLMConnectionError: If the endpoint is unreachable after retries.
            VLMAuthError: On 401/403 (no retry).
            VLMTimeoutError: On timeout after retries.
            VLMServerError: On 5xx after retries.
            VLMGatewayError: On unexpected errors.
        """
        last_exc: Optional[Exception] = None
        attempts = self._config.retry_attempts + 1  # first attempt + retries

        for attempt in range(1, attempts + 1):
            try:
                # Ensure client is built (pins DNS) then re-check pin each attempt.
                self._get_client()
                self._assert_pin_still_valid()
                start = time.monotonic()
                resp = self._get_client().request(method, path, json=json_body)
                elapsed_ms = (time.monotonic() - start) * 1000

                if resp.status_code in (401, 403):
                    raise VLMAuthError(
                        f"VLM authentication failed (HTTP {resp.status_code})"
                    )

                if 500 <= resp.status_code < 600:
                    raise VLMServerError(f"VLM server error (HTTP {resp.status_code})")

                if resp.status_code >= 400:
                    raise VLMGatewayError(
                        f"VLM request failed (HTTP {resp.status_code})"
                    )

                declared_length = resp.headers.get("content-length")
                if declared_length:
                    try:
                        if int(declared_length) > _MAX_RESPONSE_BYTES:
                            raise VLMGatewayError(
                                "VLM response exceeds the configured safety limit"
                            )
                    except ValueError as exc:
                        raise VLMGatewayError(
                            "VLM response has an invalid size"
                        ) from exc
                if len(resp.content) > _MAX_RESPONSE_BYTES:
                    raise VLMGatewayError(
                        "VLM response exceeds the configured safety limit"
                    )
                result = resp.json()
                if not isinstance(result, dict):
                    raise VLMGatewayError("VLM response must be a JSON object")
                result["processing_time_ms"] = round(elapsed_ms, 1)
                logger.debug(
                    "VLM %s %s completed in %.0fms (attempt %d)",
                    method,
                    path,
                    elapsed_ms,
                    attempt,
                )
                return result

            except VLMAuthError:
                raise  # non-retryable

            except VLMServerError as exc:
                last_exc = exc
                if attempt < attempts:
                    wait = min(2 ** (attempt - 1), 8)
                    logger.warning(
                        "VLM server error on attempt %d/%d, retrying in %ds: %s",
                        attempt,
                        attempts,
                        wait,
                        exc,
                    )
                    time.sleep(wait)
                    continue
                raise

            except VLMGatewayError:
                raise  # non-retryable (4xx, etc.)

            except httpx.ConnectError as exc:
                last_exc = exc
                if attempt < attempts:
                    wait = min(2 ** (attempt - 1), 8)
                    logger.warning(
                        "VLM connection error on attempt %d/%d, retrying in %ds",
                        attempt,
                        attempts,
                        wait,
                    )
                    time.sleep(wait)
                    continue

            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < attempts:
                    wait = min(2 ** (attempt - 1), 8)
                    logger.warning(
                        "VLM timeout on attempt %d/%d, retrying in %ds",
                        attempt,
                        attempts,
                        wait,
                    )
                    time.sleep(wait)
                    continue

            except Exception as exc:
                raise VLMGatewayError(f"Unexpected VLM error: {exc}") from exc

        # Exhausted retries
        if isinstance(last_exc, httpx.ConnectError):
            raise VLMConnectionError(
                f"VLM endpoint unreachable after {attempts} attempts"
            ) from last_exc
        if isinstance(last_exc, httpx.TimeoutException):
            raise VLMTimeoutError(
                f"VLM request timed out after {attempts} attempts"
            ) from last_exc
        raise VLMGatewayError(
            f"VLM request failed after {attempts} attempts: {last_exc}"
        ) from last_exc
