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
import time
from typing import Any, Optional

import httpx

from vlm_config import VLMConfig

logger = logging.getLogger(__name__)

# Sentinel for unavailable gateway
_DISABLED_MSG = "VLM gateway is disabled. Set VLM_ENABLED=true and configure VLM_ENDPOINT_URL."


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


class VLMGateway:
    """Client for custom VLM inference endpoints.

    Connects to vLLM, TensorRT-LLM, or any OpenAI-compatible server.
    All methods are synchronous (suitable for threading-based pipeline).
    """

    def __init__(self, config: VLMConfig) -> None:
        self._config = config
        self._client: Optional[httpx.Client] = None

    @property
    def enabled(self) -> bool:
        """Whether the VLM gateway is enabled and configured."""
        return self._config.enabled and bool(self._config.endpoint_url)

    def _get_client(self) -> httpx.Client:
        """Return a lazily-initialized httpx client."""
        if self._client is None:
            headers: dict[str, str] = {
                "Content-Type": "application/json",
                "User-Agent": "OCR-LOCAL-VLMGateway/1.0",
            }
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            self._client = httpx.Client(
                base_url=self._config.endpoint_url.rstrip("/"),
                headers=headers,
                timeout=httpx.Timeout(self._config.timeout_seconds),
            )
        return self._client

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

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
        pages: list[dict[str, Any]],
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

        # Enforce max context pages
        if len(pages) > self._config.max_context_pages:
            pages = pages[: self._config.max_context_pages]
            logger.info(
                "Truncated page list to %d pages (VLM_MAX_CONTEXT_PAGES)",
                self._config.max_context_pages,
            )

        payload: dict[str, Any] = {
            "model": self._config.model_name,
            "pages": pages,
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
        return response.get("results", [])

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
                start = time.monotonic()
                resp = self._get_client().request(method, path, json=json_body)
                elapsed_ms = (time.monotonic() - start) * 1000

                if resp.status_code in (401, 403):
                    raise VLMAuthError(
                        f"VLM authentication failed (HTTP {resp.status_code})"
                    )

                if 500 <= resp.status_code < 600:
                    raise VLMServerError(
                        f"VLM server error (HTTP {resp.status_code}): {resp.text[:200]}"
                    )

                if resp.status_code >= 400:
                    raise VLMGatewayError(
                        f"VLM request failed (HTTP {resp.status_code}): {resp.text[:200]}"
                    )

                result = resp.json()
                if isinstance(result, dict):
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
