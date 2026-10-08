"""SGLang OpenAI-compatible chat-completions OCR client boundary.

This module is intentionally separate from :mod:`vlm_gateway`. The existing
gateway speaks OCR_LOCAL's RAG-shaped analyst-assist contract
(``/v1/analyze`` and ``/v1/search``). SGLang OCR integrations must use the
OpenAI-compatible ``/v1/chat/completions`` surface through this client instead.

The client imports no model runtime packages; it is only an HTTP/payload helper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
_USER_AGENT = "OCR-LOCAL-SGLangChatCompletions/1.0"


class SGLangChatClientError(Exception):
    """Base exception for SGLang chat-completions client errors."""


class InlineImagesDisabledError(SGLangChatClientError, ValueError):
    """Raised when a caller tries to inline an image without explicit opt-in."""


@dataclass(frozen=True)
class SGLangChatConfig:
    """Configuration for the SGLang chat-completions client.

    ``allow_inline_images`` defaults to ``False`` so page images are not embedded
    in OpenAI-style message content unless an upstream config lane explicitly
    permits that behavior.
    """

    endpoint_url: str
    model_name: str
    api_key: str = ""
    timeout_seconds: int = 30
    temperature: float = 0.0
    max_tokens: int | None = None
    allow_inline_images: bool = False

    def validate(self) -> list[str]:
        """Return validation errors without contacting the configured endpoint."""
        errors: list[str] = []
        if not self.endpoint_url:
            errors.append("endpoint_url is required")
        if self.endpoint_url and not self.endpoint_url.startswith(("http://", "https://")):
            errors.append("endpoint_url must start with http:// or https://")
        if not self.model_name:
            errors.append("model_name is required")
        if not isinstance(self.timeout_seconds, int):
            errors.append("timeout_seconds must be an integer")
        elif self.timeout_seconds < 1:
            errors.append("timeout_seconds must be >= 1")
        if not isinstance(self.temperature, int | float):
            errors.append("temperature must be a number")
        elif not 0.0 <= self.temperature <= 2.0:
            errors.append("temperature must be between 0.0 and 2.0")
        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int):
                errors.append("max_tokens must be an integer when set")
            elif self.max_tokens < 1:
                errors.append("max_tokens must be >= 1 when set")
        return errors


def build_ocr_chat_completions_payload(
    *,
    config: SGLangChatConfig,
    prompt: str,
    image_b64: str | None = None,
    image_mime_type: str = "image/png",
) -> dict[str, Any]:
    """Build an OpenAI-compatible OCR prompt payload for SGLang.

    The payload is pure data and can be tested without SGLang, CUDA, model
    weights, or a running model server.
    """
    prompt_text = prompt.strip()
    if not prompt_text:
        raise ValueError("prompt is required")

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    if image_b64:
        if not config.allow_inline_images:
            raise InlineImagesDisabledError(
                "Inline image content is disabled; set allow_inline_images=True "
                "only when the upstream image policy permits it."
            )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_mime_type};base64,{image_b64}",
                },
            }
        )

    payload: dict[str, Any] = {
        "model": config.model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": config.temperature,
    }
    if config.max_tokens is not None:
        payload["max_tokens"] = config.max_tokens
    return payload


class SGLangChatCompletionsClient:
    """HTTP client for SGLang's OpenAI-compatible chat-completions endpoint."""

    def __init__(self, config: SGLangChatConfig) -> None:
        errors = config.validate()
        if errors:
            raise ValueError("; ".join(errors))
        self._config = config
        self._client: httpx.Client | None = None

    def __enter__(self) -> SGLangChatCompletionsClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            }
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"
            self._client = httpx.Client(
                base_url=self._config.endpoint_url.rstrip("/"),
                headers=headers,
                follow_redirects=False,
                timeout=httpx.Timeout(self._config.timeout_seconds),
                trust_env=False,
            )
        return self._client

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def create_ocr_completion(
        self,
        *,
        prompt: str,
        image_b64: str | None = None,
        image_mime_type: str = "image/png",
    ) -> dict[str, Any]:
        """POST an OCR request to ``/v1/chat/completions``.

        Tests should mock ``_get_client`` or use ``httpx.MockTransport``; this
        method is not a runtime probe and should not be used as validation unless
        an operator intentionally points it at a live model server.
        """
        payload = build_ocr_chat_completions_payload(
            config=self._config,
            prompt=prompt,
            image_b64=image_b64,
            image_mime_type=image_mime_type,
        )
        try:
            response = self._get_client().post(CHAT_COMPLETIONS_PATH, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SGLangChatClientError(
                "SGLang chat-completions request failed"
            ) from exc
        try:
            result = response.json()
        except ValueError as exc:
            raise SGLangChatClientError(
                "SGLang chat-completions response was not valid JSON"
            ) from exc
        if not isinstance(result, dict):
            raise SGLangChatClientError("SGLang chat-completions response was not a JSON object")
        return result
