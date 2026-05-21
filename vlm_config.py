"""VLM (Vision-Language Model) deployment configuration.

Controls connection to external VLM inference servers (vLLM, TensorRT-LLM, etc.)
for semantic document understanding. Opt-in via VLM_ENABLED=true.

Environment variables:
    VLM_ENABLED: bool (default: False) -- master toggle
    VLM_ENDPOINT_URL: str -- inference server URL (e.g. http://vlm:8080/v1)
    VLM_API_KEY: str -- API key for the inference server
    VLM_MODEL_NAME: str -- model identifier (default: "default")
    VLM_MAX_CONTEXT_PAGES: int -- max pages per analysis request (default: 5)
    VLM_TIMEOUT_SECONDS: int -- HTTP timeout for VLM requests (default: 30)
    VLM_RETRY_ATTEMPTS: int -- retry count on transient failures (default: 3)
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _bool_env(key: str, default: bool = False) -> bool:
    """Parse a boolean from an environment variable."""
    return os.environ.get(key, "").lower() in ("1", "true", "yes") if not default else os.environ.get(key, "").lower() not in ("0", "false", "no")


def _int_env(key: str, default: int, min_val: int = 1, max_val: int = 100_000) -> int:
    """Parse an integer from an environment variable with bounds."""
    raw = os.environ.get(key, str(default))
    try:
        value = int(raw)
    except (ValueError, TypeError):
        value = default
    return max(min_val, min(value, max_val))


@dataclass(frozen=True)
class VLMConfig:
    """Immutable configuration for a VLM inference endpoint.

    Attributes:
        enabled: Whether VLM features are active.
        endpoint_url: Base URL of the VLM inference server.
        api_key: API key for authenticating to the VLM server (never logged).
        model_name: Model identifier for multi-model servers.
        max_context_pages: Maximum number of pages sent per analysis request.
        timeout_seconds: HTTP request timeout.
        retry_attempts: Number of retries on transient failures.
    """

    enabled: bool = False
    endpoint_url: str = ""
    api_key: str = ""
    model_name: str = "default"
    max_context_pages: int = 5
    timeout_seconds: int = 30
    retry_attempts: int = 3

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty if valid)."""
        errors: list[str] = []
        if self.enabled and not self.endpoint_url:
            errors.append("VLM_ENDPOINT_URL is required when VLM_ENABLED=true")
        if self.endpoint_url and not self.endpoint_url.startswith(("http://", "https://")):
            errors.append(
                f"VLM_ENDPOINT_URL must start with http:// or https://, got: {self.endpoint_url}"
            )
        if self.max_context_pages < 1:
            errors.append("VLM_MAX_CONTEXT_PAGES must be >= 1")
        if self.timeout_seconds < 1:
            errors.append("VLM_TIMEOUT_SECONDS must be >= 1")
        if self.retry_attempts < 0:
            errors.append("VLM_RETRY_ATTEMPTS must be >= 0")
        return errors


def load_vlm_config() -> VLMConfig:
    """Load VLM configuration from environment variables."""
    return VLMConfig(
        enabled=_bool_env("VLM_ENABLED", default=False),
        endpoint_url=os.environ.get("VLM_ENDPOINT_URL", "").strip(),
        api_key=os.environ.get("VLM_API_KEY", "").strip(),
        model_name=os.environ.get("VLM_MODEL_NAME", "default").strip(),
        max_context_pages=_int_env("VLM_MAX_CONTEXT_PAGES", 5, min_val=1, max_val=100),
        timeout_seconds=_int_env("VLM_TIMEOUT_SECONDS", 30, min_val=1, max_val=600),
        retry_attempts=_int_env("VLM_RETRY_ATTEMPTS", 3, min_val=0, max_val=10),
    )
