"""VLM (Vision-Language Model) deployment configuration.

Controls connection to external VLM inference servers (vLLM, TensorRT-LLM, etc.)
for semantic document understanding. Opt-in via VLM_ENABLED=true.

Environment variables:
    VLM_ENABLED: bool (default: False) -- master toggle
    VLM_ENDPOINT_URL: str -- inference server URL (e.g. http://vlm:8080/v1)
    VLM_ALLOWED_HOSTS: comma-separated host allowlist for VLM egress
    VLM_ALLOW_INSECURE_LOCAL: bool -- permit http allowlisted hosts for local dev
    VLM_ALLOW_PRIVATE_EGRESS: bool -- permit private/in-cluster allowlisted hosts
    VLM_ALLOW_INLINE_IMAGES: bool -- permit inline image_b64 payloads (default: False)
    VLM_API_KEY: str -- API key for the inference server
    VLM_MODEL_NAME: str -- model identifier (default: "default")
    VLM_MAX_CONTEXT_PAGES: int -- max pages per analysis request (default: 5)
    VLM_TIMEOUT_SECONDS: int -- HTTP timeout for VLM requests (default: 30)
    VLM_RETRY_ATTEMPTS: int -- retry count on transient failures (default: 3)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from ocr_distributed.ssrf import is_private_ip


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


def _normalize_allowed_host(value: str) -> str:
    """Normalize an allowlist host entry, accepting optional scheme or port."""
    candidate = value.strip().lower()
    if not candidate:
        return ""
    parsed = urlparse(candidate if "://" in candidate else f"//{candidate}")
    host = (parsed.hostname or candidate).lower()
    try:
        port = parsed.port
    except ValueError:
        port = None
    return f"{host}:{port}" if port is not None else host


def _default_port(scheme: str) -> int | None:
    if scheme == "http":
        return 80
    if scheme == "https":
        return 443
    return None


def _allowed_endpoint_matches(configured_host: str, endpoint_url: str) -> bool:
    """Return True when an allowlist entry matches endpoint host and optional port."""
    endpoint = urlparse(endpoint_url)
    endpoint_host = (endpoint.hostname or "").lower()
    endpoint_port = endpoint.port or _default_port(endpoint.scheme)
    configured = urlparse(
        configured_host.strip().lower()
        if "://" in configured_host
        else f"//{configured_host.strip().lower()}"
    )
    configured_host_name = (configured.hostname or configured_host).lower()
    try:
        configured_port = configured.port
    except ValueError:
        configured_port = None
    if endpoint_host != configured_host_name:
        return False
    return configured_port is None or configured_port == endpoint_port


def _csv_env(key: str) -> tuple[str, ...]:
    """Parse a comma-separated environment variable into normalized host values."""
    return tuple(
        host
        for part in os.environ.get(key, "").split(",")
        if (host := _normalize_allowed_host(part))
    )


@dataclass(frozen=True)
class VLMConfig:
    """Immutable configuration for a VLM inference endpoint.

    Attributes:
        enabled: Whether VLM features are active.
        endpoint_url: Base URL of the VLM inference server.
        allowed_hosts: Explicit hostname allowlist for VLM egress.
        allow_insecure_local: Permit HTTP allowlisted endpoints for local dev.
        allow_private_egress: Permit private/in-cluster allowlisted endpoints.
        allow_inline_images: Permit inline image_b64 payloads instead of URL handoff.
        api_key: API key for authenticating to the VLM server (never logged).
        model_name: Model identifier for multi-model servers.
        max_context_pages: Maximum number of pages sent per analysis request.
        timeout_seconds: HTTP request timeout.
        retry_attempts: Number of retries on transient failures.
    """

    enabled: bool = False
    endpoint_url: str = ""
    allowed_hosts: tuple[str, ...] = ()
    allow_insecure_local: bool = False
    allow_private_egress: bool = False
    allow_inline_images: bool = False
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
        if self.enabled and self.endpoint_url:
            allowed_hosts = tuple(
                host
                for configured_host in self.allowed_hosts
                if (host := _normalize_allowed_host(configured_host))
            )
            if not allowed_hosts:
                errors.append(
                    "VLM_ALLOWED_HOSTS must include the configured VLM endpoint host "
                    "when VLM_ENABLED=true"
                )
            elif not any(
                _allowed_endpoint_matches(configured_host, self.endpoint_url)
                for configured_host in allowed_hosts
            ):
                errors.append(
                    "VLM_ENDPOINT_URL host/port is not in VLM_ALLOWED_HOSTS"
                )
            parsed = urlparse(self.endpoint_url)
            endpoint_host = (parsed.hostname or "").lower()
            if parsed.scheme == "http" and not self.allow_insecure_local:
                errors.append(
                    "VLM_ENDPOINT_URL must use HTTPS unless "
                    "VLM_ALLOW_INSECURE_LOCAL=true for local development"
                )
            if (
                parsed.scheme == "http"
                and self.allow_insecure_local
                and not (
                    self.allow_private_egress
                    and endpoint_host
                    and is_private_ip(endpoint_host)
                )
            ):
                errors.append(
                    "VLM_ALLOW_INSECURE_LOCAL only permits HTTP for private/local "
                    "endpoints when VLM_ALLOW_PRIVATE_EGRESS=true"
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
        allowed_hosts=_csv_env("VLM_ALLOWED_HOSTS"),
        allow_insecure_local=_bool_env("VLM_ALLOW_INSECURE_LOCAL", default=False),
        allow_private_egress=_bool_env("VLM_ALLOW_PRIVATE_EGRESS", default=False),
        allow_inline_images=_bool_env("VLM_ALLOW_INLINE_IMAGES", default=False),
        api_key=os.environ.get("VLM_API_KEY", "").strip(),
        model_name=os.environ.get("VLM_MODEL_NAME", "default").strip(),
        max_context_pages=_int_env("VLM_MAX_CONTEXT_PAGES", 5, min_val=1, max_val=100),
        timeout_seconds=_int_env("VLM_TIMEOUT_SECONDS", 30, min_val=1, max_val=600),
        retry_attempts=_int_env("VLM_RETRY_ATTEMPTS", 3, min_val=0, max_val=10),
    )
