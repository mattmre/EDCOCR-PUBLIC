"""OCR-side readiness preflight for the external EDC_TRANSLATION seam."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ocr_local.translation.external_client import (
    DEFAULT_TRANSLATION_READINESS_PATH,
    TranslationReadinessStatus,
    TranslationServiceClient,
)

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig


def external_translation_readiness(
    config: "PipelineConfig | None" = None,
    *,
    source_language: str = "en",
    target_language: str = "en",
) -> TranslationReadinessStatus:
    """Return whether OCR may dispatch to external EDC_TRANSLATION."""

    if not _prefer_external(config):
        return TranslationReadinessStatus(
            status="disabled",
            ready=False,
            enabled=False,
            url=None,
            message="external translation preference is disabled",
        )

    url = _configured_url(config)
    if not url:
        return TranslationReadinessStatus(
            status="unavailable",
            ready=False,
            enabled=True,
            url=None,
            message="EDC_TRANSLATION_URL is not configured",
        )

    client = TranslationServiceClient(
        url,
        api_key=os.environ.get("EDC_TRANSLATION_API_KEY"),
        timeout=_configured_timeout(config),
    )
    return client.check_readiness(
        path=_configured_readiness_path(config),
        source_language=source_language,
        target_language=target_language,
    )


def external_translation_dispatch_enabled(
    config: "PipelineConfig | None" = None,
    *,
    source_language: str = "en",
    target_language: str = "en",
) -> bool:
    """Return True only when preference is on and readiness is green."""

    return external_translation_readiness(
        config,
        source_language=source_language,
        target_language=target_language,
    ).ready


def _prefer_external(config: "PipelineConfig | None") -> bool:
    env_value = os.environ.get("EDC_TRANSLATION_PREFER_EXTERNAL", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    config_value = getattr(config, "translation_prefer_external_service", False)
    if isinstance(config_value, str):
        config_value = config_value.lower() in {"1", "true", "yes", "on"}
    return bool(config_value) or env_value


def _configured_url(config: "PipelineConfig | None") -> str | None:
    value = getattr(config, "translation_external_service_url", None)
    if value:
        return str(value)
    return os.environ.get("EDC_TRANSLATION_URL") or None


def _configured_timeout(config: "PipelineConfig | None") -> float:
    value = getattr(config, "translation_external_timeout_seconds", None)
    if value is not None:
        return float(value)
    return float(os.environ.get("EDC_TRANSLATION_TIMEOUT_SECONDS", "30"))


def _configured_readiness_path(config: "PipelineConfig | None") -> str:
    value = getattr(config, "translation_external_readiness_path", None)
    if value:
        return str(value)
    return os.environ.get(
        "EDC_TRANSLATION_READINESS_PATH",
        DEFAULT_TRANSLATION_READINESS_PATH,
    )
