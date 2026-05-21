"""PipelineConfig coverage for the external EDC_TRANSLATION seam."""

from __future__ import annotations

import pytest

from pipeline_config import PipelineConfig, create_pipeline_config


def test_external_translation_config_defaults_off():
    cfg = create_pipeline_config(env={})

    assert cfg.translation_prefer_external_service is False
    assert cfg.translation_external_service_url is None
    assert cfg.translation_external_provider_id == "passthrough"
    assert cfg.translation_external_timeout_seconds == 30.0
    assert cfg.translation_external_readiness_path == "/health"


def test_external_translation_config_from_env():
    cfg = create_pipeline_config(
        env={
            "EDC_TRANSLATION_PREFER_EXTERNAL": "true",
            "EDC_TRANSLATION_URL": "http://127.0.0.1:18080",
            "EDC_TRANSLATION_PROVIDER_ID": "deterministic_ci",
            "EDC_TRANSLATION_TIMEOUT_SECONDS": "2.5",
            "EDC_TRANSLATION_READINESS_PATH": "/api/v1/translation/readiness/auto-route",
        }
    )

    assert cfg.translation_prefer_external_service is True
    assert cfg.translation_external_service_url == "http://127.0.0.1:18080"
    assert cfg.translation_external_provider_id == "deterministic_ci"
    assert cfg.translation_external_timeout_seconds == 2.5
    assert (
        cfg.translation_external_readiness_path
        == "/api/v1/translation/readiness/auto-route"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("translation_prefer_external_service", "true"),
        ("translation_external_service_url", ""),
        ("translation_external_provider_id", ""),
        ("translation_external_timeout_seconds", 0),
        ("translation_external_timeout_seconds", 601),
        ("translation_external_readiness_path", "health"),
    ],
)
def test_external_translation_config_validation(field, value):
    kwargs = {field: value}

    with pytest.raises(ValueError):
        PipelineConfig(**kwargs)


def test_external_translation_readiness_default_off():
    from ocr_local.translation.readiness import external_translation_readiness

    cfg = create_pipeline_config(env={})

    status = external_translation_readiness(cfg)

    assert status.status == "disabled"
    assert status.ready is False
    assert status.enabled is False


def test_external_translation_readiness_requires_configured_url():
    from ocr_local.translation.readiness import external_translation_readiness

    cfg = create_pipeline_config(env={"EDC_TRANSLATION_PREFER_EXTERNAL": "true"})

    status = external_translation_readiness(cfg)

    assert status.status == "unavailable"
    assert status.ready is False
    assert "EDC_TRANSLATION_URL" in status.message
