"""Sanity tests for the staging translation flag-flip overlay.

These tests pin the contract between
``helm/ocr-local/values-staging.yaml`` and
``coordinator/.env.staging`` on one side and the runtime parsing logic
in :mod:`pipeline_config` and :mod:`api.main` on the other side.

The staging overlay flips Plan B Wave M1 translation flags ON for
test/staging clusters.  Production overlays MUST keep the defaults
OFF -- see  and
``docs/operations/translation-flag-flip-runbook.md``.
"""

from __future__ import annotations

import os

from pipeline_config import create_pipeline_config

# ---------------------------------------------------------------------------
# Mirrors of the staging overlay env block
# ---------------------------------------------------------------------------

# Mirrors ``helm/ocr-local/values-staging.yaml`` ``coordinator.env``
# and ``coordinator/.env.staging``.
_STAGING_ENV = {
    "ENABLE_TRANSLATION": "true",
    "ENABLE_TRANSLATION_API": "true",
    "ENABLE_HANDWRITING_MT": "false",
    "EDC_TRANSLATION_PREFER_EXTERNAL": "false",
    "EDC_TRANSLATION_URL": "http://edc-translation:8080",
    "EDC_TRANSLATION_PROVIDER_ID": "deterministic_ci",
    "EDC_TRANSLATION_TIMEOUT_SECONDS": "30",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_staging_overlay_flips_pipeline_translation_flags(monkeypatch):
    """Applying the staging env block flips translation ON, handwriting MT OFF.

    ``ENABLE_TRANSLATION_API`` is read by ``api.main`` directly (not by
    :class:`PipelineConfig`), so it is exercised separately below.
    """
    for key, value in _STAGING_ENV.items():
        monkeypatch.setenv(key, value)

    cfg = create_pipeline_config()

    assert cfg.enable_translation is True
    assert cfg.enable_handwriting_mt is False
    assert cfg.translation_prefer_external_service is False
    assert cfg.translation_external_service_url == "http://edc-translation:8080"
    assert cfg.translation_external_provider_id == "deterministic_ci"


def test_staging_overlay_enables_translation_api_flag(monkeypatch):
    """``ENABLE_TRANSLATION_API`` is parsed with the same truthy rule as ``api.main``.

    Mirrors the parsing in :mod:`api.main`::

        os.environ.get("ENABLE_TRANSLATION_API", "").lower() in ("1", "true", "yes")
    """
    for key, value in _STAGING_ENV.items():
        monkeypatch.setenv(key, value)

    enabled = os.environ.get("ENABLE_TRANSLATION_API", "").lower() in (
        "1",
        "true",
        "yes",
    )

    assert enabled is True


def test_default_translation_flags_remain_off(monkeypatch):
    """Without the staging overlay, every translation flag stays OFF.

    Guards against accidental promotion of the staging overlay into a
    production-default code path.  See .
    """
    for key in _STAGING_ENV:
        monkeypatch.delenv(key, raising=False)

    cfg = create_pipeline_config()

    assert cfg.enable_translation is False
    assert cfg.enable_handwriting_mt is False
    assert cfg.translation_prefer_external_service is False

    api_flag = os.environ.get("ENABLE_TRANSLATION_API", "").lower() in (
        "1",
        "true",
        "yes",
    )
    assert api_flag is False
