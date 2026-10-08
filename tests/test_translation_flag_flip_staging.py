"""Sanity tests for the staging translation flag-flip overlay.

These tests pin the contract between
``helm/ocr-local/values-staging.yaml`` and
``coordinator/.env.staging`` on one side and the runtime parsing logic
in :mod:`pipeline_config` and :mod:`api.main` on the other side.

The staging overlay flips Plan B Wave M1 translation flags ON for
test/staging clusters.  Production overlays MUST keep the defaults
OFF -- see docs/release-candidate.md and
``docs/operations/translation-flag-flip-runbook.md``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
    production-default code path.  See docs/release-candidate.md.
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


def test_staging_render_has_translation_batch_consumer():
    repo_root = Path(__file__).resolve().parents[1]
    rendered = subprocess.run(
        [
            "helm",
            "template",
            "ocr-local",
            str(repo_root / "helm" / "ocr-local"),
            "-f",
            str(repo_root / "helm" / "ocr-local" / "values-staging.yaml"),
            "--set",
            "secrets.djangoSecretKey=test",
            "--set",
            "secrets.apiKey=test-api-key",
            "--set",
            "secrets.webhookSecretKey=test-webhook-key",
            "--set",
            "secrets.piiEncryptionKey=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
            "--set",
            "secrets.celeryTaskSigningKey=test-signing-key",
            "--set",
            "secrets.postgresPassword=test",
            "--set",
            "secrets.rabbitmqPassword=test",
            "--set",
            "secrets.redisPassword=test",
            "--set",
            "secrets.flowerPassword=test-flower-password",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert 'ENABLE_TRANSLATION_API: "true"' in rendered
    assert "--queues=cpu_general,translation_batch" in rendered
    assert "--queues=ocr_gpu,cpu_general,nlp_general" in rendered
    assert "CELERY_TASK_SIGNING_KEY:" in rendered
    assert rendered.count("- name: CELERY_TASK_SIGNING_KEY") >= 3
    assert "type: Recreate" in rendered
    assert "PII_ENCRYPTION_KEY:" in rendered
    assert rendered.count("- name: PII_ENCRYPTION_KEY") >= 3
    compose = (repo_root / "coordinator" / "docker-compose.coordinator.yml").read_text(
        encoding="utf-8"
    )
    assert "-Q coordinator,translation_batch" in compose
    for compose_name in (
        "docker-compose.worker.yml",
        "docker-compose.worker.cpu.yml",
        "docker-compose.cpu-only.yml",
        "docker-compose.multi-gpu.yml",
    ):
        worker_compose = (repo_root / "coordinator" / compose_name).read_text(
            encoding="utf-8"
        )
        assert "nlp_general" in worker_compose


def test_api_bridge_binds_translation_task_to_signed_coordinator_app():
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(
        {
            "DEPLOYMENT_ENV": "production",
            "DJANGO_SETTINGS_MODULE": "coordinator.settings_test",
            "CELERY_TASK_SIGNING_KEY": "test-signing-key",
            "PII_ENCRYPTION_KEY": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
            "PYTHONPATH": f"{repo_root / 'coordinator'}:{repo_root}",
        }
    )
    probe = """
from api.coordinator_bridge import ensure_coordinator_django
ensure_coordinator_django()
from jobs.task_signing import SignedTask
from jobs.tasks_translation_batch import translate_batch_input
assert translate_batch_input.app.main == 'coordinator'
assert isinstance(translate_batch_input, SignedTask)
"""
    subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
