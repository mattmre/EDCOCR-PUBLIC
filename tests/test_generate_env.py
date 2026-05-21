"""Tests for scripts/generate_env.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import generate_env


@pytest.fixture
def template_text() -> str:
    return "\n".join([
        "DJANGO_SECRET_KEY=change-me-to-a-random-secret-key",
        "POSTGRES_PASSWORD=change-me-to-a-strong-password",
        "RABBITMQ_PASSWORD=change-me-to-a-strong-password",
        "REDIS_PASSWORD=change-me-to-a-strong-password",
        "FLOWER_PASSWORD=change-me-to-a-strong-password",
        "MINIO_ROOT_USER=change-me-to-a-minio-access-key",
        "MINIO_ROOT_PASSWORD=change-me-to-a-minio-secret-key",
        "S3_ACCESS_KEY=change-me-to-a-minio-access-key",
        "S3_SECRET_KEY=change-me-to-a-minio-secret-key",
        "CELERY_BROKER_URL=amqp://ocr_user:password@coordinator-host:5672//",
        "DATABASE_URL=postgres://ocr:password@coordinator-host:5432/ocr_coordinator",
        "DEPLOYMENT_ENV=staging",
        "",
    ])


def test_render_env_content_populates_local_s3_credentials(template_text: str):
    generated_values = {
        "django_secret": "django-secret",
        "postgres_pass": "postgres-pass",
        "rabbitmq_pass": "rabbitmq-pass",
        "redis_pass": "redis-pass",
        "flower_pass": "flower-pass",
        "metrics_key": "metrics-key",
        "erlang_cookie": "erlang-cookie-value",
        "minio_root_user": "ocrlocal-user",
        "minio_root_password": "ocrlocal-password",
    }

    rendered = generate_env._render_env_content(template_text, generated_values)

    assert "DJANGO_SECRET_KEY=django-secret" in rendered
    assert "POSTGRES_PASSWORD=postgres-pass" in rendered
    assert "RABBITMQ_PASSWORD=rabbitmq-pass" in rendered
    assert "REDIS_PASSWORD=redis-pass" in rendered
    assert "FLOWER_PASSWORD=flower-pass" in rendered
    assert "MINIO_ROOT_USER=ocrlocal-user" in rendered
    assert "MINIO_ROOT_PASSWORD=ocrlocal-password" in rendered
    assert "S3_ACCESS_KEY=ocrlocal-user" in rendered
    assert "S3_SECRET_KEY=ocrlocal-password" in rendered
    assert "amqp://ocr_user:rabbitmq-pass@" in rendered
    assert "postgres://ocr:postgres-pass@" in rendered
    assert "DEPLOYMENT_ENV=development" in rendered
    assert "METRICS_API_KEY=metrics-key" in rendered


def test_write_env_file_supports_custom_output(tmp_path: Path, template_text: str):
    template = tmp_path / ".env.example"
    output = tmp_path / ".env.local-s3"
    template.write_text(template_text, encoding="utf-8")

    generated_values = {
        "django_secret": "django-secret",
        "postgres_pass": "postgres-pass",
        "rabbitmq_pass": "rabbitmq-pass",
        "redis_pass": "redis-pass",
        "flower_pass": "flower-pass",
        "metrics_key": "metrics-key",
        "erlang_cookie": "erlang-cookie-value",
        "minio_root_user": "ocrlocal-user",
        "minio_root_password": "ocrlocal-password",
    }

    exit_code = generate_env.write_env_file(
        template,
        output,
        force=False,
        generated_values=generated_values,
    )

    assert exit_code == 0
    assert output.exists()
    rendered = output.read_text(encoding="utf-8")
    assert "MINIO_ROOT_USER=ocrlocal-user" in rendered
    assert "S3_SECRET_KEY=ocrlocal-password" in rendered
