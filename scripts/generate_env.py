#!/usr/bin/env python3
"""Generate coordinator env files from .env.example with secure random credentials.

Usage:
    python scripts/generate_env.py
    python scripts/generate_env.py --force  # overwrite existing .env
    python scripts/generate_env.py --output coordinator/.env.local-s3
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path


def _repo_base_dir() -> Path:
    """Return the repository root containing coordinator/.env.example."""
    return Path(__file__).resolve().parents[1]


def _build_generated_values() -> dict[str, str]:
    """Build the generated credentials used to render an env file."""
    minio_root_user = f"ocrlocal-{secrets.token_hex(6)}"
    minio_root_password = secrets.token_urlsafe(32)
    return {
        "django_secret": secrets.token_urlsafe(50),
        "postgres_pass": secrets.token_urlsafe(32),
        "rabbitmq_pass": secrets.token_urlsafe(32),
        "redis_pass": secrets.token_urlsafe(32),
        "flower_pass": secrets.token_urlsafe(32),
        "metrics_key": secrets.token_urlsafe(32),
        "erlang_cookie": secrets.token_hex(32),
        # Local MinIO root credentials double as the S3 integration credentials.
        "minio_root_user": minio_root_user,
        "minio_root_password": minio_root_password,
    }


def _render_env_content(content: str, generated_values: dict[str, str]) -> str:
    """Render template content with generated credentials."""
    replacements = {
        "DJANGO_SECRET_KEY=change-me-to-a-random-secret-key": (
            f"DJANGO_SECRET_KEY={generated_values['django_secret']}"
        ),
        "POSTGRES_PASSWORD=change-me-to-a-strong-password": (
            f"POSTGRES_PASSWORD={generated_values['postgres_pass']}"
        ),
        "RABBITMQ_PASSWORD=change-me-to-a-strong-password": (
            f"RABBITMQ_PASSWORD={generated_values['rabbitmq_pass']}"
        ),
        "REDIS_PASSWORD=change-me-to-a-strong-password": (
            f"REDIS_PASSWORD={generated_values['redis_pass']}"
        ),
        "FLOWER_PASSWORD=change-me-to-a-strong-password": (
            f"FLOWER_PASSWORD={generated_values['flower_pass']}"
        ),
        "RABBITMQ_ERLANG_COOKIE=change-me-in-production": (
            f"RABBITMQ_ERLANG_COOKIE={generated_values['erlang_cookie']}"
        ),
        "MINIO_ROOT_USER=change-me-to-a-minio-access-key": (
            f"MINIO_ROOT_USER={generated_values['minio_root_user']}"
        ),
        "MINIO_ROOT_PASSWORD=change-me-to-a-minio-secret-key": (
            f"MINIO_ROOT_PASSWORD={generated_values['minio_root_password']}"
        ),
        "S3_ACCESS_KEY=change-me-to-a-minio-access-key": (
            f"S3_ACCESS_KEY={generated_values['minio_root_user']}"
        ),
        "S3_SECRET_KEY=change-me-to-a-minio-secret-key": (
            f"S3_SECRET_KEY={generated_values['minio_root_password']}"
        ),
        "amqp://ocr_user:password@": (
            f"amqp://ocr_user:{generated_values['rabbitmq_pass']}@"
        ),
        "postgres://ocr:password@": (
            f"postgres://ocr:{generated_values['postgres_pass']}@"
        ),
    }

    for placeholder, replacement in replacements.items():
        content = content.replace(placeholder, replacement)

    if "METRICS_API_KEY" not in content:
        content += f"\n# Metrics API Key\nMETRICS_API_KEY={generated_values['metrics_key']}\n"

    return content.replace("DEPLOYMENT_ENV=staging", "DEPLOYMENT_ENV=development")


def write_env_file(
    template: Path,
    output: Path,
    *,
    force: bool,
    generated_values: dict[str, str] | None = None,
) -> int:
    """Write an env file from a template.

    Returns:
        0 on success, 1 on expected usage failure.
    """
    if not template.exists():
        print(f"ERROR: Template not found: {template}", file=sys.stderr)
        return 1

    if output.exists() and not force:
        print(
            f"ERROR: {output} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    content = template.read_text(encoding="utf-8")
    rendered = _render_env_content(
        content,
        generated_values or _build_generated_values(),
    )
    output.write_text(rendered, encoding="utf-8")
    print(f"Generated {output} with unique per-service credentials.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate coordinator env files with secure random credentials"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing .env file"
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=_repo_base_dir() / "coordinator" / ".env.example",
        help="Path to the env template file (default: coordinator/.env.example)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_repo_base_dir() / "coordinator" / ".env",
        help="Path to the generated env file (default: coordinator/.env)",
    )
    args = parser.parse_args()

    return write_env_file(args.template, args.output, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
