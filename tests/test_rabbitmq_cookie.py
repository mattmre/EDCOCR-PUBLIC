"""Tests for C-05: RABBITMQ_ERLANG_COOKIE randomization.

Validates that:
- generate_env.py produces a random erlang_cookie value
- generate_env.py replaces the placeholder in rendered .env content
- validate_phase7c_env.py flags the placeholder cookie in strict mode
- Helm secret.yaml always renders an erlang cookie (with or without explicit value)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import generate_env
import validate_phase7c_env as env_script

# ---------------------------------------------------------------------------
# generate_env: erlang_cookie generation
# ---------------------------------------------------------------------------


class TestGenerateEnvErlangCookie:
    """Tests for erlang cookie generation in generate_env.py."""

    def test_build_generated_values_includes_erlang_cookie(self):
        """_build_generated_values must include an erlang_cookie key."""
        values = generate_env._build_generated_values()
        assert "erlang_cookie" in values
        assert len(values["erlang_cookie"]) > 0

    def test_erlang_cookie_is_random_hex(self):
        """erlang_cookie should be a hex string (token_hex output)."""
        values = generate_env._build_generated_values()
        cookie = values["erlang_cookie"]
        # token_hex(32) produces 64 hex characters
        assert len(cookie) == 64
        assert all(c in "0123456789abcdef" for c in cookie)

    def test_erlang_cookie_is_unique_per_call(self):
        """Two calls should produce different cookies."""
        v1 = generate_env._build_generated_values()
        v2 = generate_env._build_generated_values()
        assert v1["erlang_cookie"] != v2["erlang_cookie"]

    def test_render_replaces_placeholder_cookie(self):
        """_render_env_content must replace the placeholder cookie value."""
        template = (
            "RABBITMQ_ERLANG_COOKIE=change-me-in-production\n"
            "DEPLOYMENT_ENV=staging\n"
        )
        values = {
            "django_secret": "ds",
            "postgres_pass": "pp",
            "rabbitmq_pass": "rp",
            "redis_pass": "rdp",
            "flower_pass": "fp",
            "metrics_key": "mk",
            "erlang_cookie": "abc123def456",
            "minio_root_user": "mu",
            "minio_root_password": "mp",
        }
        rendered = generate_env._render_env_content(template, values)
        assert "change-me-in-production" not in rendered
        assert "RABBITMQ_ERLANG_COOKIE=abc123def456" in rendered

    def test_render_preserves_cookie_when_not_placeholder(self):
        """If the cookie value is not the placeholder, it should stay unchanged."""
        template = (
            "RABBITMQ_ERLANG_COOKIE=already-set-to-real-value\n"
            "DEPLOYMENT_ENV=staging\n"
        )
        values = {
            "django_secret": "ds",
            "postgres_pass": "pp",
            "rabbitmq_pass": "rp",
            "redis_pass": "rdp",
            "flower_pass": "fp",
            "metrics_key": "mk",
            "erlang_cookie": "new-random-cookie",
            "minio_root_user": "mu",
            "minio_root_password": "mp",
        }
        rendered = generate_env._render_env_content(template, values)
        assert "RABBITMQ_ERLANG_COOKIE=already-set-to-real-value" in rendered

    def test_write_env_file_replaces_cookie(self, tmp_path: Path):
        """Full write_env_file flow should replace the cookie placeholder."""
        template = tmp_path / ".env.example"
        output = tmp_path / ".env"
        template.write_text(
            "RABBITMQ_ERLANG_COOKIE=change-me-in-production\n"
            "DEPLOYMENT_ENV=staging\n",
            encoding="utf-8",
        )
        values = {
            "django_secret": "ds",
            "postgres_pass": "pp",
            "rabbitmq_pass": "rp",
            "redis_pass": "rdp",
            "flower_pass": "fp",
            "metrics_key": "mk",
            "erlang_cookie": "test-cookie-value",
            "minio_root_user": "mu",
            "minio_root_password": "mp",
        }
        rc = generate_env.write_env_file(template, output, force=False, generated_values=values)
        assert rc == 0
        content = output.read_text(encoding="utf-8")
        assert "RABBITMQ_ERLANG_COOKIE=test-cookie-value" in content
        assert "change-me-in-production" not in content


# ---------------------------------------------------------------------------
# validate_phase7c_env: erlang cookie validation
# ---------------------------------------------------------------------------


class TestValidateErlangCookie:
    """Tests for erlang cookie validation in validate_phase7c_env.py."""

    def test_placeholder_cookie_fails_strict_baseline(self):
        """Placeholder cookie value must fail baseline validation in strict mode."""
        env_vars = {
            "DJANGO_SECRET_KEY": "real-secret",
            "POSTGRES_PASSWORD": "real-pass",
            "RABBITMQ_PASSWORD": "real-pass",
            "RABBITMQ_ERLANG_COOKIE": "change-me-in-production",
            "DATABASE_URL": "postgres://ocr:real@db:5432/ocr",
            "CELERY_BROKER_URL": "amqp://ocr:real@rmq:5672//",
        }
        results = env_script._validate_baseline(env_vars, strict_placeholders=True)
        cookie_result = next(r for r in results if r.key == "RABBITMQ_ERLANG_COOKIE")
        assert cookie_result.status == "placeholder", (
            f"Expected 'placeholder' status but got '{cookie_result.status}'"
        )

    def test_real_cookie_passes_strict_baseline(self):
        """A real cookie value must pass baseline validation."""
        env_vars = {
            "DJANGO_SECRET_KEY": "real-secret",
            "POSTGRES_PASSWORD": "real-pass",
            "RABBITMQ_PASSWORD": "real-pass",
            "RABBITMQ_ERLANG_COOKIE": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
            "DATABASE_URL": "postgres://ocr:real@db:5432/ocr",
            "CELERY_BROKER_URL": "amqp://ocr:real@rmq:5672//",
        }
        results = env_script._validate_baseline(env_vars, strict_placeholders=True)
        cookie_result = next(r for r in results if r.key == "RABBITMQ_ERLANG_COOKIE")
        assert cookie_result.status == "ok"

    def test_missing_cookie_fails_baseline(self):
        """Missing cookie must be flagged as missing in baseline."""
        env_vars = {
            "DJANGO_SECRET_KEY": "real-secret",
            "POSTGRES_PASSWORD": "real-pass",
            "RABBITMQ_PASSWORD": "real-pass",
            "DATABASE_URL": "postgres://ocr:real@db:5432/ocr",
            "CELERY_BROKER_URL": "amqp://ocr:real@rmq:5672//",
        }
        results = env_script._validate_baseline(env_vars, strict_placeholders=True)
        cookie_result = next(r for r in results if r.key == "RABBITMQ_ERLANG_COOKIE")
        assert cookie_result.status == "missing"

    def test_empty_cookie_fails_baseline(self):
        """Empty cookie must be flagged as empty in baseline."""
        env_vars = {
            "DJANGO_SECRET_KEY": "real-secret",
            "POSTGRES_PASSWORD": "real-pass",
            "RABBITMQ_PASSWORD": "real-pass",
            "RABBITMQ_ERLANG_COOKIE": "",
            "DATABASE_URL": "postgres://ocr:real@db:5432/ocr",
            "CELERY_BROKER_URL": "amqp://ocr:real@rmq:5672//",
        }
        results = env_script._validate_baseline(env_vars, strict_placeholders=True)
        cookie_result = next(r for r in results if r.key == "RABBITMQ_ERLANG_COOKIE")
        assert cookie_result.status == "empty"

    def test_placeholder_cookie_passes_non_strict(self):
        """Placeholder cookie should pass when strict_placeholders is False."""
        env_vars = {
            "DJANGO_SECRET_KEY": "real-secret",
            "POSTGRES_PASSWORD": "real-pass",
            "RABBITMQ_PASSWORD": "real-pass",
            "RABBITMQ_ERLANG_COOKIE": "change-me-in-production",
            "DATABASE_URL": "postgres://ocr:real@db:5432/ocr",
            "CELERY_BROKER_URL": "amqp://ocr:real@rmq:5672//",
        }
        results = env_script._validate_baseline(env_vars, strict_placeholders=False)
        cookie_result = next(r for r in results if r.key == "RABBITMQ_ERLANG_COOKIE")
        assert cookie_result.status == "ok"

    def test_run_fails_with_placeholder_cookie(self, tmp_path: Path):
        """Full run() should return non-zero when cookie has placeholder value."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "DJANGO_SECRET_KEY=real-secret-key-value\n"
            "POSTGRES_PASSWORD=real-pg-password\n"
            "RABBITMQ_PASSWORD=real-rmq-password\n"
            "RABBITMQ_ERLANG_COOKIE=change-me-in-production\n"
            "DATABASE_URL=postgres://ocr:real@db:5432/ocr\n"
            "CELERY_BROKER_URL=amqp://ocr:real@rmq:5672//\n",
            encoding="utf-8",
        )
        rc = env_script.run(
            env_path=env_file,
            report=None,
            strict_placeholders=True,
        )
        assert rc != 0, "Expected non-zero exit when cookie is placeholder"

    def test_baseline_key_count_includes_cookie(self):
        """RABBITMQ_ERLANG_COOKIE must be in BASELINE_REQUIRED_KEYS."""
        assert "RABBITMQ_ERLANG_COOKIE" in env_script.BASELINE_REQUIRED_KEYS

    def test_cookie_not_duplicated_in_production_keys(self):
        """RABBITMQ_ERLANG_COOKIE should not be in PRODUCTION_REQUIRED_KEYS (moved to baseline)."""
        assert "RABBITMQ_ERLANG_COOKIE" not in env_script.PRODUCTION_REQUIRED_KEYS


# ---------------------------------------------------------------------------
# Helm template: secret.yaml and rabbitmq-statefulset.yaml
# ---------------------------------------------------------------------------


class TestHelmTemplateErlangCookie:
    """Tests for Helm template changes (static YAML content checks)."""

    @pytest.fixture
    def secret_yaml(self) -> str:
        secret_path = (
            Path(__file__).parent.parent
            / "helm"
            / "ocr-local"
            / "templates"
            / "secret.yaml"
        )
        return secret_path.read_text(encoding="utf-8")

    @pytest.fixture
    def statefulset_yaml(self) -> str:
        statefulset_path = (
            Path(__file__).parent.parent
            / "helm"
            / "ocr-local"
            / "templates"
            / "rabbitmq-statefulset.yaml"
        )
        return statefulset_path.read_text(encoding="utf-8")

    def test_secret_yaml_has_else_branch_for_cookie(self, secret_yaml: str):
        """secret.yaml must auto-generate cookie when erlangCookieSecret is empty."""
        assert "randAlphaNum 64" in secret_yaml, (
            "secret.yaml must use randAlphaNum 64 to auto-generate cookie"
        )
        assert "{{- else }}" in secret_yaml, (
            "secret.yaml must have an else branch for cookie generation"
        )

    def test_secret_yaml_uses_lookup_for_stable_cookie(self, secret_yaml: str):
        """secret.yaml must use lookup to preserve cookie across upgrades."""
        assert "lookup" in secret_yaml, (
            "secret.yaml must use Helm lookup to check for existing secret"
        )
        assert "b64dec" in secret_yaml, (
            "secret.yaml must decode the existing cookie from base64"
        )

    def test_secret_yaml_always_renders_cookie_key(self, secret_yaml: str):
        """secret.yaml must always render RABBITMQ_ERLANG_COOKIE (no outer skip)."""
        # Both branches should output the same key name
        lines = secret_yaml.splitlines()
        cookie_key_lines = [
            line for line in lines if "RABBITMQ_ERLANG_COOKIE:" in line
        ]
        # Should appear exactly twice: once for the if branch, once for the else
        assert len(cookie_key_lines) == 2, (
            f"Expected 2 RABBITMQ_ERLANG_COOKIE lines (if/else), found {len(cookie_key_lines)}"
        )

    def test_statefulset_always_injects_cookie_env(self, statefulset_yaml: str):
        """rabbitmq-statefulset.yaml must unconditionally inject RABBITMQ_ERLANG_COOKIE."""
        assert "RABBITMQ_ERLANG_COOKIE" in statefulset_yaml
        # The env var injection should NOT be wrapped in a conditional
        lines = statefulset_yaml.splitlines()
        for i, line in enumerate(lines):
            if "name: RABBITMQ_ERLANG_COOKIE" in line:
                # Check the line before is not a conditional
                if i > 0:
                    prev = lines[i - 1].strip()
                    assert not prev.startswith("{{- if"), (
                        "RABBITMQ_ERLANG_COOKIE env injection should not be conditional"
                    )
                break
        else:
            pytest.fail("RABBITMQ_ERLANG_COOKIE env var not found in statefulset")


# ---------------------------------------------------------------------------
# Helm template: worker readiness probes ( / INFRA-003)
# ---------------------------------------------------------------------------


class TestWorkerReadinessProbes:
    """Tests that worker deployments have readiness probes to prevent
    task routing to uninitialized pods during rolling updates."""

    @pytest.fixture
    def gpu_worker_yaml(self) -> str:
        path = (
            Path(__file__).parent.parent
            / "helm"
            / "ocr-local"
            / "templates"
            / "gpu-worker-deployment.yaml"
        )
        return path.read_text(encoding="utf-8")

    @pytest.fixture
    def cpu_worker_yaml(self) -> str:
        path = (
            Path(__file__).parent.parent
            / "helm"
            / "ocr-local"
            / "templates"
            / "cpu-worker-deployment.yaml"
        )
        return path.read_text(encoding="utf-8")

    @pytest.fixture
    def cpu_ocr_worker_yaml(self) -> str:
        path = (
            Path(__file__).parent.parent
            / "helm"
            / "ocr-local"
            / "templates"
            / "cpu-ocr-worker-deployment.yaml"
        )
        return path.read_text(encoding="utf-8")

    def test_gpu_worker_has_readiness_probe(self, gpu_worker_yaml: str):
        """GPU worker must have a readinessProbe."""
        assert "readinessProbe:" in gpu_worker_yaml

    def test_gpu_worker_has_startup_probe(self, gpu_worker_yaml: str):
        """GPU worker must have a startupProbe."""
        assert "startupProbe:" in gpu_worker_yaml

    def test_gpu_worker_has_liveness_probe(self, gpu_worker_yaml: str):
        """GPU worker must have a livenessProbe."""
        assert "livenessProbe:" in gpu_worker_yaml

    def test_cpu_worker_has_readiness_probe(self, cpu_worker_yaml: str):
        """CPU worker must have a readinessProbe."""
        assert "readinessProbe:" in cpu_worker_yaml

    def test_cpu_worker_has_startup_probe(self, cpu_worker_yaml: str):
        """CPU worker must have a startupProbe."""
        assert "startupProbe:" in cpu_worker_yaml

    def test_cpu_worker_has_liveness_probe(self, cpu_worker_yaml: str):
        """CPU worker must have a livenessProbe."""
        assert "livenessProbe:" in cpu_worker_yaml

    def test_cpu_ocr_worker_has_readiness_probe(self, cpu_ocr_worker_yaml: str):
        """CPU OCR worker must have a readinessProbe."""
        assert "readinessProbe:" in cpu_ocr_worker_yaml

    def test_cpu_ocr_worker_has_startup_probe(self, cpu_ocr_worker_yaml: str):
        """CPU OCR worker must have a startupProbe."""
        assert "startupProbe:" in cpu_ocr_worker_yaml

    def test_cpu_ocr_worker_has_liveness_probe(self, cpu_ocr_worker_yaml: str):
        """CPU OCR worker must have a livenessProbe."""
        assert "livenessProbe:" in cpu_ocr_worker_yaml

    def test_readiness_probe_uses_celery_inspect(self, gpu_worker_yaml: str):
        """Readiness probe should use celery inspect ping for meaningful check."""
        # Find the readinessProbe section and verify it uses celery inspect
        lines = gpu_worker_yaml.splitlines()
        in_readiness = False
        found_celery = False
        for line in lines:
            if "readinessProbe:" in line:
                in_readiness = True
            elif in_readiness and "Probe:" in line:
                break  # Hit next probe section
            elif in_readiness and "inspect" in line:
                found_celery = True
                break
        assert found_celery, (
            "readinessProbe should use celery inspect ping"
        )
