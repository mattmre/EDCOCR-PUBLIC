"""Tests for coordinator settings environment parsing."""

import base64
import importlib
import os
from unittest.mock import patch

import pytest
from django.core.exceptions import ImproperlyConfigured


class TestJobProcessingTimeoutSetting:
    """Validate JOB_PROCESSING_TIMEOUT_MINUTES parsing in production settings."""

    def _env_base(self):
        return {
            "DJANGO_DEBUG": "True",
            "DATABASE_URL": "postgres://testuser:testpass@localhost:5432/testdb",  # pragma: allowlist secret
            "CELERY_BROKER_URL": "amqp://guest:guest@localhost:5672//",  # pragma: allowlist secret
            "DEPLOYMENT_ENV": "development",
        }

    def _reload_settings(self):
        import coordinator.settings as settings_mod

        importlib.reload(settings_mod)
        return settings_mod

    def test_default_job_processing_timeout(self):
        env = self._env_base()
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            assert settings_mod.JOB_PROCESSING_TIMEOUT_MINUTES == 30

    def test_default_s3_cache_budget_is_conservative(self):
        with patch.dict(os.environ, self._env_base(), clear=True):
            settings_mod = self._reload_settings()
            assert settings_mod.S3_CACHE_MAX_SIZE_GB == 1.0
            assert settings_mod.S3_CACHE_MAX_OBJECT_GB == 0.5

    def test_custom_job_processing_timeout(self):
        env = self._env_base()
        env["JOB_PROCESSING_TIMEOUT_MINUTES"] = "45"
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()
            assert settings_mod.JOB_PROCESSING_TIMEOUT_MINUTES == 45

    def test_invalid_job_processing_timeout_rejected(self):
        env = self._env_base()
        env["JOB_PROCESSING_TIMEOUT_MINUTES"] = "0"
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ImproperlyConfigured):
                self._reload_settings()

    def test_stage_recovery_beat_interval_is_bounded(self):
        for configured, expected in ((None, 60), ("1", 5), ("5", 5), ("500", 300)):
            env = self._env_base()
            if configured is not None:
                env["OCR_STAGE_RECOVERY_INTERVAL_SECONDS"] = configured
            with patch.dict(os.environ, env, clear=True):
                settings_mod = self._reload_settings()
                assert settings_mod.OCR_STAGE_RECOVERY_INTERVAL_SECONDS == expected
                assert settings_mod.CELERY_BEAT_SCHEDULE["recover-page-stages"] == {
                    "task": "jobs.tasks.recover_page_stages", "schedule": float(expected),
                }


class TestDatabaseSchemaSetting(TestJobProcessingTimeoutSetting):
    def test_default_schema_is_public(self):
        with patch.dict(os.environ, self._env_base(), clear=True):
            settings_mod = self._reload_settings()

        assert settings_mod.DATABASE_SCHEMA == "public"
        assert settings_mod.DATABASES["default"]["OPTIONS"]["options"] == (
            "-c search_path=public"
        )

    def test_isolated_schema_is_bound_to_connection(self):
        env = self._env_base()
        env["DATABASE_SCHEMA"] = "ocr_processing"
        with patch.dict(os.environ, env, clear=True):
            settings_mod = self._reload_settings()

        assert settings_mod.DATABASE_SCHEMA == "ocr_processing"
        assert settings_mod.DATABASES["default"]["OPTIONS"]["options"] == (
            "-c search_path=ocr_processing"
        )

    @pytest.mark.parametrize(
        "schema",
        ["", "Public", "ocr-processing", "ocr processing", "a" * 64],
    )
    def test_invalid_schema_is_rejected(self, schema):
        env = self._env_base()
        env["DATABASE_SCHEMA"] = schema
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ImproperlyConfigured, match="DATABASE_SCHEMA"):
                self._reload_settings()


class TestPresignedOnlyWorkerSetting(TestJobProcessingTimeoutSetting):
    def _worker_env(self):
        return self._env_base() | {
            "STORAGE_BACKEND": "s3", "S3_ENDPOINT": "http://minio:9000",
            "S3_BUCKET": "lab", "S3_USE_PRESIGNED_URLS": "true",
            "S3_PRESIGNED_TASK_KEY": base64.urlsafe_b64encode(b"x" * 32).decode(),
            "S3_PRESIGNED_ONLY_WORKER": "true",
            "AWS_EC2_METADATA_DISABLED": "true",
        }

    def test_credential_free_worker_starts_only_in_presigned_mode(self):
        with patch.dict(os.environ, self._worker_env(), clear=True):
            settings_mod = self._reload_settings()
        assert settings_mod.S3_PRESIGNED_ONLY_WORKER
        assert settings_mod.S3_ACCESS_KEY == settings_mod.S3_SECRET_KEY == ""

    @pytest.mark.parametrize("change", [
        {"S3_USE_PRESIGNED_URLS": "false"},
        {"S3_ACCESS_KEY": "bad"},
        {"S3_SESSION_TOKEN": "bad"},
        {"AWS_ACCESS_KEY_ID": "bad"},
        {"AWS_WEB_IDENTITY_TOKEN_FILE": "/tmp/token"},
        {"AWS_SHARED_CREDENTIALS_FILE": "/tmp/credentials"},
        {"AWS_EC2_METADATA_DISABLED": "false"},
    ])
    def test_worker_rejects_sdk_mode_or_ambient_credentials(self, change):
        with patch.dict(os.environ, self._worker_env() | change, clear=True):
            with pytest.raises(ImproperlyConfigured):
                self._reload_settings()


class TestIAMStorageSetting(TestJobProcessingTimeoutSetting):
    def _iam_env(self):
        return self._env_base() | {
            "STORAGE_BACKEND": "s3", "S3_AUTH_MODE": "iam", "S3_BUCKET": "owned-fixture",
        }

    def test_iam_configuration_accepts_provider_chain_and_default_aws_endpoint(self):
        with patch.dict(os.environ, self._iam_env(), clear=True):
            configured = self._reload_settings()
        assert configured.S3_AUTH_MODE == "iam"
        assert configured.S3_ENDPOINT == configured.S3_ACCESS_KEY == configured.S3_SECRET_KEY == ""

    @pytest.mark.parametrize("change", [
        {"S3_ACCESS_KEY": "fixture-key"}, {"S3_SECRET_KEY": "fixture-secret"},
        {"S3_SESSION_TOKEN": "fixture-token"}, {"S3_AUTH_MODE": "automatic"},
        {"S3_BUCKET": ""},
    ])
    def test_iam_configuration_rejects_ambiguous_credentials_or_missing_bucket(self, change):
        with patch.dict(os.environ, self._iam_env() | change, clear=True):
            with pytest.raises(ImproperlyConfigured):
                self._reload_settings()

    def test_default_static_mode_still_requires_explicit_credentials(self):
        with patch.dict(os.environ, self._iam_env() | {"S3_AUTH_MODE": "static"}, clear=True):
            with pytest.raises(ImproperlyConfigured, match="S3_ACCESS_KEY"):
                self._reload_settings()

    def test_iam_presigned_mode_requires_an_explicit_worker_host_policy(self):
        env = self._iam_env() | {
            "S3_USE_PRESIGNED_URLS": "true",
            "S3_PRESIGNED_TASK_KEY": base64.urlsafe_b64encode(b"x" * 32).decode(),
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ImproperlyConfigured, match="S3_PRESIGNED_ALLOWED_HOSTS"):
                self._reload_settings()
        with patch.dict(os.environ, env | {"S3_PRESIGNED_ALLOWED_HOSTS": "owned-fixture.s3.amazonaws.com"}, clear=True):
            assert self._reload_settings().S3_USE_PRESIGNED_URLS
