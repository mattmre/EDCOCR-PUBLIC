"""Tests for coordinator settings environment parsing."""

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
            "DATABASE_URL": "postgres://testuser:testpass@localhost:5432/testdb",
            "CELERY_BROKER_URL": "amqp://guest:guest@localhost:5672//",
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
