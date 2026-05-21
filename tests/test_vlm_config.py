"""Tests for VLM configuration loading and validation."""

import os
from unittest import mock

import pytest

from vlm_config import VLMConfig, load_vlm_config


class TestVLMConfig:
    """Unit tests for VLMConfig dataclass."""

    def test_default_values(self):
        cfg = VLMConfig()
        assert cfg.enabled is False
        assert cfg.endpoint_url == ""
        assert cfg.api_key == ""
        assert cfg.model_name == "default"
        assert cfg.max_context_pages == 5
        assert cfg.timeout_seconds == 30
        assert cfg.retry_attempts == 3

    def test_custom_values(self):
        cfg = VLMConfig(
            enabled=True,
            endpoint_url="http://vlm:8080/v1",
            api_key="secret123",
            model_name="qwen-vl-7b",
            max_context_pages=10,
            timeout_seconds=60,
            retry_attempts=5,
        )
        assert cfg.enabled is True
        assert cfg.endpoint_url == "http://vlm:8080/v1"
        assert cfg.api_key == "secret123"
        assert cfg.model_name == "qwen-vl-7b"
        assert cfg.max_context_pages == 10
        assert cfg.timeout_seconds == 60
        assert cfg.retry_attempts == 5

    def test_frozen_dataclass(self):
        cfg = VLMConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = True  # type: ignore[misc]


class TestVLMConfigValidation:
    """Tests for VLMConfig.validate()."""

    def test_disabled_config_is_valid(self):
        cfg = VLMConfig(enabled=False)
        assert cfg.validate() == []

    def test_enabled_without_url_is_invalid(self):
        cfg = VLMConfig(enabled=True, endpoint_url="")
        errors = cfg.validate()
        assert len(errors) == 1
        assert "VLM_ENDPOINT_URL" in errors[0]

    def test_enabled_with_valid_url(self):
        cfg = VLMConfig(enabled=True, endpoint_url="http://vlm:8080")
        assert cfg.validate() == []

    def test_invalid_url_scheme(self):
        cfg = VLMConfig(endpoint_url="ftp://vlm:8080")
        errors = cfg.validate()
        assert any("http://" in e for e in errors)

    def test_https_url_is_valid(self):
        cfg = VLMConfig(enabled=True, endpoint_url="https://vlm.example.com")
        assert cfg.validate() == []

    def test_invalid_max_context_pages(self):
        cfg = VLMConfig(max_context_pages=0)
        errors = cfg.validate()
        assert any("max_context_pages" in e.lower() for e in errors)

    def test_invalid_timeout(self):
        cfg = VLMConfig(timeout_seconds=0)
        errors = cfg.validate()
        assert any("timeout" in e.lower() for e in errors)

    def test_negative_retry_attempts(self):
        cfg = VLMConfig(retry_attempts=-1)
        errors = cfg.validate()
        assert any("retry" in e.lower() for e in errors)

    def test_multiple_validation_errors(self):
        cfg = VLMConfig(
            enabled=True,
            endpoint_url="",
            max_context_pages=0,
            timeout_seconds=0,
        )
        errors = cfg.validate()
        assert len(errors) >= 3


class TestLoadVLMConfig:
    """Tests for load_vlm_config() env var parsing."""

    def test_defaults_when_no_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = load_vlm_config()
            assert cfg.enabled is False
            assert cfg.endpoint_url == ""
            assert cfg.model_name == "default"
            assert cfg.max_context_pages == 5
            assert cfg.timeout_seconds == 30
            assert cfg.retry_attempts == 3

    def test_enabled_from_env(self):
        env = {"VLM_ENABLED": "true", "VLM_ENDPOINT_URL": "http://localhost:8080"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = load_vlm_config()
            assert cfg.enabled is True
            assert cfg.endpoint_url == "http://localhost:8080"

    def test_enabled_with_1(self):
        with mock.patch.dict(os.environ, {"VLM_ENABLED": "1"}, clear=True):
            cfg = load_vlm_config()
            assert cfg.enabled is True

    def test_enabled_with_yes(self):
        with mock.patch.dict(os.environ, {"VLM_ENABLED": "yes"}, clear=True):
            cfg = load_vlm_config()
            assert cfg.enabled is True

    def test_disabled_explicit(self):
        with mock.patch.dict(os.environ, {"VLM_ENABLED": "false"}, clear=True):
            cfg = load_vlm_config()
            assert cfg.enabled is False

    def test_api_key_from_env(self):
        with mock.patch.dict(os.environ, {"VLM_API_KEY": "  sk-test  "}, clear=True):
            cfg = load_vlm_config()
            assert cfg.api_key == "sk-test"

    def test_model_name_from_env(self):
        with mock.patch.dict(os.environ, {"VLM_MODEL_NAME": "llava-1.6"}, clear=True):
            cfg = load_vlm_config()
            assert cfg.model_name == "llava-1.6"

    def test_max_context_pages_from_env(self):
        with mock.patch.dict(os.environ, {"VLM_MAX_CONTEXT_PAGES": "10"}, clear=True):
            cfg = load_vlm_config()
            assert cfg.max_context_pages == 10

    def test_max_context_pages_clamped(self):
        with mock.patch.dict(os.environ, {"VLM_MAX_CONTEXT_PAGES": "999"}, clear=True):
            cfg = load_vlm_config()
            assert cfg.max_context_pages == 100  # max_val

    def test_timeout_from_env(self):
        with mock.patch.dict(os.environ, {"VLM_TIMEOUT_SECONDS": "120"}, clear=True):
            cfg = load_vlm_config()
            assert cfg.timeout_seconds == 120

    def test_retry_from_env(self):
        with mock.patch.dict(os.environ, {"VLM_RETRY_ATTEMPTS": "0"}, clear=True):
            cfg = load_vlm_config()
            assert cfg.retry_attempts == 0

    def test_invalid_int_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"VLM_MAX_CONTEXT_PAGES": "notanumber"}, clear=True):
            cfg = load_vlm_config()
            assert cfg.max_context_pages == 5
