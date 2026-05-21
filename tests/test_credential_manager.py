"""Tests for credential_manager module.

Covers all backends (mocked), fallback chain, placeholder detection,
missing credential handling, and the module-level convenience API.
"""

from __future__ import annotations

import base64
import json
import os
from unittest import mock

import pytest

from credential_manager import (
    CREDENTIAL_SCHEMA,
    PLACEHOLDER_VALUES,
    AWSKMSBackend,
    AWSSecretsBackend,
    CredentialBackend,
    CredentialError,
    CredentialManager,
    CredentialValidationResult,
    EnvVarBackend,
    FileBackend,
    ValidationReport,
    VaultBackend,
    get_credential,
    is_placeholder,
    reset_default_manager,
    validate_credentials,
)

# ---------------------------------------------------------------------------
# is_placeholder tests
# ---------------------------------------------------------------------------


class TestIsPlaceholder:
    """Tests for the is_placeholder function."""

    def test_empty_string(self):
        assert is_placeholder("") is True

    def test_none_like_empty(self):
        assert is_placeholder("  ") is True

    @pytest.mark.parametrize("value", sorted(PLACEHOLDER_VALUES))
    def test_exact_placeholder_values(self, value):
        assert is_placeholder(value) is True

    @pytest.mark.parametrize("value", sorted(PLACEHOLDER_VALUES))
    def test_exact_placeholder_case_insensitive(self, value):
        assert is_placeholder(value.upper()) is True

    def test_minioadmin(self):
        assert is_placeholder("minioadmin") is True

    def test_changeme(self):
        assert is_placeholder("changeme") is True

    def test_change_me_with_hyphens(self):
        assert is_placeholder("change-me-to-a-strong-password") is True

    def test_change_me_with_underscores(self):
        assert is_placeholder("change_me_secret") is True

    def test_example_substring(self):
        assert is_placeholder("example-secret-key") is True

    def test_your_prefix(self):
        assert is_placeholder("your_password_here") is True

    def test_placeholder_substring(self):
        assert is_placeholder("some-placeholder-value") is True

    def test_replace_me(self):
        assert is_placeholder("replace-me-in-production") is True

    def test_real_password(self):
        assert is_placeholder("x9kF!mQ2pL@w3nR7") is False

    def test_real_secret_key(self):
        assert is_placeholder("dj4kBm2nPq8rYw5tXz1aGh6cEf9iJl0o") is False

    def test_uuid_style(self):
        assert is_placeholder("a1b2c3d4-e5f6-7890-abcd-ef1234567890") is False

    def test_url_with_placeholder_password(self):
        assert is_placeholder("amqp://user:password@host:5672//") is True

    def test_url_with_changeme_password(self):
        assert is_placeholder("postgres://ocr:change-me@host:5432/db") is True

    def test_url_with_real_password(self):
        assert is_placeholder("amqp://ocr_user:x9kF!mQ2pL@host:5672//") is False

    def test_url_with_minioadmin_user(self):
        assert is_placeholder("http://minioadmin:minioadmin@minio:9000") is True

    def test_xxx_substring(self):
        assert is_placeholder("xxx-placeholder") is True

    def test_todo_substring(self):
        assert is_placeholder("todo-replace") is True

    def test_fixme_substring(self):
        assert is_placeholder("fixme-secret") is True


# ---------------------------------------------------------------------------
# EnvVarBackend tests
# ---------------------------------------------------------------------------


class TestEnvVarBackend:
    """Tests for the EnvVarBackend."""

    def test_available_always_true(self):
        backend = EnvVarBackend()
        assert backend.available() is True

    def test_get_existing_var(self):
        backend = EnvVarBackend()
        with mock.patch.dict(os.environ, {"TEST_CRED_KEY": "my_secret"}):
            assert backend.get("TEST_CRED_KEY") == "my_secret"

    def test_get_missing_var(self):
        backend = EnvVarBackend()
        with mock.patch.dict(os.environ, {}, clear=True):
            assert backend.get("NONEXISTENT_KEY_12345") is None

    def test_get_empty_var(self):
        backend = EnvVarBackend()
        with mock.patch.dict(os.environ, {"EMPTY_CRED": ""}):
            assert backend.get("EMPTY_CRED") is None

    def test_get_whitespace_only(self):
        backend = EnvVarBackend()
        with mock.patch.dict(os.environ, {"WS_CRED": "   "}):
            assert backend.get("WS_CRED") is None

    def test_get_strips_whitespace(self):
        backend = EnvVarBackend()
        with mock.patch.dict(os.environ, {"PADDED_CRED": "  value  "}):
            assert backend.get("PADDED_CRED") == "value"

    def test_name(self):
        assert EnvVarBackend.name == "env"


# ---------------------------------------------------------------------------
# VaultBackend tests
# ---------------------------------------------------------------------------


class TestVaultBackend:
    """Tests for the VaultBackend (mocked hvac)."""

    def test_name(self):
        assert VaultBackend.name == "vault"

    def test_available_without_hvac(self):
        backend = VaultBackend()
        with mock.patch.dict("sys.modules", {"hvac": None}):
            # Import will fail, so available() should return False
            assert backend.available() is False

    def test_available_without_env_vars(self):
        backend = VaultBackend()
        with mock.patch.dict(os.environ, {}, clear=True):
            assert backend.available() is False

    def test_available_with_config(self):
        backend = VaultBackend()
        mock_hvac = mock.MagicMock()
        with mock.patch.dict(
            os.environ, {"VAULT_ADDR": "https://vault:8200", "VAULT_TOKEN": "tok"}
        ):
            with mock.patch.dict("sys.modules", {"hvac": mock_hvac}):
                assert backend.available() is True

    def test_get_loads_secrets_from_vault(self):
        backend = VaultBackend()
        mock_hvac = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_hvac.Client.return_value = mock_client
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {
                "data": {
                    "POSTGRES_PASSWORD": "vault_password",
                    "DJANGO_SECRET_KEY": "vault_django_key",
                }
            }
        }
        with mock.patch.dict(
            os.environ,
            {
                "VAULT_ADDR": "https://vault:8200",
                "VAULT_TOKEN": "tok",
                "VAULT_SECRET_PATH": "secret/data/ocr-local",
            },
        ):
            with mock.patch.dict("sys.modules", {"hvac": mock_hvac}):
                result = backend.get("POSTGRES_PASSWORD")
                assert result == "vault_password"

    def test_get_returns_none_for_missing_key(self):
        backend = VaultBackend()
        mock_hvac = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_hvac.Client.return_value = mock_client
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"OTHER_KEY": "value"}}
        }
        with mock.patch.dict(
            os.environ, {"VAULT_ADDR": "https://vault:8200", "VAULT_TOKEN": "tok"}
        ):
            with mock.patch.dict("sys.modules", {"hvac": mock_hvac}):
                assert backend.get("NONEXISTENT") is None

    def test_get_handles_vault_exception(self):
        backend = VaultBackend()
        mock_hvac = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_hvac.Client.return_value = mock_client
        mock_client.secrets.kv.v2.read_secret_version.side_effect = Exception("Vault error")
        with mock.patch.dict(
            os.environ, {"VAULT_ADDR": "https://vault:8200", "VAULT_TOKEN": "tok"}
        ):
            with mock.patch.dict("sys.modules", {"hvac": mock_hvac}):
                assert backend.get("POSTGRES_PASSWORD") is None

    def test_get_no_hvac_installed(self):
        backend = VaultBackend()
        # Ensure hvac cannot be imported
        with mock.patch.dict(os.environ, {"VAULT_ADDR": "https://vault:8200", "VAULT_TOKEN": "tok"}):
            with mock.patch("builtins.__import__", side_effect=_make_import_blocker("hvac")):
                assert backend.get("POSTGRES_PASSWORD") is None

    def test_vault_namespace_passed(self):
        backend = VaultBackend()
        mock_hvac = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_hvac.Client.return_value = mock_client
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {}}
        }
        with mock.patch.dict(
            os.environ,
            {
                "VAULT_ADDR": "https://vault:8200",
                "VAULT_TOKEN": "tok",
                "VAULT_NAMESPACE": "my-namespace",
            },
        ):
            with mock.patch.dict("sys.modules", {"hvac": mock_hvac}):
                backend.get("SOME_KEY")
                mock_hvac.Client.assert_called_once_with(
                    url="https://vault:8200", token="tok", namespace="my-namespace"
                )


def _make_import_blocker(blocked_module: str):
    """Create an __import__ side_effect that blocks a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _blocker(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"Mocked: {blocked_module} not installed")
        return real_import(name, *args, **kwargs)

    return _blocker


# ---------------------------------------------------------------------------
# AWSSecretsBackend tests
# ---------------------------------------------------------------------------


class TestAWSSecretsBackend:
    """Tests for the AWSSecretsBackend (mocked boto3)."""

    def test_name(self):
        assert AWSSecretsBackend.name == "aws-secrets"

    def test_available_without_boto3(self):
        backend = AWSSecretsBackend()
        with mock.patch.dict("sys.modules", {"boto3": None}):
            assert backend.available() is False

    def test_available_without_env(self):
        backend = AWSSecretsBackend()
        with mock.patch.dict(os.environ, {}, clear=True):
            assert backend.available() is False

    def test_available_with_config(self):
        backend = AWSSecretsBackend()
        mock_boto3 = mock.MagicMock()
        with mock.patch.dict(os.environ, {"AWS_SECRET_NAME": "ocr-local/creds"}):
            with mock.patch.dict("sys.modules", {"boto3": mock_boto3}):
                assert backend.available() is True

    def test_get_loads_from_aws(self):
        backend = AWSSecretsBackend()
        mock_boto3 = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"POSTGRES_PASSWORD": "aws_password"})
        }
        with mock.patch.dict(
            os.environ,
            {"AWS_SECRET_NAME": "ocr-local/creds", "AWS_REGION_NAME": "us-west-2"},
        ):
            with mock.patch.dict("sys.modules", {"boto3": mock_boto3}):
                assert backend.get("POSTGRES_PASSWORD") == "aws_password"

    def test_get_returns_none_for_missing(self):
        backend = AWSSecretsBackend()
        mock_boto3 = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"OTHER": "value"})
        }
        with mock.patch.dict(os.environ, {"AWS_SECRET_NAME": "test"}):
            with mock.patch.dict("sys.modules", {"boto3": mock_boto3}):
                assert backend.get("NONEXISTENT") is None

    def test_get_handles_exception(self):
        backend = AWSSecretsBackend()
        mock_boto3 = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.side_effect = Exception("AWS error")
        with mock.patch.dict(os.environ, {"AWS_SECRET_NAME": "test"}):
            with mock.patch.dict("sys.modules", {"boto3": mock_boto3}):
                assert backend.get("POSTGRES_PASSWORD") is None

    def test_default_region(self):
        backend = AWSSecretsBackend()
        mock_boto3 = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {"SecretString": "{}"}
        with mock.patch.dict(os.environ, {"AWS_SECRET_NAME": "test"}, clear=False):
            env_copy = dict(os.environ)
            env_copy.pop("AWS_REGION_NAME", None)
            with mock.patch.dict(os.environ, env_copy, clear=True):
                with mock.patch.dict("sys.modules", {"boto3": mock_boto3}):
                    backend.get("KEY")
                    mock_boto3.client.assert_called_with(
                        "secretsmanager", region_name="us-east-1"
                    )


# ---------------------------------------------------------------------------
# FileBackend tests
# ---------------------------------------------------------------------------


class TestFileBackend:
    """Tests for the FileBackend."""

    def test_name(self):
        assert FileBackend.name == "file"

    def test_available_without_path(self):
        backend = FileBackend()
        with mock.patch.dict(os.environ, {}, clear=True):
            assert backend.available() is False

    def test_available_with_nonexistent_path(self):
        backend = FileBackend()
        with mock.patch.dict(os.environ, {"CREDENTIAL_FILE_PATH": "/nonexistent"}):
            assert backend.available() is False

    def test_get_plain_json(self, tmp_path):
        backend = FileBackend()
        creds = {"POSTGRES_PASSWORD": "file_pw", "S3_ACCESS_KEY": "file_ak"}
        cred_file = tmp_path / "creds.json"
        cred_file.write_text(json.dumps(creds), encoding="utf-8")
        with mock.patch.dict(
            os.environ,
            {"CREDENTIAL_FILE_PATH": str(cred_file), "CREDENTIAL_FILE_ENCODING": "plain"},
        ):
            assert backend.get("POSTGRES_PASSWORD") == "file_pw"
            assert backend.get("S3_ACCESS_KEY") == "file_ak"
            assert backend.get("NONEXISTENT") is None

    def test_get_base64_json(self, tmp_path):
        backend = FileBackend()
        creds = {"DJANGO_SECRET_KEY": "b64_django_key"}
        encoded = base64.b64encode(json.dumps(creds).encode())
        cred_file = tmp_path / "creds.json.b64"
        cred_file.write_bytes(encoded)
        with mock.patch.dict(
            os.environ,
            {"CREDENTIAL_FILE_PATH": str(cred_file), "CREDENTIAL_FILE_ENCODING": "base64"},
        ):
            assert backend.get("DJANGO_SECRET_KEY") == "b64_django_key"

    def test_get_nonexistent_file(self):
        backend = FileBackend()
        with mock.patch.dict(
            os.environ, {"CREDENTIAL_FILE_PATH": "/this/path/does/not/exist.json"}
        ):
            assert backend.get("ANY_KEY") is None

    def test_get_invalid_json(self, tmp_path):
        backend = FileBackend()
        cred_file = tmp_path / "bad.json"
        cred_file.write_text("not valid json{{{", encoding="utf-8")
        with mock.patch.dict(os.environ, {"CREDENTIAL_FILE_PATH": str(cred_file)}):
            assert backend.get("KEY") is None

    def test_available_with_valid_file(self, tmp_path):
        backend = FileBackend()
        cred_file = tmp_path / "creds.json"
        cred_file.write_text(json.dumps({"key": "val"}), encoding="utf-8")
        with mock.patch.dict(os.environ, {"CREDENTIAL_FILE_PATH": str(cred_file)}):
            assert backend.available() is True


# ---------------------------------------------------------------------------
# CredentialManager tests
# ---------------------------------------------------------------------------


class TestCredentialManager:
    """Tests for the CredentialManager orchestrator."""

    def test_default_backends_includes_env(self):
        manager = CredentialManager()
        names = [b.name for b in manager.backends]
        assert "env" in names

    def test_default_backends_includes_all_five(self):
        manager = CredentialManager()
        names = [b.name for b in manager.backends]
        assert names == ["env", "vault", "aws-secrets", "aws-kms", "file"]

    def test_custom_backends(self):
        b1 = EnvVarBackend()
        manager = CredentialManager(backends=[b1])
        assert manager.backends == [b1]

    def test_get_from_env(self):
        manager = CredentialManager(backends=[EnvVarBackend()])
        with mock.patch.dict(os.environ, {"MY_SECRET": "env_val"}):
            assert manager.get("MY_SECRET") == "env_val"

    def test_get_returns_default_when_missing(self):
        manager = CredentialManager(backends=[EnvVarBackend()])
        with mock.patch.dict(os.environ, {}, clear=True):
            assert manager.get("MISSING_KEY_999", "fallback") == "fallback"

    def test_get_returns_none_by_default(self):
        manager = CredentialManager(backends=[EnvVarBackend()])
        with mock.patch.dict(os.environ, {}, clear=True):
            assert manager.get("MISSING_KEY_999") is None

    def test_get_required_success(self):
        manager = CredentialManager(backends=[EnvVarBackend()])
        with mock.patch.dict(os.environ, {"REQUIRED_KEY": "value"}):
            assert manager.get_required("REQUIRED_KEY") == "value"

    def test_get_required_raises_on_missing(self):
        manager = CredentialManager(backends=[EnvVarBackend()])
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LookupError, match="Required credential"):
                manager.get_required("MISSING_REQUIRED")

    def test_fallback_chain_prefers_first_backend(self):
        """First backend that returns a value wins."""

        class FixedBackend(CredentialBackend):
            name = "fixed"

            def __init__(self, data):
                self._data = data

            def get(self, key):
                return self._data.get(key)

            def available(self):
                return True

        b1 = FixedBackend({"KEY": "from_b1"})
        b2 = FixedBackend({"KEY": "from_b2"})
        manager = CredentialManager(backends=[b1, b2])
        assert manager.get("KEY") == "from_b1"

    def test_fallback_chain_tries_next_on_none(self):
        """If first backend returns None, try next."""

        class FixedBackend(CredentialBackend):
            name = "fixed"

            def __init__(self, data):
                self._data = data

            def get(self, key):
                return self._data.get(key)

            def available(self):
                return True

        b1 = FixedBackend({})
        b2 = FixedBackend({"KEY": "from_b2"})
        manager = CredentialManager(backends=[b1, b2])
        assert manager.get("KEY") == "from_b2"

    def test_fallback_chain_skips_erroring_backend(self):
        """If a backend raises, skip to next."""

        class ErrorBackend(CredentialBackend):
            name = "error"

            def get(self, key):
                raise RuntimeError("Backend failed")

            def available(self):
                return True

        class FixedBackend(CredentialBackend):
            name = "fixed"

            def __init__(self, data):
                self._data = data

            def get(self, key):
                return self._data.get(key)

            def available(self):
                return True

        b1 = ErrorBackend()
        b2 = FixedBackend({"KEY": "fallback_value"})
        manager = CredentialManager(backends=[b1, b2])
        assert manager.get("KEY") == "fallback_value"

    def test_available_backends(self):
        manager = CredentialManager(backends=[EnvVarBackend()])
        assert manager.available_backends() == ["env"]


# ---------------------------------------------------------------------------
# validate_credentials tests
# ---------------------------------------------------------------------------


class TestValidateCredentials:
    """Tests for the validate_credentials function."""

    def _make_manager(self, env_vars: dict[str, str]) -> CredentialManager:
        """Create a manager backed by a mock environment."""

        class DictBackend(CredentialBackend):
            name = "dict"

            def __init__(self, data):
                self._data = data

            def get(self, key):
                return self._data.get(key)

            def available(self):
                return True

        return CredentialManager(backends=[DictBackend(env_vars)])

    def test_all_credentials_present_passes(self):
        env = {
            "S3_ACCESS_KEY": "real_access_key",
            "S3_SECRET_KEY": "real_secret_key",
            "S3_ENDPOINT": "http://minio:9000",
            "DJANGO_SECRET_KEY": "dj4kBm2nPq8rYw5tXz1aGh6cEf9iJl0o",
            "POSTGRES_PASSWORD": "x9kF!mQ2pL@w3nR7",
            "DATABASE_URL": "postgres://ocr:x9kF!mQ2pL@host:5432/db",
            "RABBITMQ_PASSWORD": "rmq_strong_pass_123",
            "CELERY_BROKER_URL": "amqp://ocr_user:rmq_strong_pass_123@host:5672//",
            "REDIS_PASSWORD": "redis_strong_123",
            "METRICS_API_KEY": "met_strong_key_456",
            "OCR_API_KEY": "ocr_strong_key_789",
        }
        with mock.patch.dict(os.environ, {"STORAGE_BACKEND": "s3"}):
            report = validate_credentials(self._make_manager(env), strict=True)
            assert report.passed is True
            assert len(report.errors) == 0

    def test_missing_required_credential_fails(self):
        # Missing DJANGO_SECRET_KEY which is required_when=always
        env = {
            "POSTGRES_PASSWORD": "strong_pass",
            "DATABASE_URL": "postgres://x:y@host:5432/db",
            "RABBITMQ_PASSWORD": "rmq_pass",
            "CELERY_BROKER_URL": "amqp://x:y@host:5672//",
        }
        report = validate_credentials(self._make_manager(env), strict=True)
        assert report.passed is False
        missing = [r for r in report.results if r.name == "DJANGO_SECRET_KEY"]
        assert len(missing) == 1
        assert missing[0].status == "missing"

    def test_placeholder_credential_fails_strict(self):
        env = {
            "DJANGO_SECRET_KEY": "change-me-to-a-random-secret-key",
            "POSTGRES_PASSWORD": "strong_pass",
            "DATABASE_URL": "postgres://x:y@host:5432/db",
            "RABBITMQ_PASSWORD": "rmq_pass",
            "CELERY_BROKER_URL": "amqp://x:y@host:5672//",
        }
        report = validate_credentials(self._make_manager(env), strict=True)
        assert report.passed is False
        placeholder = [r for r in report.results if r.name == "DJANGO_SECRET_KEY"]
        assert placeholder[0].status == "placeholder"

    def test_placeholder_credential_ok_without_strict(self):
        env = {
            "DJANGO_SECRET_KEY": "change-me-to-a-random-secret-key",
            "POSTGRES_PASSWORD": "change-me",
            "DATABASE_URL": "postgres://x:change-me@host:5432/db",
            "RABBITMQ_PASSWORD": "change-me",
            "CELERY_BROKER_URL": "amqp://x:change-me@host:5672//",
        }
        report = validate_credentials(self._make_manager(env), strict=False)
        # All "always" required creds are present and strict=False
        always_required = [
            r for r in report.results if r.status not in ("ok", "not_required")
        ]
        assert len(always_required) == 0

    def test_optional_credentials_not_required(self):
        env = {
            "DJANGO_SECRET_KEY": "real_key",
            "POSTGRES_PASSWORD": "strong_pass",
            "DATABASE_URL": "postgres://x:y@host:5432/db",
            "RABBITMQ_PASSWORD": "rmq_pass",
            "CELERY_BROKER_URL": "amqp://x:y@host:5672//",
            # REDIS_PASSWORD, METRICS_API_KEY, OCR_API_KEY deliberately omitted
        }
        report = validate_credentials(self._make_manager(env), strict=True)
        # S3 keys not required when STORAGE_BACKEND != s3
        assert report.passed is True

    def test_s3_keys_required_when_backend_is_s3(self):
        env = {
            "DJANGO_SECRET_KEY": "real_key",
            "POSTGRES_PASSWORD": "strong_pass",
            "DATABASE_URL": "postgres://x:y@host:5432/db",
            "RABBITMQ_PASSWORD": "rmq_pass",
            "CELERY_BROKER_URL": "amqp://x:y@host:5672//",
            # S3 keys deliberately omitted
        }
        with mock.patch.dict(os.environ, {"STORAGE_BACKEND": "s3"}):
            report = validate_credentials(self._make_manager(env), strict=True)
            assert report.passed is False
            missing_names = {r.name for r in report.errors}
            assert "S3_ACCESS_KEY" in missing_names
            assert "S3_SECRET_KEY" in missing_names
            assert "S3_ENDPOINT" in missing_names

    def test_minioadmin_detected_as_placeholder(self):
        env = {
            "S3_ACCESS_KEY": "minioadmin",
            "S3_SECRET_KEY": "minioadmin",
            "S3_ENDPOINT": "http://minio:9000",
            "DJANGO_SECRET_KEY": "real_key",
            "POSTGRES_PASSWORD": "strong_pass",
            "DATABASE_URL": "postgres://x:y@host:5432/db",
            "RABBITMQ_PASSWORD": "rmq_pass",
            "CELERY_BROKER_URL": "amqp://x:y@host:5672//",
        }
        with mock.patch.dict(os.environ, {"STORAGE_BACKEND": "s3"}):
            report = validate_credentials(self._make_manager(env), strict=True)
            assert report.passed is False
            placeholder_names = {r.name for r in report.errors}
            assert "S3_ACCESS_KEY" in placeholder_names
            assert "S3_SECRET_KEY" in placeholder_names

    def test_report_summary(self):
        env = {
            "DJANGO_SECRET_KEY": "real_key",
            "POSTGRES_PASSWORD": "strong_pass",
            "DATABASE_URL": "postgres://x:y@host:5432/db",
            "RABBITMQ_PASSWORD": "rmq_pass",
            "CELERY_BROKER_URL": "amqp://x:y@host:5672//",
        }
        report = validate_credentials(self._make_manager(env), strict=True)
        summary = report.summary()
        assert "PASS" in summary or "FAIL" in summary

    def test_uses_default_manager_when_none(self):
        with mock.patch.dict(
            os.environ,
            {
                "DJANGO_SECRET_KEY": "real_key",
                "POSTGRES_PASSWORD": "strong_pass",
                "DATABASE_URL": "postgres://x:y@host:5432/db",
                "RABBITMQ_PASSWORD": "rmq_pass",
                "CELERY_BROKER_URL": "amqp://x:y@host:5672//",
            },
        ):
            report = validate_credentials(strict=True)
            assert isinstance(report, ValidationReport)


# ---------------------------------------------------------------------------
# Module-level get_credential tests
# ---------------------------------------------------------------------------


class TestGetCredential:
    """Tests for the module-level get_credential convenience function."""

    def setup_method(self):
        reset_default_manager()

    def teardown_method(self):
        reset_default_manager()

    def test_get_from_env(self):
        with mock.patch.dict(os.environ, {"MY_TEST_CRED": "test_val"}):
            assert get_credential("MY_TEST_CRED") == "test_val"

    def test_get_returns_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert get_credential("NONEXISTENT_999", "my_default") == "my_default"

    def test_get_returns_none_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert get_credential("NONEXISTENT_999") is None

    def test_singleton_reused(self):
        with mock.patch.dict(os.environ, {"KEY1": "val1"}):
            get_credential("KEY1")
            # Calling again should reuse the same manager
            result = get_credential("KEY1")
            assert result == "val1"


# ---------------------------------------------------------------------------
# ValidationReport tests
# ---------------------------------------------------------------------------


class TestValidationReport:
    """Tests for the ValidationReport dataclass."""

    def test_passed_when_all_ok(self):
        report = ValidationReport(
            results=[
                CredentialValidationResult("K1", "ok", "Set"),
                CredentialValidationResult("K2", "not_required", "Optional"),
            ]
        )
        assert report.passed is True

    def test_failed_when_missing(self):
        report = ValidationReport(
            results=[
                CredentialValidationResult("K1", "ok", "Set"),
                CredentialValidationResult("K2", "missing", "Not found"),
            ]
        )
        assert report.passed is False

    def test_failed_when_placeholder(self):
        report = ValidationReport(
            results=[
                CredentialValidationResult("K1", "placeholder", "Placeholder detected"),
            ]
        )
        assert report.passed is False

    def test_errors_property(self):
        report = ValidationReport(
            results=[
                CredentialValidationResult("K1", "ok", ""),
                CredentialValidationResult("K2", "missing", ""),
                CredentialValidationResult("K3", "placeholder", ""),
                CredentialValidationResult("K4", "not_required", ""),
            ]
        )
        errors = report.errors
        assert len(errors) == 2
        assert {e.name for e in errors} == {"K2", "K3"}

    def test_summary_pass(self):
        report = ValidationReport(
            results=[CredentialValidationResult("K1", "ok", "")]
        )
        assert "PASS" in report.summary()

    def test_summary_fail(self):
        report = ValidationReport(
            results=[CredentialValidationResult("K1", "missing", "")]
        )
        assert "FAIL" in report.summary()

    def test_empty_report_passes(self):
        report = ValidationReport()
        assert report.passed is True


# ---------------------------------------------------------------------------
# CREDENTIAL_SCHEMA tests
# ---------------------------------------------------------------------------


class TestCredentialSchema:
    """Verify the credential schema covers expected keys."""

    def test_schema_has_s3_keys(self):
        assert "S3_ACCESS_KEY" in CREDENTIAL_SCHEMA
        assert "S3_SECRET_KEY" in CREDENTIAL_SCHEMA
        assert "S3_ENDPOINT" in CREDENTIAL_SCHEMA

    def test_schema_has_django_key(self):
        assert "DJANGO_SECRET_KEY" in CREDENTIAL_SCHEMA

    def test_schema_has_postgres(self):
        assert "POSTGRES_PASSWORD" in CREDENTIAL_SCHEMA
        assert "DATABASE_URL" in CREDENTIAL_SCHEMA

    def test_schema_has_rabbitmq(self):
        assert "RABBITMQ_PASSWORD" in CREDENTIAL_SCHEMA
        assert "CELERY_BROKER_URL" in CREDENTIAL_SCHEMA

    def test_schema_has_optional_keys(self):
        assert "REDIS_PASSWORD" in CREDENTIAL_SCHEMA
        assert "METRICS_API_KEY" in CREDENTIAL_SCHEMA
        assert "OCR_API_KEY" in CREDENTIAL_SCHEMA

    def test_all_entries_have_description(self):
        for key, entry in CREDENTIAL_SCHEMA.items():
            assert "description" in entry, f"{key} missing description"

    def test_all_entries_have_required_when(self):
        for key, entry in CREDENTIAL_SCHEMA.items():
            assert "required_when" in entry, f"{key} missing required_when"

    def test_s3_keys_conditional(self):
        for key in ("S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_ENDPOINT"):
            assert CREDENTIAL_SCHEMA[key]["required_when"] == "STORAGE_BACKEND=s3"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and integration-style tests."""

    def test_credential_backend_base_class(self):
        backend = CredentialBackend()
        with pytest.raises(NotImplementedError):
            backend.get("KEY")
        assert backend.available() is False

    def test_file_backend_non_dict_json(self, tmp_path):
        """If the JSON file contains a list instead of a dict, handle gracefully."""
        backend = FileBackend()
        cred_file = tmp_path / "list.json"
        cred_file.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        with mock.patch.dict(os.environ, {"CREDENTIAL_FILE_PATH": str(cred_file)}):
            assert backend.get("KEY") is None

    def test_vault_caches_secrets(self):
        """Secrets are loaded once and cached."""
        backend = VaultBackend()
        mock_hvac = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_hvac.Client.return_value = mock_client
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"K": "V"}}
        }
        with mock.patch.dict(
            os.environ, {"VAULT_ADDR": "https://v:8200", "VAULT_TOKEN": "t"}
        ):
            with mock.patch.dict("sys.modules", {"hvac": mock_hvac}):
                backend.get("K")
                backend.get("K")
                # Should only call read_secret_version once
                assert mock_client.secrets.kv.v2.read_secret_version.call_count == 1

    def test_aws_caches_secrets(self):
        """AWS secrets are loaded once and cached."""
        backend = AWSSecretsBackend()
        mock_boto3 = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"K": "V"})
        }
        with mock.patch.dict(os.environ, {"AWS_SECRET_NAME": "test"}):
            with mock.patch.dict("sys.modules", {"boto3": mock_boto3}):
                backend.get("K")
                backend.get("K")
                assert mock_client.get_secret_value.call_count == 1

    def test_manager_get_with_all_backends_returning_none(self):
        """If all backends return None, default is returned."""

        class NoneBackend(CredentialBackend):
            name = "none"

            def get(self, key):
                return None

            def available(self):
                return True

        manager = CredentialManager(backends=[NoneBackend(), NoneBackend()])
        assert manager.get("KEY", "default_val") == "default_val"

    def test_placeholder_detection_not_too_aggressive(self):
        """Ensure normal strong passwords are not flagged."""
        safe_values = [
            "aG5kLm9jcl9sb2NhbC5jcmVkZW50aWFscw==",  # base64
            "sk-proj-1234567890abcdef",
            "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
            "postgres://ocr:aG5kLm9@db:5432/ocr_coordinator",
            "my-strong-p4$$w0rd!",
            "12345678901234567890123456789012",
        ]
        for val in safe_values:
            assert is_placeholder(val) is False, f"False positive for: {val}"


# ---------------------------------------------------------------------------
# Vault KV v1 / v2 path construction tests
# ---------------------------------------------------------------------------


class TestVaultKVVersions:
    """Tests for Vault KV v1 and KV v2 path construction."""

    def test_default_kv_version_is_2(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            env_copy = dict(os.environ)
            env_copy.pop("VAULT_KV_VERSION", None)
            with mock.patch.dict(os.environ, env_copy, clear=True):
                backend = VaultBackend()
                assert backend.vault_kv_version == 2

    def test_kv_version_from_env(self):
        with mock.patch.dict(os.environ, {"VAULT_KV_VERSION": "1"}):
            backend = VaultBackend()
            assert backend.vault_kv_version == 1

    def test_kv_version_from_constructor(self):
        backend = VaultBackend(vault_kv_version=1)
        assert backend.vault_kv_version == 1

    def test_constructor_overrides_env(self):
        with mock.patch.dict(os.environ, {"VAULT_KV_VERSION": "1"}):
            backend = VaultBackend(vault_kv_version=2)
            assert backend.vault_kv_version == 2

    def test_kv_v2_calls_read_secret_version(self):
        backend = VaultBackend(vault_kv_version=2)
        mock_hvac = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_hvac.Client.return_value = mock_client
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"DB_PASS": "v2_value"}}
        }
        with mock.patch.dict(
            os.environ,
            {
                "VAULT_ADDR": "https://vault:8200",
                "VAULT_TOKEN": "tok",
                "VAULT_SECRET_PATH": "secret/data/ocr-local",
            },
        ):
            with mock.patch.dict("sys.modules", {"hvac": mock_hvac}):
                assert backend.get("DB_PASS") == "v2_value"
                mock_client.secrets.kv.v2.read_secret_version.assert_called_once()

    def test_kv_v1_calls_read_secret(self):
        backend = VaultBackend(vault_kv_version=1)
        mock_hvac = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_hvac.Client.return_value = mock_client
        mock_client.secrets.kv.v1.read_secret.return_value = {
            "data": {"DB_PASS": "v1_value"}
        }
        with mock.patch.dict(
            os.environ,
            {
                "VAULT_ADDR": "https://vault:8200",
                "VAULT_TOKEN": "tok",
                "VAULT_SECRET_PATH": "secret/ocr-local",
            },
        ):
            with mock.patch.dict("sys.modules", {"hvac": mock_hvac}):
                assert backend.get("DB_PASS") == "v1_value"
                mock_client.secrets.kv.v1.read_secret.assert_called_once_with(
                    path="ocr-local",
                    mount_point="secret",
                )

    def test_kv_v1_strips_data_segment(self):
        """If user accidentally includes /data/ in path for v1, strip it."""
        backend = VaultBackend(vault_kv_version=1)
        mock_hvac = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_hvac.Client.return_value = mock_client
        mock_client.secrets.kv.v1.read_secret.return_value = {
            "data": {"KEY": "val"}
        }
        with mock.patch.dict(
            os.environ,
            {
                "VAULT_ADDR": "https://vault:8200",
                "VAULT_TOKEN": "tok",
                # User accidentally includes /data/ from a v2 example
                "VAULT_SECRET_PATH": "secret/data/ocr-local",
            },
        ):
            with mock.patch.dict("sys.modules", {"hvac": mock_hvac}):
                assert backend.get("KEY") == "val"
                # Should have called with path "ocr-local", mount "secret"
                mock_client.secrets.kv.v1.read_secret.assert_called_once_with(
                    path="ocr-local",
                    mount_point="secret",
                )

    def test_kv_v2_unwraps_nested_data(self):
        """KV v2 responses have .data.data; KV v1 has .data."""
        backend = VaultBackend(vault_kv_version=2)
        mock_hvac = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_hvac.Client.return_value = mock_client
        # v2 wraps in nested data
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {
                "data": {"NESTED_KEY": "nested_val"},
                "metadata": {"version": 1},
            }
        }
        with mock.patch.dict(
            os.environ,
            {"VAULT_ADDR": "https://vault:8200", "VAULT_TOKEN": "tok"},
        ):
            with mock.patch.dict("sys.modules", {"hvac": mock_hvac}):
                assert backend.get("NESTED_KEY") == "nested_val"

    def test_kv_v2_custom_mount_point(self):
        """Custom mount points (not 'secret') should split correctly."""
        backend = VaultBackend(vault_kv_version=2)
        mock_hvac = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_hvac.Client.return_value = mock_client
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"API_KEY": "custom_val"}}
        }
        with mock.patch.dict(
            os.environ,
            {
                "VAULT_ADDR": "https://vault:8200",
                "VAULT_TOKEN": "tok",
                "VAULT_SECRET_PATH": "custom-mount/data/my-secrets",
            },
        ):
            with mock.patch.dict("sys.modules", {"hvac": mock_hvac}):
                assert backend.get("API_KEY") == "custom_val"
                mock_client.secrets.kv.v2.read_secret_version.assert_called_once_with(
                    path="my-secrets",
                    mount_point="custom-mount",
                    raise_on_deleted_version=True,
                )


# ---------------------------------------------------------------------------
# AWS KMS backend tests
# ---------------------------------------------------------------------------


class TestAWSKMSBackend:
    """Tests for the AWSKMSBackend (mocked boto3)."""

    def test_name(self):
        assert AWSKMSBackend.name == "aws-kms"

    def test_available_without_boto3(self):
        backend = AWSKMSBackend()
        with mock.patch.dict("sys.modules", {"boto3": None}):
            assert backend.available() is False

    def test_available_without_kms_env_vars(self):
        backend = AWSKMSBackend()
        mock_boto3 = mock.MagicMock()
        # Remove all KMS_ENC_ prefixed vars
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("KMS_ENC_")}
        with mock.patch.dict(os.environ, clean_env, clear=True):
            with mock.patch.dict("sys.modules", {"boto3": mock_boto3}):
                assert backend.available() is False

    def test_available_with_kms_env_vars(self):
        backend = AWSKMSBackend()
        mock_boto3 = mock.MagicMock()
        with mock.patch.dict(os.environ, {"KMS_ENC_DB_PASS": "ciphertext"}):
            with mock.patch.dict("sys.modules", {"boto3": mock_boto3}):
                assert backend.available() is True

    def test_decrypt_success(self):
        backend = AWSKMSBackend()
        mock_boto3 = mock.MagicMock()
        mock_botocore_exceptions = mock.MagicMock()
        mock_kms = mock.MagicMock()
        mock_boto3.client.return_value = mock_kms

        plaintext = b"my_secret_password"
        ciphertext_b64 = base64.b64encode(b"encrypted_blob").decode()

        mock_kms.decrypt.return_value = {"Plaintext": plaintext}

        with mock.patch.dict(
            os.environ,
            {"KMS_ENC_DB_PASS": ciphertext_b64, "AWS_KMS_REGION": "us-west-2"},
        ):
            with mock.patch.dict(
                "sys.modules",
                {"boto3": mock_boto3, "botocore": mock.MagicMock(), "botocore.exceptions": mock_botocore_exceptions},
            ):
                # Mock ClientError to be a real-ish exception class
                mock_botocore_exceptions.ClientError = type(
                    "ClientError",
                    (Exception,),
                    {"__init__": lambda self, resp, op: (
                        super(type(self), self).__init__(str(resp)),
                        setattr(self, "response", resp),
                        setattr(self, "operation_name", op),
                    )[-1]},
                )
                assert backend.get("DB_PASS") == "my_secret_password"

    def test_invalid_ciphertext_raises_credential_error(self):
        backend = AWSKMSBackend()
        mock_boto3 = mock.MagicMock()
        mock_kms = mock.MagicMock()
        mock_boto3.client.return_value = mock_kms

        # Create a real exception class that matches botocore.exceptions.ClientError
        class MockClientError(Exception):
            def __init__(self, response, operation_name):
                self.response = response
                self.operation_name = operation_name
                super().__init__(str(response))

        mock_botocore_exc = mock.MagicMock()
        mock_botocore_exc.ClientError = MockClientError

        mock_kms.decrypt.side_effect = MockClientError(
            {"Error": {"Code": "InvalidCiphertextException", "Message": "bad"}},
            "Decrypt",
        )

        ciphertext_b64 = base64.b64encode(b"bad_blob").decode()
        with mock.patch.dict(os.environ, {"KMS_ENC_SOME_KEY": ciphertext_b64}):
            with mock.patch.dict(
                "sys.modules",
                {
                    "boto3": mock_boto3,
                    "botocore": mock.MagicMock(),
                    "botocore.exceptions": mock_botocore_exc,
                },
            ):
                with pytest.raises(CredentialError, match="corrupt or was encrypted"):
                    backend.get("SOME_KEY")

    def test_access_denied_raises_credential_error(self):
        backend = AWSKMSBackend()
        mock_boto3 = mock.MagicMock()
        mock_kms = mock.MagicMock()
        mock_boto3.client.return_value = mock_kms

        class MockClientError(Exception):
            def __init__(self, response, operation_name):
                self.response = response
                self.operation_name = operation_name
                super().__init__(str(response))

        mock_botocore_exc = mock.MagicMock()
        mock_botocore_exc.ClientError = MockClientError

        mock_kms.decrypt.side_effect = MockClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
            "Decrypt",
        )

        ciphertext_b64 = base64.b64encode(b"blob").decode()
        with mock.patch.dict(os.environ, {"KMS_ENC_SECRET": ciphertext_b64}):
            with mock.patch.dict(
                "sys.modules",
                {
                    "boto3": mock_boto3,
                    "botocore": mock.MagicMock(),
                    "botocore.exceptions": mock_botocore_exc,
                },
            ):
                with pytest.raises(CredentialError, match="IAM permissions are missing"):
                    backend.get("SECRET")

    def test_transient_error_retries_with_backoff(self):
        backend = AWSKMSBackend()
        mock_boto3 = mock.MagicMock()
        mock_kms = mock.MagicMock()
        mock_boto3.client.return_value = mock_kms

        class MockClientError(Exception):
            def __init__(self, response, operation_name):
                self.response = response
                self.operation_name = operation_name
                super().__init__(str(response))

        mock_botocore_exc = mock.MagicMock()
        mock_botocore_exc.ClientError = MockClientError

        # Fail twice with transient error, succeed on third
        transient_error = MockClientError(
            {"Error": {"Code": "KMSInternalException", "Message": "transient"}},
            "Decrypt",
        )
        mock_kms.decrypt.side_effect = [
            transient_error,
            transient_error,
            {"Plaintext": b"recovered_value"},
        ]

        ciphertext_b64 = base64.b64encode(b"blob").decode()
        with mock.patch.dict(os.environ, {"KMS_ENC_RETRY_KEY": ciphertext_b64}):
            with mock.patch.dict(
                "sys.modules",
                {
                    "boto3": mock_boto3,
                    "botocore": mock.MagicMock(),
                    "botocore.exceptions": mock_botocore_exc,
                },
            ):
                with mock.patch("credential_manager.time.sleep") as mock_sleep:
                    result = backend.get("RETRY_KEY")
                    assert result == "recovered_value"
                    assert mock_kms.decrypt.call_count == 3
                    # Exponential backoff: 1s, 2s
                    assert mock_sleep.call_count == 2
                    mock_sleep.assert_any_call(1)
                    mock_sleep.assert_any_call(2)

    def test_transient_error_exhausts_retries(self):
        backend = AWSKMSBackend()
        mock_boto3 = mock.MagicMock()
        mock_kms = mock.MagicMock()
        mock_boto3.client.return_value = mock_kms

        class MockClientError(Exception):
            def __init__(self, response, operation_name):
                self.response = response
                self.operation_name = operation_name
                super().__init__(str(response))

        mock_botocore_exc = mock.MagicMock()
        mock_botocore_exc.ClientError = MockClientError

        transient_error = MockClientError(
            {"Error": {"Code": "KMSInternalException", "Message": "transient"}},
            "Decrypt",
        )
        mock_kms.decrypt.side_effect = [transient_error] * 3

        ciphertext_b64 = base64.b64encode(b"blob").decode()
        with mock.patch.dict(os.environ, {"KMS_ENC_FAIL_KEY": ciphertext_b64}):
            with mock.patch.dict(
                "sys.modules",
                {
                    "boto3": mock_boto3,
                    "botocore": mock.MagicMock(),
                    "botocore.exceptions": mock_botocore_exc,
                },
            ):
                with mock.patch("credential_manager.time.sleep"):
                    result = backend.get("FAIL_KEY")
                    # Should return None after exhausting retries
                    assert result is None
                    assert mock_kms.decrypt.call_count == 3


# ---------------------------------------------------------------------------
# Credential rotation callback tests
# ---------------------------------------------------------------------------


class TestCredentialRotation:
    """Tests for credential refresh/rotation hooks."""

    def test_no_refresh_thread_by_default(self):
        manager = CredentialManager(backends=[EnvVarBackend()])
        assert manager._refresh_thread is None

    def test_refresh_thread_starts_when_interval_set(self):
        manager = CredentialManager(
            backends=[EnvVarBackend()],
            refresh_interval_seconds=60,
        )
        assert manager._refresh_thread is not None
        assert manager._refresh_thread.daemon is True
        manager.stop_refresh()

    def test_stop_refresh_stops_thread(self):
        manager = CredentialManager(
            backends=[EnvVarBackend()],
            refresh_interval_seconds=60,
        )
        assert manager._refresh_thread is not None
        manager.stop_refresh()
        assert manager._refresh_thread is None

    def test_callback_invoked_on_value_change(self):
        """When a credential value changes, the callback is called."""
        refreshed_keys: list[str] = []

        class MutableBackend(CredentialBackend):
            name = "mutable"

            def __init__(self):
                self.data: dict[str, str] = {"KEY": "original"}

            def get(self, key):
                return self.data.get(key)

            def available(self):
                return True

        backend = MutableBackend()
        manager = CredentialManager(
            backends=[backend],
            on_credential_refreshed=lambda name: refreshed_keys.append(name),
        )

        # Access the credential to populate the cache
        assert manager.get("KEY") == "original"

        # Simulate a rotation
        backend.data["KEY"] = "rotated"

        # Manually trigger the refresh check
        manager._check_for_rotated_credentials()

        assert refreshed_keys == ["KEY"]
        assert manager._credential_cache["KEY"] == "rotated"

    def test_callback_not_invoked_when_value_unchanged(self):
        """When a credential stays the same, callback is not called."""
        refreshed_keys: list[str] = []

        class StaticBackend(CredentialBackend):
            name = "static"

            def get(self, key):
                return "same_value" if key == "KEY" else None

            def available(self):
                return True

        manager = CredentialManager(
            backends=[StaticBackend()],
            on_credential_refreshed=lambda name: refreshed_keys.append(name),
        )

        manager.get("KEY")
        manager._check_for_rotated_credentials()

        assert refreshed_keys == []

    def test_callback_exception_does_not_crash(self):
        """If the callback raises, the refresh loop continues."""

        class MutableBackend(CredentialBackend):
            name = "mutable"

            def __init__(self):
                self.data: dict[str, str] = {"K1": "v1", "K2": "v2"}

            def get(self, key):
                return self.data.get(key)

            def available(self):
                return True

        backend = MutableBackend()
        call_log: list[str] = []

        def bad_callback(name: str) -> None:
            call_log.append(name)
            if name == "K1":
                raise RuntimeError("callback exploded")

        manager = CredentialManager(
            backends=[backend],
            on_credential_refreshed=bad_callback,
        )

        # Populate cache
        manager.get("K1")
        manager.get("K2")

        # Rotate both
        backend.data["K1"] = "new_v1"
        backend.data["K2"] = "new_v2"

        # Should not raise
        manager._check_for_rotated_credentials()

        # Both should have been attempted despite K1 callback failing
        assert "K1" in call_log
        assert "K2" in call_log

    def test_refresh_only_checks_previously_accessed_keys(self):
        """Rotation check only re-fetches keys that have been get()-ed."""

        class TrackingBackend(CredentialBackend):
            name = "tracking"

            def __init__(self):
                self.calls: list[str] = []

            def get(self, key):
                self.calls.append(key)
                return "val"

            def available(self):
                return True

        backend = TrackingBackend()
        manager = CredentialManager(
            backends=[backend],
        )

        # Only access KEY_A, not KEY_B
        manager.get("KEY_A")
        backend.calls.clear()

        manager._check_for_rotated_credentials()

        # Should only re-fetch KEY_A
        assert backend.calls == ["KEY_A"]


# ---------------------------------------------------------------------------
# CredentialError tests
# ---------------------------------------------------------------------------


class TestCredentialError:
    """Tests for the CredentialError exception."""

    def test_is_exception(self):
        assert issubclass(CredentialError, Exception)

    def test_message(self):
        err = CredentialError("test message")
        assert str(err) == "test message"

    def test_kms_raises_credential_error_through_manager(self):
        """CredentialError from KMS backend propagates through CredentialManager."""

        class KMSErrorBackend(CredentialBackend):
            name = "kms-error"

            def get(self, key):
                raise CredentialError("KMS key not found")

            def available(self):
                return True

        manager = CredentialManager(backends=[KMSErrorBackend()])
        with pytest.raises(CredentialError, match="KMS key not found"):
            manager.get("ANY_KEY")
