"""Tests for the credential bridge and production wiring.

Verifies that:
- coordinator.credential_bridge imports and re-exports correctly
- The fallback path works when credential_manager is not importable
- get_credential reads from env vars via the bridge
- coordinator settings uses _get_credential for sensitive values
- api/cloud_storage.py uses _get_credential for S3/Azure keys
- scripts/migrate_nfs_to_s3.py uses _get_credential for S3 keys
- coordinator/jobs/apps.py startup validation logic works
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest import mock

# ---------------------------------------------------------------------------
# credential_bridge import tests
# ---------------------------------------------------------------------------


class TestCredentialBridgeImport:
    """Verify the bridge module imports and re-exports correctly."""

    def test_bridge_imports_successfully(self):
        """The bridge module should import without errors."""
        # credential_manager.py is at the repo root, so it should be importable
        # when the project root is on sys.path (which tests/ always has).
        from coordinator.credential_bridge import get_credential

        assert callable(get_credential)

    def test_bridge_exports_validate_credentials(self):
        from coordinator.credential_bridge import validate_credentials

        assert callable(validate_credentials)

    def test_bridge_exports_credential_manager_class(self):
        from coordinator.credential_bridge import CredentialManager

        # Should be the real class, not None, because credential_manager is available
        assert CredentialManager is not None

    def test_bridge_get_credential_reads_env(self):
        from coordinator.credential_bridge import get_credential

        with mock.patch.dict(os.environ, {"BRIDGE_TEST_KEY": "bridge_val"}):
            assert get_credential("BRIDGE_TEST_KEY") == "bridge_val"

    def test_bridge_get_credential_returns_default(self):
        from coordinator.credential_bridge import get_credential

        with mock.patch.dict(os.environ, {}, clear=True):
            assert get_credential("NONEXISTENT_BRIDGE_KEY", "fallback") == "fallback"

    def test_bridge_get_credential_returns_none_by_default(self):
        from coordinator.credential_bridge import get_credential

        with mock.patch.dict(os.environ, {}, clear=True):
            assert get_credential("NONEXISTENT_BRIDGE_KEY") is None


# ---------------------------------------------------------------------------
# Fallback behaviour when credential_manager is absent
# ---------------------------------------------------------------------------


class TestCredentialBridgeFallback:
    """Verify that the bridge falls back to os.environ when credential_manager
    cannot be imported."""

    @staticmethod
    def _swap_and_reload():
        """Block credential_manager and re-import the bridge module.

        Returns ``(bridge_module, restore_func)`` where *restore_func*
        should be called in a ``finally`` block.
        """
        saved_cm = sys.modules.get("credential_manager", _SENTINEL)
        sys.modules.pop("credential_manager", None)
        sys.modules["credential_manager"] = None  # block import
        sys.modules.pop("coordinator.credential_bridge", None)

        bridge = importlib.import_module("coordinator.credential_bridge")

        def _restore():
            sys.modules.pop("coordinator.credential_bridge", None)
            if saved_cm is _SENTINEL:
                sys.modules.pop("credential_manager", None)
            else:
                sys.modules["credential_manager"] = saved_cm

        return bridge, _restore

    def test_fallback_get_credential_reads_env(self):
        """When credential_manager is missing, get_credential should still
        read from os.environ."""
        bridge, restore = self._swap_and_reload()
        try:
            with mock.patch.dict(os.environ, {"FALLBACK_KEY": "fallback_value"}):
                assert bridge.get_credential("FALLBACK_KEY") == "fallback_value"
            # CredentialManager should be None in fallback mode
            assert bridge.CredentialManager is None
        finally:
            restore()

    def test_fallback_validate_credentials_returns_passing(self):
        """Fallback validate_credentials should return a report that passes."""
        bridge, restore = self._swap_and_reload()
        try:
            report = bridge.validate_credentials()
            assert report.passed is True
            assert report.errors == []
        finally:
            restore()


# Sentinel for distinguishing "key absent" from "key is None" in sys.modules
_SENTINEL = object()


# ---------------------------------------------------------------------------
# coordinator/coordinator/settings.py integration
# ---------------------------------------------------------------------------


class TestSettingsCredentialWiring:
    """Verify that settings.py uses _get_credential for sensitive values."""

    def test_settings_imports_get_credential(self):
        """The settings module must define _get_credential."""
        # We can check the source file for the import line without loading
        # Django settings (which requires a full Django env).
        import pathlib

        settings_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "coordinator"
            / "coordinator"
            / "settings.py"
        )
        source = settings_path.read_text(encoding="utf-8")

        # Verify the bridge import is present
        assert "from coordinator.credential_bridge import get_credential" in source

    def test_settings_uses_get_credential_for_s3(self):
        """S3_ACCESS_KEY and S3_SECRET_KEY must use _get_credential."""
        import pathlib

        settings_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "coordinator"
            / "coordinator"
            / "settings.py"
        )
        source = settings_path.read_text(encoding="utf-8")

        assert "_get_credential('S3_ACCESS_KEY'" in source
        assert "_get_credential('S3_SECRET_KEY'" in source

    def test_settings_uses_get_credential_for_django_secret(self):
        """DJANGO_SECRET_KEY must use _get_credential."""
        import pathlib

        settings_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "coordinator"
            / "coordinator"
            / "settings.py"
        )
        source = settings_path.read_text(encoding="utf-8")

        assert "_get_credential('DJANGO_SECRET_KEY'" in source


# ---------------------------------------------------------------------------
# api/cloud_storage.py integration
# ---------------------------------------------------------------------------


class TestCloudStorageCredentialWiring:
    """Verify that cloud_storage.py uses _get_credential for secrets."""

    def test_cloud_storage_has_credential_import(self):
        """cloud_storage.py must import get_credential with fallback."""
        import pathlib

        cs_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "api"
            / "cloud_storage.py"
        )
        source = cs_path.read_text(encoding="utf-8")

        assert "from credential_manager import get_credential" in source

    def test_cloud_storage_uses_get_credential_for_s3(self):
        import pathlib

        cs_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "api"
            / "cloud_storage.py"
        )
        source = cs_path.read_text(encoding="utf-8")

        assert '_get_credential("S3_ACCESS_KEY"' in source
        assert '_get_credential("S3_SECRET_KEY"' in source

    def test_cloud_storage_uses_get_credential_for_azure(self):
        import pathlib

        cs_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "api"
            / "cloud_storage.py"
        )
        source = cs_path.read_text(encoding="utf-8")

        assert '_get_credential("AZURE_STORAGE_CONNECTION_STRING"' in source
        assert '_get_credential("AZURE_STORAGE_KEY"' in source

    def test_cloud_storage_from_env_s3_uses_credential_manager(self):
        """StorageConfig.from_env() for S3 provider must go through
        _get_credential for access_key and secret_key."""
        from api.cloud_storage import StorageConfig

        with mock.patch.dict(
            os.environ,
            {
                "STORAGE_PROVIDER": "s3",
                "S3_ENDPOINT_URL": "http://minio:9000",
                "S3_ACCESS_KEY": "env_access_key",
                "S3_SECRET_KEY": "env_secret_key",
                "S3_BUCKET": "test-bucket",
                "S3_REGION": "us-east-1",
                "S3_PREFIX": "",
            },
        ):
            config = StorageConfig.from_env("s3")
            assert config.access_key == "env_access_key"
            assert config.secret_key == "env_secret_key"


# ---------------------------------------------------------------------------
# scripts/migrate_nfs_to_s3.py integration
# ---------------------------------------------------------------------------


class TestMigrateScriptCredentialWiring:
    """Verify that migrate_nfs_to_s3.py uses _get_credential."""

    def test_migrate_script_has_credential_import(self):
        import pathlib

        script_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "scripts"
            / "migrate_nfs_to_s3.py"
        )
        source = script_path.read_text(encoding="utf-8")

        assert "from credential_manager import get_credential" in source

    def test_migrate_script_uses_get_credential_for_s3(self):
        import pathlib

        script_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "scripts"
            / "migrate_nfs_to_s3.py"
        )
        source = script_path.read_text(encoding="utf-8")

        assert '_get_credential("S3_ACCESS_KEY")' in source
        assert '_get_credential("S3_SECRET_KEY")' in source


# ---------------------------------------------------------------------------
# Helm chart secret.yaml integration
# ---------------------------------------------------------------------------


class TestHelmSecretWiring:
    """Verify that the Helm secret template requires S3 secrets when
    storage.backend is s3."""

    def test_helm_secret_requires_s3_access_key(self):
        import pathlib

        tpl_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "helm"
            / "ocr-local"
            / "templates"
            / "secret.yaml"
        )
        source = tpl_path.read_text(encoding="utf-8")

        assert 'required "secrets.s3AccessKey is required when storage.backend=s3"' in source

    def test_helm_secret_requires_s3_secret_key(self):
        import pathlib

        tpl_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "helm"
            / "ocr-local"
            / "templates"
            / "secret.yaml"
        )
        source = tpl_path.read_text(encoding="utf-8")

        assert 'required "secrets.s3SecretKey is required when storage.backend=s3"' in source

    def test_helm_secret_requires_s3_endpoint(self):
        import pathlib

        tpl_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "helm"
            / "ocr-local"
            / "templates"
            / "secret.yaml"
        )
        source = tpl_path.read_text(encoding="utf-8")

        assert 'required "secrets.s3Endpoint is required when storage.backend=s3"' in source

    def test_helm_secret_requires_s3_bucket(self):
        import pathlib

        tpl_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "helm"
            / "ocr-local"
            / "templates"
            / "secret.yaml"
        )
        source = tpl_path.read_text(encoding="utf-8")

        assert 'required "secrets.s3Bucket is required when storage.backend=s3"' in source


# ---------------------------------------------------------------------------
# coordinator/jobs/apps.py startup validation
# ---------------------------------------------------------------------------


class TestStartupValidation:
    """Verify that JobsConfig._validate_credentials works correctly.

    We test ``ready()`` indirectly by checking the source for the
    DEPLOYMENT_ENV guard, and test ``_validate_credentials`` directly
    since it is a static method that does not require Django app setup.
    """

    @staticmethod
    def _load_jobs_apps_module():
        """Load coordinator/jobs/apps.py with a stub django.apps.AppConfig."""
        apps_path = (
            Path(__file__).resolve().parent.parent
            / "coordinator"
            / "jobs"
            / "apps.py"
        )
        module_name = "_test_jobs_apps"
        sentinel = object()
        saved_modules = {
            name: sys.modules.get(name, sentinel)
            for name in ("django", "django.apps", module_name)
        }
        django_module = types.ModuleType("django")
        django_apps_module = types.ModuleType("django.apps")
        django_module.apps = django_apps_module
        django_apps_module.AppConfig = type("AppConfig", (), {})

        spec = importlib.util.spec_from_file_location(module_name, apps_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)

        sys.modules["django"] = django_module
        sys.modules["django.apps"] = django_apps_module
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        def _restore():
            for name, original in saved_modules.items():
                if original is sentinel:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

        return module, _restore

    def test_ready_calls_validate_in_production_source(self):
        """The ready() method must check DEPLOYMENT_ENV=production and call
        _validate_credentials.  We verify via source inspection rather than
        constructing a full Django AppConfig (which needs a running app registry)."""
        import pathlib

        apps_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "coordinator"
            / "jobs"
            / "apps.py"
        )
        source = apps_path.read_text(encoding="utf-8")

        assert "'DEPLOYMENT_ENV'" in source
        assert "'production'" in source
        assert "_validate_credentials" in source

    def test_validate_credentials_logs_warnings_on_failure(self):
        """When validation finds issues, they should be logged as warnings."""
        mock_result = mock.MagicMock()
        mock_result.status = "missing"
        mock_result.message = "Required credential 'S3_ACCESS_KEY' not found"

        mock_report = mock.MagicMock()
        mock_report.passed = False
        mock_report.errors = [mock_result]

        module, restore = self._load_jobs_apps_module()
        try:
            with mock.patch(
                "coordinator.credential_bridge.validate_credentials",
                return_value=mock_report,
            ):
                with mock.patch.object(module, "logger") as mock_logger:
                    module.JobsConfig._validate_credentials()
                    mock_logger.warning.assert_called()
        finally:
            restore()

    def test_validate_credentials_logs_success(self):
        """When validation passes, it should log success info."""
        mock_report = mock.MagicMock()
        mock_report.passed = True
        mock_report.summary.return_value = "PASS: 5/5 credentials validated"

        module, restore = self._load_jobs_apps_module()
        try:
            with mock.patch(
                "coordinator.credential_bridge.validate_credentials",
                return_value=mock_report,
            ):
                with mock.patch.object(module, "logger") as mock_logger:
                    module.JobsConfig._validate_credentials()
                    mock_logger.info.assert_called()
        finally:
            restore()

    def test_validate_credentials_handles_import_error(self):
        """If credential_bridge is unavailable, validation should not raise."""
        module, restore = self._load_jobs_apps_module()
        try:
            with mock.patch(
                "coordinator.credential_bridge.validate_credentials",
                side_effect=ImportError("No module"),
            ):
                module.JobsConfig._validate_credentials()
        finally:
            restore()

    def test_validate_credentials_handles_unexpected_error(self):
        """If validation raises an unexpected error, startup must not block."""
        module, restore = self._load_jobs_apps_module()
        try:
            with mock.patch(
                "coordinator.credential_bridge.validate_credentials",
                side_effect=RuntimeError("Unexpected"),
            ):
                module.JobsConfig._validate_credentials()
        finally:
            restore()
