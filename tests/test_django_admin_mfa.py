"""Tests for C-07: MFA for Django admin using django-otp.

Validates that:
- ADMIN_MFA_REQUIRED defaults to true when DEPLOYMENT_ENV=production
- ADMIN_MFA_REQUIRED defaults to false when DEPLOYMENT_ENV is not production
- Explicit ADMIN_MFA_REQUIRED env var overrides the default
- MFA middleware applies to /admin/ paths
- MFA middleware skips non-admin paths
- MFA middleware skips when MFA is not required
- MFA middleware skips for unauthenticated users
- MFA middleware passes through login/logout/otp pages
- MFA middleware marks session after OTP verification
- django-otp is listed in coordinator requirements
- Settings include django_otp in INSTALLED_APPS
- OTPMiddleware is in MIDDLEWARE
- MFAMiddleware is in MIDDLEWARE after OTPMiddleware
- TOTPDevice is registered in admin

Django models and middleware are tested via direct import of the mfa module
with mocked Django primitives (same pattern as test_audit_deletion_events.py).
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
COORDINATOR_DIR = ROOT_DIR / "coordinator"


def _read_coordinator_requirements() -> str:
    return (COORDINATOR_DIR / "requirements.txt").read_text()


def _read_settings_source() -> str:
    return (COORDINATOR_DIR / "coordinator" / "settings.py").read_text()


def _read_admin_source() -> str:
    return (COORDINATOR_DIR / "jobs" / "admin.py").read_text()


# ---------------------------------------------------------------------------
# Dependency / wiring tests (no Django needed)
# ---------------------------------------------------------------------------


class TestDependencies:
    """Verify django-otp and qrcode are in coordinator requirements."""

    def test_django_otp_in_requirements(self):
        reqs = _read_coordinator_requirements()
        assert "django-otp" in reqs

    def test_qrcode_in_requirements(self):
        reqs = _read_coordinator_requirements()
        assert "qrcode" in reqs


class TestSettingsWiring:
    """Verify settings.py includes django_otp apps and middleware."""

    def test_django_otp_in_installed_apps(self):
        src = _read_settings_source()
        assert "'django_otp'" in src or '"django_otp"' in src

    def test_otp_totp_in_installed_apps(self):
        src = _read_settings_source()
        assert "django_otp.plugins.otp_totp" in src

    def test_otp_middleware_in_middleware(self):
        src = _read_settings_source()
        assert "django_otp.middleware.OTPMiddleware" in src

    def test_mfa_middleware_in_middleware(self):
        src = _read_settings_source()
        assert "jobs.mfa.MFAMiddleware" in src

    def test_otp_middleware_after_auth(self):
        """OTPMiddleware must appear after AuthenticationMiddleware."""
        src = _read_settings_source()
        auth_pos = src.index("AuthenticationMiddleware")
        otp_pos = src.index("OTPMiddleware")
        assert otp_pos > auth_pos

    def test_mfa_middleware_after_otp(self):
        """MFAMiddleware must appear after OTPMiddleware."""
        src = _read_settings_source()
        otp_pos = src.index("django_otp.middleware.OTPMiddleware")
        mfa_pos = src.index("jobs.mfa.MFAMiddleware")
        assert mfa_pos > otp_pos


class TestAdminWiring:
    """Verify admin.py imports and registers TOTPDevice."""

    def test_totp_device_import(self):
        src = _read_admin_source()
        assert "TOTPDevice" in src

    def test_totp_device_admin_import(self):
        src = _read_admin_source()
        assert "TOTPDeviceAdmin" in src

    def test_totp_device_registration(self):
        src = _read_admin_source()
        assert "admin.site.register(TOTPDevice" in src


# ---------------------------------------------------------------------------
# is_mfa_required() logic tests
# ---------------------------------------------------------------------------


class TestIsMfaRequired:
    """Test the is_mfa_required() function from jobs.mfa."""

    def _import_is_mfa_required(self):
        """Import the function, injecting a minimal Django mock if needed."""
        # Ensure coordinator package is importable
        if str(COORDINATOR_DIR) not in sys.path:
            sys.path.insert(0, str(COORDINATOR_DIR))

        # We need minimal django mocks so that `from django.http import ...`
        # works at import time.
        django_mocks = {}
        for mod_name in [
            "django",
            "django.http",
            "django.urls",
        ]:
            if mod_name not in sys.modules:
                m = types.ModuleType(mod_name)
                sys.modules[mod_name] = m
                django_mocks[mod_name] = m

        # Provide stub classes/functions that the module references at import
        sys.modules["django.http"].HttpRequest = type("HttpRequest", (), {})
        sys.modules["django.http"].HttpResponse = type("HttpResponse", (), {})
        sys.modules["django.http"].HttpResponseRedirect = type(
            "HttpResponseRedirect", (), {"__init__": lambda self, url: None}
        )
        sys.modules["django.urls"].reverse = lambda name, **kw: "/admin/login/"

        try:
            # Force re-import to pick up env var changes
            if "jobs.mfa" in sys.modules:
                del sys.modules["jobs.mfa"]

            from jobs.mfa import is_mfa_required

            return is_mfa_required
        finally:
            # Clean up injected mocks
            for mod_name in django_mocks:
                if mod_name in sys.modules and sys.modules[mod_name] is django_mocks[mod_name]:
                    del sys.modules[mod_name]

    def test_default_true_when_production(self):
        with mock.patch.dict(
            os.environ,
            {"DEPLOYMENT_ENV": "production"},
            clear=False,
        ):
            # Remove explicit override if present
            os.environ.pop("ADMIN_MFA_REQUIRED", None)
            fn = self._import_is_mfa_required()
            assert fn() is True

    def test_default_false_when_development(self):
        with mock.patch.dict(
            os.environ,
            {"DEPLOYMENT_ENV": "development"},
            clear=False,
        ):
            os.environ.pop("ADMIN_MFA_REQUIRED", None)
            fn = self._import_is_mfa_required()
            assert fn() is False

    def test_default_false_when_staging(self):
        with mock.patch.dict(
            os.environ,
            {"DEPLOYMENT_ENV": "staging"},
            clear=False,
        ):
            os.environ.pop("ADMIN_MFA_REQUIRED", None)
            fn = self._import_is_mfa_required()
            assert fn() is False

    def test_default_false_when_env_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            fn = self._import_is_mfa_required()
            assert fn() is False

    def test_explicit_true_overrides_development(self):
        with mock.patch.dict(
            os.environ,
            {"DEPLOYMENT_ENV": "development", "ADMIN_MFA_REQUIRED": "true"},
            clear=False,
        ):
            fn = self._import_is_mfa_required()
            assert fn() is True

    def test_explicit_false_overrides_production(self):
        with mock.patch.dict(
            os.environ,
            {"DEPLOYMENT_ENV": "production", "ADMIN_MFA_REQUIRED": "false"},
            clear=False,
        ):
            fn = self._import_is_mfa_required()
            assert fn() is False

    def test_accepts_1_as_true(self):
        with mock.patch.dict(
            os.environ,
            {"ADMIN_MFA_REQUIRED": "1"},
            clear=False,
        ):
            fn = self._import_is_mfa_required()
            assert fn() is True

    def test_accepts_yes_as_true(self):
        with mock.patch.dict(
            os.environ,
            {"ADMIN_MFA_REQUIRED": "yes"},
            clear=False,
        ):
            fn = self._import_is_mfa_required()
            assert fn() is True


# ---------------------------------------------------------------------------
# MFAMiddleware behaviour tests
# ---------------------------------------------------------------------------


class TestMFAMiddleware:
    """Test the MFAMiddleware class."""

    @pytest.fixture(autouse=True)
    def _setup_django_mocks(self):
        """Inject minimal Django mocks so the module can be imported."""
        if str(COORDINATOR_DIR) not in sys.path:
            sys.path.insert(0, str(COORDINATOR_DIR))

        self._django_mocks = {}
        for mod_name in [
            "django",
            "django.http",
            "django.urls",
        ]:
            if mod_name not in sys.modules:
                m = types.ModuleType(mod_name)
                sys.modules[mod_name] = m
                self._django_mocks[mod_name] = m

        # Stubs
        sys.modules["django.http"].HttpRequest = type("HttpRequest", (), {})
        sys.modules["django.http"].HttpResponse = type("HttpResponse", (), {})

        class FakeRedirect:
            def __init__(self, url):
                self.url = url

        sys.modules["django.http"].HttpResponseRedirect = FakeRedirect
        sys.modules["django.urls"].reverse = lambda name, **kw: "/admin/login/"

        # Force re-import
        if "jobs.mfa" in sys.modules:
            del sys.modules["jobs.mfa"]

        from jobs.mfa import MFAMiddleware

        self.MFAMiddleware = MFAMiddleware
        self.FakeRedirect = FakeRedirect

        yield

        # Cleanup
        for mod_name in list(self._django_mocks):
            if mod_name in sys.modules and sys.modules[mod_name] is self._django_mocks[mod_name]:
                del sys.modules[mod_name]
        if "jobs.mfa" in sys.modules:
            del sys.modules["jobs.mfa"]

    def _make_request(self, path, authenticated=True, otp_device=None, session=None):
        """Create a mock request object."""
        request = MagicMock()
        request.path = path
        request.session = session if session is not None else {}
        if authenticated:
            request.user = MagicMock()
            request.user.is_authenticated = True
            request.user.otp_device = otp_device
        else:
            request.user = MagicMock()
            request.user.is_authenticated = False
        return request

    def test_skips_non_admin_paths(self):
        """Non-admin paths should pass through without MFA checks."""
        sentinel = object()
        get_response = MagicMock(return_value=sentinel)
        mw = self.MFAMiddleware(get_response)

        with mock.patch.dict(os.environ, {"ADMIN_MFA_REQUIRED": "true"}):
            request = self._make_request("/api/v1/metrics/")
            result = mw(request)

        assert result is sentinel
        get_response.assert_called_once_with(request)

    def test_skips_when_mfa_not_required(self):
        """Admin paths should pass through when MFA is disabled."""
        sentinel = object()
        get_response = MagicMock(return_value=sentinel)
        mw = self.MFAMiddleware(get_response)

        with mock.patch.dict(os.environ, {"ADMIN_MFA_REQUIRED": "false"}):
            request = self._make_request("/admin/jobs/job/")
            result = mw(request)

        assert result is sentinel
        get_response.assert_called_once_with(request)

    def test_skips_unauthenticated_users(self):
        """Unauthenticated users should pass through to the login page."""
        sentinel = object()
        get_response = MagicMock(return_value=sentinel)
        mw = self.MFAMiddleware(get_response)

        with mock.patch.dict(os.environ, {"ADMIN_MFA_REQUIRED": "true"}):
            request = self._make_request("/admin/jobs/job/", authenticated=False)
            result = mw(request)

        assert result is sentinel

    def test_skips_when_session_verified(self):
        """Users with session verification flag should pass through."""
        sentinel = object()
        get_response = MagicMock(return_value=sentinel)
        mw = self.MFAMiddleware(get_response)

        with mock.patch.dict(os.environ, {"ADMIN_MFA_REQUIRED": "true"}):
            session = {"_otp_verified": True}
            request = self._make_request("/admin/jobs/job/", session=session)
            result = mw(request)

        assert result is sentinel

    def test_passes_through_with_otp_device(self):
        """Users verified by OTPMiddleware (otp_device set) should pass."""
        sentinel = object()
        get_response = MagicMock(return_value=sentinel)
        mw = self.MFAMiddleware(get_response)

        with mock.patch.dict(os.environ, {"ADMIN_MFA_REQUIRED": "true"}):
            otp_device = MagicMock()
            session = {}
            request = self._make_request(
                "/admin/jobs/job/",
                otp_device=otp_device,
                session=session,
            )
            result = mw(request)

        assert result is sentinel
        assert session.get("_otp_verified") is True

    def test_allows_login_page(self):
        """The login page should not be redirected (avoids loops)."""
        sentinel = object()
        get_response = MagicMock(return_value=sentinel)
        mw = self.MFAMiddleware(get_response)

        with mock.patch.dict(os.environ, {"ADMIN_MFA_REQUIRED": "true"}):
            request = self._make_request("/admin/login/")
            result = mw(request)

        assert result is sentinel

    def test_allows_logout_page(self):
        sentinel = object()
        get_response = MagicMock(return_value=sentinel)
        mw = self.MFAMiddleware(get_response)

        with mock.patch.dict(os.environ, {"ADMIN_MFA_REQUIRED": "true"}):
            request = self._make_request("/admin/logout/")
            result = mw(request)

        assert result is sentinel

    def test_allows_otp_pages(self):
        sentinel = object()
        get_response = MagicMock(return_value=sentinel)
        mw = self.MFAMiddleware(get_response)

        with mock.patch.dict(os.environ, {"ADMIN_MFA_REQUIRED": "true"}):
            request = self._make_request("/admin/otp/totp/totpdevice/add/")
            result = mw(request)

        assert result is sentinel

    def test_redirects_unverified_authenticated_user(self):
        """An authenticated user without OTP verification should be redirected."""
        get_response = MagicMock(return_value=object())
        mw = self.MFAMiddleware(get_response)

        with mock.patch.dict(os.environ, {"ADMIN_MFA_REQUIRED": "true"}):
            request = self._make_request("/admin/jobs/job/")
            result = mw(request)

        # Should be a redirect, not the sentinel
        assert isinstance(result, self.FakeRedirect)
        assert "/admin/login/" in result.url
        get_response.assert_not_called()

    def test_redirect_includes_next_param(self):
        """The redirect URL should include the original path as ?next=."""
        get_response = MagicMock(return_value=object())
        mw = self.MFAMiddleware(get_response)

        with mock.patch.dict(os.environ, {"ADMIN_MFA_REQUIRED": "true"}):
            request = self._make_request("/admin/jobs/job/123/change/")
            result = mw(request)

        assert isinstance(result, self.FakeRedirect)
        assert "next=/admin/jobs/job/123/change/" in result.url


# ---------------------------------------------------------------------------
# Module file existence test
# ---------------------------------------------------------------------------


class TestMFAModuleExists:
    """Verify the MFA module file exists at the expected path."""

    def test_mfa_module_file_exists(self):
        mfa_path = COORDINATOR_DIR / "jobs" / "mfa.py"
        assert mfa_path.exists(), f"Expected {mfa_path} to exist"

    def test_mfa_module_has_is_mfa_required(self):
        src = (COORDINATOR_DIR / "jobs" / "mfa.py").read_text()
        assert "def is_mfa_required" in src

    def test_mfa_module_has_middleware_class(self):
        src = (COORDINATOR_DIR / "jobs" / "mfa.py").read_text()
        assert "class MFAMiddleware" in src

    def test_mfa_module_has_session_key(self):
        src = (COORDINATOR_DIR / "jobs" / "mfa.py").read_text()
        assert "MFA_VERIFIED_SESSION_KEY" in src
