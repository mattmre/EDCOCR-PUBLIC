"""MFA (Multi-Factor Authentication) configuration for Django admin.

Provides TOTP-based MFA enforcement for the Django admin interface using
django-otp.  Controlled by the ``ADMIN_MFA_REQUIRED`` environment variable:

* **production** (``DEPLOYMENT_ENV=production``): MFA is required by default.
* **development / staging**: MFA is *not* required by default.

Set ``ADMIN_MFA_REQUIRED=true`` or ``ADMIN_MFA_REQUIRED=false`` to override.
"""

from __future__ import annotations

import logging
import os

from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.urls import reverse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

_TRUE_VALUES = {"1", "true", "yes"}


def is_mfa_required() -> bool:
    """Return True when admin users must complete TOTP verification.

    * ``ADMIN_MFA_REQUIRED`` env var takes precedence if set.
    * Otherwise defaults to ``True`` when ``DEPLOYMENT_ENV=production``,
      ``False`` for any other environment value.
    """
    explicit = os.environ.get("ADMIN_MFA_REQUIRED", "").strip().lower()
    if explicit:
        return explicit in _TRUE_VALUES

    deployment_env = os.environ.get("DEPLOYMENT_ENV", "development").strip().lower()
    return deployment_env == "production"


# ---------------------------------------------------------------------------
# Session key used to track OTP verification state
# ---------------------------------------------------------------------------

MFA_VERIFIED_SESSION_KEY = "_otp_verified"


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class MFAMiddleware:
    """Enforce TOTP verification for admin paths when MFA is required.

    Behaviour:
    * Only intercepts requests whose ``path`` starts with ``/admin/``.
    * Passes through immediately when ``is_mfa_required()`` returns False.
    * Passes through if the user already has a verified session flag.
    * Passes through for unauthenticated users (let Django handle login).
    * Passes through for the OTP verify/setup page itself (to avoid loops).
    * Redirects all other admin requests to the django-otp verify page.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        if not request.path.startswith("/admin/"):
            return self.get_response(request)

        if not is_mfa_required():
            return self.get_response(request)

        # Let unauthenticated users through to the login page
        if not hasattr(request, "user") or not request.user.is_authenticated:
            return self.get_response(request)

        # Already verified this session
        if request.session.get(MFA_VERIFIED_SESSION_KEY):
            return self.get_response(request)

        # Check if user has a confirmed TOTP device and is verified via
        # django-otp's OTPMiddleware (sets request.user.otp_device)
        if getattr(request.user, "otp_device", None) is not None:
            # OTPMiddleware has verified the user; mark the session
            request.session[MFA_VERIFIED_SESSION_KEY] = True
            return self.get_response(request)

        # Allow access to the login and OTP-related admin pages to avoid
        # redirect loops.  The django-otp login view lives at /admin/login/.
        _passthrough_prefixes = (
            "/admin/login/",
            "/admin/logout/",
            "/admin/otp/",
        )
        if any(request.path.startswith(p) for p in _passthrough_prefixes):
            return self.get_response(request)

        # Redirect to the admin login page (which, with OTPAdminSite or
        # OTPMiddleware, will require TOTP verification).
        login_url = reverse("admin:login")
        logger.info(
            "MFA verification required for user=%s path=%s",
            request.user,
            request.path,
        )
        return HttpResponseRedirect(f"{login_url}?next={request.path}")
