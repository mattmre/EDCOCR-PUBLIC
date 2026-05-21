"""Tests for P1 auth and security header fixes (, , ).

Covers:
- Anonymous auth defaults to viewer role (not admin)
- ANONYMOUS_ROLE env var overrides the default
- APIKEY_ROLE default is now "operator" (not "admin")
- APIKEY_ROLE=admin env var restores admin behavior
- SecurityHeadersMiddleware adds standard security headers
- CSP is skipped for docs/openapi/redoc paths
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Anonymous auth defaults to viewer role
# ---------------------------------------------------------------------------


class TestAnonymousRoleDefault:
    """Anonymous identity should default to viewer, not admin."""

    def test_anonymous_identity_gets_viewer_role(self, tmp_path):
        """When ALLOW_UNAUTHENTICATED=true and no API key, role is viewer."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir(exist_ok=True)
        output.mkdir(exist_ok=True)

        with patch("api.config.SOURCE_FOLDER", str(source)), \
             patch("api.config.OUTPUT_FOLDER", str(output)), \
             patch("api.config.OCR_API_KEY", ""), \
             patch("api.auth.OCR_API_KEY", ""), \
             patch("api.config.ALLOW_UNAUTHENTICATED", True), \
             patch("api.auth.ALLOW_UNAUTHENTICATED", True), \
             patch("api.config.ANONYMOUS_ROLE", "viewer"), \
             patch("api.auth.ANONYMOUS_ROLE", "viewer"), \
             patch("api.config.OAUTH2_ENABLED", False), \
             patch("api.oauth2_config.OAUTH2_ENABLED", False), \
             patch("api.oauth2_config.APIKEY_ROLE", "operator"), \
             patch("api.job_manager.config") as mock_config:
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64

            from api.main import create_app

            app = create_app()
            app.state.limiter.enabled = False
            app.state.limiter.reset()
            tc = TestClient(app)

            # Viewer can list jobs (read-only)
            resp = tc.get("/api/v1/jobs")
            assert resp.status_code == 200

            # Viewer cannot submit (requires operator or admin)
            pdf_path = source / "test.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 test")
            with open(pdf_path, "rb") as f:
                resp = tc.post(
                    "/api/v1/jobs",
                    files={"file": ("test.pdf", f, "application/pdf")},
                    data={"priority": "normal"},
                )
            assert resp.status_code == 403

    def test_anonymous_role_env_var_override(self, tmp_path):
        """ANONYMOUS_ROLE env var can escalate anonymous users to operator."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir(exist_ok=True)
        output.mkdir(exist_ok=True)

        with patch("api.config.SOURCE_FOLDER", str(source)), \
             patch("api.config.OUTPUT_FOLDER", str(output)), \
             patch("api.config.OCR_API_KEY", ""), \
             patch("api.auth.OCR_API_KEY", ""), \
             patch("api.config.ALLOW_UNAUTHENTICATED", True), \
             patch("api.auth.ALLOW_UNAUTHENTICATED", True), \
             patch("api.config.ANONYMOUS_ROLE", "operator"), \
             patch("api.auth.ANONYMOUS_ROLE", "operator"), \
             patch("api.config.OAUTH2_ENABLED", False), \
             patch("api.oauth2_config.OAUTH2_ENABLED", False), \
             patch("api.oauth2_config.APIKEY_ROLE", "operator"), \
             patch("api.job_manager.config") as mock_config:
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64

            from api.main import create_app

            app = create_app()
            app.state.limiter.enabled = False
            app.state.limiter.reset()
            tc = TestClient(app)

            # Operator can list jobs
            resp = tc.get("/api/v1/jobs")
            assert resp.status_code == 200

            # Operator can submit jobs
            pdf_path = source / "test.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 test")
            with open(pdf_path, "rb") as f:
                resp = tc.post(
                    "/api/v1/jobs",
                    files={"file": ("test.pdf", f, "application/pdf")},
                    data={"priority": "normal"},
                )
            # 201 (submitted) -- not 403
            assert resp.status_code == 201

    def test_config_anonymous_role_default(self):
        """api.config.ANONYMOUS_ROLE defaults to 'viewer'."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove ANONYMOUS_ROLE if set
            env = {k: v for k, v in os.environ.items() if k != "ANONYMOUS_ROLE"}
            with patch.dict(os.environ, env, clear=True):
                import api.config

                importlib.reload(api.config)
                assert api.config.ANONYMOUS_ROLE == "viewer"

    def test_config_anonymous_role_from_env(self):
        """api.config.ANONYMOUS_ROLE reads from ANONYMOUS_ROLE env var."""
        with patch.dict(os.environ, {"ANONYMOUS_ROLE": "operator"}, clear=False):
            import api.config

            importlib.reload(api.config)
            assert api.config.ANONYMOUS_ROLE == "operator"


# ---------------------------------------------------------------------------
# APIKEY_ROLE defaults to "operator"
# ---------------------------------------------------------------------------


class TestApiKeyRoleDefault:
    """APIKEY_ROLE should default to operator, not admin."""

    def test_apikey_role_defaults_to_operator(self):
        """APIKEY_ROLE defaults to 'operator' when env var is unset."""
        env = {k: v for k, v in os.environ.items() if k != "APIKEY_ROLE"}
        with patch.dict(os.environ, env, clear=True):
            import api.oauth2_config

            importlib.reload(api.oauth2_config)
            assert api.oauth2_config.APIKEY_ROLE == "operator"

    def test_apikey_role_env_override_to_admin(self):
        """APIKEY_ROLE=admin env var restores admin behavior."""
        with patch.dict(os.environ, {"APIKEY_ROLE": "admin"}, clear=False):
            import api.oauth2_config

            importlib.reload(api.oauth2_config)
            assert api.oauth2_config.APIKEY_ROLE == "admin"


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------


def _make_test_app() -> FastAPI:
    """Create a minimal FastAPI app with SecurityHeadersMiddleware."""
    from api.security_headers import SecurityHeadersMiddleware

    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/api/v1/test")
    async def test_endpoint():
        return {"status": "ok"}

    @app.get("/docs")
    async def docs_endpoint():
        return {"docs": True}

    @app.get("/openapi.json")
    async def openapi_endpoint():
        return {"openapi": "3.0.0"}

    @app.get("/redoc")
    async def redoc_endpoint():
        return {"redoc": True}

    return app


class TestSecurityHeadersMiddleware:
    """Security headers middleware injects required headers."""

    @pytest.fixture()
    def client(self):
        app = _make_test_app()
        return TestClient(app)

    def test_x_content_type_options(self, client):
        """X-Content-Type-Options: nosniff is set on all responses."""
        resp = client.get("/api/v1/test")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"

    def test_x_frame_options(self, client):
        """X-Frame-Options: DENY is set on all responses."""
        resp = client.get("/api/v1/test")
        assert resp.headers["X-Frame-Options"] == "DENY"

    def test_x_xss_protection(self, client):
        """X-XSS-Protection header is set on all responses."""
        resp = client.get("/api/v1/test")
        assert resp.headers["X-XSS-Protection"] == "1; mode=block"

    def test_referrer_policy(self, client):
        """Referrer-Policy is set on all responses."""
        resp = client.get("/api/v1/test")
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_permissions_policy(self, client):
        """Permissions-Policy is set on all responses."""
        resp = client.get("/api/v1/test")
        assert resp.headers["Permissions-Policy"] == "geolocation=(), microphone=(), camera=()"

    def test_csp_on_api_path(self, client):
        """Content-Security-Policy is set on non-docs API paths."""
        resp = client.get("/api/v1/test")
        assert resp.headers["Content-Security-Policy"] == "default-src 'self'"

    def test_csp_skipped_on_docs(self, client):
        """Content-Security-Policy is NOT set on /docs path."""
        resp = client.get("/docs")
        assert "Content-Security-Policy" not in resp.headers

    def test_csp_skipped_on_openapi_json(self, client):
        """Content-Security-Policy is NOT set on /openapi.json path."""
        resp = client.get("/openapi.json")
        assert "Content-Security-Policy" not in resp.headers

    def test_csp_skipped_on_redoc(self, client):
        """Content-Security-Policy is NOT set on /redoc path."""
        resp = client.get("/redoc")
        assert "Content-Security-Policy" not in resp.headers

    def test_other_headers_still_set_on_docs(self, client):
        """Non-CSP security headers are still set on docs paths."""
        resp = client.get("/docs")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_csp_policy_env_var(self):
        """CSP_POLICY env var overrides the default policy."""
        custom_policy = "default-src 'self'; script-src 'self'"
        with patch("api.security_headers.CSP_POLICY", custom_policy):
            app = _make_test_app()
            tc = TestClient(app)
            resp = tc.get("/api/v1/test")
            assert resp.headers["Content-Security-Policy"] == custom_policy

    def test_no_hsts_header(self, client):
        """Strict-Transport-Security is NOT set (must be set by reverse proxy)."""
        resp = client.get("/api/v1/test")
        assert "Strict-Transport-Security" not in resp.headers
