"""Tests for CORS middleware configuration.

Verifies that CORS headers are present when CORS_ALLOWED_ORIGINS is
configured and absent when it is not.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient


class TestCorsMiddleware:
    """Verify CORS middleware opt-in behavior."""

    def test_cors_headers_present_when_origins_configured(self):
        """CORS headers appear on preflight when CORS_ALLOWED_ORIGINS is set."""
        with patch("api.config.CORS_ALLOWED_ORIGINS", ("https://app.example.com",)):
            with patch("api.main.CORS_ALLOWED_ORIGINS", ("https://app.example.com",)):
                # Re-create the app with CORS enabled
                from api.main import create_app

                app = create_app()
                client = TestClient(app)

                # Send a CORS preflight request
                response = client.options(
                    "/api/v1/health",
                    headers={
                        "Origin": "https://app.example.com",
                        "Access-Control-Request-Method": "GET",
                    },
                )
                assert "access-control-allow-origin" in response.headers
                assert response.headers["access-control-allow-origin"] == "https://app.example.com"

    def test_cors_headers_absent_when_origins_empty(self):
        """No CORS headers when CORS_ALLOWED_ORIGINS is empty (default)."""
        with patch("api.config.CORS_ALLOWED_ORIGINS", ()):
            with patch("api.main.CORS_ALLOWED_ORIGINS", ()):
                from api.main import create_app

                app = create_app()
                client = TestClient(app)

                response = client.options(
                    "/api/v1/health",
                    headers={
                        "Origin": "https://evil.example.com",
                        "Access-Control-Request-Method": "GET",
                    },
                )
                # CORS middleware not registered -- no CORS headers
                assert "access-control-allow-origin" not in response.headers

    def test_cors_rejects_unlisted_origin(self):
        """CORS middleware does not allow origins not in the configured list."""
        with patch("api.config.CORS_ALLOWED_ORIGINS", ("https://trusted.example.com",)):
            with patch("api.main.CORS_ALLOWED_ORIGINS", ("https://trusted.example.com",)):
                from api.main import create_app

                app = create_app()
                client = TestClient(app)

                response = client.options(
                    "/api/v1/health",
                    headers={
                        "Origin": "https://evil.example.com",
                        "Access-Control-Request-Method": "GET",
                    },
                )
                # Origin not in allowed list -- should not echo it back
                cors_origin = response.headers.get("access-control-allow-origin", "")
                assert cors_origin != "https://evil.example.com"

    def test_cors_config_env_var_parsing(self):
        """CORS_ALLOWED_ORIGINS env var is correctly parsed into tuple."""
        from api.config import _parse_csv_values

        result = _parse_csv_values("https://a.com, https://b.com , https://c.com")
        assert result == ("https://a.com", "https://b.com", "https://c.com")

    def test_cors_config_empty_string(self):
        """Empty CORS_ALLOWED_ORIGINS env var produces empty tuple."""
        from api.config import _parse_csv_values

        result = _parse_csv_values("")
        assert result == ()
