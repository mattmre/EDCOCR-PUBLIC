"""Security tests for auth compare_digest hardening and docs gating.

Covers:
- AUTH-BUG: Latin-1 / non-ASCII X-API-Key values must not crash the
  auth middleware (secrets.compare_digest raises TypeError on chars > 127).
- /docs, /redoc, /openapi.json must not be mounted by default
  (EXPOSE_API_DOCS=false); setting EXPOSE_API_DOCS=true restores them.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _build_app(tmp_path, *, api_key: str = "test-api-key-12345",
                expose_docs: bool = False):
    """Build a FastAPI TestClient with mocked config."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)

    with patch("api.config.SOURCE_FOLDER", str(source)), \
         patch("api.config.OUTPUT_FOLDER", str(output)), \
         patch("api.config.OCR_API_KEY", api_key), \
         patch("api.auth.OCR_API_KEY", api_key), \
         patch("api.config.ALLOW_UNAUTHENTICATED", False), \
         patch("api.auth.ALLOW_UNAUTHENTICATED", False), \
         patch("api.config.ENABLE_MULTITENANCY", False), \
         patch("api.auth.ENABLE_MULTITENANCY", False), \
         patch("api.config.EXPOSE_API_DOCS", expose_docs), \
         patch("api.auth.EXPOSE_API_DOCS", expose_docs), \
         patch("api.main.EXPOSE_API_DOCS", expose_docs), \
         patch("api.config.OAUTH2_ENABLED", False), \
         patch("api.oauth2_config.OAUTH2_ENABLED", False), \
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
        yield TestClient(app)


# ---------------------------------------------------------------------------
# AUTH-BUG: Non-ASCII API key headers must not crash the middleware
# ---------------------------------------------------------------------------


class TestSafeCompareHelper:
    """Direct unit tests on the _safe_compare_api_key helper.

    This is the primary defence against AUTH-BUG. The helper is called
    synchronously from the middleware, so a direct unit test exercises the
    code path that an attacker-controlled header would take.
    """

    def test_ascii_match(self):
        from api.auth import _safe_compare_api_key

        assert _safe_compare_api_key("abc", "abc") is True

    def test_ascii_mismatch(self):
        from api.auth import _safe_compare_api_key

        assert _safe_compare_api_key("abc", "xyz") is False

    def test_latin1_provided_returns_false(self):
        """Latin-1 high chars (128-255) raise TypeError in compare_digest."""
        from api.auth import _safe_compare_api_key

        # "\u00ff" is the canonical case discovered by Hypothesis (AUTH-BUG).
        assert _safe_compare_api_key("\u00ff", "abc") is False

    def test_latin1_both_sides_returns_false(self):
        from api.auth import _safe_compare_api_key

        assert _safe_compare_api_key("\u00ff", "\u00ee") is False

    def test_non_ascii_unicode_returns_false(self):
        """High codepoints (> 255) must not crash either."""
        from api.auth import _safe_compare_api_key

        assert _safe_compare_api_key("caf\u00e9-key", "abc") is False
        assert _safe_compare_api_key("\U0001F512-key", "abc") is False

    def test_latin1_never_matches_ascii(self):
        """A latin-1 provided key must never match an ASCII expected key."""
        from api.auth import _safe_compare_api_key

        assert _safe_compare_api_key("\u00ffvalid", "valid") is False
        assert _safe_compare_api_key("valid\u00ff", "valid") is False

    def test_empty_inputs_return_false(self):
        from api.auth import _safe_compare_api_key

        assert _safe_compare_api_key("", "abc") is False
        assert _safe_compare_api_key("abc", "") is False
        assert _safe_compare_api_key("", "") is False

    def test_none_inputs_return_false(self):
        """Defensive: should not crash if caller passes None."""
        from api.auth import _safe_compare_api_key

        assert _safe_compare_api_key(None, "abc") is False  # type: ignore[arg-type]
        assert _safe_compare_api_key("abc", None) is False  # type: ignore[arg-type]

    def test_does_not_raise_type_error(self):
        """Regression guard: the helper must never raise on bad input."""
        import secrets as _secrets

        from api.auth import _safe_compare_api_key

        # Confirm the underlying call would raise TypeError without the wrapper.
        with pytest.raises(TypeError):
            _secrets.compare_digest("\u00ff", "abc")

        # The wrapper swallows it.
        try:
            result = _safe_compare_api_key("\u00ff", "abc")
        except TypeError:
            pytest.fail("_safe_compare_api_key must not raise TypeError")
        assert result is False


class TestLatin1ApiKeyMiddleware:
    """End-to-end middleware tests using latin-1 header values.

    HTTP headers are latin-1 encoded per RFC 7230, so bytes 128-255 can
    legally reach the middleware.  These are exactly the codepoints that
    make ``secrets.compare_digest`` raise ``TypeError`` when passed as str.
    """

    def test_latin1_header_returns_401_not_500(self, tmp_path):
        """A latin-1 (>= 128) char in X-API-Key must produce a clean 401.

        httpx's ``TestClient`` defaults to ASCII header encoding so we
        build the request manually with raw bytes to mimic what a
        non-compliant HTTP client would put on the wire.
        """
        with _build_app(tmp_path) as tc:
            # Use raw-byte headers via httpx Request to bypass the client-side
            # ASCII guard.  The header name is ASCII; the value is latin-1.
            import httpx

            request = httpx.Request(
                "GET",
                "http://testserver/api/v1/jobs",
                headers=[
                    (b"x-api-key", b"\xffnot-a-real-key"),
                ],
            )
            resp = tc.send(request)
            assert resp.status_code == 401, (
                f"Expected 401 for latin-1 header, got {resp.status_code}: {resp.text}"
            )
            body = resp.json()
            assert body.get("error") == "unauthorized"

    def test_valid_ascii_api_key_still_works(self, tmp_path):
        """Regression guard: valid ASCII keys still authorize."""
        with _build_app(tmp_path, api_key="valid-ascii-key") as tc:
            resp = tc.get(
                "/api/v1/jobs",
                headers={"X-API-Key": "valid-ascii-key"},
            )
            assert resp.status_code == 200

    def test_wrong_ascii_api_key_returns_401(self, tmp_path):
        """Regression guard: wrong ASCII keys still return 401."""
        with _build_app(tmp_path, api_key="valid-ascii-key") as tc:
            resp = tc.get(
                "/api/v1/jobs",
                headers={"X-API-Key": "wrong-ascii-key"},
            )
            assert resp.status_code == 401


# ---------------------------------------------------------------------------
# OpenAPI docs must be gated by EXPOSE_API_DOCS
# ---------------------------------------------------------------------------


class TestDocsGating:
    """/docs, /redoc, /openapi.json must require EXPOSE_API_DOCS=true."""

    @pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
    def test_docs_not_mounted_by_default(self, tmp_path, path):
        """When EXPOSE_API_DOCS=false, docs routes are not mounted.

        We send a valid API key so that auth succeeds and the only possible
        failure is routing (404 when the path is truly unmounted).  An
        attacker without a key would see 401 instead; either way the schema
        content is unreachable, but this test pins the actual FastAPI
        behaviour under the intended deployment config.
        """
        with _build_app(tmp_path, expose_docs=False) as tc:
            resp = tc.get(path, headers={"X-API-Key": "test-api-key-12345"})
            assert resp.status_code == 404, (
                f"Expected {path} to be unmounted, got {resp.status_code}"
            )

    @pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
    def test_docs_unauthenticated_does_not_leak_schema(self, tmp_path, path):
        """Unauthenticated access to docs paths must not return the schema."""
        with _build_app(tmp_path, expose_docs=False) as tc:
            resp = tc.get(path)
            # Either 401 (auth rejected first) or 404 (route not mounted)
            # is acceptable -- both hide the schema from unauthenticated
            # callers.  The critical assertion is that we never return
            # 200 or schema content.
            assert resp.status_code in (401, 404)
            assert "openapi" not in resp.text.lower() or resp.status_code != 200

    @pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
    def test_docs_mounted_when_opted_in(self, tmp_path, path):
        """When EXPOSE_API_DOCS=true, docs routes are reachable."""
        with _build_app(tmp_path, expose_docs=True) as tc:
            resp = tc.get(path)
            # Docs paths are exempt from auth in the legacy behaviour, so
            # they should return 200 without an API key.
            assert resp.status_code == 200, (
                f"Expected {path} to be reachable, got {resp.status_code}"
            )

    def test_openapi_schema_content_when_enabled(self, tmp_path):
        """When docs are enabled, /openapi.json returns a valid schema."""
        with _build_app(tmp_path, expose_docs=True) as tc:
            resp = tc.get("/openapi.json")
            assert resp.status_code == 200
            schema = resp.json()
            assert schema.get("openapi", "").startswith("3.")
            assert "paths" in schema

    def test_docs_disabled_does_not_affect_health(self, tmp_path):
        """Health probes remain exempt regardless of EXPOSE_API_DOCS."""
        with _build_app(tmp_path, expose_docs=False) as tc:
            resp = tc.get("/api/v1/health")
            assert resp.status_code == 200
