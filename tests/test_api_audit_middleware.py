"""Tests for API audit middleware: masked API key logging, health skip, JSONL format.

These tests complement ``test_api_audit.py`` by focusing on the masked API key
feature added to :func:`api.audit.build_api_audit_event` and validating the
convenience re-exports in ``api.audit_middleware``.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.audit import mask_api_key, reset_chain_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_client(app, client_addr=None):
    """Create a TestClient that works across Starlette versions."""
    if client_addr is None:
        return TestClient(app)
    try:
        return TestClient(app, client=client_addr)
    except TypeError:
        return TestClient(app)


def _read_audit_lines(audit_path: Path) -> list[dict]:
    """Read JSONL audit records from disk."""
    with open(audit_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_audit_chain():
    """Reset the audit hash chain between tests for isolation."""
    reset_chain_state()
    yield
    reset_chain_state()


@pytest.fixture()
def audit_app(tmp_path):
    """Provide an isolated app and its derived audit-log path."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    audit_path = output / "logs" / "api-audit.jsonl"

    with ExitStack() as stack:
        stack.enter_context(patch("api.auth.OCR_API_KEY", "abcdefgh-long-api-token"))
        stack.enter_context(patch("api.config.SOURCE_FOLDER", str(source)))
        stack.enter_context(patch("api.config.OUTPUT_FOLDER", str(output)))
        stack.enter_context(patch("api.config.API_AUDIT_LOG_ENABLED", True))
        stack.enter_context(patch("api.config.API_AUDIT_LOG_PATH", ""))
        stack.enter_context(patch("api.config.API_AUDIT_EXCLUDE_HEALTH", False))
        mock_config = stack.enter_context(patch("api.job_manager.config"))
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 1
        mock_config.MAX_CONCURRENT_JOBS = 64

        from api.main import create_app

        app = create_app()
        app.state.limiter.enabled = False
        app.state.limiter.reset()
        yield app, audit_path


# ---------------------------------------------------------------------------
# Unit tests for mask_api_key
# ---------------------------------------------------------------------------


class TestMaskApiKey:
    """Unit tests for the mask_api_key helper."""

    def _make_request(self, api_key: str | None = None):
        """Build a minimal mock request with an optional X-API-Key header."""
        from unittest.mock import MagicMock

        request = MagicMock()
        headers = {}
        if api_key is not None:
            headers["X-API-Key"] = api_key
        request.headers = headers
        return request

    def test_long_key_shows_first_eight_chars(self):
        """Keys longer than 8 chars show first 8 + '***'."""
        request = self._make_request("abcdefghijklmnop")
        assert mask_api_key(request) == "abcdefgh***"

    def test_short_key_shows_first_four_chars(self):
        """Keys with 8 or fewer chars show first 4 + '***'."""
        request = self._make_request("abcdefgh")
        assert mask_api_key(request) == "abcd***"

    def test_very_short_key(self):
        """Very short keys still mask safely."""
        request = self._make_request("abc")
        assert mask_api_key(request) == "abc***"

    def test_no_api_key_returns_none(self):
        """Missing X-API-Key header returns None."""
        request = self._make_request(None)
        assert mask_api_key(request) is None

    def test_empty_api_key_returns_none(self):
        """Empty X-API-Key header returns None."""
        request = self._make_request("")
        assert mask_api_key(request) is None


# ---------------------------------------------------------------------------
# Integration tests: masked key appears in audit events
# ---------------------------------------------------------------------------


class TestAuditMaskedKeyIntegration:
    """Verify that audit JSONL entries include a masked API key."""

    def test_authenticated_request_has_masked_key(self, audit_app):
        """Authorized requests should have api_key_masked with first 8 chars."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        response = client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "abcdefgh-long-api-token"},
        )
        assert response.status_code == 200

        events = _read_audit_lines(audit_path)
        assert len(events) == 1
        assert events[0]["api_key_masked"] == "abcdefgh***"

    def test_unauthorized_request_has_masked_key(self, audit_app):
        """Failed auth requests still record the masked key that was tried."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        response = client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "wrong-key-value-here"},
        )
        assert response.status_code == 401

        events = _read_audit_lines(audit_path)
        assert len(events) == 1
        assert events[0]["api_key_masked"] == "wrong-ke***"

    def test_no_key_request_has_null_masked_key(self, audit_app):
        """Requests without X-API-Key should have api_key_masked=null."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        # Health endpoint is exempt from auth, so no key needed
        client.get("/api/v1/health")

        events = _read_audit_lines(audit_path)
        assert len(events) == 1
        assert events[0]["api_key_masked"] is None

    def test_raw_key_never_in_audit_file(self, audit_app):
        """The full raw API key must never appear in the audit file."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "abcdefgh-long-api-token"},
        )
        client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "bad-key-secret-12345"},
        )

        with open(audit_path, "r", encoding="utf-8") as handle:
            raw = handle.read()

        assert "abcdefgh-long-api-token" not in raw
        assert "bad-key-secret-12345" not in raw


# ---------------------------------------------------------------------------
# Integration tests: health check exclusion
# ---------------------------------------------------------------------------


class TestHealthCheckSkip:
    """Verify that health checks can be excluded from audit logs."""

    def test_health_skipped_when_configured(self, tmp_path):
        """With API_AUDIT_EXCLUDE_HEALTH=True, /api/v1/health is not logged."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()
        audit_path = output / "logs" / "api-audit.jsonl"

        with ExitStack() as stack:
            stack.enter_context(
                patch("api.auth.OCR_API_KEY", "abcdefgh-long-api-token")
            )
            stack.enter_context(patch("api.config.SOURCE_FOLDER", str(source)))
            stack.enter_context(patch("api.config.OUTPUT_FOLDER", str(output)))
            stack.enter_context(patch("api.config.API_AUDIT_LOG_ENABLED", True))
            stack.enter_context(patch("api.config.API_AUDIT_LOG_PATH", ""))
            stack.enter_context(patch("api.config.API_AUDIT_EXCLUDE_HEALTH", True))
            mock_config = stack.enter_context(patch("api.job_manager.config"))
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64

            from api.main import create_app

            app = create_app()
            app.state.limiter.enabled = False
            app.state.limiter.reset()
            client = _build_test_client(app)

            # Health should be excluded
            response = client.get("/api/v1/health")
            assert response.status_code == 200

            # Regular endpoints should still be logged
            client.get(
                "/api/v1/jobs",
                headers={"X-API-Key": "abcdefgh-long-api-token"},
            )

        events = _read_audit_lines(audit_path)
        assert len(events) == 1
        assert events[0]["path"] == "/api/v1/jobs"

    def test_health_logged_when_exclusion_off(self, audit_app):
        """With API_AUDIT_EXCLUDE_HEALTH=False (default), health is logged."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get("/api/v1/health")

        events = _read_audit_lines(audit_path)
        assert len(events) == 1
        assert events[0]["path"] == "/api/v1/health"


# ---------------------------------------------------------------------------
# JSONL format validity
# ---------------------------------------------------------------------------


class TestJsonlFormatValidity:
    """Verify that every audit entry is valid JSONL with expected fields."""

    def test_each_line_is_valid_json(self, audit_app):
        """Every line in the audit file must parse as valid JSON."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        # Generate a few audit entries
        client.get("/api/v1/health")
        client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "abcdefgh-long-api-token"},
        )
        client.get("/api/v1/jobs", headers={"X-API-Key": "bad-key"})

        with open(audit_path, "r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    pytest.fail(f"Line {i + 1} is not valid JSON: {line!r}")

    def test_required_fields_present(self, audit_app):
        """Each audit entry must contain all required fields."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "abcdefgh-long-api-token"},
        )

        events = _read_audit_lines(audit_path)
        assert len(events) == 1

        required_fields = {
            "timestamp",
            "request_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "client_ip",
            "api_key_masked",
            "auth_method",
            "auth_outcome",
            "hash",
            "prev_hash",
        }
        event = events[0]
        missing = required_fields - set(event.keys())
        assert not missing, f"Missing required fields: {missing}"

    def test_duration_is_positive_float(self, audit_app):
        """duration_ms must be a positive numeric value."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get("/api/v1/health")

        events = _read_audit_lines(audit_path)
        assert len(events) == 1
        assert isinstance(events[0]["duration_ms"], (int, float))
        assert events[0]["duration_ms"] >= 0

    def test_status_code_is_integer(self, audit_app):
        """status_code must be an integer."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get("/api/v1/health")

        events = _read_audit_lines(audit_path)
        assert isinstance(events[0]["status_code"], int)

    def test_timestamp_is_iso8601(self, audit_app):
        """Timestamp must be a valid ISO 8601 string."""
        import datetime

        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get("/api/v1/health")

        events = _read_audit_lines(audit_path)
        ts = events[0]["timestamp"]
        # Should parse without error
        parsed = datetime.datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None, "Timestamp must be timezone-aware"


# ---------------------------------------------------------------------------
# Re-export module tests
# ---------------------------------------------------------------------------


class TestAuditMiddlewareModule:
    """Verify that api.audit_middleware re-exports work correctly."""

    def test_import_middleware_class(self):
        from api.audit_middleware import ApiAuditMiddleware

        assert ApiAuditMiddleware is not None

    def test_import_mask_api_key(self):
        from api.audit_middleware import mask_api_key as imported_mask

        assert imported_mask is mask_api_key

    def test_import_verify_audit_chain(self):
        from api.audit_middleware import verify_audit_chain

        assert callable(verify_audit_chain)

    def test_import_build_api_audit_event(self):
        from api.audit_middleware import build_api_audit_event

        assert callable(build_api_audit_event)

    def test_all_exports(self):
        import api.audit_middleware as mod

        assert hasattr(mod, "__all__")
        assert "ApiAuditMiddleware" in mod.__all__
        assert "mask_api_key" in mod.__all__
