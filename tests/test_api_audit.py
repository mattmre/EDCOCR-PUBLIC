"""Tests for request-level API audit logging with hash-chain integrity."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.audit import (
    _compute_event_hash,
    _extract_job_id,
    reset_chain_state,
    verify_audit_chain,
)


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
        stack.enter_context(patch("api.auth.OCR_API_KEY", "unit-test-api-token"))
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


class TestApiAuditLogging:
    """Test structured request-level audit logging behavior."""

    def test_authenticated_request_is_logged(self, audit_app):
        app, audit_path = audit_app
        client = _build_test_client(app, ("203.0.113.10", 50001))

        response = client.get(
            "/api/v1/jobs",
            headers={
                "X-API-Key": "unit-test-api-token",
                "X-Request-ID": "req-audit-001",
            },
        )

        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "req-audit-001"

        events = _read_audit_lines(audit_path)
        assert len(events) == 1

        event = events[0]
        assert event["request_id"] == "req-audit-001"
        assert event["method"] == "GET"
        assert event["path"] == "/api/v1/jobs"
        assert event["status_code"] == 200
        assert event["client_ip"] in {"203.0.113.10", "testclient"}
        assert event["auth_method"] == "apikey"
        assert event["auth_outcome"] == "authorized"
        assert event["subject"] == "apikey"
        assert "X-API-Key" not in event
        assert "Authorization" not in event

    def test_unauthorized_request_logs_without_query_secret(self, audit_app):
        app, audit_path = audit_app
        client = _build_test_client(app, ("198.51.100.44", 50002))

        response = client.get("/api/v1/jobs?token=secret-value")

        assert response.status_code == 401

        events = _read_audit_lines(audit_path)
        assert len(events) == 1

        event = events[0]
        assert event["path"] == "/api/v1/jobs"
        assert event["query_present"] is True
        assert event["status_code"] == 401
        assert event["auth_outcome"] == "missing_api_key"
        assert event["auth_method"] == ""
        assert event["subject"] == ""

        with open(audit_path, "r", encoding="utf-8") as handle:
            raw_text = handle.read()
        assert "secret-value" not in raw_text
        assert "unit-test-api-token" not in raw_text

    def test_exempt_health_request_is_logged(self, audit_app):
        app, audit_path = audit_app
        client = _build_test_client(app)

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        assert response.headers["X-Request-ID"].startswith("req_")

        events = _read_audit_lines(audit_path)
        assert len(events) == 1
        assert events[0]["path"] == "/api/v1/health"
        assert events[0]["auth_outcome"] == "exempt"
        assert events[0]["auth_method"] == ""

    def test_audit_logging_can_be_disabled(self, tmp_path):
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()
        audit_path = output / "logs" / "api-audit.jsonl"

        with ExitStack() as stack:
            stack.enter_context(patch("api.auth.OCR_API_KEY", "unit-test-api-token"))
            stack.enter_context(patch("api.config.SOURCE_FOLDER", str(source)))
            stack.enter_context(patch("api.config.OUTPUT_FOLDER", str(output)))
            stack.enter_context(patch("api.config.API_AUDIT_LOG_ENABLED", False))
            stack.enter_context(patch("api.config.API_AUDIT_LOG_PATH", ""))
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

            response = client.get(
                "/api/v1/jobs",
                headers={"X-API-Key": "unit-test-api-token"},
            )

        assert response.status_code == 200
        assert not audit_path.exists()


class TestAuditHashChain:
    """Test SHA-256 hash chain integrity on audit entries."""

    def test_first_entry_has_null_prev_hash(self, audit_app):
        """First audit entry should have prev_hash=None (genesis)."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "unit-test-api-token"},
        )

        events = _read_audit_lines(audit_path)
        assert len(events) == 1
        assert events[0]["prev_hash"] is None
        assert events[0]["hash"] is not None
        assert len(events[0]["hash"]) == 64  # SHA-256 hex digest

    def test_sequential_entries_are_hash_linked(self, audit_app):
        """Each entry's prev_hash should match the previous entry's hash."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        # Make three requests to form a chain
        client.get("/api/v1/health")
        client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "unit-test-api-token"},
        )
        client.get("/api/v1/health")

        events = _read_audit_lines(audit_path)
        assert len(events) == 3

        assert events[0]["prev_hash"] is None
        assert events[1]["prev_hash"] == events[0]["hash"]
        assert events[2]["prev_hash"] == events[1]["hash"]

    def test_hash_is_deterministic(self, audit_app):
        """Recomputing the hash from event fields should match stored hash."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "unit-test-api-token"},
        )

        events = _read_audit_lines(audit_path)
        event = events[0]

        # Recompute hash excluding the hash field
        hashable = {k: v for k, v in event.items() if k != "hash"}
        event_bytes = json.dumps(hashable, sort_keys=True, default=str).encode("utf-8")
        recomputed = hashlib.sha256(event_bytes).hexdigest()

        assert recomputed == event["hash"]

    def test_chain_verifies_via_utility(self, audit_app):
        """verify_audit_chain should pass for an untampered chain."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get("/api/v1/health")
        client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "unit-test-api-token"},
        )
        client.get("/api/v1/health")

        is_valid, message = verify_audit_chain(str(audit_path))
        assert is_valid is True
        assert "3 entries" in message

    def test_tampered_entry_detected(self, audit_app):
        """verify_audit_chain should detect a modified entry."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get("/api/v1/health")
        client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "unit-test-api-token"},
        )

        # Tamper with the stored file
        events = _read_audit_lines(audit_path)
        events[0]["status_code"] = 999
        with open(audit_path, "w", encoding="utf-8") as handle:
            for ev in events:
                handle.write(json.dumps(ev) + "\n")

        is_valid, message = verify_audit_chain(str(audit_path))
        assert is_valid is False
        assert "Tampered entry" in message

    def test_broken_chain_link_detected(self, audit_app):
        """verify_audit_chain should detect a broken prev_hash link."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get("/api/v1/health")
        client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "unit-test-api-token"},
        )

        events = _read_audit_lines(audit_path)
        events[1]["prev_hash"] = "0" * 64
        # Recompute hash to avoid tamper detection on hash field
        hashable = {k: v for k, v in events[1].items() if k != "hash"}
        events[1]["hash"] = hashlib.sha256(
            json.dumps(hashable, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        with open(audit_path, "w", encoding="utf-8") as handle:
            for ev in events:
                handle.write(json.dumps(ev) + "\n")

        is_valid, message = verify_audit_chain(str(audit_path))
        assert is_valid is False
        assert "Broken chain" in message

    def test_verify_empty_audit_file(self, tmp_path):
        """Verifying an empty audit file should succeed."""
        empty_file = tmp_path / "empty-audit.jsonl"
        empty_file.write_text("", encoding="utf-8")

        is_valid, message = verify_audit_chain(str(empty_file))
        assert is_valid is True
        assert "Empty audit chain" in message

    def test_verify_nonexistent_file(self, tmp_path):
        """Verifying a nonexistent file should fail gracefully."""
        is_valid, message = verify_audit_chain(str(tmp_path / "no-such-file.jsonl"))
        assert is_valid is False
        assert "Failed to load" in message

    def test_verify_corrupted_jsonl(self, tmp_path):
        """Verifying a file with invalid JSON should fail gracefully."""
        bad_file = tmp_path / "bad-audit.jsonl"
        bad_file.write_text('{"valid": true}\nnot json\n', encoding="utf-8")

        is_valid, message = verify_audit_chain(str(bad_file))
        assert is_valid is False
        assert "Failed to load" in message


class TestJobIdExtraction:
    """Test job ID extraction from URL path parameters."""

    def test_job_id_extracted_from_status_path(self, audit_app):
        """Job ID should be extracted when present in the URL path."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        response = client.get(
            "/api/v1/jobs/job_1a2b3c4d5e6f",
            headers={"X-API-Key": "unit-test-api-token"},
        )

        # The job doesn't exist, so we get 404, but audit still records
        assert response.status_code == 404

        events = _read_audit_lines(audit_path)
        assert len(events) == 1
        assert events[0]["job_id"] == "job_1a2b3c4d5e6f"

    def test_no_job_id_for_list_endpoint(self, audit_app):
        """No job_id should appear for the jobs list endpoint."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "unit-test-api-token"},
        )

        events = _read_audit_lines(audit_path)
        assert len(events) == 1
        assert events[0]["job_id"] is None

    def test_no_job_id_for_health_endpoint(self, audit_app):
        """No job_id should appear for the health endpoint."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get("/api/v1/health")

        events = _read_audit_lines(audit_path)
        assert len(events) == 1
        assert events[0]["job_id"] is None

    def test_job_id_from_result_path(self, audit_app):
        """Job ID should be extracted from nested paths like /result."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get(
            "/api/v1/jobs/job_aabbccddeeff/result",
            headers={"X-API-Key": "unit-test-api-token"},
        )

        events = _read_audit_lines(audit_path)
        assert len(events) == 1
        assert events[0]["job_id"] == "job_aabbccddeeff"


class TestJobIdExtractionUnit:
    """Unit tests for the _extract_job_id helper."""

    def test_valid_job_id_path(self):
        assert _extract_job_id("/api/v1/jobs/job_1a2b3c4d5e6f") == "job_1a2b3c4d5e6f"

    def test_job_id_from_result_subpath(self):
        assert _extract_job_id("/api/v1/jobs/job_aabbccddeeff/result") == "job_aabbccddeeff"

    def test_job_id_from_retry_subpath(self):
        assert _extract_job_id("/api/v1/jobs/job_112233445566/retry") == "job_112233445566"

    def test_no_job_id_for_list(self):
        assert _extract_job_id("/api/v1/jobs") is None

    def test_no_job_id_for_health(self):
        assert _extract_job_id("/api/v1/health") is None

    def test_no_job_id_for_invalid_format(self):
        assert _extract_job_id("/api/v1/jobs/invalid_id") is None

    def test_no_job_id_for_short_hex(self):
        assert _extract_job_id("/api/v1/jobs/job_1a2b3c") is None


class TestHealthExclusion:
    """Test configurable health endpoint exclusion from audit logs."""

    def test_health_excluded_when_configured(self, tmp_path):
        """Health endpoint should be skipped when API_AUDIT_EXCLUDE_HEALTH is set."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()
        audit_path = output / "logs" / "api-audit.jsonl"

        with ExitStack() as stack:
            stack.enter_context(patch("api.auth.OCR_API_KEY", "unit-test-api-token"))
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

            # Jobs should still be logged
            client.get(
                "/api/v1/jobs",
                headers={"X-API-Key": "unit-test-api-token"},
            )

        events = _read_audit_lines(audit_path)
        assert len(events) == 1
        assert events[0]["path"] == "/api/v1/jobs"

    def test_health_logged_when_exclusion_disabled(self, audit_app):
        """Health requests should be logged when exclusion is off (default)."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get("/api/v1/health")

        events = _read_audit_lines(audit_path)
        assert len(events) == 1
        assert events[0]["path"] == "/api/v1/health"


class TestComputeEventHash:
    """Unit tests for _compute_event_hash helper."""

    def test_hash_excludes_hash_field(self):
        """The hash field itself should not be included in the computation."""
        event = {"a": 1, "b": 2, "hash": "should_be_excluded"}
        h1 = _compute_event_hash(event)

        event_no_hash = {"a": 1, "b": 2}
        h2 = _compute_event_hash(event_no_hash)

        assert h1 == h2

    def test_hash_is_deterministic(self):
        """Same input should always produce the same hash."""
        event = {"method": "GET", "path": "/test", "status_code": 200}
        assert _compute_event_hash(event) == _compute_event_hash(event)

    def test_different_events_different_hashes(self):
        """Different events should produce different hashes."""
        e1 = {"method": "GET", "path": "/a"}
        e2 = {"method": "GET", "path": "/b"}
        assert _compute_event_hash(e1) != _compute_event_hash(e2)

    def test_hash_length(self):
        """Hash should be 64 hex chars (SHA-256)."""
        event = {"test": "data"}
        h = _compute_event_hash(event)
        assert len(h) == 64
        int(h, 16)  # Should be valid hex


class TestResetChainState:
    """Test chain state reset for test isolation."""

    def test_reset_clears_prev_hash(self, audit_app):
        """After reset, next entry should have prev_hash=None."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        # First request establishes chain
        client.get("/api/v1/health")
        events = _read_audit_lines(audit_path)
        assert events[0]["prev_hash"] is None

        # Second request links to first
        client.get("/api/v1/health")
        events = _read_audit_lines(audit_path)
        assert events[1]["prev_hash"] == events[0]["hash"]

        # Reset chain state
        reset_chain_state()

        # Third request should have prev_hash=None again
        client.get("/api/v1/health")
        events = _read_audit_lines(audit_path)
        assert events[2]["prev_hash"] is None


class TestApiKeyMasking:
    """Test that raw API keys never appear in audit logs."""

    def test_raw_api_key_never_in_audit_file(self, audit_app):
        """Raw API keys must never appear anywhere in the JSONL file."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        # Authorized request
        client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "unit-test-api-token"},
        )
        # Unauthorized request with a different key
        client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "bad-key-value-12345"},
        )

        with open(audit_path, "r", encoding="utf-8") as handle:
            raw = handle.read()

        assert "unit-test-api-token" not in raw
        assert "bad-key-value-12345" not in raw

    def test_no_request_body_in_audit(self, audit_app):
        """Audit events should not contain any request body fields."""
        app, audit_path = audit_app
        client = _build_test_client(app)

        client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "unit-test-api-token"},
        )

        events = _read_audit_lines(audit_path)
        event = events[0]

        # No body-related keys
        assert "body" not in event
        assert "request_body" not in event
        assert "payload" not in event
        assert "content" not in event
