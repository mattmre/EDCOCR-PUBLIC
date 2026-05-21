"""Tests for webhook notification feature — URL validation, signing, payload, delivery."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    """FastAPI TestClient with isolated DB and temp dirs."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with patch("api.config.SOURCE_FOLDER", str(source)), \
         patch("api.config.OUTPUT_FOLDER", str(output)), \
         patch("api.config.WEBHOOK_ALLOW_HTTP", True), \
         patch("api.config.WEBHOOK_ALLOW_PRIVATE", True), \
         patch("api.job_manager.config") as mock_config:
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 1
        mock_config.MAX_CONCURRENT_JOBS = 64
        mock_config.WEBHOOK_TIMEOUT = 30
        mock_config.WEBHOOK_MAX_RETRIES = 3
        mock_config.WEBHOOK_SECRET = ""

        from api.main import create_app

        app = create_app()
        app.state.limiter.enabled = False
        app.state.limiter.reset()

        from fastapi.testclient import TestClient

        yield TestClient(app)


@pytest.fixture()
def sample_pdf(tmp_path) -> Path:
    """Create a minimal PDF file for upload testing."""
    pdf = tmp_path / "test_doc.pdf"
    pdf.write_bytes(
        b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
    )
    return pdf


# ---------------------------------------------------------------------------
# URL Validation Tests
# ---------------------------------------------------------------------------


class TestWebhookURLValidation:
    def test_rejects_http_by_default(self):
        from api.webhooks import validate_webhook_url

        with pytest.raises(ValueError, match="HTTPS"):
            validate_webhook_url("http://example.com/webhook")

    def test_allows_http_when_configured(self):
        from api.webhooks import validate_webhook_url

        result = validate_webhook_url(
            "http://example.com/webhook",
            allow_http=True,
            allow_private=True,
        )
        assert result == "http://example.com/webhook"

    def test_rejects_empty_url(self):
        from api.webhooks import validate_webhook_url

        with pytest.raises(ValueError, match="empty"):
            validate_webhook_url("")

    def test_rejects_whitespace_only_url(self):
        from api.webhooks import validate_webhook_url

        with pytest.raises(ValueError, match="empty"):
            validate_webhook_url("   ")

    def test_rejects_too_long_url(self):
        from api.webhooks import validate_webhook_url

        long_url = "https://example.com/" + "a" * 2040
        with pytest.raises(ValueError, match="2048"):
            validate_webhook_url(long_url)

    def test_accepts_valid_https_url(self):
        from api.webhooks import validate_webhook_url

        result = validate_webhook_url(
            "https://example.com/webhook/callback",
            allow_private=True,
        )
        assert result == "https://example.com/webhook/callback"

    def test_rejects_unsupported_scheme(self):
        from api.webhooks import validate_webhook_url

        with pytest.raises(ValueError, match="unsupported scheme"):
            validate_webhook_url("ftp://example.com/webhook")

    def test_rejects_missing_scheme(self):
        from api.webhooks import validate_webhook_url

        with pytest.raises(ValueError, match="scheme"):
            validate_webhook_url("example.com/webhook")

    def test_rejects_missing_hostname(self):
        from api.webhooks import validate_webhook_url

        with pytest.raises(ValueError, match="hostname"):
            validate_webhook_url("https://")

    def test_rejects_localhost_by_default(self):
        from api.webhooks import validate_webhook_url

        with pytest.raises(ValueError, match="localhost"):
            validate_webhook_url("https://localhost/webhook")

    def test_allows_localhost_when_private_allowed(self):
        from api.webhooks import validate_webhook_url

        result = validate_webhook_url(
            "https://localhost/webhook",
            allow_private=True,
        )
        assert result == "https://localhost/webhook"

    def test_strips_whitespace(self):
        from api.webhooks import validate_webhook_url

        result = validate_webhook_url(
            "  https://example.com/webhook  ",
            allow_private=True,
        )
        assert result == "https://example.com/webhook"


# ---------------------------------------------------------------------------
# HMAC Signing Tests
# ---------------------------------------------------------------------------


class TestWebhookSigning:
    def test_hmac_signature_computation(self):
        from api.webhooks import compute_signature

        payload = '{"event":"job.completed"}'
        secret = "test-secret"
        timestamp = 1700000000

        sig = compute_signature(payload, secret, timestamp)
        assert sig.startswith("sha256=")

        # Verify manually
        message = f"{timestamp}.{payload}"
        expected = hmac.new(
            secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        assert sig == f"sha256={expected}"

    def test_signature_deterministic(self):
        from api.webhooks import compute_signature

        payload = '{"event":"job.completed","job_id":"test123"}'
        secret = "my-secret"
        timestamp = 1700000000

        sig1 = compute_signature(payload, secret, timestamp)
        sig2 = compute_signature(payload, secret, timestamp)
        assert sig1 == sig2

    def test_signature_different_for_different_payloads(self):
        from api.webhooks import compute_signature

        secret = "shared-secret"
        timestamp = 1700000000

        sig1 = compute_signature('{"event":"job.completed"}', secret, timestamp)
        sig2 = compute_signature('{"event":"job.failed"}', secret, timestamp)
        assert sig1 != sig2

    def test_signature_different_for_different_secrets(self):
        from api.webhooks import compute_signature

        payload = '{"event":"job.completed"}'
        timestamp = 1700000000

        sig1 = compute_signature(payload, "secret-a", timestamp)
        sig2 = compute_signature(payload, "secret-b", timestamp)
        assert sig1 != sig2

    def test_signature_different_for_different_timestamps(self):
        from api.webhooks import compute_signature

        payload = '{"event":"job.completed"}'
        secret = "test-secret"

        sig1 = compute_signature(payload, secret, 1700000000)
        sig2 = compute_signature(payload, secret, 1700000001)
        assert sig1 != sig2


# ---------------------------------------------------------------------------
# Payload Tests
# ---------------------------------------------------------------------------


class TestWebhookPayload:
    def _make_job(self, **kwargs):
        """Create a mock job object."""
        job = MagicMock()
        job.job_id = kwargs.get("job_id", "job_abc123")
        job.status = kwargs.get("status", "completed")
        job.source_file = kwargs.get("source_file", "test.pdf")
        job.started_at = kwargs.get(
            "started_at", datetime(2025, 1, 1, tzinfo=timezone.utc)
        )
        job.completed_at = kwargs.get(
            "completed_at", datetime(2025, 1, 1, 0, 5, tzinfo=timezone.utc)
        )
        job.processing_time = kwargs.get("processing_time", 300.0)
        job.pages_completed = kwargs.get("pages_completed", 10)
        job.total_pages = kwargs.get("total_pages", 10)
        job.error_message = kwargs.get("error_message", None)
        return job

    def test_completed_payload_schema(self):
        from api.webhooks import build_webhook_payload

        job = self._make_job(status="completed")
        payload = build_webhook_payload(job, "job.completed")

        assert payload["event"] == "job.completed"
        assert payload["job_id"] == "job_abc123"
        assert payload["status"] == "completed"
        assert payload["source_file"] == "test.pdf"
        assert payload["error_message"] is None
        assert "timestamp" in payload
        assert "processing" in payload
        assert payload["processing"]["pages_completed"] == 10

    def test_failed_payload_schema(self):
        from api.webhooks import build_webhook_payload

        job = self._make_job(
            status="failed", error_message="Pipeline exited with code 1"
        )
        payload = build_webhook_payload(job, "job.failed")

        assert payload["event"] == "job.failed"
        assert payload["status"] == "failed"
        assert payload["error_message"] == "Pipeline exited with code 1"

    def test_payload_has_required_fields(self):
        from api.webhooks import build_webhook_payload

        job = self._make_job()
        payload = build_webhook_payload(job, "job.completed")

        required_fields = {
            "event",
            "timestamp",
            "job_id",
            "status",
            "source_file",
            "processing",
        }
        assert required_fields.issubset(set(payload.keys()))

    def test_payload_processing_fields(self):
        from api.webhooks import build_webhook_payload

        job = self._make_job(
            processing_time=123.4, pages_completed=5, total_pages=8
        )
        payload = build_webhook_payload(job, "job.completed")

        proc = payload["processing"]
        assert proc["processing_time_seconds"] == 123.4
        assert proc["pages_completed"] == 5
        assert proc["total_pages"] == 8
        assert proc["started_at"] is not None
        assert proc["completed_at"] is not None

    def test_payload_serializes_to_json(self):
        from api.webhooks import build_webhook_payload

        job = self._make_job()
        payload = build_webhook_payload(job, "job.completed")
        # Should not raise
        json_str = json.dumps(payload)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["job_id"] == "job_abc123"


# ---------------------------------------------------------------------------
# Delivery Tests
# ---------------------------------------------------------------------------


class TestWebhookDelivery:
    @pytest.fixture(autouse=True)
    def _allow_test_urls(self):
        """Bypass SSRF validation in delivery tests.

        Delivery tests call deliver_webhook() directly (not via the API client),
        so the client fixture patches for WEBHOOK_ALLOW_* are not active.
        Mock validate_webhook_url to a no-op since URL validation is covered
        by TestWebhookURLValidation.
        """
        with patch(
            "api.webhooks.validate_webhook_url",
            side_effect=lambda url, **kw: url,
        ):
            yield

    def _create_job_in_db(self, session_factory, **kwargs):
        """Create a completed job in the test database."""
        from api.database import Job

        session = session_factory()
        job = Job(
            job_id=kwargs.get("job_id", "job_test123"),
            status=kwargs.get("status", "completed"),
            source_file=kwargs.get("source_file", "test.pdf"),
            priority="normal",
            webhook_url=kwargs.get(
                "webhook_url", "https://example.com/webhook"
            ),
            webhook_secret=kwargs.get("webhook_secret", "test-secret"),
            started_at=datetime(2025, 1, 1, tzinfo=timezone.utc).replace(
                tzinfo=None
            ),
            completed_at=datetime(2025, 1, 1, 0, 5, tzinfo=timezone.utc).replace(
                tzinfo=None
            ),
            processing_time=300.0,
            pages_completed=10,
            total_pages=10,
        )
        session.add(job)
        session.commit()
        job_id = kwargs.get("job_id", "job_test123")
        session.close()
        return job_id

    def test_delivery_success(self):
        from api.database import Job, get_session_factory
        from api.webhooks import deliver_webhook

        factory = get_session_factory()
        job_id = self._create_job_in_db(factory)

        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("api.webhooks._safe_opener.open", return_value=mock_response):
            deliver_webhook(job_id, factory, webhook_max_retries=3)

        session = factory()
        job = session.get(Job, job_id)
        assert job.webhook_status == "delivered"
        assert job.webhook_attempts == 1
        assert job.webhook_last_error is None
        session.close()

    def test_delivery_retries_on_error(self):
        from api.database import Job, get_session_factory
        from api.webhooks import deliver_webhook

        factory = get_session_factory()
        job_id = self._create_job_in_db(factory)

        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        call_count = 0

        def mock_urlopen(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection refused")
            return mock_response

        with patch("api.webhooks._safe_opener.open", side_effect=mock_urlopen), \
             patch("time.sleep"):
            deliver_webhook(job_id, factory, webhook_max_retries=3)

        session = factory()
        job = session.get(Job, job_id)
        assert job.webhook_status == "delivered"
        assert job.webhook_attempts == 3
        session.close()

    def test_delivery_gives_up_after_max_retries(self):
        from api.database import Job, get_session_factory
        from api.webhooks import deliver_webhook

        factory = get_session_factory()
        job_id = self._create_job_in_db(factory)

        def mock_urlopen(*args, **kwargs):
            raise ConnectionError("Connection refused")

        with patch("api.webhooks._safe_opener.open", side_effect=mock_urlopen), \
             patch("time.sleep"):
            deliver_webhook(job_id, factory, webhook_max_retries=2)

        session = factory()
        job = session.get(Job, job_id)
        assert job.webhook_status == "failed"
        assert job.webhook_attempts == 3  # 1 initial + 2 retries
        assert "Connection refused" in job.webhook_last_error
        session.close()

    def test_delivery_updates_status_in_db(self):
        from api.database import Job, get_session_factory
        from api.webhooks import deliver_webhook

        factory = get_session_factory()
        job_id = self._create_job_in_db(factory)

        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("api.webhooks._safe_opener.open", return_value=mock_response):
            deliver_webhook(job_id, factory, webhook_max_retries=0)

        session = factory()
        job = session.get(Job, job_id)
        assert job.webhook_status == "delivered"
        session.close()

    def test_no_delivery_when_url_is_none(self):
        from api.database import get_session_factory
        from api.webhooks import deliver_webhook

        factory = get_session_factory()
        job_id = self._create_job_in_db(factory, webhook_url=None)

        with patch("api.webhooks._safe_opener.open") as mock_urlopen:
            deliver_webhook(job_id, factory, webhook_max_retries=0)

        # urlopen should never be called
        mock_urlopen.assert_not_called()

    def test_delivery_sends_correct_headers(self):
        from api.database import get_session_factory
        from api.webhooks import deliver_webhook

        factory = get_session_factory()
        job_id = self._create_job_in_db(factory, webhook_secret="my-secret")

        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        captured_request = None

        def mock_urlopen(req, **kwargs):
            nonlocal captured_request
            captured_request = req
            return mock_response

        with patch("api.webhooks._safe_opener.open", side_effect=mock_urlopen):
            deliver_webhook(job_id, factory, webhook_max_retries=0)

        assert captured_request is not None
        assert captured_request.get_header("Content-type") == "application/json"
        assert (
            captured_request.get_header("X-webhook-event") == "job.completed"
        )
        assert captured_request.get_header("X-webhook-job-id") == job_id
        assert captured_request.get_header("X-webhook-signature").startswith(
            "sha256="
        )
        assert captured_request.get_header("X-webhook-timestamp") is not None

    def test_delivery_skips_non_terminal_status(self):
        from api.database import get_session_factory
        from api.webhooks import deliver_webhook

        factory = get_session_factory()
        job_id = self._create_job_in_db(factory, status="processing")

        with patch("api.webhooks._safe_opener.open") as mock_urlopen:
            deliver_webhook(job_id, factory, webhook_max_retries=0)

        mock_urlopen.assert_not_called()

    def test_delivery_no_signature_when_no_secret(self):
        from api.database import get_session_factory
        from api.webhooks import deliver_webhook

        factory = get_session_factory()
        job_id = self._create_job_in_db(factory, webhook_secret=None)

        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        captured_request = None

        def mock_urlopen(req, **kwargs):
            nonlocal captured_request
            captured_request = req
            return mock_response

        with patch("api.webhooks._safe_opener.open", side_effect=mock_urlopen):
            deliver_webhook(
                job_id,
                factory,
                webhook_max_retries=0,
                webhook_secret_default="",
            )

        assert captured_request is not None
        # No signature header when no secret
        assert captured_request.get_header("X-webhook-signature") is None


# ---------------------------------------------------------------------------
# Integration Tests (API level)
# ---------------------------------------------------------------------------


class TestWebhookIntegration:
    def test_submit_with_webhook_url_stores_url(self, client, sample_pdf):
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={
                    "priority": "normal",
                    "webhook_url": "http://localhost:9999/callback",
                    "webhook_secret": "my-secret",
                },
            )
        assert resp.status_code == 201
        job_id = resp.json()["job_id"]

        # Verify stored in DB (encrypted at rest — SEC-001)
        from api.config import decrypt_webhook_secret
        from api.database import Job, get_session_factory

        session = get_session_factory()()
        job = session.get(Job, job_id)
        assert job.webhook_url == "http://localhost:9999/callback"
        # Stored value must be encrypted, NOT plaintext
        assert job.webhook_secret != "my-secret"
        assert decrypt_webhook_secret(job.webhook_secret) == "my-secret"
        session.close()

    def test_submit_without_webhook_preserves_behavior(
        self, client, sample_pdf
    ):
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"priority": "normal"},
            )
        assert resp.status_code == 201
        job_id = resp.json()["job_id"]

        from api.database import Job, get_session_factory

        session = get_session_factory()()
        job = session.get(Job, job_id)
        assert job.webhook_url is None
        assert job.webhook_secret is None
        session.close()

    def test_retry_preserves_webhook_url(self, client, sample_pdf):
        # Submit job with webhook
        # Prevent async pipeline thread from racing this test's manual
        # status transitions (which can intermittently produce 409 on retry).
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            with open(sample_pdf, "rb") as f:
                resp = client.post(
                    "/api/v1/jobs",
                    files={"file": ("test.pdf", f, "application/pdf")},
                    data={
                        "priority": "normal",
                        "webhook_url": "http://localhost:9999/callback",
                        "webhook_secret": "retry-secret",
                    },
                )
        job_id = resp.json()["job_id"]

        # Force into failed status
        from api.database import Job, get_session_factory

        session = get_session_factory()()
        job = session.get(Job, job_id)
        job.status = "failed"
        job.error_message = "Test failure"
        session.commit()
        session.close()

        # Retry
        resp = client.post(f"/api/v1/jobs/{job_id}/retry")
        assert resp.status_code == 201
        new_job_id = resp.json()["job_id"]

        # Verify webhook preserved on retry job (encrypted at rest — SEC-001)
        from api.config import decrypt_webhook_secret

        session = get_session_factory()()
        new_job = session.get(Job, new_job_id)
        assert new_job.webhook_url == "http://localhost:9999/callback"
        # Stored value must be encrypted, NOT plaintext
        assert new_job.webhook_secret != "retry-secret"
        assert decrypt_webhook_secret(new_job.webhook_secret) == "retry-secret"
        session.close()

    def test_webhook_secret_not_in_response(self, client, sample_pdf):
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={
                    "priority": "normal",
                    "webhook_url": "http://localhost:9999/callback",
                    "webhook_secret": "super-secret",
                },
            )
        assert resp.status_code == 201
        submit_data = resp.json()

        # Secret must NOT appear in submit response
        assert "webhook_secret" not in json.dumps(submit_data)

        # Secret must NOT appear in status response
        job_id = submit_data["job_id"]
        status_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert "webhook_secret" not in json.dumps(status_data)

    def test_webhook_status_in_status_response(self, client, sample_pdf):
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"priority": "normal"},
            )
        job_id = resp.json()["job_id"]

        resp = client.get(f"/api/v1/jobs/{job_id}")
        data = resp.json()
        # webhook_status should be present (null for jobs without webhooks)
        assert "webhook_status" in data
        assert data["webhook_status"] is None

    def test_submit_rejects_invalid_webhook_url(self, client, sample_pdf):
        """Submitting with an invalid webhook URL returns 422."""
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={
                    "priority": "normal",
                    "webhook_url": "ftp://invalid-scheme.com/webhook",
                },
            )
        assert resp.status_code == 422
