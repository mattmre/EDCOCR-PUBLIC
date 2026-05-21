"""Tests for webhook DLQ write failure logging.

Verifies that when the dead-letter queue write fails during webhook
delivery, the error is logged explicitly instead of being silently dropped.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch


def _make_mock_job(job_id="job-dlq-test", status="completed"):
    """Create a mock Job object for webhook delivery."""
    job = MagicMock()
    job.job_id = job_id
    job.status = status
    job.source_file = "test.pdf"
    job.started_at = MagicMock()
    job.started_at.isoformat.return_value = "2026-04-07T00:00:00"
    job.completed_at = MagicMock()
    job.completed_at.isoformat.return_value = "2026-04-07T00:01:00"
    job.processing_time = 60.0
    job.pages_completed = 5
    job.total_pages = 5
    job.error_message = None
    job.webhook_url = "https://example.com/webhook"
    job.webhook_secret = None
    job.webhook_status = None
    job.webhook_attempts = 0
    job.webhook_last_error = None
    return job


class TestWebhookDlqWriteFailureLogging:
    """Verify DLQ write failures are logged, not silently dropped."""

    @patch("api.webhooks.validate_webhook_url")
    @patch("api.webhooks._safe_opener")
    def test_dlq_write_failure_is_logged(self, mock_opener, mock_validate, caplog):
        """When add_to_dlq raises, the error is logged at ERROR level."""
        from api.webhooks import deliver_webhook

        mock_job = _make_mock_job()
        mock_session = MagicMock()
        mock_session.get.return_value = mock_job
        mock_session_factory = MagicMock(return_value=mock_session)

        # Make the HTTP delivery fail on every attempt (to trigger DLQ path)
        mock_opener.open.side_effect = OSError("Connection refused")

        # Make the DLQ write fail
        dlq_error = RuntimeError("DLQ database locked")
        with patch(
            "api.webhook_dlq.add_to_dlq", side_effect=dlq_error,
        ):
            with caplog.at_level(logging.ERROR, logger="api.webhooks"):
                deliver_webhook(
                    "job-dlq-test",
                    mock_session_factory,
                    webhook_max_retries=0,
                )

        # Verify the DLQ write failure was logged
        dlq_messages = [
            r for r in caplog.records
            if "DLQ write failed" in r.message and "job-dlq-test" in r.message
        ]
        assert len(dlq_messages) >= 1, (
            "Expected DLQ write failure to be logged at ERROR level"
        )
        assert dlq_messages[0].levelno == logging.ERROR

    @patch("api.webhooks.validate_webhook_url")
    @patch("api.webhooks._safe_opener")
    def test_dlq_write_failure_does_not_crash_delivery(
        self, mock_opener, mock_validate
    ):
        """DLQ write failure must not crash the delivery thread."""
        from api.webhooks import deliver_webhook

        mock_job = _make_mock_job()
        mock_session = MagicMock()
        mock_session.get.return_value = mock_job
        mock_session_factory = MagicMock(return_value=mock_session)

        mock_opener.open.side_effect = OSError("Connection refused")

        with patch(
            "api.webhook_dlq.add_to_dlq",
            side_effect=RuntimeError("DLQ disk full"),
        ):
            # Should NOT raise -- delivery function handles DLQ failure gracefully
            deliver_webhook(
                "job-dlq-test",
                mock_session_factory,
                webhook_max_retries=0,
            )

        # Verify session was properly closed
        mock_session.close.assert_called_once()

    @patch("api.webhooks.validate_webhook_url")
    @patch("api.webhooks._safe_opener")
    def test_dlq_write_failure_includes_exception_detail(
        self, mock_opener, mock_validate, caplog
    ):
        """DLQ write failure log includes the exception string."""
        from api.webhooks import deliver_webhook

        mock_job = _make_mock_job()
        mock_session = MagicMock()
        mock_session.get.return_value = mock_job
        mock_session_factory = MagicMock(return_value=mock_session)

        mock_opener.open.side_effect = OSError("Connection refused")

        with patch(
            "api.webhook_dlq.add_to_dlq",
            side_effect=PermissionError("read-only filesystem"),
        ):
            with caplog.at_level(logging.ERROR, logger="api.webhooks"):
                deliver_webhook(
                    "job-dlq-test",
                    mock_session_factory,
                    webhook_max_retries=0,
                )

        dlq_messages = [
            r for r in caplog.records
            if "DLQ write failed" in r.message
        ]
        assert len(dlq_messages) >= 1
        # Verify the exception message is included
        assert "read-only filesystem" in dlq_messages[0].message
