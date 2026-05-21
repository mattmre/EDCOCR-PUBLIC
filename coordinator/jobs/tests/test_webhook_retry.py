"""Tests for webhook delivery retry logic.

Run with: cd coordinator && python -m pytest jobs/tests/test_webhook_retry.py -v
"""

import sys
import types
import uuid
from unittest.mock import MagicMock, patch

import pytest

django = pytest.importorskip("django")

from django.test import TestCase  # noqa: E402

from jobs.models import Job  # noqa: E402
from jobs.tasks import (  # noqa: E402
    _WEBHOOK_RETRY_DELAYS,
    _deliver_webhook_with_retry,
    _send_webhook,
)


def _make_fake_ssrf(mock_opener=None, validate_side_effect=None):
    """Build fake ocr_distributed.ssrf module for sys.modules injection."""
    fake_pkg = types.ModuleType("ocr_distributed")
    fake_ssrf = types.ModuleType("ocr_distributed.ssrf")
    fake_ssrf.safe_opener = mock_opener or MagicMock()
    if validate_side_effect:
        fake_ssrf.validate_webhook_url = MagicMock(side_effect=validate_side_effect)
    else:
        fake_ssrf.validate_webhook_url = MagicMock()
    fake_pkg.ssrf = fake_ssrf
    return {
        "ocr_distributed": fake_pkg,
        "ocr_distributed.ssrf": fake_ssrf,
    }


class TestDeliverWebhookWithRetry(TestCase):
    """Unit tests for _deliver_webhook_with_retry."""

    @patch("jobs.tasks.time.sleep")
    def test_succeeds_on_first_attempt(self, mock_sleep):
        """Webhook delivered on first try -- no retries needed."""
        mock_opener = MagicMock()
        fake_modules = _make_fake_ssrf(mock_opener=mock_opener)
        with patch.dict(sys.modules, fake_modules):
            result = _deliver_webhook_with_retry(
                "https://example.com/hook",
                b'{"event":"job.completed"}',
                {"Content-Type": "application/json"},
            )
        assert result is True
        assert mock_opener.open.call_count == 1
        mock_sleep.assert_not_called()

    @patch("jobs.tasks.time.sleep")
    def test_succeeds_on_second_attempt(self, mock_sleep):
        """Transient failure on first attempt, success on second."""
        mock_opener = MagicMock()
        mock_opener.open.side_effect = [ConnectionError("timeout"), None]
        fake_modules = _make_fake_ssrf(mock_opener=mock_opener)
        with patch.dict(sys.modules, fake_modules):
            result = _deliver_webhook_with_retry(
                "https://example.com/hook",
                b'{"event":"job.completed"}',
                {"Content-Type": "application/json"},
            )
        assert result is True
        assert mock_opener.open.call_count == 2
        mock_sleep.assert_called_once_with(_WEBHOOK_RETRY_DELAYS[0])

    @patch("jobs.tasks.time.sleep")
    def test_fails_after_all_attempts(self, mock_sleep):
        """All 3 attempts fail -- returns False."""
        mock_opener = MagicMock()
        mock_opener.open.side_effect = ConnectionError("network down")
        fake_modules = _make_fake_ssrf(mock_opener=mock_opener)
        with patch.dict(sys.modules, fake_modules):
            result = _deliver_webhook_with_retry(
                "https://example.com/hook",
                b'{"event":"job.completed"}',
                {"Content-Type": "application/json"},
                max_attempts=3,
            )
        assert result is False
        assert mock_opener.open.call_count == 3
        # Two sleeps (between attempt 1->2 and 2->3)
        assert mock_sleep.call_count == 2

    @patch("jobs.tasks.time.sleep")
    def test_custom_max_attempts(self, mock_sleep):
        """Respects custom max_attempts parameter."""
        mock_opener = MagicMock()
        mock_opener.open.side_effect = ConnectionError("fail")
        fake_modules = _make_fake_ssrf(mock_opener=mock_opener)
        with patch.dict(sys.modules, fake_modules):
            result = _deliver_webhook_with_retry(
                "https://example.com/hook",
                b"{}",
                {},
                max_attempts=1,
            )
        assert result is False
        assert mock_opener.open.call_count == 1
        mock_sleep.assert_not_called()


class TestSendWebhookIntegration(TestCase):
    """Integration tests for _send_webhook with retry."""

    def _make_job(self, **kwargs):
        defaults = {
            "job_id": uuid.uuid4(),
            "source_file": "test.pdf",
            "status": Job.Status.COMPLETED,
            "webhook_url": "https://example.com/hook",
            "webhook_secret": "test-secret",
        }
        defaults.update(kwargs)
        return Job.objects.create(**defaults)

    @patch("jobs.tasks._deliver_webhook_with_retry", return_value=True)
    def test_delivered_status_on_success(self, mock_deliver):
        """Job webhook_status set to 'delivered' on success."""
        fake_modules = _make_fake_ssrf()
        job = self._make_job()
        with patch.dict(sys.modules, fake_modules):
            _send_webhook(job, {"pages": 5})
        job.refresh_from_db()
        assert job.webhook_status == "delivered"
        mock_deliver.assert_called_once()

    @patch("jobs.tasks._deliver_webhook_with_retry", return_value=False)
    def test_failed_status_after_exhausted_retries(self, mock_deliver):
        """Job webhook_status set to 'failed' when all retries exhausted."""
        fake_modules = _make_fake_ssrf()
        job = self._make_job()
        with patch.dict(sys.modules, fake_modules):
            _send_webhook(job, {"pages": 5})
        job.refresh_from_db()
        assert job.webhook_status == "failed"

    def test_url_validation_failure_skips_delivery(self):
        """URL validation failure immediately marks failed, no delivery attempted."""
        fake_modules = _make_fake_ssrf(
            validate_side_effect=ValueError("blocked"),
        )
        job = self._make_job()
        with (
            patch.dict(sys.modules, fake_modules),
            patch("jobs.tasks._deliver_webhook_with_retry") as mock_deliver,
        ):
            _send_webhook(job, {"pages": 5})
            mock_deliver.assert_not_called()
        job.refresh_from_db()
        assert job.webhook_status == "failed"


class TestWebhookRetryDelays(TestCase):
    """Verify retry delay constants are sensible."""

    def test_delays_are_increasing(self):
        assert len(_WEBHOOK_RETRY_DELAYS) >= 2
        for i in range(1, len(_WEBHOOK_RETRY_DELAYS)):
            assert _WEBHOOK_RETRY_DELAYS[i] >= _WEBHOOK_RETRY_DELAYS[i - 1]

    def test_delays_are_positive(self):
        for delay in _WEBHOOK_RETRY_DELAYS:
            assert delay > 0
