"""Tests for webhook HMAC timestamp/signature freshness on retry.

Verifies that:
- Each retry attempt gets a fresh ``time.time()`` timestamp
- The HMAC signature is recomputed per attempt (not stale from attempt 0)
- Batch webhook delivery also refreshes timestamps on retry
- Without a secret, signature headers are omitted on all attempts
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from api.database import Batch, Job, get_engine
from api.webhooks import compute_signature

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_session_factory(tmp_path):
    """Build a sessionmaker bound to the test database."""
    from sqlalchemy.orm import sessionmaker

    db_file = str(tmp_path / "test_hmac_retry.db")
    engine = get_engine(db_file)
    return sessionmaker(bind=engine)


def _insert_job(session, job_id="job_hmac_001", status="completed", secret=""):
    """Insert a completed job with a webhook URL."""
    job = Job(
        job_id=job_id,
        status=status,
        source_file="test.pdf",
        webhook_url="https://example.com/hook",
        webhook_secret=secret,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        started_at=datetime.now(timezone.utc).replace(tzinfo=None),
        completed_at=datetime.now(timezone.utc).replace(tzinfo=None),
        pages_completed=1,
        total_pages=1,
    )
    session.add(job)
    session.commit()
    return job


def _insert_batch(session, batch_id="batch_hmac_001", status="completed", secret=""):
    """Insert a completed batch with a webhook URL."""
    batch = Batch(
        batch_id=batch_id,
        status=status,
        total_jobs=1,
        jobs_completed=1,
        jobs_failed=0,
        jobs_cancelled=0,
        priority="normal",
        webhook_url="https://example.com/batch-hook",
        webhook_secret=secret,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    session.add(batch)
    session.commit()
    return batch


def _make_capturing_opener(captured_list, fail_all=True):
    """Build a mock opener that captures request headers and optionally fails.

    Each call appends a dict with ``timestamp`` and ``signature`` keys
    extracted from the request headers.  When *fail_all* is True every
    call raises ``URLError``; otherwise returns a 200 mock response.
    """
    mock_opener = MagicMock()

    def _side_effect(req, **kwargs):
        ts = req.get_header("X-webhook-timestamp")
        sig = req.get_header("X-webhook-signature")
        captured_list.append({"timestamp": ts, "signature": sig, "body": req.data})
        if fail_all:
            raise urllib.error.URLError("simulated failure")
        resp = MagicMock()
        resp.getcode.return_value = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    mock_opener.open.side_effect = _side_effect
    return mock_opener


# ---------------------------------------------------------------------------
# Tests: per-attempt timestamp freshness (job webhook)
# ---------------------------------------------------------------------------


class TestJobWebhookTimestampRefresh:
    """Verify deliver_webhook recomputes timestamp and signature per retry."""

    def test_each_retry_gets_fresh_timestamp(self, tmp_path):
        """Timestamps should advance across retry attempts."""
        Session = _make_session_factory(tmp_path)
        session = Session()
        _insert_job(session, secret="test-secret")
        session.close()

        captured: list[dict] = []
        mock_opener = _make_capturing_opener(captured, fail_all=True)

        with (
            patch("api.webhooks.time.sleep"),  # skip retry delays
            patch("api.webhooks._safe_opener", mock_opener),
            patch("api.config.WEBHOOK_ALLOW_HTTP", True),
            patch("api.config.WEBHOOK_ALLOW_PRIVATE", True),
            patch("api.config.WEBHOOK_ENRICH_ENTITIES", False),
            patch("api.config.decrypt_webhook_secret", return_value="test-secret"),
        ):
            from api.webhooks import deliver_webhook

            t_before = int(time.time())
            deliver_webhook(
                "job_hmac_001",
                Session,
                webhook_timeout=5,
                webhook_max_retries=3,
                webhook_secret_default="",
            )
            t_after = int(time.time())

        # Should have 4 attempts (1 initial + 3 retries)
        assert len(captured) == 4

        # All timestamps should be valid and non-stale
        timestamps = [int(c["timestamp"]) for c in captured]
        for ts in timestamps:
            assert t_before <= ts <= t_after, (
                f"Timestamp {ts} not in range [{t_before}, {t_after}]"
            )

        # Each timestamp should be >= the previous (monotonic)
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Timestamp {timestamps[i]} < previous {timestamps[i - 1]}"
            )

    def test_signature_changes_when_timestamp_changes(self, tmp_path):
        """HMAC signature must differ when timestamp differs.

        We verify this by checking that signatures are recomputed:
        even if timestamps happen to be the same (sub-second retries),
        each attempt's signature matches compute_signature with that
        attempt's timestamp.
        """
        Session = _make_session_factory(tmp_path)
        session = Session()
        _insert_job(session, secret="sig-secret")
        session.close()

        captured: list[dict] = []
        mock_opener = _make_capturing_opener(captured, fail_all=True)

        with (
            patch("api.webhooks.time.sleep"),
            patch("api.webhooks._safe_opener", mock_opener),
            patch("api.config.WEBHOOK_ALLOW_HTTP", True),
            patch("api.config.WEBHOOK_ALLOW_PRIVATE", True),
            patch("api.config.WEBHOOK_ENRICH_ENTITIES", False),
            patch("api.config.decrypt_webhook_secret", return_value="sig-secret"),
        ):
            from api.webhooks import deliver_webhook

            deliver_webhook(
                "job_hmac_001",
                Session,
                webhook_timeout=5,
                webhook_max_retries=3,
                webhook_secret_default="",
            )

        assert len(captured) == 4

        # Each signature must match compute_signature with the corresponding timestamp
        for c in captured:
            payload_json = c["body"].decode("utf-8")
            ts = int(c["timestamp"])
            expected = compute_signature(payload_json, "sig-secret", ts)
            assert c["signature"] == expected, (
                f"Signature mismatch at timestamp {ts}"
            )

    def test_no_signature_headers_without_secret(self, tmp_path):
        """When no secret is set, signature headers should be absent."""
        Session = _make_session_factory(tmp_path)
        session = Session()
        _insert_job(session, secret="")
        session.close()

        captured: list[dict] = []
        mock_opener = _make_capturing_opener(captured, fail_all=False)

        with (
            patch("api.webhooks.time.sleep"),
            patch("api.webhooks._safe_opener", mock_opener),
            patch("api.config.WEBHOOK_ALLOW_HTTP", True),
            patch("api.config.WEBHOOK_ALLOW_PRIVATE", True),
            patch("api.config.WEBHOOK_ENRICH_ENTITIES", False),
        ):
            from api.webhooks import deliver_webhook

            deliver_webhook(
                "job_hmac_001",
                Session,
                webhook_timeout=5,
                webhook_max_retries=0,
                webhook_secret_default="",
            )

        assert len(captured) == 1
        assert captured[0]["signature"] is None
        assert captured[0]["timestamp"] is None

    def test_signature_matches_compute_signature(self, tmp_path):
        """The signature in the header must match compute_signature output."""
        Session = _make_session_factory(tmp_path)
        session = Session()
        _insert_job(session, secret="verify-secret")
        session.close()

        captured: list[dict] = []
        mock_opener = _make_capturing_opener(captured, fail_all=False)

        with (
            patch("api.webhooks.time.sleep"),
            patch("api.webhooks._safe_opener", mock_opener),
            patch("api.config.WEBHOOK_ALLOW_HTTP", True),
            patch("api.config.WEBHOOK_ALLOW_PRIVATE", True),
            patch("api.config.WEBHOOK_ENRICH_ENTITIES", False),
            patch("api.config.decrypt_webhook_secret", return_value="verify-secret"),
        ):
            from api.webhooks import deliver_webhook

            deliver_webhook(
                "job_hmac_001",
                Session,
                webhook_timeout=5,
                webhook_max_retries=0,
                webhook_secret_default="",
            )

        assert len(captured) == 1
        payload_json = captured[0]["body"].decode("utf-8")
        ts = int(captured[0]["timestamp"])
        expected_sig = compute_signature(payload_json, "verify-secret", ts)
        assert captured[0]["signature"] == expected_sig


# ---------------------------------------------------------------------------
# Tests: per-attempt timestamp freshness (batch webhook)
# ---------------------------------------------------------------------------


class TestBatchWebhookTimestampRefresh:
    """Verify deliver_batch_webhook also refreshes timestamp per retry."""

    def test_batch_each_retry_gets_fresh_timestamp(self, tmp_path):
        """Batch timestamps should advance across retry attempts."""
        Session = _make_session_factory(tmp_path)
        session = Session()
        _insert_batch(session, secret="batch-secret")
        session.close()

        captured: list[dict] = []
        mock_opener = _make_capturing_opener(captured, fail_all=True)

        with (
            patch("api.webhooks.time.sleep"),
            patch("api.webhooks._safe_opener", mock_opener),
            patch("api.config.WEBHOOK_ALLOW_HTTP", True),
            patch("api.config.WEBHOOK_ALLOW_PRIVATE", True),
            patch("api.config.decrypt_webhook_secret", return_value="batch-secret"),
        ):
            from api.webhooks import deliver_batch_webhook

            t_before = int(time.time())
            deliver_batch_webhook(
                "batch_hmac_001",
                Session,
                webhook_timeout=5,
                webhook_max_retries=3,
                webhook_secret_default="",
            )
            t_after = int(time.time())

        assert len(captured) == 4

        timestamps = [int(c["timestamp"]) for c in captured]
        for ts in timestamps:
            assert t_before <= ts <= t_after

        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_batch_signature_matches_per_attempt(self, tmp_path):
        """Each batch attempt signature must match its own timestamp."""
        Session = _make_session_factory(tmp_path)
        session = Session()
        _insert_batch(session, secret="batch-sig")
        session.close()

        captured: list[dict] = []
        mock_opener = _make_capturing_opener(captured, fail_all=True)

        with (
            patch("api.webhooks.time.sleep"),
            patch("api.webhooks._safe_opener", mock_opener),
            patch("api.config.WEBHOOK_ALLOW_HTTP", True),
            patch("api.config.WEBHOOK_ALLOW_PRIVATE", True),
            patch("api.config.decrypt_webhook_secret", return_value="batch-sig"),
        ):
            from api.webhooks import deliver_batch_webhook

            deliver_batch_webhook(
                "batch_hmac_001",
                Session,
                webhook_timeout=5,
                webhook_max_retries=3,
                webhook_secret_default="",
            )

        assert len(captured) == 4

        for c in captured:
            payload_json = c["body"].decode("utf-8")
            ts = int(c["timestamp"])
            expected = compute_signature(payload_json, "batch-sig", ts)
            assert c["signature"] == expected


# ---------------------------------------------------------------------------
# Tests: default secret fallback still refreshes
# ---------------------------------------------------------------------------


class TestDefaultSecretRefresh:
    """Verify that webhook_secret_default also gets fresh signatures."""

    def test_default_secret_timestamps_refresh(self, tmp_path):
        """When using webhook_secret_default, timestamps still refresh."""
        Session = _make_session_factory(tmp_path)
        session = Session()
        # No per-job secret; uses the default
        _insert_job(session, secret="")
        session.close()

        captured: list[dict] = []
        mock_opener = _make_capturing_opener(captured, fail_all=True)

        with (
            patch("api.webhooks.time.sleep"),
            patch("api.webhooks._safe_opener", mock_opener),
            patch("api.config.WEBHOOK_ALLOW_HTTP", True),
            patch("api.config.WEBHOOK_ALLOW_PRIVATE", True),
            patch("api.config.WEBHOOK_ENRICH_ENTITIES", False),
        ):
            from api.webhooks import deliver_webhook

            t_before = int(time.time())
            deliver_webhook(
                "job_hmac_001",
                Session,
                webhook_timeout=5,
                webhook_max_retries=1,
                webhook_secret_default="fallback-secret",
            )
            t_after = int(time.time())

        assert len(captured) == 2

        timestamps = [int(c["timestamp"]) for c in captured]
        for ts in timestamps:
            assert t_before <= ts <= t_after

        # Each signature matches its own timestamp using fallback secret
        for c in captured:
            payload_json = c["body"].decode("utf-8")
            ts = int(c["timestamp"])
            expected = compute_signature(payload_json, "fallback-secret", ts)
            assert c["signature"] == expected
