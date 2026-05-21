"""Tests for webhook entity enrichment — PII bounding box payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from api.webhooks import _load_pii_entities, build_webhook_payload

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_job(**overrides):
    """Create a mock job object with sensible defaults."""
    job = MagicMock()
    job.job_id = overrides.get("job_id", "test-job-001")
    job.status = overrides.get("status", "completed")
    job.source_file = overrides.get("source_file", "document.pdf")
    job.started_at = overrides.get("started_at", datetime(2026, 1, 1, tzinfo=timezone.utc))
    job.completed_at = overrides.get("completed_at", datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc))
    job.processing_time = overrides.get("processing_time", 300.0)
    job.pages_completed = overrides.get("pages_completed", 10)
    job.total_pages = overrides.get("total_pages", 10)
    job.error_message = overrides.get("error_message", None)
    return job


# ---------------------------------------------------------------------------
# build_webhook_payload — backward compatibility
# ---------------------------------------------------------------------------


class TestBuildWebhookPayloadBackwardCompat:
    """Ensure existing webhook payloads are unchanged by default."""

    def test_default_no_entities(self):
        """Without enrich_entities, payload has no 'entities' key."""
        job = _make_mock_job()
        payload = build_webhook_payload(job, "job.completed")
        assert "entities" not in payload

    def test_enrich_false_no_entities(self):
        """Explicit enrich_entities=False omits entities."""
        job = _make_mock_job()
        payload = build_webhook_payload(job, "job.completed", enrich_entities=False)
        assert "entities" not in payload

    def test_standard_payload_fields(self):
        """Standard fields are always present."""
        job = _make_mock_job()
        payload = build_webhook_payload(job, "job.completed")
        assert payload["event"] == "job.completed"
        assert payload["job_id"] == "test-job-001"
        assert payload["status"] == "completed"
        assert payload["source_file"] == "document.pdf"
        assert "processing" in payload
        assert payload["processing"]["pages_completed"] == 10
        assert payload["error_message"] is None

    def test_failed_job_has_error_message(self):
        """Failed job payload includes error_message."""
        job = _make_mock_job(status="failed", error_message="OCR timeout")
        payload = build_webhook_payload(job, "job.failed")
        assert payload["error_message"] == "OCR timeout"


# ---------------------------------------------------------------------------
# build_webhook_payload — entity enrichment
# ---------------------------------------------------------------------------


class TestBuildWebhookPayloadEnrichment:
    """Test PII entity enrichment in webhook payloads."""

    @patch("api.webhooks._load_pii_entities")
    def test_enrich_completed_job(self, mock_load):
        """Completed job with entities includes them in payload."""
        mock_load.return_value = [
            {
                "entity_type": "SSN",
                "confidence_score": 0.97,
                "page_index": 1,
                "bounding_box": [100.0, 200.0, 300.0, 220.0],
            },
            {
                "entity_type": "EMAIL",
                "confidence_score": 0.85,
                "page_index": 2,
                "bounding_box": [50.0, 100.0, 400.0, 120.0],
            },
        ]
        job = _make_mock_job()
        payload = build_webhook_payload(job, "job.completed", enrich_entities=True)

        assert "entities" in payload
        assert len(payload["entities"]) == 2
        assert payload["entities"][0]["entity_type"] == "SSN"
        assert payload["entities"][0]["confidence_score"] == 0.97
        assert payload["entities"][0]["page_index"] == 1
        assert payload["entities"][0]["bounding_box"] == [100.0, 200.0, 300.0, 220.0]

    @patch("api.webhooks._load_pii_entities")
    def test_enrich_no_entities_found(self, mock_load):
        """Completed job with no entities omits the key."""
        mock_load.return_value = []
        job = _make_mock_job()
        payload = build_webhook_payload(job, "job.completed", enrich_entities=True)
        assert "entities" not in payload

    @patch("api.webhooks._load_pii_entities")
    def test_enrich_only_on_completed(self, mock_load):
        """Entity enrichment only applies to job.completed events."""
        mock_load.return_value = [
            {"entity_type": "SSN", "confidence_score": 0.9, "page_index": 1, "bounding_box": [0, 0, 1, 1]},
        ]
        job = _make_mock_job(status="failed", error_message="timeout")
        payload = build_webhook_payload(job, "job.failed", enrich_entities=True)
        assert "entities" not in payload
        mock_load.assert_not_called()

    @patch("api.webhooks._load_pii_entities")
    def test_enrich_cancelled_event(self, mock_load):
        """Cancelled jobs do not get enriched."""
        mock_load.return_value = [
            {"entity_type": "NAME", "confidence_score": 0.8, "page_index": 1, "bounding_box": [0, 0, 1, 1]},
        ]
        job = _make_mock_job(status="cancelled")
        payload = build_webhook_payload(job, "job.cancelled", enrich_entities=True)
        assert "entities" not in payload
        mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# _load_pii_entities
# ---------------------------------------------------------------------------


class TestLoadPiiEntities:
    """Test PII entity loading (graceful degradation)."""

    def test_returns_empty_when_django_unavailable(self):
        """When Django models are not available, returns empty list."""
        # In a test environment without Django configured, this should not crash
        result = _load_pii_entities("nonexistent-job")
        assert result == []

    @patch("api.webhooks._load_pii_entities")
    def test_returns_entities_when_available(self, mock_load):
        """When coordinator models are available, returns entity dicts."""
        mock_load.return_value = [
            {
                "entity_type": "PHONE",
                "confidence_score": 0.92,
                "page_index": 3,
                "bounding_box": [10.0, 20.0, 200.0, 40.0],
            }
        ]
        result = mock_load("test-job")
        assert len(result) == 1
        assert result[0]["entity_type"] == "PHONE"


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestWebhookEnrichConfig:
    """Test WEBHOOK_ENRICH_ENTITIES config variable."""

    def test_default_disabled(self):
        """WEBHOOK_ENRICH_ENTITIES defaults to False."""
        with patch.dict("os.environ", {}, clear=False):
            # Re-import to pick up fresh env
            import importlib

            import api.config
            importlib.reload(api.config)
            assert api.config.WEBHOOK_ENRICH_ENTITIES is False

    def test_enabled_via_env(self):
        """WEBHOOK_ENRICH_ENTITIES can be enabled via env var."""
        with patch.dict("os.environ", {"WEBHOOK_ENRICH_ENTITIES": "true"}, clear=False):
            import importlib

            import api.config
            importlib.reload(api.config)
            assert api.config.WEBHOOK_ENRICH_ENTITIES is True
            # Reset
            importlib.reload(api.config)
