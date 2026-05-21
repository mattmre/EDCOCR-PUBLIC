"""Tests for Server-Sent Events (SSE) streaming endpoint.

Covers the GET /api/v1/jobs/{job_id}/stream endpoint including:
- Job existence validation (404 for missing jobs)
- SSE event format (event: <type>\\ndata: <json>\\n\\n)
- Terminal-state event sequences (completed, failed, cancelled)
- Progress event emission
- Invalid job ID rejection
- Stream timeout behaviour
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.database import Job, get_engine, get_session_factory, reset_engine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    """FastAPI TestClient with isolated DB, no auth, fast SSE polling."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with (
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.config.OCR_API_KEY", ""),
        patch("api.auth.OCR_API_KEY", ""),
        patch("api.config.ALLOW_UNAUTHENTICATED", True),
        patch("api.auth.ALLOW_UNAUTHENTICATED", True),
        patch.dict(os.environ, {
            "SSE_POLL_INTERVAL": "0.05",
            "SSE_STREAM_TIMEOUT": "5",
        }),
    ):
        from api.main import create_app

        app = create_app()
        app.state.limiter.enabled = False
        yield TestClient(app)


@pytest.fixture()
def auth_client(tmp_path):
    """FastAPI TestClient with API key auth enabled."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with (
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.config.OCR_API_KEY", "test-secret-key"),
        patch("api.auth.OCR_API_KEY", "test-secret-key"),
        patch("api.config.ALLOW_UNAUTHENTICATED", False),
        patch("api.auth.ALLOW_UNAUTHENTICATED", False),
        patch.dict(os.environ, {
            "SSE_POLL_INTERVAL": "0.05",
            "SSE_STREAM_TIMEOUT": "5",
        }),
    ):
        from api.main import create_app

        app = create_app()
        app.state.limiter.enabled = False
        yield TestClient(app)


def _create_job(
    job_id: str,
    status: str = "submitted",
    *,
    total_pages: int | None = None,
    pages_completed: int = 0,
    current_stage: str | None = None,
    error_message: str | None = None,
    result_path: str | None = None,
    processing_time: float | None = None,
):
    """Insert a Job record directly via the session factory."""
    sf = get_session_factory()
    session = sf()
    try:
        job = Job(
            job_id=job_id,
            source_file="test.pdf",
            status=status,
            priority="normal",
            total_pages=total_pages,
            pages_completed=pages_completed,
            current_stage=current_stage,
            error_message=error_message,
            result_path=result_path,
            processing_time=processing_time,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(job)
        session.commit()
    finally:
        session.close()


def _parse_sse_stream(response) -> list[dict]:
    """Parse SSE frames from a streaming response into structured dicts.

    Each returned dict has keys: ``event`` and ``data`` (parsed JSON).
    """
    events: list[dict] = []
    current_event: str | None = None
    for line in response.iter_lines():
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_str = line.split(":", 1)[1].strip()
            events.append({
                "event": current_event,
                "data": json.loads(data_str),
            })
            current_event = None
    return events


# ---------------------------------------------------------------------------
# SSE event formatter unit tests
# ---------------------------------------------------------------------------

class TestSSEEventFormat:
    """Unit tests for the _sse_event helper function."""

    def test_format_basic(self):
        from api.routers.jobs import _sse_event

        event = _sse_event("status", {"job_id": "j1", "status": "processing"})
        assert event.startswith("event: status\n")
        assert "data: " in event
        assert event.endswith("\n\n")

    def test_data_is_valid_json(self):
        from api.routers.jobs import _sse_event

        event = _sse_event("progress", {"job_id": "j1", "percent": 42.5})
        data_line = [line for line in event.split("\n") if line.startswith("data:")][0]
        payload = json.loads(data_line.split(":", 1)[1].strip())
        assert payload["job_id"] == "j1"
        assert payload["percent"] == 42.5

    def test_all_event_types(self):
        from api.routers.jobs import _sse_event

        for event_type in ("status", "progress", "result", "error", "done"):
            frame = _sse_event(event_type, {"ok": True})
            assert f"event: {event_type}" in frame
            assert '"ok": true' in frame


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------

class TestSSEStreamEndpoint:
    """Integration tests for GET /api/v1/jobs/{job_id}/stream."""

    def test_nonexistent_job_returns_404(self, client):
        r = client.get("/api/v1/jobs/job_aabbccddeeff/stream")
        assert r.status_code == 404

    def test_invalid_job_id_returns_400(self, client):
        r = client.get("/api/v1/jobs/bad-id/stream")
        assert r.status_code == 400

    def test_completed_job_stream(self, client):
        """A completed job should emit status, result, and done events."""
        _create_job(
            "job_aabbccddeeff",
            status="completed",
            result_path="/tmp/out",
            processing_time=12.5,
            pages_completed=10,
        )

        with client.stream("GET", "/api/v1/jobs/job_aabbccddeeff/stream") as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            assert resp.headers.get("cache-control") == "no-cache"
            assert resp.headers.get("x-accel-buffering") == "no"

            events = _parse_sse_stream(resp)

        event_types = [e["event"] for e in events]
        assert "status" in event_types
        assert "result" in event_types
        assert "done" in event_types

        # Verify result event payload
        result_events = [e for e in events if e["event"] == "result"]
        assert len(result_events) == 1
        result_data = result_events[0]["data"]
        assert result_data["job_id"] == "job_aabbccddeeff"
        assert result_data["status"] == "completed"
        assert result_data["result_path"] == "/tmp/out"
        assert result_data["processing_time_seconds"] == 12.5
        assert result_data["pages_processed"] == 10

    def test_failed_job_stream(self, client):
        """A failed job should emit status, error, and done events."""
        _create_job(
            "job_aabbccddeeff",
            status="failed",
            error_message="Pipeline crashed",
        )

        with client.stream("GET", "/api/v1/jobs/job_aabbccddeeff/stream") as resp:
            assert resp.status_code == 200
            events = _parse_sse_stream(resp)

        event_types = [e["event"] for e in events]
        assert "status" in event_types
        assert "error" in event_types
        assert "done" in event_types

        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["data"]["error"] == "Pipeline crashed"
        assert error_events[0]["data"]["status"] == "failed"

    def test_cancelled_job_stream(self, client):
        """A cancelled job should emit status and done events (no error)."""
        _create_job("job_aabbccddeeff", status="cancelled")

        with client.stream("GET", "/api/v1/jobs/job_aabbccddeeff/stream") as resp:
            assert resp.status_code == 200
            events = _parse_sse_stream(resp)

        event_types = [e["event"] for e in events]
        assert "status" in event_types
        assert "done" in event_types
        # cancelled should NOT emit an error event
        assert "error" not in event_types

    def test_status_event_includes_previous(self, client):
        """The status event should carry previous_status (None for first)."""
        _create_job("job_aabbccddeeff", status="completed")

        with client.stream("GET", "/api/v1/jobs/job_aabbccddeeff/stream") as resp:
            events = _parse_sse_stream(resp)

        status_events = [e for e in events if e["event"] == "status"]
        assert len(status_events) >= 1
        assert status_events[0]["data"]["previous_status"] is None
        assert status_events[0]["data"]["status"] == "completed"

    def test_progress_event_with_pages(self, client):
        """A processing job with pages should emit a progress event."""
        _create_job(
            "job_aabbccddeeff",
            status="processing",
            total_pages=20,
            pages_completed=5,
            current_stage="ocr",
        )

        collected_events: list[dict] = []

        # We need to break out of the stream after receiving the first
        # progress event, since this job stays in 'processing' forever.
        with client.stream("GET", "/api/v1/jobs/job_aabbccddeeff/stream") as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    evt = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    collected_events.append({"event": evt, "data": data})
                    # Break after receiving at least one progress event
                    if evt == "progress":
                        break

        progress_events = [e for e in collected_events if e["event"] == "progress"]
        assert len(progress_events) >= 1
        p = progress_events[0]["data"]
        assert p["pages_completed"] == 5
        assert p["total_pages"] == 20
        assert p["percent"] == 25.0
        assert p["current_stage"] == "ocr"

    def test_auth_required_without_key(self, auth_client):
        """SSE endpoint should require authentication like other job endpoints."""
        _create_job("job_aabbccddeeff", status="completed")

        r = auth_client.get("/api/v1/jobs/job_aabbccddeeff/stream")
        assert r.status_code == 401

    def test_auth_accepted_with_key(self, auth_client):
        """SSE endpoint should accept valid API key."""
        _create_job("job_aabbccddeeff", status="completed")

        with auth_client.stream(
            "GET",
            "/api/v1/jobs/job_aabbccddeeff/stream",
            headers={"X-API-Key": "test-secret-key"},
        ) as resp:
            assert resp.status_code == 200
            events = _parse_sse_stream(resp)

        event_types = [e["event"] for e in events]
        assert "done" in event_types

    def test_stream_response_headers(self, client):
        """Verify SSE-specific response headers for proxy compatibility."""
        _create_job("job_aabbccddeeff", status="completed")

        with client.stream("GET", "/api/v1/jobs/job_aabbccddeeff/stream") as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            assert resp.headers.get("cache-control") == "no-cache"
            assert resp.headers.get("x-accel-buffering") == "no"
            # Drain the stream
            for _ in resp.iter_lines():
                pass


class TestSSEStreamTimeout:
    """Tests for SSE stream timeout behaviour."""

    def test_timeout_emits_error_and_done(self, tmp_path):
        """When the stream times out, error + done events should be emitted."""
        reset_engine()
        db_file = str(tmp_path / "test_timeout.db")
        with (
            patch("api.config.DB_PATH", db_file),
            patch("api.database.DB_PATH", db_file),
            patch("api.config.SOURCE_FOLDER", str(tmp_path / "src")),
            patch("api.config.OUTPUT_FOLDER", str(tmp_path / "out")),
            patch("api.config.OCR_API_KEY", ""),
            patch("api.auth.OCR_API_KEY", ""),
            patch("api.config.ALLOW_UNAUTHENTICATED", True),
            patch("api.auth.ALLOW_UNAUTHENTICATED", True),
            patch.dict(os.environ, {
                "SSE_POLL_INTERVAL": "0.05",
                "SSE_STREAM_TIMEOUT": "0.15",  # Very short timeout
            }),
        ):
            reset_engine()
            (tmp_path / "src").mkdir(exist_ok=True)
            (tmp_path / "out").mkdir(exist_ok=True)
            get_engine(db_file)

            # Create a job that stays in 'submitted' (non-terminal).
            _create_job("job_aabbccddeeff", status="submitted")

            from api.main import create_app

            app = create_app()
            app.state.limiter.enabled = False
            tc = TestClient(app)

            with tc.stream("GET", "/api/v1/jobs/job_aabbccddeeff/stream") as resp:
                assert resp.status_code == 200
                events = _parse_sse_stream(resp)

            event_types = [e["event"] for e in events]
            assert "error" in event_types
            assert "done" in event_types

            # The last error should mention timeout
            error_events = [e for e in events if e["event"] == "error"]
            assert any("timeout" in e["data"].get("error", "").lower() for e in error_events)

        reset_engine()
