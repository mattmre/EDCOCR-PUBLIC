"""Tests for review queue API endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.database import get_engine, reset_engine
from api.review_queue import ReviewQueue, ReviewStatus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Give each test a fresh SQLite database."""
    reset_engine()
    db_file = str(tmp_path / "test_review.db")
    with patch("api.config.DB_PATH", db_file), \
         patch("api.database.DB_PATH", db_file), \
         patch("api.review_queue.DB_PATH", db_file):
        reset_engine()
        get_engine(db_file)
        yield
        reset_engine()


@pytest.fixture()
def review_queue(tmp_path):
    """Provide a ReviewQueue sharing the test database."""
    db_file = str(tmp_path / "test_review.db")
    return ReviewQueue(db_path=db_file)


@pytest.fixture()
def client(tmp_path, review_queue):
    """FastAPI TestClient with isolated DB and review queue."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)

    with patch("api.config.SOURCE_FOLDER", str(source)), \
         patch("api.config.OUTPUT_FOLDER", str(output)), \
         patch("api.config.OCR_API_KEY", "test-key-review"), \
         patch("api.config.ALLOW_UNAUTHENTICATED", False), \
         patch("api.auth.OCR_API_KEY", "test-key-review"), \
         patch("api.auth.ALLOW_UNAUTHENTICATED", False), \
         patch("api.routers.review._review_queue", review_queue), \
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
        c = TestClient(app)
        c.headers.update({"X-API-Key": "test-key-review"})
        yield c


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _seed_items(review_queue, count=3):
    """Add some review items and return their IDs."""
    items = []
    for i in range(count):
        item = review_queue.add(
            job_id=f"job_{i:012x}",
            reason="low_confidence",
            confidence=0.3 + i * 0.1,
            quality_classification="degraded",
            metadata={"page_count": i + 1},
        )
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# GET /api/v1/review/queue
# ---------------------------------------------------------------------------


class TestListReviewQueue:
    def test_list_empty_queue(self, client):
        resp = client.get("/api/v1/review/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_pending_items(self, client, review_queue):
        _seed_items(review_queue, 5)
        resp = client.get("/api/v1/review/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 5

    def test_list_with_reason_filter(self, client, review_queue):
        review_queue.add(job_id="job_000000000001", reason="low_confidence")
        review_queue.add(job_id="job_000000000002", reason="degraded_quality")
        review_queue.add(job_id="job_000000000003", reason="low_confidence")

        resp = client.get("/api/v1/review/queue?reason=low_confidence")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert all(i["reason"] == "low_confidence" for i in data["items"])

    def test_list_with_status_filter(self, client, review_queue):
        item = review_queue.add(job_id="job_000000000001", reason="low_confidence")
        review_queue.add(job_id="job_000000000002", reason="low_confidence")
        review_queue.decide(item.review_id, ReviewStatus.APPROVED, reviewer="a")

        resp = client.get("/api/v1/review/queue?status=approved")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["status"] == "approved"

    def test_list_with_pagination(self, client, review_queue):
        _seed_items(review_queue, 10)
        resp = client.get("/api/v1/review/queue?limit=3&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 3
        assert data["total"] == 10

    def test_list_item_has_all_fields(self, client, review_queue):
        review_queue.add(
            job_id="job_000000000001",
            reason="low_confidence",
            confidence=0.42,
            quality_classification="degraded",
            metadata={"key": "value"},
        )
        resp = client.get("/api/v1/review/queue")
        item = resp.json()["items"][0]
        assert "review_id" in item
        assert "job_id" in item
        assert "reason" in item
        assert "confidence" in item
        assert "quality_classification" in item
        assert "status" in item
        assert "reviewer" in item
        assert "decision_notes" in item
        assert "created_at" in item
        assert "reviewed_at" in item
        assert "metadata" in item
        assert item["metadata"] == {"key": "value"}


# ---------------------------------------------------------------------------
# GET /api/v1/review/stats
# ---------------------------------------------------------------------------


class TestReviewStats:
    def test_stats_empty_queue(self, client):
        resp = client.get("/api/v1/review/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pending"] == 0
        assert data["total"] == 0

    def test_stats_with_items(self, client, review_queue):
        i1 = review_queue.add(job_id="job_000000000001", reason="low_confidence")
        review_queue.add(job_id="job_000000000002", reason="degraded_quality")
        review_queue.add(job_id="job_000000000003", reason="low_confidence")
        review_queue.decide(i1.review_id, ReviewStatus.APPROVED, reviewer="a")

        resp = client.get("/api/v1/review/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pending"] == 2
        assert data["approved"] == 1
        assert data["rejected"] == 0
        assert data["reprocess"] == 0
        assert data["total"] == 3


# ---------------------------------------------------------------------------
# GET /api/v1/review/{review_id}
# ---------------------------------------------------------------------------


class TestGetReviewItem:
    def test_get_existing_item(self, client, review_queue):
        item = review_queue.add(
            job_id="job_000000000001",
            reason="low_confidence",
            confidence=0.38,
        )
        resp = client.get(f"/api/v1/review/{item.review_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["review_id"] == item.review_id
        assert data["confidence"] == 0.38

    def test_get_nonexistent_item(self, client):
        resp = client.get("/api/v1/review/rev_aabbccddee01")
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "review_not_found"

    def test_get_invalid_id_format(self, client):
        resp = client.get("/api/v1/review/bad_format")
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_review_id"

    def test_get_invalid_id_too_short(self, client):
        resp = client.get("/api/v1/review/rev_abc")
        assert resp.status_code == 400

    def test_get_invalid_id_wrong_prefix(self, client):
        resp = client.get("/api/v1/review/job_aabbccddee01")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/v1/review/{review_id}/decision
# ---------------------------------------------------------------------------


class TestReviewDecision:
    def test_approve_decision(self, client, review_queue):
        item = review_queue.add(job_id="job_000000000001", reason="low_confidence")
        resp = client.post(
            f"/api/v1/review/{item.review_id}/decision",
            json={"status": "approved", "reviewer": "alice", "notes": "Looks good"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "approved"
        assert data["reviewer"] == "alice"
        assert data["decision_notes"] == "Looks good"
        assert data["reviewed_at"] != ""

    def test_reject_decision(self, client, review_queue):
        item = review_queue.add(job_id="job_000000000001", reason="low_confidence")
        resp = client.post(
            f"/api/v1/review/{item.review_id}/decision",
            json={"status": "rejected", "reviewer": "bob"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

    def test_reprocess_decision(self, client, review_queue):
        item = review_queue.add(
            job_id="job_000000000001", reason="dpi_escalation_failed"
        )
        resp = client.post(
            f"/api/v1/review/{item.review_id}/decision",
            json={"status": "reprocess", "notes": "Try 600 DPI"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "reprocess"

    def test_decision_nonexistent_item(self, client):
        resp = client.post(
            "/api/v1/review/rev_aabbccddee01/decision",
            json={"status": "approved"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "review_not_found"

    def test_decision_invalid_review_id(self, client):
        resp = client.post(
            "/api/v1/review/invalid_id/decision",
            json={"status": "approved"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_review_id"

    def test_decision_invalid_status(self, client, review_queue):
        item = review_queue.add(job_id="job_000000000001", reason="low_confidence")
        resp = client.post(
            f"/api/v1/review/{item.review_id}/decision",
            json={"status": "invalid_status"},
        )
        assert resp.status_code == 422  # Pydantic validation error (Literal)

    def test_decision_already_decided(self, client, review_queue):
        item = review_queue.add(job_id="job_000000000001", reason="low_confidence")
        # First decision succeeds
        resp1 = client.post(
            f"/api/v1/review/{item.review_id}/decision",
            json={"status": "approved", "reviewer": "alice"},
        )
        assert resp1.status_code == 200

        # Second decision should fail with 409
        resp2 = client.post(
            f"/api/v1/review/{item.review_id}/decision",
            json={"status": "rejected", "reviewer": "bob"},
        )
        assert resp2.status_code == 409
        assert resp2.json()["detail"]["error"] == "already_decided"

    def test_decision_without_reviewer(self, client, review_queue):
        item = review_queue.add(job_id="job_000000000001", reason="low_confidence")
        resp = client.post(
            f"/api/v1/review/{item.review_id}/decision",
            json={"status": "approved"},
        )
        assert resp.status_code == 200
        assert resp.json()["reviewer"] == ""

    def test_decision_updates_database(self, client, review_queue):
        item = review_queue.add(job_id="job_000000000001", reason="low_confidence")
        client.post(
            f"/api/v1/review/{item.review_id}/decision",
            json={"status": "approved", "reviewer": "charlie"},
        )
        # Verify directly in the queue
        fetched = review_queue.get(item.review_id)
        assert fetched.status == "approved"
        assert fetched.reviewer == "charlie"


# ---------------------------------------------------------------------------
# POST /api/v1/review/{review_id}/certify
# ---------------------------------------------------------------------------


class TestReviewCertify:
    def test_certify_marks_review_approved_and_certified(self, client, review_queue):
        item = review_queue.add(job_id="job_000000000001", reason="low_confidence")
        resp = client.post(
            f"/api/v1/review/{item.review_id}/certify",
            json={
                "auth_method": "hardware_token",
                "auth_token": "local-test-token",
                "notes": "operator strong-auth proof",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "approved"
        assert data["metadata"]["certified"] is True
        assert data["metadata"]["certification"]["auth_method"] == "hardware_token"
        assert "local-test-token" not in data["decision_notes"]

    def test_certify_rejects_already_decided_item(self, client, review_queue):
        item = review_queue.add(job_id="job_000000000001", reason="low_confidence")
        review_queue.decide(item.review_id, ReviewStatus.APPROVED, reviewer="alice")

        resp = client.post(
            f"/api/v1/review/{item.review_id}/certify",
            json={"auth_method": "hardware_token", "auth_token": "local-test-token"},
        )

        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "already_decided"


# ---------------------------------------------------------------------------
# Auth enforcement
# ---------------------------------------------------------------------------


class TestReviewAuth:
    def test_viewer_cannot_list_queue(self, tmp_path, review_queue):
        """Viewer role should be rejected with 403."""
        db_file = str(tmp_path / "test_review.db")

        with patch("api.config.SOURCE_FOLDER", str(tmp_path / "s")), \
             patch("api.config.OUTPUT_FOLDER", str(tmp_path / "o")), \
             patch("api.config.DB_PATH", db_file), \
             patch("api.database.DB_PATH", db_file), \
             patch("api.review_queue.DB_PATH", db_file), \
             patch("api.config.OCR_API_KEY", "test-key"), \
             patch("api.auth.OCR_API_KEY", "test-key"), \
             patch("api.config.ALLOW_UNAUTHENTICATED", False), \
             patch("api.auth.ALLOW_UNAUTHENTICATED", False), \
             patch("api.routers.review._review_queue", review_queue), \
             patch("api.job_manager.config") as mock_config:
            (tmp_path / "s").mkdir(exist_ok=True)
            (tmp_path / "o").mkdir(exist_ok=True)
            mock_config.SOURCE_FOLDER = str(tmp_path / "s")
            mock_config.OUTPUT_FOLDER = str(tmp_path / "o")
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64

            reset_engine()
            get_engine(db_file)

            from api.main import create_app
            app = create_app()
            app.state.limiter.enabled = False
            app.state.limiter.reset()

            # Patch identity to return viewer role
            from api.identity import AuthIdentity
            viewer_identity = AuthIdentity(
                subject="viewer_user", role="viewer", auth_method="apikey"
            )

            with patch("api.identity.get_identity", return_value=viewer_identity):
                test_client = TestClient(app)

                resp = test_client.get(
                    "/api/v1/review/queue",
                    headers={"X-API-Key": "test-key"},
                )
                assert resp.status_code == 403

                resp = test_client.get(
                    "/api/v1/review/stats",
                    headers={"X-API-Key": "test-key"},
                )
                assert resp.status_code == 403

            reset_engine()
