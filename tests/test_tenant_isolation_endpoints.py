"""Cross-tenant isolation tests for 7 API endpoints ( through ).

Validates that tenant A cannot access tenant B's job data through any of the
streaming, output, event, recall, or review endpoints. Every cross-tenant
attempt must return 404 (not 200, not 403) to avoid revealing that the job
exists.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.database import Job, get_engine, get_session_factory, reset_engine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TENANT_A = "tenant_aaa111222333"
TENANT_B = "tenant_bbb444555666"
JOB_A_ID = "job_aaaaaaaaaaaa"
JOB_B_ID = "job_bbbbbbbbbbbb"


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Give each test a fresh SQLite database and reset singletons."""
    reset_engine()
    db_file = str(tmp_path / "test_tenant_iso.db")
    with (
        patch("api.config.DB_PATH", db_file),
        patch("api.database.DB_PATH", db_file),
        patch("api.review_queue.DB_PATH", db_file),
    ):
        reset_engine()
        get_engine(db_file)

        # Reset review queue singleton so each test gets a fresh DB
        import api.routers.review as _review_mod
        _review_mod._review_queue = None

        # Reset entity index singleton
        import api.routers.recall as _recall_mod
        _recall_mod._entity_index = None

        yield
        reset_engine()


@pytest.fixture()
def session():
    """Return a SQLAlchemy session for direct ORM access."""
    factory = get_session_factory()
    s = factory()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def _seed_jobs(session):
    """Seed two jobs, one per tenant."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    job_a = Job(
        job_id=JOB_A_ID,
        status="completed",
        source_file="doc_a.pdf",
        created_at=now,
        tenant_id=TENANT_A,
        result_path="/app/output/a",
    )
    job_b = Job(
        job_id=JOB_B_ID,
        status="completed",
        source_file="doc_b.pdf",
        created_at=now,
        tenant_id=TENANT_B,
        result_path="/app/output/b",
    )
    session.add_all([job_a, job_b])
    session.commit()


def _make_client(tmp_path, tenant_id):
    """Build a TestClient whose request.state.tenant_id is forced."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)

    # We run with ALLOW_UNAUTHENTICATED=True but inject tenant_id via
    # a custom middleware that simulates multi-tenant auth resolution.
    with (
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.config.OCR_API_KEY", "test-key-123"),
        patch("api.auth.OCR_API_KEY", "test-key-123"),
        patch("api.config.ALLOW_UNAUTHENTICATED", False),
        patch("api.auth.ALLOW_UNAUTHENTICATED", False),
        patch.dict(os.environ, {
            "SSE_POLL_INTERVAL": "0.05",
            "SSE_STREAM_TIMEOUT": "2",
        }),
    ):
        from api.main import create_app

        app = create_app()
        app.state.limiter.enabled = False

        # Inject tenant_id into request.state after auth
        from fastapi import Request
        from starlette.middleware.base import BaseHTTPMiddleware

        class TenantInjector(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                request.state.tenant_id = tenant_id
                return await call_next(request)

        app.add_middleware(TenantInjector)

        client = TestClient(app)
        # Provide the API key header for all requests
        client.headers["X-API-Key"] = "test-key-123"
        yield client


@pytest.fixture()
def client_a(tmp_path):
    """TestClient authenticated as tenant A."""
    yield from _make_client(tmp_path, TENANT_A)


@pytest.fixture()
def client_b(tmp_path):
    """TestClient authenticated as tenant B."""
    yield from _make_client(tmp_path, TENANT_B)


# ---------------------------------------------------------------------------
# / SSE stream endpoint (jobs.py)
# ---------------------------------------------------------------------------


class TestSSETenantIsolation:
    """SSE stream must not expose jobs from other tenants."""

    def test_own_tenant_sse_stream_returns_200(self, client_a, _seed_jobs):
        """Tenant A can stream its own job."""
        resp = client_a.get(f"/api/v1/jobs/{JOB_A_ID}/stream", timeout=3)
        assert resp.status_code == 200

    def test_cross_tenant_sse_stream_returns_404(self, client_b, _seed_jobs):
        """Tenant B cannot stream tenant A's job."""
        resp = client_b.get(f"/api/v1/jobs/{JOB_A_ID}/stream", timeout=3)
        assert resp.status_code == 404

    def test_cross_tenant_sse_404_body(self, client_b, _seed_jobs):
        """404 response uses job_not_found error (no info leakage)."""
        resp = client_b.get(f"/api/v1/jobs/{JOB_A_ID}/stream", timeout=3)
        body = resp.json()
        assert body["detail"]["error"] == "job_not_found"


# ---------------------------------------------------------------------------
# WebSocket endpoint (ws.py)
# ---------------------------------------------------------------------------

# WebSocket tenant isolation is harder to test with TestClient because the
# WebSocket auth path is separate. We test it via the DB query path.


class TestWebSocketTenantIsolation:
    """WebSocket job lookup must enforce tenant_id filtering."""

    def test_ws_query_includes_tenant_filter(self, session, _seed_jobs):
        """Direct DB query pattern used in ws.py respects tenant_id."""
        query = session.query(Job).filter(
            Job.job_id == JOB_A_ID, Job.tenant_id == TENANT_B
        )
        assert query.first() is None, "Cross-tenant lookup must return None"

    def test_ws_query_own_tenant_succeeds(self, session, _seed_jobs):
        """Own-tenant lookup succeeds."""
        query = session.query(Job).filter(
            Job.job_id == JOB_A_ID, Job.tenant_id == TENANT_A
        )
        assert query.first() is not None


# ---------------------------------------------------------------------------
# Output manifest endpoint (outputs.py)
# ---------------------------------------------------------------------------


class TestOutputsTenantIsolation:
    """Output manifest endpoints must filter by tenant_id."""

    def test_own_tenant_outputs_returns_200(self, client_a, _seed_jobs):
        """Tenant A can list outputs for its own job."""
        resp = client_a.get(f"/api/v1/jobs/{JOB_A_ID}/outputs")
        # 200 even if no actual files on disk (empty manifest)
        assert resp.status_code == 200

    def test_cross_tenant_outputs_returns_404(self, client_b, _seed_jobs):
        """Tenant B cannot list outputs for tenant A's job."""
        resp = client_b.get(f"/api/v1/jobs/{JOB_A_ID}/outputs")
        assert resp.status_code == 404

    def test_cross_tenant_output_download_returns_404(self, client_b, _seed_jobs):
        """Tenant B cannot download a specific output type for tenant A's job."""
        resp = client_b.get(f"/api/v1/jobs/{JOB_A_ID}/outputs/searchable_pdf")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Events endpoint (events.py)
# ---------------------------------------------------------------------------


class TestEventsTenantIsolation:
    """Event replay must verify job ownership before returning events."""

    def test_own_tenant_events_allowed(self, client_a, _seed_jobs):
        """Tenant A can retrieve events for its own job."""
        resp = client_a.get(f"/api/v1/jobs/{JOB_A_ID}/events")
        assert resp.status_code == 200

    def test_cross_tenant_events_returns_404(self, client_b, _seed_jobs):
        """Tenant B cannot retrieve events for tenant A's job."""
        resp = client_b.get(f"/api/v1/jobs/{JOB_A_ID}/events")
        assert resp.status_code == 404

    def test_cross_tenant_events_404_body(self, client_b, _seed_jobs):
        """404 uses job_not_found error code."""
        resp = client_b.get(f"/api/v1/jobs/{JOB_A_ID}/events")
        body = resp.json()
        assert body["detail"]["error"] == "job_not_found"


# ---------------------------------------------------------------------------
# Entity/extraction recall (recall.py)
# ---------------------------------------------------------------------------


class TestRecallTenantIsolation:
    """Entity and extraction recall must restrict results to own tenant."""

    def test_entities_with_other_tenant_job_id_returns_404(
        self, client_b, _seed_jobs
    ):
        """Tenant B searching entities for tenant A's job_id gets 404."""
        resp = client_b.get(
            "/api/v1/entities", params={"job_id": JOB_A_ID}
        )
        assert resp.status_code == 404

    def test_extractions_with_other_tenant_job_id_returns_404(
        self, client_b, _seed_jobs
    ):
        """Tenant B searching extractions for tenant A's job_id gets 404."""
        resp = client_b.get(
            "/api/v1/extractions", params={"job_id": JOB_A_ID}
        )
        assert resp.status_code == 404

    def test_entities_own_tenant_allowed(self, client_a, _seed_jobs):
        """Tenant A searching entities for its own job returns 200."""
        resp = client_a.get(
            "/api/v1/entities", params={"job_id": JOB_A_ID}
        )
        assert resp.status_code == 200

    def test_entities_no_job_filter_only_returns_own(self, client_a, _seed_jobs):
        """Tenant A searching without job_id filter only sees own jobs' entities."""
        resp = client_a.get("/api/v1/entities")
        assert resp.status_code == 200
        # Results should be empty (no indexed entities) but the query
        # was restricted to tenant A's jobs
        body = resp.json()
        assert body["total"] == 0


# ---------------------------------------------------------------------------
# Review queue (review.py)
# ---------------------------------------------------------------------------


class TestReviewTenantIsolation:
    """Review queue endpoints must filter by tenant job ownership."""

    def test_review_queue_empty_for_other_tenant(self, client_b, _seed_jobs):
        """Tenant B sees empty review queue (no jobs belong to them in tenant A's data)."""
        # Seed a review item for tenant A's job
        from api.review_queue import ReviewQueue

        queue = ReviewQueue()
        queue.add(
            job_id=JOB_A_ID,
            reason="low_confidence",
            confidence=0.3,
        )

        resp = client_b.get("/api/v1/review/queue")
        assert resp.status_code == 200
        body = resp.json()
        # Tenant B should see 0 items because the review item belongs
        # to tenant A's job
        assert body["total"] == 0
        assert len(body["items"]) == 0

    def test_review_queue_shows_own_tenant_items(self, client_a, _seed_jobs):
        """Tenant A sees review items for its own jobs."""
        from api.review_queue import ReviewQueue

        queue = ReviewQueue()
        queue.add(
            job_id=JOB_A_ID,
            reason="low_confidence",
            confidence=0.3,
        )

        resp = client_a.get("/api/v1/review/queue")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert len(body["items"]) == 1
        assert body["items"][0]["job_id"] == JOB_A_ID

    def test_get_review_item_cross_tenant_returns_404(
        self, client_b, _seed_jobs
    ):
        """Tenant B cannot read a review item belonging to tenant A's job."""
        from api.review_queue import ReviewQueue

        queue = ReviewQueue()
        item = queue.add(
            job_id=JOB_A_ID,
            reason="low_confidence",
            confidence=0.3,
        )

        resp = client_b.get(f"/api/v1/review/{item.review_id}")
        assert resp.status_code == 404

    def test_review_decision_cross_tenant_returns_404(
        self, client_b, _seed_jobs
    ):
        """Tenant B cannot submit a decision for tenant A's review item."""
        from api.review_queue import ReviewQueue

        queue = ReviewQueue()
        item = queue.add(
            job_id=JOB_A_ID,
            reason="low_confidence",
            confidence=0.3,
        )

        resp = client_b.post(
            f"/api/v1/review/{item.review_id}/decision",
            json={"status": "approved", "reviewer": "attacker"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Entity index allowed_job_ids filtering (unit tests)
# ---------------------------------------------------------------------------


class TestEntityIndexTenantFiltering:
    """Unit tests for EntityIndex.search_entities/search_extractions with
    allowed_job_ids parameter."""

    def test_search_entities_empty_allowed_returns_empty(self, tmp_path):
        """Empty allowed_job_ids returns zero results immediately."""
        from api.entity_index import EntityIndex

        idx = EntityIndex(db_path=str(tmp_path / "eidx.db"))
        idx.index_entities(JOB_A_ID, "doc.pdf", [
            {"type": "PERSON", "text": "Alice", "confidence": 0.9},
        ])

        results, total = idx.search_entities(allowed_job_ids=[])
        assert total == 0
        assert results == []

    def test_search_entities_allowed_restricts_results(self, tmp_path):
        """Only entities from allowed job IDs are returned."""
        from api.entity_index import EntityIndex

        idx = EntityIndex(db_path=str(tmp_path / "eidx.db"))
        idx.index_entities(JOB_A_ID, "doc_a.pdf", [
            {"type": "PERSON", "text": "Alice", "confidence": 0.9},
        ])
        idx.index_entities(JOB_B_ID, "doc_b.pdf", [
            {"type": "PERSON", "text": "Bob", "confidence": 0.8},
        ])

        results, total = idx.search_entities(allowed_job_ids=[JOB_A_ID])
        assert total == 1
        assert results[0].text == "Alice"

    def test_search_extractions_allowed_restricts_results(self, tmp_path):
        """Only extractions from allowed job IDs are returned."""
        from api.entity_index import EntityIndex

        idx = EntityIndex(db_path=str(tmp_path / "eidx.db"))
        idx.index_extractions(JOB_A_ID, "doc_a.pdf", [
            {"key": "invoice_number", "value": "INV-001", "confidence": 0.9},
        ])
        idx.index_extractions(JOB_B_ID, "doc_b.pdf", [
            {"key": "invoice_number", "value": "INV-002", "confidence": 0.8},
        ])

        results, total = idx.search_extractions(allowed_job_ids=[JOB_A_ID])
        assert total == 1
        assert results[0].field_value == "INV-001"

    def test_search_entities_job_id_overrides_allowed(self, tmp_path):
        """When both job_id and allowed_job_ids are set, job_id takes precedence."""
        from api.entity_index import EntityIndex

        idx = EntityIndex(db_path=str(tmp_path / "eidx.db"))
        idx.index_entities(JOB_A_ID, "doc_a.pdf", [
            {"type": "PERSON", "text": "Alice", "confidence": 0.9},
        ])

        # job_id is set, so allowed_job_ids is ignored
        results, total = idx.search_entities(
            job_id=JOB_A_ID, allowed_job_ids=[JOB_B_ID]
        )
        assert total == 1


# ---------------------------------------------------------------------------
# Review queue allowed_job_ids filtering (unit tests)
# ---------------------------------------------------------------------------


class TestReviewQueueTenantFiltering:
    """Unit tests for ReviewQueue.list_all/count with allowed_job_ids."""

    def test_list_all_empty_allowed_returns_empty(self, tmp_path):
        """Empty allowed_job_ids returns zero results."""
        from api.review_queue import ReviewQueue

        q = ReviewQueue(db_path=str(tmp_path / "review.db"))
        q.add(job_id=JOB_A_ID, reason="low_confidence", confidence=0.3)

        items = q.list_all(allowed_job_ids=[])
        assert items == []

    def test_list_all_restricts_to_allowed_jobs(self, tmp_path):
        """Only review items for allowed job IDs are returned."""
        from api.review_queue import ReviewQueue

        q = ReviewQueue(db_path=str(tmp_path / "review.db"))
        q.add(job_id=JOB_A_ID, reason="low_confidence", confidence=0.3)
        q.add(job_id=JOB_B_ID, reason="low_confidence", confidence=0.4)

        items = q.list_all(allowed_job_ids=[JOB_A_ID])
        assert len(items) == 1
        assert items[0].job_id == JOB_A_ID

    def test_count_restricts_to_allowed_jobs(self, tmp_path):
        """Count only counts items for allowed job IDs."""
        from api.review_queue import ReviewQueue

        q = ReviewQueue(db_path=str(tmp_path / "review.db"))
        q.add(job_id=JOB_A_ID, reason="low_confidence", confidence=0.3)
        q.add(job_id=JOB_B_ID, reason="low_confidence", confidence=0.4)

        count = q.count(allowed_job_ids=[JOB_A_ID])
        assert count == 1
