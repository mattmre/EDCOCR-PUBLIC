"""Tests for D6 server-side jobs filtering on ``GET /api/v1/jobs``.

Covers status multi-select, tenant isolation (gotcha #80), date ranges,
the ``q`` substring filter (with LIKE-injection guard), sort orders, and
top-level ``total`` count.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

# These tests import the API stack which requires sqlalchemy/fastapi.
# Skip cleanly if those are unavailable (mirrors sdk-tests CI pattern).
pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_API_KEY", "test-key-d6")
    monkeypatch.setenv("OCR_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("OCR_SOURCE_DIR", str(tmp_path / "source"))
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "true")
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "source").mkdir(exist_ok=True)


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from api.main import create_app

    app = create_app()
    app.state.limiter.enabled = False
    app.state.limiter.reset()
    return TestClient(app)


@pytest.fixture()
def seeded_jobs(client):
    """Seed a deterministic set of Job rows for filter tests."""
    from api.database import Job, get_session_factory

    session = get_session_factory()()
    base = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc).replace(tzinfo=None)
    rows = [
        Job(
            job_id="job_aaaaaaaa0001",
            source_file="alpha_invoice.pdf",
status="completed",
            tenant_id=None,
            created_at=base,
            processing_time=12.0,
        ),
        Job(
            job_id="job_aaaaaaaa0002",
            source_file="bravo_contract.pdf",
status="failed",
            tenant_id=None,
            created_at=base + timedelta(minutes=1),
            processing_time=3.0,
        ),
        Job(
            job_id="job_aaaaaaaa0003",
            source_file="charlie_50%-discount.pdf",
status="processing",
            tenant_id=None,
            created_at=base + timedelta(minutes=2),
            processing_time=None,
        ),
        Job(
            job_id="job_aaaaaaaa0004",
            source_file="delta_report.pdf",
status="completed",
            tenant_id="tenant_a",
            created_at=base + timedelta(minutes=3),
            processing_time=42.0,
        ),
        Job(
            job_id="job_aaaaaaaa0005",
            source_file="echo_brief.pdf",
status="completed",
            tenant_id="tenant_b",
            created_at=base + timedelta(minutes=4),
            processing_time=7.5,
        ),
    ]
    for row in rows:
        session.merge(row)
    session.commit()
    session.close()
    return rows


class TestStatusFilter:
    def test_single_status_legacy(self, client, seeded_jobs):
        resp = client.get("/api/v1/jobs?status=completed")
        assert resp.status_code == 200
        data = resp.json()
        statuses = {j["status"] for j in data["jobs"]}
        assert statuses == {"completed"}
        assert data["total"] == 3

    def test_multi_status(self, client, seeded_jobs):
        resp = client.get("/api/v1/jobs?status=failed&status=processing")
        assert resp.status_code == 200
        data = resp.json()
        statuses = {j["status"] for j in data["jobs"]}
        assert statuses == {"failed", "processing"}
        assert data["total"] == 2

    def test_invalid_status_returns_422(self, client, seeded_jobs):
        resp = client.get("/api/v1/jobs?status=not_a_real_status")
        assert resp.status_code == 422


class TestSubstringSearch:
    def test_q_matches_source_file(self, client, seeded_jobs):
        resp = client.get("/api/v1/jobs?q=alpha")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["jobs"][0]["source_file"] == "alpha_invoice.pdf"

    def test_q_matches_job_id(self, client, seeded_jobs):
        resp = client.get("/api/v1/jobs?q=aaaaaaaa0002")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["jobs"][0]["job_id"] == "job_aaaaaaaa0002"

    def test_q_like_injection_percent_is_escaped(self, client, seeded_jobs):
        # A naive LIKE would treat "50%" as a wildcard.  With proper
        # escaping the literal "50%" only matches charlie_50%-discount.
        resp = client.get("/api/v1/jobs?q=50%25")  # URL-encoded %
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["jobs"][0]["job_id"] == "job_aaaaaaaa0003"

    def test_q_like_injection_underscore_is_escaped(self, client, seeded_jobs):
        # "_invoice" should match literal underscore -- only the alpha row.
        resp = client.get("/api/v1/jobs?q=_invoice")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["jobs"][0]["job_id"] == "job_aaaaaaaa0001"


class TestDateRange:
    def test_submitted_after_filters(self, client, seeded_jobs):
        cutoff = "2026-04-27T12:02:30"
        resp = client.get(f"/api/v1/jobs?submitted_after={cutoff}")
        assert resp.status_code == 200
        data = resp.json()
        # Rows after 12:02:30 -> indices 3,4 (delta, echo).
        ids = {j["job_id"] for j in data["jobs"]}
        assert ids == {"job_aaaaaaaa0004", "job_aaaaaaaa0005"}
        assert data["total"] == 2

    def test_submitted_before_filters(self, client, seeded_jobs):
        cutoff = "2026-04-27T12:01:30"
        resp = client.get(f"/api/v1/jobs?submitted_before={cutoff}")
        assert resp.status_code == 200
        data = resp.json()
        ids = {j["job_id"] for j in data["jobs"]}
        assert ids == {"job_aaaaaaaa0001", "job_aaaaaaaa0002"}
        assert data["total"] == 2

    def test_inverted_range_returns_422(self, client, seeded_jobs):
        resp = client.get(
            "/api/v1/jobs?submitted_after=2026-04-27T12:05:00"
            "&submitted_before=2026-04-27T12:00:00"
        )
        assert resp.status_code == 422


class TestSort:
    def test_submitted_at_asc(self, client, seeded_jobs):
        resp = client.get("/api/v1/jobs?sort=submitted_at_asc")
        assert resp.status_code == 200
        data = resp.json()
        ids = [j["job_id"] for j in data["jobs"]]
        assert ids == [
            "job_aaaaaaaa0001",
            "job_aaaaaaaa0002",
            "job_aaaaaaaa0003",
            "job_aaaaaaaa0004",
            "job_aaaaaaaa0005",
        ]

    def test_duration_desc_pushes_nulls_last(self, client, seeded_jobs):
        resp = client.get("/api/v1/jobs?sort=duration_desc")
        assert resp.status_code == 200
        data = resp.json()
        ids = [j["job_id"] for j in data["jobs"]]
        # 42.0 > 12.0 > 7.5 > 3.0 > NULL
        assert ids[0] == "job_aaaaaaaa0004"
        assert ids[-1] == "job_aaaaaaaa0003"

    def test_invalid_sort_returns_422(self, client, seeded_jobs):
        resp = client.get("/api/v1/jobs?sort=garbage")
        assert resp.status_code == 422


class TestTotalAndPagination:
    def test_total_reflects_pre_pagination_count(self, client, seeded_jobs):
        resp = client.get("/api/v1/jobs?limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["jobs"]) == 2
        assert data["total"] == 5

    def test_total_with_filter(self, client, seeded_jobs):
        resp = client.get("/api/v1/jobs?status=completed&limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["jobs"]) == 1
        assert data["total"] == 3


class TestTenantIsolation:
    """Gotcha #80: non-admin tenant scope is enforced silently (404/scope rewrite)."""

    def test_anonymous_caller_drops_foreign_tenant_filter(self, client, seeded_jobs):
        # Unauthenticated mode -- caller_tenant_id is None, not platform-admin,
        # so explicit tenant_id is dropped and all jobs are returned.
        resp = client.get("/api/v1/jobs?tenant_id=tenant_a")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
