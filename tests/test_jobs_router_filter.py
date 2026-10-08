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
def client(monkeypatch):
    from fastapi.testclient import TestClient

    # api.database is imported by the shared DB fixture before this fixture
    # sets its environment.  Pin the auth module's import-time constants here
    # so this test remains deterministic when run after auth-heavy modules.
    import api.auth as api_auth

    monkeypatch.setattr(api_auth, "OCR_API_KEY", "")
    monkeypatch.setattr(api_auth, "ALLOW_UNAUTHENTICATED", True)

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
    @pytest.mark.parametrize("after,before", [
        ("2026-04-27T12:01:00", "2026-04-27T12:03:00Z"),
        ("2026-04-27T12:01:00Z", "2026-04-27T12:03:00"),
        ("2026-04-27T08:01:00-04:00", "2026-04-27T14:03:00+02:00"),
        ("2026-04-27T12:01:00", "2026-04-27T12:03:00"),
    ])
    def test_utc_date_bounds_filter_persisted_rows(self, client, seeded_jobs, after, before):
        response = client.get("/api/v1/jobs", params={
            "submitted_after": after, "submitted_before": before,
            "sort": "submitted_at_asc", "limit": 1, "offset": 1,
        })
        assert response.status_code == 200
        data = response.json()
        assert (data["total"], data["limit"], data["offset"]) == (3, 1, 1)
        assert [job["job_id"] for job in data["jobs"]] == ["job_aaaaaaaa0003"]

    @pytest.mark.parametrize("after,before", [
        ("2026-04-27T12:04:00", "2026-04-27T12:03:00Z"),
        ("2026-04-27T08:04:00-04:00", "2026-04-27T12:03:00"),
        ("0001-01-01T00:00:00+01:00", "2026-04-27T12:03:00"),
    ])
    def test_invalid_utc_date_range_is_422(self, client, seeded_jobs, after, before):
        response = client.get("/api/v1/jobs", params={
            "submitted_after": after, "submitted_before": before,
        })
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_date_range"

    def test_manager_normalizes_offset_for_internal_callers(self, client, seeded_jobs):
        from api.job_manager import get_manager

        jobs, total = get_manager().list_jobs(
            submitted_after=datetime.fromisoformat("2026-04-27T08:01:00-04:00"),
            submitted_before=datetime.fromisoformat("2026-04-27T14:03:00+02:00"),
            sort="submitted_at_asc",
        )
        assert total == 3
        assert [job.job_id for job in jobs] == [
            "job_aaaaaaaa0002", "job_aaaaaaaa0003", "job_aaaaaaaa0004",
        ]

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

    def test_tenant_key_foreign_tenant_filter_is_403(
        self, client, seeded_jobs, monkeypatch
    ):
        # r3-BE01: tenant-scoped caller passing a foreign ?tenant_id= gets an
        # explicit 403 (dashboard parity), never a silent 200 rewrite.
        import api.authz as api_authz

        monkeypatch.setattr(
            api_authz, "require_tenant_scope_or_platform", lambda request: "tenant_a"
        )
        resp = client.get("/api/v1/jobs?tenant_id=tenant_b")
        assert resp.status_code == 403
        assert resp.json()["detail"]["error"] == "tenant_mismatch"

    def test_tenant_key_own_tenant_filter_passes(
        self, client, seeded_jobs, monkeypatch
    ):
        import api.authz as api_authz

        monkeypatch.setattr(
            api_authz, "require_tenant_scope_or_platform", lambda request: "tenant_a"
        )
        resp = client.get("/api/v1/jobs?tenant_id=tenant_a")
        assert resp.status_code == 200
