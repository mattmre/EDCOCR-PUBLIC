"""Tests for M-22: DELETE /api/v1/admin/tenants/{tenant_id}/purge endpoint.

Validates that:
- Purge endpoint exists in the admin router
- LITIGATION_HOLD returns 409
- Successful deletion returns expected response shape with counts
- Missing tenant_id returns 404
- Invalid tenant_id format returns 400
- Requires platform admin permission
- Unauthenticated requests are rejected
- Cascading deletion removes jobs, usage records, and API keys
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.database import (
    Job,
    TenantApiKey,
    UsageRecord,
    get_session_factory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def sample_tenant(session):
    """Create and return a sample active tenant."""
    from api.tenant_manager import create_tenant

    return create_tenant(
        name="Purge Test Corp",
        tier="standard",
        max_concurrent_jobs=4,
        max_pages_per_month=10000,
        admin_email="admin@purgetest.example",
        session=session,
    )


@pytest.fixture()
def platform_admin_client(tmp_path, sample_tenant, session, monkeypatch):
    """FastAPI TestClient with multi-tenancy enabled and a platform-admin key."""
    monkeypatch.delenv("LITIGATION_HOLD", raising=False)

    from api.tenant_manager import create_api_key

    _, raw_key = create_api_key(
        sample_tenant.tenant_id,
        name="platform-admin-key",
        permissions=["submit", "read", "admin", "platform_admin"],
        session=session,
    )

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with (
        patch("api.config.ENABLE_MULTITENANCY", True),
        patch("api.main.ENABLE_MULTITENANCY", True),
        patch("api.auth.ENABLE_MULTITENANCY", True),
        patch("api.routers.admin.ENABLE_MULTITENANCY", True),
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.config.OCR_API_KEY", ""),
        patch("api.auth.OCR_API_KEY", ""),
        patch("api.config.ALLOW_UNAUTHENTICATED", False),
        patch("api.auth.ALLOW_UNAUTHENTICATED", False),
        patch("api.job_manager.config") as mock_config,
    ):
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
        yield TestClient(app), raw_key, sample_tenant


@pytest.fixture()
def tenant_admin_client(tmp_path, sample_tenant, session, monkeypatch):
    """FastAPI TestClient with a non-platform tenant admin key (scoped to own tenant)."""
    monkeypatch.delenv("LITIGATION_HOLD", raising=False)

    from api.tenant_manager import create_api_key

    _, raw_key = create_api_key(
        sample_tenant.tenant_id,
        name="tenant-admin-key",
        permissions=["submit", "read", "admin"],
        session=session,
    )

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with (
        patch("api.config.ENABLE_MULTITENANCY", True),
        patch("api.main.ENABLE_MULTITENANCY", True),
        patch("api.auth.ENABLE_MULTITENANCY", True),
        patch("api.routers.admin.ENABLE_MULTITENANCY", True),
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.config.OCR_API_KEY", ""),
        patch("api.auth.OCR_API_KEY", ""),
        patch("api.config.ALLOW_UNAUTHENTICATED", False),
        patch("api.auth.ALLOW_UNAUTHENTICATED", False),
        patch("api.job_manager.config") as mock_config,
    ):
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
        yield TestClient(app), raw_key, sample_tenant


# ===========================================================================
# Tests: endpoint exists in router
# ===========================================================================


class TestPurgeEndpointExists:
    """Verify the purge endpoint is registered and reachable."""

    def test_purge_endpoint_responds(self, platform_admin_client):
        """DELETE /api/v1/admin/tenants/{id}/purge should not return 405."""
        client, key, tenant = platform_admin_client
        resp = client.delete(
            f"/api/v1/admin/tenants/{tenant.tenant_id}/purge",
            headers={"X-API-Key": key},
        )
        # Should be 200 (success) not 405 (method not allowed) or 404 (route not found)
        assert resp.status_code in (200, 409), f"Unexpected status: {resp.status_code}"


# ===========================================================================
# Tests: LITIGATION_HOLD returns 409
# ===========================================================================


class TestLitigationHold:
    """LITIGATION_HOLD env var should block purge with 409."""

    def test_litigation_hold_true_returns_409(self, platform_admin_client, monkeypatch):
        """When LITIGATION_HOLD=true, purge returns 409 Conflict."""
        monkeypatch.setenv("LITIGATION_HOLD", "true")
        client, key, tenant = platform_admin_client
        resp = client.delete(
            f"/api/v1/admin/tenants/{tenant.tenant_id}/purge",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 409
        body = resp.json()
        assert "litigation_hold" in body["detail"]["error"]

    def test_litigation_hold_one_returns_409(self, platform_admin_client, monkeypatch):
        """LITIGATION_HOLD=1 also triggers the 409 block."""
        monkeypatch.setenv("LITIGATION_HOLD", "1")
        client, key, tenant = platform_admin_client
        resp = client.delete(
            f"/api/v1/admin/tenants/{tenant.tenant_id}/purge",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 409

    def test_litigation_hold_yes_returns_409(self, platform_admin_client, monkeypatch):
        """LITIGATION_HOLD=yes also triggers the 409 block."""
        monkeypatch.setenv("LITIGATION_HOLD", "yes")
        client, key, tenant = platform_admin_client
        resp = client.delete(
            f"/api/v1/admin/tenants/{tenant.tenant_id}/purge",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 409

    def test_litigation_hold_unset_allows_purge(self, platform_admin_client, monkeypatch):
        """When LITIGATION_HOLD is unset, purge proceeds normally."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        client, key, tenant = platform_admin_client
        resp = client.delete(
            f"/api/v1/admin/tenants/{tenant.tenant_id}/purge",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 200


# ===========================================================================
# Tests: successful deletion returns expected shape
# ===========================================================================


class TestSuccessfulPurge:
    """Successful purge returns correct response with deletion counts."""

    def test_purge_returns_expected_shape(self, platform_admin_client, monkeypatch):
        """Response includes tenant_id and deleted counts."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        client, key, tenant = platform_admin_client
        resp = client.delete(
            f"/api/v1/admin/tenants/{tenant.tenant_id}/purge",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["tenant_id"] == tenant.tenant_id
        assert "deleted" in body
        assert "jobs" in body["deleted"]
        assert "usage_records" in body["deleted"]
        assert "api_keys" in body["deleted"]

    def test_purge_deletes_jobs(self, platform_admin_client, session, monkeypatch):
        """Purge removes all jobs belonging to the tenant."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        client, key, tenant = platform_admin_client
        tid = tenant.tenant_id  # capture before purge deletes the ORM object

        # Create jobs for this tenant
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for i in range(3):
            job = Job(
                job_id=f"purge-test-job-{i}",
                status="completed",
                source_file=f"test{i}.pdf",
                created_at=now,
                tenant_id=tid,
            )
            session.add(job)
        session.commit()

        resp = client.delete(
            f"/api/v1/admin/tenants/{tid}/purge",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"]["jobs"] == 3

        # Verify jobs are gone from the database
        session.expire_all()
        remaining = session.query(Job).filter(Job.tenant_id == tid).count()
        assert remaining == 0

    def test_purge_deletes_usage_records(self, platform_admin_client, session, monkeypatch):
        """Purge removes all usage records for the tenant."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        client, key, tenant = platform_admin_client

        # Create usage records
        for period in ("2026-01", "2026-02", "2026-03"):
            usage = UsageRecord(
                tenant_id=tenant.tenant_id,
                period=period,
                jobs_submitted=10,
                pages_processed=100,
            )
            session.add(usage)
        session.commit()

        resp = client.delete(
            f"/api/v1/admin/tenants/{tenant.tenant_id}/purge",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"]["usage_records"] == 3

    def test_purge_deletes_api_keys(self, platform_admin_client, session, monkeypatch):
        """Purge removes all API keys for the tenant."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        client, key, tenant = platform_admin_client
        tid = tenant.tenant_id  # capture before purge deletes the ORM object

        # The platform_admin_client fixture already creates one key;
        # create additional keys
        from api.tenant_manager import create_api_key

        create_api_key(
            tid,
            name="extra-key-1",
            permissions=["submit"],
            session=session,
        )
        create_api_key(
            tid,
            name="extra-key-2",
            permissions=["read"],
            session=session,
        )

        pre_count = (
            session.query(TenantApiKey)
            .filter(TenantApiKey.tenant_id == tid)
            .count()
        )
        assert pre_count >= 3  # at least admin key + 2 extras

        resp = client.delete(
            f"/api/v1/admin/tenants/{tid}/purge",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"]["api_keys"] == pre_count

        # Verify keys are gone
        session.expire_all()
        remaining = (
            session.query(TenantApiKey)
            .filter(TenantApiKey.tenant_id == tid)
            .count()
        )
        assert remaining == 0

    def test_purge_removes_tenant_record(self, platform_admin_client, session, monkeypatch):
        """After purge, the tenant record no longer exists in the database."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        client, key, tenant = platform_admin_client
        tid = tenant.tenant_id  # capture before purge deletes the ORM object

        resp = client.delete(
            f"/api/v1/admin/tenants/{tid}/purge",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 200

        session.expire_all()
        from api.tenant_manager import get_tenant

        db_tenant = get_tenant(tid, session=session)
        assert db_tenant is None


# ===========================================================================
# Tests: missing tenant returns 404
# ===========================================================================


class TestMissingTenant:
    """Purge returns 404 for nonexistent tenant."""

    def test_nonexistent_tenant_returns_404(self, platform_admin_client, monkeypatch):
        """Purge on a non-existing tenant_id returns 404."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        client, key, _ = platform_admin_client
        resp = client.delete(
            "/api/v1/admin/tenants/tenant_000000000000/purge",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 404
        body = resp.json()
        assert "not_found" in body["detail"]["error"]


# ===========================================================================
# Tests: invalid tenant_id format returns 400
# ===========================================================================


class TestInvalidTenantId:
    """Purge returns 400 for malformed tenant_id."""

    def test_invalid_format_returns_400(self, platform_admin_client, monkeypatch):
        """Malformed tenant_id returns 400 Bad Request."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        client, key, _ = platform_admin_client
        resp = client.delete(
            "/api/v1/admin/tenants/bad-id/purge",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 400


# ===========================================================================
# Tests: auth requirements
# ===========================================================================


class TestPurgeAuth:
    """Purge endpoint enforces authentication and authorization."""

    def test_requires_platform_admin(self, tenant_admin_client, monkeypatch):
        """Non-platform admin gets 403."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        client, key, tenant = tenant_admin_client
        resp = client.delete(
            f"/api/v1/admin/tenants/{tenant.tenant_id}/purge",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 403

    def test_no_auth_rejected(self, platform_admin_client, monkeypatch):
        """Request without auth header returns 401."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        client, _, tenant = platform_admin_client
        resp = client.delete(
            f"/api/v1/admin/tenants/{tenant.tenant_id}/purge",
        )
        assert resp.status_code == 401


# ===========================================================================
# Tests: purge_tenant_data unit tests
# ===========================================================================


class TestPurgeTenantDataUnit:
    """Unit tests for tenant_manager.purge_tenant_data()."""

    def test_purge_returns_counts(self, sample_tenant, session, monkeypatch):
        """purge_tenant_data returns deletion counts dict."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        from api.tenant_manager import purge_tenant_data

        result = purge_tenant_data(sample_tenant.tenant_id, session=session)
        assert result is not None
        assert result["tenant_id"] == sample_tenant.tenant_id
        assert "deleted" in result
        assert result["deleted"]["jobs"] == 0  # no jobs seeded
        assert result["deleted"]["usage_records"] == 0
        # At least 0 keys (none created in this fixture at the unit level)
        assert isinstance(result["deleted"]["api_keys"], int)

    def test_purge_nonexistent_returns_none(self, session, monkeypatch):
        """purge_tenant_data returns None for missing tenant."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        from api.tenant_manager import purge_tenant_data

        result = purge_tenant_data("tenant_000000000000", session=session)
        assert result is None

    def test_purge_removes_all_data(self, sample_tenant, session, monkeypatch):
        """purge_tenant_data removes jobs, usage, keys, and tenant."""
        monkeypatch.delenv("LITIGATION_HOLD", raising=False)
        from api.tenant_manager import create_api_key, purge_tenant_data

        # Seed data
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        job = Job(
            job_id="unit-purge-job",
            status="completed",
            source_file="test.pdf",
            created_at=now,
            tenant_id=sample_tenant.tenant_id,
        )
        session.add(job)

        usage = UsageRecord(
            tenant_id=sample_tenant.tenant_id,
            period="2026-03",
            jobs_submitted=5,
        )
        session.add(usage)
        session.commit()

        create_api_key(
            sample_tenant.tenant_id,
            name="test-key",
            permissions=["submit"],
            session=session,
        )

        result = purge_tenant_data(sample_tenant.tenant_id, session=session)
        assert result["deleted"]["jobs"] == 1
        assert result["deleted"]["usage_records"] == 1
        assert result["deleted"]["api_keys"] == 1

        # Tenant record should be gone
        from api.database import Tenant

        tenant = session.get(Tenant, sample_tenant.tenant_id)
        assert tenant is None
