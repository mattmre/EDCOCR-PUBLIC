"""Tests for multi-tenancy expansion: delete endpoint, tenant dashboard, cost bridge."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.database import (
    Job,
    TenantApiKey,
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
        name="Expansion Corp",
        tier="standard",
        max_concurrent_jobs=4,
        max_pages_per_month=10000,
        admin_email="admin@expansion.test",
        session=session,
    )


@pytest.fixture()
def platform_admin_client(tmp_path, sample_tenant, session):
    """FastAPI TestClient with multi-tenancy enabled and a platform-admin key."""
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
def tenant_admin_client(tmp_path, sample_tenant, session):
    """FastAPI TestClient with a non-platform tenant admin key (scoped to own tenant)."""
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


@pytest.fixture()
def dashboard_client(tmp_path, sample_tenant, session):
    """FastAPI TestClient with both dashboard and multi-tenancy enabled."""
    from api.tenant_manager import create_api_key

    _, raw_key = create_api_key(
        sample_tenant.tenant_id,
        name="dashboard-admin-key",
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
        patch.dict("os.environ", {"ENABLE_DASHBOARD": "true", "ENABLE_MULTITENANCY": "true"}),
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
# DELETE tenant endpoint
# ===========================================================================


class TestDeleteTenant:
    """Tests for DELETE /api/v1/admin/tenants/{tenant_id}."""

    def test_delete_tenant_soft_deletes(self, platform_admin_client, session):
        """DELETE sets tenant status to 'deleted' (not physically removed)."""
        client, key, tenant = platform_admin_client
        resp = client.delete(
            f"/api/v1/admin/tenants/{tenant.tenant_id}",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "deleted"
        assert body["tenant_id"] == tenant.tenant_id

        # Verify in database — record still exists (expire stale cache first)
        session.expire_all()
        from api.tenant_manager import get_tenant

        db_tenant = get_tenant(tenant.tenant_id, session=session)
        assert db_tenant is not None
        assert db_tenant.status == "deleted"

    def test_delete_tenant_revokes_all_keys(self, platform_admin_client, session):
        """DELETE revokes all active API keys for the tenant."""
        client, key, tenant = platform_admin_client

        # Create additional keys before deletion
        from api.tenant_manager import create_api_key

        extra_key_id, _ = create_api_key(
            tenant.tenant_id,
            name="extra-key",
            permissions=["submit", "read"],
            session=session,
        )

        resp = client.delete(
            f"/api/v1/admin/tenants/{tenant.tenant_id}",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 200

        # All keys for this tenant should be revoked
        keys = (
            session.query(TenantApiKey)
            .filter(TenantApiKey.tenant_id == tenant.tenant_id)
            .all()
        )
        for k in keys:
            assert k.status == "revoked", f"Key {k.key_id} should be revoked"

    def test_delete_tenant_not_found(self, platform_admin_client):
        """DELETE returns 404 for nonexistent tenant."""
        client, key, _ = platform_admin_client
        resp = client.delete(
            "/api/v1/admin/tenants/tenant_000000000000",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 404

    def test_delete_tenant_invalid_id_format(self, platform_admin_client):
        """DELETE returns 400 for malformed tenant_id."""
        client, key, _ = platform_admin_client
        resp = client.delete(
            "/api/v1/admin/tenants/bad-id",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 400

    def test_delete_tenant_requires_platform_admin(self, tenant_admin_client):
        """DELETE requires platform_admin permission, not just admin."""
        client, key, tenant = tenant_admin_client
        resp = client.delete(
            f"/api/v1/admin/tenants/{tenant.tenant_id}",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 403

    def test_delete_tenant_no_auth_rejected(self, platform_admin_client):
        """DELETE without auth header is rejected."""
        client, _, tenant = platform_admin_client
        resp = client.delete(
            f"/api/v1/admin/tenants/{tenant.tenant_id}",
        )
        assert resp.status_code == 401


# ===========================================================================
# delete_tenant() unit tests
# ===========================================================================


class TestDeleteTenantManager:
    """Unit tests for tenant_manager.delete_tenant()."""

    def test_delete_returns_updated_tenant(self, sample_tenant, session):
        from api.tenant_manager import delete_tenant

        result = delete_tenant(sample_tenant.tenant_id, session=session)
        assert result is not None
        assert result.status == "deleted"

    def test_delete_nonexistent_returns_none(self, session):
        from api.tenant_manager import delete_tenant

        result = delete_tenant("tenant_000000000000", session=session)
        assert result is None

    def test_delete_revokes_keys(self, sample_tenant, session):
        from api.tenant_manager import create_api_key, delete_tenant

        k1_id, _ = create_api_key(
            sample_tenant.tenant_id,
            permissions=["submit"],
            session=session,
        )
        k2_id, _ = create_api_key(
            sample_tenant.tenant_id,
            permissions=["read"],
            session=session,
        )

        delete_tenant(sample_tenant.tenant_id, session=session)

        key1 = session.get(TenantApiKey, k1_id)
        key2 = session.get(TenantApiKey, k2_id)
        assert key1.status == "revoked"
        assert key2.status == "revoked"

    def test_delete_idempotent(self, sample_tenant, session):
        """Deleting an already-deleted tenant still succeeds."""
        from api.tenant_manager import delete_tenant

        delete_tenant(sample_tenant.tenant_id, session=session)
        result = delete_tenant(sample_tenant.tenant_id, session=session)
        assert result is not None
        assert result.status == "deleted"


# ===========================================================================
# Tenant-scoped dashboard
# ===========================================================================


class TestTenantDashboard:
    """Tests for GET /api/v1/dashboard/tenant/{tenant_id}."""

    def test_tenant_dashboard_returns_filtered_data(self, dashboard_client, session):
        """Tenant dashboard returns job metrics scoped to the tenant."""
        client, key, tenant = dashboard_client

        # Insert some jobs for this tenant
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for i in range(3):
            job = Job(
                job_id=f"job-dash-{i}",
                status="completed",
                source_file=f"test{i}.pdf",
                created_at=now - timedelta(seconds=60),
                completed_at=now,
                pages_completed=5,
                processing_time=2.5,
                tenant_id=tenant.tenant_id,
            )
            session.add(job)

        # Insert a job for a different tenant (should be excluded)
        other_job = Job(
            job_id="job-other-tenant",
            status="completed",
            source_file="other.pdf",
            created_at=now - timedelta(seconds=60),
            completed_at=now,
            pages_completed=100,
            processing_time=10.0,
            tenant_id="tenant_999999999999",
        )
        session.add(other_job)
        session.commit()

        resp = client.get(
            f"/api/v1/dashboard/tenant/{tenant.tenant_id}",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["tenant_id"] == tenant.tenant_id
        assert body["jobs"]["completed"] == 3
        assert body["jobs"]["total"] == 3
        # The other tenant's job should not be counted
        assert body["jobs"]["completed"] != 4

    def test_tenant_dashboard_empty(self, dashboard_client):
        """Tenant dashboard returns zeroes when no jobs exist."""
        client, key, tenant = dashboard_client
        resp = client.get(
            f"/api/v1/dashboard/tenant/{tenant.tenant_id}",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["jobs"]["total"] == 0
        assert body["jobs"]["completed"] == 0

    def test_tenant_dashboard_invalid_id(self, dashboard_client):
        """Tenant dashboard returns 400 for invalid tenant_id."""
        client, key, _ = dashboard_client
        resp = client.get(
            "/api/v1/dashboard/tenant/bad-id",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 400

    def test_tenant_dashboard_requires_multitenancy(self, tmp_path, sample_tenant, session):
        """Tenant dashboard returns 404 when ENABLE_MULTITENANCY is off."""
        from api.tenant_manager import create_api_key

        _, raw_key = create_api_key(
            sample_tenant.tenant_id,
            name="no-mt-key",
            permissions=["submit", "read", "admin", "platform_admin"],
            session=session,
        )

        source = tmp_path / "src"
        output = tmp_path / "out"
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
            patch.dict("os.environ", {"ENABLE_DASHBOARD": "true", "ENABLE_MULTITENANCY": "false"}),
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
            tc = TestClient(app)

            resp = tc.get(
                f"/api/v1/dashboard/tenant/{sample_tenant.tenant_id}",
                headers={"X-API-Key": raw_key},
            )
            assert resp.status_code == 404


# ===========================================================================
# Cost bridge
# ===========================================================================


class TestCostBridge:
    """Tests for cost_tracking_bridge and cost-bridge admin endpoint."""

    def test_cost_bridge_returns_expected_format(self, sample_tenant, session):
        """build_cost_tracking_bridge returns cost_tracking.py-compatible schema."""
        from api.tenant_manager import build_cost_tracking_bridge
        from api.usage import _get_or_create_usage

        # Seed some usage data
        record = _get_or_create_usage(
            sample_tenant.tenant_id, "2026-03", session
        )
        record.pages_processed = 100
        record.storage_bytes_used = 1024**3  # 1 GiB
        record.api_calls = 50
        record.processing_seconds = 3600.0
        record.jobs_submitted = 10
        session.commit()

        with patch("api.usage._current_period", return_value="2026-03"):
            report = build_cost_tracking_bridge(
                sample_tenant.tenant_id, session=session
            )

        assert report is not None
        assert report["tenant_id"] == sample_tenant.tenant_id

        # Verify usage section matches cost_tracking.TenantUsage schema
        usage = report["usage"]
        assert usage["pages_processed"] == 100
        assert usage["gpu_seconds"] == 3600.0
        assert usage["storage_bytes"] == 1024**3
        assert usage["api_calls"] == 50
        assert usage["jobs_submitted"] == 10

        # Verify cost section matches cost_tracking.TenantUsage.estimated_cost() schema
        cost = report["cost"]
        assert "page_cost" in cost
        assert "gpu_cost" in cost
        assert "storage_cost" in cost
        assert "api_cost" in cost
        assert "total_cost" in cost
        assert cost["currency"] == "USD"
        assert cost["total_cost"] > 0

    def test_cost_bridge_no_usage_returns_none(self, sample_tenant, session):
        """build_cost_tracking_bridge returns None when no usage exists."""
        from api.tenant_manager import build_cost_tracking_bridge

        result = build_cost_tracking_bridge(
            sample_tenant.tenant_id, session=session
        )
        assert result is None

    def test_cost_bridge_endpoint(self, platform_admin_client, session):
        """GET /api/v1/admin/tenants/{id}/cost-bridge returns cost report."""
        client, key, tenant = platform_admin_client

        # Seed usage
        from api.usage import _get_or_create_usage

        record = _get_or_create_usage(tenant.tenant_id, "2026-03", session)
        record.pages_processed = 50
        record.api_calls = 25
        session.commit()

        with patch("api.usage._current_period", return_value="2026-03"):
            resp = client.get(
                f"/api/v1/admin/tenants/{tenant.tenant_id}/cost-bridge",
                headers={"X-API-Key": key},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["tenant_id"] == tenant.tenant_id
        assert "cost" in body
        assert "usage" in body
        assert body["cost"]["currency"] == "USD"

    def test_cost_bridge_endpoint_no_usage_returns_zeroed(
        self, platform_admin_client
    ):
        """GET cost-bridge with no usage returns a zeroed report."""
        client, key, tenant = platform_admin_client
        resp = client.get(
            f"/api/v1/admin/tenants/{tenant.tenant_id}/cost-bridge",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["cost"]["total_cost"] == 0.0
        assert body["usage"]["pages_processed"] == 0

    def test_cost_bridge_endpoint_tenant_not_found(self, platform_admin_client):
        """GET cost-bridge for nonexistent tenant returns 404."""
        client, key, _ = platform_admin_client
        resp = client.get(
            "/api/v1/admin/tenants/tenant_000000000000/cost-bridge",
            headers={"X-API-Key": key},
        )
        assert resp.status_code == 404

    def test_cost_bridge_endpoint_requires_auth(self, platform_admin_client):
        """GET cost-bridge without auth is rejected."""
        client, _, tenant = platform_admin_client
        resp = client.get(
            f"/api/v1/admin/tenants/{tenant.tenant_id}/cost-bridge",
        )
        assert resp.status_code == 401


# ===========================================================================
# Auth requirement coverage
# ===========================================================================


class TestAuthRequirements:
    """Verify all new endpoints enforce authentication."""

    def test_delete_no_auth(self, platform_admin_client):
        client, _, tenant = platform_admin_client
        resp = client.delete(f"/api/v1/admin/tenants/{tenant.tenant_id}")
        assert resp.status_code == 401

    def test_cost_bridge_no_auth(self, platform_admin_client):
        client, _, tenant = platform_admin_client
        resp = client.get(f"/api/v1/admin/tenants/{tenant.tenant_id}/cost-bridge")
        assert resp.status_code == 401

    def test_tenant_dashboard_no_auth(self, dashboard_client):
        client, _, tenant = dashboard_client
        resp = client.get(f"/api/v1/dashboard/tenant/{tenant.tenant_id}")
        assert resp.status_code == 401
