"""Tests for M-25: Per-tenant rate limiting with admin configuration.

Validates:
- Default rate limit applies when no tenant-specific config exists
- Per-tenant rate limit overrides the default
- Rate limit key extraction from API key / tenant_id
- Admin endpoint sets, gets, and deletes per-tenant rate limits
- Rate limit string validation
- Cache invalidation and refresh
- Dynamic limit resolver behavior
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

# Ensure multi-tenancy and test DB path are configured before importing API modules
os.environ.setdefault("ENABLE_MULTITENANCY", "true")
os.environ.setdefault("ALLOW_UNAUTHENTICATED", "true")


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Use a fresh SQLite database for each test."""
    db_path = str(tmp_path / "test_rate_limit.db")
    monkeypatch.setenv("ENABLE_MULTITENANCY", "true")
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "true")

    import api.database as db_mod

    db_mod.reset_engine()
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    db_mod.get_engine(db_path)

    # Clear the tenant rate limiter cache between tests
    from api.tenant_rate_limiter import invalidate_cache

    invalidate_cache()

    yield

    db_mod.reset_engine()


@pytest.fixture()
def _create_tenant():
    """Helper to create a tenant in the test database."""
    from api.tenant_manager import create_tenant

    def _make(name="test-tenant", tier="standard"):
        tenant = create_tenant(name=name, tier=tier)
        return tenant.tenant_id

    return _make


# ---------------------------------------------------------------------------
# Unit tests: validate_rate_limit_string
# ---------------------------------------------------------------------------


class TestRateLimitValidation:
    def test_valid_formats(self):
        from api.tenant_rate_limiter import validate_rate_limit_string

        assert validate_rate_limit_string("100/minute") is True
        assert validate_rate_limit_string("10/second") is True
        assert validate_rate_limit_string("500/hour") is True
        assert validate_rate_limit_string("1000/day") is True
        assert validate_rate_limit_string("5000/month") is True
        assert validate_rate_limit_string("50000/year") is True

    def test_invalid_formats(self):
        from api.tenant_rate_limiter import validate_rate_limit_string

        assert validate_rate_limit_string("") is False
        assert validate_rate_limit_string("100") is False
        assert validate_rate_limit_string("abc/minute") is False
        assert validate_rate_limit_string("100/week") is False
        assert validate_rate_limit_string("100 per minute") is False
        assert validate_rate_limit_string("-1/minute") is False
        assert validate_rate_limit_string("100/MINUTE") is False


# ---------------------------------------------------------------------------
# Unit tests: tenant_key_func
# ---------------------------------------------------------------------------


class TestTenantKeyFunc:
    def test_extracts_tenant_id_when_multitenancy_enabled(self, monkeypatch):
        monkeypatch.setattr("api.tenant_rate_limiter.ENABLE_MULTITENANCY", True)

        from api.tenant_rate_limiter import tenant_key_func

        request = MagicMock()
        request.state.tenant_id = "tenant_abc123def456"
        assert tenant_key_func(request) == "tenant_abc123def456"

    def test_falls_back_to_ip_when_no_tenant(self, monkeypatch):
        monkeypatch.setattr("api.tenant_rate_limiter.ENABLE_MULTITENANCY", True)

        from api.tenant_rate_limiter import tenant_key_func

        request = MagicMock(spec=[])
        # request.state has no tenant_id attribute
        request.state = MagicMock(spec=[])
        request.client = MagicMock()
        request.client.host = "192.168.1.100"
        # Remove tenant_id so getattr returns None
        assert not hasattr(request.state, "tenant_id")

        result = tenant_key_func(request)
        assert result == "192.168.1.100"

    def test_falls_back_to_ip_when_multitenancy_disabled(self, monkeypatch):
        monkeypatch.setattr("api.tenant_rate_limiter.ENABLE_MULTITENANCY", False)

        from api.tenant_rate_limiter import tenant_key_func

        request = MagicMock()
        request.state.tenant_id = "tenant_abc123def456"
        request.client.host = "10.0.0.1"

        result = tenant_key_func(request)
        assert result == "10.0.0.1"


# ---------------------------------------------------------------------------
# Unit tests: set/get/delete tenant rate limits
# ---------------------------------------------------------------------------


class TestTenantRateLimitCRUD:
    def test_set_and_get(self, _create_tenant):
        from api.tenant_rate_limiter import (
            get_tenant_rate_limit,
            set_tenant_rate_limit,
        )

        tenant_id = _create_tenant()
        set_tenant_rate_limit(tenant_id, "200/minute")
        assert get_tenant_rate_limit(tenant_id) == "200/minute"

    def test_get_returns_none_when_unset(self, _create_tenant):
        from api.tenant_rate_limiter import get_tenant_rate_limit

        tenant_id = _create_tenant()
        assert get_tenant_rate_limit(tenant_id) is None

    def test_update_existing(self, _create_tenant):
        from api.tenant_rate_limiter import (
            get_tenant_rate_limit,
            set_tenant_rate_limit,
        )

        tenant_id = _create_tenant()
        set_tenant_rate_limit(tenant_id, "100/minute")
        set_tenant_rate_limit(tenant_id, "500/hour")
        assert get_tenant_rate_limit(tenant_id) == "500/hour"

    def test_delete(self, _create_tenant):
        from api.tenant_rate_limiter import (
            delete_tenant_rate_limit,
            get_tenant_rate_limit,
            set_tenant_rate_limit,
        )

        tenant_id = _create_tenant()
        set_tenant_rate_limit(tenant_id, "100/minute")
        assert delete_tenant_rate_limit(tenant_id) is True
        assert get_tenant_rate_limit(tenant_id) is None

    def test_delete_nonexistent(self, _create_tenant):
        from api.tenant_rate_limiter import delete_tenant_rate_limit

        tenant_id = _create_tenant()
        assert delete_tenant_rate_limit(tenant_id) is False

    def test_set_invalid_format_raises(self, _create_tenant):
        from api.tenant_rate_limiter import set_tenant_rate_limit

        tenant_id = _create_tenant()
        with pytest.raises(ValueError, match="Invalid rate limit format"):
            set_tenant_rate_limit(tenant_id, "not-a-rate")

    def test_cache_invalidation(self, _create_tenant):
        from api.tenant_rate_limiter import (
            get_tenant_rate_limit,
            invalidate_cache,
            set_tenant_rate_limit,
        )

        tenant_id = _create_tenant()
        set_tenant_rate_limit(tenant_id, "300/minute")
        assert get_tenant_rate_limit(tenant_id) == "300/minute"

        invalidate_cache()
        # After invalidation, should reload from DB on next access
        assert get_tenant_rate_limit(tenant_id) == "300/minute"


# ---------------------------------------------------------------------------
# Unit tests: dynamic limit resolver
# ---------------------------------------------------------------------------


class TestDynamicLimitResolver:
    def test_returns_tenant_limit_when_configured(self, _create_tenant, monkeypatch):
        monkeypatch.setattr("api.tenant_rate_limiter.ENABLE_MULTITENANCY", True)

        from api.tenant_rate_limiter import (
            _make_dynamic_limit_resolver,
            set_tenant_rate_limit,
        )

        tenant_id = _create_tenant()
        set_tenant_rate_limit(tenant_id, "200/minute")

        resolver = _make_dynamic_limit_resolver(lambda: "60/minute")
        assert resolver(tenant_id) == "200/minute"

    def test_falls_back_to_default_when_no_tenant_config(self, _create_tenant, monkeypatch):
        monkeypatch.setattr("api.tenant_rate_limiter.ENABLE_MULTITENANCY", True)

        from api.tenant_rate_limiter import _make_dynamic_limit_resolver

        tenant_id = _create_tenant()
        resolver = _make_dynamic_limit_resolver(lambda: "60/minute")
        assert resolver(tenant_id) == "60/minute"

    def test_falls_back_for_non_tenant_key(self, monkeypatch):
        monkeypatch.setattr("api.tenant_rate_limiter.ENABLE_MULTITENANCY", True)

        from api.tenant_rate_limiter import _make_dynamic_limit_resolver

        resolver = _make_dynamic_limit_resolver(lambda: "60/minute")
        assert resolver("192.168.1.100") == "60/minute"

    def test_falls_back_when_multitenancy_disabled(self, monkeypatch):
        monkeypatch.setattr("api.tenant_rate_limiter.ENABLE_MULTITENANCY", False)

        from api.tenant_rate_limiter import _make_dynamic_limit_resolver

        resolver = _make_dynamic_limit_resolver(lambda: "60/minute")
        assert resolver("tenant_abc123def456") == "60/minute"

    def test_get_dynamic_default_rate(self, monkeypatch):
        monkeypatch.setattr("api.tenant_rate_limiter.ENABLE_MULTITENANCY", False)

        from api.tenant_rate_limiter import get_dynamic_default_rate

        resolver = get_dynamic_default_rate()
        # Should return the default rate for non-tenant key
        result = resolver("192.168.1.1")
        assert "/" in result  # Must be a valid rate string

    def test_get_dynamic_submit_rate(self, monkeypatch):
        monkeypatch.setattr("api.tenant_rate_limiter.ENABLE_MULTITENANCY", False)

        from api.tenant_rate_limiter import get_dynamic_submit_rate

        resolver = get_dynamic_submit_rate()
        result = resolver("192.168.1.1")
        assert "/" in result


# ---------------------------------------------------------------------------
# Unit tests: limits.py tenant-aware key func
# ---------------------------------------------------------------------------


class TestLimitsKeyFunc:
    def test_key_func_returns_tenant_id(self, monkeypatch):
        monkeypatch.setattr("api.limits.ENABLE_MULTITENANCY", True)

        from api.limits import _tenant_aware_key_func

        request = MagicMock()
        request.state.tenant_id = "tenant_abc123def456"
        assert _tenant_aware_key_func(request) == "tenant_abc123def456"

    def test_key_func_returns_ip_when_no_tenant(self, monkeypatch):
        monkeypatch.setattr("api.limits.ENABLE_MULTITENANCY", True)

        from api.limits import _tenant_aware_key_func

        request = MagicMock(spec=[])
        request.state = MagicMock(spec=[])
        request.client = MagicMock()
        request.client.host = "10.0.0.5"
        request.headers = {}

        result = _tenant_aware_key_func(request)
        assert result == "10.0.0.5"


# ---------------------------------------------------------------------------
# Integration tests: admin API endpoints
# ---------------------------------------------------------------------------


class TestAdminRateLimitEndpoints:
    """Test the admin rate-limit endpoints via the FastAPI test client."""

    @pytest.fixture(autouse=True)
    def _setup_app(self, monkeypatch):
        """Create a test client with multi-tenancy enabled."""
        monkeypatch.setenv("ENABLE_MULTITENANCY", "true")
        monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "true")

        # Reload config to pick up env vars
        import importlib

        import api.config
        importlib.reload(api.config)
        monkeypatch.setattr("api.config.ENABLE_MULTITENANCY", True)

        # Patch auth middleware to always pass for admin tests
        from api.main import create_app

        # Patch the ENABLE_MULTITENANCY in all needed places
        monkeypatch.setattr("api.auth.ENABLE_MULTITENANCY", True)
        monkeypatch.setattr("api.main.ENABLE_MULTITENANCY", True)

        app = create_app()

        from starlette.testclient import TestClient

        self.client = TestClient(app)

    @pytest.fixture()
    def tenant_id(self, _create_tenant):
        return _create_tenant(name="rate-limit-test-tenant")

    def test_set_rate_limit(self, tenant_id):
        resp = self.client.put(
            f"/api/v1/admin/tenants/{tenant_id}/rate-limit",
            json={"rate_limit": "100/minute"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == tenant_id
        assert data["rate_limit"] == "100/minute"

    def test_get_rate_limit(self, tenant_id):
        # Set first
        self.client.put(
            f"/api/v1/admin/tenants/{tenant_id}/rate-limit",
            json={"rate_limit": "200/hour"},
        )
        resp = self.client.get(
            f"/api/v1/admin/tenants/{tenant_id}/rate-limit",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == tenant_id
        assert data["rate_limit"] == "200/hour"

    def test_get_rate_limit_unset(self, tenant_id):
        resp = self.client.get(
            f"/api/v1/admin/tenants/{tenant_id}/rate-limit",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["rate_limit"] is None

    def test_delete_rate_limit(self, tenant_id):
        self.client.put(
            f"/api/v1/admin/tenants/{tenant_id}/rate-limit",
            json={"rate_limit": "50/second"},
        )
        resp = self.client.delete(
            f"/api/v1/admin/tenants/{tenant_id}/rate-limit",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["rate_limit"] is None

        # Verify it's gone
        resp = self.client.get(
            f"/api/v1/admin/tenants/{tenant_id}/rate-limit",
        )
        assert resp.json()["rate_limit"] is None

    def test_set_invalid_format_rejected(self, tenant_id):
        resp = self.client.put(
            f"/api/v1/admin/tenants/{tenant_id}/rate-limit",
            json={"rate_limit": "not-valid"},
        )
        assert resp.status_code == 422  # Pydantic validation

    def test_set_for_nonexistent_tenant(self):
        resp = self.client.put(
            "/api/v1/admin/tenants/tenant_000000000000/rate-limit",
            json={"rate_limit": "100/minute"},
        )
        assert resp.status_code == 404

    def test_update_existing_rate_limit(self, tenant_id):
        self.client.put(
            f"/api/v1/admin/tenants/{tenant_id}/rate-limit",
            json={"rate_limit": "100/minute"},
        )
        resp = self.client.put(
            f"/api/v1/admin/tenants/{tenant_id}/rate-limit",
            json={"rate_limit": "500/hour"},
        )
        assert resp.status_code == 200
        assert resp.json()["rate_limit"] == "500/hour"

    def test_invalid_tenant_id_format(self):
        resp = self.client.put(
            "/api/v1/admin/tenants/bad-id/rate-limit",
            json={"rate_limit": "100/minute"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Database model tests
# ---------------------------------------------------------------------------


class TestTenantRateLimitModel:
    def test_model_created_in_db(self, _create_tenant):
        from api.database import TenantRateLimit, get_session_factory

        tenant_id = _create_tenant()
        session = get_session_factory()()
        try:
            row = TenantRateLimit(
                tenant_id=tenant_id, rate_limit="100/minute"
            )
            session.add(row)
            session.commit()

            loaded = session.get(TenantRateLimit, tenant_id)
            assert loaded is not None
            assert loaded.rate_limit == "100/minute"
        finally:
            session.close()

    def test_model_upsert(self, _create_tenant):
        from api.database import TenantRateLimit, get_session_factory

        tenant_id = _create_tenant()
        session = get_session_factory()()
        try:
            row = TenantRateLimit(
                tenant_id=tenant_id, rate_limit="100/minute"
            )
            session.add(row)
            session.commit()

            # Update
            loaded = session.get(TenantRateLimit, tenant_id)
            loaded.rate_limit = "500/hour"
            session.commit()

            reloaded = session.get(TenantRateLimit, tenant_id)
            assert reloaded.rate_limit == "500/hour"
        finally:
            session.close()
