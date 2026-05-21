"""Tests for multi-tenancy: tenant models, API keys, quotas, usage, isolation, admin API."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.database import (
    Job,
    TenantApiKey,
    UsageRecord,
    get_engine,
    get_session_factory,
    reset_engine,
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
        name="Test Corp",
        tier="standard",
        max_concurrent_jobs=4,
        max_pages_per_month=10000,
        admin_email="admin@testcorp.com",
        session=session,
    )


@pytest.fixture()
def sample_tenant_with_key(sample_tenant, session):
    """Create a tenant with an API key, return (tenant, key_id, raw_key)."""
    from api.tenant_manager import create_api_key

    key_id, raw_key = create_api_key(
        sample_tenant.tenant_id,
        name="test-key",
        permissions=["submit", "read", "admin"],
        session=session,
    )
    return sample_tenant, key_id, raw_key


@pytest.fixture()
def mt_client(tmp_path, sample_tenant_with_key):
    """FastAPI TestClient with multi-tenancy enabled and an admin tenant key."""
    tenant, key_id, raw_key = sample_tenant_with_key
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
        yield TestClient(app), raw_key


@pytest.fixture()
def platform_admin_client(tmp_path, sample_tenant, session):
    """FastAPI TestClient with a platform-admin tenant key."""
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
        yield TestClient(app), raw_key


@pytest.fixture()
def legacy_client(tmp_path):
    """FastAPI TestClient with multi-tenancy DISABLED (legacy single-key mode)."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with (
        patch("api.config.ENABLE_MULTITENANCY", False),
        patch("api.auth.ENABLE_MULTITENANCY", False),
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.config.OCR_API_KEY", "legacy-test-key"),
        patch("api.auth.OCR_API_KEY", "legacy-test-key"),
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
        yield TestClient(app)


# ===========================================================================
# TestTenantModel
# ===========================================================================


class TestTenantModel:
    """Tests for Tenant ORM model CRUD."""

    def test_create_tenant(self, session):
        from api.tenant_manager import create_tenant

        tenant = create_tenant(name="Acme Inc", session=session)
        assert tenant.tenant_id.startswith("tenant_")
        assert len(tenant.tenant_id) == len("tenant_") + 12
        assert tenant.name == "Acme Inc"
        assert tenant.status == "active"
        assert tenant.tier == "standard"

    def test_create_tenant_custom_tier(self, session):
        from api.tenant_manager import create_tenant

        tenant = create_tenant(
            name="Enterprise Corp",
            tier="enterprise",
            max_concurrent_jobs=20,
            max_pages_per_month=1_000_000,
            session=session,
        )
        assert tenant.tier == "enterprise"
        assert tenant.max_concurrent_jobs == 20
        assert tenant.max_pages_per_month == 1_000_000

    def test_get_tenant(self, sample_tenant, session):
        from api.tenant_manager import get_tenant

        fetched = get_tenant(sample_tenant.tenant_id, session=session)
        assert fetched is not None
        assert fetched.tenant_id == sample_tenant.tenant_id
        assert fetched.name == "Test Corp"

    def test_get_tenant_not_found(self, session):
        from api.tenant_manager import get_tenant

        result = get_tenant("tenant_nonexistent", session=session)
        assert result is None

    def test_list_tenants(self, session):
        from api.tenant_manager import create_tenant, list_tenants

        create_tenant(name="A Corp", session=session)
        create_tenant(name="B Corp", session=session)
        tenants = list_tenants(session=session)
        assert len(tenants) >= 2

    def test_list_tenants_filter_status(self, session):
        from api.tenant_manager import create_tenant, list_tenants, suspend_tenant

        t1 = create_tenant(name="Active Corp", session=session)
        t2 = create_tenant(name="Suspended Corp", session=session)
        suspend_tenant(t2.tenant_id, session=session)

        active = list_tenants(status_filter="active", session=session)
        suspended = list_tenants(status_filter="suspended", session=session)
        assert any(t.tenant_id == t1.tenant_id for t in active)
        assert any(t.tenant_id == t2.tenant_id for t in suspended)

    def test_update_tenant(self, sample_tenant, session):
        from api.tenant_manager import update_tenant

        updated = update_tenant(
            sample_tenant.tenant_id,
            name="Updated Corp",
            max_concurrent_jobs=10,
            session=session,
        )
        assert updated.name == "Updated Corp"
        assert updated.max_concurrent_jobs == 10
        assert updated.updated_at is not None

    def test_update_tenant_not_found(self, session):
        from api.tenant_manager import update_tenant

        result = update_tenant("tenant_nonexistent", name="Nothing", session=session)
        assert result is None

    def test_suspend_tenant(self, sample_tenant, session):
        from api.tenant_manager import suspend_tenant

        result = suspend_tenant(sample_tenant.tenant_id, session=session)
        assert result.status == "suspended"

    def test_activate_tenant(self, sample_tenant, session):
        from api.tenant_manager import activate_tenant, suspend_tenant

        suspend_tenant(sample_tenant.tenant_id, session=session)
        result = activate_tenant(sample_tenant.tenant_id, session=session)
        assert result.status == "active"

    def test_tenant_allowed_features_json(self, session):
        from api.tenant_manager import create_tenant

        tenant = create_tenant(
            name="Feature Corp",
            allowed_features=["docintel", "ner", "extraction"],
            session=session,
        )
        features = json.loads(tenant.allowed_features)
        assert "docintel" in features
        assert "ner" in features

    def test_update_tenant_allowed_features(self, sample_tenant, session):
        from api.tenant_manager import update_tenant

        updated = update_tenant(
            sample_tenant.tenant_id,
            allowed_features=["docintel"],
            session=session,
        )
        features = json.loads(updated.allowed_features)
        assert features == ["docintel"]


# ===========================================================================
# TestTenantApiKey
# ===========================================================================


class TestTenantApiKey:
    """Tests for API key creation, resolution, and revocation."""

    def test_create_api_key(self, sample_tenant, session):
        from api.tenant_manager import create_api_key

        key_id, raw_key = create_api_key(
            sample_tenant.tenant_id,
            name="my-key",
            session=session,
        )
        assert key_id.startswith("key_")
        assert raw_key.startswith("ocr_")
        assert len(raw_key) > 20

    def test_create_api_key_nonexistent_tenant(self, session):
        from api.tenant_manager import create_api_key

        with pytest.raises(ValueError, match="Tenant not found"):
            create_api_key("tenant_nonexistent", session=session)

    def test_resolve_by_hash(self, sample_tenant, session):
        from api.tenant_manager import (
            create_api_key,
            hash_api_key,
            resolve_tenant_by_key,
        )

        key_id, raw_key = create_api_key(
            sample_tenant.tenant_id,
            name="resolve-test",
            session=session,
        )
        key_hash = hash_api_key(raw_key)
        result = resolve_tenant_by_key(key_hash, session=session)
        assert result is not None
        tenant, key_record = result
        assert tenant.tenant_id == sample_tenant.tenant_id
        assert key_record.key_id == key_id

    def test_resolve_wrong_hash_returns_none(self, session):
        from api.tenant_manager import resolve_tenant_by_key

        result = resolve_tenant_by_key("deadbeef" * 8, session=session)
        assert result is None

    def test_revoke_api_key(self, sample_tenant, session):
        from api.tenant_manager import (
            create_api_key,
            hash_api_key,
            resolve_tenant_by_key,
            revoke_api_key,
        )

        key_id, raw_key = create_api_key(
            sample_tenant.tenant_id,
            name="revoke-test",
            session=session,
        )
        assert revoke_api_key(key_id, session=session) is True

        # Revoked key should not resolve
        result = resolve_tenant_by_key(hash_api_key(raw_key), session=session)
        assert result is None

    def test_revoke_nonexistent_key(self, session):
        from api.tenant_manager import revoke_api_key

        assert revoke_api_key("key_nonexistent0", session=session) is False

    def test_expired_key_not_resolved(self, sample_tenant, session):
        from api.tenant_manager import (
            create_api_key,
            hash_api_key,
            resolve_tenant_by_key,
        )

        past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        key_id, raw_key = create_api_key(
            sample_tenant.tenant_id,
            name="expired-key",
            expires_at=past,
            session=session,
        )
        result = resolve_tenant_by_key(hash_api_key(raw_key), session=session)
        assert result is None

    def test_suspended_tenant_key_not_resolved(self, sample_tenant, session):
        from api.tenant_manager import (
            create_api_key,
            hash_api_key,
            resolve_tenant_by_key,
            suspend_tenant,
        )

        key_id, raw_key = create_api_key(
            sample_tenant.tenant_id,
            name="suspended-test",
            session=session,
        )
        suspend_tenant(sample_tenant.tenant_id, session=session)
        result = resolve_tenant_by_key(hash_api_key(raw_key), session=session)
        assert result is None

    def test_api_key_custom_permissions(self, sample_tenant, session):
        from api.tenant_manager import (
            create_api_key,
            hash_api_key,
            resolve_tenant_by_key,
        )

        key_id, raw_key = create_api_key(
            sample_tenant.tenant_id,
            permissions=["submit", "read", "admin"],
            session=session,
        )
        result = resolve_tenant_by_key(hash_api_key(raw_key), session=session)
        assert result is not None
        _, key_record = result
        perms = json.loads(key_record.permissions)
        assert "admin" in perms

    def test_last_used_at_updated(self, sample_tenant, session):
        from api.tenant_manager import (
            create_api_key,
            hash_api_key,
            resolve_tenant_by_key,
        )

        key_id, raw_key = create_api_key(
            sample_tenant.tenant_id,
            session=session,
        )
        result = resolve_tenant_by_key(hash_api_key(raw_key), session=session)
        assert result is not None
        _, key_record = result
        assert key_record.last_used_at is not None


# ===========================================================================
# TestApiKeyHashing
# ===========================================================================


class TestApiKeyHashing:
    """Tests for SHA-256 API key hashing."""

    def test_hash_deterministic(self):
        from api.tenant_manager import hash_api_key

        key = "ocr_test1234567890abcdef"
        h1 = hash_api_key(key)
        h2 = hash_api_key(key)
        assert h1 == h2

    def test_hash_is_sha256_hex(self):
        from api.tenant_manager import hash_api_key

        key = "ocr_test_key_for_hashing"
        h = hash_api_key(key)
        assert len(h) == 64
        int(h, 16)  # Should not raise

    def test_hash_matches_stdlib(self):
        from api.tenant_manager import hash_api_key

        key = "ocr_verify_against_stdlib"
        expected = hashlib.sha256(key.encode("utf-8")).hexdigest()
        assert hash_api_key(key) == expected

    def test_different_keys_different_hashes(self):
        from api.tenant_manager import hash_api_key

        h1 = hash_api_key("ocr_key_alpha")
        h2 = hash_api_key("ocr_key_beta")
        assert h1 != h2

    def test_raw_key_never_stored(self, sample_tenant, session):
        """Verify that the raw API key does not appear in the database."""
        from api.tenant_manager import create_api_key

        _, raw_key = create_api_key(
            sample_tenant.tenant_id,
            session=session,
        )
        # Query all key records
        records = session.query(TenantApiKey).all()
        for r in records:
            assert r.api_key_hash != raw_key
            assert raw_key not in (r.api_key_hash or "")


# ===========================================================================
# TestQuotaEnforcement
# ===========================================================================


class TestQuotaEnforcement:
    """Tests for quota checking logic."""

    def test_check_job_quota_under_limit(self, sample_tenant, session):
        from api.quota import check_job_quota

        assert check_job_quota(sample_tenant, session=session) is True

    def test_check_job_quota_at_limit(self, sample_tenant, session):
        from api.quota import QuotaExceededError, check_job_quota

        # Create active jobs up to the limit
        for i in range(sample_tenant.max_concurrent_jobs):
            job = Job(
                job_id=f"job_{i:012d}",
                status="processing",
                source_file="test.pdf",
                tenant_id=sample_tenant.tenant_id,
            )
            session.add(job)
        session.commit()

        with pytest.raises(QuotaExceededError) as exc_info:
            check_job_quota(sample_tenant, session=session)
        assert exc_info.value.limit_type == "concurrent_jobs"
        assert exc_info.value.current == sample_tenant.max_concurrent_jobs

    def test_completed_jobs_dont_count(self, sample_tenant, session):
        from api.quota import check_job_quota

        # Completed jobs should not count toward concurrent limit
        for i in range(10):
            job = Job(
                job_id=f"job_{i:012d}",
                status="completed",
                source_file="test.pdf",
                tenant_id=sample_tenant.tenant_id,
            )
            session.add(job)
        session.commit()
        assert check_job_quota(sample_tenant, session=session) is True

    def test_check_page_quota_under_limit(self, sample_tenant, session):
        from api.quota import check_page_quota

        assert check_page_quota(sample_tenant, 100, session=session) is True

    def test_check_page_quota_exceeded(self, sample_tenant, session):
        from api.quota import QuotaExceededError, check_page_quota
        from api.usage import record_pages_processed

        # Record pages up to near the limit
        record_pages_processed(sample_tenant.tenant_id, 9950, session=session)

        with pytest.raises(QuotaExceededError) as exc_info:
            check_page_quota(sample_tenant, 100, session=session)
        assert exc_info.value.limit_type == "pages_per_month"

    def test_check_storage_quota_under_limit(self, sample_tenant, session):
        from api.quota import check_storage_quota

        assert check_storage_quota(sample_tenant, 1024, session=session) is True

    def test_check_storage_quota_exceeded(self, sample_tenant, session):
        from api.quota import QuotaExceededError, check_storage_quota
        from api.usage import record_storage_used

        # Record storage up to near the limit
        record_storage_used(
            sample_tenant.tenant_id,
            sample_tenant.max_storage_bytes - 100,
            session=session,
        )

        with pytest.raises(QuotaExceededError) as exc_info:
            check_storage_quota(sample_tenant, 200, session=session)
        assert exc_info.value.limit_type == "storage_bytes"

    def test_quota_exceeded_error_attributes(self):
        from api.quota import QuotaExceededError

        err = QuotaExceededError("tenant_abc", "concurrent_jobs", 4, 4)
        assert err.tenant_id == "tenant_abc"
        assert err.limit_type == "concurrent_jobs"
        assert err.current == 4
        assert err.maximum == 4
        assert "concurrent_jobs" in str(err)


# ===========================================================================
# TestUsageTracking
# ===========================================================================


class TestUsageTracking:
    """Tests for usage recording and retrieval."""

    def test_record_job_submitted(self, sample_tenant, session):
        from api.usage import get_usage, record_job_submitted

        record_job_submitted(sample_tenant.tenant_id, session=session)
        usage = get_usage(sample_tenant.tenant_id, session=session)
        assert usage is not None
        assert usage.jobs_submitted == 1

    def test_record_multiple_jobs(self, sample_tenant, session):
        from api.usage import get_usage, record_job_submitted

        for _ in range(5):
            record_job_submitted(sample_tenant.tenant_id, session=session)
        usage = get_usage(sample_tenant.tenant_id, session=session)
        assert usage.jobs_submitted == 5

    def test_record_pages_processed(self, sample_tenant, session):
        from api.usage import get_usage, record_pages_processed

        record_pages_processed(sample_tenant.tenant_id, 42, session=session)
        usage = get_usage(sample_tenant.tenant_id, session=session)
        assert usage.pages_processed == 42

    def test_record_pages_accumulates(self, sample_tenant, session):
        from api.usage import get_usage, record_pages_processed

        record_pages_processed(sample_tenant.tenant_id, 10, session=session)
        record_pages_processed(sample_tenant.tenant_id, 20, session=session)
        usage = get_usage(sample_tenant.tenant_id, session=session)
        assert usage.pages_processed == 30

    def test_record_storage_used(self, sample_tenant, session):
        from api.usage import get_usage, record_storage_used

        record_storage_used(sample_tenant.tenant_id, 1048576, session=session)
        usage = get_usage(sample_tenant.tenant_id, session=session)
        assert usage.storage_bytes_used == 1048576

    def test_record_api_call(self, sample_tenant, session):
        from api.usage import get_usage, record_api_call

        record_api_call(sample_tenant.tenant_id, session=session)
        record_api_call(sample_tenant.tenant_id, session=session)
        usage = get_usage(sample_tenant.tenant_id, session=session)
        assert usage.api_calls == 2

    def test_record_processing_seconds(self, sample_tenant, session):
        from api.usage import get_usage, record_processing_seconds

        record_processing_seconds(sample_tenant.tenant_id, 12.5, session=session)
        record_processing_seconds(sample_tenant.tenant_id, 7.5, session=session)
        usage = get_usage(sample_tenant.tenant_id, session=session)
        assert usage.processing_seconds == pytest.approx(20.0)

    def test_build_cost_summary_uses_provider_agnostic_rates(self, sample_tenant, session):
        from api.usage import build_cost_summary, get_usage, record_processing_seconds

        usage = UsageRecord(
            tenant_id=sample_tenant.tenant_id,
            period="2026-03",
            jobs_submitted=4,
            pages_processed=10,
            storage_bytes_used=2 * 1024**3,
            api_calls=8,
            processing_seconds=0.0,
        )
        session.add(usage)
        session.commit()

        with (
            patch("api.config.TENANT_COST_PER_PAGE_USD", 0.02),
            patch("api.usage.config.TENANT_COST_PER_PAGE_USD", 0.02),
            patch("api.config.TENANT_COST_PER_GIB_INGESTED_USD", 0.5),
            patch("api.usage.config.TENANT_COST_PER_GIB_INGESTED_USD", 0.5),
            patch("api.config.TENANT_COST_PER_API_CALL_USD", 0.01),
            patch("api.usage.config.TENANT_COST_PER_API_CALL_USD", 0.01),
            patch("api.config.TENANT_COST_PER_PROCESSING_HOUR_USD", 3.0),
            patch("api.usage.config.TENANT_COST_PER_PROCESSING_HOUR_USD", 3.0),
        ):
            record_processing_seconds(sample_tenant.tenant_id, 1800, session=session)
            usage = get_usage(sample_tenant.tenant_id, period="2026-03", session=session)
            summary = build_cost_summary(usage)

        assert summary["page_cost_usd"] == pytest.approx(0.2)
        assert summary["storage_ingest_cost_usd"] == pytest.approx(1.0)
        assert summary["api_call_cost_usd"] == pytest.approx(0.08)
        assert summary["processing_cost_usd"] == pytest.approx(1.5)
        assert summary["total_cost_usd"] == pytest.approx(2.78)

    def test_monthly_period_format(self, sample_tenant, session):
        from api.usage import get_usage, record_job_submitted

        record_job_submitted(sample_tenant.tenant_id, session=session)
        usage = get_usage(sample_tenant.tenant_id, session=session)
        # Period should be YYYY-MM format
        assert len(usage.period) == 7
        assert usage.period[4] == "-"

    def test_get_usage_specific_period(self, sample_tenant, session):
        from api.usage import get_usage

        # No usage recorded for a future period
        result = get_usage(sample_tenant.tenant_id, period="2099-12", session=session)
        assert result is None

    def test_different_periods_separate(self, sample_tenant, session):
        """Usage records for different months are independent."""
        record1 = UsageRecord(
            tenant_id=sample_tenant.tenant_id,
            period="2026-01",
            jobs_submitted=10,
            pages_processed=100,
        )
        record2 = UsageRecord(
            tenant_id=sample_tenant.tenant_id,
            period="2026-02",
            jobs_submitted=5,
            pages_processed=50,
        )
        session.add_all([record1, record2])
        session.commit()

        from api.usage import get_usage

        u1 = get_usage(sample_tenant.tenant_id, period="2026-01", session=session)
        u2 = get_usage(sample_tenant.tenant_id, period="2026-02", session=session)
        assert u1.jobs_submitted == 10
        assert u2.jobs_submitted == 5


# ===========================================================================
# TestTenantIsolation
# ===========================================================================


class TestTenantIsolation:
    """Tests for data isolation between tenants."""

    def test_tenant_a_cannot_see_tenant_b_jobs(self, session):
        from api.tenant_manager import create_tenant

        tenant_a = create_tenant(name="Tenant A", session=session)
        tenant_b = create_tenant(name="Tenant B", session=session)

        # Create jobs for each tenant
        job_a = Job(
            job_id="job_aaaaaaaaaaaa",
            status="completed",
            source_file="a.pdf",
            tenant_id=tenant_a.tenant_id,
        )
        job_b = Job(
            job_id="job_bbbbbbbbbbbb",
            status="completed",
            source_file="b.pdf",
            tenant_id=tenant_b.tenant_id,
        )
        session.add_all([job_a, job_b])
        session.commit()

        # Query jobs for tenant A
        a_jobs = (
            session.query(Job)
            .filter(Job.tenant_id == tenant_a.tenant_id)
            .all()
        )
        b_jobs = (
            session.query(Job)
            .filter(Job.tenant_id == tenant_b.tenant_id)
            .all()
        )

        assert len(a_jobs) == 1
        assert a_jobs[0].job_id == "job_aaaaaaaaaaaa"
        assert len(b_jobs) == 1
        assert b_jobs[0].job_id == "job_bbbbbbbbbbbb"

    def test_tenant_usage_isolated(self, session):
        from api.tenant_manager import create_tenant
        from api.usage import get_usage, record_job_submitted

        tenant_a = create_tenant(name="Usage A", session=session)
        tenant_b = create_tenant(name="Usage B", session=session)

        record_job_submitted(tenant_a.tenant_id, session=session)
        record_job_submitted(tenant_a.tenant_id, session=session)
        record_job_submitted(tenant_b.tenant_id, session=session)

        usage_a = get_usage(tenant_a.tenant_id, session=session)
        usage_b = get_usage(tenant_b.tenant_id, session=session)
        assert usage_a.jobs_submitted == 2
        assert usage_b.jobs_submitted == 1

    def test_tenant_api_keys_isolated(self, session):
        from api.tenant_manager import (
            create_api_key,
            create_tenant,
            hash_api_key,
            resolve_tenant_by_key,
        )

        t1 = create_tenant(name="Key Tenant 1", session=session)
        t2 = create_tenant(name="Key Tenant 2", session=session)

        _, key1 = create_api_key(t1.tenant_id, session=session)
        _, key2 = create_api_key(t2.tenant_id, session=session)

        result1 = resolve_tenant_by_key(hash_api_key(key1), session=session)
        result2 = resolve_tenant_by_key(hash_api_key(key2), session=session)

        assert result1[0].tenant_id == t1.tenant_id
        assert result2[0].tenant_id == t2.tenant_id

    def test_quota_check_per_tenant(self, session):
        """Each tenant has independent job quota."""
        from api.quota import QuotaExceededError, check_job_quota
        from api.tenant_manager import create_tenant

        t1 = create_tenant(name="Quota T1", max_concurrent_jobs=1, session=session)
        t2 = create_tenant(name="Quota T2", max_concurrent_jobs=1, session=session)

        # Fill t1's quota
        job = Job(
            job_id="job_t1filled0001",
            status="processing",
            source_file="test.pdf",
            tenant_id=t1.tenant_id,
        )
        session.add(job)
        session.commit()

        # t1 should be over quota
        with pytest.raises(QuotaExceededError):
            check_job_quota(t1, session=session)

        # t2 should still be fine
        assert check_job_quota(t2, session=session) is True


# ===========================================================================
# TestBackwardCompatibility
# ===========================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility when ENABLE_MULTITENANCY is false."""

    def test_legacy_auth_works(self, legacy_client):
        """With multitenancy disabled, legacy X-API-Key auth works."""
        resp = legacy_client.get(
            "/api/v1/health",
            headers={"X-API-Key": "legacy-test-key"},
        )
        assert resp.status_code == 200

    def test_legacy_wrong_key_rejected(self, legacy_client):
        """With multitenancy disabled, wrong key is rejected."""
        resp = legacy_client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_admin_endpoint_not_registered(self, legacy_client):
        """Admin endpoints should not exist when multitenancy is disabled."""
        resp = legacy_client.get(
            "/api/v1/admin/tenants",
            headers={"X-API-Key": "legacy-test-key"},
        )
        assert resp.status_code == 404

    def test_jobs_have_null_tenant_id(self, session):
        """Pre-existing jobs should have tenant_id=None."""
        job = Job(
            job_id="job_legacy000001",
            status="completed",
            source_file="legacy.pdf",
        )
        session.add(job)
        session.commit()

        fetched = session.get(Job, "job_legacy000001")
        assert fetched.tenant_id is None


# ===========================================================================
# TestAdminAPI
# ===========================================================================


class TestAdminAPI:
    """Tests for admin API endpoints (require multi-tenancy + admin permission)."""

    def test_create_tenant_endpoint(self, platform_admin_client):
        client, api_key = platform_admin_client
        resp = client.post(
            "/api/v1/admin/tenants",
            json={"name": "New Tenant", "tier": "enterprise"},
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "New Tenant"
        assert data["tier"] == "enterprise"
        assert data["tenant_id"].startswith("tenant_")

    def test_list_tenants_endpoint(self, platform_admin_client):
        client, api_key = platform_admin_client
        resp = client.get(
            "/api/v1/admin/tenants",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1  # At least the sample tenant

    def test_list_tenants_scopes_tenant_admin_to_self(self, mt_client, session):
        client, api_key = mt_client
        from api.tenant_manager import create_tenant

        create_tenant(name="Other Tenant", session=session)

        resp = client.get(
            "/api/v1/admin/tenants",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Test Corp"

    def test_get_tenant_endpoint(self, mt_client, sample_tenant):
        client, api_key = mt_client
        resp = client.get(
            f"/api/v1/admin/tenants/{sample_tenant.tenant_id}",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == sample_tenant.tenant_id
        assert data["name"] == "Test Corp"

    def test_get_tenant_not_found(self, mt_client):
        client, api_key = mt_client
        resp = client.get(
            "/api/v1/admin/tenants/tenant_000000000000",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 404

    def test_update_tenant_endpoint(self, mt_client, sample_tenant):
        client, api_key = mt_client
        resp = client.put(
            f"/api/v1/admin/tenants/{sample_tenant.tenant_id}",
            json={"name": "Renamed Corp", "max_concurrent_jobs": 8},
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Renamed Corp"
        assert data["max_concurrent_jobs"] == 8

    def test_suspend_tenant_endpoint(self, platform_admin_client):
        client, api_key = platform_admin_client

        # Create a second tenant to suspend (don't suspend the admin's own tenant)
        resp = client.post(
            "/api/v1/admin/tenants",
            json={"name": "Suspend Target"},
            headers={"X-API-Key": api_key},
        )
        target_id = resp.json()["tenant_id"]

        resp = client.post(
            f"/api/v1/admin/tenants/{target_id}/suspend",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "suspended"

    def test_activate_tenant_endpoint(self, platform_admin_client):
        client, api_key = platform_admin_client

        # Create and suspend a tenant, then activate
        resp = client.post(
            "/api/v1/admin/tenants",
            json={"name": "Activate Target"},
            headers={"X-API-Key": api_key},
        )
        target_id = resp.json()["tenant_id"]

        client.post(
            f"/api/v1/admin/tenants/{target_id}/suspend",
            headers={"X-API-Key": api_key},
        )

        resp = client.post(
            f"/api/v1/admin/tenants/{target_id}/activate",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    def test_create_api_key_endpoint(self, mt_client, sample_tenant):
        client, api_key = mt_client
        resp = client.post(
            f"/api/v1/admin/tenants/{sample_tenant.tenant_id}/keys",
            json={"name": "new-api-key", "permissions": ["submit", "read"]},
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["key_id"].startswith("key_")
        assert data["api_key"].startswith("ocr_")
        assert data["name"] == "new-api-key"
        assert "submit" in data["permissions"]

    def test_revoke_api_key_endpoint(self, mt_client, sample_tenant):
        client, api_key = mt_client

        # Create a key to revoke
        resp = client.post(
            f"/api/v1/admin/tenants/{sample_tenant.tenant_id}/keys",
            json={"name": "to-revoke"},
            headers={"X-API-Key": api_key},
        )
        key_id = resp.json()["key_id"]

        resp = client.delete(
            f"/api/v1/admin/tenants/{sample_tenant.tenant_id}/keys/{key_id}",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 204

    def test_revoke_nonexistent_key_endpoint(self, mt_client, sample_tenant):
        client, api_key = mt_client
        resp = client.delete(
            f"/api/v1/admin/tenants/{sample_tenant.tenant_id}/keys/key_000000000000",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 404

    def test_get_usage_endpoint(self, mt_client, sample_tenant):
        client, api_key = mt_client
        resp = client.get(
            f"/api/v1/admin/tenants/{sample_tenant.tenant_id}/usage",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == sample_tenant.tenant_id
        assert "period" in data
        assert "jobs_submitted" in data
        assert "estimated_costs" in data
        assert data["api_calls"] == 0

    def test_forbidden_request_does_not_increment_api_calls(
        self,
        mt_client,
        sample_tenant,
        session,
    ):
        client, api_key = mt_client
        from api.tenant_manager import create_tenant
        from api.usage import get_usage

        other_tenant = create_tenant(name="Other Tenant", session=session)

        resp = client.get(
            f"/api/v1/admin/tenants/{other_tenant.tenant_id}",
            headers={"X-API-Key": api_key},
        )

        assert resp.status_code == 403
        usage = get_usage(sample_tenant.tenant_id, session=session)
        assert usage is None or usage.api_calls == 0

    def test_empty_period_usage_response_keeps_rate_snapshot(
        self,
        mt_client,
        sample_tenant,
    ):
        client, api_key = mt_client

        with (
            patch("api.config.TENANT_COST_PER_PAGE_USD", 0.02),
            patch("api.usage.config.TENANT_COST_PER_PAGE_USD", 0.02),
            patch("api.config.TENANT_COST_PER_GIB_INGESTED_USD", 0.5),
            patch("api.usage.config.TENANT_COST_PER_GIB_INGESTED_USD", 0.5),
            patch("api.config.TENANT_COST_PER_API_CALL_USD", 0.01),
            patch("api.usage.config.TENANT_COST_PER_API_CALL_USD", 0.01),
            patch("api.config.TENANT_COST_PER_PROCESSING_HOUR_USD", 3.0),
            patch("api.usage.config.TENANT_COST_PER_PROCESSING_HOUR_USD", 3.0),
        ):
            resp = client.get(
                f"/api/v1/admin/tenants/{sample_tenant.tenant_id}/usage?period=2025-12",
                headers={"X-API-Key": api_key},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["api_calls"] == 0
        assert data["estimated_costs"]["total_cost_usd"] == pytest.approx(0.0)
        assert data["estimated_costs"]["rates"]["per_page_usd"] == pytest.approx(0.02)
        assert data["estimated_costs"]["rates"]["per_gib_ingested_usd"] == pytest.approx(
            0.5
        )
        assert data["estimated_costs"]["rates"]["per_api_call_usd"] == pytest.approx(0.01)
        assert data["estimated_costs"]["rates"]["per_processing_hour_usd"] == pytest.approx(
            3.0
        )

    def test_usage_endpoint_returns_processing_and_cost_fields(self, mt_client, sample_tenant, session):
        client, api_key = mt_client
        session.add(
            UsageRecord(
                tenant_id=sample_tenant.tenant_id,
                period="2026-03",
                jobs_submitted=2,
                pages_processed=40,
                storage_bytes_used=1024**3,
                api_calls=0,
                processing_seconds=7200.0,
            )
        )
        session.commit()

        with (
            patch("api.config.TENANT_COST_PER_PAGE_USD", 0.01),
            patch("api.usage.config.TENANT_COST_PER_PAGE_USD", 0.01),
            patch("api.config.TENANT_COST_PER_GIB_INGESTED_USD", 0.25),
            patch("api.usage.config.TENANT_COST_PER_GIB_INGESTED_USD", 0.25),
            patch("api.config.TENANT_COST_PER_API_CALL_USD", 0.0),
            patch("api.usage.config.TENANT_COST_PER_API_CALL_USD", 0.0),
            patch("api.config.TENANT_COST_PER_PROCESSING_HOUR_USD", 2.0),
            patch("api.usage.config.TENANT_COST_PER_PROCESSING_HOUR_USD", 2.0),
        ):
            resp = client.get(
                f"/api/v1/admin/tenants/{sample_tenant.tenant_id}/usage?period=2026-03",
                headers={"X-API-Key": api_key},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["processing_seconds"] == pytest.approx(7200.0)
        assert data["estimated_costs"]["page_cost_usd"] == pytest.approx(0.4)
        assert data["estimated_costs"]["storage_ingest_cost_usd"] == pytest.approx(0.25)
        assert data["estimated_costs"]["processing_cost_usd"] == pytest.approx(4.0)

    def test_get_slo_endpoint_returns_empty_window_defaults(self, mt_client, sample_tenant):
        client, api_key = mt_client
        resp = client.get(
            f"/api/v1/admin/tenants/{sample_tenant.tenant_id}/slo",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == sample_tenant.tenant_id
        assert data["jobs_total"] == 0
        assert data["success_rate"] == pytest.approx(1.0)
        assert data["status"]["overall_met"] is True

    def test_get_slo_endpoint_reports_failure_when_targets_missed(
        self,
        mt_client,
        sample_tenant,
        session,
    ):
        client, api_key = mt_client
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        session.add_all(
            [
                Job(
                    job_id="job_slo_1",
                    status="completed",
                    source_file="a.pdf",
                    tenant_id=sample_tenant.tenant_id,
                    created_at=now - timedelta(hours=1),
                    started_at=now - timedelta(minutes=50),
                    completed_at=now - timedelta(minutes=20),
                    pages_completed=10,
                    processing_time=1800.0,
                ),
                Job(
                    job_id="job_slo_2",
                    status="failed",
                    source_file="b.pdf",
                    tenant_id=sample_tenant.tenant_id,
                    created_at=now - timedelta(hours=2),
                    started_at=now - timedelta(hours=2),
                    completed_at=now - timedelta(hours=1, minutes=30),
                    pages_completed=1,
                    processing_time=2000.0,
                ),
            ]
        )
        session.commit()

        with (
            patch("api.config.TENANT_SLO_WINDOW_HOURS", 24),
            patch("api.slo.config.TENANT_SLO_WINDOW_HOURS", 24),
            patch("api.config.TENANT_SLO_TARGET_SUCCESS_RATE", 0.99),
            patch("api.slo.config.TENANT_SLO_TARGET_SUCCESS_RATE", 0.99),
            patch("api.config.TENANT_SLO_TARGET_P95_PROCESSING_SECONDS", 1200.0),
            patch("api.slo.config.TENANT_SLO_TARGET_P95_PROCESSING_SECONDS", 1200.0),
        ):
            resp = client.get(
                f"/api/v1/admin/tenants/{sample_tenant.tenant_id}/slo",
                headers={"X-API-Key": api_key},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs_total"] == 2
        assert data["completed_jobs"] == 1
        assert data["failed_jobs"] == 1
        assert data["success_rate"] == pytest.approx(0.5)
        assert data["p95_processing_seconds"] == pytest.approx(1800.0)
        assert data["status"]["success_rate_met"] is False
        assert data["status"]["p95_processing_met"] is False
        assert data["status"]["overall_met"] is False

    def test_get_slo_endpoint_reports_passing_window(self, mt_client, sample_tenant, session):
        client, api_key = mt_client
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        session.add_all(
            [
                Job(
                    job_id="job_slo_pass_1",
                    status="completed",
                    source_file="a.pdf",
                    tenant_id=sample_tenant.tenant_id,
                    created_at=now - timedelta(hours=1),
                    started_at=now - timedelta(minutes=55),
                    completed_at=now - timedelta(minutes=45),
                    pages_completed=20,
                    processing_time=300.0,
                ),
                Job(
                    job_id="job_slo_pass_2",
                    status="completed",
                    source_file="b.pdf",
                    tenant_id=sample_tenant.tenant_id,
                    created_at=now - timedelta(hours=3),
                    started_at=now - timedelta(hours=3),
                    completed_at=now - timedelta(hours=2, minutes=45),
                    pages_completed=10,
                    processing_time=600.0,
                ),
            ]
        )
        session.commit()

        with (
            patch("api.config.TENANT_SLO_TARGET_SUCCESS_RATE", 0.9),
            patch("api.slo.config.TENANT_SLO_TARGET_SUCCESS_RATE", 0.9),
            patch("api.config.TENANT_SLO_TARGET_P95_PROCESSING_SECONDS", 900.0),
            patch("api.slo.config.TENANT_SLO_TARGET_P95_PROCESSING_SECONDS", 900.0),
        ):
            resp = client.get(
                f"/api/v1/admin/tenants/{sample_tenant.tenant_id}/slo?window_hours=24",
                headers={"X-API-Key": api_key},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success_rate"] == pytest.approx(1.0)
        assert data["pages_processed"] == 30
        assert data["throughput_jobs_per_hour"] == pytest.approx(2 / 24, abs=1e-6)
        assert data["throughput_pages_per_hour"] == pytest.approx(30 / 24, abs=1e-6)
        assert data["status"]["overall_met"] is True

    def test_get_slo_endpoint_rejects_non_positive_window(self, mt_client, sample_tenant):
        client, api_key = mt_client

        resp = client.get(
            f"/api/v1/admin/tenants/{sample_tenant.tenant_id}/slo?window_hours=0",
            headers={"X-API-Key": api_key},
        )

        assert resp.status_code == 422

    def test_tenant_admin_cannot_create_other_tenant(self, mt_client):
        client, api_key = mt_client
        resp = client.post(
            "/api/v1/admin/tenants",
            json={"name": "Forbidden Tenant"},
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 403

    def test_tenant_admin_cannot_access_other_tenant(self, mt_client, session):
        client, api_key = mt_client
        from api.tenant_manager import create_tenant

        other_tenant = create_tenant(name="Other Tenant", session=session)

        resp = client.get(
            f"/api/v1/admin/tenants/{other_tenant.tenant_id}",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 403

    def test_tenant_admin_cannot_manage_other_tenant_keys(self, mt_client, session):
        client, api_key = mt_client
        from api.tenant_manager import create_api_key, create_tenant

        other_tenant = create_tenant(name="Other Tenant", session=session)
        key_id, _ = create_api_key(
            other_tenant.tenant_id,
            permissions=["submit", "read"],
            session=session,
        )

        resp = client.delete(
            f"/api/v1/admin/tenants/{other_tenant.tenant_id}/keys/{key_id}",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 403

    def test_tenant_admin_cannot_create_platform_admin_key(self, mt_client, sample_tenant):
        client, api_key = mt_client
        resp = client.post(
            f"/api/v1/admin/tenants/{sample_tenant.tenant_id}/keys",
            json={"name": "escalate", "permissions": ["submit", "read", "admin", "platform_admin"]},
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 403

    def test_admin_requires_auth(self, mt_client):
        client, _ = mt_client
        resp = client.get("/api/v1/admin/tenants")
        assert resp.status_code == 401

    def test_non_admin_key_rejected(self, mt_client, sample_tenant, session):
        client, _ = mt_client
        from api.tenant_manager import create_api_key

        # Create a key with only read permission (no admin)
        _, reader_key = create_api_key(
            sample_tenant.tenant_id,
            permissions=["read"],
            session=session,
        )
        resp = client.get(
            "/api/v1/admin/tenants",
            headers={"X-API-Key": reader_key},
        )
        assert resp.status_code == 403

    def test_invalid_tenant_id_format(self, mt_client):
        client, api_key = mt_client
        resp = client.get(
            "/api/v1/admin/tenants/bad_format",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 400

    def test_invalid_key_id_format(self, mt_client, sample_tenant):
        client, api_key = mt_client
        resp = client.delete(
            f"/api/v1/admin/tenants/{sample_tenant.tenant_id}/keys/bad_format",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 400


# ===========================================================================
# TestTenantJobAPIIsolation
# ===========================================================================


class TestTenantJobAPIIsolation:
    """API-level tenant isolation for job visibility and control."""

    def test_tenant_job_list_returns_only_owned_jobs(self, mt_client, sample_tenant, session):
        client, api_key = mt_client
        from api.tenant_manager import create_tenant

        other_tenant = create_tenant(name="Other Tenant", session=session)
        session.add_all(
            [
                Job(
                    job_id="job_aaaaaaaaaaaa",
                    status="completed",
                    source_file="self.pdf",
                    tenant_id=sample_tenant.tenant_id,
                ),
                Job(
                    job_id="job_bbbbbbbbbbbb",
                    status="completed",
                    source_file="other.pdf",
                    tenant_id=other_tenant.tenant_id,
                ),
            ]
        )
        session.commit()

        resp = client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["jobs"][0]["job_id"] == "job_aaaaaaaaaaaa"

    def test_tenant_cannot_fetch_other_tenant_job(self, mt_client, session):
        client, api_key = mt_client
        from api.tenant_manager import create_tenant

        other_tenant = create_tenant(name="Other Tenant", session=session)
        session.add(
            Job(
                job_id="job_cccccccccccc",
                status="completed",
                source_file="other.pdf",
                tenant_id=other_tenant.tenant_id,
            )
        )
        session.commit()

        resp = client.get(
            "/api/v1/jobs/job_cccccccccccc",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 404

    def test_tenant_cannot_cancel_other_tenant_job(self, mt_client, session):
        client, api_key = mt_client
        from api.tenant_manager import create_tenant

        other_tenant = create_tenant(name="Other Tenant", session=session)
        session.add(
            Job(
                job_id="job_dddddddddddd",
                status="processing",
                source_file="other.pdf",
                tenant_id=other_tenant.tenant_id,
            )
        )
        session.commit()

        resp = client.delete(
            "/api/v1/jobs/job_dddddddddddd",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 404


# ===========================================================================
# TestTenantQuotaAPI
# ===========================================================================


class TestTenantQuotaAPI:
    """Quota checks are enforced on the API submission path."""

    def test_submit_enforces_concurrent_job_quota(self, mt_client, sample_tenant, session):
        client, api_key = mt_client
        sample_tenant.max_concurrent_jobs = 1
        session.commit()

        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            resp1 = client.post(
                "/api/v1/jobs",
                files={"file": ("first.pdf", b"first-content", "application/pdf")},
                headers={"X-API-Key": api_key},
            )
            resp2 = client.post(
                "/api/v1/jobs",
                files={"file": ("second.pdf", b"second-content", "application/pdf")},
                headers={"X-API-Key": api_key},
            )

        assert resp1.status_code == 201
        assert resp2.status_code == 429
        assert resp2.json()["detail"]["error"] == "quota_exceeded"
        assert resp2.json()["detail"]["details"]["limit_type"] == "concurrent_jobs"

    def test_submit_enforces_storage_quota(self, mt_client, sample_tenant, session):
        client, api_key = mt_client
        sample_tenant.max_storage_bytes = 8
        session.commit()

        resp = client.post(
            "/api/v1/jobs",
            files={"file": ("large.pdf", b"0123456789", "application/pdf")},
            headers={"X-API-Key": api_key},
        )

        assert resp.status_code == 429
        assert resp.json()["detail"]["error"] == "quota_exceeded"
        assert resp.json()["detail"]["details"]["limit_type"] == "storage_bytes"


# ===========================================================================
# TestMigration
# ===========================================================================


class TestMigration:
    """Tests for schema migration — adding tenant tables to existing DB."""

    def test_ensure_tenant_schema_creates_tables(self, tmp_path):
        """Migration should create tenant tables on a fresh DB."""
        reset_engine()
        db_file = str(tmp_path / "migration_test.db")
        with (
            patch("api.config.DB_PATH", db_file),
            patch("api.database.DB_PATH", db_file),
        ):
            reset_engine()
            engine = get_engine(db_file)
            from sqlalchemy import inspect as sa_inspect

            insp = sa_inspect(engine)
            tables = insp.get_table_names()
            assert "tenants" in tables
            assert "tenant_api_keys" in tables
            assert "usage_records" in tables
            usage_indexes = {idx["name"] for idx in insp.get_indexes("usage_records")}
            assert "uq_usage_records_tenant_period" in usage_indexes
            reset_engine()

    def test_jobs_table_has_tenant_id(self, tmp_path):
        """Migration should add tenant_id column to jobs table."""
        reset_engine()
        db_file = str(tmp_path / "migration_col_test.db")
        with (
            patch("api.config.DB_PATH", db_file),
            patch("api.database.DB_PATH", db_file),
        ):
            reset_engine()
            engine = get_engine(db_file)
            from sqlalchemy import inspect as sa_inspect

            insp = sa_inspect(engine)
            columns = {col["name"] for col in insp.get_columns("jobs")}
            assert "tenant_id" in columns
            reset_engine()

    def test_existing_jobs_have_null_tenant(self, session):
        """Jobs created before multitenancy should have tenant_id=None."""
        job = Job(
            job_id="job_premigration",
            status="completed",
            source_file="old.pdf",
        )
        session.add(job)
        session.commit()

        fetched = session.get(Job, "job_premigration")
        assert fetched.tenant_id is None

    def test_migration_idempotent(self, tmp_path):
        """Running migration twice should not fail."""
        reset_engine()
        db_file = str(tmp_path / "idem_test.db")
        with (
            patch("api.config.DB_PATH", db_file),
            patch("api.database.DB_PATH", db_file),
        ):
            reset_engine()
            engine = get_engine(db_file)

            # Run migration again manually
            from api.database import _ensure_tenant_schema

            _ensure_tenant_schema(engine)  # Should not raise
            reset_engine()


# ===========================================================================
# TestMultiTenantAuth
# ===========================================================================


class TestMultiTenantAuth:
    """Tests for authentication flow with multi-tenancy enabled."""

    def test_tenant_key_authenticates(self, mt_client):
        """A valid tenant API key should authenticate successfully."""
        client, api_key = mt_client
        resp = client.get(
            "/api/v1/health",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200

    def test_invalid_key_rejected(self, mt_client):
        """An invalid key should be rejected with 401."""
        client, _ = mt_client
        resp = client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "ocr_invalid_key_does_not_exist"},
        )
        assert resp.status_code == 401

    def test_no_key_rejected(self, mt_client):
        """Missing API key should return 401."""
        client, _ = mt_client
        # Health is exempt, try jobs endpoint
        resp = client.get("/api/v1/jobs")
        assert resp.status_code == 401

    def test_health_exempt_from_auth(self, mt_client):
        """Health endpoint should work without API key."""
        client, _ = mt_client
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
