"""Tests for Role-Based Access Control (RBAC) on API endpoints.

Covers:
- TestAuthIdentity: create with all fields, minimal fields, frozen immutability, defaults
- TestRequireRole: admin always passes, operator allowed, viewer denied, viewer allowed,
  no identity defaults to admin
- TestAPIKeyAuth: API key submit/list/health, wrong key rejected, no credentials rejected
- TestBearerAuth: valid Bearer token, viewer denied on submit, viewer can list/read,
  admin can cancel, invalid token falls through
- TestDualAuth: API key takes priority over Bearer when both provided, API key still
  works with OAuth2 enabled
- TestUnauthenticated: no credentials with ALLOW_UNAUTHENTICATED=true gets viewer role
- TestEndpointProtection: submit_job requires admin/operator, cancel_job requires
  admin/operator, retry_job requires admin/operator, list_jobs allows all roles,
  get_job_status allows all roles
- TestAPIKeyRoleConfig: APIKEY_ROLE=operator can read, APIKEY_ROLE=viewer denied submit
- TestWebSocketAuth: token auth frame validates JWT, api_key param works,
  invalid token rejected, no credentials rejected, token query param removed
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.identity import AuthIdentity

# ---------------------------------------------------------------------------
# JWT helpers -- generate JWTs for testing using PyJWT
# ---------------------------------------------------------------------------

try:
    import json as _json

    import jwt as pyjwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from jwt.algorithms import RSAAlgorithm

    _TEST_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _TEST_PUBLIC_KEY = _TEST_PRIVATE_KEY.public_key()

    _TEST_PRIVATE_PEM = _TEST_PRIVATE_KEY.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()

    _TEST_PUBLIC_PEM = _TEST_PUBLIC_KEY.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()

    _TEST_JWK = _json.loads(RSAAlgorithm.to_jwk(_TEST_PUBLIC_KEY))
    _TEST_JWK["kid"] = "rbac-test-key"
    _TEST_JWK["use"] = "sig"
    _TEST_JWK["alg"] = "RS256"
    _TEST_JWKS_KEYS = [_TEST_JWK]
    _HAS_JOSE = True  # Keep variable name for skipif compatibility
except ImportError:
    _HAS_JOSE = False
    _TEST_PRIVATE_PEM = ""
    _TEST_JWKS_KEYS = []


def _make_jwt(role: str = "operator", subject: str = "test-user") -> str:
    """Create a signed JWT with the given role."""
    now = int(time.time())
    payload = {
        "sub": subject,
        "iss": "https://test-issuer.example.com",
        "aud": "test-audience",
        "iat": now,
        "nbf": now,
        "exp": now + 3600,
        "roles": [role],
    }
    return pyjwt.encode(payload, _TEST_PRIVATE_PEM, algorithm="RS256")


def _make_sample_pdf(path: Path) -> Path:
    """Create a minimal valid PDF file."""
    path.write_bytes(
        b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
    )
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_oauth2():
    """Reset OAuth2 module state between tests."""
    from api.oauth2 import reset_jwks_cache

    reset_jwks_cache()
    yield
    reset_jwks_cache()


@pytest.fixture()
def client(tmp_path):
    """FastAPI TestClient with API key auth and isolated dirs."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)

    with patch("api.config.SOURCE_FOLDER", str(source)), \
         patch("api.config.OUTPUT_FOLDER", str(output)), \
         patch("api.config.OCR_API_KEY", "test-key-123"), \
         patch("api.auth.OCR_API_KEY", "test-key-123"), \
         patch("api.config.OAUTH2_ENABLED", False), \
         patch("api.oauth2_config.OAUTH2_ENABLED", False), \
         patch("api.oauth2_config.APIKEY_ROLE", "admin"), \
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
        yield TestClient(app)


@pytest.fixture()
def oauth2_client(tmp_path):
    """FastAPI TestClient with OAuth2 enabled + API key + JWKS cache."""
    import threading

    from api.oauth2 import JWKSCache, set_jwks_cache

    # Pre-populate the JWKS cache with test keys
    cache = JWKSCache.__new__(JWKSCache)
    cache._jwks_uri = "https://test-issuer.example.com/.well-known/jwks.json"
    cache._keys = _TEST_JWKS_KEYS
    cache._fetched_at = time.time()
    cache._ttl = 3600
    cache._lock = threading.Lock()
    set_jwks_cache(cache)

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)

    with patch("api.config.SOURCE_FOLDER", str(source)), \
         patch("api.config.OUTPUT_FOLDER", str(output)), \
         patch("api.config.OCR_API_KEY", "test-key-123"), \
         patch("api.auth.OCR_API_KEY", "test-key-123"), \
         patch("api.config.OAUTH2_ENABLED", True), \
         patch("api.oauth2_config.OAUTH2_ENABLED", True), \
         patch("api.oauth2_config.OAUTH2_ISSUER", "https://test-issuer.example.com"), \
         patch("api.oauth2_config.OAUTH2_AUDIENCE", "test-audience"), \
         patch("api.oauth2_config.OAUTH2_JWKS_URI", "https://test-issuer.example.com/.well-known/jwks.json"), \
         patch("api.oauth2_config.OAUTH2_ALGORITHMS", ["RS256"]), \
         patch("api.oauth2_config.OAUTH2_ROLE_CLAIM", "roles"), \
         patch("api.oauth2_config.OAUTH2_ADMIN_ROLE", "admin"), \
         patch("api.oauth2_config.OAUTH2_OPERATOR_ROLE", "operator"), \
         patch("api.oauth2_config.OAUTH2_VIEWER_ROLE", "viewer"), \
         patch("api.oauth2_config.OAUTH2_DEFAULT_ROLE", "viewer"), \
         patch("api.oauth2_config.APIKEY_ROLE", "admin"), \
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
        yield TestClient(app)


@pytest.fixture()
def unauth_client(tmp_path):
    """FastAPI TestClient with no API key and ALLOW_UNAUTHENTICATED=true."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)

    with patch("api.config.SOURCE_FOLDER", str(source)), \
         patch("api.config.OUTPUT_FOLDER", str(output)), \
         patch("api.config.OCR_API_KEY", ""), \
         patch("api.auth.OCR_API_KEY", ""), \
         patch("api.config.ALLOW_UNAUTHENTICATED", True), \
         patch("api.auth.ALLOW_UNAUTHENTICATED", True), \
         patch("api.config.ANONYMOUS_ROLE", "viewer"), \
         patch("api.auth.ANONYMOUS_ROLE", "viewer"), \
         patch("api.config.OAUTH2_ENABLED", False), \
         patch("api.oauth2_config.OAUTH2_ENABLED", False), \
         patch("api.oauth2_config.APIKEY_ROLE", "operator"), \
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
        yield TestClient(app)


# ---------------------------------------------------------------------------
# TestAuthIdentity -- dataclass creation
# ---------------------------------------------------------------------------

class TestAuthIdentity:
    """Test AuthIdentity dataclass creation and properties."""

    def test_create_with_all_fields(self):
        """AuthIdentity can be created with all fields populated."""
        identity = AuthIdentity(
            subject="user-123",
            role="operator",
            auth_method="oauth2",
            email="user@example.com",
            name="Test User",
            claims={"sub": "user-123", "roles": ["operator"]},
        )
        assert identity.subject == "user-123"
        assert identity.role == "operator"
        assert identity.auth_method == "oauth2"
        assert identity.email == "user@example.com"
        assert identity.name == "Test User"
        assert "sub" in identity.claims

    def test_create_with_minimal_fields(self):
        """AuthIdentity can be created with only required fields."""
        identity = AuthIdentity(
            subject="apikey",
            role="admin",
            auth_method="apikey",
        )
        assert identity.subject == "apikey"
        assert identity.role == "admin"
        assert identity.auth_method == "apikey"

    def test_defaults_for_optional_fields(self):
        """Optional fields have correct defaults."""
        identity = AuthIdentity(subject="test", role="viewer", auth_method="apikey")
        assert identity.email is None
        assert identity.name is None
        assert identity.claims == {}

    def test_identity_is_frozen(self):
        """AuthIdentity is immutable (frozen dataclass)."""
        identity = AuthIdentity(subject="test", role="viewer", auth_method="apikey")
        with pytest.raises(AttributeError):
            identity.role = "admin"

    def test_get_identity_returns_default_when_no_state(self):
        """get_identity returns admin identity when no identity is set on request."""
        from api.identity import get_identity

        mock_request = MagicMock()
        mock_request.state = MagicMock(spec=[])  # No identity attribute
        identity = get_identity(mock_request)
        assert identity.role == "admin"
        assert identity.auth_method == "none"
        assert identity.subject == "anonymous"


# ---------------------------------------------------------------------------
# TestRequireRole -- unit tests for the dependency factory
# ---------------------------------------------------------------------------

class TestRequireRole:
    """Test the require_role() dependency factory in isolation."""

    @pytest.mark.anyio
    async def test_admin_always_passes(self):
        """Admin role passes any require_role() check."""
        from api.identity import require_role

        mock_request = MagicMock()
        mock_request.state.identity = AuthIdentity(
            subject="admin-user", role="admin", auth_method="apikey"
        )

        checker = require_role("operator")
        # Should not raise
        await checker(mock_request)

    @pytest.mark.anyio
    async def test_operator_passes_when_allowed(self):
        """Operator passes when 'operator' is in allowed roles."""
        from api.identity import require_role

        mock_request = MagicMock()
        mock_request.state.identity = AuthIdentity(
            subject="op-user", role="operator", auth_method="oauth2"
        )

        checker = require_role("admin", "operator")
        await checker(mock_request)

    @pytest.mark.anyio
    async def test_viewer_denied_on_write_endpoint(self):
        """Viewer is denied when only admin/operator are allowed."""
        from fastapi import HTTPException

        from api.identity import require_role

        mock_request = MagicMock()
        mock_request.state.identity = AuthIdentity(
            subject="view-user", role="viewer", auth_method="oauth2"
        )
        mock_request.url.path = "/api/v1/jobs"

        checker = require_role("admin", "operator")
        with pytest.raises(HTTPException) as exc_info:
            await checker(mock_request)
        assert exc_info.value.status_code == 403

    @pytest.mark.anyio
    async def test_viewer_passes_when_allowed(self):
        """Viewer passes when 'viewer' is in allowed roles."""
        from api.identity import require_role

        mock_request = MagicMock()
        mock_request.state.identity = AuthIdentity(
            subject="view-user", role="viewer", auth_method="oauth2"
        )

        checker = require_role("viewer", "operator", "admin")
        await checker(mock_request)

    @pytest.mark.anyio
    async def test_no_identity_defaults_admin(self):
        """When no identity is set, get_identity() returns admin which always passes."""
        from api.identity import require_role

        mock_request = MagicMock()
        mock_request.state = MagicMock(spec=[])  # No identity attribute

        checker = require_role("operator")
        # Should not raise because default is admin
        await checker(mock_request)

    @pytest.mark.anyio
    async def test_403_includes_role_info_in_detail(self):
        """403 response detail includes the caller's role and required roles."""
        from fastapi import HTTPException

        from api.identity import require_role

        mock_request = MagicMock()
        mock_request.state.identity = AuthIdentity(
            subject="view-user", role="viewer", auth_method="oauth2"
        )
        mock_request.url.path = "/api/v1/jobs"

        checker = require_role("operator")
        with pytest.raises(HTTPException) as exc_info:
            await checker(mock_request)

        detail = exc_info.value.detail
        assert "viewer" in detail["message"]
        assert "operator" in detail["message"]


# ---------------------------------------------------------------------------
# TestAPIKeyAuth -- API key backward compatibility
# ---------------------------------------------------------------------------

class TestAPIKeyAuth:
    """Verify existing API key auth still works and gets correct role."""

    def test_apikey_submit_job_allowed(self, client, tmp_path):
        """API key user can submit jobs (admin role by default)."""
        pdf = _make_sample_pdf(tmp_path / "source" / "test.pdf")

        with open(pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"priority": "normal"},
                headers={"X-API-Key": "test-key-123"},
            )
        assert resp.status_code == 201

    def test_apikey_list_jobs_allowed(self, client):
        """API key user can list jobs."""
        resp = client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200

    def test_apikey_health_no_auth_required(self, client):
        """Health endpoint requires no auth."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_wrong_apikey_rejected(self, client):
        """Wrong API key is rejected with 401."""
        resp = client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_no_credentials_rejected(self, client):
        """Request with no credentials is rejected with 401."""
        resp = client.get("/api/v1/jobs")
        assert resp.status_code == 401

    def test_apikey_sets_identity_with_apikey_role(self, client):
        """API key auth sets the APIKEY_ROLE (admin by default)."""
        resp = client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-key-123"},
        )
        # If APIKEY_ROLE=admin, the user should be able to access any endpoint
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# TestBearerAuth -- OAuth2 Bearer token integration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_JOSE, reason="PyJWT not installed")
class TestBearerAuth:
    """Test OAuth2 Bearer token auth through the middleware."""

    def test_bearer_operator_can_submit(self, oauth2_client, tmp_path):
        """OAuth2 user with operator role can submit jobs."""
        pdf = _make_sample_pdf(tmp_path / "source" / "test.pdf")

        token = _make_jwt(role="operator")
        with open(pdf, "rb") as f:
            resp = oauth2_client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"priority": "normal"},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 201

    def test_bearer_admin_can_submit(self, oauth2_client, tmp_path):
        """OAuth2 user with admin role can submit jobs."""
        pdf = _make_sample_pdf(tmp_path / "source" / "test.pdf")

        token = _make_jwt(role="admin")
        with open(pdf, "rb") as f:
            resp = oauth2_client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"priority": "normal"},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 201

    def test_bearer_viewer_cannot_submit(self, oauth2_client, tmp_path):
        """OAuth2 user with viewer role is denied from submitting jobs."""
        pdf = _make_sample_pdf(tmp_path / "source" / "test.pdf")

        token = _make_jwt(role="viewer")
        with open(pdf, "rb") as f:
            resp = oauth2_client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"priority": "normal"},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 403

    def test_bearer_viewer_can_list_jobs(self, oauth2_client):
        """OAuth2 user with viewer role can list jobs (read-only)."""
        token = _make_jwt(role="viewer")
        resp = oauth2_client.get(
            "/api/v1/jobs",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_bearer_viewer_can_get_job_status(self, oauth2_client):
        """OAuth2 user with viewer role can get individual job status."""
        token = _make_jwt(role="viewer")
        resp = oauth2_client.get(
            "/api/v1/jobs/job_000000000000",
            headers={"Authorization": f"Bearer {token}"},
        )
        # 404 because job doesn't exist -- RBAC should not block
        assert resp.status_code == 404

    def test_bearer_admin_can_cancel(self, oauth2_client):
        """OAuth2 admin can cancel jobs."""
        token = _make_jwt(role="admin")
        resp = oauth2_client.delete(
            "/api/v1/jobs/job_000000000000",
            headers={"Authorization": f"Bearer {token}"},
        )
        # 404 is expected -- RBAC should not block
        assert resp.status_code == 404

    def test_bearer_viewer_cannot_cancel(self, oauth2_client):
        """OAuth2 viewer cannot cancel jobs."""
        token = _make_jwt(role="viewer")
        resp = oauth2_client.delete(
            "/api/v1/jobs/job_000000000000",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_bearer_viewer_cannot_retry(self, oauth2_client):
        """OAuth2 viewer cannot retry jobs."""
        token = _make_jwt(role="viewer")
        resp = oauth2_client.post(
            "/api/v1/jobs/job_000000000000/retry",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_invalid_bearer_token_falls_through(self, oauth2_client):
        """Invalid Bearer token falls through to 401 when API key is required."""
        resp = oauth2_client.get(
            "/api/v1/jobs",
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# TestDualAuth -- API key + Bearer coexistence
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_JOSE, reason="PyJWT not installed")
class TestDualAuth:
    """Test behavior when both API key and Bearer token are provided."""

    def test_apikey_takes_priority_over_bearer(self, oauth2_client):
        """When both API key and Bearer token are provided, API key wins."""
        token = _make_jwt(role="viewer")
        resp = oauth2_client.get(
            "/api/v1/jobs",
            headers={
                "X-API-Key": "test-key-123",
                "Authorization": f"Bearer {token}",
            },
        )
        # API key = admin role, should work even though Bearer has viewer
        assert resp.status_code == 200

    def test_apikey_still_works_with_oauth2_enabled(self, oauth2_client):
        """API key auth still works when OAuth2 is also enabled."""
        resp = oauth2_client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200

    def test_bad_apikey_rejects_even_with_valid_bearer(self, oauth2_client):
        """A bad API key results in 401 even if a valid Bearer token is present."""
        token = _make_jwt(role="admin")
        resp = oauth2_client.get(
            "/api/v1/jobs",
            headers={
                "X-API-Key": "wrong-key",
                "Authorization": f"Bearer {token}",
            },
        )
        # Bad API key rejects immediately -- doesn't fall through to Bearer
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# TestUnauthenticated -- ALLOW_UNAUTHENTICATED=true behavior
# ---------------------------------------------------------------------------

class TestUnauthenticatedMode:
    """Test behavior when ALLOW_UNAUTHENTICATED=true and no API key."""

    def test_unauthenticated_gets_viewer_role(self, unauth_client):
        """Unauthenticated mode assigns viewer role (least privilege)."""
        resp = unauth_client.get("/api/v1/jobs")
        assert resp.status_code == 200

    def test_unauthenticated_cannot_submit(self, unauth_client, tmp_path):
        """Unauthenticated mode denies job submission (viewer lacks submit)."""
        pdf = _make_sample_pdf(tmp_path / "source" / "test.pdf")

        with open(pdf, "rb") as f:
            resp = unauth_client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"priority": "normal"},
            )
        assert resp.status_code == 403

    def test_unauthenticated_cannot_cancel(self, unauth_client):
        """Unauthenticated mode denies cancel (viewer lacks cancel)."""
        resp = unauth_client.delete("/api/v1/jobs/job_000000000000")
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# TestEndpointProtection -- permission matrix for all RBAC-protected endpoints
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_JOSE, reason="PyJWT not installed")
class TestEndpointProtection:
    """Test the permission matrix for each endpoint type."""

    def test_submit_requires_admin_or_operator(self, oauth2_client, tmp_path):
        """POST /api/v1/jobs requires admin or operator role."""
        pdf = _make_sample_pdf(tmp_path / "source" / "test.pdf")

        # Viewer should be denied
        token = _make_jwt(role="viewer")
        with open(pdf, "rb") as f:
            resp = oauth2_client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"priority": "normal"},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 403

    def test_cancel_requires_admin_or_operator(self, oauth2_client):
        """DELETE /api/v1/jobs/{id} requires admin or operator role."""
        token = _make_jwt(role="viewer")
        resp = oauth2_client.delete(
            "/api/v1/jobs/job_000000000000",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_retry_requires_admin_or_operator(self, oauth2_client):
        """POST /api/v1/jobs/{id}/retry requires admin or operator role."""
        token = _make_jwt(role="viewer")
        resp = oauth2_client.post(
            "/api/v1/jobs/job_000000000000/retry",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_list_jobs_allows_all_roles(self, oauth2_client):
        """GET /api/v1/jobs allows viewer role (no require_role on list)."""
        token = _make_jwt(role="viewer")
        resp = oauth2_client.get(
            "/api/v1/jobs",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_get_job_status_allows_all_roles(self, oauth2_client):
        """GET /api/v1/jobs/{id} allows viewer role (no require_role on status)."""
        token = _make_jwt(role="viewer")
        resp = oauth2_client.get(
            "/api/v1/jobs/job_000000000000",
            headers={"Authorization": f"Bearer {token}"},
        )
        # 404 because job doesn't exist -- but RBAC did not block
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestAPIKeyRoleConfig -- APIKEY_ROLE configuration
# ---------------------------------------------------------------------------

class TestAPIKeyRoleConfig:
    """Test that APIKEY_ROLE configuration is respected."""

    def test_apikey_role_operator_can_read(self, tmp_path):
        """API key user with APIKEY_ROLE=operator can list jobs."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir(exist_ok=True)
        output.mkdir(exist_ok=True)

        with patch("api.config.SOURCE_FOLDER", str(source)), \
             patch("api.config.OUTPUT_FOLDER", str(output)), \
             patch("api.config.OCR_API_KEY", "test-key-123"), \
             patch("api.auth.OCR_API_KEY", "test-key-123"), \
             patch("api.oauth2_config.APIKEY_ROLE", "operator"), \
             patch("api.config.OAUTH2_ENABLED", False), \
             patch("api.oauth2_config.OAUTH2_ENABLED", False), \
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
            tc = TestClient(app)

            resp = tc.get(
                "/api/v1/jobs",
                headers={"X-API-Key": "test-key-123"},
            )
            assert resp.status_code == 200

    def test_apikey_role_viewer_denied_submit(self, tmp_path):
        """API key user with APIKEY_ROLE=viewer cannot submit jobs."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir(exist_ok=True)
        output.mkdir(exist_ok=True)

        with patch("api.config.SOURCE_FOLDER", str(source)), \
             patch("api.config.OUTPUT_FOLDER", str(output)), \
             patch("api.config.OCR_API_KEY", "test-key-123"), \
             patch("api.auth.OCR_API_KEY", "test-key-123"), \
             patch("api.oauth2_config.APIKEY_ROLE", "viewer"), \
             patch("api.config.OAUTH2_ENABLED", False), \
             patch("api.oauth2_config.OAUTH2_ENABLED", False), \
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
            tc = TestClient(app)

            pdf = _make_sample_pdf(source / "test.pdf")

            with open(pdf, "rb") as f:
                resp = tc.post(
                    "/api/v1/jobs",
                    files={"file": ("test.pdf", f, "application/pdf")},
                    data={"priority": "normal"},
                    headers={"X-API-Key": "test-key-123"},
                )
            assert resp.status_code == 403


# ---------------------------------------------------------------------------
# TestWebSocketAuth -- WebSocket authentication via auth frame
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_JOSE, reason="PyJWT not installed")
class TestWebSocketAuth:
    """Test WebSocket authentication using auth frames.

    OAuth2 tokens are no longer accepted as query parameters.
    They must be sent via ``{"type": "auth", "token": "..."}``.
    """

    @pytest.mark.anyio
    async def test_ws_token_auth_frame_validates_jwt(self):
        """Auth frame with token validates JWT when OAuth2 is enabled."""
        import json as _json
        from unittest.mock import AsyncMock

        from api.routers.ws import _authenticate_ws

        mock_ws = MagicMock()
        mock_ws.state = SimpleNamespace()
        mock_ws.headers = {}
        token = _make_jwt(role="operator")
        mock_ws.receive_text = AsyncMock(
            return_value=_json.dumps({"type": "auth", "token": token}),
        )

        with patch("api.routers.ws.OCR_API_KEY", ""), \
             patch("api.oauth2_config.OAUTH2_ENABLED", True), \
             patch("api.oauth2.validate_jwt", return_value={"sub": "user", "roles": ["operator"]}), \
             patch("api.oauth2.extract_role", return_value="operator"):
            result = await _authenticate_ws(mock_ws)

        assert result is True
        assert mock_ws.state.identity.role == "operator"

    @pytest.mark.anyio
    async def test_ws_api_key_param_works(self):
        """WebSocket api_key parameter authenticates successfully."""
        from api.routers.ws import _authenticate_ws

        mock_ws = MagicMock()
        mock_ws.state = SimpleNamespace()

        with patch("api.routers.ws.OCR_API_KEY", "test-key-123"), \
             patch("api.oauth2_config.APIKEY_ROLE", "admin"):
            result = await _authenticate_ws(mock_ws, api_key="test-key-123")

        assert result is True
        assert mock_ws.state.identity.role == "admin"

    @pytest.mark.anyio
    async def test_ws_invalid_token_auth_frame_rejected(self):
        """Auth frame with invalid JWT token is rejected."""
        import json as _json
        from unittest.mock import AsyncMock

        from api.routers.ws import _authenticate_ws

        mock_ws = MagicMock()
        mock_ws.state = SimpleNamespace()
        mock_ws.headers = {}
        mock_ws.receive_text = AsyncMock(
            return_value=_json.dumps({"type": "auth", "token": "bad-token"}),
        )

        with patch("api.routers.ws.OCR_API_KEY", ""), \
             patch("api.oauth2_config.OAUTH2_ENABLED", True), \
             patch("api.oauth2.validate_jwt", side_effect=ValueError("invalid")):
            result = await _authenticate_ws(mock_ws)

        assert result is False

    @pytest.mark.anyio
    async def test_ws_no_credentials_rejected(self):
        """WebSocket with no credentials is rejected when auth is configured."""
        from unittest.mock import AsyncMock

        from api.routers.ws import _authenticate_ws

        mock_ws = MagicMock()
        mock_ws.state = SimpleNamespace()
        mock_ws.headers = {}
        mock_ws.receive_text = AsyncMock(
            return_value='{"type":"auth"}',
        )

        with patch("api.routers.ws.OCR_API_KEY", "test-key-123"):
            result = await _authenticate_ws(mock_ws)

        assert result is False

    @pytest.mark.anyio
    async def test_ws_no_auth_configured_allows_all(self):
        """WebSocket allows all connections when no auth is configured."""
        from api.routers.ws import _authenticate_ws

        mock_ws = MagicMock()
        mock_ws.state = SimpleNamespace()

        with patch("api.routers.ws.OCR_API_KEY", ""), \
             patch("api.oauth2_config.OAUTH2_ENABLED", False), \
             patch("api.auth.ALLOW_UNAUTHENTICATED", True), \
             patch("api.config.ANONYMOUS_ROLE", "viewer"):
            result = await _authenticate_ws(mock_ws)

        assert result is True
        assert mock_ws.state.identity.role == "viewer"

    @pytest.mark.anyio
    async def test_ws_wrong_api_key_rejected(self):
        """WebSocket with wrong API key is rejected."""
        from api.routers.ws import _authenticate_ws

        mock_ws = MagicMock()
        mock_ws.state = SimpleNamespace()

        with patch("api.routers.ws.OCR_API_KEY", "test-key-123"):
            result = await _authenticate_ws(mock_ws, api_key="wrong-key")

        assert result is False

    @pytest.mark.anyio
    async def test_ws_token_auth_frame_ignored_when_oauth2_disabled(self):
        """Auth frame token is ignored when OAuth2 is disabled."""
        import json as _json
        from unittest.mock import AsyncMock

        from api.routers.ws import _authenticate_ws

        mock_ws = MagicMock()
        mock_ws.state = SimpleNamespace()
        mock_ws.headers = {}
        mock_ws.receive_text = AsyncMock(
            return_value=_json.dumps({"type": "auth", "token": "some-token"}),
        )

        with patch("api.routers.ws.OCR_API_KEY", ""), \
             patch("api.oauth2_config.OAUTH2_ENABLED", False), \
             patch("api.auth.ALLOW_UNAUTHENTICATED", False):
            result = await _authenticate_ws(mock_ws)

        assert result is False

    @pytest.mark.anyio
    async def test_ws_oauth2_only_requires_token(self):
        """OAuth2-only deployments must not allow anonymous WebSocket access."""
        from unittest.mock import AsyncMock

        from api.routers.ws import _authenticate_ws

        mock_ws = MagicMock()
        mock_ws.state = SimpleNamespace()
        mock_ws.headers = {}
        mock_ws.receive_text = AsyncMock(
            return_value='{"type":"auth"}',
        )

        with patch("api.routers.ws.OCR_API_KEY", ""), \
             patch("api.oauth2_config.OAUTH2_ENABLED", True), \
             patch("api.auth.ALLOW_UNAUTHENTICATED", False):
            result = await _authenticate_ws(mock_ws)

        assert result is False

    @pytest.mark.anyio
    async def test_ws_oauth2_only_valid_token_auth_frame_sets_identity(self):
        """OAuth2-only deployments accept valid JWTs via auth frame and attach identity."""
        import json as _json
        from unittest.mock import AsyncMock

        from api.routers.ws import _authenticate_ws

        mock_ws = MagicMock()
        mock_ws.state = SimpleNamespace()
        mock_ws.headers = {}
        mock_ws.receive_text = AsyncMock(
            return_value=_json.dumps({"type": "auth", "token": "valid-token"}),
        )

        claims = {"sub": "viewer-user", "roles": ["viewer"], "email": "viewer@example.com"}
        with patch("api.routers.ws.OCR_API_KEY", ""), \
             patch("api.oauth2_config.OAUTH2_ENABLED", True), \
             patch("api.auth.ALLOW_UNAUTHENTICATED", False), \
             patch("api.oauth2.validate_jwt", return_value=claims), \
             patch("api.oauth2.extract_role", return_value="viewer"):
            result = await _authenticate_ws(mock_ws)

        assert result is True
        assert mock_ws.state.identity.subject == "viewer-user"
        assert mock_ws.state.identity.role == "viewer"

    @pytest.mark.anyio
    async def test_ws_token_query_param_no_longer_accepted(self):
        """Endpoint signature no longer accepts token as a query parameter."""
        import inspect as _inspect

        from api.routers.ws import job_progress_ws

        sig = _inspect.signature(job_progress_ws)
        param_names = list(sig.parameters.keys())
        assert "token" not in param_names, \
            "token query parameter must be removed ( credential leakage)"
