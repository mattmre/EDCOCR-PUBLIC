"""E2E smoke tests for API server.

These tests verify the API server starts correctly and core endpoints respond.
They do not require GPU, Docker, or external services.
Designed to run in CI as a lightweight integration gate.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.database import get_engine, reset_engine

# API key used by the smoke test client
_SMOKE_KEY = "smoke-test-key"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Give each test a fresh SQLite database and temp dirs."""
    reset_engine()
    db_file = str(tmp_path / "smoke_jobs.db")
    output = tmp_path / "output"
    output.mkdir()
    source = tmp_path / "source"
    source.mkdir()

    with patch("api.config.DB_PATH", db_file), \
         patch("api.database.DB_PATH", db_file), \
         patch("api.config.SOURCE_FOLDER", str(source)), \
         patch("api.config.OUTPUT_FOLDER", str(output)):
        reset_engine()
        get_engine(db_file)
        yield
        reset_engine()


@pytest.fixture()
def client(tmp_path):
    """FastAPI TestClient with auth enabled and isolated DB/dirs."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)

    with patch("api.auth.OCR_API_KEY", _SMOKE_KEY), \
         patch("api.config.SOURCE_FOLDER", str(source)), \
         patch("api.config.OUTPUT_FOLDER", str(output)), \
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
# Server Startup
# ---------------------------------------------------------------------------


class TestServerStartup:
    """Verify the FastAPI app boots without errors."""

    def test_app_creates_successfully(self):
        from api.main import create_app

        app = create_app()
        assert app is not None
        assert app.title == "EDCOCR API"

    def test_routes_registered(self):
        from api.main import create_app

        app = create_app()
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/api/v1/health" in route_paths
        assert "/api/v1/health/detailed" in route_paths
        # /docs and /redoc are conditional on EXPOSE_API_DOCS env var

    def test_jobs_route_registered(self):
        from api.main import create_app

        app = create_app()
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        jobs_routes = [p for p in route_paths if p.startswith("/api/v1/jobs")]
        assert len(jobs_routes) > 0


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealthSmoke:
    def test_health_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"

    def test_health_no_auth_required(self, client):
        """Health endpoint should work without API key."""
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_health_response_shape(self, client):
        r = client.get("/api/v1/health")
        data = r.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data

    def test_detailed_health_returns_200(self, client):
        r = client.get("/api/v1/health/detailed")
        assert r.status_code == 200
        data = r.json()
        assert "checks" in data
        assert isinstance(data["checks"], dict)


# ---------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------


class TestOpenAPISmoke:
    @pytest.fixture(autouse=True)
    def _enable_docs(self):
        with patch("api.main.EXPOSE_API_DOCS", True), \
             patch("api.auth.EXPOSE_API_DOCS", True):
            yield

    def test_openapi_spec_accessible(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        assert spec["info"]["title"] == "EDCOCR API"

    def test_docs_page_loads(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_redoc_page_loads(self, client):
        r = client.get("/redoc")
        assert r.status_code == 200

    def test_openapi_has_security_schemes(self, client):
        spec = client.get("/openapi.json").json()
        schemes = spec.get("components", {}).get("securitySchemes", {})
        assert "ApiKeyAuth" in schemes
        assert schemes["ApiKeyAuth"]["type"] == "apiKey"
        assert schemes["ApiKeyAuth"]["in"] == "header"
        assert schemes["ApiKeyAuth"]["name"] == "X-API-Key"

    def test_openapi_has_paths(self, client):
        spec = client.get("/openapi.json").json()
        paths = spec.get("paths", {})
        assert "/api/v1/health" in paths
        assert len(paths) > 3


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthSmoke:
    def test_unauthenticated_request_rejected(self, client):
        """Requests without an API key should be rejected."""
        r = client.get("/api/v1/jobs")
        assert r.status_code in (401, 403)

    def test_valid_api_key_accepted(self, client):
        """Requests with the correct API key should succeed."""
        r = client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": _SMOKE_KEY},
        )
        assert r.status_code == 200

    def test_invalid_api_key_rejected(self, client):
        """Requests with a wrong API key should be rejected."""
        r = client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code in (401, 403)

    def test_exempt_paths_no_auth(self, client):
        """Auth-exempt paths should respond without credentials.

        Only health endpoints are unconditionally exempt.  OpenAPI docs
        (/docs, /redoc, /openapi.json) are gated behind the
        ``EXPOSE_API_DOCS`` env var and are not mounted by default,
        so they should return 401 or 404 here (never 200).
        """
        r = client.get("/api/v1/health")
        assert r.status_code == 200

        for path in ("/docs", "/openapi.json", "/redoc"):
            r = client.get(path)
            assert r.status_code in (401, 404), (
                f"{path} returned {r.status_code} -- docs should be gated by "
                "EXPOSE_API_DOCS and must not be reachable unauthenticated"
            )


# ---------------------------------------------------------------------------
# Job Lifecycle
# ---------------------------------------------------------------------------


class TestJobLifecycleSmoke:
    def test_list_jobs_empty(self, client):
        r = client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": _SMOKE_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)
        assert data["total"] == 0

    def test_get_nonexistent_job(self, client):
        r = client.get(
            "/api/v1/jobs/job_000000000000",
            headers={"X-API-Key": _SMOKE_KEY},
        )
        assert r.status_code == 404

    def test_cancel_nonexistent_job(self, client):
        r = client.delete(
            "/api/v1/jobs/job_000000000000",
            headers={"X-API-Key": _SMOKE_KEY},
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Batch Lifecycle
# ---------------------------------------------------------------------------


class TestBatchSmoke:
    def test_list_batches_empty(self, client):
        r = client.get(
            "/api/v1/jobs/batch",
            headers={"X-API-Key": _SMOKE_KEY},
        )
        assert r.status_code == 200
        data = r.json()
        assert "batches" in data
        assert isinstance(data["batches"], list)

    def test_get_nonexistent_batch(self, client):
        r = client.get(
            "/api/v1/jobs/batch/batch_000000000000",
            headers={"X-API-Key": _SMOKE_KEY},
        )
        assert r.status_code in (400, 404)


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandlingSmoke:
    def test_unknown_endpoint_404(self, client):
        r = client.get(
            "/api/v1/nonexistent",
            headers={"X-API-Key": _SMOKE_KEY},
        )
        assert r.status_code in (404, 405)

    def test_malformed_job_submit(self, client):
        """Submit without required fields should fail cleanly."""
        r = client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": _SMOKE_KEY},
        )
        assert r.status_code in (400, 422)

    def test_invalid_job_id_format(self, client):
        """Non-conforming job ID should return 400."""
        r = client.get(
            "/api/v1/jobs/invalid-id-format",
            headers={"X-API-Key": _SMOKE_KEY},
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Version Consistency
# ---------------------------------------------------------------------------


class TestVersionSmoke:
    def test_version_in_health(self, client):
        from version import __version__

        r = client.get("/api/v1/health")
        assert r.json()["version"] == __version__

    def test_version_in_openapi(self):
        """Verify version is reflected in the OpenAPI spec.

        Rebuilds the app with EXPOSE_API_DOCS enabled ( gates it off
        by default) so ``/openapi.json`` is mounted.
        """
        from version import __version__

        with patch("api.main.EXPOSE_API_DOCS", True), \
             patch("api.auth.EXPOSE_API_DOCS", True):
            from api.main import create_app
            app = create_app()
            app.state.limiter.enabled = False
            app.state.limiter.reset()
            with TestClient(app) as c:
                spec = c.get("/openapi.json").json()
        assert spec["info"]["version"] == __version__

    def test_version_not_unknown(self, client):
        r = client.get("/api/v1/health")
        assert r.json()["version"] != "unknown"
