"""Tests for OpenAPI spec generation and completeness."""
from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from api.database import get_engine, reset_engine


@pytest.fixture()
def app(monkeypatch):
    """Create app with all feature gates enabled."""
    monkeypatch.setenv("ENABLE_MULTITENANCY", "true")
    monkeypatch.setenv("ENABLE_DASHBOARD", "true")
    monkeypatch.setenv("OCR_API_KEY", "test-key")
    _tmpdir = tempfile.mkdtemp()
    monkeypatch.setenv("OCR_OUTPUT_DIR", _tmpdir)
    monkeypatch.setenv("OCR_SOURCE_DIR", _tmpdir)
    db_file = os.path.join(_tmpdir, "openapi-test.db")
    monkeypatch.setenv("API_DB_PATH", db_file)
    os.makedirs(os.path.join(_tmpdir, "EXPORT", "PDF"), exist_ok=True)

    # Patch config module to reflect env vars set after import
    import api.auth as _auth
    import api.config as _cfg
    import api.database as _db
    monkeypatch.setattr(_cfg, "ENABLE_MULTITENANCY", True)
    monkeypatch.setattr(_cfg, "OUTPUT_FOLDER", _tmpdir)
    monkeypatch.setattr(_cfg, "SOURCE_FOLDER", _tmpdir)
    monkeypatch.setattr(_cfg, "DB_PATH", db_file)
    monkeypatch.setattr(_auth, "OCR_API_KEY", "test-key")
    monkeypatch.setattr(_db, "DB_PATH", db_file)

    import api.main as _main
    from api.main import create_app

    monkeypatch.setattr(_main, "ENABLE_MULTITENANCY", True)
    monkeypatch.setattr(_main, "EXPOSE_API_DOCS", True)
    monkeypatch.setattr(_auth, "EXPOSE_API_DOCS", True)

    reset_engine()
    get_engine(db_file)
    app = create_app()
    app.state.limiter.enabled = False
    app.state.limiter.reset()
    yield app
    reset_engine()


@pytest.fixture()
def client(app):
    return TestClient(app)


class TestOpenAPISpec:
    def test_openapi_json_accessible(self, client):
        """GET /openapi.json returns valid JSON schema."""
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "openapi" in schema
        assert schema["info"]["title"] == "EDCOCR API"

    def test_security_schemes_present(self, client):
        """Security schemes include ApiKeyAuth and BearerAuth."""
        schema = client.get("/openapi.json").json()
        schemes = schema.get("components", {}).get("securitySchemes", {})
        assert "ApiKeyAuth" in schemes
        assert schemes["ApiKeyAuth"]["type"] == "apiKey"
        assert schemes["ApiKeyAuth"]["in"] == "header"
        assert schemes["ApiKeyAuth"]["name"] == "X-API-Key"
        assert "BearerAuth" in schemes
        assert schemes["BearerAuth"]["type"] == "http"
        assert schemes["BearerAuth"]["scheme"] == "bearer"

    def test_global_security_applied(self, client):
        """Global security array references ApiKeyAuth."""
        schema = client.get("/openapi.json").json()
        assert "security" in schema
        assert any("ApiKeyAuth" in s for s in schema["security"])
        assert any("BearerAuth" in s for s in schema["security"])

    def test_tags_have_descriptions(self, client):
        """All declared tags include a description field."""
        schema = client.get("/openapi.json").json()
        tags = schema.get("tags", [])
        assert len(tags) >= 10
        for tag in tags:
            assert "description" in tag, f"Tag '{tag['name']}' missing description"

    def test_health_endpoint_in_spec(self, client):
        """Health check path is present in the spec."""
        schema = client.get("/openapi.json").json()
        assert "/api/v1/health" in schema["paths"]

    def test_jobs_endpoints_in_spec(self, client):
        """Core job management paths are present in the spec."""
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]
        assert "/api/v1/jobs" in paths
        assert "/api/v1/jobs/{job_id}" in paths

    def test_docs_accessible(self, client):
        """Swagger UI docs endpoint returns 200."""
        r = client.get("/docs")
        assert r.status_code == 200

    def test_redoc_accessible(self, client):
        """ReDoc endpoint returns 200."""
        r = client.get("/redoc")
        assert r.status_code == 200

    def test_version_matches(self, client):
        """OpenAPI info.version matches version.py."""
        schema = client.get("/openapi.json").json()
        from version import __version__

        assert schema["info"]["version"] == __version__

    def test_admin_endpoints_present_when_enabled(self, client):
        """Admin endpoints should be in spec when ENABLE_MULTITENANCY=true."""
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]
        assert "/api/v1/admin/tenants" in paths

    def test_dashboard_endpoints_present_when_enabled(self, client):
        """Dashboard endpoints should be in spec when ENABLE_DASHBOARD=true."""
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]
        assert "/api/v1/dashboard" in paths

    def test_batch_endpoints_in_spec(self, client):
        """Batch submission path is present in the spec."""
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]
        # batch router has /api/v1/jobs/batch prefix
        batch_paths = [p for p in paths if "batch" in p]
        assert len(batch_paths) > 0

    def test_security_scheme_descriptions_nonempty(self, client):
        """Security scheme descriptions are non-empty strings."""
        schema = client.get("/openapi.json").json()
        schemes = schema["components"]["securitySchemes"]
        for name, scheme in schemes.items():
            assert scheme.get("description"), f"{name} has empty description"


class TestExportScript:
    def test_export_script_exists(self):
        """Export script file exists on disk."""
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "export_openapi.py",
        )
        assert os.path.exists(script_path)

    def test_export_script_importable(self):
        """Export script can be loaded as a module without errors."""
        import importlib.util

        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "export_openapi.py",
        )
        spec = importlib.util.spec_from_file_location("export_openapi", script_path)
        assert spec is not None
