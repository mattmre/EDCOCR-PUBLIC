"""Tests for consolidated health check endpoint."""

import os
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.database import get_engine, reset_engine


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    db_file = str(tmp_path / "test_health.db")

    operator_placeholder = "placeholder"
    with patch.dict(os.environ, {
        "OCR_OUTPUT_DIR": str(output),
        "OCR_SOURCE_DIR": str(source),
        "OCR_API_KEY": operator_placeholder,
        "API_DB_PATH": db_file,
    }), \
         patch("api.config.DB_PATH", db_file), \
         patch("api.database.DB_PATH", db_file), \
         patch("api.config.SOURCE_FOLDER", str(source)), \
         patch("api.config.OUTPUT_FOLDER", str(output)), \
         patch("api.auth.OCR_API_KEY", "test-key"):
        reset_engine()
        get_engine(db_file)
        yield
        reset_engine()


@pytest.fixture
def client():
    from api.main import create_app

    app = create_app()
    app.state.limiter.enabled = False
    app.state.limiter.reset()
    with TestClient(app) as test_client:
        yield test_client


class TestBasicHealthUnchanged:
    """Verify the original /api/v1/health endpoint still works unchanged."""

    def test_health_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_no_auth_required(self, client):
        r = client.get("/api/v1/health", headers={"X-API-Key": ""})
        assert r.status_code == 200

    @pytest.mark.parametrize("path", ["/api/v1/ready", "/api/v1/readiness"])
    def test_readiness_aliases_return_basic_health(self, client, path):
        r = client.get(path)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data


class TestDetailedHealth:
    def test_detailed_returns_200(self, client):
        r = client.get("/api/v1/health/detailed")
        assert r.status_code == 200
        data = r.json()
        assert "checks" in data
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data

    def test_detailed_no_auth_required(self, client):
        r = client.get("/api/v1/health/detailed")
        assert r.status_code == 200

    def test_detailed_includes_database_check(self, client):
        r = client.get("/api/v1/health/detailed")
        checks = r.json()["checks"]
        assert "database" in checks
        assert checks["database"]["status"] in ("healthy", "degraded", "unhealthy")

    def test_detailed_includes_disk_checks(self, client):
        r = client.get("/api/v1/health/detailed")
        checks = r.json()["checks"]
        assert "disk_output" in checks
        assert "disk_source" in checks

    def test_detailed_includes_models_check(self, client):
        r = client.get("/api/v1/health/detailed")
        checks = r.json()["checks"]
        assert "models" in checks

    def test_detailed_includes_pipeline_check(self, client):
        r = client.get("/api/v1/health/detailed")
        checks = r.json()["checks"]
        assert "pipeline" in checks

    def test_detailed_includes_external_translation_check(self, client):
        r = client.get("/api/v1/health/detailed")
        checks = r.json()["checks"]
        assert "external_translation" in checks
        assert checks["external_translation"]["status"] in (
            "healthy",
            "degraded",
            "unhealthy",
        )

    def test_external_translation_readiness_endpoint(self, client):
        from api.models import SubsystemCheck

        with patch("api.routers.health._check_external_translation") as mock_ext:
            mock_ext.return_value = SubsystemCheck(
                status="degraded",
                message="EDC_TRANSLATION unreachable",
                latency_ms=12.3,
            )
            r = client.get("/api/v1/translation/readiness")

        assert r.status_code == 200
        assert r.json() == {
            "status": "degraded",
            "message": "EDC_TRANSLATION unreachable",
            "latency_ms": 12.3,
        }

    def test_detailed_status_healthy_when_all_pass(self, client):
        from api.models import SubsystemCheck
        with patch("api.routers.health._check_database") as mock_db, \
             patch("api.routers.health._check_disk") as mock_disk, \
             patch("api.routers.health._check_models") as mock_models, \
             patch("api.routers.health._check_heartbeat") as mock_hb, \
             patch("api.routers.health._check_external_translation") as mock_ext:
            mock_db.return_value = SubsystemCheck(status="healthy", message="OK")
            mock_disk.return_value = SubsystemCheck(status="healthy", message="OK")
            mock_models.return_value = SubsystemCheck(status="healthy", message="OK")
            mock_hb.return_value = SubsystemCheck(status="healthy", message="OK")
            mock_ext.return_value = SubsystemCheck(status="healthy", message="disabled")
            r = client.get("/api/v1/health/detailed")
        data = r.json()
        assert data["status"] == "healthy"

    def test_detailed_backward_compatible(self, client):
        """Detailed response extends basic response -- has all basic fields."""
        r = client.get("/api/v1/health/detailed")
        data = r.json()
        # All fields from basic health must be present
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "jobs" in data


class TestSubsystemChecks:
    def test_check_database_healthy(self):
        from api.routers.health import _check_database
        result = _check_database()
        assert result.status == "healthy"
        assert result.latency_ms is not None
        assert result.latency_ms >= 0

    def test_check_disk_healthy(self, tmp_path):
        from api.routers.health import _check_disk
        result = _check_disk(str(tmp_path), "Test")
        assert result.status == "healthy"
        assert "GB free" in result.message

    def test_check_disk_missing_dir(self, tmp_path):
        from api.routers.health import _check_disk
        missing = str(tmp_path / "does_not_exist_subdir")
        result = _check_disk(missing, "Test")
        assert result.status == "unhealthy"
        assert "not found" in result.message

    def test_check_heartbeat_missing(self):
        from api.routers.health import _check_heartbeat
        with patch.dict(os.environ, {"HEALTHCHECK_FILE": "/nonexistent/heartbeat"}):
            result = _check_heartbeat()
            assert result.status == "degraded"

    def test_check_heartbeat_fresh(self, tmp_path):
        from api.routers.health import _check_heartbeat
        hb = tmp_path / "heartbeat"
        hb.write_text("ok")
        with patch.dict(os.environ, {"HEALTHCHECK_FILE": str(hb)}):
            result = _check_heartbeat()
            assert result.status == "healthy"

    def test_check_heartbeat_stale(self, tmp_path):
        from api.routers.health import _check_heartbeat
        hb = tmp_path / "heartbeat"
        hb.write_text("ok")
        # Make file appear old
        old_time = time.time() - 130
        os.utime(str(hb), (old_time, old_time))
        with patch.dict(os.environ, {"HEALTHCHECK_FILE": str(hb)}):
            result = _check_heartbeat()
            assert result.status == "unhealthy"

    def test_check_heartbeat_aging(self, tmp_path):
        from api.routers.health import _check_heartbeat
        hb = tmp_path / "heartbeat"
        hb.write_text("ok")
        # Make file appear 70 seconds old (between 60 and 120 threshold)
        old_time = time.time() - 70
        os.utime(str(hb), (old_time, old_time))
        with patch.dict(os.environ, {"HEALTHCHECK_FILE": str(hb)}):
            result = _check_heartbeat()
            assert result.status == "degraded"
            assert "aging" in result.message.lower()

    def test_check_models_degraded_when_missing(self):
        from api.routers.health import _check_models
        # In test environment, FastText model typically does not exist
        result = _check_models()
        # Either healthy (if the model happens to exist) or degraded
        assert result.status in ("healthy", "degraded")

    def test_overall_unhealthy_propagates(self, client):
        """If any subsystem is unhealthy, overall should be unhealthy."""
        from api.models import SubsystemCheck
        with patch("api.routers.health._check_database") as mock_db:
            mock_db.return_value = SubsystemCheck(status="unhealthy", message="DB down")
            r = client.get("/api/v1/health/detailed")
            assert r.json()["status"] == "unhealthy"

    def test_overall_degraded_propagates(self, client):
        """If worst subsystem is degraded (none unhealthy), overall should be degraded."""
        from api.models import SubsystemCheck
        with patch("api.routers.health._check_database") as mock_db, \
             patch("api.routers.health._check_disk") as mock_disk, \
             patch("api.routers.health._check_models") as mock_models, \
             patch("api.routers.health._check_heartbeat") as mock_hb, \
             patch("api.routers.health._check_external_translation") as mock_ext:
            mock_db.return_value = SubsystemCheck(status="healthy", message="OK")
            mock_disk.return_value = SubsystemCheck(status="healthy", message="OK")
            mock_models.return_value = SubsystemCheck(status="degraded", message="no model")
            mock_hb.return_value = SubsystemCheck(status="healthy", message="OK")
            mock_ext.return_value = SubsystemCheck(status="healthy", message="disabled")
            r = client.get("/api/v1/health/detailed")
            assert r.json()["status"] == "degraded"

    def test_check_disk_low_space(self, tmp_path):
        """Disk check returns degraded when free space is below threshold."""
        from api.routers.health import _check_disk
        # Use an impossibly high threshold to force degraded
        result = _check_disk(str(tmp_path), "Test", min_free_gb=999999.0)
        assert result.status == "degraded"
        assert "threshold" in result.message

    def test_check_external_translation_disabled_is_healthy(self):
        from api.routers.health import _check_external_translation

        with patch.dict(os.environ, {"EDC_TRANSLATION_PREFER_EXTERNAL": "false"}):
            result = _check_external_translation()

        assert result.status == "healthy"
        assert "disabled" in result.message
