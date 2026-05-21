"""Tests for recall API endpoints (entities, extractions, recall stats)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _env_setup(tmp_path, monkeypatch):
    """Set up environment for API tests."""
    monkeypatch.setenv("OCR_API_KEY", "test-key-recall")
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "")
    monkeypatch.setenv("OCR_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUT_FOLDER", str(tmp_path))
    monkeypatch.setenv("API_DB_PATH", str(tmp_path / "test_jobs.db"))

    # Patch api.config values that are read at import time
    monkeypatch.setattr("api.config.OUTPUT_FOLDER", str(tmp_path))
    monkeypatch.setattr("api.config.DB_PATH", str(tmp_path / "test_jobs.db"))
    monkeypatch.setattr("api.config.OCR_API_KEY", "test-key-recall")
    monkeypatch.setattr("api.config.ALLOW_UNAUTHENTICATED", False)
    # Also patch auth module to ensure consistent auth enforcement
    try:
        monkeypatch.setattr("api.auth.OCR_API_KEY", "test-key-recall")
        monkeypatch.setattr("api.auth.ALLOW_UNAUTHENTICATED", False)
    except AttributeError:
        pass


@pytest.fixture()
def entity_index(tmp_path):
    """Create an EntityIndex backed by a temporary database."""
    from api.entity_index import EntityIndex

    db_path = str(tmp_path / "test_entity_index.db")
    return EntityIndex(db_path=db_path)


@pytest.fixture()
def populated_entity_index(entity_index):
    """Pre-populate the entity index with sample data."""
    entity_index.index_entities("job_aaa111bbb222", "report.pdf", [
        {"type": "PERSON", "text": "Alice Johnson", "confidence": 0.95, "source": "ner", "page": 1},
        {"type": "DATE", "text": "2026-03-15", "confidence": 0.88, "source": "extraction", "page": 1},
        {"type": "ORG", "text": "TestCorp Inc", "confidence": 0.92, "source": "ner", "page": 2},
    ])
    entity_index.index_extractions("job_aaa111bbb222", "report.pdf", [
        {"key": "invoice_number", "value": "INV-001", "confidence": 0.90, "page": 1},
        {"key": "total_amount", "value": "$2,500.00", "confidence": 0.87, "page": 2},
    ])
    return entity_index


@pytest.fixture()
def client(tmp_path, populated_entity_index):
    """Create a test client with the recall router active."""
    # Reset database engine state for clean test
    from api import database

    database.reset_engine()
    database._engine = None
    database._SessionLocal = None

    # Patch the recall router singleton to use our populated index
    import api.routers.recall as recall_mod

    recall_mod._entity_index = populated_entity_index

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)

    with patch("api.config.SOURCE_FOLDER", str(source)), \
         patch("api.config.OUTPUT_FOLDER", str(output)), \
         patch("api.config.OCR_API_KEY", "test-key-recall"), \
         patch("api.config.ALLOW_UNAUTHENTICATED", False), \
         patch("api.auth.OCR_API_KEY", "test-key-recall"), \
         patch("api.auth.ALLOW_UNAUTHENTICATED", False), \
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
        with TestClient(app) as c:
            yield c

    # Clean up singleton
    recall_mod._entity_index = None
    database.reset_engine()


def _auth_header():
    return {"X-API-Key": "test-key-recall"}


# ---------------------------------------------------------------------------
# Entity search endpoint
# ---------------------------------------------------------------------------


class TestSearchEntities:
    def test_search_all(self, client):
        """GET /api/v1/entities returns all entities."""
        resp = client.get("/api/v1/entities", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["results"]) == 3
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_search_by_type(self, client):
        """Filter entities by type."""
        resp = client.get(
            "/api/v1/entities",
            params={"type": "PERSON"},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["entity_type"] == "PERSON"
        assert data["results"][0]["text"] == "Alice Johnson"

    def test_search_by_text(self, client):
        """Search entities by text query."""
        resp = client.get(
            "/api/v1/entities",
            params={"q": "Alice"},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["text"] == "Alice Johnson"

    def test_search_by_job_id(self, client):
        """Filter entities by job ID."""
        resp = client.get(
            "/api/v1/entities",
            params={"job_id": "job_aaa111bbb222"},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 3

    def test_search_by_min_confidence(self, client):
        """Filter entities by minimum confidence."""
        resp = client.get(
            "/api/v1/entities",
            params={"min_confidence": 0.9},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        for result in data["results"]:
            assert result["confidence"] >= 0.9

    def test_search_pagination(self, client):
        """Test limit and offset parameters."""
        resp = client.get(
            "/api/v1/entities",
            params={"limit": 1, "offset": 0},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["total"] == 3
        assert data["limit"] == 1
        assert data["offset"] == 0

    def test_search_no_results(self, client):
        """Search with no matches returns empty."""
        resp = client.get(
            "/api/v1/entities",
            params={"type": "NONEXISTENT"},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["results"] == []

    def test_result_shape(self, client):
        """Verify response structure has expected keys."""
        resp = client.get("/api/v1/entities", headers=_auth_header())
        data = resp.json()
        result = data["results"][0]
        assert "entity_id" in result
        assert "job_id" in result
        assert "entity_type" in result
        assert "text" in result
        assert "confidence" in result
        assert "source" in result
        assert "page" in result
        assert "document_name" in result


# ---------------------------------------------------------------------------
# Extraction search endpoint
# ---------------------------------------------------------------------------


class TestSearchExtractions:
    def test_search_all(self, client):
        """GET /api/v1/extractions returns all extractions."""
        resp = client.get("/api/v1/extractions", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_search_by_field(self, client):
        """Filter extractions by field name."""
        resp = client.get(
            "/api/v1/extractions",
            params={"field": "invoice_number"},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["field_name"] == "invoice_number"

    def test_search_by_value(self, client):
        """Search extractions by value query."""
        resp = client.get(
            "/api/v1/extractions",
            params={"q": "$2,500"},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["field_value"] == "$2,500.00"

    def test_search_by_job_id(self, client):
        """Filter extractions by job ID."""
        resp = client.get(
            "/api/v1/extractions",
            params={"job_id": "job_aaa111bbb222"},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_search_pagination(self, client):
        """Test limit and offset for extractions."""
        resp = client.get(
            "/api/v1/extractions",
            params={"limit": 1, "offset": 0},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["total"] == 2

    def test_result_shape(self, client):
        """Verify extraction response structure."""
        resp = client.get("/api/v1/extractions", headers=_auth_header())
        data = resp.json()
        result = data["results"][0]
        assert "extraction_id" in result
        assert "job_id" in result
        assert "field_name" in result
        assert "field_value" in result
        assert "confidence" in result
        assert "page" in result
        assert "document_name" in result


# ---------------------------------------------------------------------------
# Recall stats endpoint
# ---------------------------------------------------------------------------


class TestRecallStats:
    def test_stats(self, client):
        """GET /api/v1/recall/stats returns index statistics."""
        resp = client.get("/api/v1/recall/stats", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_entities"] == 3
        assert data["total_extractions"] == 2
        assert data["unique_entity_types"] == 3  # PERSON, DATE, ORG
        assert data["unique_field_names"] == 2  # invoice_number, total_amount
        assert data["jobs_indexed"] == 1


# ---------------------------------------------------------------------------
# Auth enforcement
# ---------------------------------------------------------------------------


class TestAuthEnforcement:
    def test_entities_requires_auth(self, client):
        """Entity search requires authentication."""
        resp = client.get("/api/v1/entities")
        assert resp.status_code in (401, 403)

    def test_extractions_requires_auth(self, client):
        """Extraction search requires authentication."""
        resp = client.get("/api/v1/extractions")
        assert resp.status_code in (401, 403)

    def test_stats_requires_auth(self, client):
        """Stats endpoint requires authentication."""
        resp = client.get("/api/v1/recall/stats")
        assert resp.status_code in (401, 403)

    def test_wrong_api_key(self, client):
        """Wrong API key is rejected."""
        resp = client.get(
            "/api/v1/entities",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code in (401, 403)
