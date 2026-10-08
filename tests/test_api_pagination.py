"""Tests for the standardized pagination dependency (maintenance review).

Verifies that all list endpoints accept limit/offset (canonical) and
page/per_page (backward-compatible aliases) and that max limit is enforced.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from api.deps import MAX_PAGE_LIMIT, normalize_pagination

# ---------------------------------------------------------------------------
# Unit tests for normalize_pagination helper
# ---------------------------------------------------------------------------


class TestNormalizePagination:
    """Direct unit tests for the pure pagination normalizer."""

    def test_defaults(self):
        limit, offset = normalize_pagination(limit=50, offset=0)
        assert limit == 50
        assert offset == 0

    def test_explicit_limit_offset(self):
        limit, offset = normalize_pagination(limit=10, offset=20)
        assert limit == 10
        assert offset == 20

    def test_legacy_page_per_page(self):
        """page=2 + per_page=10 -> offset=10, limit=10."""
        limit, offset = normalize_pagination(limit=50, offset=0, page=2, per_page=10)
        assert limit == 10
        assert offset == 10

    def test_legacy_page_only(self):
        """page=3 without per_page -> offset=(3-1)*50=100, limit=50."""
        limit, offset = normalize_pagination(limit=50, offset=0, page=3)
        assert limit == 50
        assert offset == 100

    def test_legacy_per_page_only(self):
        """per_page=25 without page -> limit=25, offset=0."""
        limit, offset = normalize_pagination(limit=50, offset=0, per_page=25)
        assert limit == 25
        assert offset == 0

    def test_page_1_per_page_maps_to_zero_offset(self):
        limit, offset = normalize_pagination(limit=50, offset=0, page=1, per_page=5)
        assert limit == 5
        assert offset == 0

    def test_max_limit_constant(self):
        assert MAX_PAGE_LIMIT == 200

    def test_direct_inputs_are_bounded(self):
        with pytest.raises(ValueError, match="limit"):
            normalize_pagination(limit=0, offset=0)
        with pytest.raises(ValueError, match="page"):
            normalize_pagination(limit=50, offset=0, page=0)
        with pytest.raises(ValueError, match="per_page"):
            normalize_pagination(limit=50, offset=0, per_page=201)

    def test_dependency_maps_legacy_validation_to_422(self):
        from fastapi import HTTPException

        from api.deps import pagination_params

        with pytest.raises(HTTPException) as exc_info:
            pagination_params(50, 0, page=0, per_page=None)
        assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# Integration tests: list_jobs endpoint pagination
# ---------------------------------------------------------------------------


class TestJobsListPagination:
    """Verify /api/v1/jobs list endpoint with both pagination idioms."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OCR_API_KEY", "test-key-pagination")
        monkeypatch.setenv("OCR_OUTPUT_DIR", str(tmp_path / "output"))
        monkeypatch.setenv("OCR_SOURCE_DIR", str(tmp_path / "source"))
        monkeypatch.setenv("ALLOW_UNAUTHENTICATED", "true")
        (tmp_path / "output").mkdir(exist_ok=True)
        (tmp_path / "source").mkdir(exist_ok=True)

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient

        from api.main import create_app

        # Auth configuration is imported as module constants.  Patch both
        # sides explicitly so this fixture remains valid after tests that
        # exercise an authenticated app; environment changes alone do not
        # undo a previously imported API key.
        with (
            patch("api.config.OCR_API_KEY", ""),
            patch("api.auth.OCR_API_KEY", ""),
            patch("api.config.ALLOW_UNAUTHENTICATED", True),
            patch("api.auth.ALLOW_UNAUTHENTICATED", True),
            patch("api.config.ENABLE_MULTITENANCY", False),
            patch("api.auth.ENABLE_MULTITENANCY", False),
        ):
            app = create_app()
            app.state.limiter.enabled = False
            app.state.limiter.reset()
            c = TestClient(app)
            yield c

    def test_default_pagination(self, client):
        resp = client.get("/api/v1/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_explicit_limit_offset(self, client):
        resp = client.get("/api/v1/jobs?limit=10&offset=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["limit"] == 10
        assert data["offset"] == 5

    def test_legacy_page_per_page(self, client):
        resp = client.get("/api/v1/jobs?page=2&per_page=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["limit"] == 10
        assert data["offset"] == 10

    def test_limit_exceeds_max_returns_422(self, client):
        resp = client.get("/api/v1/jobs?limit=201")
        assert resp.status_code == 422

    def test_negative_offset_returns_422(self, client):
        resp = client.get("/api/v1/jobs?offset=-1")
        assert resp.status_code == 422

    def test_per_page_exceeds_max_returns_422(self, client):
        resp = client.get("/api/v1/jobs?per_page=201")
        assert resp.status_code == 422
