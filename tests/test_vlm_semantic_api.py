"""Tests for the semantic search API endpoint."""

import os
from unittest import mock

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.auth import api_key_middleware
from api.database import get_engine, reset_engine
from api.event_store import reset_event_store
from vlm_config import VLMConfig
from vlm_gateway import (
    VLMAuthError,
    VLMConnectionError,
    VLMGatewayError,
    VLMTimeoutError,
)


async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Return 429 with JSON error when rate limit is exceeded."""
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please try again later.",
        },
    )


def _create_semantic_test_app() -> FastAPI:
    """Build a minimal app containing only the semantic router under test.

    Using the full application factory here is unnecessarily broad: it pulls in
    unrelated routers and transitive OCR dependencies, and reloading those
    modules can invalidate active patches on ``api.routers.semantic``.
    """
    from api import limits as limits_mod
    from api.routers import semantic as semantic_mod

    app = FastAPI()
    app.middleware("http")(api_key_middleware)
    app.state.limiter = limits_mod.limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(semantic_mod.router)
    return app


@pytest.fixture(autouse=True)
def _allow_unauthenticated():
    """Allow unauthenticated access for API tests."""
    with mock.patch.dict(os.environ, {"ALLOW_UNAUTHENTICATED": "true"}, clear=False):
        yield


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Give each test a fresh writable runtime surface."""
    reset_engine()
    reset_event_store()
    output = tmp_path / "output"
    output.mkdir()
    db_file = str(output / "semantic-tests.db")
    event_store_file = str(output / "semantic-event-store.db")
    event_stream_file = str(output / "logs" / "semantic-events.jsonl")
    audit_log_file = str(output / "logs" / "semantic-audit.jsonl")
    webhook_dlq_file = str(output / "logs" / "semantic-webhook-dlq.jsonl")
    with mock.patch("api.config.OUTPUT_FOLDER", str(output)), \
         mock.patch("api.config.DB_PATH", db_file), \
         mock.patch("api.database.DB_PATH", db_file), \
         mock.patch("api.config.EVENT_STORE_PATH", event_store_file), \
         mock.patch("api.config.API_EVENT_STREAM_PATH", event_stream_file), \
         mock.patch("api.config.API_AUDIT_LOG_PATH", audit_log_file), \
         mock.patch("api.config.WEBHOOK_DLQ_PATH", webhook_dlq_file), \
         mock.patch("api.config.EVENT_STORE_ENABLED", False), \
         mock.patch("api.config.API_EVENT_STREAM_ENABLED", False), \
         mock.patch("api.config.API_AUDIT_LOG_ENABLED", False):
        reset_engine()
        reset_event_store()
        get_engine(db_file)
        yield
        reset_event_store()
        reset_engine()


@pytest.fixture()
def _vlm_enabled_config():
    """Patch load_vlm_config to return an enabled config."""
    cfg = VLMConfig(
        enabled=True,
        endpoint_url="http://vlm-test:8080",
        api_key="test-key",
        model_name="test-model",
        max_context_pages=5,
        timeout_seconds=30,
        retry_attempts=0,
    )
    with mock.patch("api.routers.semantic.load_vlm_config", return_value=cfg):
        yield cfg


@pytest.fixture()
def _vlm_disabled_config():
    """Patch load_vlm_config to return a disabled config."""
    cfg = VLMConfig(enabled=False)
    with mock.patch("api.routers.semantic.load_vlm_config", return_value=cfg):
        yield cfg


@pytest.fixture()
def client():
    """Create an isolated test client for the semantic endpoints."""
    app = _create_semantic_test_app()
    app.state.limiter.enabled = False
    app.state.limiter.reset()
    with TestClient(app) as test_client:
        yield test_client
    app.state.limiter.reset()


class TestSemanticSearchEndpoint:
    """Tests for POST /api/v1/search/semantic."""

    def test_vlm_disabled_returns_503(self, client, _vlm_disabled_config):
        resp = client.post(
            "/api/v1/search/semantic",
            json={"query": "test"},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert data["detail"]["error"] == "vlm_disabled"

    def test_successful_search(self, client, _vlm_enabled_config):
        mock_results = [
            {
                "text": "Found text passage.",
                "score": 0.88,
                "page": 2,
                "document_id": "doc_001",
                "bbox": [100, 200, 500, 240],
            },
        ]
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.semantic_search.return_value = mock_results

            resp = client.post(
                "/api/v1/search/semantic",
                json={"query": "search text"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "search text"
        assert len(data["results"]) == 1
        assert data["results"][0]["score"] == 0.88
        assert data["results"][0]["page"] == 2
        assert data["total"] == 1
        assert data["model"] == "test-model"

    def test_search_with_document_scope(self, client, _vlm_enabled_config):
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.semantic_search.return_value = []

            resp = client.post(
                "/api/v1/search/semantic",
                json={"query": "test", "document_id": "doc_specific", "max_results": 5},
            )

        assert resp.status_code == 200
        instance.semantic_search.assert_called_once_with(
            query="test",
            document_id="doc_specific",
            max_results=5,
            min_score=0.0,
        )

    def test_search_with_min_score(self, client, _vlm_enabled_config):
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.semantic_search.return_value = []

            resp = client.post(
                "/api/v1/search/semantic",
                json={"query": "test", "min_score": 0.7},
            )

        assert resp.status_code == 200
        instance.semantic_search.assert_called_once_with(
            query="test",
            document_id=None,
            max_results=10,
            min_score=0.7,
        )

    def test_empty_query_rejected(self, client, _vlm_enabled_config):
        resp = client.post(
            "/api/v1/search/semantic",
            json={"query": ""},
        )
        assert resp.status_code == 422

    def test_connection_error_returns_502(self, client, _vlm_enabled_config):
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.semantic_search.side_effect = VLMConnectionError("unreachable")

            resp = client.post(
                "/api/v1/search/semantic",
                json={"query": "test"},
            )

        assert resp.status_code == 502
        assert resp.json()["detail"]["error"] == "vlm_connection_error"

    def test_auth_error_returns_502(self, client, _vlm_enabled_config):
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.semantic_search.side_effect = VLMAuthError("unauthorized")

            resp = client.post(
                "/api/v1/search/semantic",
                json={"query": "test"},
            )

        assert resp.status_code == 502
        assert resp.json()["detail"]["error"] == "vlm_auth_error"

    def test_timeout_returns_504(self, client, _vlm_enabled_config):
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.semantic_search.side_effect = VLMTimeoutError("timeout")

            resp = client.post(
                "/api/v1/search/semantic",
                json={"query": "test"},
            )

        assert resp.status_code == 504
        assert resp.json()["detail"]["error"] == "vlm_timeout"

    def test_generic_gateway_error_returns_502(self, client, _vlm_enabled_config):
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.semantic_search.side_effect = VLMGatewayError("something broke")

            resp = client.post(
                "/api/v1/search/semantic",
                json={"query": "test"},
            )

        assert resp.status_code == 502

    def test_max_results_validation(self, client, _vlm_enabled_config):
        resp = client.post(
            "/api/v1/search/semantic",
            json={"query": "test", "max_results": 0},
        )
        assert resp.status_code == 422

    def test_max_results_over_limit(self, client, _vlm_enabled_config):
        resp = client.post(
            "/api/v1/search/semantic",
            json={"query": "test", "max_results": 200},
        )
        assert resp.status_code == 422

    def test_min_score_over_1(self, client, _vlm_enabled_config):
        resp = client.post(
            "/api/v1/search/semantic",
            json={"query": "test", "min_score": 1.5},
        )
        assert resp.status_code == 422


class TestDocumentAnalysisEndpoint:
    """Tests for POST /api/v1/search/analyze."""

    def test_vlm_disabled_returns_503(self, client, _vlm_disabled_config):
        resp = client.post(
            "/api/v1/search/analyze",
            json={"pages": [{"page_number": 1, "text": "hello"}]},
        )
        assert resp.status_code == 503

    def test_successful_analysis(self, client, _vlm_enabled_config):
        mock_result = {
            "entities": [{"type": "PERSON", "text": "Jane Doe"}],
            "summary": "Test summary.",
            "relationships": [{"subject": "Jane Doe", "predicate": "signed", "object": "contract"}],
            "confidence": 0.91,
            "model": "test-model",
            "processing_time_ms": 150.5,
        }
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.analyze_document.return_value = mock_result

            resp = client.post(
                "/api/v1/search/analyze",
                json={
                    "pages": [{"page_number": 1, "text": "Jane Doe signed the contract."}],
                    "prompt": "Extract entities and relationships.",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["entities"]) == 1
        assert data["entities"][0]["type"] == "PERSON"
        assert data["summary"] == "Test summary."
        assert data["confidence"] == 0.91
        assert data["processing_time_ms"] == 150.5

    def test_empty_pages_rejected(self, client, _vlm_enabled_config):
        resp = client.post(
            "/api/v1/search/analyze",
            json={"pages": []},
        )
        assert resp.status_code == 422

    def test_analysis_connection_error(self, client, _vlm_enabled_config):
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.analyze_document.side_effect = VLMConnectionError("refused")

            resp = client.post(
                "/api/v1/search/analyze",
                json={"pages": [{"page_number": 1, "text": "test"}]},
            )

        assert resp.status_code == 502

    def test_analysis_with_optional_fields(self, client, _vlm_enabled_config):
        mock_result = {"entities": [], "summary": "", "relationships": [], "confidence": 0.0}
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.analyze_document.return_value = mock_result

            resp = client.post(
                "/api/v1/search/analyze",
                json={
                    "pages": [{"page_number": 1, "text": "hello"}],
                    "prompt": "Analyze this.",
                    "document_id": "doc_abc",
                },
            )

        assert resp.status_code == 200
        instance.analyze_document.assert_called_once_with(
            pages=[{"page_number": 1, "text": "hello"}],
            prompt="Analyze this.",
            document_id="doc_abc",
        )


class TestVLMHealthEndpoint:
    """Tests for GET /api/v1/search/vlm/health."""

    def test_disabled_returns_status(self, client, _vlm_disabled_config):
        resp = client.get("/api/v1/search/vlm/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["vlm_enabled"] is False
        assert data["vlm_reachable"] is False

    def test_enabled_and_reachable(self, client, _vlm_enabled_config):
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.health_check.return_value = True

            resp = client.get("/api/v1/search/vlm/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["vlm_enabled"] is True
        assert data["vlm_reachable"] is True
        assert data["model_name"] == "test-model"
        assert data["endpoint_configured"] is True

    def test_enabled_but_unreachable(self, client, _vlm_enabled_config):
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.health_check.return_value = False

            resp = client.get("/api/v1/search/vlm/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["vlm_enabled"] is True
        assert data["vlm_reachable"] is False

    def test_health_check_exception_handled(self, client, _vlm_enabled_config):
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.health_check.side_effect = Exception("unexpected")

            resp = client.get("/api/v1/search/vlm/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["vlm_reachable"] is False


class TestVLMConfigMisconfigured:
    """Tests for invalid VLM configuration."""

    def test_enabled_without_url_returns_503(self, client):
        cfg = VLMConfig(enabled=True, endpoint_url="")
        with mock.patch("api.routers.semantic.load_vlm_config", return_value=cfg):
            resp = client.post(
                "/api/v1/search/semantic",
                json={"query": "test"},
            )
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "vlm_misconfigured"

    def test_enabled_with_bad_url_scheme_returns_503(self, client):
        cfg = VLMConfig(enabled=True, endpoint_url="ftp://invalid")
        with mock.patch("api.routers.semantic.load_vlm_config", return_value=cfg):
            resp = client.post(
                "/api/v1/search/semantic",
                json={"query": "test"},
            )
        assert resp.status_code == 503


class TestGatewayCleanup:
    """Tests that gateway resources are cleaned up after requests."""

    def test_gateway_closed_after_successful_search(self, client, _vlm_enabled_config):
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.semantic_search.return_value = []

            client.post("/api/v1/search/semantic", json={"query": "test"})

            instance.close.assert_called_once()

    def test_gateway_closed_after_error(self, client, _vlm_enabled_config):
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.semantic_search.side_effect = VLMConnectionError("err")

            client.post("/api/v1/search/semantic", json={"query": "test"})

            instance.close.assert_called_once()

    def test_gateway_closed_after_analysis(self, client, _vlm_enabled_config):
        mock_result = {"entities": [], "summary": "", "relationships": [], "confidence": 0.0}
        with mock.patch("api.routers.semantic.VLMGateway") as MockGW:
            instance = MockGW.return_value
            instance.analyze_document.return_value = mock_result

            client.post(
                "/api/v1/search/analyze",
                json={"pages": [{"page_number": 1, "text": "test"}]},
            )

            instance.close.assert_called_once()


class TestVLMImportGuard:
    """VLM gateway import must be guarded so the API still starts.

    Simulates VLM dependencies being unavailable by patching the
    ``VLM_AVAILABLE`` flag on the router module.
    """

    def test_semantic_search_returns_503_when_vlm_unavailable(self, client):
        from api.routers import semantic as semantic_mod

        with mock.patch.object(semantic_mod, "VLM_AVAILABLE", False):
            resp = client.post("/api/v1/search/semantic", json={"query": "test"})
        assert resp.status_code == 503
        body = resp.json()
        assert "VLM dependencies" in body["detail"]
        assert "requirements-vlm.txt" in body["detail"]

    def test_analyze_returns_503_when_vlm_unavailable(self, client):
        from api.routers import semantic as semantic_mod

        with mock.patch.object(semantic_mod, "VLM_AVAILABLE", False):
            resp = client.post(
                "/api/v1/search/analyze",
                json={"pages": [{"page_number": 1, "text": "sample"}]},
            )
        assert resp.status_code == 503
        body = resp.json()
        assert "VLM dependencies" in body["detail"]

    def test_health_reports_unavailable_when_vlm_missing(self, client):
        from api.routers import semantic as semantic_mod

        with mock.patch.object(semantic_mod, "VLM_AVAILABLE", False):
            resp = client.get("/api/v1/search/vlm/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["vlm_enabled"] is False
        assert body["vlm_reachable"] is False
        assert body["endpoint_configured"] is False
