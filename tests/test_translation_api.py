"""Tests for the translation REST endpoint -- Plan B M1-PR5.

These tests use FastAPI's ``TestClient`` against an isolated app
instance built from ``api.routers.translation`` directly, bypassing the
auth middleware so the endpoint contract can be exercised
deterministically without bringing up the full ``api.main`` stack.

Authentication is exercised via a small auth-middleware test that
mirrors the production stack.
"""
from __future__ import annotations

from typing import Iterator

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Build an isolated FastAPI app with only the translation router."""
    from api.routers.translation import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


def _valid_payload(**overrides) -> dict:
    payload = {
        "source_text": "hello",
        "target_languages": ["fr"],
    }
    payload.update(overrides)
    return payload


def test_submit_requires_api_key(monkeypatch):
    """When mounted via api.main, missing X-API-Key -> 401.

    Built directly here to exercise the auth middleware behaviour.
    """
    from fastapi import FastAPI as _FastAPI

    monkeypatch.setenv("OCR_API_KEY", "test-key-abc")
    monkeypatch.setenv("ENABLE_TRANSLATION_API", "true")

    # Reload api.config + api.auth so they pick up the env vars.
    import importlib

    import api.auth as _auth
    import api.config as _config

    importlib.reload(_config)
    importlib.reload(_auth)

    from api.routers.translation import router

    app = _FastAPI()
    app.middleware("http")(_auth.api_key_middleware)
    app.include_router(router)
    c = TestClient(app)
    resp = c.post("/api/v1/translation/jobs", json=_valid_payload())
    assert resp.status_code == 401


def test_submit_no_source_raises_422(client):
    payload = {"target_languages": ["fr"]}
    resp = client.post("/api/v1/translation/jobs", json=payload)
    assert resp.status_code == 422


def test_submit_with_source_text_ok(client):
    resp = client.post("/api/v1/translation/jobs", json=_valid_payload())
    assert resp.status_code == 202


def test_submit_returns_job_id(client):
    resp = client.post("/api/v1/translation/jobs", json=_valid_payload())
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    # uuid4 string length is 36
    assert len(body["job_id"]) == 36


def test_submit_certified_false_in_response(client):
    resp = client.post("/api/v1/translation/jobs", json=_valid_payload())
    assert resp.status_code == 202
    assert resp.json()["certified"] is False


def test_submit_empty_target_languages_422(client):
    payload = _valid_payload(target_languages=[])
    resp = client.post("/api/v1/translation/jobs", json=payload)
    assert resp.status_code == 422


def test_submit_with_source_job_id_ok(client):
    payload = {
        "source_job_id": "job_abc123",
        "target_languages": ["fr"],
    }
    resp = client.post("/api/v1/translation/jobs", json=payload)
    assert resp.status_code == 202


def test_submit_with_source_uri_ok(client):
    payload = {
        "source_uri": "s3://bucket/key.pdf",
        "target_languages": ["es"],
    }
    resp = client.post("/api/v1/translation/jobs", json=payload)
    assert resp.status_code == 202


def test_response_has_target_languages(client):
    resp = client.post(
        "/api/v1/translation/jobs",
        json=_valid_payload(target_languages=["fr", "es"]),
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["target_languages"] == ["fr", "es"]


def test_submit_status_is_queued(client):
    resp = client.post("/api/v1/translation/jobs", json=_valid_payload())
    assert resp.json()["status"] == "queued"


def test_submit_certify_after_review_default_false():
    """certify_after_review must default to False (Pydantic default)."""
    from api.routers.translation import TranslationJobSubmit

    req = TranslationJobSubmit(
        source_text="hello", target_languages=["fr"]
    )
    assert req.certify_after_review is False


def test_translation_router_prefix():
    """Endpoint is mounted at /api/v1/translation/jobs."""
    from api.routers.translation import router

    paths = {route.path for route in router.routes}
    assert "/api/v1/translation/jobs" in paths


def test_submit_large_source_text_handled(client):
    """10k chars accepted (no 413 in the stub endpoint)."""
    payload = _valid_payload(source_text="a" * 10000)
    resp = client.post("/api/v1/translation/jobs", json=payload)
    assert resp.status_code == 202


def test_submit_multi_target_echo(client):
    payload = _valid_payload(target_languages=["fr", "es", "de"])
    resp = client.post("/api/v1/translation/jobs", json=payload)
    assert resp.status_code == 202
    body = resp.json()
    assert set(body["target_languages"]) == {"fr", "es", "de"}
