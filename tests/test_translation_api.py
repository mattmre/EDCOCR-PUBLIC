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


@pytest.fixture(autouse=True)
def _reset_translation_limiter():
    from api.limits import limiter
    from api.routers import translation as translation_router

    # Some policy tests reload api.limits.  The route decorator keeps the
    # limiter object that existed when the router was imported, so reset both
    # references or a later test can receive a misleading 429 instead of the
    # router's explicit non-operational 501 response.
    limiters = (limiter, translation_router.limiter)
    for route_limiter in limiters:
        route_limiter.enabled = False
        route_limiter.reset()
    yield
    for route_limiter in limiters:
        route_limiter.reset()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Build an isolated FastAPI app with only the translation router."""
    from api.routers.translation import router

    app = FastAPI()
    from api.identity import AuthIdentity

    @app.middleware("http")
    async def _operator_identity(request, call_next):
        request.state.identity = AuthIdentity(
            subject="translation-test",
            role="operator",
            auth_method="test",
        )
        return await call_next(request)

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
    assert resp.status_code == 501


def test_submit_does_not_return_synthetic_job_id(client):
    resp = client.post("/api/v1/translation/jobs", json=_valid_payload())
    assert resp.status_code == 501
    body = resp.json()
    assert "job_id" not in body
    assert "no job was created" in body["detail"]


def test_submit_certified_false_in_response(client):
    resp = client.post("/api/v1/translation/jobs", json=_valid_payload())
    assert resp.status_code == 501
    assert "no job was created" in resp.json()["detail"]


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
    assert resp.status_code == 501


def test_submit_with_source_uri_ok(client):
    payload = {
        "source_uri": "s3://bucket/key.pdf",
        "target_languages": ["es"],
    }
    resp = client.post("/api/v1/translation/jobs", json=payload)
    assert resp.status_code == 501


def test_submit_does_not_claim_target_languages_without_dispatch(client):
    resp = client.post(
        "/api/v1/translation/jobs",
        json=_valid_payload(target_languages=["fr", "es"]),
    )
    assert resp.status_code == 501
    assert "no job was created" in resp.json()["detail"]


def test_submit_status_is_not_queued(client):
    resp = client.post("/api/v1/translation/jobs", json=_valid_payload())
    assert resp.status_code == 501
    assert "queued" not in resp.json()["detail"].lower()


def test_submit_certify_after_review_default_false():
    """certify_after_review must default to False (Pydantic default)."""
    from api.routers.translation import TranslationJobSubmit

    req = TranslationJobSubmit(source_text="hello", target_languages=["fr"])
    assert req.certify_after_review is False


def test_translation_router_prefix():
    """Endpoint is mounted at /api/v1/translation/jobs."""
    from api.routers.translation import router

    paths = {route.path for route in router.routes}
    assert "/api/v1/translation/jobs" in paths


def test_submit_large_source_text_handled(client):
    """10k chars reaches the explicit non-operational boundary."""
    payload = _valid_payload(source_text="a" * 10000)
    resp = client.post("/api/v1/translation/jobs", json=payload)
    assert resp.status_code == 501


def test_submit_multi_target_echo(client):
    payload = _valid_payload(target_languages=["fr", "es", "de"])
    resp = client.post("/api/v1/translation/jobs", json=payload)
    assert resp.status_code == 501
    assert "no job was created" in resp.json()["detail"]


def test_language_tags_are_canonicalized_and_duplicate_tags_rejected():
    from pydantic import ValidationError

    from api.routers.translation import TranslationJobSubmit

    request = TranslationJobSubmit(
        source_text="hello",
        source_language="EN-latn-us",
        target_languages=["FR-ca"],
    )
    assert request.source_language == "en-Latn-US"
    assert request.target_languages == ["fr-CA"]

    with pytest.raises(ValidationError, match="unique"):
        TranslationJobSubmit(
            source_text="hello",
            target_languages=["fr", "FR"],
        )
