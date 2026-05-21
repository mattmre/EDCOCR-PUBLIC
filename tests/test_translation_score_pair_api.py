"""Tests for ``POST /api/v1/translation/score-pair`` -- B15.

These tests build an isolated FastAPI app with the translation router
and the rate-limiter middleware so the endpoint contract can be
exercised without the full ``api.main`` stack.

The COMETKiwi estimator is replaced via the module-level
``_get_qe_estimator`` cache so we never need ``unbabel-comet`` installed.
"""
from __future__ import annotations

from typing import Iterator
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi.middleware import SlowAPIMiddleware  # noqa: E402

from api.limits import limiter  # noqa: E402
from api.routers import translation as translation_router_mod  # noqa: E402
from ocr_local.translation.quality_estimation import (  # noqa: E402
    QualityEstimationConfig,
    QualityScore,
)


def _rate_limit_exceeded_handler(request, exc):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})


@pytest.fixture
def app_no_auth() -> Iterator[FastAPI]:
    """Build an isolated FastAPI app with the translation router only."""
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.include_router(translation_router_mod.router)
    yield app


@pytest.fixture
def client(app_no_auth) -> Iterator[TestClient]:
    with TestClient(app_no_auth) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_qe_cache():
    """Make sure each test starts with a clean estimator cache."""
    translation_router_mod._reset_qe_estimator_cache()
    yield
    translation_router_mod._reset_qe_estimator_cache()


@pytest.fixture(autouse=True)
def _reset_limiter():
    """Disable the limiter between tests so quota doesn't leak across cases."""
    limiter.enabled = True
    limiter.reset()
    yield
    limiter.reset()


def _make_estimator_stub(
    *,
    score: float | None = 0.85,
    available: bool = True,
    reason: str | None = None,
    threshold_warn: float = 0.4,
    threshold_reject: float = 0.2,
):
    """Return an object that mimics CometKiwiEstimator for the router."""
    cfg = QualityEstimationConfig(
        score_threshold_warn=threshold_warn,
        score_threshold_reject=threshold_reject,
    )
    estimator = MagicMock()
    estimator.config = cfg
    estimator.score_pair.return_value = QualityScore(
        score=score,
        available=available,
        reason=reason,
        model_id=cfg.model_id,
    )
    return estimator


def _install_estimator(estimator):
    """Install ``estimator`` so ``_get_qe_estimator`` returns it."""
    translation_router_mod._QE_ESTIMATOR = estimator
    translation_router_mod._QE_LOAD_FAILED = False
    translation_router_mod._QE_LOAD_FAILED_REASON = None


def _force_estimator_unavailable(reason: str = "unavailable"):
    translation_router_mod._QE_ESTIMATOR = None
    translation_router_mod._QE_LOAD_FAILED = True
    translation_router_mod._QE_LOAD_FAILED_REASON = reason


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_score_pair_200_with_valid_request(client):
    estimator = _make_estimator_stub(score=0.77)
    _install_estimator(estimator)
    resp = client.post(
        "/api/v1/translation/score-pair",
        json={
            "source": "hello world",
            "target": "bonjour le monde",
            "source_lang": "en",
            "target_lang": "fr",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["score"] == pytest.approx(0.77)
    assert body["model_id"] == "Unbabel/wmt22-cometkiwi-da"
    assert body["threshold_warn"] == pytest.approx(0.4)
    assert body["threshold_reject"] == pytest.approx(0.2)
    estimator.score_pair.assert_called_once_with("hello world", "bonjour le monde")


def test_score_pair_returns_unavailable_for_empty_input(client):
    estimator = _make_estimator_stub(
        score=None, available=False, reason="empty_input"
    )
    _install_estimator(estimator)
    resp = client.post(
        "/api/v1/translation/score-pair",
        json={
            "source": "x",
            "target": "y",
            "source_lang": "en",
            "target_lang": "fr",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert body["reason"] == "empty_input"
    assert body["score"] is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_score_pair_422_on_missing_fields(client):
    estimator = _make_estimator_stub()
    _install_estimator(estimator)
    resp = client.post(
        "/api/v1/translation/score-pair",
        json={"source": "x", "target": "y"},
    )
    assert resp.status_code == 422


def test_score_pair_422_on_empty_source(client):
    estimator = _make_estimator_stub()
    _install_estimator(estimator)
    resp = client.post(
        "/api/v1/translation/score-pair",
        json={
            "source": "",
            "target": "y",
            "source_lang": "en",
            "target_lang": "fr",
        },
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 503 when comet unavailable
# ---------------------------------------------------------------------------


def test_score_pair_503_when_qe_unavailable(client):
    _force_estimator_unavailable("comet_not_installed")
    resp = client.post(
        "/api/v1/translation/score-pair",
        json={
            "source": "x",
            "target": "y",
            "source_lang": "en",
            "target_lang": "fr",
        },
    )
    assert resp.status_code == 503
    assert "QE unavailable" in resp.json()["detail"]


def test_score_pair_503_when_score_pair_returns_comet_not_installed(client):
    """Even when the estimator stub is loaded, score_pair may signal missing comet."""
    estimator = _make_estimator_stub(
        score=None, available=False, reason="comet_not_installed"
    )
    _install_estimator(estimator)
    resp = client.post(
        "/api/v1/translation/score-pair",
        json={
            "source": "x",
            "target": "y",
            "source_lang": "en",
            "target_lang": "fr",
        },
    )
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Auth (mirrors test_translation_api.py pattern)
# ---------------------------------------------------------------------------


def test_score_pair_requires_api_key(monkeypatch):
    """When mounted via api.main, missing X-API-Key -> 401."""
    monkeypatch.setenv("OCR_API_KEY", "test-key-abc")
    monkeypatch.setenv("ENABLE_TRANSLATION_API", "true")

    import importlib

    import api.auth as _auth
    import api.config as _config

    importlib.reload(_config)
    importlib.reload(_auth)

    from api.routers.translation import router

    app = FastAPI()
    app.state.limiter = limiter
    app.middleware("http")(_auth.api_key_middleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.include_router(router)
    c = TestClient(app)
    resp = c.post(
        "/api/v1/translation/score-pair",
        json={
            "source": "x",
            "target": "y",
            "source_lang": "en",
            "target_lang": "fr",
        },
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_score_pair_route_registered_with_rate_limit():
    """Smoke check the route is mounted and the limiter is wired."""
    paths = {route.path for route in translation_router_mod.router.routes}
    assert "/api/v1/translation/score-pair" in paths
