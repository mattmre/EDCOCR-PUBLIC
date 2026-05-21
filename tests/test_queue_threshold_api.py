"""Tests for OCR queue threshold configuration endpoints."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.database import get_engine, reset_engine
from api.queue_alerting import get_queue_monitor


@pytest.fixture(autouse=True)
def _isolate_api(tmp_path):
    db_file = str(tmp_path / "test_queue_thresholds.db")
    thresholds_file = str(tmp_path / "queue_thresholds.json")
    with patch.dict(
        os.environ,
        {
            "ALLOW_UNAUTHENTICATED": "true",
            "ANONYMOUS_ROLE": "admin",
            "ENABLE_DASHBOARD": "true",
            "API_DB_PATH": db_file,
            "OCR_QUEUE_THRESHOLDS_PATH": thresholds_file,
        },
    ), patch("api.config.DB_PATH", db_file), patch("api.database.DB_PATH", db_file), \
        patch("api.config.OCR_API_KEY", ""), \
        patch("api.config.ALLOW_UNAUTHENTICATED", True), \
        patch("api.config.ANONYMOUS_ROLE", "admin"), \
        patch("api.auth.OCR_API_KEY", ""), \
        patch("api.auth.ALLOW_UNAUTHENTICATED", True), \
        patch("api.auth.ANONYMOUS_ROLE", "admin"):
        import api.queue_alerting as queue_alerting

        queue_alerting._monitor = None
        reset_engine()
        get_engine(db_file)
        get_queue_monitor().reset()
        yield thresholds_file
        get_queue_monitor().reset()
        queue_alerting._monitor = None
        reset_engine()


@pytest.fixture()
def client():
    from api.main import create_app

    app = create_app()
    app.state.limiter.enabled = False
    app.state.limiter.reset()
    with TestClient(app) as test_client:
        yield test_client


def test_update_queue_threshold_returns_config(client):
    response = client.put(
        "/api/v1/queues/ocr_gpu/threshold",
        json={
            "warning_depth": 25,
            "critical_depth": 50,
            "warning_wait_seconds": 60,
            "critical_wait_seconds": 120,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "queue_name": "ocr_gpu",
        "warning_depth": 25,
        "critical_depth": 50,
        "warning_wait_seconds": 60.0,
        "critical_wait_seconds": 120.0,
    }


def test_get_queue_threshold_after_update(client):
    client.put(
        "/api/v1/queues/ocr_cpu/threshold",
        json={
            "warning_depth": 10,
            "critical_depth": 20,
            "warning_wait_seconds": 30,
            "critical_wait_seconds": 45,
        },
    )

    response = client.get("/api/v1/queues/ocr_cpu/threshold")

    assert response.status_code == 200
    assert response.json()["warning_depth"] == 10
    assert response.json()["critical_wait_seconds"] == 45.0


def test_threshold_update_persists_to_config_file(client, _isolate_api):
    client.put(
        "/api/v1/queues/ocr_gpu/threshold",
        json={
            "warning_depth": 25,
            "critical_depth": 50,
            "warning_wait_seconds": 60,
            "critical_wait_seconds": 120,
        },
    )

    with open(_isolate_api, encoding="utf-8") as fh:
        payload = json.load(fh)

    assert payload[0]["queue_name"] == "ocr_gpu"
    assert payload[0]["critical_wait_seconds"] == 120.0


def test_list_queue_thresholds(client):
    client.put(
        "/api/v1/queues/ocr_gpu/threshold",
        json={
            "warning_depth": 25,
            "critical_depth": 50,
            "warning_wait_seconds": 60,
            "critical_wait_seconds": 120,
        },
    )

    response = client.get("/api/v1/queues/thresholds")

    assert response.status_code == 200
    assert response.json() == [
        {
            "queue_name": "ocr_gpu",
            "warning_depth": 25,
            "critical_depth": 50,
            "warning_wait_seconds": 60.0,
            "critical_wait_seconds": 120.0,
        }
    ]


def test_get_missing_queue_threshold_returns_404(client):
    response = client.get("/api/v1/queues/missing/threshold")

    assert response.status_code == 404


def test_invalid_threshold_ordering_returns_422(client):
    response = client.put(
        "/api/v1/queues/ocr_gpu/threshold",
        json={
            "warning_depth": 100,
            "critical_depth": 50,
            "warning_wait_seconds": 60,
            "critical_wait_seconds": 120,
        },
    )

    assert response.status_code == 422
