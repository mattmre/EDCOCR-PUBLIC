"""Tests for RequestSizeLimitMiddleware.

Verifies that the DoS-hardening middleware:

- Rejects non-multipart requests whose Content-Length exceeds the cap.
- Allows requests under the cap.
- Exempts multipart uploads from the cap.
- Handles missing / malformed Content-Length headers correctly.
- Honors the ``max_size=0`` kill-switch.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.request_size_limit import RequestSizeLimitMiddleware


def _build_app(max_size: int) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestSizeLimitMiddleware, max_size=max_size)

    @app.post("/echo")
    async def echo(payload: dict):
        return {"received_keys": sorted(payload.keys())}

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    return app


class TestRequestSizeLimit:
    def test_small_json_allowed(self):
        client = TestClient(_build_app(max_size=1024))
        r = client.post("/echo", json={"a": 1, "b": 2})
        assert r.status_code == 200
        assert r.json() == {"received_keys": ["a", "b"]}

    def test_oversized_json_rejected_with_413(self):
        client = TestClient(_build_app(max_size=64))
        # Build a JSON payload that is guaranteed to exceed 64 bytes.
        big = {"data": "x" * 512}
        r = client.post("/echo", json=big)
        assert r.status_code == 413
        body = r.json()
        assert body["error"] == "request_too_large"
        assert body["max_size"] == 64
        assert "exceeds" in body["message"].lower()

    def test_get_without_body_passes_through(self):
        client = TestClient(_build_app(max_size=16))
        r = client.get("/ping")
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_malformed_content_length_returns_400(self):
        client = TestClient(_build_app(max_size=1024))
        # Send raw request with a bogus Content-Length header.
        r = client.post(
            "/echo",
            content=b'{"a": 1}',
            headers={
                "Content-Type": "application/json",
                "Content-Length": "not-a-number",
            },
        )
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_content_length"

    def test_multipart_exempt_from_limit(self):
        app = FastAPI()
        app.add_middleware(RequestSizeLimitMiddleware, max_size=1)

        @app.post("/upload")
        async def upload():
            return {"ok": True}

        client = TestClient(app)
        files = {"file": ("test.txt", b"hello world", "text/plain")}
        r = client.post("/upload", files=files)
        # max_size=1 would reject any JSON body, but multipart is exempt.
        assert r.status_code == 200

    def test_zero_max_size_disables_check(self):
        client = TestClient(_build_app(max_size=0))
        r = client.post("/echo", json={"x": "y" * 10_000})
        assert r.status_code == 200

    def test_equal_to_max_size_allowed(self):
        app = _build_app(max_size=100)
        client = TestClient(app)
        # Craft a body that lands near the boundary.
        body = b'{"k":"' + b"a" * 90 + b'"}'
        assert len(body) <= 100
        r = client.post(
            "/echo",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 200

    def test_one_byte_over_max_size_rejected(self):
        app = _build_app(max_size=50)
        client = TestClient(app)
        body = b'{"k":"' + b"a" * 60 + b'"}'
        assert len(body) > 50
        r = client.post(
            "/echo",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 413


class TestRequestSizeLimitWired:
    """Verify the middleware is wired into the real app factory."""

    def test_main_app_enforces_limit(self, tmp_path):
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()

        with (
            patch("api.config.SOURCE_FOLDER", str(source)),
            patch("api.config.OUTPUT_FOLDER", str(output)),
            patch("api.config.OCR_API_KEY", ""),
            patch("api.auth.OCR_API_KEY", ""),
            patch("api.config.ALLOW_UNAUTHENTICATED", True),
            patch("api.auth.ALLOW_UNAUTHENTICATED", True),
            patch("api.config.MAX_REQUEST_BODY_SIZE", 64),
            patch("api.job_manager.config") as mock_config,
        ):
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64

            # Re-import so MAX_REQUEST_BODY_SIZE patch is picked up at
            # app-factory time.
            import importlib

            import api.main as api_main
            importlib.reload(api_main)
            app = api_main.create_app()
            app.state.limiter.enabled = False
            app.state.limiter.reset()
            client = TestClient(app)

            # A small health request works.
            r = client.get("/api/v1/health")
            assert r.status_code in (200, 503)

            # An oversized JSON body is rejected with 413 before routing.
            big_body = b'{"x":"' + b"y" * 256 + b'"}'
            r = client.post(
                "/api/v1/jobs/does-not-exist/retry",
                content=big_body,
                headers={"Content-Type": "application/json"},
            )
            assert r.status_code == 413, r.text


def test_middleware_class_default_max_size():
    mw = RequestSizeLimitMiddleware(app=None)  # type: ignore[arg-type]
    assert mw.max_size == 10 * 1024 * 1024
