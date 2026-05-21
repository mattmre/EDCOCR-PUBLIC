"""Tests for REST API hardening features (auth, rate limiting, queue capacity)."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# Starlette's TestClient gained a `client=` kwarg in newer releases.
# Keep tests compatible with both signatures.
def _build_test_client(app, client_addr=None):
    if client_addr is None:
        return TestClient(app)
    try:
        return TestClient(app, client=client_addr)
    except TypeError:
        return TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    """FastAPI TestClient with isolated DB and temp dirs."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with patch("api.config.SOURCE_FOLDER", str(source)), \
         patch("api.config.OUTPUT_FOLDER", str(output)), \
         patch("api.job_manager.config") as mock_config:
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 1
        mock_config.MAX_CONCURRENT_JOBS = 64

        from api.main import create_app
        app = create_app()
        yield TestClient(app)


@pytest.fixture()
def sample_pdf(tmp_path) -> Path:
    """Create a minimal PDF file for upload testing."""
    pdf = tmp_path / "test_doc.pdf"
    # Minimal valid PDF
    pdf.write_bytes(
        b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
    )
    return pdf


# ---------------------------------------------------------------------------
# API Key Authentication Tests
# ---------------------------------------------------------------------------

class TestAPIKeyAuth:
    def test_startup_fails_when_auth_key_missing_without_override(self):
        """App startup fails unless empty OCR_API_KEY has explicit override."""
        from api.main import create_app

        with patch("api.auth.OCR_API_KEY", ""), patch(
            "api.auth.ALLOW_UNAUTHENTICATED", False
        ):
            with pytest.raises(RuntimeError, match="OCR_API_KEY"):
                create_app()

    def test_startup_allows_empty_key_with_explicit_override(self):
        """App startup succeeds when ALLOW_UNAUTHENTICATED=true is set."""
        from api.main import create_app

        with patch("api.auth.OCR_API_KEY", ""), patch(
            "api.auth.ALLOW_UNAUTHENTICATED", True
        ):
            app = create_app()
            client = TestClient(app)
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200

    def test_request_without_key_when_required(self, tmp_path):
        """When OCR_API_KEY is set, requests without key get 401."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()

        with patch("api.auth.OCR_API_KEY", "unit-test-api-token"), \
             patch("api.config.SOURCE_FOLDER", str(source)), \
             patch("api.config.OUTPUT_FOLDER", str(output)), \
             patch("api.job_manager.config") as mock_config:
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64

            from api.main import create_app
            app = create_app()
            client = TestClient(app)
            resp = client.get("/api/v1/jobs")
            assert resp.status_code == 401

    def test_request_with_valid_key(self, tmp_path):
        """When OCR_API_KEY is set, requests with valid key succeed."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()

        with patch("api.auth.OCR_API_KEY", "unit-test-api-token"), \
             patch("api.config.SOURCE_FOLDER", str(source)), \
             patch("api.config.OUTPUT_FOLDER", str(output)), \
             patch("api.job_manager.config") as mock_config:
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64

            from api.main import create_app
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/v1/jobs", headers={"X-API-Key": "unit-test-api-token"})
            assert resp.status_code == 200

    def test_health_exempt_from_auth(self, tmp_path):
        """Health endpoint works without API key even when auth is enabled."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()

        with patch("api.auth.OCR_API_KEY", "unit-test-api-token"), \
             patch("api.config.SOURCE_FOLDER", str(source)), \
             patch("api.config.OUTPUT_FOLDER", str(output)), \
             patch("api.job_manager.config") as mock_config:
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64

            from api.main import create_app
            app = create_app()
            client = TestClient(app)

            resp = client.get("/api/v1/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# Ingress Allowlist Tests
# ---------------------------------------------------------------------------

class TestIngressAllowlist:
    def test_allowlisted_client_can_access_api(self, tmp_path):
        """When API_ALLOWED_IPS is configured, allowlisted clients are permitted."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()

        with patch("api.auth.API_ALLOWED_IPS", ("203.0.113.10", "testclient")), \
             patch("api.config.SOURCE_FOLDER", str(source)), \
             patch("api.config.OUTPUT_FOLDER", str(output)), \
             patch("api.job_manager.config") as mock_config:
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64

            from api.main import create_app
            app = create_app()
            client = _build_test_client(app, ("203.0.113.10", 50001))

            resp = client.get("/api/v1/jobs")
            assert resp.status_code == 200

    def test_non_allowlisted_client_gets_403(self, tmp_path):
        """When API_ALLOWED_IPS is configured, non-allowlisted clients are denied."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()

        with patch("api.auth.API_ALLOWED_IPS", ("203.0.113.10",)), \
             patch("api.config.SOURCE_FOLDER", str(source)), \
             patch("api.config.OUTPUT_FOLDER", str(output)), \
             patch("api.job_manager.config") as mock_config:
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64

            from api.main import create_app
            app = create_app()
            client = _build_test_client(app, ("198.51.100.44", 50002))

            resp = client.get("/api/v1/jobs")
            assert resp.status_code == 403
            assert resp.json()["error"] == "forbidden"

    def test_exempt_health_endpoint_ignores_allowlist(self, tmp_path):
        """Health endpoint remains reachable even when allowlist is configured."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()

        with patch("api.auth.API_ALLOWED_IPS", ("203.0.113.10",)), \
             patch("api.config.SOURCE_FOLDER", str(source)), \
             patch("api.config.OUTPUT_FOLDER", str(output)), \
             patch("api.job_manager.config") as mock_config:
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64

            from api.main import create_app
            app = create_app()
            client = _build_test_client(app, ("198.51.100.44", 50003))

            resp = client.get("/api/v1/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# Queue Capacity Tests
# ---------------------------------------------------------------------------

class TestQueueCapacity:
    def test_submit_rejected_when_queue_full(self, tmp_path, sample_pdf):
        """Submit returns 429 when MAX_CONCURRENT_JOBS reached."""
        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()

        with patch("api.config.SOURCE_FOLDER", str(source)), \
             patch("api.config.OUTPUT_FOLDER", str(output)), \
             patch("api.config.MAX_CONCURRENT_JOBS", 2), \
             patch("api.job_manager.config") as mock_config:
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 2

            from api.main import create_app
            app = create_app()
            app.state.limiter.enabled = False
            app.state.limiter.reset()
            client = TestClient(app)

            # Submit 2 jobs to fill queue
            for i in range(2):
                with open(sample_pdf, "rb") as f:
                    resp = client.post(
                        "/api/v1/jobs",
                        files={"file": (f"test{i}.pdf", f, "application/pdf")},
                        data={"priority": "normal"},
                    )
                assert resp.status_code == 201

            # Third submission should be rejected
            with open(sample_pdf, "rb") as f:
                resp = client.post(
                    "/api/v1/jobs",
                    files={"file": ("test3.pdf", f, "application/pdf")},
                    data={"priority": "normal"},
                )
            assert resp.status_code == 429
            assert resp.json()["detail"]["error"] == "queue_full"

    def test_submit_accepted_when_queue_has_space(self, client, sample_pdf):
        """Submit succeeds when queue is not full."""
        with patch("api.config.MAX_CONCURRENT_JOBS", 10):
            with open(sample_pdf, "rb") as f:
                resp = client.post(
                    "/api/v1/jobs",
                    files={"file": ("test.pdf", f, "application/pdf")},
                    data={"priority": "normal"},
                )
            assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Rate Limiting Tests
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_rate_limit_uses_default_from_env(self):
        """Rate limiter respects OCR_RATE_LIMIT env var."""
        with patch.dict(os.environ, {"OCR_RATE_LIMIT": "5/minute"}):
            import importlib  # noqa: E402

            from api import limits
            importlib.reload(limits)
            assert limits.get_default_rate() == "5/minute"

    def test_submit_rate_limit_uses_env(self):
        """Submit rate limiter respects OCR_SUBMIT_RATE_LIMIT env var."""
        with patch.dict(os.environ, {"OCR_SUBMIT_RATE_LIMIT": "3/minute"}):
            import importlib  # noqa: E402

            from api import limits
            importlib.reload(limits)
            assert limits.get_submit_rate() == "3/minute"

    def test_rate_limiter_blocks_after_limit(self, tmp_path):
        """Rate limiter returns 429 after exceeding the request limit."""
        import importlib

        from slowapi import Limiter
        from slowapi.util import get_remote_address

        source = tmp_path / "source"
        output = tmp_path / "output"
        source.mkdir()
        output.mkdir()

        # Create a strict limiter (2 requests/minute)
        strict_limiter = Limiter(key_func=get_remote_address)

        with patch("api.config.SOURCE_FOLDER", str(source)), \
             patch("api.config.OUTPUT_FOLDER", str(output)), \
             patch("api.limits.limiter", strict_limiter), \
             patch("api.limits.get_default_rate", return_value="2/minute"), \
             patch("api.limits.get_submit_rate", return_value="2/minute"), \
             patch("api.job_manager.config") as mock_config:
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64

            # Reload jobs router so decorators pick up patched limiter & rates
            from api.routers import jobs as jobs_mod
            importlib.reload(jobs_mod)

            from api.main import create_app
            app = create_app()
            app.state.limiter = strict_limiter
            client = TestClient(app)

            # Make requests up to and beyond the limit
            statuses = []
            for _ in range(5):
                resp = client.get("/api/v1/jobs")
                statuses.append(resp.status_code)

            # At least one response should be 429
            assert 429 in statuses, f"Expected 429 in {statuses}"

            # Restore original modules for other tests
            from api import limits as limits_mod
            importlib.reload(limits_mod)
            importlib.reload(jobs_mod)


class TestPathConfiguration:
    def test_source_folder_honors_ocr_source_dir_alias(self, tmp_path):
        """Config honors OCR_SOURCE_DIR when SOURCE_FOLDER is unset."""
        source = tmp_path / "alias-source"

        from api import config as config_mod

        with patch.dict(
            os.environ,
            {"OCR_SOURCE_DIR": str(source), "SOURCE_FOLDER": ""},
            clear=False,
        ):
            config_mod = importlib.reload(config_mod)
            assert config_mod.SOURCE_FOLDER == str(source)

        importlib.reload(config_mod)

    def test_output_folder_honors_ocr_output_dir_alias(self, tmp_path):
        """Config and DB startup honor OCR_OUTPUT_DIR when OUTPUT_FOLDER is unset."""
        output = tmp_path / "alias-output"

        from api import config as config_mod
        from api import database as database_mod

        with patch.dict(
            os.environ,
            {"OCR_OUTPUT_DIR": str(output), "OUTPUT_FOLDER": "", "API_DB_PATH": ""},
            clear=False,
        ):
            config_mod = importlib.reload(config_mod)
            database_mod = importlib.reload(database_mod)
            database_mod.reset_engine()

            engine = database_mod.get_engine()

            assert config_mod.OUTPUT_FOLDER == str(output)
            assert Path(config_mod.DB_PATH) == output / "jobs.db"
            assert Path(config_mod.DB_PATH).parent.exists()
            assert engine is not None

        database_mod.reset_engine()
        importlib.reload(config_mod)
        importlib.reload(database_mod)


# ---------------------------------------------------------------------------
# Path Traversal Tests
# ---------------------------------------------------------------------------

class TestPathTraversal:
    def test_source_path_rejects_traversal(self):
        """Form parser rejects source_path with path traversal characters."""
        from fastapi import HTTPException

        from api.deps import parse_job_submit_form

        with pytest.raises(HTTPException) as exc_info:
            parse_job_submit_form(
                source_path="../../../etc/passwd",
                priority="normal",
                enable_docintel=False,
                docintel_mode="full",
            )
        assert exc_info.value.status_code == 422

    def test_source_path_allows_safe_paths(self):
        """Form parser allows valid safe paths under SOURCE_FOLDER."""
        from api.deps import parse_job_submit_form

        with patch("api.config.SOURCE_FOLDER", "C:\\app\\ocr_source"):
            result = parse_job_submit_form(
                source_path="C:\\app\\ocr_source\\documents\\report.pdf",
                priority="normal",
                enable_docintel=False,
                docintel_mode="full",
            )
        assert result.source_path == "C:\\app\\ocr_source\\documents\\report.pdf"

    def test_source_path_rejects_paths_outside_ingest_root(self):
        """Form parser rejects absolute paths outside SOURCE_FOLDER."""
        from fastapi import HTTPException

        from api.deps import parse_job_submit_form

        with patch("api.config.SOURCE_FOLDER", "C:\\app\\ocr_source"):
            with pytest.raises(HTTPException) as exc_info:
                parse_job_submit_form(
                    source_path="C:\\etc\\passwd",
                    priority="normal",
                    enable_docintel=False,
                    docintel_mode="full",
                )
        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["error"] == "path_not_allowed"


# ---------------------------------------------------------------------------
# Dependency Injection Tests
# ---------------------------------------------------------------------------

class TestFormParsing:
    def test_parse_job_submit_form_defaults(self):
        """Form parser applies correct defaults."""
        from api.deps import parse_job_submit_form

        # Simulate default form values
        result = parse_job_submit_form(
            source_path=None,
            priority="normal",
            enable_docintel=False,
            docintel_mode="full",
            processing_timeout_minutes=None,
        )

        assert result.source_path is None
        assert result.priority == "normal"
        assert result.enable_docintel is False
        assert result.docintel_mode == "full"
        assert result.processing_timeout_minutes is None

    def test_parse_job_submit_form_validates_priority(self):
        """Form parser validates priority via Pydantic, raises HTTPException."""
        from fastapi import HTTPException

        from api.deps import parse_job_submit_form

        with pytest.raises(HTTPException) as exc_info:
            parse_job_submit_form(
                source_path=None,
                priority="invalid",
                enable_docintel=False,
                docintel_mode="full",
                processing_timeout_minutes=None,
            )
        assert exc_info.value.status_code == 422

    def test_parse_job_submit_form_validates_processing_timeout(self):
        """Form parser rejects non-positive processing timeout overrides."""
        from fastapi import HTTPException

        from api.deps import parse_job_submit_form

        with pytest.raises(HTTPException) as exc_info:
            parse_job_submit_form(
                source_path=None,
                priority="normal",
                enable_docintel=False,
                docintel_mode="full",
                processing_timeout_minutes=0,
            )
        assert exc_info.value.status_code == 422

    def test_parse_job_submit_form_returns_structured_webhook_error(self):
        """Invalid webhook URLs return a structured API error payload."""
        from fastapi import HTTPException

        from api.deps import parse_job_submit_form

        with pytest.raises(HTTPException) as exc_info:
            parse_job_submit_form(
                source_path=None,
                priority="normal",
                enable_docintel=False,
                docintel_mode="full",
                webhook_url="ftp://invalid-scheme.local/webhook",
            )

        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["error"] == "invalid_webhook_url"
