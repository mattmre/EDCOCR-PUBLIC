"""Tests for the OCR REST API — jobs and health endpoints."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

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

    from api import event_store

    # API tests create durable events while exercising submission/retry.  Do
    # not let a module singleton from another test retain its production path
    # or SQLite connection.
    event_store.reset_event_store()
    with (
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.config.EVENT_STORE_PATH", str(output / "event_store.db")),
        patch("api.config.OCR_API_KEY", ""),
        patch("api.auth.OCR_API_KEY", ""),
        patch("api.config.ANONYMOUS_ROLE", "operator"),
        patch("api.auth.ANONYMOUS_ROLE", "operator"),
        patch("api.auth.ALLOW_UNAUTHENTICATED", True),
        patch("api.config.ALLOW_UNAUTHENTICATED", True),
        patch("api.identity._anonymous_role", return_value="operator"),
        patch.dict(os.environ, {"OCR_ALLOW_CLIENT_URGENT": "true"}),
        patch("api.job_manager.config") as mock_config,
    ):
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 1
        mock_config.MAX_CONCURRENT_JOBS = 64

        from api.main import create_app

        app = create_app()
        from api import limits as api_limits
        from api import main as api_main

        api_limits.limiter.enabled = False
        api_main.limiter.enabled = False
        app.state.limiter.enabled = False
        app.state.limiter.reset()
        try:
            yield TestClient(app)
        finally:
            event_store.reset_event_store()


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
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_includes_job_counts(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "jobs" in data


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------


class TestJobSubmission:
    def test_submit_with_file_upload(self, client, sample_pdf):
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"priority": "normal"},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "submitted"
        assert "job_id" in data
        assert data["source_file"] == "test.pdf"
        assert data["priority"] == "normal"
        assert "links" in data

    def test_submit_with_source_path(self, client, sample_pdf, tmp_path):
        # Place file where source_path can find it
        src = tmp_path / "source" / "input.pdf"
        src.write_bytes(sample_pdf.read_bytes())

        resp = client.post(
            "/api/v1/jobs",
            data={"source_path": str(src), "priority": "urgent"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["source_file"] == "input.pdf"
        assert data["priority"] == "urgent"

    def test_submit_no_file_or_path_returns_400(self, client):
        resp = client.post("/api/v1/jobs", data={"priority": "normal"})
        assert resp.status_code == 400

    def test_submit_invalid_priority_returns_400(self, client, sample_pdf):
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"priority": "invalid"},
            )
        # Pydantic validation returns 422 for invalid enum values
        assert resp.status_code == 422

    def test_submit_nonexistent_path_returns_404(self, client):
        from api import config

        resp = client.post(
            "/api/v1/jobs",
            data={"source_path": str(Path(config.SOURCE_FOLDER) / "missing.pdf")},
        )
        assert resp.status_code == 404

    def test_submit_source_path_outside_ingest_root_returns_422(
        self, client, sample_pdf, tmp_path
    ):
        outside = tmp_path / "outside" / "input.pdf"
        outside.parent.mkdir(exist_ok=True)
        outside.write_bytes(sample_pdf.read_bytes())

        resp = client.post(
            "/api/v1/jobs",
            data={"source_path": str(outside)},
        )

        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "path_not_allowed"


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------


class TestJobStatus:
    def test_get_status_of_submitted_job(self, client, sample_pdf):
        with open(sample_pdf, "rb") as f:
            submit = client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
            )
        job_id = submit.json()["job_id"]

        resp = client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert "progress" in data

    def test_get_status_not_found(self, client):
        resp = client.get("/api/v1/jobs/job_000000000000")
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "job_not_found"

    def test_failed_status_exposes_tenant_scoped_terminal_reason(
        self, client, sample_pdf
    ):
        with patch("api.job_manager.JobManager._ensure_workers", return_value=None):
            with open(sample_pdf, "rb") as f:
                submit = client.post(
                    "/api/v1/jobs",
                    files={"file": ("test.pdf", f, "application/pdf")},
                )
        job_id = submit.json()["job_id"]

        from api.database import Job, get_session_factory

        session = get_session_factory()()
        job = session.get(Job, job_id)
        job.status = "failed"
        job.error_message = "Job interrupted by API restart before completion"
        session.commit()
        session.close()

        failed = client.get(f"/api/v1/jobs/{job_id}")
        assert failed.status_code == 200
        assert failed.json()["error_message"] == (
            "Job interrupted by API restart before completion"
        )


# ---------------------------------------------------------------------------
# Job listing
# ---------------------------------------------------------------------------


class TestJobListing:
    def test_list_empty(self, client):
        resp = client.get("/api/v1/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    def test_list_with_jobs(self, client, sample_pdf):
        # Submit 2 jobs
        for _ in range(2):
            with open(sample_pdf, "rb") as f:
                client.post(
                    "/api/v1/jobs",
                    files={"file": ("test.pdf", f, "application/pdf")},
                )

        resp = client.get("/api/v1/jobs")
        data = resp.json()
        assert data["total"] == 2
        assert len(data["jobs"]) == 2

    def test_list_filter_by_status(self, client, sample_pdf):
        with open(sample_pdf, "rb") as f:
            client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
            )

        # Filter for completed (should be empty since job just submitted)
        resp = client.get("/api/v1/jobs?status=completed")
        data = resp.json()
        assert data["total"] == 0


# ---------------------------------------------------------------------------
# Job result
# ---------------------------------------------------------------------------


class TestJobResult:
    def _create_completed_job_with_pdf_artifact(self, tmp_path):
        from api.database import Job, get_session_factory

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        result_dir = tmp_path / "output" / job_id
        pdf_dir = result_dir / "EXPORT" / "PDF"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = pdf_dir / "result.pdf"
        artifact_path.write_bytes(
            b"%PDF-1.0\n1 0 obj<</Type/Catalog>>endobj\n%%EOF\n",
        )
        text_dir = result_dir / "EXPORT" / "TEXT"
        text_dir.mkdir(parents=True, exist_ok=True)
        (text_dir / "result.txt").write_text("owned OCR text", encoding="utf-8")
        structure_dir = result_dir / "EXPORT" / "STRUCTURE"
        structure_dir.mkdir(parents=True, exist_ok=True)
        (structure_dir / "result.json").write_text(
            '{"pages": [{"page": 1}]}',
            encoding="utf-8",
        )

        session = get_session_factory()()
        try:
            job = Job(
                job_id=job_id,
                source_file="test.pdf",
                status="completed",
                result_path=str(result_dir),
            )
            session.add(job)
            session.commit()
        finally:
            session.close()

        return job_id, artifact_path

    def test_result_not_complete_returns_409(self, client, sample_pdf):
        # Keep the job in "submitted" so this assertion does not race a fast
        # background pipeline completion on CI.
        with patch("api.job_manager.JobManager._ensure_workers", return_value=None):
            with open(sample_pdf, "rb") as f:
                submit = client.post(
                    "/api/v1/jobs",
                    files={"file": ("test.pdf", f, "application/pdf")},
                )
        job_id = submit.json()["job_id"]

        resp = client.get(f"/api/v1/jobs/{job_id}/result")
        assert resp.status_code == 409

    def test_result_not_found(self, client):
        resp = client.get("/api/v1/jobs/job_000000000000/result")
        assert resp.status_code == 404

    def test_download_not_found(self, client):
        resp = client.get("/api/v1/jobs/job_000000000000/result/download?type=pdf")
        assert resp.status_code == 404

    @pytest.mark.parametrize(
        ("artifact_type", "expected_content_type", "expected_payload"),
        [
            ("pdf", "application/pdf", b"%PDF-1.0"),
            ("text", "text/plain", b"owned OCR text"),
            ("structure", "application/json", b'"pages"'),
        ],
    )
    def test_download_valid_result_artifacts(
        self,
        client,
        tmp_path,
        artifact_type,
        expected_content_type,
        expected_payload,
    ):
        job_id, _ = self._create_completed_job_with_pdf_artifact(tmp_path)

        resp = client.get(
            f"/api/v1/jobs/{job_id}/result/download",
            params={"type": artifact_type},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith(expected_content_type)
        assert expected_payload in resp.content

    def test_download_rejects_malicious_artifact_type(self, client, tmp_path):
        job_id, _ = self._create_completed_job_with_pdf_artifact(tmp_path)

        resp = client.get(
            f"/api/v1/jobs/{job_id}/result/download",
            params={"type": "../../etc/passwd"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "artifact_not_found"

    def test_download_rejects_artifact_outside_job_result_root(self, client, tmp_path):
        from api.database import Job, get_session_factory

        job_id, _ = self._create_completed_job_with_pdf_artifact(tmp_path)
        outside_artifact = tmp_path / "outside.pdf"
        outside_artifact.write_bytes(
            b"%PDF-1.0\n1 0 obj<</Type/Catalog>>endobj\n%%EOF\n",
        )

        session = get_session_factory()()
        try:
            job = session.get(Job, job_id)
            job.result_path = str(outside_artifact.parent)
            session.commit()
        finally:
            session.close()

        resp = client.get(f"/api/v1/jobs/{job_id}/result/download?type=pdf")

        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "output_unavailable"

    @pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
    def test_download_refuses_linked_result_artifact(
        self,
        client,
        tmp_path,
        link_kind,
    ):
        job_id, artifact_path = self._create_completed_job_with_pdf_artifact(tmp_path)
        foreign = tmp_path / "foreign.pdf"
        foreign.write_bytes(b"foreign tenant evidence")
        artifact_path.unlink()
        try:
            if link_kind == "symlink":
                artifact_path.symlink_to(foreign)
            else:
                os.link(foreign, artifact_path)
        except OSError:
            pytest.skip(f"{link_kind} is not supported")

        resp = client.get(f"/api/v1/jobs/{job_id}/result/download?type=pdf")

        assert resp.status_code == 404
        assert b"foreign tenant evidence" not in resp.content

    def test_download_refuses_parent_root_replacement(self, client, tmp_path):
        job_id, artifact_path = self._create_completed_job_with_pdf_artifact(tmp_path)
        result_root = artifact_path.parents[2]
        original_root = result_root.with_name(f"{result_root.name}-original")

        from api.output_manifest import build_manifest_from_root_fd as real_build

        def _swap_after_discovery(*args, **kwargs):
            manifest = real_build(*args, **kwargs)
            result_root.rename(original_root)
            foreign_pdf = result_root / "EXPORT" / "PDF" / "result.pdf"
            foreign_pdf.parent.mkdir(parents=True)
            foreign_pdf.write_bytes(b"foreign tenant evidence")
            return manifest

        with patch(
            "api.routers.jobs.build_manifest_from_root_fd",
            side_effect=_swap_after_discovery,
        ):
            resp = client.get(f"/api/v1/jobs/{job_id}/result/download?type=pdf")

        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "artifact_not_found"
        assert b"foreign tenant evidence" not in resp.content

    def test_download_refuses_pdf_directory_replacement(self, client, tmp_path):
        job_id, artifact_path = self._create_completed_job_with_pdf_artifact(tmp_path)
        pdf_dir = artifact_path.parent
        original_pdf_dir = pdf_dir.with_name("PDF-authorized")
        from api.output_manifest import build_manifest_from_root_fd as real_build

        def _swap_child_after_discovery(*args, **kwargs):
            manifest = real_build(*args, **kwargs)
            pdf_dir.rename(original_pdf_dir)
            pdf_dir.mkdir()
            (pdf_dir / artifact_path.name).write_bytes(b"foreign tenant evidence")
            return manifest

        with patch(
            "api.routers.jobs.build_manifest_from_root_fd",
            side_effect=_swap_child_after_discovery,
        ):
            resp = client.get(f"/api/v1/jobs/{job_id}/result/download?type=pdf")

        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "artifact_not_found"
        assert b"foreign tenant evidence" not in resp.content


# ---------------------------------------------------------------------------
# Job cancellation
# ---------------------------------------------------------------------------


class TestJobCancellation:
    def test_cancel_submitted_job(self, client, sample_pdf):
        with open(sample_pdf, "rb") as f:
            submit = client.post(
                "/api/v1/jobs",
                files={"file": ("test.pdf", f, "application/pdf")},
            )
        job_id = submit.json()["job_id"]

        resp = client.delete(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Job retry
# ---------------------------------------------------------------------------


class TestJobRetry:
    def _submit_and_fail(self, client, sample_pdf) -> str:
        """Submit a job and force it into 'failed' status."""
        # Prevent the background pipeline thread from racing the manual status
        # override below. Without this, the async worker can restore the job to
        # processing/completed before retry runs, which makes the test flaky.
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            with open(sample_pdf, "rb") as f:
                submit = client.post(
                    "/api/v1/jobs",
                    files={"file": ("test.pdf", f, "application/pdf")},
                )
        job_id = submit.json()["job_id"]

        # Force job into failed status via DB
        from api.database import Job, get_session_factory

        session = get_session_factory()()
        job = session.get(Job, job_id)
        job.status = "failed"
        job.error_message = "Test failure"
        session.commit()
        session.close()
        return job_id

    def test_retry_failed_job_returns_201(self, client, sample_pdf):
        with patch("api.job_manager.JobManager._ensure_workers", return_value=None):
            job_id = self._submit_and_fail(client, sample_pdf)
            resp = client.post(f"/api/v1/jobs/{job_id}/retry")
        assert resp.status_code == 201
        data = resp.json()
        assert data["job_id"] != job_id
        assert data["status"] == "submitted"
        assert data["source_file"] == "test.pdf"

    def test_retry_nonexistent_job_returns_404(self, client):
        resp = client.post("/api/v1/jobs/job_000000000000/retry")
        assert resp.status_code == 404

    def test_retry_active_job_returns_409(self, client, sample_pdf):
        # Keep the job in the initial submitted state so retry checks the
        # active-job guard instead of racing a background failure transition.
        with patch("api.job_manager.JobManager._ensure_workers", return_value=None):
            with open(sample_pdf, "rb") as f:
                submit = client.post(
                    "/api/v1/jobs",
                    files={"file": ("test.pdf", f, "application/pdf")},
                )
        job_id = submit.json()["job_id"]
        # Job is still "submitted" (not failed) — retry should be 409
        resp = client.post(f"/api/v1/jobs/{job_id}/retry")
        assert resp.status_code == 409

    def test_retry_cancelled_job_returns_201(self, client, sample_pdf):
        with patch("api.job_manager.JobManager._ensure_workers", return_value=None):
            with open(sample_pdf, "rb") as f:
                submit = client.post(
                    "/api/v1/jobs",
                    files={"file": ("test.pdf", f, "application/pdf")},
                )
            job_id = submit.json()["job_id"]

            # Cancel it first
            client.delete(f"/api/v1/jobs/{job_id}")

            # Retry the cancelled job
            resp = client.post(f"/api/v1/jobs/{job_id}/retry")
        assert resp.status_code == 201
        data = resp.json()
        assert data["job_id"] != job_id

    def test_retry_preserves_settings(self, client, sample_pdf):
        """Retry should preserve original job settings, including timeout overrides."""
        # Prevent the async pipeline thread from racing this test's manual
        # failure transition, which can intermittently turn the retry into 409.
        with patch("api.job_manager.JobManager._ensure_workers", return_value=None):
            with open(sample_pdf, "rb") as f:
                submit = client.post(
                    "/api/v1/jobs",
                    files={"file": ("test.pdf", f, "application/pdf")},
                    data={
                        "priority": "urgent",
                        "enable_docintel": "true",
                        "docintel_mode": "tables_only",
                        "processing_timeout_minutes": "7",
                    },
                )
            job_id = submit.json()["job_id"]

            # Force fail
            from api.database import Job, get_session_factory

            session = get_session_factory()()
            job = session.get(Job, job_id)
            job.status = "failed"
            session.commit()
            session.close()

            resp = client.post(f"/api/v1/jobs/{job_id}/retry")
            assert resp.status_code == 201
            assert resp.json()["priority"] == "urgent"
            assert resp.json()["status"] == "submitted"

        retry_job_id = resp.json()["job_id"]
        status = client.get(f"/api/v1/jobs/{retry_job_id}")
        settings = status.json()["settings"]
        assert settings["enable_docintel"] is True
        assert settings["docintel_mode"] == "tables_only"
        assert settings["processing_timeout_minutes"] == 7

    def test_cancel_nonexistent_returns_404(self, client):
        resp = client.delete("/api/v1/jobs/job_000000000000")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Database model tests
# ---------------------------------------------------------------------------


class TestDatabaseModel:
    def test_job_settings_serialization(self):
        from api.database import Job

        job = Job(
            job_id="test_123",
            source_file="doc.pdf",
        )
        job.settings = {"enable_docintel": True, "mode": "full"}
        assert job.settings["enable_docintel"] is True
        assert job.settings["mode"] == "full"

    def test_job_percent_complete(self):
        from api.database import Job

        job = Job(job_id="test_456", source_file="doc.pdf")
        job.total_pages = 100
        job.pages_completed = 42
        assert job.percent_complete() == 42.0

    def test_job_percent_complete_zero_pages(self):
        from api.database import Job

        job = Job(job_id="test_789", source_file="doc.pdf")
        assert job.percent_complete() == 0.0


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestPydanticModels:
    def test_job_submit_request_defaults(self):
        from api.models import JobSubmitRequest

        req = JobSubmitRequest()
        assert req.priority == "normal"
        assert req.enable_docintel is False
        assert req.docintel_mode == "full"
        assert req.processing_timeout_minutes is None

    def test_error_response(self):
        from api.models import ErrorResponse

        err = ErrorResponse(error="test", message="Test error")
        assert err.error == "test"
        assert err.details == {}

    def test_health_response(self):
        from api.models import HealthResponse

        h = HealthResponse(status="healthy", version="0.4.0", uptime_seconds=100.0)
        assert h.status == "healthy"
