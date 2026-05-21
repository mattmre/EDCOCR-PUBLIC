"""Tests for the OCR REST API — jobs and health endpoints."""

from __future__ import annotations

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
        app.state.limiter.enabled = False
        app.state.limiter.reset()
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

    def test_submit_source_path_outside_ingest_root_returns_422(self, client, sample_pdf, tmp_path):
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

    def test_download_rejects_malicious_artifact_type(self, client, tmp_path):
        job_id, _ = self._create_completed_job_with_pdf_artifact(tmp_path)

        resp = client.get(
            f"/api/v1/jobs/{job_id}/result/download",
            params={"type": "../../etc/passwd"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "artifact_not_found"

    def test_download_rejects_artifact_outside_job_result_root(self, client, tmp_path):
        job_id, _ = self._create_completed_job_with_pdf_artifact(tmp_path)
        outside_artifact = tmp_path / "outside.pdf"
        outside_artifact.write_bytes(
            b"%PDF-1.0\n1 0 obj<</Type/Catalog>>endobj\n%%EOF\n",
        )

        with patch(
            "api.job_manager.JobManager.get_result_artifacts",
            return_value={"pdf": str(outside_artifact)},
        ):
            resp = client.get(f"/api/v1/jobs/{job_id}/result/download?type=pdf")

        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "path_not_allowed"


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
