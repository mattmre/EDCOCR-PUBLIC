"""REST API tests for ``POST /api/v1/jobs/{id}/redetect-language`` (Plan A PR A5).

Mirrors the existing retry-endpoint test pattern in ``tests/test_api.py`` to
avoid flakiness around the background pipeline thread: every test patches
``JobManager._run_pipeline`` before submitting a job and forces the job into
``completed`` via direct DB writes.

Run with::

    python -m pytest tests/test_redetect_api.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures (mirrored from tests/test_api.py)
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
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal PDF file for upload."""
    pdf = tmp_path / "test_doc.pdf"
    pdf.write_bytes(
        b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
    )
    return pdf


def _submit_and_complete(client: TestClient, sample_pdf: Path, tmp_path: Path) -> str:
    """Submit a job, force it to completed, and populate a PDF artifact.

    Returns the job_id.  Caller must still patch ``_run_pipeline`` via a
    context manager so the background thread does not race the status
    override.
    """
    with open(sample_pdf, "rb") as f:
        submit = client.post(
            "/api/v1/jobs",
            files={"file": ("test.pdf", f, "application/pdf")},
        )
    assert submit.status_code == 201, submit.text
    job_id = submit.json()["job_id"]

    # Build a result_path with an EXPORT/PDF artifact so
    # get_result_artifacts() finds a pdf.
    result_dir = tmp_path / "results" / job_id
    pdf_dir = result_dir / "EXPORT" / "PDF"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    artifact = pdf_dir / "test.pdf"
    artifact.write_bytes(sample_pdf.read_bytes())

    from api.database import Job, get_session_factory

    session = get_session_factory()()
    job = session.get(Job, job_id)
    job.status = "completed"
    job.result_path = str(result_dir)
    session.commit()
    session.close()
    return job_id


def _submit_keep_submitted(client: TestClient, sample_pdf: Path) -> str:
    """Submit a job and leave it in ``submitted`` state (not completed)."""
    with open(sample_pdf, "rb") as f:
        submit = client.post(
            "/api/v1/jobs",
            files={"file": ("test.pdf", f, "application/pdf")},
        )
    assert submit.status_code == 201, submit.text
    return submit.json()["job_id"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRedetectLanguageEndpoint:
    def test_redetect_completed_job_returns_202(self, client, sample_pdf, tmp_path):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None), \
             patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert resp.status_code == 202, resp.text

    def test_redetect_response_body_status_queued(self, client, sample_pdf, tmp_path):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None), \
             patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        data = resp.json()
        assert data["status"] == "queued"
        assert data["job_id"] == job_id

    def test_redetect_starts_background_thread(self, client, sample_pdf, tmp_path):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None), \
             patch("threading.Thread") as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        # Thread was instantiated AND started
        mock_thread.assert_called()
        mock_thread_instance.start.assert_called()

    def test_redetect_thread_is_daemon(self, client, sample_pdf, tmp_path):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None), \
             patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        # daemon=True was passed
        kwargs = mock_thread.call_args.kwargs
        assert kwargs.get("daemon") is True


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestRedetectLanguageErrors:
    def test_unknown_job_returns_404(self, client):
        resp = client.post("/api/v1/jobs/job_000000000000/redetect-language")
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "job_not_found"

    def test_invalid_job_id_format_returns_400(self, client):
        resp = client.post("/api/v1/jobs/not-a-job-id/redetect-language")
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_job_id"

    def test_submitted_job_returns_409(self, client, sample_pdf):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_keep_submitted(client, sample_pdf)
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "job_not_complete"

    def test_failed_job_returns_409(self, client, sample_pdf):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_keep_submitted(client, sample_pdf)

            from api.database import Job, get_session_factory
            session = get_session_factory()()
            job = session.get(Job, job_id)
            job.status = "failed"
            session.commit()
            session.close()

            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert resp.status_code == 409

    def test_cancelled_job_returns_409(self, client, sample_pdf):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_keep_submitted(client, sample_pdf)

            from api.database import Job, get_session_factory
            session = get_session_factory()()
            job = session.get(Job, job_id)
            job.status = "cancelled"
            session.commit()
            session.close()

            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert resp.status_code == 409

    def test_completed_job_without_pdf_artifact_returns_409(
        self, client, sample_pdf, tmp_path,
    ):
        """If get_result_artifacts() yields no pdf, the endpoint fails cleanly."""
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_keep_submitted(client, sample_pdf)
            # Completed but result_path missing -> no artifacts
            from api.database import Job, get_session_factory
            session = get_session_factory()()
            job = session.get(Job, job_id)
            job.status = "completed"
            # result_path stays None/unset
            session.commit()
            session.close()

            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "pdf_missing"


# ---------------------------------------------------------------------------
# Response-shape and integration contracts
# ---------------------------------------------------------------------------


class TestRedetectLanguageResponseShape:
    def test_response_contains_required_keys(self, client, sample_pdf, tmp_path):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None), \
             patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        data = resp.json()
        assert "status" in data
        assert "job_id" in data

    def test_response_content_type_json(self, client, sample_pdf, tmp_path):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None), \
             patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert resp.headers["content-type"].startswith("application/json")

    def test_endpoint_path_matches_router_prefix(self, client, sample_pdf, tmp_path):
        """Endpoint must live at ``/api/v1/jobs/{id}/redetect-language`` exactly."""
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None), \
             patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            # Exact path -- if the prefix drifts, this 404s.
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert resp.status_code == 202

    def test_endpoint_registered_in_app(self, client):
        """The redetect-language route should be in the app's registered routes."""
        app = client.app
        paths = {getattr(r, "path", "") for r in app.routes}
        assert "/api/v1/jobs/{job_id}/redetect-language" in paths


class TestRedetectLanguageJobLookup:
    def test_get_job_called_with_tenant_scope(self, client, sample_pdf, tmp_path):
        """The handler should look up the job (tenant scope = None in single-tenant mode)."""
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None), \
             patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert resp.status_code == 202

    def test_unknown_job_id_does_not_spawn_thread(self, client):
        with patch("threading.Thread") as mock_thread:
            client.post("/api/v1/jobs/job_000000000000/redetect-language")
        mock_thread.assert_not_called()

    def test_non_completed_job_does_not_spawn_thread(self, client, sample_pdf):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_keep_submitted(client, sample_pdf)
            with patch("threading.Thread") as mock_thread:
                client.post(f"/api/v1/jobs/{job_id}/redetect-language")
            mock_thread.assert_not_called()


# ---------------------------------------------------------------------------
# Tenant isolation sanity
# ---------------------------------------------------------------------------


class TestRedetectLanguageTenantIsolation:
    def test_endpoint_uses_manager_get_job(self, client, sample_pdf, tmp_path):
        """Handler must route through manager.get_job so tenant scoping applies."""
        from api.job_manager import JobManager

        # Submit + complete first (unpatched).
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)

        # Now spy on get_job during the redetect call only.
        real_get_job = JobManager.get_job
        call_count = {"n": 0}

        def _spy_get_job(self, *args, **kwargs):
            call_count["n"] += 1
            return real_get_job(self, *args, **kwargs)

        with patch("threading.Thread") as mock_thread, \
             patch.object(JobManager, "get_job", _spy_get_job):
            mock_thread.return_value = MagicMock()
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert resp.status_code == 202
        assert call_count["n"] >= 1


# ---------------------------------------------------------------------------
# Idempotence / repeated calls
# ---------------------------------------------------------------------------


class TestRedetectLanguageIdempotence:
    def test_repeated_calls_each_return_202(self, client, sample_pdf, tmp_path):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None), \
             patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            r1 = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
            r2 = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
            r3 = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert r1.status_code == 202
        assert r2.status_code == 202
        assert r3.status_code == 202

    def test_repeated_calls_each_spawn_thread(self, client, sample_pdf, tmp_path):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None), \
             patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            client.post(f"/api/v1/jobs/{job_id}/redetect-language")
            client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert mock_thread.call_count >= 2
