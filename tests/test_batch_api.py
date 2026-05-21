"""Tests for the batch job API -- integration tests via TestClient."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.database import Batch, Job, get_engine, get_session_factory, reset_engine

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

    with (
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.job_manager.config") as mock_config,
        patch("api.job_manager.JobManager._run_pipeline"),
    ):
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 1
        mock_config.MAX_CONCURRENT_JOBS = 64
        mock_config.WEBHOOK_TIMEOUT = 30
        mock_config.WEBHOOK_MAX_RETRIES = 0
        mock_config.WEBHOOK_SECRET = ""

        from api.main import create_app

        app = create_app()
        app.state.limiter.enabled = False
        app.state.limiter.reset()
        yield TestClient(app)


@pytest.fixture()
def sample_pdf(tmp_path) -> Path:
    """Create a minimal PDF file for upload testing."""
    pdf = tmp_path / "test_doc.pdf"
    pdf.write_bytes(
        b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
    )
    return pdf


@pytest.fixture()
def sample_pdf_2(tmp_path) -> Path:
    """Create a second minimal PDF file for upload testing."""
    pdf = tmp_path / "test_doc_2.pdf"
    pdf.write_bytes(
        b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
    )
    return pdf


def _insert_batch(batch_id, status="submitted", total_jobs=0, **kwargs):
    """Insert a batch directly into the database."""
    factory = get_session_factory()
    session = factory()
    batch = Batch(
        batch_id=batch_id,
        status=status,
        total_jobs=total_jobs,
        priority=kwargs.pop("priority", "normal"),
    )
    for key, value in kwargs.items():
        setattr(batch, key, value)
    session.add(batch)
    session.commit()
    session.close()


def _insert_job(job_id, status="submitted", batch_id=None, **kwargs):
    """Insert a job directly into the database."""
    factory = get_session_factory()
    session = factory()
    job = Job(
        job_id=job_id,
        status=status,
        source_file=kwargs.pop("source_file", "test.pdf"),
        priority=kwargs.pop("priority", "normal"),
        batch_id=batch_id,
    )
    for key, value in kwargs.items():
        setattr(job, key, value)
    session.add(job)
    session.commit()
    session.close()


# ---------------------------------------------------------------------------
# Submit batch
# ---------------------------------------------------------------------------


class TestBatchSubmission:
    def test_submit_batch_with_multiple_files(self, client, sample_pdf, sample_pdf_2):
        """Submitting multiple files returns 201 with batch info."""
        with open(sample_pdf, "rb") as f1, open(sample_pdf_2, "rb") as f2:
            resp = client.post(
                "/api/v1/jobs/batch",
                files=[
                    ("files", ("test1.pdf", f1, "application/pdf")),
                    ("files", ("test2.pdf", f2, "application/pdf")),
                ],
                data={"priority": "normal"},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "submitted"
        assert "batch_id" in data
        assert data["batch_id"].startswith("batch_")
        assert data["total_jobs"] == 2
        assert len(data["jobs"]) == 2
        assert data["priority"] == "normal"
        assert "links" in data

    def test_submit_batch_with_single_file(self, client, sample_pdf):
        """A batch with one file is valid."""
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/batch",
                files=[("files", ("test.pdf", f, "application/pdf"))],
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_jobs"] == 1

    def test_submit_batch_no_files_returns_400(self, client):
        """Submitting a batch with no files or paths returns 400."""
        resp = client.post(
            "/api/v1/jobs/batch",
            data={"priority": "normal"},
        )
        assert resp.status_code == 400

    def test_submit_batch_with_priority(self, client, sample_pdf):
        """Batch respects the priority field."""
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/batch",
                files=[("files", ("test.pdf", f, "application/pdf"))],
                data={"priority": "urgent"},
            )
        assert resp.status_code == 201
        assert resp.json()["priority"] == "urgent"

    def test_submit_batch_invalid_priority_returns_422(self, client, sample_pdf):
        """Invalid priority returns 422."""
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/batch",
                files=[("files", ("test.pdf", f, "application/pdf"))],
                data={"priority": "invalid"},
            )
        assert resp.status_code == 422

    def test_submit_batch_exceeds_max_size(self, client, sample_pdf):
        """Submitting more files than MAX_BATCH_SIZE returns 413."""
        with patch("api.config.MAX_BATCH_SIZE", 1):
            with open(sample_pdf, "rb") as f1:
                content = f1.read()
            resp = client.post(
                "/api/v1/jobs/batch",
                files=[
                    ("files", ("test1.pdf", content, "application/pdf")),
                    ("files", ("test2.pdf", content, "application/pdf")),
                ],
            )
        # Batch size exceeded is "payload too large" -- must be 413, not 400.
        assert resp.status_code == 413
        body = resp.json()
        assert body["detail"]["error"] == "batch_too_large"
        assert "exceeds" in body["detail"]["message"].lower()

    def test_submit_batch_child_jobs_have_batch_id(self, client, sample_pdf):
        """Child jobs created by batch have batch_id set."""
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/batch",
                files=[("files", ("test.pdf", f, "application/pdf"))],
            )
        data = resp.json()
        batch_id = data["batch_id"]
        job_id = data["jobs"][0]["job_id"]

        # Verify via list_jobs filter
        list_resp = client.get(f"/api/v1/jobs?batch_id={batch_id}")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        assert list_data["total"] == 1
        assert list_data["jobs"][0]["job_id"] == job_id

    def test_submit_batch_with_source_paths(self, client, sample_pdf, tmp_path):
        """Batch submission accepts source_paths under SOURCE_FOLDER."""
        src = tmp_path / "source" / "batch-input.pdf"
        src.write_bytes(sample_pdf.read_bytes())

        resp = client.post(
            "/api/v1/jobs/batch",
            data={"source_paths": json.dumps([str(src)])},
        )

        assert resp.status_code == 201
        data = resp.json()
        assert data["total_jobs"] == 1
        assert data["jobs"][0]["source_file"] == "batch-input.pdf"

    def test_submit_batch_rejects_source_paths_outside_ingest_root(self, client, sample_pdf, tmp_path):
        """Batch submission rejects source_paths outside SOURCE_FOLDER."""
        outside = tmp_path / "outside" / "batch-input.pdf"
        outside.parent.mkdir(exist_ok=True)
        outside.write_bytes(sample_pdf.read_bytes())

        resp = client.post(
            "/api/v1/jobs/batch",
            data={"source_paths": json.dumps([str(outside)])},
        )

        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "path_not_allowed"


# ---------------------------------------------------------------------------
# Batch status
# ---------------------------------------------------------------------------


class TestBatchStatus:
    def test_get_batch_status(self, client, sample_pdf):
        """Get status of a submitted batch."""
        with open(sample_pdf, "rb") as f:
            submit = client.post(
                "/api/v1/jobs/batch",
                files=[("files", ("test.pdf", f, "application/pdf"))],
            )
        batch_id = submit.json()["batch_id"]

        resp = client.get(f"/api/v1/jobs/batch/{batch_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_id"] == batch_id
        assert "progress" in data
        assert "jobs" in data
        assert data["total_jobs"] == 1

    def test_get_batch_not_found(self, client):
        """Getting a non-existent batch returns 404."""
        resp = client.get("/api/v1/jobs/batch/batch_000000000000")
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "batch_not_found"

    def test_get_batch_invalid_id_format(self, client):
        """Invalid batch ID format returns 400."""
        resp = client.get("/api/v1/jobs/batch/invalid-id")
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_batch_id"

    def test_batch_progress_info(self, client):
        """Batch progress reflects child job statuses."""
        _insert_batch("batch_aaaaaaaaaaaa", status="processing", total_jobs=3)
        _insert_job("job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_bbbbbbbbbbbb", status="processing", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_cccccccccccc", status="submitted", batch_id="batch_aaaaaaaaaaaa")

        resp = client.get("/api/v1/jobs/batch/batch_aaaaaaaaaaaa")
        assert resp.status_code == 200
        progress = resp.json()["progress"]
        assert progress["completed"] == 1
        assert progress["processing"] == 1
        assert progress["submitted"] == 1


# ---------------------------------------------------------------------------
# Cancel batch
# ---------------------------------------------------------------------------


class TestBatchCancel:
    def test_cancel_batch(self, client, sample_pdf):
        """Cancelling a batch returns 200."""
        with open(sample_pdf, "rb") as f:
            submit = client.post(
                "/api/v1/jobs/batch",
                files=[("files", ("test.pdf", f, "application/pdf"))],
            )
        batch_id = submit.json()["batch_id"]

        resp = client.delete(f"/api/v1/jobs/batch/{batch_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"

    def test_cancel_batch_not_found(self, client):
        """Cancelling a non-existent batch returns 404."""
        resp = client.delete("/api/v1/jobs/batch/batch_000000000000")
        assert resp.status_code == 404

    def test_cancel_batch_invalid_id(self, client):
        """Cancelling with invalid batch ID returns 400."""
        resp = client.delete("/api/v1/jobs/batch/not-a-batch")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Retry batch
# ---------------------------------------------------------------------------


class TestBatchRetry:
    def test_retry_batch_with_failed_jobs(self, client, tmp_path):
        """Retrying a batch with failed jobs creates new child jobs."""
        _insert_batch("batch_aaaaaaaaaaaa", status="failed", total_jobs=1)

        # Create a source dir with a file for the failed job
        source_dir = tmp_path / "source" / "job_aaaaaaaaaaaa"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "test.pdf").write_bytes(
            b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
        )
        _insert_job(
            "job_aaaaaaaaaaaa",
            status="failed",
            batch_id="batch_aaaaaaaaaaaa",
        )

        with patch("api.config.SOURCE_FOLDER", str(tmp_path / "source")):
            resp = client.post("/api/v1/jobs/batch/batch_aaaaaaaaaaaa/retry")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "processing"

    def test_retry_batch_not_found(self, client):
        """Retrying a non-existent batch returns 404."""
        resp = client.post("/api/v1/jobs/batch/batch_000000000000/retry")
        assert resp.status_code == 404

    def test_retry_batch_no_failed_jobs(self, client):
        """Retrying a batch with no failed jobs returns 409."""
        _insert_batch("batch_aaaaaaaaaaaa", status="completed", total_jobs=1)
        _insert_job(
            "job_aaaaaaaaaaaa",
            status="completed",
            batch_id="batch_aaaaaaaaaaaa",
        )

        resp = client.post("/api/v1/jobs/batch/batch_aaaaaaaaaaaa/retry")
        assert resp.status_code == 409

    def test_retry_batch_invalid_id(self, client):
        """Retrying with an invalid batch ID returns 400."""
        resp = client.post("/api/v1/jobs/batch/bad-id/retry")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# List batches
# ---------------------------------------------------------------------------


class TestListBatches:
    def test_list_batches_empty(self, client):
        """Listing batches when none exist returns empty list."""
        resp = client.get("/api/v1/jobs/batch")
        assert resp.status_code == 200
        data = resp.json()
        assert data["batches"] == []
        assert data["total"] == 0
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_list_batches_returns_all(self, client):
        """Listing batches returns all batches."""
        _insert_batch("batch_aaaaaaaaaaaa", total_jobs=1)
        _insert_batch("batch_bbbbbbbbbbbb", total_jobs=2)

        resp = client.get("/api/v1/jobs/batch")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["batches"]) == 2

    def test_list_batches_status_filter(self, client):
        """Listing batches with status filter returns matching batches."""
        _insert_batch("batch_aaaaaaaaaaaa", status="submitted", total_jobs=1)
        _insert_batch("batch_bbbbbbbbbbbb", status="completed", total_jobs=2)

        resp = client.get("/api/v1/jobs/batch?status=completed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["batches"][0]["batch_id"] == "batch_bbbbbbbbbbbb"

    def test_list_batches_pagination(self, client):
        """Listing batches supports limit/offset pagination."""
        _insert_batch("batch_aaaaaaaaaaaa", total_jobs=1)
        _insert_batch("batch_bbbbbbbbbbbb", total_jobs=1)
        _insert_batch("batch_cccccccccccc", total_jobs=1)

        resp = client.get("/api/v1/jobs/batch?limit=2&offset=0")
        data = resp.json()
        assert data["total"] == 3
        assert len(data["batches"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 0

        resp = client.get("/api/v1/jobs/batch?limit=2&offset=2")
        data = resp.json()
        assert data["total"] == 3
        assert len(data["batches"]) == 1
        assert data["limit"] == 2
        assert data["offset"] == 2

    def test_list_batches_pagination_legacy(self, client):
        """Listing batches still supports legacy page/per_page."""
        _insert_batch("batch_aaaaaaaaaaaa", total_jobs=1)
        _insert_batch("batch_bbbbbbbbbbbb", total_jobs=1)
        _insert_batch("batch_cccccccccccc", total_jobs=1)

        resp = client.get("/api/v1/jobs/batch?page=1&per_page=2")
        data = resp.json()
        assert data["total"] == 3
        assert len(data["batches"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 0

        resp = client.get("/api/v1/jobs/batch?page=2&per_page=2")
        data = resp.json()
        assert data["total"] == 3
        assert len(data["batches"]) == 1
        assert data["limit"] == 2
        assert data["offset"] == 2

    def test_list_batches_includes_job_progress(self, client):
        """Each batch in the list includes progress info."""
        _insert_batch("batch_aaaaaaaaaaaa", total_jobs=2)
        _insert_job("job_aaaaaaaaaaaa", status="completed", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_bbbbbbbbbbbb", status="processing", batch_id="batch_aaaaaaaaaaaa")

        resp = client.get("/api/v1/jobs/batch")
        data = resp.json()
        assert len(data["batches"]) == 1
        batch = data["batches"][0]
        assert "progress" in batch
        assert batch["progress"]["completed"] == 1
        assert batch["progress"]["processing"] == 1
        assert "jobs" in batch
        assert len(batch["jobs"]) == 2

    def test_list_batches_page_beyond_results(self, client):
        """Requesting a page beyond results returns empty list with total."""
        _insert_batch("batch_aaaaaaaaaaaa", total_jobs=1)
        resp = client.get("/api/v1/jobs/batch?page=999")
        data = resp.json()
        assert data["total"] == 1
        assert len(data["batches"]) == 0


# ---------------------------------------------------------------------------
# List jobs with batch_id filter
# ---------------------------------------------------------------------------


class TestListJobsBatchFilter:
    def test_list_jobs_with_batch_id_filter(self, client):
        """Listing jobs with batch_id filter returns only batch members."""
        _insert_batch("batch_aaaaaaaaaaaa", total_jobs=2)
        _insert_job("job_aaaaaaaaaaaa", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_bbbbbbbbbbbb", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_cccccccccccc")  # Not in batch

        resp = client.get("/api/v1/jobs?batch_id=batch_aaaaaaaaaaaa")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        job_ids = {j["job_id"] for j in data["jobs"]}
        assert "job_aaaaaaaaaaaa" in job_ids
        assert "job_bbbbbbbbbbbb" in job_ids
        assert "job_cccccccccccc" not in job_ids

    def test_list_jobs_without_batch_filter(self, client):
        """Without batch_id filter, all jobs are returned."""
        _insert_job("job_aaaaaaaaaaaa", batch_id="batch_aaaaaaaaaaaa")
        _insert_job("job_bbbbbbbbbbbb")

        resp = client.get("/api/v1/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2


# ---------------------------------------------------------------------------
# Auth required (when API key is set)
# ---------------------------------------------------------------------------


class TestBatchAuth:
    def test_batch_requires_auth_when_key_set(self, tmp_path, sample_pdf):
        """Batch endpoints require API key when OCR_API_KEY is configured."""
        reset_engine()

    def test_batch_mutations_require_operator_role(self, tmp_path, sample_pdf):
        """Anonymous viewer mode can read batches but cannot mutate them."""
        reset_engine()
        db_file = str(tmp_path / "test_rbac.db")
        source = tmp_path / "source_rbac"
        output = tmp_path / "output_rbac"
        source.mkdir()
        output.mkdir()

        with (
            patch("api.config.DB_PATH", db_file),
            patch("api.database.DB_PATH", db_file),
            patch("api.config.SOURCE_FOLDER", str(source)),
            patch("api.config.OUTPUT_FOLDER", str(output)),
            patch("api.config.OCR_API_KEY", ""),
            patch("api.config.ALLOW_UNAUTHENTICATED", True),
            patch("api.config.ANONYMOUS_ROLE", "viewer"),
            patch("api.auth.OCR_API_KEY", ""),
            patch("api.auth.ALLOW_UNAUTHENTICATED", True),
            patch("api.auth.ANONYMOUS_ROLE", "viewer"),
            patch("api.job_manager.config") as mock_config,
            patch("api.job_manager.JobManager._run_pipeline"),
        ):
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64
            mock_config.WEBHOOK_TIMEOUT = 30
            mock_config.WEBHOOK_MAX_RETRIES = 0
            mock_config.WEBHOOK_SECRET = ""

            reset_engine()
            get_engine(db_file)

            from api.main import create_app

            app = create_app()
            app.state.limiter.enabled = False
            test_client = TestClient(app)

            with open(sample_pdf, "rb") as f:
                submit = test_client.post(
                    "/api/v1/jobs/batch",
                    files=[("files", ("test.pdf", f, "application/pdf"))],
                )
            assert submit.status_code == 403

            _insert_batch("batch_aaaaaaaaaaaa", status="submitted", total_jobs=1)
            _insert_job("job_aaaaaaaaaaaa", status="failed", batch_id="batch_aaaaaaaaaaaa")

            assert test_client.get("/api/v1/jobs/batch/batch_aaaaaaaaaaaa").status_code == 200
            assert test_client.delete("/api/v1/jobs/batch/batch_aaaaaaaaaaaa").status_code == 403
            assert test_client.post("/api/v1/jobs/batch/batch_aaaaaaaaaaaa/retry").status_code == 403

        reset_engine()
        db_file = str(tmp_path / "test_auth.db")
        source = tmp_path / "source_auth"
        output = tmp_path / "output_auth"
        source.mkdir()
        output.mkdir()

        with (
            patch("api.config.DB_PATH", db_file),
            patch("api.database.DB_PATH", db_file),
            patch("api.config.SOURCE_FOLDER", str(source)),
            patch("api.config.OUTPUT_FOLDER", str(output)),
            patch("api.auth.OCR_API_KEY", "test-key-12345"),
            patch("api.auth.ALLOW_UNAUTHENTICATED", False),
            patch("api.job_manager.config") as mock_config,
            patch("api.job_manager.JobManager._run_pipeline"),
        ):
            mock_config.SOURCE_FOLDER = str(source)
            mock_config.OUTPUT_FOLDER = str(output)
            mock_config.PIPELINE_SCRIPT = "echo"
            mock_config.PIPELINE_POLL_INTERVAL = 1
            mock_config.MAX_CONCURRENT_JOBS = 64
            mock_config.WEBHOOK_TIMEOUT = 30
            mock_config.WEBHOOK_MAX_RETRIES = 0
            mock_config.WEBHOOK_SECRET = ""

            reset_engine()
            get_engine(db_file)

            from api.main import create_app

            app = create_app()
            app.state.limiter.enabled = False
            test_client = TestClient(app)

            # No key -- should get 401
            with open(sample_pdf, "rb") as f:
                resp = test_client.post(
                    "/api/v1/jobs/batch",
                    files=[("files", ("test.pdf", f, "application/pdf"))],
                )
            assert resp.status_code == 401

            # With key -- should succeed
            with open(sample_pdf, "rb") as f:
                resp = test_client.post(
                    "/api/v1/jobs/batch",
                    files=[("files", ("test.pdf", f, "application/pdf"))],
                    headers={"X-API-Key": "test-key-12345"},
                )
            assert resp.status_code == 201

        reset_engine()


# ---------------------------------------------------------------------------
# derive_batch_status pure function
# ---------------------------------------------------------------------------


class _FakeJob:
    """Stub with a ``status`` attribute for derive_batch_status tests."""

    def __init__(self, status: str):
        self.status = status


class TestDeriveBatchStatus:
    def test_empty_returns_submitted(self):
        from api.batch_manager import derive_batch_status
        assert derive_batch_status([]) == "submitted"

    def test_all_completed(self):
        from api.batch_manager import derive_batch_status
        jobs = [_FakeJob("completed"), _FakeJob("completed"), _FakeJob("completed")]
        assert derive_batch_status(jobs) == "completed"

    def test_all_failed(self):
        from api.batch_manager import derive_batch_status
        jobs = [_FakeJob("failed"), _FakeJob("failed")]
        assert derive_batch_status(jobs) == "failed"

    def test_all_cancelled(self):
        from api.batch_manager import derive_batch_status
        jobs = [_FakeJob("cancelled"), _FakeJob("cancelled")]
        assert derive_batch_status(jobs) == "cancelled"

    def test_mixed_terminal_returns_partial_failure(self):
        from api.batch_manager import derive_batch_status
        jobs = [_FakeJob("completed"), _FakeJob("failed")]
        assert derive_batch_status(jobs) == "partial_failure"

    def test_completed_and_cancelled_returns_partial_failure(self):
        from api.batch_manager import derive_batch_status
        jobs = [_FakeJob("completed"), _FakeJob("cancelled")]
        assert derive_batch_status(jobs) == "partial_failure"

    def test_three_way_terminal_mix(self):
        from api.batch_manager import derive_batch_status
        jobs = [_FakeJob("completed"), _FakeJob("failed"), _FakeJob("cancelled")]
        assert derive_batch_status(jobs) == "partial_failure"

    def test_processing_in_progress(self):
        from api.batch_manager import derive_batch_status
        jobs = [_FakeJob("processing"), _FakeJob("completed")]
        assert derive_batch_status(jobs) == "processing"

    def test_all_submitted(self):
        from api.batch_manager import derive_batch_status
        jobs = [_FakeJob("submitted"), _FakeJob("submitted")]
        assert derive_batch_status(jobs) == "submitted"

    def test_submitted_and_completed_returns_submitted(self):
        from api.batch_manager import derive_batch_status
        jobs = [_FakeJob("submitted"), _FakeJob("completed")]
        assert derive_batch_status(jobs) == "submitted"

    def test_single_completed(self):
        from api.batch_manager import derive_batch_status
        assert derive_batch_status([_FakeJob("completed")]) == "completed"

    def test_single_failed(self):
        from api.batch_manager import derive_batch_status
        assert derive_batch_status([_FakeJob("failed")]) == "failed"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestBatchEdgeCases:
    def test_batch_id_format(self, client, sample_pdf):
        """Batch IDs follow the expected format."""
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/batch",
                files=[("files", ("test.pdf", f, "application/pdf"))],
            )
        batch_id = resp.json()["batch_id"]
        assert batch_id.startswith("batch_")
        assert len(batch_id) == len("batch_") + 12

    def test_batch_child_job_ids_are_unique(self, client, sample_pdf):
        """All child job IDs in a batch are unique."""
        content = sample_pdf.read_bytes()
        resp = client.post(
            "/api/v1/jobs/batch",
            files=[
                ("files", ("a.pdf", content, "application/pdf")),
                ("files", ("b.pdf", content, "application/pdf")),
                ("files", ("c.pdf", content, "application/pdf")),
            ],
        )
        job_ids = [j["job_id"] for j in resp.json()["jobs"]]
        assert len(job_ids) == len(set(job_ids))

    def test_batch_links_contain_batch_id(self, client, sample_pdf):
        """Submit response links reference the batch_id."""
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/batch",
                files=[("files", ("test.pdf", f, "application/pdf"))],
            )
        data = resp.json()
        assert data["batch_id"] in data["links"]["self"]

    def test_submit_then_cancel_then_status(self, client, sample_pdf):
        """Full lifecycle: submit -> cancel -> verify cancelled status."""
        with open(sample_pdf, "rb") as f:
            submit = client.post(
                "/api/v1/jobs/batch",
                files=[("files", ("test.pdf", f, "application/pdf"))],
            )
        batch_id = submit.json()["batch_id"]

        client.delete(f"/api/v1/jobs/batch/{batch_id}")

        status = client.get(f"/api/v1/jobs/batch/{batch_id}")
        assert status.json()["status"] == "cancelled"

    def test_batch_settings_preserved(self, client, sample_pdf):
        """Batch settings are persisted and returned in status."""
        with open(sample_pdf, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/batch",
                files=[("files", ("test.pdf", f, "application/pdf"))],
                data={
                    "enable_docintel": "true",
                    "docintel_mode": "layout_only",
                    "processing_timeout_minutes": "9",
                },
            )
        batch_id = resp.json()["batch_id"]

        status = client.get(f"/api/v1/jobs/batch/{batch_id}")
        settings = status.json()["settings"]
        assert settings["enable_docintel"] is True
        assert settings["docintel_mode"] == "layout_only"
        assert settings["processing_timeout_minutes"] == 9

    def test_batch_source_paths_invalid_json(self, client):
        """Invalid JSON for source_paths returns 422."""
        resp = client.post(
            "/api/v1/jobs/batch",
            data={"source_paths": "not-json-at-all"},
        )
        assert resp.status_code == 422

    def test_batch_source_paths_not_array(self, client):
        """source_paths must be a JSON array, not a scalar."""
        resp = client.post(
            "/api/v1/jobs/batch",
            data={"source_paths": '"just-a-string"'},
        )
        assert resp.status_code == 422

    def test_cancel_already_cancelled_is_idempotent(self, client, sample_pdf):
        """Cancelling an already-cancelled batch returns 200."""
        with open(sample_pdf, "rb") as f:
            submit = client.post(
                "/api/v1/jobs/batch",
                files=[("files", ("test.pdf", f, "application/pdf"))],
            )
        batch_id = submit.json()["batch_id"]

        client.delete(f"/api/v1/jobs/batch/{batch_id}")
        resp = client.delete(f"/api/v1/jobs/batch/{batch_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestBatchModels:
    def test_batch_submit_request_defaults(self):
        from api.batch_models import BatchSubmitRequest
        req = BatchSubmitRequest()
        assert req.priority == "normal"
        assert req.enable_docintel is False
        assert req.docintel_mode == "full"
        assert req.processing_timeout_minutes is None
        assert req.source_paths is None

    def test_batch_progress_info_defaults(self):
        from api.batch_models import BatchProgressInfo
        info = BatchProgressInfo()
        assert info.submitted == 0
        assert info.percent_complete == 0.0

    def test_batch_list_response_shape(self):
        from api.batch_models import BatchListResponse
        resp = BatchListResponse(batches=[], total=0, limit=50, offset=0)
        assert resp.total == 0
        assert resp.batches == []
        assert resp.limit == 50
        assert resp.offset == 0

    def test_batch_list_response_legacy_fields(self):
        from api.batch_models import BatchListResponse
        resp = BatchListResponse(batches=[], total=0, limit=20, offset=0, page=1, per_page=20)
        assert resp.page == 1
        assert resp.per_page == 20


# ---------------------------------------------------------------------------
# BatchManager unit tests
# ---------------------------------------------------------------------------


class TestBatchManagerUnit:
    def test_get_batch_returns_none_for_missing(self):
        from api.batch_manager import BatchManager
        factory = get_session_factory()
        mgr = BatchManager(factory)
        assert mgr.get_batch("batch_nonexistent0") is None

    def test_get_batch_jobs_returns_empty_for_missing(self):
        from api.batch_manager import BatchManager
        factory = get_session_factory()
        mgr = BatchManager(factory)
        assert mgr.get_batch_jobs("batch_nonexistent0") == []

    def test_list_batches_empty(self):
        from api.batch_manager import BatchManager
        factory = get_session_factory()
        mgr = BatchManager(factory)
        batches, total = mgr.list_batches()
        assert batches == []
        assert total == 0

    def test_list_batches_with_status_filter(self):
        from api.batch_manager import BatchManager
        _insert_batch("batch_aaaaaaaaaaaa", status="submitted", total_jobs=1)
        _insert_batch("batch_bbbbbbbbbbbb", status="completed", total_jobs=2)

        factory = get_session_factory()
        mgr = BatchManager(factory)
        batches, total = mgr.list_batches(status="completed")
        assert total == 1
        assert batches[0].batch_id == "batch_bbbbbbbbbbbb"

    def test_list_batches_pagination(self):
        from api.batch_manager import BatchManager
        _insert_batch("batch_aaaaaaaaaaaa", total_jobs=1)
        _insert_batch("batch_bbbbbbbbbbbb", total_jobs=1)
        _insert_batch("batch_cccccccccccc", total_jobs=1)

        factory = get_session_factory()
        mgr = BatchManager(factory)
        batches, total = mgr.list_batches(limit=2, offset=0)
        assert total == 3
        assert len(batches) == 2

    def test_list_batches_pagination_legacy(self):
        from api.batch_manager import BatchManager
        _insert_batch("batch_aaaaaaaaaaaa", total_jobs=1)
        _insert_batch("batch_bbbbbbbbbbbb", total_jobs=1)
        _insert_batch("batch_cccccccccccc", total_jobs=1)

        factory = get_session_factory()
        mgr = BatchManager(factory)
        batches, total = mgr.list_batches(page=1, per_page=2)
        assert total == 3
        assert len(batches) == 2

    def test_check_batch_completion_noop_for_missing(self):
        from api.batch_manager import BatchManager
        factory = get_session_factory()
        mgr = BatchManager(factory)
        # Should not raise
        mgr.check_batch_completion("batch_nonexistent0")
