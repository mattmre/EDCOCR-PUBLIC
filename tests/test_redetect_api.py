"""REST API tests for ``POST /api/v1/jobs/{id}/redetect-language`` (Plan A PR A5).

Mirrors the existing retry-endpoint test pattern in ``tests/test_api.py`` to
avoid flakiness around the background pipeline thread: every test patches
``JobManager._run_pipeline`` before submitting a job and forces the job into
``completed`` via direct DB writes.

Run with::

    python -m pytest tests/test_redetect_api.py -v
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from threading import Thread as _REAL_THREAD
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ocr_local.features.custody import CustodyChain

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

    from api import event_store

    # Keep durable lifecycle events inside this fixture's writable sandbox;
    # otherwise a prior app can leave the singleton pointed at the container
    # default (/app/event_store.db), which is intentionally unwritable in the
    # non-root test process.
    event_store.reset_event_store()

    with (
        patch("api.config.SOURCE_FOLDER", str(source)),
        patch("api.config.OUTPUT_FOLDER", str(output)),
        patch("api.config.EVENT_STORE_PATH", str(output / "event_store.db")),
        patch("api.config.OCR_API_KEY", ""),
        patch("api.config.ALLOW_UNAUTHENTICATED", True),
        patch("api.config.ANONYMOUS_ROLE", "operator"),
        patch("api.auth.OCR_API_KEY", ""),
        patch("api.auth.ALLOW_UNAUTHENTICATED", True),
        patch("api.auth.ANONYMOUS_ROLE", "operator"),
        patch("api.routers.jobs.Thread", _REAL_THREAD),
        patch("api.job_manager.config") as mock_config,
    ):
        mock_config.SOURCE_FOLDER = str(source)
        mock_config.OUTPUT_FOLDER = str(output)
        mock_config.PIPELINE_SCRIPT = "echo"
        mock_config.PIPELINE_POLL_INTERVAL = 1
        mock_config.MAX_CONCURRENT_JOBS = 64

        from api.main import create_app

        app = create_app()
        app.state.limiter.enabled = False
        app.state.limiter.reset()
        try:
            yield TestClient(app)
        finally:
            event_store.reset_event_store()


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
    # JobManager only serves artifacts below the configured output root.
    # Keep the synthetic result inside the same canonical root as the client
    # fixture instead of relying on the old unconstrained temp path.
    result_dir = tmp_path / "output" / job_id
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
    source_hash = job.source_hash
    session.close()

    from ocr_local.features.custody import CustodyChain

    chain = CustodyChain(
        job_id, str(sample_pdf), str(result_dir / "EXPORT" / "CUSTODY")
    )
    chain.append_event(
        "file_ingested",
        {"source_path": str(sample_pdf), "file_hash": source_hash},
    )
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


class _ImmediateThread:
    """Run the redetect target synchronously while its bound fds are live."""

    def __init__(self, *, target, daemon):
        self._target = target
        self.daemon = daemon

    def start(self):
        self._target()


class _CapturedThread:
    """Capture a queued target so tests can replace lexical paths first."""

    target = None

    def __init__(self, *, target, daemon):
        type(self).target = target
        self.daemon = daemon

    def start(self):
        return None


def _job_source_hash(job_id: str) -> str:
    from api.database import Job, get_session_factory

    session = get_session_factory()()
    try:
        job = session.get(Job, job_id)
        assert job is not None
        return str(job.source_hash)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRedetectLanguageEndpoint:
    def test_redetect_completed_job_returns_202(self, client, sample_pdf, tmp_path):
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert resp.status_code == 202, resp.text

    def test_redetect_response_body_status_queued(self, client, sample_pdf, tmp_path):
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        data = resp.json()
        assert data["status"] == "queued"
        assert data["job_id"] == job_id
        assert data["operation_id"].startswith("redetect_")

    def test_redetect_starts_background_thread(self, client, sample_pdf, tmp_path):
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread") as mock_thread,
        ):
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        # Thread was instantiated AND started
        mock_thread.assert_called()
        mock_thread_instance.start.assert_called()

    def test_redetect_thread_is_daemon(self, client, sample_pdf, tmp_path):
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread") as mock_thread,
        ):
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
        self,
        client,
        sample_pdf,
        tmp_path,
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
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        data = resp.json()
        assert "status" in data
        assert "job_id" in data

    def test_response_content_type_json(self, client, sample_pdf, tmp_path):
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert resp.headers["content-type"].startswith("application/json")

    def test_endpoint_path_matches_router_prefix(self, client, sample_pdf, tmp_path):
        """Endpoint must live at ``/api/v1/jobs/{id}/redetect-language`` exactly."""
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            # Exact path -- if the prefix drifts, this 404s.
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert resp.status_code == 202

    def test_endpoint_registered_in_app(self):
        """The jobs router should register the complete redetect path."""
        from api.routers.jobs import router

        paths = {getattr(route, "path", "") for route in router.routes}
        assert "/api/v1/jobs/{job_id}/redetect-language" in paths


class TestRedetectLanguageJobLookup:
    def test_get_job_called_with_tenant_scope(self, client, sample_pdf, tmp_path):
        """The handler should look up the job (tenant scope = None in single-tenant mode)."""
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert resp.status_code == 202

    def test_unknown_job_id_does_not_spawn_thread(self, client):
        with patch("api.routers.jobs.Thread") as mock_thread:
            client.post("/api/v1/jobs/job_000000000000/redetect-language")
        mock_thread.assert_not_called()

    def test_non_completed_job_does_not_spawn_thread(self, client, sample_pdf):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_keep_submitted(client, sample_pdf)
            with patch("api.routers.jobs.Thread") as mock_thread:
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

        with (
            patch("api.routers.jobs.Thread") as mock_thread,
            patch.object(JobManager, "get_job", _spy_get_job),
        ):
            mock_thread.return_value = MagicMock()
            resp = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert resp.status_code == 202
        assert call_count["n"] >= 1


# ---------------------------------------------------------------------------
# Idempotence / repeated calls
# ---------------------------------------------------------------------------


class TestRedetectLanguageIdempotence:
    def test_repeated_calls_each_return_202(self, client, sample_pdf, tmp_path):
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            r1 = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
            r2 = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
            r3 = client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert r1.status_code == 202
        assert r2.status_code == 202
        assert r3.status_code == 202

    def test_repeated_calls_each_spawn_thread(self, client, sample_pdf, tmp_path):
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            client.post(f"/api/v1/jobs/{job_id}/redetect-language")
            client.post(f"/api/v1/jobs/{job_id}/redetect-language")
        assert mock_thread.call_count >= 2


class TestRedetectLanguageBoundOutputs:
    def test_manifest_failure_closes_bound_job_descriptors(
        self,
        client,
        sample_pdf,
        tmp_path,
    ):
        from api.routers import jobs as jobs_router

        captured = []
        real_bind = jobs_router._bind_job_directory_under_root

        def _capture_binding(*args, **kwargs):
            binding = real_bind(*args, **kwargs)
            captured.append(binding)
            return binding

        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
        with (
            patch(
                "api.routers.jobs._bind_job_directory_under_root",
                side_effect=_capture_binding,
            ),
            patch(
                "api.routers.jobs.build_manifest_from_root_fd",
                side_effect=ValueError("bounded manifest failure"),
            ),
            patch("api.routers.jobs.Thread") as mock_thread,
        ):
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 409, response.text
        assert response.json()["detail"]["error"] == "output_changed"
        assert len(captured) == 1
        assert captured[0].job_fd == -1
        assert captured[0].storage_fd == -1
        mock_thread.assert_not_called()

    def test_conflicting_job_and_source_hash_authorities_fail_ambiguous(
        self,
        client,
        sample_pdf,
        tmp_path,
    ):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)

        from ocr_local.features.custody import CustodyChain

        custody_dir = tmp_path / "output" / job_id / "EXPORT" / "CUSTODY"
        (custody_dir / f"{job_id}.custody.jsonl").unlink()
        job_chain = CustodyChain(job_id, str(sample_pdf), str(custody_dir))
        job_chain.append_event("file_ingested", {"file_hash": "0" * 64})
        hash_chain = CustodyChain(
            "resume_hash_match", str(sample_pdf), str(custody_dir)
        )
        hash_chain.append_event(
            "file_ingested",
            {"file_hash": _job_source_hash(job_id)},
        )

        with patch("api.routers.jobs.Thread") as mock_thread:
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 409, response.text
        assert response.json()["detail"]["error"] == "custody_chain_ambiguous"
        mock_thread.assert_not_called()

    def test_pathlike_verified_custody_id_is_not_canonical_authority(
        self,
        client,
        sample_pdf,
        tmp_path,
    ):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)

        from ocr_local.features.custody import CustodyChain

        custody_dir = tmp_path / "output" / job_id / "EXPORT" / "CUSTODY"
        (custody_dir / f"{job_id}.custody.jsonl").unlink()
        pathlike = CustodyChain("../evil", str(sample_pdf))
        pathlike.append_event(
            "file_ingested",
            {"file_hash": _job_source_hash(job_id)},
        )
        (custody_dir / "evil.custody.jsonl").write_text(
            "\n".join(json.dumps(event) for event in pathlike.events) + "\n",
            encoding="utf-8",
        )

        with patch("api.routers.jobs.Thread") as mock_thread:
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 409, response.text
        assert response.json()["detail"]["error"] == "custody_chain_invalid"
        mock_thread.assert_not_called()

    def test_appends_only_to_canonical_existing_chain(
        self,
        client,
        sample_pdf,
        tmp_path,
    ):
        """A pipeline resume id wins; no source-hash-prefix fork is created."""
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)

        from ocr_local.features.custody import CustodyChain

        result_dir = tmp_path / "output" / job_id
        custody_dir = result_dir / "EXPORT" / "CUSTODY"
        source_hash = _job_source_hash(job_id)

        # Replace the API-style seed chain with a production-style resume id.
        (custody_dir / f"{job_id}.custody.jsonl").unlink()
        canonical_id = "resume_pipeline_document"
        canonical = CustodyChain(canonical_id, str(sample_pdf), str(custody_dir))
        canonical.append_event("file_ingested", {"file_hash": source_hash})

        legacy_prefix_id = source_hash[:16]
        legacy_prefix = CustodyChain(
            legacy_prefix_id,
            str(sample_pdf),
            str(custody_dir),
        )
        legacy_prefix.append_event("file_ingested", {"file_hash": "0" * 64})
        language_dir = result_dir / "EXPORT" / "LANGUAGE"
        language_dir.mkdir(parents=True, exist_ok=True)
        target = language_dir / "test.language.json"
        target.write_bytes(b"OLD SIDECAR")

        observed = {}

        def _fake_redetect(**kwargs):
            observed.update(kwargs)
            return {
                "status": "ok",
                "sidecar_filename": "test.language.json",
                "sidecar_bytes": b'{"new": true}',
                "primary_language": "en",
                "page_count": 1,
            }

        with (
            patch("api.routers.jobs.Thread", _ImmediateThread),
            patch(
                "ocr_local.features.language_detection.redetect_document",
                side_effect=_fake_redetect,
            ),
        ):
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 202, response.text
        assert observed["document_id"] == canonical_id
        assert observed["write_sidecar"] is False
        assert target.read_bytes() == b'{"new": true}'
        updated = CustodyChain.load_from_file(canonical._filepath)
        assert len(updated.events) == 3
        assert [event["event_type"] for event in updated.events[-2:]] == [
            "LANGUAGE_REDETECT_PREPARED",
            "LANGUAGE_REDETECTED",
        ]
        assert (
            updated.events[-1]["data"]["artifact_sha256"]
            == hashlib.sha256(target.read_bytes()).hexdigest()
        )
        assert updated.events[-1]["prev_hash"] == updated.events[-2]["hash"]
        assert list((result_dir / ".redetect").iterdir()) == []
        assert len(CustodyChain.load_from_file(legacy_prefix._filepath).events) == 1
        assert {path.name for path in custody_dir.glob("*.custody.jsonl")} == {
            f"{canonical_id}.custody.jsonl",
            f"{legacy_prefix_id}.custody.jsonl",
        }

    @pytest.mark.parametrize(
        ("document_id", "event_type", "data", "expected_error"),
        [
            (
                "unattested_resume",
                "file_ingested",
                {"source_path": "document.pdf"},
                "custody_chain_foreign",
            ),
            (
                None,
                "file_ingested",
                {"file_hash": "b" * 64},
                "custody_chain_foreign",
            ),
            (
                None,
                "file_ingested",
                {"file_hash": "not-a-sha256"},
                "custody_chain_invalid",
            ),
            (
                "late_hash_only",
                "assembly_complete",
                {"source_hash": "JOB_HASH"},
                "custody_chain_foreign",
            ),
        ],
    )
    def test_redetect_refuses_noncanonical_custody_authority(
        self,
        client,
        sample_pdf,
        tmp_path,
        document_id,
        event_type,
        data,
        expected_error,
    ):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
        custody_dir = tmp_path / "output" / job_id / "EXPORT" / "CUSTODY"
        for path in custody_dir.glob("*.custody.jsonl"):
            path.unlink()
        chain = CustodyChain(document_id or job_id, str(sample_pdf), str(custody_dir))
        event_data = dict(data)
        if event_data.get("source_hash") == "JOB_HASH":
            event_data["source_hash"] = _job_source_hash(job_id)
        chain.append_event(event_type, event_data)

        with patch("api.routers.jobs.Thread") as mock_thread:
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 409, response.text
        assert response.json()["detail"]["error"] == expected_error
        mock_thread.assert_not_called()

    def test_queued_custody_file_swap_cannot_redirect_append(
        self,
        client,
        sample_pdf,
        tmp_path,
        caplog,
    ):
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread", _CapturedThread),
        ):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 202, response.text
        custody_dir = tmp_path / "output" / job_id / "EXPORT" / "CUSTODY"
        selected = custody_dir / f"{job_id}.custody.jsonl"
        authorized = custody_dir / "authorized.custody.jsonl"
        selected.rename(authorized)
        replacement = CustodyChain(job_id, "foreign.pdf", str(custody_dir))
        replacement.append_event("file_ingested", {"file_hash": "b" * 64})

        with patch(
            "ocr_local.features.language_detection.redetect_document",
            return_value={
                "status": "ok",
                "primary_language": "en",
                "page_count": 1,
                "sidecar_filename": "test.language.json",
                "sidecar_bytes": b"{}",
            },
        ) as mock_redetect:
            _CapturedThread.target()

        mock_redetect.assert_called_once()
        assert len(CustodyChain.load_from_file(authorized).events) == 1
        assert len(CustodyChain.load_from_file(replacement._filepath).events) == 1
        language_dir = tmp_path / "output" / job_id / "EXPORT" / "LANGUAGE"
        assert list(language_dir.iterdir()) == []
        assert "canonical custody inode changed" in caplog.text

    def test_custody_tip_change_leaves_existing_sidecar_untouched(
        self,
        client,
        sample_pdf,
        tmp_path,
        caplog,
    ):
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread", _CapturedThread),
        ):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            language_dir = tmp_path / "output" / job_id / "EXPORT" / "LANGUAGE"
            language_dir.mkdir(parents=True, exist_ok=True)
            target = language_dir / "test.language.json"
            target.write_bytes(b"PRIOR SIDECAR")
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        custody_dir = tmp_path / "output" / job_id / "EXPORT" / "CUSTODY"
        changed = CustodyChain(job_id, str(sample_pdf), str(custody_dir))
        changed.append_event("processing_resumed", {"reason": "concurrent"})
        with patch(
            "ocr_local.features.language_detection.redetect_document",
            return_value={
                "status": "ok",
                "sidecar_filename": "test.language.json",
                "sidecar_bytes": b"UNTRACKED SIDECAR",
            },
        ):
            _CapturedThread.target()

        assert response.status_code == 202, response.text
        assert target.read_bytes() == b"PRIOR SIDECAR"
        assert [path.name for path in language_dir.iterdir()] == [target.name]
        loaded = CustodyChain.load_from_file(changed._filepath)
        assert len(loaded.events) == 2
        assert all(
            event["event_type"] != "LANGUAGE_REDETECT_PREPARED"
            for event in loaded.events
        )
        assert "inode changed after selection" in caplog.text

    def test_publish_failure_restores_prior_and_retains_prepared_evidence(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
    ):
        from api.routers import jobs as jobs_router

        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
        result_dir = tmp_path / "output" / job_id
        language_dir = result_dir / "EXPORT" / "LANGUAGE"
        language_dir.mkdir(parents=True, exist_ok=True)
        target = language_dir / "test.language.json"
        target.write_bytes(b"PRIOR SIDECAR")
        real_link = os.link

        def fail_stage_publish(src, dst, *args, **kwargs):
            if src == jobs_router._REDETECT_STAGE_NAME:
                raise OSError("injected publish failure")
            return real_link(src, dst, *args, **kwargs)

        monkeypatch.setattr(jobs_router.os, "link", fail_stage_publish)
        with (
            patch("api.routers.jobs.Thread", _ImmediateThread),
            patch(
                "ocr_local.features.language_detection.redetect_document",
                return_value={
                    "status": "ok",
                    "sidecar_filename": "test.language.json",
                    "sidecar_bytes": b"NEW SIDECAR",
                },
            ),
        ):
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 202, response.text
        assert target.read_bytes() == b"PRIOR SIDECAR"
        custody_path = result_dir / "EXPORT" / "CUSTODY" / f"{job_id}.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(custody_path))
        assert [event["event_type"] for event in loaded.events] == [
            "file_ingested",
            "LANGUAGE_REDETECT_PREPARED",
        ]
        assert list((result_dir / ".redetect").iterdir()) == []

    @pytest.mark.parametrize("with_prior", [False, True])
    @pytest.mark.parametrize(
        "signal_after_real_append",
        [False, True],
        ids=["before-real-commit-append", "after-real-commit-append"],
    )
    def test_commit_append_control_flow_uses_transaction_outcome(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
        with_prior,
        signal_after_real_append,
    ):
        """Only a proven rollback after the PREPARED savepoint permits restore."""
        import custody as custody_module

        class CommitAppendSignal(BaseException):
            pass

        signal = CommitAppendSignal("injected commit append control flow")
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread", _CapturedThread),
        ):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            result_dir = tmp_path / "output" / job_id
            language_dir = result_dir / "EXPORT" / "LANGUAGE"
            language_dir.mkdir(parents=True, exist_ok=True)
            target = language_dir / "test.language.json"
            if with_prior:
                target.write_bytes(b"PRIOR SIDECAR")
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        real_append = custody_module._BoundCustodyTransaction.append_event

        def inject_commit_signal(transaction, event_type, data=None):
            if event_type == "LANGUAGE_REDETECTED" and not signal_after_real_append:
                raise signal
            event = real_append(transaction, event_type, data)
            if event_type == "LANGUAGE_REDETECTED" and signal_after_real_append:
                raise signal
            return event

        monkeypatch.setattr(
            custody_module._BoundCustodyTransaction,
            "append_event",
            inject_commit_signal,
        )
        with patch(
            "ocr_local.features.language_detection.redetect_document",
            return_value={
                "status": "ok",
                "sidecar_filename": "test.language.json",
                "sidecar_bytes": b"UNCOMMITTED SIDECAR",
            },
        ):
            with pytest.raises(CommitAppendSignal) as raised:
                _CapturedThread.target()

        assert raised.value is signal
        assert response.status_code == 202, response.text
        if with_prior:
            assert target.read_bytes() == b"PRIOR SIDECAR"
        else:
            assert not target.exists()
        custody_path = result_dir / "EXPORT" / "CUSTODY" / f"{job_id}.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(custody_path))
        assert [event["event_type"] for event in loaded.events] == [
            "file_ingested",
            "LANGUAGE_REDETECT_PREPARED",
        ]
        assert (
            loaded.events[-1]["data"]["operation_id"] == response.json()["operation_id"]
        )
        assert list((result_dir / ".redetect").iterdir()) == []

    @pytest.mark.parametrize("with_prior", [False, True])
    @pytest.mark.parametrize(
        "failure_point",
        ["unlock", "evidence_fd_close"],
    )
    def test_post_boundary_control_flow_preserves_committed_redetect(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
        with_prior,
        failure_point,
    ):
        """Cleanup signals after the bound commit cannot authorize restoration."""
        import custody as custody_module

        class PostBoundarySignal(BaseException):
            pass

        signal = PostBoundarySignal(f"injected after real {failure_point}")
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread", _CapturedThread),
        ):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            result_dir = tmp_path / "output" / job_id
            language_dir = result_dir / "EXPORT" / "LANGUAGE"
            language_dir.mkdir(parents=True, exist_ok=True)
            target = language_dir / "test.language.json"
            if with_prior:
                target.write_bytes(b"PRIOR SIDECAR")
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        injected = False
        if failure_point == "unlock":
            real_unlock = custody_module._unlock_file_strict

            def finish_unlock_then_signal(lock_file):
                nonlocal injected
                real_unlock(lock_file)
                if not injected:
                    injected = True
                    raise signal

            cleanup_patch = patch.object(
                custody_module,
                "_unlock_file_strict",
                side_effect=finish_unlock_then_signal,
            )
        else:
            real_close = os.close
            custody_leaf = f"/{job_id}.custody.jsonl"

            def finish_evidence_close_then_signal(fd):
                nonlocal injected
                try:
                    descriptor_path = os.readlink(f"/proc/self/fd/{fd}")
                except OSError:
                    descriptor_path = ""
                real_close(fd)
                if not injected and descriptor_path.endswith(custody_leaf):
                    injected = True
                    raise signal

            cleanup_patch = patch.object(
                custody_module.os,
                "close",
                side_effect=finish_evidence_close_then_signal,
            )

        with (
            cleanup_patch,
            patch(
                "ocr_local.features.language_detection.redetect_document",
                return_value={
                    "status": "ok",
                    "sidecar_filename": "test.language.json",
                    "sidecar_bytes": b"COMMITTED SIDECAR",
                },
            ),
        ):
            with pytest.raises(PostBoundarySignal) as raised:
                _CapturedThread.target()

        assert injected
        assert raised.value is signal
        assert response.status_code == 202, response.text
        assert target.read_bytes() == b"COMMITTED SIDECAR"
        custody_path = result_dir / "EXPORT" / "CUSTODY" / f"{job_id}.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(custody_path))
        assert [event["event_type"] for event in loaded.events] == [
            "file_ingested",
            "LANGUAGE_REDETECT_PREPARED",
            "LANGUAGE_REDETECTED",
        ]
        operation_dir = result_dir / ".redetect" / response.json()["operation_id"]
        assert operation_dir.is_dir()
        if with_prior:
            assert (operation_dir / "prior.language.json").read_bytes() == (
                b"PRIOR SIDECAR"
            )

    @pytest.mark.parametrize("with_prior", [False, True])
    def test_foreign_tail_rollback_refusal_preserves_recovery_state(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
        caplog,
        with_prior,
    ):
        """An unproven post-COMMITTED tail is rollback_uncertain, never restorable."""
        import custody as custody_module

        class ForeignTailSignal(BaseException):
            pass

        signal = ForeignTailSignal("injected after foreign custody tail")
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread", _CapturedThread),
        ):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            result_dir = tmp_path / "output" / job_id
            language_dir = result_dir / "EXPORT" / "LANGUAGE"
            language_dir.mkdir(parents=True, exist_ok=True)
            target = language_dir / "test.language.json"
            if with_prior:
                target.write_bytes(b"PRIOR SIDECAR")
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        real_append = custody_module._BoundCustodyTransaction.append_event

        def append_foreign_tail_then_signal(transaction, event_type, data=None):
            event = real_append(transaction, event_type, data)
            if event_type != "LANGUAGE_REDETECTED":
                return event
            _foreign_event, encoded = custody_module._generated_event(
                document_id=transaction.document_id,
                prev_hash=event["hash"],
                event_type="processing_resumed",
                data={"reason": "foreign-tail-control"},
            )
            view = memoryview(encoded)
            while view:
                written = os.write(transaction.custody_fd, view)
                assert written > 0
                view = view[written:]
            os.fsync(transaction.custody_fd)
            raise signal

        monkeypatch.setattr(
            custody_module._BoundCustodyTransaction,
            "append_event",
            append_foreign_tail_then_signal,
        )
        with patch(
            "ocr_local.features.language_detection.redetect_document",
            return_value={
                "status": "ok",
                "sidecar_filename": "test.language.json",
                "sidecar_bytes": b"RECOVERY SIDECAR",
            },
        ):
            with pytest.raises(ForeignTailSignal) as raised:
                _CapturedThread.target()

        assert raised.value is signal
        if hasattr(signal, "add_note"):
            assert any(
                "outcome=rollback_uncertain" in note for note in signal.__notes__
            )
        assert "custody outcome rollback_uncertain" in caplog.text
        assert response.status_code == 202, response.text
        assert target.read_bytes() == b"RECOVERY SIDECAR"
        custody_path = result_dir / "EXPORT" / "CUSTODY" / f"{job_id}.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(custody_path))
        assert [event["event_type"] for event in loaded.events] == [
            "file_ingested",
            "LANGUAGE_REDETECT_PREPARED",
            "LANGUAGE_REDETECTED",
            "processing_resumed",
        ]
        assert loaded.verify_chain()[0] is True
        operation_dir = result_dir / ".redetect" / response.json()["operation_id"]
        assert operation_dir.is_dir()
        if with_prior:
            assert (operation_dir / "prior.language.json").read_bytes() == (
                b"PRIOR SIDECAR"
            )

    def test_unjournaled_committed_bytes_do_not_authorize_restore(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        """Durable COMMITTED bytes missing from the journal are not restorable."""
        import custody as custody_module

        class UnjournaledSignal(BaseException):
            pass

        signal = UnjournaledSignal("injected after unjournaled COMMITTED bytes")
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread", _CapturedThread),
        ):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            result_dir = tmp_path / "output" / job_id
            language_dir = result_dir / "EXPORT" / "LANGUAGE"
            language_dir.mkdir(parents=True, exist_ok=True)
            target = language_dir / "test.language.json"
            target.write_bytes(b"PRIOR SIDECAR")
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        real_append = custody_module._BoundCustodyTransaction.append_event

        def write_unjournaled_committed(transaction, event_type, data=None):
            if event_type != "LANGUAGE_REDETECTED":
                return real_append(transaction, event_type, data)
            _event, encoded = custody_module._generated_event(
                document_id=transaction.document_id,
                prev_hash=transaction.chain_head,
                event_type=event_type,
                data=data,
            )
            view = memoryview(encoded)
            while view:
                written = os.write(transaction.custody_fd, view)
                assert written > 0
                view = view[written:]
            os.fsync(transaction.custody_fd)
            raise signal

        monkeypatch.setattr(
            custody_module._BoundCustodyTransaction,
            "append_event",
            write_unjournaled_committed,
        )
        with patch(
            "ocr_local.features.language_detection.redetect_document",
            return_value={
                "status": "ok",
                "sidecar_filename": "test.language.json",
                "sidecar_bytes": b"RECOVERY SIDECAR",
            },
        ):
            with pytest.raises(UnjournaledSignal) as raised:
                _CapturedThread.target()

        assert raised.value is signal
        assert "custody outcome rollback_uncertain" in caplog.text
        assert target.read_bytes() == b"RECOVERY SIDECAR"
        custody_path = result_dir / "EXPORT" / "CUSTODY" / f"{job_id}.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(custody_path))
        assert [event["event_type"] for event in loaded.events][-1] == (
            "LANGUAGE_REDETECTED"
        )
        assert (result_dir / ".redetect" / response.json()["operation_id"]).is_dir()

    def test_stage_unlink_failure_quarantines_publish_and_restores_prior(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
    ):
        """The temporary two-name link state never exposes an uncommitted sidecar."""
        from api.routers import jobs as jobs_router

        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
        result_dir = tmp_path / "output" / job_id
        language_dir = result_dir / "EXPORT" / "LANGUAGE"
        language_dir.mkdir(parents=True, exist_ok=True)
        target = language_dir / "test.language.json"
        target.write_bytes(b"PRIOR SIDECAR")
        real_unlink = os.unlink
        injected = False

        def fail_first_stage_unlink(path, *args, **kwargs):
            nonlocal injected
            if path == jobs_router._REDETECT_STAGE_NAME and not injected:
                injected = True
                raise OSError("injected stage unlink failure")
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(jobs_router.os, "unlink", fail_first_stage_unlink)
        with (
            patch("api.routers.jobs.Thread", _ImmediateThread),
            patch(
                "ocr_local.features.language_detection.redetect_document",
                return_value={
                    "status": "ok",
                    "sidecar_filename": "test.language.json",
                    "sidecar_bytes": b"UNCOMMITTED SIDECAR",
                },
            ),
        ):
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 202, response.text
        assert injected
        assert target.read_bytes() == b"PRIOR SIDECAR"
        assert [path.name for path in language_dir.iterdir()] == [target.name]
        custody_path = result_dir / "EXPORT" / "CUSTODY" / f"{job_id}.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(custody_path))
        assert [event["event_type"] for event in loaded.events] == [
            "file_ingested",
            "LANGUAGE_REDETECT_PREPARED",
        ]
        assert list((result_dir / ".redetect").iterdir()) == []

    def test_restore_refuses_in_place_rewrite_of_published_inode(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        """Restore never deletes bytes written concurrently into the published inode."""
        import custody as custody_module

        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
        result_dir = tmp_path / "output" / job_id
        language_dir = result_dir / "EXPORT" / "LANGUAGE"
        language_dir.mkdir(parents=True, exist_ok=True)
        target = language_dir / "test.language.json"
        target.write_bytes(b"PRIOR SIDECAR")
        real_append = custody_module._BoundCustodyTransaction.append_event

        def rewrite_then_fail_commit(transaction, event_type, data=None):
            if event_type == "LANGUAGE_REDETECTED":
                target.write_bytes(b"CONCURRENT SIDECAR")
                raise OSError("injected commit failure after concurrent rewrite")
            return real_append(transaction, event_type, data)

        monkeypatch.setattr(
            custody_module._BoundCustodyTransaction,
            "append_event",
            rewrite_then_fail_commit,
        )
        with (
            patch("api.routers.jobs.Thread", _ImmediateThread),
            patch(
                "ocr_local.features.language_detection.redetect_document",
                return_value={
                    "status": "ok",
                    "sidecar_filename": "test.language.json",
                    "sidecar_bytes": b"NEW SIDECAR",
                },
            ),
        ):
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 202, response.text
        assert target.read_bytes() == b"CONCURRENT SIDECAR"
        custody_path = result_dir / "EXPORT" / "CUSTODY" / f"{job_id}.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(custody_path))
        assert [event["event_type"] for event in loaded.events] == [
            "file_ingested",
            "LANGUAGE_REDETECT_PREPARED",
        ]
        operation_dir = result_dir / ".redetect" / response.json()["operation_id"]
        assert (operation_dir / "prior.language.json").exists()
        assert "restore failed" in caplog.text

    def test_final_artifact_authority_failure_rolls_back_committed_event(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
    ):
        """A target rewrite after COMMITTED append is still pre-commit authority loss."""
        import custody as custody_module

        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
        result_dir = tmp_path / "output" / job_id
        language_dir = result_dir / "EXPORT" / "LANGUAGE"
        language_dir.mkdir(parents=True, exist_ok=True)
        target = language_dir / "test.language.json"
        target.write_bytes(b"PRIOR SIDECAR")
        real_append = custody_module._BoundCustodyTransaction.append_event
        injected = False

        def rewrite_after_committed_append(transaction, event_type, data=None):
            nonlocal injected
            event = real_append(transaction, event_type, data)
            if event_type == "LANGUAGE_REDETECTED":
                target.write_bytes(b"CONCURRENT AFTER COMMIT APPEND")
                injected = True
            return event

        monkeypatch.setattr(
            custody_module._BoundCustodyTransaction,
            "append_event",
            rewrite_after_committed_append,
        )
        with (
            patch("api.routers.jobs.Thread", _ImmediateThread),
            patch(
                "ocr_local.features.language_detection.redetect_document",
                return_value={
                    "status": "ok",
                    "sidecar_filename": "test.language.json",
                    "sidecar_bytes": b"NEW SIDECAR",
                },
            ),
        ):
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 202, response.text
        assert injected
        assert target.read_bytes() == b"CONCURRENT AFTER COMMIT APPEND"
        custody_path = result_dir / "EXPORT" / "CUSTODY" / f"{job_id}.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(custody_path))
        assert [event["event_type"] for event in loaded.events] == [
            "file_ingested",
            "LANGUAGE_REDETECT_PREPARED",
        ]

    @pytest.mark.parametrize("with_prior", [False, True])
    def test_baseexception_final_authority_failure_restores_before_reraise(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
        with_prior,
    ):
        """Control-flow exceptions use the same precommit restore path."""
        import custody as custody_module

        class FinalAuthoritySignal(BaseException):
            pass

        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread", _CapturedThread),
        ):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            result_dir = tmp_path / "output" / job_id
            language_dir = result_dir / "EXPORT" / "LANGUAGE"
            language_dir.mkdir(parents=True, exist_ok=True)
            target = language_dir / "test.language.json"
            if with_prior:
                target.write_bytes(b"PRIOR SIDECAR")
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        real_transaction = custody_module.open_bound_chain_transaction

        @contextmanager
        def inject_final_signal(**kwargs):
            original_check = kwargs["final_authority_check"]

            def signal_after_valid_authority() -> None:
                original_check()
                raise FinalAuthoritySignal("injected final authority signal")

            kwargs["final_authority_check"] = signal_after_valid_authority
            with real_transaction(**kwargs) as transaction:
                yield transaction

        monkeypatch.setattr(
            custody_module,
            "open_bound_chain_transaction",
            inject_final_signal,
        )
        with patch(
            "ocr_local.features.language_detection.redetect_document",
            return_value={
                "status": "ok",
                "sidecar_filename": "test.language.json",
                "sidecar_bytes": b"UNCOMMITTED SIDECAR",
            },
        ):
            with pytest.raises(FinalAuthoritySignal):
                _CapturedThread.target()

        assert response.status_code == 202, response.text
        if with_prior:
            assert target.read_bytes() == b"PRIOR SIDECAR"
        else:
            assert not target.exists()
        custody_path = result_dir / "EXPORT" / "CUSTODY" / f"{job_id}.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(custody_path))
        assert [event["event_type"] for event in loaded.events] == [
            "file_ingested",
            "LANGUAGE_REDETECT_PREPARED",
        ]
        assert list((result_dir / ".redetect").iterdir()) == []

    def test_commit_failure_restores_prior_without_committed_event(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
    ):
        import custody as custody_module

        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
        result_dir = tmp_path / "output" / job_id
        language_dir = result_dir / "EXPORT" / "LANGUAGE"
        language_dir.mkdir(parents=True, exist_ok=True)
        target = language_dir / "test.language.json"
        target.write_bytes(b"PRIOR SIDECAR")
        real_append = custody_module._BoundCustodyTransaction.append_event

        def fail_commit(transaction, event_type, data=None):
            if event_type == "LANGUAGE_REDETECTED":
                raise OSError("injected custody commit failure")
            return real_append(transaction, event_type, data)

        monkeypatch.setattr(
            custody_module._BoundCustodyTransaction,
            "append_event",
            fail_commit,
        )
        with (
            patch("api.routers.jobs.Thread", _ImmediateThread),
            patch(
                "ocr_local.features.language_detection.redetect_document",
                return_value={
                    "status": "ok",
                    "sidecar_filename": "test.language.json",
                    "sidecar_bytes": b"NEW SIDECAR",
                },
            ),
        ):
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 202, response.text
        assert target.read_bytes() == b"PRIOR SIDECAR"
        custody_path = result_dir / "EXPORT" / "CUSTODY" / f"{job_id}.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(custody_path))
        assert [event["event_type"] for event in loaded.events] == [
            "file_ingested",
            "LANGUAGE_REDETECT_PREPARED",
        ]
        assert list((result_dir / ".redetect").iterdir()) == []

    def test_cleanup_failure_after_commit_preserves_published_artifact(
        self,
        client,
        sample_pdf,
        tmp_path,
        caplog,
    ):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
        result_dir = tmp_path / "output" / job_id
        baseline_fds = len(os.listdir("/proc/self/fd"))
        with (
            patch("api.routers.jobs.Thread", _ImmediateThread),
            patch(
                "ocr_local.features.language_detection.redetect_document",
                return_value={
                    "status": "ok",
                    "sidecar_filename": "test.language.json",
                    "sidecar_bytes": b"COMMITTED SIDECAR",
                    "primary_language": "en",
                    "page_count": 1,
                },
            ),
            patch(
                "api.routers.jobs._cleanup_redetect_operation",
                side_effect=OSError("injected cleanup failure"),
            ),
        ):
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 202, response.text
        target = result_dir / "EXPORT" / "LANGUAGE" / "test.language.json"
        assert target.read_bytes() == b"COMMITTED SIDECAR"
        custody_path = result_dir / "EXPORT" / "CUSTODY" / f"{job_id}.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(custody_path))
        committed = loaded.events[-1]
        assert committed["event_type"] == "LANGUAGE_REDETECTED"
        assert (
            committed["data"]["artifact_sha256"]
            == hashlib.sha256(target.read_bytes()).hexdigest()
        )
        assert "staging cleanup failed" in caplog.text
        assert len(os.listdir("/proc/self/fd")) <= baseline_fds

        outputs = client.get(f"/api/v1/jobs/{job_id}/outputs")
        assert outputs.status_code == 200, outputs.text
        assert all(
            ".redetect" not in artifact["relative_path"]
            for artifact in outputs.json()["artifacts"]
        )

    def test_post_commit_validation_failure_never_restores_prior(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        from api.routers import jobs as jobs_router

        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
        result_dir = tmp_path / "output" / job_id
        language_dir = result_dir / "EXPORT" / "LANGUAGE"
        language_dir.mkdir(parents=True, exist_ok=True)
        target = language_dir / "test.language.json"
        target.write_bytes(b"PRIOR SIDECAR")
        real_assert = jobs_router._assert_redetect_name_snapshot
        checks = 0

        def fail_second_snapshot_check(*args, **kwargs):
            nonlocal checks
            checks += 1
            if checks == 2:
                raise OSError("injected post-commit validation failure")
            return real_assert(*args, **kwargs)

        monkeypatch.setattr(
            jobs_router,
            "_assert_redetect_name_snapshot",
            fail_second_snapshot_check,
        )
        with (
            patch("api.routers.jobs.Thread", _ImmediateThread),
            patch(
                "ocr_local.features.language_detection.redetect_document",
                return_value={
                    "status": "ok",
                    "sidecar_filename": "test.language.json",
                    "sidecar_bytes": b"COMMITTED SIDECAR",
                },
            ),
        ):
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 202, response.text
        assert checks == 2
        assert target.read_bytes() == b"COMMITTED SIDECAR"
        custody_path = result_dir / "EXPORT" / "CUSTODY" / f"{job_id}.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(custody_path))
        assert loaded.events[-1]["event_type"] == "LANGUAGE_REDETECTED"
        assert (
            loaded.events[-1]["data"]["artifact_sha256"]
            == hashlib.sha256(target.read_bytes()).hexdigest()
        )
        assert "custody outcome committed" in caplog.text
        operation_dir = result_dir / ".redetect" / response.json()["operation_id"]
        assert (operation_dir / "prior.language.json").read_bytes() == b"PRIOR SIDECAR"

    def test_custody_context_exit_failure_rolls_back_and_restores_prior(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
    ):
        import custody as custody_module

        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
        result_dir = tmp_path / "output" / job_id
        language_dir = result_dir / "EXPORT" / "LANGUAGE"
        language_dir.mkdir(parents=True, exist_ok=True)
        target = language_dir / "test.language.json"
        target.write_bytes(b"PRIOR SIDECAR")
        real_assert = custody_module._assert_chain_name_inode
        checks = 0

        def fail_transaction_exit(*args, **kwargs):
            nonlocal checks
            checks += 1
            if checks == 10:
                raise custody_module._CustodyAuthorityError(
                    "injected context-exit authority failure"
                )
            return real_assert(*args, **kwargs)

        monkeypatch.setattr(
            custody_module,
            "_assert_chain_name_inode",
            fail_transaction_exit,
        )
        with (
            patch("api.routers.jobs.Thread", _ImmediateThread),
            patch(
                "ocr_local.features.language_detection.redetect_document",
                return_value={
                    "status": "ok",
                    "sidecar_filename": "test.language.json",
                    "sidecar_bytes": b"UNCOMMITTED SIDECAR",
                },
            ),
        ):
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 202, response.text
        assert checks == 10
        assert target.read_bytes() == b"PRIOR SIDECAR"
        custody_path = result_dir / "EXPORT" / "CUSTODY" / f"{job_id}.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(custody_path))
        assert [event["event_type"] for event in loaded.events] == [
            "file_ingested",
            "LANGUAGE_REDETECT_PREPARED",
        ]
        assert list((result_dir / ".redetect").iterdir()) == []

    @pytest.mark.parametrize("with_prior", [False, True])
    def test_publish_never_clobbers_target_appearing_in_link_gap(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
        with_prior,
    ):
        from api.routers import jobs as jobs_router

        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
        result_dir = tmp_path / "output" / job_id
        language_dir = result_dir / "EXPORT" / "LANGUAGE"
        language_dir.mkdir(parents=True, exist_ok=True)
        target = language_dir / "test.language.json"
        if with_prior:
            target.write_bytes(b"PRIOR SIDECAR")
        real_link = os.link
        injected = False

        def inject_concurrent_target(src, dst, *args, **kwargs):
            nonlocal injected
            if src == jobs_router._REDETECT_STAGE_NAME and not injected:
                injected = True
                concurrent_fd = os.open(
                    dst,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=kwargs["dst_dir_fd"],
                )
                try:
                    os.write(concurrent_fd, b"CONCURRENT SIDECAR")
                    os.fsync(concurrent_fd)
                finally:
                    os.close(concurrent_fd)
            return real_link(src, dst, *args, **kwargs)

        monkeypatch.setattr(jobs_router.os, "link", inject_concurrent_target)
        with (
            patch("api.routers.jobs.Thread", _ImmediateThread),
            patch(
                "ocr_local.features.language_detection.redetect_document",
                return_value={
                    "status": "ok",
                    "sidecar_filename": "test.language.json",
                    "sidecar_bytes": b"REDETECT SIDECAR",
                },
            ),
        ):
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 202, response.text
        assert injected is True
        assert target.read_bytes() == b"CONCURRENT SIDECAR"
        custody_path = result_dir / "EXPORT" / "CUSTODY" / f"{job_id}.custody.jsonl"
        loaded = CustodyChain.load_from_file(str(custody_path))
        assert [event["event_type"] for event in loaded.events] == [
            "file_ingested",
            "LANGUAGE_REDETECT_PREPARED",
        ]
        operation_dir = result_dir / ".redetect" / response.json()["operation_id"]
        if with_prior:
            assert (operation_dir / "prior.language.json").read_bytes() == (
                b"PRIOR SIDECAR"
            )
        else:
            assert not operation_dir.exists()

    def test_redetect_custody_directory_bound_is_fail_closed(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
    ):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
        monkeypatch.setattr(
            "api.routers.jobs._REDETECT_MAX_CUSTODY_DIRECTORY_ENTRIES",
            0,
        )

        with patch("api.routers.jobs.Thread") as mock_thread:
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 409, response.text
        assert response.json()["detail"]["error"] == ("custody_chain_bounds_exceeded")
        mock_thread.assert_not_called()

    @pytest.mark.parametrize(
        ("limit_name", "limit_value"),
        [
            ("_REDETECT_MAX_CUSTODY_CANDIDATES", 1),
            ("_REDETECT_MAX_CUSTODY_AGGREGATE_BYTES", 1),
            ("_REDETECT_MAX_CUSTODY_EVENTS", 1),
        ],
    )
    def test_redetect_custody_aggregate_bounds_are_fail_closed(
        self,
        client,
        sample_pdf,
        tmp_path,
        monkeypatch,
        limit_name,
        limit_value,
    ):
        with patch("api.job_manager.JobManager._run_pipeline", return_value=None):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
        custody_dir = tmp_path / "output" / job_id / "EXPORT" / "CUSTODY"
        if limit_name in {
            "_REDETECT_MAX_CUSTODY_CANDIDATES",
            "_REDETECT_MAX_CUSTODY_EVENTS",
        }:
            foreign = CustodyChain("foreign_chain", str(sample_pdf), str(custody_dir))
            foreign.append_event("file_ingested", {"file_hash": "b" * 64})
        monkeypatch.setattr(f"api.routers.jobs.{limit_name}", limit_value)

        with patch("api.routers.jobs.Thread") as mock_thread:
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 409, response.text
        assert response.json()["detail"]["error"] == ("custody_chain_bounds_exceeded")
        mock_thread.assert_not_called()

    @pytest.mark.parametrize("child_name", ["LANGUAGE", "CUSTODY"])
    def test_queued_child_directory_swap_cannot_redirect_write(
        self,
        child_name,
        client,
        sample_pdf,
        tmp_path,
    ):
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread", _CapturedThread),
        ):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 202, response.text
        assert _CapturedThread.target is not None

        result_dir = tmp_path / "output" / job_id
        child = result_dir / "EXPORT" / child_name
        authorized = child.with_name(f"{child_name}.authorized")
        foreign = tmp_path / f"foreign-{child_name.lower()}"
        foreign.mkdir()
        child.rename(authorized)
        child.symlink_to(foreign, target_is_directory=True)

        with patch(
            "ocr_local.features.language_detection.redetect_document"
        ) as mock_redetect:
            _CapturedThread.target()

        mock_redetect.assert_not_called()
        assert list(foreign.iterdir()) == []

    def test_queued_job_root_swap_cannot_redirect_write(
        self,
        client,
        sample_pdf,
        tmp_path,
    ):
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread", _CapturedThread),
        ):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 202, response.text
        assert _CapturedThread.target is not None

        result_dir = tmp_path / "output" / job_id
        authorized = result_dir.with_name(f"{job_id}.authorized")
        result_dir.rename(authorized)
        result_dir.mkdir()

        with patch(
            "ocr_local.features.language_detection.redetect_document"
        ) as mock_redetect:
            _CapturedThread.target()

        mock_redetect.assert_not_called()
        assert list(result_dir.iterdir()) == []

    def test_queued_pdf_directory_swap_uses_retained_authorized_bytes(
        self,
        client,
        sample_pdf,
        tmp_path,
    ):
        """The worker never reopens a PDF child replaced after queueing."""
        with (
            patch("api.job_manager.JobManager._run_pipeline", return_value=None),
            patch("api.routers.jobs.Thread", _CapturedThread),
        ):
            job_id = _submit_and_complete(client, sample_pdf, tmp_path)
            response = client.post(f"/api/v1/jobs/{job_id}/redetect-language")

        assert response.status_code == 202, response.text
        assert _CapturedThread.target is not None

        pdf_dir = tmp_path / "output" / job_id / "EXPORT" / "PDF"
        authorized = pdf_dir.with_name("PDF-authorized")
        pdf_dir.rename(authorized)
        pdf_dir.mkdir()
        (pdf_dir / "test.pdf").write_bytes(b"FOREIGN TENANT PDF")
        observed = {}

        def _observe_redetect(**kwargs):
            observed.update(kwargs)
            return {
                "status": "ok",
                "sidecar_filename": "test.language.json",
                "sidecar_bytes": b"{}",
            }

        with patch(
            "ocr_local.features.language_detection.redetect_document",
            side_effect=_observe_redetect,
        ):
            _CapturedThread.target()

        assert observed["pdf_bytes"] == sample_pdf.read_bytes()
        assert observed["pdf_bytes"] != b"FOREIGN TENANT PDF"
