"""Tests for the edcocr-sdk PyPI package (sdk/python/src/edcocr_sdk).

All HTTP interactions are mocked -- no live server is required.
Tests run as part of the main test suite via pytest.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure the SDK source is importable from the repo checkout
_SDK_SRC = str(Path(__file__).resolve().parent.parent / "sdk" / "python" / "src")
if _SDK_SRC not in sys.path:
    sys.path.insert(0, _SDK_SRC)

from edcocr_sdk import (  # noqa: E402
    AsyncEDCOCRClient,
    AuthenticationError,
    ConflictError,
    HealthResponse,
    Job,
    JobListResponse,
    JobResult,
    JobStatus,
    JobSubmitResult,
    NotFoundError,
    EDCOCRClient,
    OCRLocalError,
    Priority,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
    __version__,
)
from edcocr_sdk.client import (  # noqa: E402
    SDK_VERSION,
    USER_AGENT,
    _build_headers,
    _build_submit_data,
    _raise_for_status,
)
from edcocr_sdk.models import (  # noqa: E402
    BatchItem,
    BatchSubmission,
    DocIntelMode,
    ErrorDetail,
    JobProgress,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEALTH_JSON = {
    "status": "healthy",
    "version": "0.0.0-test",
    "uptime_seconds": 123.4,
    "jobs": {"submitted": 1, "processing": 2, "completed": 10},
}

_JOB_STATUS_JSON = {
    "job_id": "job_abc123def456",
    "status": "processing",
    "created_at": "2026-03-15T10:00:00",
    "started_at": "2026-03-15T10:00:01",
    "completed_at": None,
    "priority": "normal",
    "source_file": "test.pdf",
    "progress": {
        "total_pages": 5,
        "pages_completed": 2,
        "percent_complete": 40.0,
        "current_stage": "ocr",
    },
    "settings": {},
    "webhook_status": None,
}

_JOB_SUBMIT_JSON = {
    "job_id": "job_abc123def456",
    "status": "submitted",
    "created_at": "2026-03-15T10:00:00",
    "priority": "normal",
    "source_file": "test.pdf",
    "estimated_pages": 5,
    "links": {
        "self": "/api/v1/jobs/job_abc123def456",
        "result": "/api/v1/jobs/job_abc123def456/result",
    },
}

_JOB_RESULT_JSON = {
    "job_id": "job_abc123def456",
    "status": "completed",
    "completed_at": "2026-03-15T10:05:00",
    "processing_time_seconds": 300.0,
    "artifacts": {"pdf": "/download/pdf", "text": "/download/text"},
    "metadata": {"pages_processed": 5},
}

_JOB_LIST_JSON = {
    "jobs": [_JOB_STATUS_JSON],
    "total": 1,
    "limit": 50,
    "offset": 0,
}

_OUTPUTS_JSON = {
    "job_id": "job_abc123def456",
    "artifacts": [
        {
            "type": "ocr_text",
            "path": "job_abc123def456.txt",
            "download_url": "/api/v1/jobs/job_abc123def456/outputs/ocr_text",
        }
    ],
    "schema_versions": {"manifest": "1.0"},
}

_DOCUMENT_BUNDLE_JSON = {
    "schema_version": "DocumentBundle.v1",
    "document": {"document_id": "job_abc123def456", "source_file": "test.pdf"},
    "pages": [{"page_number": 1, "text": "OCR text"}],
    "spans": [],
    "custody": {"chain_head": "abc123", "evidence": []},
    "artifacts": [],
    "metadata": {},
}

_EVIDENCE_BUNDLE_JSON = {
    "schema_version": "ocr-evidence-bundle-v1",
    "job_id": "job_abc123def456",
    "source_file": "test.pdf",
    "status": "completed",
    "custody": {"available": True, "valid": True, "chain_head": "abc123"},
    "document_bundle_sha256": "def456",
    "document_bundle_url": "/api/v1/jobs/job_abc123def456/document-bundle",
    "artifacts": [],
}

_BATCH_SUBMIT_JSON = {
    "batch_id": "batch_abc123def456",
    "status": "submitted",
    "created_at": "2026-03-15T10:00:00",
    "total_jobs": 1,
    "priority": "normal",
    "jobs": [{"job_id": "job_abc123def456", "source_file": "test.pdf", "status": "submitted"}],
    "links": {"self": "/api/v1/jobs/batch/batch_abc123def456"},
}

_BATCH_LIST_JSON = {
    "batches": [_BATCH_SUBMIT_JSON],
    "total": 1,
    "limit": 50,
    "offset": 0,
}


class _FakeResponse:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, status_code: int = 200, json_data=None, content=b"", text=""):
        self.status_code = status_code
        self._json = json_data
        self.content = content
        self._text = text
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._json

    @property
    def text(self):
        if self._text:
            return self._text
        if self._json is not None:
            return json.dumps(self._json)
        return ""


def _mock_httpx_client():
    """Return a MagicMock that pretends to be an httpx.Client."""
    mock = MagicMock()
    mock.is_closed = False
    return mock


def _inject_mock_client(sdk_client):
    """Inject a mock httpx client into the SDK client, bypassing the property."""
    mock = _mock_httpx_client()
    sdk_client._client = mock
    return mock


def _inject_async_mock_client(sdk_client):
    """Inject a mock async httpx client into the SDK client."""
    mock = MagicMock()
    mock.is_closed = False
    return mock


# ---------------------------------------------------------------------------
# Version and metadata
# ---------------------------------------------------------------------------


class TestVersionAndMetadata:
    def test_version_string(self):
        assert __version__ == SDK_VERSION

    def test_sdk_version_matches(self):
        # SDK_VERSION should be a valid semver string
        parts = SDK_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_user_agent_format(self):
        assert USER_AGENT == f"edcocr-sdk-python/{SDK_VERSION}"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TestEnumerations:
    def test_job_status_values(self):
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.SUBMITTED.value == "submitted"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_priority_values(self):
        assert Priority.URGENT.value == "urgent"
        assert Priority.NORMAL.value == "normal"
        assert Priority.LOW.value == "low"

    def test_docintel_mode_values(self):
        assert DocIntelMode.LAYOUT_ONLY.value == "layout_only"
        assert DocIntelMode.TABLES_ONLY.value == "tables_only"
        assert DocIntelMode.FULL.value == "full"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModels:
    def test_health_response_parse(self):
        h = HealthResponse.model_validate(_HEALTH_JSON)
        assert h.status == "healthy"
        assert h.version == "0.0.0-test"
        assert h.uptime_seconds == 123.4
        assert h.jobs["completed"] == 10

    def test_job_parse(self):
        j = Job.model_validate(_JOB_STATUS_JSON)
        assert j.job_id == "job_abc123def456"
        assert j.status == "processing"
        assert j.priority == "normal"
        assert j.progress is not None
        assert j.progress.percent_complete == 40.0

    def test_job_is_terminal(self):
        j = Job(job_id="j1", status="processing")
        assert not j.is_terminal
        assert not j.is_success

        j2 = Job(job_id="j2", status="completed")
        assert j2.is_terminal
        assert j2.is_success

        j3 = Job(job_id="j3", status="failed")
        assert j3.is_terminal
        assert not j3.is_success

        j4 = Job(job_id="j4", status="cancelled")
        assert j4.is_terminal
        assert not j4.is_success

    def test_job_submit_result_parse(self):
        r = JobSubmitResult.model_validate(_JOB_SUBMIT_JSON)
        assert r.job_id == "job_abc123def456"
        assert r.status == "submitted"
        assert r.estimated_pages == 5
        assert r.links is not None

    def test_job_result_parse(self):
        r = JobResult.model_validate(_JOB_RESULT_JSON)
        assert r.processing_time_seconds == 300.0
        assert "pdf" in r.artifacts

    def test_job_list_response_parse(self):
        r = JobListResponse.model_validate(_JOB_LIST_JSON)
        assert r.total == 1
        assert len(r.jobs) == 1
        assert r.jobs[0].job_id == "job_abc123def456"

    def test_job_progress_defaults(self):
        p = JobProgress()
        assert p.total_pages == 0
        assert p.current_stage == "submitted"

    def test_error_detail_parse(self):
        e = ErrorDetail(error="not_found", message="Job not found")
        assert e.error == "not_found"
        assert e.details == {}

    def test_batch_submission(self):
        b = BatchSubmission(items=[
            BatchItem(source_path="/data/doc.pdf", priority=Priority.URGENT),
            BatchItem(source_path="/data/doc2.pdf", enable_docintel=True),
        ])
        assert len(b.items) == 2
        assert b.items[0].priority == Priority.URGENT
        assert b.items[1].enable_docintel is True


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_base_exception_attributes(self):
        exc = OCRLocalError("test error", status_code=500, response_body="body")
        assert str(exc) == "test error"
        assert exc.status_code == 500
        assert exc.response_body == "body"

    def test_base_exception_repr(self):
        exc = OCRLocalError("test", status_code=400)
        assert "OCRLocalError" in repr(exc)
        assert "400" in repr(exc)

    def test_exception_hierarchy(self):
        assert issubclass(AuthenticationError, OCRLocalError)
        assert issubclass(NotFoundError, OCRLocalError)
        assert issubclass(RateLimitError, OCRLocalError)
        assert issubclass(ValidationError, OCRLocalError)
        assert issubclass(ConflictError, OCRLocalError)
        assert issubclass(ServerError, OCRLocalError)
        assert issubclass(TimeoutError, OCRLocalError)

    def test_raise_for_status_success(self):
        resp = _FakeResponse(200)
        _raise_for_status(resp)  # Should not raise

    def test_raise_for_status_401(self):
        with pytest.raises(AuthenticationError) as exc_info:
            _raise_for_status(_FakeResponse(401, text="Unauthorized"))
        assert exc_info.value.status_code == 401

    def test_raise_for_status_403(self):
        with pytest.raises(AuthenticationError):
            _raise_for_status(_FakeResponse(403))

    def test_raise_for_status_404(self):
        with pytest.raises(NotFoundError) as exc_info:
            _raise_for_status(_FakeResponse(404))
        assert exc_info.value.status_code == 404

    def test_raise_for_status_409(self):
        with pytest.raises(ConflictError):
            _raise_for_status(_FakeResponse(409))

    def test_raise_for_status_422(self):
        with pytest.raises(ValidationError):
            _raise_for_status(_FakeResponse(422))

    def test_raise_for_status_429(self):
        with pytest.raises(RateLimitError):
            _raise_for_status(_FakeResponse(429))

    def test_raise_for_status_500(self):
        with pytest.raises(ServerError):
            _raise_for_status(_FakeResponse(500))

    def test_raise_for_status_502(self):
        with pytest.raises(ServerError) as exc_info:
            _raise_for_status(_FakeResponse(502))
        assert exc_info.value.status_code == 502

    def test_raise_for_status_generic_4xx(self):
        with pytest.raises(OCRLocalError):
            _raise_for_status(_FakeResponse(418))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_build_headers_with_key(self):
        h = _build_headers("my-secret-key")
        assert h["X-API-Key"] == "my-secret-key"
        assert "User-Agent" in h

    def test_build_headers_no_key(self):
        h = _build_headers("")
        assert "X-API-Key" not in h
        assert "User-Agent" in h

    def test_build_submit_data_empty(self):
        d = _build_submit_data()
        assert d == {}

    def test_build_submit_data_all_fields(self):
        webhook_placeholder = "placeholder"
        d = _build_submit_data(
            enable_docintel=True,
            docintel_mode="full",
            webhook_url="https://example.com/hook",
            webhook_secret=webhook_placeholder,
            priority="urgent",
            processing_timeout_minutes=60,
        )
        assert d["enable_docintel"] == "true"
        assert d["docintel_mode"] == "full"
        assert d["webhook_url"] == "https://example.com/hook"
        assert d["webhook_secret"] == webhook_placeholder
        assert d["priority"] == "urgent"
        assert d["processing_timeout_minutes"] == "60"


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class TestSyncClient:
    def test_init_strips_trailing_slash(self):
        c = EDCOCRClient("http://localhost:8000/", api_key="k")
        assert c.base_url == "http://localhost:8000"

    def test_context_manager(self):
        with EDCOCRClient("http://localhost:8000") as client:
            assert client is not None

    def test_submit_requires_input(self):
        client = EDCOCRClient("http://localhost:8000")
        with pytest.raises(ValueError, match="at least one"):
            client.submit_job()

    def test_submit_file_not_found(self):
        client = EDCOCRClient("http://localhost:8000")
        with pytest.raises(FileNotFoundError):
            client.submit_job(file_path="/nonexistent/file.pdf")

    def test_health_check(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.get.return_value = _FakeResponse(200, json_data=_HEALTH_JSON)
        health = client.health_check()
        assert health.status == "healthy"
        assert health.version == "0.0.0-test"

    def test_get_job(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.get.return_value = _FakeResponse(200, json_data=_JOB_STATUS_JSON)
        job = client.get_job("job_abc123def456")
        assert job.job_id == "job_abc123def456"
        assert job.status == "processing"

    def test_list_jobs(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.get.return_value = _FakeResponse(200, json_data=_JOB_LIST_JSON)
        result = client.list_jobs(status="processing")
        assert result.total == 1
        assert len(result.jobs) == 1

    def test_get_result(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.get.return_value = _FakeResponse(200, json_data=_JOB_RESULT_JSON)
        result = client.get_result("job_abc123def456")
        assert result.processing_time_seconds == 300.0
        assert "pdf" in result.artifacts

    def test_download_artifact(self, tmp_path):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        pdf_bytes = b"%PDF-1.4 fake content"
        mock.get.return_value = _FakeResponse(200, content=pdf_bytes)
        out = tmp_path / "result.pdf"
        content = client.download_artifact(
            "job_abc123def456", "pdf", output_path=out
        )
        assert content == pdf_bytes
        assert out.read_bytes() == pdf_bytes

    def test_download_artifact_no_save(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.get.return_value = _FakeResponse(200, content=b"text content")
        content = client.download_artifact("job_abc123def456", "text")
        assert content == b"text content"

    def test_list_outputs(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.get.return_value = _FakeResponse(200, json_data=_OUTPUTS_JSON)
        outputs = client.list_outputs("job_abc123def456")
        assert outputs["artifacts"][0]["type"] == "ocr_text"
        mock.get.assert_called_once_with("/api/v1/jobs/job_abc123def456/outputs")

    def test_get_document_bundle(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.get.return_value = _FakeResponse(200, json_data=_DOCUMENT_BUNDLE_JSON)
        bundle = client.get_document_bundle("job_abc123def456")
        assert bundle["schema_version"] == "DocumentBundle.v1"
        mock.get.assert_called_once_with("/api/v1/jobs/job_abc123def456/document-bundle")

    def test_export_document_bundle(self, tmp_path):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.get.return_value = _FakeResponse(200, json_data=_DOCUMENT_BUNDLE_JSON)
        out = tmp_path / "bundle.json"
        bundle = client.export_document_bundle("job_abc123def456", out)
        assert bundle["document"]["document_id"] == "job_abc123def456"
        assert json.loads(out.read_text(encoding="utf-8"))["schema_version"] == "DocumentBundle.v1"

    def test_get_evidence_bundle_and_verify_custody(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.get.return_value = _FakeResponse(200, json_data=_EVIDENCE_BUNDLE_JSON)
        evidence = client.get_evidence_bundle("job_abc123def456")
        assert evidence["custody"]["valid"] is True
        assert client.verify_custody("job_abc123def456") is True

    def test_submit_batch_with_source_paths(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.post.return_value = _FakeResponse(201, json_data=_BATCH_SUBMIT_JSON)
        batch = client.submit_batch(source_paths=["/data/doc.pdf"], priority="urgent")
        assert batch["batch_id"] == "batch_abc123def456"
        args, kwargs = mock.post.call_args
        assert args == ("/api/v1/jobs/batch",)
        assert json.loads(kwargs["data"]["source_paths"]) == ["/data/doc.pdf"]
        assert kwargs["data"]["priority"] == "urgent"

    def test_submit_batch_requires_input(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        with pytest.raises(ValueError, match="at least one"):
            client.submit_batch()

    def test_list_and_get_batch(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.get.side_effect = [
            _FakeResponse(200, json_data=_BATCH_LIST_JSON),
            _FakeResponse(200, json_data=_BATCH_SUBMIT_JSON),
        ]
        batches = client.list_batches(status="submitted")
        batch = client.get_batch("batch_abc123def456")
        assert batches["total"] == 1
        assert batch["batch_id"] == "batch_abc123def456"

    def test_cancel_job(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        cancelled = dict(_JOB_STATUS_JSON, status="cancelled")
        mock.delete.return_value = _FakeResponse(200, json_data=cancelled)
        job = client.cancel_job("job_abc123def456")
        assert job.status == "cancelled"

    def test_retry_job(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.post.return_value = _FakeResponse(201, json_data=_JOB_SUBMIT_JSON)
        result = client.retry_job("job_abc123def456")
        assert result.job_id == "job_abc123def456"

    def test_submit_with_file_obj(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.post.return_value = _FakeResponse(201, json_data=_JOB_SUBMIT_JSON)
        file_obj = io.BytesIO(b"fake pdf content")
        result = client.submit_job(file_obj=file_obj, filename="test.pdf")
        assert result.job_id == "job_abc123def456"

    def test_submit_with_source_path(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.post.return_value = _FakeResponse(201, json_data=_JOB_SUBMIT_JSON)
        result = client.submit_job(source_path="/data/doc.pdf")
        assert result.job_id == "job_abc123def456"

    def test_submit_with_real_file(self, tmp_path):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")
        mock.post.return_value = _FakeResponse(201, json_data=_JOB_SUBMIT_JSON)
        result = client.submit_job(file_path=pdf)
        assert result.job_id == "job_abc123def456"

    def test_submit_with_all_options(self, tmp_path):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        webhook_placeholder = "placeholder"
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")
        mock.post.return_value = _FakeResponse(201, json_data=_JOB_SUBMIT_JSON)
        result = client.submit_job(
            file_path=pdf,
            enable_docintel=True,
            docintel_mode="full",
            priority="urgent",
            webhook_url="https://example.com/hook",
            webhook_secret=webhook_placeholder,
            processing_timeout_minutes=30,
        )
        assert result.status == "submitted"

    def test_wait_for_completion_immediate(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        completed = dict(_JOB_STATUS_JSON, status="completed")
        mock.get.return_value = _FakeResponse(200, json_data=completed)
        job = client.wait_for_completion(
            "job_abc123def456", poll_interval=0.01
        )
        assert job.is_terminal

    def test_wait_for_completion_timeout(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        processing = dict(_JOB_STATUS_JSON, status="processing")
        mock.get.return_value = _FakeResponse(200, json_data=processing)
        with pytest.raises(TimeoutError, match="did not complete"):
            client.wait_for_completion(
                "job_abc123def456", poll_interval=0.01, timeout=0.05
            )

    def test_close_no_client(self):
        client = EDCOCRClient("http://localhost:8000")
        client.close()  # Should not raise

    def test_health_check_auth_error(self):
        client = EDCOCRClient("http://localhost:8000", api_key="bad")
        mock = _inject_mock_client(client)
        mock.get.return_value = _FakeResponse(401, text="Unauthorized")
        with pytest.raises(AuthenticationError):
            client.health_check()

    def test_get_job_not_found(self):
        client = EDCOCRClient("http://localhost:8000", api_key="k")
        mock = _inject_mock_client(client)
        mock.get.return_value = _FakeResponse(404)
        with pytest.raises(NotFoundError):
            client.get_job("job_000000000000")


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class TestAsyncClient:
    def test_init_strips_trailing_slash(self):
        c = AsyncEDCOCRClient("http://localhost:8000/", api_key="k")
        assert c.base_url == "http://localhost:8000"

    @pytest.mark.asyncio
    async def test_health_check(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock

        async def mock_get(*a, **kw):
            return _FakeResponse(200, json_data=_HEALTH_JSON)

        mock.get = mock_get
        health = await client.health_check()
        assert health.status == "healthy"

    @pytest.mark.asyncio
    async def test_get_job(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock

        async def mock_get(*a, **kw):
            return _FakeResponse(200, json_data=_JOB_STATUS_JSON)

        mock.get = mock_get
        job = await client.get_job("job_abc123def456")
        assert job.job_id == "job_abc123def456"

    @pytest.mark.asyncio
    async def test_submit_requires_input(self):
        client = AsyncEDCOCRClient("http://localhost:8000")
        with pytest.raises(ValueError, match="at least one"):
            await client.submit_job()

    @pytest.mark.asyncio
    async def test_submit_file_not_found(self):
        client = AsyncEDCOCRClient("http://localhost:8000")
        with pytest.raises(FileNotFoundError):
            await client.submit_job(file_path="/nonexistent/file.pdf")

    @pytest.mark.asyncio
    async def test_submit_with_source_path(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock

        async def mock_post(*a, **kw):
            return _FakeResponse(201, json_data=_JOB_SUBMIT_JSON)

        mock.post = mock_post
        result = await client.submit_job(source_path="/data/doc.pdf")
        assert result.job_id == "job_abc123def456"

    @pytest.mark.asyncio
    async def test_list_jobs(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock

        async def mock_get(*a, **kw):
            return _FakeResponse(200, json_data=_JOB_LIST_JSON)

        mock.get = mock_get
        result = await client.list_jobs()
        assert result.total == 1

    @pytest.mark.asyncio
    async def test_get_result(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock

        async def mock_get(*a, **kw):
            return _FakeResponse(200, json_data=_JOB_RESULT_JSON)

        mock.get = mock_get
        result = await client.get_result("job_abc123def456")
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_download_artifact(self, tmp_path):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock

        async def mock_get(*a, **kw):
            return _FakeResponse(200, content=b"pdf bytes")

        mock.get = mock_get
        out = tmp_path / "out.pdf"
        content = await client.download_artifact(
            "job_abc123def456", output_path=out
        )
        assert content == b"pdf bytes"
        assert out.exists()

    @pytest.mark.asyncio
    async def test_list_outputs(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock

        async def mock_get(*a, **kw):
            return _FakeResponse(200, json_data=_OUTPUTS_JSON)

        mock.get = mock_get
        outputs = await client.list_outputs("job_abc123def456")
        assert outputs["artifacts"][0]["type"] == "ocr_text"

    @pytest.mark.asyncio
    async def test_get_document_bundle(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock

        async def mock_get(*a, **kw):
            return _FakeResponse(200, json_data=_DOCUMENT_BUNDLE_JSON)

        mock.get = mock_get
        bundle = await client.get_document_bundle("job_abc123def456")
        assert bundle["schema_version"] == "DocumentBundle.v1"

    @pytest.mark.asyncio
    async def test_get_evidence_bundle_and_verify_custody(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock

        async def mock_get(*a, **kw):
            return _FakeResponse(200, json_data=_EVIDENCE_BUNDLE_JSON)

        mock.get = mock_get
        evidence = await client.get_evidence_bundle("job_abc123def456")
        assert evidence["custody"]["valid"] is True
        assert await client.verify_custody("job_abc123def456") is True

    @pytest.mark.asyncio
    async def test_submit_batch_with_source_paths(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock

        async def mock_post(*a, **kw):
            return _FakeResponse(201, json_data=_BATCH_SUBMIT_JSON)

        mock.post = mock_post
        batch = await client.submit_batch(source_paths=["/data/doc.pdf"])
        assert batch["batch_id"] == "batch_abc123def456"

    @pytest.mark.asyncio
    async def test_list_and_get_batch(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock
        responses = [
            _FakeResponse(200, json_data=_BATCH_LIST_JSON),
            _FakeResponse(200, json_data=_BATCH_SUBMIT_JSON),
        ]

        async def mock_get(*a, **kw):
            return responses.pop(0)

        mock.get = mock_get
        batches = await client.list_batches()
        batch = await client.get_batch("batch_abc123def456")
        assert batches["total"] == 1
        assert batch["batch_id"] == "batch_abc123def456"

    @pytest.mark.asyncio
    async def test_cancel_job(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock
        cancelled = dict(_JOB_STATUS_JSON, status="cancelled")

        async def mock_delete(*a, **kw):
            return _FakeResponse(200, json_data=cancelled)

        mock.delete = mock_delete
        job = await client.cancel_job("job_abc123def456")
        assert job.status == "cancelled"

    @pytest.mark.asyncio
    async def test_retry_job(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock

        async def mock_post(*a, **kw):
            return _FakeResponse(201, json_data=_JOB_SUBMIT_JSON)

        mock.post = mock_post
        result = await client.retry_job("job_abc123def456")
        assert result.job_id == "job_abc123def456"

    @pytest.mark.asyncio
    async def test_wait_for_completion_immediate(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock
        completed = dict(_JOB_STATUS_JSON, status="completed")

        async def mock_get(*a, **kw):
            return _FakeResponse(200, json_data=completed)

        mock.get = mock_get
        job = await client.wait_for_completion(
            "job_abc123def456", poll_interval=0.01
        )
        assert job.is_terminal

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock
        processing = dict(_JOB_STATUS_JSON, status="processing")

        async def mock_get(*a, **kw):
            return _FakeResponse(200, json_data=processing)

        mock.get = mock_get
        with pytest.raises(TimeoutError, match="did not complete"):
            await client.wait_for_completion(
                "job_abc123def456", poll_interval=0.01, timeout=0.05
            )

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with AsyncEDCOCRClient("http://localhost:8000") as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_close_no_client(self):
        client = AsyncEDCOCRClient("http://localhost:8000")
        await client.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_health_check_server_error(self):
        client = AsyncEDCOCRClient("http://localhost:8000", api_key="k")
        mock = MagicMock()
        mock.is_closed = False
        client._client = mock

        async def mock_get(*a, **kw):
            return _FakeResponse(500)

        mock.get = mock_get
        with pytest.raises(ServerError):
            await client.health_check()


# ---------------------------------------------------------------------------
# All exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Verify that all documented symbols are importable from the package root."""

    def test_all_exports(self):
        import edcocr_sdk

        expected = {
            "EDCOCRClient",
            "AsyncEDCOCRClient",
            "BatchItem",
            "BatchSubmission",
            "DocIntelMode",
            "ErrorDetail",
            "HealthResponse",
            "Job",
            "JobListResponse",
            "JobProgress",
            "JobResult",
            "JobStatus",
            "JobSubmitResult",
            "Priority",
            "AuthenticationError",
            "ConflictError",
            "NotFoundError",
            "OCRLocalError",
            "RateLimitError",
            "ServerError",
            "TimeoutError",
            "ValidationError",
        }
        actual = set(edcocr_sdk.__all__)
        assert expected == actual
