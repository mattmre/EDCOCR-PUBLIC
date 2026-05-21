"""Tests for the Python SDK client (sdk.python_client).

All HTTP interactions are mocked — no real network calls are made.
"""

import io
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# The SDK module uses a lazy import for `requests`, so we can import the
# module itself without the library installed.  We patch at the session level.
from sdk.python_client import (
    AuthenticationError,
    HealthInfo,
    JobInfo,
    JobStatus,
    NotFoundError,
    OcrClient,
    OcrClientError,
    ServerError,
    TimeoutError,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mock_response(status_code=200, json_data=None, text="", content=b"",
                   content_type="application/json"):
    """Build a mock ``requests.Response``."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-type": content_type}
    resp.json.return_value = json_data or {}
    resp.text = text or (str(json_data) if json_data else "")
    resp.content = content
    return resp


# ------------------------------------------------------------------
# JobStatus enum
# ------------------------------------------------------------------


class TestJobStatus:
    """Validate enum members."""

    def test_has_five_members(self):
        assert len(JobStatus) == 5

    def test_values(self):
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_lookup_by_value(self):
        assert JobStatus("completed") is JobStatus.COMPLETED


# ------------------------------------------------------------------
# JobInfo dataclass
# ------------------------------------------------------------------


class TestJobInfo:
    """Test JobInfo construction, serialisation, and predicates."""

    def test_defaults(self):
        info = JobInfo()
        assert info.job_id == ""
        assert info.status == ""
        assert info.filename == ""
        assert info.pages == 0
        assert info.created_at == ""
        assert info.completed_at == ""
        assert info.error == ""
        assert info.progress == 0.0

    def test_from_dict(self):
        data = {
            "job_id": "job_abc123",
            "status": "completed",
            "filename": "scan.pdf",
            "pages": 5,
            "created_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:01:00",
            "error": "",
            "progress": 100.0,
        }
        info = JobInfo.from_dict(data)
        assert info.job_id == "job_abc123"
        assert info.status == "completed"
        assert info.filename == "scan.pdf"
        assert info.pages == 5
        assert info.progress == 100.0

    def test_from_dict_alt_keys(self):
        """Server may use alternate key names."""
        data = {
            "id": "alt-id",
            "status": "processing",
            "original_filename": "alt.pdf",
            "total_pages": 3,
            "error_message": "oops",
        }
        info = JobInfo.from_dict(data)
        assert info.job_id == "alt-id"
        assert info.filename == "alt.pdf"
        assert info.pages == 3
        assert info.error == "oops"

    def test_to_dict(self):
        info = JobInfo(job_id="j1", status="queued", filename="f.pdf", pages=2)
        d = info.to_dict()
        assert d["job_id"] == "j1"
        assert d["status"] == "queued"
        assert d["filename"] == "f.pdf"
        assert d["pages"] == 2

    def test_is_complete_true_for_terminal_states(self):
        for st in ("completed", "failed", "cancelled"):
            assert JobInfo(status=st).is_complete is True

    def test_is_complete_false_for_active_states(self):
        for st in ("queued", "processing"):
            assert JobInfo(status=st).is_complete is False

    def test_is_success(self):
        assert JobInfo(status="completed").is_success is True
        assert JobInfo(status="failed").is_success is False
        assert JobInfo(status="queued").is_success is False


# ------------------------------------------------------------------
# HealthInfo dataclass
# ------------------------------------------------------------------


class TestHealthInfo:
    """Test HealthInfo construction and serialisation."""

    def test_defaults(self):
        h = HealthInfo()
        assert h.status == ""
        assert h.version == ""
        assert h.uptime_seconds == 0.0

    def test_from_dict(self):
        data = {"status": "healthy", "version": "1.2.3", "uptime_seconds": 42.5}
        h = HealthInfo.from_dict(data)
        assert h.status == "healthy"
        assert h.version == "1.2.3"
        assert h.uptime_seconds == 42.5

    def test_from_dict_alt_uptime_key(self):
        data = {"status": "ok", "version": "0.1", "uptime": 99.0}
        h = HealthInfo.from_dict(data)
        assert h.uptime_seconds == 99.0

    def test_to_dict(self):
        h = HealthInfo(status="healthy", version="2.0", uptime_seconds=10.0)
        d = h.to_dict()
        assert d == {"status": "healthy", "version": "2.0", "uptime_seconds": 10.0}


# ------------------------------------------------------------------
# Exception hierarchy
# ------------------------------------------------------------------


class TestOcrClientError:
    """Base exception carries status_code and response_body."""

    def test_message(self):
        exc = OcrClientError("boom")
        assert str(exc) == "boom"

    def test_status_code(self):
        exc = OcrClientError("fail", status_code=418)
        assert exc.status_code == 418

    def test_response_body(self):
        exc = OcrClientError("fail", response_body='{"err":"x"}')
        assert exc.response_body == '{"err":"x"}'

    def test_defaults(self):
        exc = OcrClientError("msg")
        assert exc.status_code == 0
        assert exc.response_body == ""


class TestAuthenticationError:
    def test_is_ocr_client_error(self):
        assert issubclass(AuthenticationError, OcrClientError)

    def test_instance(self):
        exc = AuthenticationError("denied", status_code=401)
        assert isinstance(exc, OcrClientError)
        assert exc.status_code == 401


class TestNotFoundError:
    def test_is_ocr_client_error(self):
        assert issubclass(NotFoundError, OcrClientError)

    def test_instance(self):
        exc = NotFoundError("missing", status_code=404)
        assert isinstance(exc, OcrClientError)


class TestTimeoutError:
    def test_is_ocr_client_error(self):
        assert issubclass(TimeoutError, OcrClientError)


class TestServerError:
    def test_is_ocr_client_error(self):
        assert issubclass(ServerError, OcrClientError)


# ------------------------------------------------------------------
# OcrClient — construction
# ------------------------------------------------------------------


class TestOcrClientConstruction:
    """Test client initialisation."""

    def test_base_url_trailing_slash_stripped(self):
        c = OcrClient("http://host:8000/")
        assert c.base_url == "http://host:8000"

    def test_base_url_no_trailing_slash(self):
        c = OcrClient("http://host:8000")
        assert c.base_url == "http://host:8000"

    def test_api_key_stored(self):
        c = OcrClient("http://h", api_key="k123")
        assert c.api_key == "k123"

    def test_defaults(self):
        c = OcrClient("http://h")
        assert c.timeout == 30.0
        assert c.max_retries == 3

    @patch("sdk.python_client._get_requests")
    def test_session_sets_api_key_header(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h", api_key="secret")
        _ = c.session
        assert mock_session.headers["X-API-Key"] == "secret"

    @patch("sdk.python_client._get_requests")
    def test_session_sets_user_agent(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        _ = c.session
        assert "ocr-local-python-sdk" in mock_session.headers["User-Agent"]


# ------------------------------------------------------------------
# OcrClient._check_response
# ------------------------------------------------------------------


class TestCheckResponse:
    """Test HTTP status-to-exception mapping."""

    def _client(self):
        return OcrClient("http://h")

    def test_200_no_error(self):
        self._client()._check_response(_mock_response(200))

    def test_201_no_error(self):
        self._client()._check_response(_mock_response(201))

    def test_401_raises_auth_error(self):
        with pytest.raises(AuthenticationError) as exc_info:
            self._client()._check_response(_mock_response(401))
        assert exc_info.value.status_code == 401

    def test_403_raises_auth_error(self):
        with pytest.raises(AuthenticationError) as exc_info:
            self._client()._check_response(_mock_response(403))
        assert exc_info.value.status_code == 403

    def test_404_raises_not_found(self):
        with pytest.raises(NotFoundError) as exc_info:
            self._client()._check_response(_mock_response(404))
        assert exc_info.value.status_code == 404

    def test_500_raises_server_error(self):
        with pytest.raises(ServerError) as exc_info:
            self._client()._check_response(_mock_response(500))
        assert exc_info.value.status_code == 500

    def test_422_raises_generic_error(self):
        with pytest.raises(OcrClientError) as exc_info:
            self._client()._check_response(_mock_response(422, text='{"detail":"bad"}'))
        assert exc_info.value.status_code == 422


# ------------------------------------------------------------------
# OcrClient._request
# ------------------------------------------------------------------


class TestRequest:
    """Test retry logic and response parsing."""

    @patch("sdk.python_client._get_requests")
    def test_retry_on_connection_error(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.side_effect = [
            ConnectionError("conn refused"),
            _mock_response(200, json_data={"ok": True}),
        ]
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h", max_retries=3)
        with patch("sdk.python_client.time.sleep"):
            result = c._request("GET", "/test")

        assert result == {"ok": True}
        assert mock_session.request.call_count == 2

    @patch("sdk.python_client._get_requests")
    def test_204_returns_empty_dict(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(204)
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        assert c._request("DELETE", "/api/v1/jobs/x") == {}

    @patch("sdk.python_client._get_requests")
    def test_all_retries_exhausted(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.side_effect = ConnectionError("down")
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h", max_retries=2)
        with patch("sdk.python_client.time.sleep"):
            with pytest.raises(OcrClientError, match="retries"):
                c._request("GET", "/fail")

    @patch("sdk.python_client._get_requests")
    def test_non_json_response(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        resp = _mock_response(200, text="plain text", content_type="text/plain")
        mock_session.request.return_value = resp
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        result = c._request("GET", "/text")
        assert "raw" in result


# ------------------------------------------------------------------
# OcrClient.health
# ------------------------------------------------------------------


class TestHealth:
    @patch("sdk.python_client._get_requests")
    def test_health_returns_health_info(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(
            200, json_data={"status": "healthy", "version": "1.0", "uptime_seconds": 55.2}
        )
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        h = c.health()
        assert isinstance(h, HealthInfo)
        assert h.status == "healthy"
        assert h.version == "1.0"
        assert h.uptime_seconds == 55.2


# ------------------------------------------------------------------
# OcrClient.submit
# ------------------------------------------------------------------


class TestSubmit:
    @patch("sdk.python_client._get_requests")
    def test_submit_with_file_path(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(
            201, json_data={"job_id": "job_aaa111bbb222", "status": "submitted"}
        )
        mock_get_req.return_value.Session.return_value = mock_session

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"%PDF-1.4 test content")
            tmp_path = tmp.name

        try:
            c = OcrClient("http://h", api_key="k")
            job = c.submit(file_path=tmp_path)
            assert isinstance(job, JobInfo)
            assert job.job_id == "job_aaa111bbb222"
            assert job.status == "submitted"
            # Verify POST was called with multipart
            call_args = mock_session.request.call_args
            assert call_args[0][0] == "POST"
        finally:
            os.unlink(tmp_path)

    @patch("sdk.python_client._get_requests")
    def test_submit_with_file_obj(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(
            201, json_data={"job_id": "job_ccc333ddd444", "status": "submitted"}
        )
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        buf = io.BytesIO(b"fake pdf content")
        job = c.submit(file_obj=buf, filename="test.pdf")
        assert job.job_id == "job_ccc333ddd444"

    @patch("sdk.python_client._get_requests")
    def test_submit_with_options(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(
            201, json_data={"job_id": "job_eee555fff666", "status": "submitted"}
        )
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        buf = io.BytesIO(b"data")
        c.submit(
            file_obj=buf,
            filename="doc.pdf",
            enable_docintel=True,
            webhook_url="https://hook.example.com",
            priority="high",
        )
        call_kwargs = mock_session.request.call_args[1]
        assert call_kwargs["data"]["enable_docintel"] == "true"
        assert call_kwargs["data"]["webhook_url"] == "https://hook.example.com"
        assert call_kwargs["data"]["priority"] == "high"

    def test_submit_file_not_found(self):
        c = OcrClient("http://h")
        with pytest.raises(FileNotFoundError, match="not_a_real_file"):
            c.submit(file_path="/tmp/not_a_real_file.pdf")

    def test_submit_no_input_raises_value_error(self):
        c = OcrClient("http://h")
        with pytest.raises(ValueError, match="Either file_path or file_obj"):
            c.submit()


# ------------------------------------------------------------------
# OcrClient.get_job
# ------------------------------------------------------------------


class TestGetJob:
    @patch("sdk.python_client._get_requests")
    def test_get_job(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(
            200, json_data={
                "job_id": "job_abc123def456",
                "status": "processing",
                "progress": 45.0,
            }
        )
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        job = c.get_job("job_abc123def456")
        assert isinstance(job, JobInfo)
        assert job.status == "processing"
        assert job.progress == 45.0


# ------------------------------------------------------------------
# OcrClient.list_jobs
# ------------------------------------------------------------------


class TestListJobs:
    @patch("sdk.python_client._get_requests")
    def test_list_returns_list_of_job_info(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(
            200, json_data={
                "jobs": [
                    {"job_id": "job_111111111111", "status": "completed"},
                    {"job_id": "job_222222222222", "status": "queued"},
                ],
                "total": 2,
            }
        )
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        jobs = c.list_jobs()
        assert len(jobs) == 2
        assert all(isinstance(j, JobInfo) for j in jobs)
        assert jobs[0].job_id == "job_111111111111"

    @patch("sdk.python_client._get_requests")
    def test_list_handles_raw_list_response(self, mock_get_req):
        """Some APIs return a bare JSON array."""
        mock_session = MagicMock()
        mock_session.headers = {}
        resp = _mock_response(200)
        resp.json.return_value = [
            {"job_id": "job_aaaaaaaaaaaa", "status": "completed"},
        ]
        mock_session.request.return_value = resp
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        # Patch _request to return list directly
        c._request = MagicMock(return_value=[{"job_id": "job_aaaaaaaaaaaa", "status": "completed"}])
        jobs = c.list_jobs()
        assert len(jobs) == 1

    @patch("sdk.python_client._get_requests")
    def test_list_with_params(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(
            200, json_data={"jobs": [], "total": 0}
        )
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        c.list_jobs(status="completed", limit=10, offset=5)

        call_kwargs = mock_session.request.call_args[1]
        assert call_kwargs["params"]["status"] == "completed"
        assert call_kwargs["params"]["limit"] == 10
        assert call_kwargs["params"]["offset"] == 5


# ------------------------------------------------------------------
# OcrClient.cancel_job
# ------------------------------------------------------------------


class TestCancelJob:
    @patch("sdk.python_client._get_requests")
    def test_cancel_success(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(
            200, json_data={"job_id": "job_aaa111bbb222", "status": "cancelled"}
        )
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        assert c.cancel_job("job_aaa111bbb222") is True

    @patch("sdk.python_client._get_requests")
    def test_cancel_not_found_returns_false(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(404)
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        assert c.cancel_job("job_nonexistent0") is False


# ------------------------------------------------------------------
# OcrClient.download_result
# ------------------------------------------------------------------


class TestDownloadResult:
    @patch("sdk.python_client._get_requests")
    def test_download_returns_bytes(self, mock_get_req):
        mock_resp = _mock_response(200, content=b"PDF-binary-data")
        mock_get_req.return_value.get.return_value = mock_resp

        c = OcrClient("http://h", api_key="k")
        data = c.download_result("job_aaa111bbb222")
        assert data == b"PDF-binary-data"

    @patch("sdk.python_client._get_requests")
    def test_download_saves_to_file(self, mock_get_req):
        payload = b"saved-content-bytes"
        mock_resp = _mock_response(200, content=payload)
        mock_get_req.return_value.get.return_value = mock_resp

        c = OcrClient("http://h")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name

        try:
            c.download_result("job_aaa111bbb222", output_path=tmp_path)
            assert open(tmp_path, "rb").read() == payload
        finally:
            os.unlink(tmp_path)

    @patch("sdk.python_client._get_requests")
    def test_download_passes_api_key_header(self, mock_get_req):
        mock_resp = _mock_response(200, content=b"x")
        mock_get_req.return_value.get.return_value = mock_resp

        c = OcrClient("http://h", api_key="my-secret-key")
        c.download_result("job_aaa111bbb222")
        call_kwargs = mock_get_req.return_value.get.call_args[1]
        assert call_kwargs["headers"]["X-API-Key"] == "my-secret-key"


# ------------------------------------------------------------------
# OcrClient.wait_for_result
# ------------------------------------------------------------------


class TestWaitForResult:
    @patch("sdk.python_client._get_requests")
    def test_immediate_completion(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(
            200, json_data={"job_id": "job_aaa111bbb222", "status": "completed"}
        )
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        job = c.wait_for_result("job_aaa111bbb222")
        assert job.is_success is True

    @patch("sdk.python_client.time.sleep")
    @patch("sdk.python_client._get_requests")
    def test_polling_until_complete(self, mock_get_req, mock_sleep):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.side_effect = [
            _mock_response(200, json_data={"job_id": "j1", "status": "processing"}),
            _mock_response(200, json_data={"job_id": "j1", "status": "processing"}),
            _mock_response(200, json_data={"job_id": "j1", "status": "completed"}),
        ]
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        job = c.wait_for_result("j1", poll_interval=0.01)
        assert job.is_complete is True
        assert mock_sleep.call_count == 2

    @patch("sdk.python_client.time.time")
    @patch("sdk.python_client.time.sleep")
    @patch("sdk.python_client._get_requests")
    def test_timeout(self, mock_get_req, mock_sleep, mock_time):
        # Simulate time advancing past the timeout
        mock_time.side_effect = [0.0, 10.0, 20.0]

        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.return_value = _mock_response(
            200, json_data={"job_id": "j2", "status": "processing"}
        )
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        with pytest.raises(TimeoutError, match="did not complete"):
            c.wait_for_result("j2", poll_interval=0.01, timeout=5.0)


# ------------------------------------------------------------------
# OcrClient.submit_and_wait
# ------------------------------------------------------------------


class TestSubmitAndWait:
    @patch("sdk.python_client._get_requests")
    def test_submit_and_wait(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.request.side_effect = [
            # submit
            _mock_response(201, json_data={"job_id": "job_aaa111bbb222", "status": "submitted"}),
            # wait_for_result — first poll returns completed
            _mock_response(200, json_data={"job_id": "job_aaa111bbb222", "status": "completed"}),
        ]
        mock_get_req.return_value.Session.return_value = mock_session

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"%PDF-test")
            tmp_path = tmp.name

        try:
            c = OcrClient("http://h")
            job = c.submit_and_wait(tmp_path)
            assert job.is_success is True
            assert job.job_id == "job_aaa111bbb222"
        finally:
            os.unlink(tmp_path)


# ------------------------------------------------------------------
# OcrClient.close / context manager
# ------------------------------------------------------------------


class TestCloseAndContextManager:
    @patch("sdk.python_client._get_requests")
    def test_close_clears_session(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_get_req.return_value.Session.return_value = mock_session

        c = OcrClient("http://h")
        _ = c.session  # trigger lazy init
        assert c._session is not None

        c.close()
        assert c._session is None
        mock_session.close.assert_called_once()

    @patch("sdk.python_client._get_requests")
    def test_close_noop_when_no_session(self, mock_get_req):
        c = OcrClient("http://h")
        c.close()  # should not raise

    @patch("sdk.python_client._get_requests")
    def test_context_manager(self, mock_get_req):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_get_req.return_value.Session.return_value = mock_session

        with OcrClient("http://h") as c:
            _ = c.session  # trigger lazy init
        # After exiting context, session should be closed
        mock_session.close.assert_called_once()
