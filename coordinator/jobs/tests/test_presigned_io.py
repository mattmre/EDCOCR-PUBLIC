"""Tests for Phase 7C presigned URL HTTP I/O helpers (worker side).

Tests the worker-side download/upload functions that use presigned URLs
with stdlib urllib -- no S3 credentials required.
"""

import logging
import os
import tempfile
import traceback
from contextlib import nullcontext
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest
from django.test import SimpleTestCase

from jobs.presigned_io import (
    NoRedirectHandler,
    PresignedTransferError,
    _no_redirect_opener,
    _validate_presigned_url,
    delete_presigned,
    download_presigned,
    upload_presigned,
)


@pytest.mark.parametrize("operation", ("get", "put", "delete"))
def test_http_error_never_exposes_bearer_query_in_exception_or_worker_log(
    operation, tmp_path, caplog
):
    url = "https://s3.example.com/jobs/job-1/a.pdf?X-Amz-Signature=SECRET"
    failure = HTTPError(url, 403, "Forbidden", None, BytesIO(b"denied"))
    local = tmp_path / "a.pdf"
    local.write_bytes(b"payload")
    with (
        patch.dict(os.environ, {"S3_PRESIGNED_ALLOWED_HOSTS": "s3.example.com"}),
        patch("jobs.presigned_io._pinned_presigned_url", return_value=nullcontext(url)),
        patch("jobs.presigned_io._no_redirect_opener.open", side_effect=failure),
    ):
        with pytest.raises(PresignedTransferError) as captured:
            if operation == "get":
                download_presigned(url, str(local), job_id="job-1", expected_key="jobs/job-1/a.pdf")
            elif operation == "put":
                upload_presigned(str(local), url, job_id="job-1", expected_key="jobs/job-1/a.pdf")
            else:
                delete_presigned(url, job_id="job-1", expected_key="jobs/job-1/a.pdf")
    assert str(captured.value) == "presigned HTTP status 403"
    assert "SECRET" not in "".join(traceback.format_exception(captured.value))
    with caplog.at_level(logging.ERROR):
        logging.getLogger("jobs.tasks").error("staged transfer failed: %s", captured.value)
    assert "SECRET" not in caplog.text


class TestDeletePresigned(SimpleTestCase):
    @patch("jobs.presigned_io._pinned_presigned_url")
    @patch("jobs.presigned_io._no_redirect_opener.open")
    def test_deletes_exact_job_bound_key(self, mock_open, mock_pin):
        url = "https://s3.example.com/jobs/job-1/output/result.json?delete"
        mock_pin.return_value = nullcontext(url)
        response = MagicMock()
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = response
        with patch.dict(
            os.environ,
            {"S3_PRESIGNED_ALLOWED_HOSTS": "s3.example.com"},
            clear=False,
        ):
            delete_presigned(
                url,
                job_id="job-1",
                expected_key="jobs/job-1/output/result.json",
            )

        request = mock_open.call_args.args[0]
        self.assertEqual(request.get_method(), "DELETE")


class TestDownloadPresigned(SimpleTestCase):
    """Tests for download_presigned() HTTP GET helper."""

    def setUp(self):
        self._allowlist = patch.dict(
            os.environ,
            {
                "S3_PRESIGNED_ALLOWED_HOSTS": "s3.example.com",
                "S3_PRESIGNED_ALLOW_HTTP": "false",
                "S3_PRESIGNED_ALLOW_PRIVATE": "false",
            },
            clear=False,
        )
        self._allowlist.start()
        self.addCleanup(self._allowlist.stop)
        self._pin = patch(
            "jobs.presigned_io._pinned_presigned_url",
            side_effect=lambda url: nullcontext(url),
        )
        self._pin.start()
        self.addCleanup(self._pin.stop)

    @patch("jobs.presigned_io._no_redirect_opener.open")
    def test_downloads_to_local_path(self, mock_urlopen):
        """download_presigned writes response bytes to the specified local path."""
        # Mock urlopen to return a response with known content
        content = b"PDF-file-content-here"
        mock_response = MagicMock()
        mock_response.read.side_effect = [content, b""]
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "downloaded.pdf")
            observations = []
            result = download_presigned(
                "https://s3.example.com/key?get", local_path,
                transfer_observer=lambda event, count: observations.append((event, count)),
            )

            self.assertEqual(result, local_path)
            self.assertTrue(os.path.isfile(local_path))
            with open(local_path, "rb") as f:
                self.assertEqual(f.read(), content)
            self.assertEqual(observations, [("get_attempt", 0), ("get_success_bytes", len(content))])

    @patch("jobs.presigned_io._no_redirect_opener.open")
    def test_creates_parent_dirs(self, mock_urlopen):
        """download_presigned creates parent directories if they do not exist."""
        content = b"some-data"
        mock_response = MagicMock()
        mock_response.read.side_effect = [content, b""]
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a nested path that does not yet exist
            local_path = os.path.join(tmpdir, "deep", "nested", "dir", "output.pdf")
            self.assertFalse(os.path.exists(os.path.dirname(local_path)))

            download_presigned("https://s3.example.com/key?get", local_path)

            self.assertTrue(os.path.isdir(os.path.dirname(local_path)))
            self.assertTrue(os.path.isfile(local_path))

    @patch("jobs.presigned_io._no_redirect_opener.open")
    def test_raises_on_http_error(self, mock_urlopen):
        """A failed bearer request reports status without exposing its URL."""
        mock_urlopen.side_effect = HTTPError(
            url="https://s3.example.com/key?get",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=BytesIO(b"Access Denied"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "output.pdf")
            observations = []
            with self.assertRaises(PresignedTransferError) as ctx:
                download_presigned(
                    "https://s3.example.com/key?get", local_path,
                    transfer_observer=lambda event, count: observations.append((event, count)),
                )

            self.assertEqual(str(ctx.exception), "presigned HTTP status 403")
            # File should not exist after error
            self.assertFalse(os.path.isfile(local_path))
            self.assertEqual(observations, [("get_attempt", 0)])


class TestUploadPresigned(SimpleTestCase):
    """Tests for upload_presigned() HTTP PUT helper."""

    def setUp(self):
        self._allowlist = patch.dict(
            os.environ,
            {
                "S3_PRESIGNED_ALLOWED_HOSTS": "s3.example.com",
                "S3_PRESIGNED_ALLOW_HTTP": "false",
                "S3_PRESIGNED_ALLOW_PRIVATE": "false",
            },
            clear=False,
        )
        self._allowlist.start()
        self.addCleanup(self._allowlist.stop)
        self._pin = patch(
            "jobs.presigned_io._pinned_presigned_url",
            side_effect=lambda url: nullcontext(url),
        )
        self._pin.start()
        self.addCleanup(self._pin.stop)

    @patch("jobs.presigned_io._no_redirect_opener.open")
    def test_uploads_file_as_streaming_body(self, mock_urlopen):
        """upload_presigned passes a readable body instead of materializing it."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "upload.pdf")
            file_content = b"PDF-content-to-upload"
            with open(local_path, "wb") as f:
                f.write(file_content)

            upload_presigned(local_path, "https://s3.example.com/key?put")

            # Verify urlopen was called
            mock_urlopen.assert_called_once()
            # Extract the Request object passed to urlopen
            call_args = mock_urlopen.call_args
            request_obj = call_args[0][0]
            self.assertTrue(callable(getattr(request_obj.data, "read", None)))
            self.assertEqual(request_obj.get_method(), "PUT")
            self.assertEqual(request_obj.full_url, "https://s3.example.com/key?put")

    @patch("jobs.presigned_io._no_redirect_opener.open")
    def test_sets_content_length_header(self, mock_urlopen):
        """upload_presigned sets Content-Length header matching the file size."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "upload.pdf")
            file_content = b"x" * 12345
            with open(local_path, "wb") as f:
                f.write(file_content)

            upload_presigned(local_path, "https://s3.example.com/key?put")

            # Extract the Request object and check headers
            call_args = mock_urlopen.call_args
            request_obj = call_args[0][0]
            content_length = request_obj.get_header("Content-length")
            self.assertEqual(content_length, str(len(file_content)))

    @patch("jobs.presigned_io._no_redirect_opener.open")
    def test_rejects_upload_over_transfer_cap(self, mock_urlopen):
        mock_urlopen.return_value = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "oversized.pdf")
            with open(local_path, "wb") as f:
                f.write(b"x" * 8)

            with patch.dict(
                os.environ,
                {"S3_PRESIGNED_MAX_TRANSFER_BYTES": "7"},
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    upload_presigned(local_path, "https://s3.example.com/key?put")
            mock_urlopen.assert_not_called()

    @patch("jobs.presigned_io._no_redirect_opener.open")
    def test_rejects_download_over_transfer_cap_and_removes_partial_file(
        self, mock_urlopen
    ):
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "8"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "oversized.pdf")
            with patch.dict(
                os.environ,
                {"S3_PRESIGNED_MAX_TRANSFER_BYTES": "7"},
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    download_presigned("https://s3.example.com/key?get", local_path)
            self.assertFalse(os.path.exists(local_path))

    @patch("jobs.presigned_io._no_redirect_opener.open")
    def test_download_respects_smaller_per_call_source_reservation(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "8"}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "source.pdf")
            with self.assertRaisesRegex(ValueError, "exceeds"):
                download_presigned(
                    "https://s3.example.com/key?get", local_path, max_bytes=7
                )
            self.assertFalse(os.path.exists(local_path))


class TestPresignedUrlBinding(SimpleTestCase):
    """R2-RC-01 residual: scheme/host/job-path binding for kwargs URLs."""

    def setUp(self):
        self._allowlist = patch.dict(
            os.environ,
            {
                "S3_PRESIGNED_ALLOWED_HOSTS": "s3.example.com",
                "S3_PRESIGNED_ALLOW_HTTP": "false",
                "S3_PRESIGNED_ALLOW_PRIVATE": "false",
            },
            clear=False,
        )
        self._allowlist.start()
        self.addCleanup(self._allowlist.stop)

    def test_rejects_non_http_scheme(self):
        with self.assertRaises(ValueError):
            _validate_presigned_url("file:///etc/passwd")

    def test_rejects_metadata_host(self):
        with self.assertRaises(ValueError):
            _validate_presigned_url("http://169.254.169.254/latest/meta-data/")

    def test_rejects_localhost_without_allowlist(self):
        with patch.dict(
            os.environ,
            {"S3_PRESIGNED_ALLOWED_HOSTS": "", "S3_ENDPOINT": ""},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                _validate_presigned_url("http://localhost/jobs/x/source/a.pdf")

    def test_allows_loopback_when_explicitly_allowlisted(self):
        with patch.dict(
            os.environ,
            {
                "S3_PRESIGNED_ALLOWED_HOSTS": "127.0.0.1:9000",
                "S3_PRESIGNED_ALLOW_HTTP": "true",
                "S3_PRESIGNED_ALLOW_PRIVATE": "true",
                "S3_ENDPOINT": "",
            },
            clear=False,
        ):
            _validate_presigned_url(
                "http://127.0.0.1:9000/bucket/jobs/j1/source/a.pdf",
                job_id="j1",
            )

    def test_job_path_bind_required(self):
        url = "https://s3.example.com/bucket/jobs/other-job/source/a.pdf"
        with self.assertRaises(ValueError) as ctx:
            _validate_presigned_url(url, job_id="job-abc")
        self.assertIn("not bound", str(ctx.exception))

    def test_job_path_bind_accepts_matching(self):
        url = "https://s3.example.com/bucket/jobs/job-abc/source/a.pdf?X-Amz=1"
        _validate_presigned_url(url, job_id="job-abc")  # no raise

    def test_allowlist_rejects_foreign_host(self):
        with patch.dict(
            os.environ,
            {"S3_PRESIGNED_ALLOWED_HOSTS": "minio.internal", "S3_ENDPOINT": ""},
            clear=False,
        ):
            with self.assertRaises(ValueError) as ctx:
                _validate_presigned_url(
                    "https://evil.example.com/jobs/j1/source/a.pdf",
                    job_id="j1",
                )
            self.assertIn("allowlist", str(ctx.exception))

    def test_empty_allowlist_fails_closed_in_single_tenant_mode(self):
        with patch.dict(
            os.environ,
            {
                "S3_PRESIGNED_ALLOWED_HOSTS": "",
                "S3_ENDPOINT": "",
                "ENABLE_MULTITENANCY": "false",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "allowlist is empty"):
                _validate_presigned_url(
                    "https://s3.example.com/jobs/j1/source/a.pdf",
                    job_id="j1",
                )

    def test_job_path_binding_uses_segments_and_rejects_parent_paths(self):
        with patch.dict(
            os.environ,
            {"S3_PRESIGNED_ALLOWED_HOSTS": "s3.example.com"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                _validate_presigned_url(
                    "https://s3.example.com/notjobs/j1/jobs/j10/file",
                    job_id="j1",
                )
            with self.assertRaisesRegex(ValueError, r"\.\."):
                _validate_presigned_url(
                    "https://s3.example.com/jobs/j1/../other/file",
                    job_id="j1",
                )

    def test_opener_has_real_redirect_and_proxy_guards(self):
        import urllib.request

        handler_types = {type(handler) for handler in _no_redirect_opener.handlers}
        self.assertIn(NoRedirectHandler, handler_types)
        self.assertNotIn(urllib.request.HTTPRedirectHandler, handler_types)
        # An empty ProxyHandler is consumed by build_opener as the explicit
        # replacement for the environment-backed default, so no proxy handler
        # remains in the final opener.
        self.assertFalse(
            any(
                isinstance(handler, urllib.request.ProxyHandler)
                for handler in _no_redirect_opener.handlers
            )
        )

    def test_allowlist_accepts_listed_host(self):
        with patch.dict(
            os.environ,
            {
                "S3_PRESIGNED_ALLOWED_HOSTS": "minio.internal",
                "S3_ENDPOINT": "",
            },
            clear=False,
        ):
            _validate_presigned_url(
                "https://minio.internal/bucket/jobs/j1/source/a.pdf",
                job_id="j1",
            )
