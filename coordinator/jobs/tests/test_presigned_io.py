"""Tests for Phase 7C presigned URL HTTP I/O helpers (worker side).

Tests the worker-side download/upload functions that use presigned URLs
with stdlib urllib -- no S3 credentials required.
"""

import os
import tempfile
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from django.test import SimpleTestCase

from jobs.presigned_io import download_presigned, upload_presigned


class TestDownloadPresigned(SimpleTestCase):
    """Tests for download_presigned() HTTP GET helper."""

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
            result = download_presigned("https://s3.example.com/key?get", local_path)

            self.assertEqual(result, local_path)
            self.assertTrue(os.path.isfile(local_path))
            with open(local_path, "rb") as f:
                self.assertEqual(f.read(), content)

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
        """download_presigned propagates HTTPError from urlopen (e.g., 403 expired)."""
        mock_urlopen.side_effect = HTTPError(
            url="https://s3.example.com/key?get",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=BytesIO(b"Access Denied"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "output.pdf")
            with self.assertRaises(HTTPError) as ctx:
                download_presigned("https://s3.example.com/key?get", local_path)

            self.assertEqual(ctx.exception.code, 403)
            # File should not exist after error
            self.assertFalse(os.path.isfile(local_path))


class TestUploadPresigned(SimpleTestCase):
    """Tests for upload_presigned() HTTP PUT helper."""

    @patch("jobs.presigned_io._no_redirect_opener.open")
    def test_uploads_file_data(self, mock_urlopen):
        """upload_presigned reads the local file and sends its content via PUT request."""
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
            self.assertEqual(request_obj.data, file_content)
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
