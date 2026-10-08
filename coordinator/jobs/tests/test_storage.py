"""Tests for Phase 7 storage backend abstraction."""

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

from botocore.exceptions import ClientError
from django.test import SimpleTestCase

from jobs.storage import (
    CachedS3Backend,
    NFSBackend,
    S3Backend,
    _BoundedReader,
    create_storage_backend,
)


def test_presigned_only_backend_never_constructs_sdk_or_accepts_sdk_io(tmp_path):
    backend = create_storage_backend(
        backend_name="s3", nfs_root=str(tmp_path), s3_endpoint="http://minio:9000",
        s3_bucket="lab", presigned_only_worker=True,
    )
    assert isinstance(backend, S3Backend)
    assert backend.backend_name == "s3"
    (tmp_path / "source.pdf").write_bytes(b"source")
    for operation in (
        lambda: backend.object_revision("jobs/j/source.pdf"),
        lambda: backend.download_file("jobs/j/source.pdf", str(tmp_path / "out.pdf")),
        lambda: backend.upload_file(str(tmp_path / "source.pdf"), "jobs/j/source.pdf"),
        lambda: backend.exists("jobs/j/source.pdf"),
        lambda: backend.presigned_url("jobs/j/source.pdf"),
    ):
        try:
            operation()
        except RuntimeError as exc:
            assert "forbidden in presigned-only worker" in str(exc)
        else:
            raise AssertionError("SDK operation unexpectedly succeeded")
    with tempfile.TemporaryDirectory() as cache_dir:
        cached = CachedS3Backend(inner=backend, cache_dir=cache_dir, max_size_bytes=1024 * 1024)
        assert cached._source_cache is not None


def test_presigned_only_backend_rejects_supplied_credentials_or_client(tmp_path):
    for overrides in ({"s3_access_key": "key"}, {"s3_secret_key": "secret"}, {"s3_client": object()}):
        try:
            create_storage_backend(backend_name="s3", nfs_root=str(tmp_path), presigned_only_worker=True, **overrides)
        except ValueError as exc:
            assert "must not receive S3 credentials" in str(exc)
        else:
            raise AssertionError("credentialed presigned-only backend unexpectedly created")


def test_s3_presigned_upload_uses_signature_v4_without_region():
    from urllib.parse import parse_qs, urlsplit

    backend = S3Backend(
        endpoint="http://127.0.0.1:9000",
        bucket="test-signing",
        access_key="local-test-key",
        secret_key="local-test-secret",
    )
    query = parse_qs(urlsplit(backend.presigned_upload_url("pages/1.txt")).query)
    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]


def test_iam_provider_refreshes_presigned_credentials_without_static_keys(monkeypatch, tmp_path):
    """Real botocore provider/signing against an owned loopback credential fixture."""
    import json
    import threading
    from datetime import datetime, timedelta, timezone
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlsplit

    import boto3

    state = {"generation": 1, "requests": 0}

    class Credentials(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"] += 1
            body = json.dumps({
                "AccessKeyId": "owned-fixture-access",
                "SecretAccessKey": "owned-fixture-secret",
                "Token": f"owned-fixture-token-{state['generation']}",
                "Expiration": (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    for name in tuple(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent-credentials"))
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Credentials)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", f"http://127.0.0.1:{server.server_port}/credentials")
        backend = S3Backend("", "owned-fixture-bucket", "", "", region="us-east-1", auth_mode="iam")
        before = parse_qs(urlsplit(backend.presigned_upload_url("jobs/j/page.txt")).query)
        # Exercise the actual worker policy on the SDK-generated URL too.
        from jobs.presigned_io import _validate_presigned_url

        signed_url = backend.presigned_upload_url("jobs/j/page.txt")
        monkeypatch.setenv("S3_PRESIGNED_ALLOWED_HOSTS", urlsplit(signed_url).hostname)
        _validate_presigned_url(signed_url, job_id="j", expected_key="jobs/j/page.txt")
        monkeypatch.setenv("S3_PRESIGNED_ALLOWED_HOSTS", "other-owned-fixture.s3.amazonaws.com")
        try:
            _validate_presigned_url(signed_url, job_id="j", expected_key="jobs/j/page.txt")
        except ValueError as exc:
            assert "not in allowlist" in str(exc)
        else:
            raise AssertionError("worker accepted a signed URL outside its configured endpoint")
        state["generation"] = 2
        after = parse_qs(urlsplit(backend.presigned_upload_url("jobs/j/page.txt")).query)
        assert before["X-Amz-Security-Token"] == ["owned-fixture-token-1"]
        assert after["X-Amz-Security-Token"] == ["owned-fixture-token-2"]
        assert before["X-Amz-Signature"] != after["X-Amz-Signature"]
        assert state["requests"] >= 2
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_explicit_temporary_s3_credentials_include_session_token():
    from urllib.parse import parse_qs, urlsplit

    backend = S3Backend("https://s3.example.test", "lab", "fixture-key", "fixture-secret",
                        session_token="fixture-session-token")
    query = parse_qs(urlsplit(backend.presigned_url("jobs/j/source.pdf")).query)
    assert query["X-Amz-Security-Token"] == ["fixture-session-token"]


def test_layoutlm_presigned_worker_storage_cannot_construct_sdk(monkeypatch, tmp_path):
    from django.test import override_settings

    from jobs import tasks_layoutlm

    with override_settings(STORAGE_BACKEND="s3", S3_AUTH_MODE="iam",
                           S3_PRESIGNED_ONLY_WORKER=True, S3_CACHE_DIR=str(tmp_path)):
        backend = tasks_layoutlm._get_storage_backend()
        try:
            backend.presigned_url("jobs/j/source.pdf")
        except RuntimeError as exc:
            assert "forbidden in presigned-only worker" in str(exc)
        else:
            raise AssertionError("LayoutLM worker unexpectedly reached the SDK")


class TestNFSBackend(SimpleTestCase):
    """Tests for NFS-backed storage behavior."""

    def test_upload_download_delete_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = NFSBackend(root=tmpdir)

            src_path = os.path.join(tmpdir, "src.txt")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write("hello-storage")

            key = "jobs/abc/output/test.txt"
            uploaded_path = backend.upload_file(src_path, key)
            assert uploaded_path.endswith(
                "jobs\\abc\\output\\test.txt"
            ) or uploaded_path.endswith("jobs/abc/output/test.txt")
            assert backend.exists(key)

            download_path = os.path.join(tmpdir, "download", "test.txt")
            backend.download_file(key, download_path)
            assert os.path.isfile(download_path)
            with open(download_path, "r", encoding="utf-8") as f:
                assert f.read() == "hello-storage"

            backend.delete(key)
            assert backend.exists(key) is False

    def test_delete_is_noop_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = NFSBackend(root=tmpdir)
            backend.delete("jobs/does-not-exist.txt")
            assert backend.exists("jobs/does-not-exist.txt") is False

    def test_path_jail_rejects_sibling_prefix_escape(self):
        """A root named ``root`` must not authorize sibling ``root_evil``."""
        with tempfile.TemporaryDirectory() as parent:
            root = os.path.join(parent, "root")
            sibling = os.path.join(parent, "root_evil")
            os.makedirs(root)
            os.makedirs(sibling)
            backend = NFSBackend(root=root)

            with self.assertRaises(ValueError):
                backend.to_absolute_path("../root_evil/secret.txt")

    def test_path_jail_rejects_symlink_escape(self):
        """A key traversing an in-root symlink must not reach outside root."""
        with tempfile.TemporaryDirectory() as parent:
            root = os.path.join(parent, "root")
            outside = os.path.join(parent, "outside")
            os.makedirs(root)
            os.makedirs(outside)
            os.symlink(outside, os.path.join(root, "linked"))
            backend = NFSBackend(root=root)

            with self.assertRaises(ValueError):
                backend.to_absolute_path("linked/secret.txt")

    def test_path_jail_rejects_in_root_cross_job_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = NFSBackend(root=tmpdir)
            os.makedirs(os.path.join(tmpdir, "jobs", "job-b"))
            os.symlink(
                os.path.join(tmpdir, "jobs", "job-b"),
                os.path.join(tmpdir, "jobs", "job-a"),
            )
            with self.assertRaises(ValueError):
                backend.to_absolute_path("jobs/job-a/source.pdf")

    def test_upload_rejects_hard_linked_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = NFSBackend(root=tmpdir)
            source = os.path.join(tmpdir, "source.txt")
            alias = os.path.join(tmpdir, "alias.txt")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write("secret")
            os.link(source, alias)
            with self.assertRaisesRegex(RuntimeError, "multiple hard links"):
                backend.upload_file(alias, "jobs/job-a/source.txt")

    def test_download_rejects_existing_destination_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = NFSBackend(root=tmpdir)
            source = backend.to_absolute_path("jobs/job-a/source.txt")
            os.makedirs(os.path.dirname(source), exist_ok=True)
            with open(source, "w", encoding="utf-8") as handle:
                handle.write("secret")
            outside = os.path.join(tmpdir, "outside.txt")
            with open(outside, "w", encoding="utf-8") as handle:
                handle.write("keep")
            destination = os.path.join(tmpdir, "download.txt")
            os.symlink(outside, destination)
            with self.assertRaisesRegex(RuntimeError, "copy target"):
                backend.download_file("jobs/job-a/source.txt", destination)
            with open(outside, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "keep")

    def test_open_key_reopens_current_regular_file_without_following_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = NFSBackend(root=tmpdir)
            path = os.path.join(tmpdir, "jobs", "job-a", "artifact.txt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(b"bound")

            with backend.open_key("jobs/job-a/artifact.txt") as handle:
                self.assertEqual(handle.read(), b"bound")

            outside = os.path.join(tmpdir, "outside.txt")
            with open(outside, "wb") as handle:
                handle.write(b"outside")
            os.unlink(path)
            try:
                os.symlink(outside, path)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable on this platform")
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                with backend.open_key("jobs/job-a/artifact.txt"):
                    pass

    def test_download_file_allows_basename_target_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = NFSBackend(root=tmpdir)
            key = "jobs/abc/source.txt"
            source = backend.to_absolute_path(key)
            os.makedirs(os.path.dirname(source), exist_ok=True)
            with open(source, "w", encoding="utf-8") as f:
                f.write("basename-target")

            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                backend.download_file(key, "download.txt")
                assert os.path.isfile(os.path.join(tmpdir, "download.txt"))
                with open(
                    os.path.join(tmpdir, "download.txt"), "r", encoding="utf-8"
                ) as f:
                    assert f.read() == "basename-target"
            finally:
                os.chdir(cwd)

    def test_list_objects_returns_all_files_under_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = NFSBackend(root=tmpdir)

            # Create test files
            for key in [
                "jobs/1/source.pdf",
                "jobs/1/temp/page1.pdf",
                "jobs/2/source.pdf",
            ]:
                target = backend.to_absolute_path(key)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "w", encoding="utf-8") as f:
                    f.write("test")

            # List jobs/1
            keys = backend.list_objects("jobs/1")
            assert len(keys) == 2
            assert "jobs/1/source.pdf" in keys
            assert "jobs/1/temp/page1.pdf" in keys

            # List all jobs
            keys = backend.list_objects("jobs")
            assert len(keys) == 3

    def test_list_objects_returns_empty_for_missing_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = NFSBackend(root=tmpdir)
            keys = backend.list_objects("jobs/nonexistent")
            assert keys == []


class TestS3Backend(SimpleTestCase):
    """Tests for S3-compatible backend behavior."""

    def _backend(self):
        client = MagicMock()
        backend = S3Backend(
            endpoint="http://s3.local",
            bucket="ocr-local",
            access_key="key",
            secret_key="secret",
            region="us-east-1",
            client=client,
        )
        return backend, client

    def test_upload_download_delete(self):
        backend, client = self._backend()
        source = tempfile.NamedTemporaryFile(delete=False)
        try:
            source.write(b"source")
            source.close()

            locator = backend.upload_file(source.name, "jobs/1/source.pdf")
            assert locator == "s3://ocr-local/jobs/1/source.pdf"
            client.upload_fileobj.assert_called_once()
            assert client.upload_fileobj.call_args.args[1:] == (
                "ocr-local",
                "jobs/1/source.pdf",
            )

            backend.download_file("jobs/1/source.pdf", "/tmp/output/source.pdf")
            client.download_fileobj.assert_called_once()
            assert client.download_fileobj.call_args.args[0:2] == (
                "ocr-local",
                "jobs/1/source.pdf",
            )

            backend.delete("jobs/1/source.pdf")
            client.delete_object.assert_called_once_with(
                Bucket="ocr-local", Key="jobs/1/source.pdf"
            )
        finally:
            os.unlink(source.name)

    def test_upload_enforces_transfer_cap(self):
        """A credentialed S3 upload rejects an oversized source before SDK I/O."""
        backend, client = self._backend()
        previous = os.environ.get("S3_STORAGE_MAX_TRANSFER_BYTES")
        os.environ["S3_STORAGE_MAX_TRANSFER_BYTES"] = "7"
        try:
            source = tempfile.NamedTemporaryFile(delete=False)
            try:
                source.write(b"12345678")
                source.close()
                with self.assertRaisesRegex(
                    ValueError, "S3_STORAGE_MAX_TRANSFER_BYTES"
                ):
                    backend.upload_file(source.name, "jobs/1/source.pdf")
                client.upload_fileobj.assert_not_called()
            finally:
                os.unlink(source.name)
        finally:
            if previous is None:
                os.environ.pop("S3_STORAGE_MAX_TRANSFER_BYTES", None)
            else:
                os.environ["S3_STORAGE_MAX_TRANSFER_BYTES"] = previous

    def test_download_allows_basename_target_path(self):
        backend, client = self._backend()

        try:
            backend.download_file("jobs/1/source.pdf", "download.pdf")
        finally:
            if os.path.exists("download.pdf"):
                os.unlink("download.pdf")

        client.download_fileobj.assert_called_once()
        assert client.download_fileobj.call_args.args[0:2] == (
            "ocr-local",
            "jobs/1/source.pdf",
        )

    def test_download_enforces_transfer_cap(self):
        """A credentialed S3 stream cannot write beyond the local safety cap."""

        class StreamingClient:
            def download_fileobj(self, _bucket, _key, destination):
                destination.write(b"12345678")

        backend = S3Backend(
            endpoint="http://s3.local",
            bucket="ocr-local",
            access_key="key",
            secret_key="secret",
            client=StreamingClient(),
        )
        previous = os.environ.get("S3_STORAGE_MAX_TRANSFER_BYTES")
        os.environ["S3_STORAGE_MAX_TRANSFER_BYTES"] = "7"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(
                    ValueError, "S3_STORAGE_MAX_TRANSFER_BYTES"
                ):
                    backend.download_file(
                        "jobs/1/source.pdf",
                        os.path.join(tmpdir, "source.pdf"),
                    )
        finally:
            if previous is None:
                os.environ.pop("S3_STORAGE_MAX_TRANSFER_BYTES", None)
            else:
                os.environ["S3_STORAGE_MAX_TRANSFER_BYTES"] = previous

    def test_download_enforces_per_call_cap_before_disk_write(self):
        class StreamingClient:
            def download_fileobj(self, _bucket, _key, destination):
                destination.write(b"12345678")

        backend = S3Backend(
            endpoint="http://s3.local",
            bucket="ocr-local",
            access_key="key",
            secret_key="secret",
            client=StreamingClient(),
        )
        previous = os.environ.get("S3_STORAGE_MAX_TRANSFER_BYTES")
        os.environ["S3_STORAGE_MAX_TRANSFER_BYTES"] = "1024"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                target = os.path.join(tmpdir, "bundle.txt")
                with self.assertRaisesRegex(ValueError, "S3_STORAGE_MAX_TRANSFER_BYTES"):
                    backend.download_file(
                        "jobs/1/text.txt", target, max_bytes=7
                    )
                assert not os.path.exists(target)
                assert os.listdir(tmpdir) == []
        finally:
            if previous is None:
                os.environ.pop("S3_STORAGE_MAX_TRANSFER_BYTES", None)
            else:
                os.environ["S3_STORAGE_MAX_TRANSFER_BYTES"] = previous

    def test_exists_true_when_head_object_succeeds(self):
        backend, client = self._backend()
        client.head_object.return_value = {"ETag": "abc"}

        assert backend.exists("jobs/1/file.pdf") is True
        client.head_object.assert_called_once_with(
            Bucket="ocr-local", Key="jobs/1/file.pdf"
        )

    def test_exists_false_when_head_object_raises(self):
        backend, client = self._backend()
        client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )

        assert backend.exists("jobs/1/file.pdf") is False

    def test_exists_propagates_non_service_errors(self):
        backend, client = self._backend()
        client.head_object.side_effect = RuntimeError("network failure")

        with self.assertRaisesRegex(RuntimeError, "network failure"):
            backend.exists("jobs/1/file.pdf")

    def test_presigned_url(self):
        backend, client = self._backend()
        client.generate_presigned_url.return_value = "https://example/presigned"

        url = backend.presigned_url("jobs/1/file.pdf", expires=120)
        assert url == "https://example/presigned"
        client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "ocr-local", "Key": "jobs/1/file.pdf"},
            ExpiresIn=120,
        )

    def test_list_objects_with_pagination(self):
        backend, client = self._backend()

        # Mock paginator
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "jobs/1/file1.pdf"}, {"Key": "jobs/1/file2.pdf"}]},
            {"Contents": [{"Key": "jobs/1/file3.pdf"}]},
        ]
        client.get_paginator.return_value = mock_paginator

        keys = backend.list_objects("jobs/1")

        assert keys == ["jobs/1/file1.pdf", "jobs/1/file2.pdf", "jobs/1/file3.pdf"]
        client.get_paginator.assert_called_once_with("list_objects_v2")
        mock_paginator.paginate.assert_called_once_with(
            Bucket="ocr-local", Prefix="jobs/1"
        )

    def test_list_objects_empty_result(self):
        backend, client = self._backend()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": []}]
        client.get_paginator.return_value = mock_paginator

        keys = backend.list_objects("jobs/nonexistent")
        assert keys == []

    def test_list_objects_accepts_directory_style_trailing_slash(self):
        backend, client = self._backend()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": []}]
        client.get_paginator.return_value = mock_paginator

        assert backend.list_objects("jobs/1/") == []
        mock_paginator.paginate.assert_called_once_with(
            Bucket="ocr-local", Prefix="jobs/1/"
        )

    def test_key_and_prefix_traversal_are_rejected(self):
        backend, _ = self._backend()

        for key in ("jobs/../secret", "jobs//secret", "/jobs/secret", "jobs\\secret"):
            with self.assertRaises(ValueError):
                backend._validate_key(key)
        for prefix in ("jobs/../", "jobs//", "/jobs/", "jobs\\"):
            with self.assertRaises(ValueError):
                backend.list_objects(prefix)


class TestCreateStorageBackend(SimpleTestCase):
    """Tests backend factory behavior."""

    def test_create_nfs_backend(self):
        backend = create_storage_backend(
            backend_name="nfs",
            nfs_root="/shared",
        )
        assert isinstance(backend, NFSBackend)
        assert backend.backend_name == "nfs"

    def test_create_s3_backend(self):
        mock_client = MagicMock()
        backend = create_storage_backend(
            backend_name="s3",
            nfs_root="/shared",
            s3_endpoint="http://s3.local",
            s3_bucket="ocr-local",
            s3_access_key="key",
            s3_secret_key="secret",
            s3_client=mock_client,
        )
        assert isinstance(backend, S3Backend)
        assert backend.backend_name == "s3"

    def test_unsupported_backend_raises(self):
        try:
            create_storage_backend(
                backend_name="invalid",
                nfs_root="/shared",
            )
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "Unsupported storage backend" in str(exc)


class TestS3BatchDelete(SimpleTestCase):
    """Tests for S3Backend.delete_many batch delete functionality."""

    def _backend(self):
        client = MagicMock()
        backend = S3Backend(
            endpoint="http://s3.local",
            bucket="ocr-local",
            access_key="key",
            secret_key="secret",
            region="us-east-1",
            client=client,
        )
        return backend, client

    def test_batch_delete_calls_delete_objects_api(self):
        """delete_many should call client.delete_objects with correct Objects list and Quiet: True."""
        backend, client = self._backend()
        client.delete_objects.return_value = {"Errors": []}

        result = backend.delete_many(["k1", "k2", "k3"])

        client.delete_objects.assert_called_once_with(
            Bucket="ocr-local",
            Delete={
                "Objects": [{"Key": "k1"}, {"Key": "k2"}, {"Key": "k3"}],
                "Quiet": True,
            },
        )
        assert result == 3

    def test_batch_delete_handles_over_1000_keys(self):
        """delete_many should split keys into batches of 1000."""
        backend, client = self._backend()
        client.delete_objects.return_value = {"Errors": []}

        keys = [f"key-{i}" for i in range(1500)]
        result = backend.delete_many(keys)

        assert client.delete_objects.call_count == 2
        # First batch should have 1000 keys
        first_call_objects = client.delete_objects.call_args_list[0][1]["Delete"][
            "Objects"
        ]
        assert len(first_call_objects) == 1000
        # Second batch should have 500 keys
        second_call_objects = client.delete_objects.call_args_list[1][1]["Delete"][
            "Objects"
        ]
        assert len(second_call_objects) == 500
        assert result == 1500

    def test_batch_delete_returns_correct_count(self):
        """Per-key S3 delete errors must fail closed, not return a partial count."""
        backend, client = self._backend()
        client.delete_objects.return_value = {
            "Errors": [
                {"Key": "k2", "Code": "AccessDenied", "Message": "Denied"},
                {"Key": "k4", "Code": "InternalError", "Message": "Error"},
            ]
        }

        with self.assertRaisesRegex(RuntimeError, "per-key errors"):
            backend.delete_many(["k1", "k2", "k3", "k4", "k5"])

    def test_batch_delete_falls_back_on_api_error(self):
        """When delete_objects raises, should fall back to sequential delete calls."""
        backend, client = self._backend()
        client.delete_objects.side_effect = RuntimeError("S3 API error")

        keys = ["k1", "k2", "k3"]
        result = backend.delete_many(keys)

        # Should have attempted delete_objects once
        client.delete_objects.assert_called_once()
        # Should fall back to sequential delete_object calls
        assert client.delete_object.call_count == 3
        client.delete_object.assert_any_call(Bucket="ocr-local", Key="k1")
        client.delete_object.assert_any_call(Bucket="ocr-local", Key="k2")
        client.delete_object.assert_any_call(Bucket="ocr-local", Key="k3")
        assert result == 3

    def test_batch_delete_empty_list_returns_zero(self):
        """delete_many with empty list should return 0 with no API calls."""
        backend, client = self._backend()

        result = backend.delete_many([])

        assert result == 0
        client.delete_objects.assert_not_called()
        client.delete_object.assert_not_called()

    def test_nfs_batch_delete_uses_sequential_fallback(self):
        """NFSBackend.delete_many should use the base class sequential fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = NFSBackend(root=tmpdir)

            # Create test files
            keys = []
            for i in range(3):
                key = f"jobs/test/file{i}.txt"
                keys.append(key)
                target = backend.to_absolute_path(key)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "w", encoding="utf-8") as f:
                    f.write(f"content-{i}")

            # Verify files exist
            for key in keys:
                assert backend.exists(key)

            result = backend.delete_many(keys)

            assert result == 3
            for key in keys:
                assert backend.exists(key) is False


class TestCachedS3Backend(SimpleTestCase):
    """Tests for CachedS3Backend worker-local caching behavior."""

    def test_source_object_limit_respects_configured_cache_bound(self):
        inner = MagicMock(spec=S3Backend)
        with tempfile.TemporaryDirectory() as cache_dir:
            cached = CachedS3Backend(
                inner=inner, cache_dir=cache_dir,
                max_size_bytes=1024, max_object_bytes=128,
            )
            assert cached._source_cache.max_size_bytes == 512
            assert cached._source_cache.max_object_bytes == 128

    def test_default_budget_is_one_gib_but_explicit_encrypted_pvc_profile_can_expand(self):
        inner = MagicMock(spec=S3Backend)
        with tempfile.TemporaryDirectory() as cache_dir:
            default = CachedS3Backend(inner=inner, cache_dir=cache_dir)
            assert default.max_size_bytes == 1024**3
            assert default._source_cache.max_object_bytes == 512 * 1024**2
            expanded = CachedS3Backend(
                inner=inner, cache_dir=cache_dir,
                max_size_bytes=8 * 1024**3,
                max_object_bytes=4 * 1024**3,
            )
            assert expanded._source_cache.max_size_bytes == 4 * 1024**3
            assert expanded._source_cache.max_object_bytes == 4 * 1024**3

    def _make_inner(self):
        """Create a mock S3Backend for use as the inner backend."""
        inner = MagicMock(spec=S3Backend)
        inner.backend_name = "s3"
        inner.endpoint = "http://s3.local"
        inner.bucket = "ocr-local"
        inner.region = "us-east-1"
        inner.object_revision.return_value = ("v1", '"etag-one"', 18)
        inner.delete_many.return_value = 3
        return inner

    def test_cache_miss_downloads_and_populates(self):
        """On cache miss, should download from inner and populate cache file."""
        inner = self._make_inner()

        def _fake_download(bucket_or_key, key_or_local, local_path=None):
            # inner.download_file(key, cache_tmp) writes to local_path
            target = key_or_local if local_path is None else local_path
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                f.write(b"downloaded-content")
            return target

        inner.download_file.side_effect = _fake_download

        with (
            tempfile.TemporaryDirectory() as cache_dir,
            tempfile.TemporaryDirectory() as target_dir,
        ):
            cached = CachedS3Backend(inner=inner, cache_dir=cache_dir)

            target_path = os.path.join(target_dir, "output.pdf")
            cached.download_file("jobs/1/source.pdf", target_path)

            # Inner download should have been called
            inner.download_file.assert_called_once()
            # Target file should exist with correct content
            assert os.path.isfile(target_path)
            with open(target_path, "rb") as f:
                assert f.read() == b"downloaded-content"

    def test_cache_hit_enforces_per_call_download_cap(self):
        inner = self._make_inner()
        with (
            tempfile.TemporaryDirectory() as cache_dir,
            tempfile.TemporaryDirectory() as target_dir,
        ):
            cached = CachedS3Backend(inner=inner, cache_dir=cache_dir)
            cache_path = cached._cache_path("jobs/1/source.pdf", inner.object_revision())
            with open(cache_path, "wb") as handle:
                handle.write(b"too-large")
            target = os.path.join(target_dir, "bundle.txt")

            with self.assertRaisesRegex(ValueError, "max_bytes"):
                cached.download_file("jobs/1/source.pdf", target, max_bytes=4)

            assert not os.path.exists(target)
            inner.download_file.assert_not_called()

    def test_cache_miss_passes_per_call_download_cap(self):
        inner = self._make_inner()
        inner.object_revision.return_value = ("v1", '"etag-small"', 5)

        def bounded_download(_key, local_path, *, max_bytes=None):
            assert max_bytes == 8
            with open(local_path, "wb") as handle:
                handle.write(b"small")
            return local_path

        inner.download_file.side_effect = bounded_download
        with (
            tempfile.TemporaryDirectory() as cache_dir,
            tempfile.TemporaryDirectory() as target_dir,
        ):
            cached = CachedS3Backend(inner=inner, cache_dir=cache_dir)
            target = os.path.join(target_dir, "bundle.txt")

            cached.download_file("jobs/1/text.txt", target, max_bytes=8)

            inner.download_file.assert_called_once()
            assert Path(target).read_bytes() == b"small"

    def test_cache_hit_avoids_download(self):
        """On cache hit, should copy from cache without calling inner."""
        inner = self._make_inner()
        inner.object_revision.return_value = (
            "v1", '"etag-cached"', len(b"cached-content")
        )

        with (
            tempfile.TemporaryDirectory() as cache_dir,
            tempfile.TemporaryDirectory() as target_dir,
        ):
            cached = CachedS3Backend(inner=inner, cache_dir=cache_dir)

            # Pre-populate cache entry for the key
            cache_path = cached._cache_path("jobs/1/source.pdf", inner.object_revision())
            with open(cache_path, "wb") as f:
                f.write(b"cached-content")

            target_path = os.path.join(target_dir, "output.pdf")
            cached.download_file("jobs/1/source.pdf", target_path)

            # Inner download should NOT have been called
            inner.download_file.assert_not_called()
            # Target file should have cached content
            assert os.path.isfile(target_path)
            with open(target_path, "rb") as f:
                assert f.read() == b"cached-content"

    def test_same_key_misses_share_one_temp_download(self):
        """Concurrent misses must not race on the shared cache temp path."""
        inner = self._make_inner()
        inner.object_revision.return_value = ("v1", '"etag-consistent"', len(b"one-consistent-value"))

        def _fake_download(_key, local_path):
            time.sleep(0.02)
            with open(local_path, "wb") as handle:
                handle.write(b"one-consistent-value")
            return local_path

        inner.download_file.side_effect = _fake_download
        with (
            tempfile.TemporaryDirectory() as cache_dir,
            tempfile.TemporaryDirectory() as target_dir,
        ):
            cached = CachedS3Backend(inner=inner, cache_dir=cache_dir)
            targets = [os.path.join(target_dir, f"out-{i}.pdf") for i in range(2)]
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(
                    pool.map(
                        lambda path: cached.download_file("same-key", path), targets
                    )
                )

            assert inner.download_file.call_count == 1
            assert [Path(path).read_bytes() for path in targets] == [
                b"one-consistent-value",
                b"one-consistent-value",
            ]

    def test_lru_eviction_removes_oldest(self):
        """When cache exceeds max_size_bytes, oldest files should be evicted."""
        inner = self._make_inner()

        with tempfile.TemporaryDirectory() as cache_dir:
            # Total budget is split between source leases and artifacts;
            # artifact half is 100 bytes, so 50 + 60 triggers eviction.
            cached = CachedS3Backend(
                inner=inner, cache_dir=cache_dir, max_size_bytes=200
            )

            # Create cache files with different ages and sizes
            old_file = os.path.join(cache_dir, "old_entry")
            with open(old_file, "wb") as f:
                f.write(b"x" * 50)
            # Set old mtime
            old_mtime = time.time() - 3600
            os.utime(old_file, (old_mtime, old_mtime))

            new_file = os.path.join(cache_dir, "new_entry")
            with open(new_file, "wb") as f:
                f.write(b"y" * 60)

            # Total = 110 bytes > max_size_bytes of 100
            # Eviction should run and remove oldest until below 80% (80 bytes)
            cached._maybe_evict()

            # Old file should be evicted
            assert not os.path.exists(old_file)
            # New file should remain (60 bytes < 80 byte target)
            assert os.path.exists(new_file)

    def test_delete_removes_cache_entry(self):
        """delete should remove cache file and delegate to inner.delete."""
        inner = self._make_inner()

        with tempfile.TemporaryDirectory() as cache_dir:
            cached = CachedS3Backend(inner=inner, cache_dir=cache_dir)

            # Pre-populate cache
            cache_path = cached._cache_path("jobs/1/file.pdf", inner.object_revision())
            with open(cache_path, "wb") as f:
                f.write(b"cached")

            assert os.path.exists(cache_path)

            cached.delete("jobs/1/file.pdf")

            # Cache entry should be removed
            assert not os.path.exists(cache_path)
            # Inner delete should be called
            inner.delete.assert_called_once_with("jobs/1/file.pdf")

    def test_delete_many_cleans_cache(self):
        """delete_many should remove cache entries and delegate to inner.delete_many."""
        inner = self._make_inner()
        inner.delete_many.return_value = 2

        with tempfile.TemporaryDirectory() as cache_dir:
            cached = CachedS3Backend(inner=inner, cache_dir=cache_dir)

            # Pre-populate cache entries
            keys = ["jobs/1/a.pdf", "jobs/1/b.pdf"]
            cache_paths = []
            for key in keys:
                cache_path = cached._cache_path(key, inner.object_revision())
                with open(cache_path, "wb") as f:
                    f.write(b"cached")
                cache_paths.append(cache_path)

            result = cached.delete_many(keys)

            # Cache entries should be removed
            for cp in cache_paths:
                assert not os.path.exists(cp)
            # Inner delete_many should be called
            inner.delete_many.assert_called_once_with(keys)
            assert result == 2

    def test_backend_name_delegates_to_inner(self):
        """backend_name should return the inner backend's name."""
        inner = self._make_inner()
        inner.backend_name = "s3"

        with tempfile.TemporaryDirectory() as cache_dir:
            cached = CachedS3Backend(inner=inner, cache_dir=cache_dir)
            assert cached.backend_name == "s3"

    def test_partial_download_cleanup(self):
        """On download failure, .tmp file should be cleaned up."""
        inner = self._make_inner()
        inner.download_file.side_effect = RuntimeError("network error")

        with (
            tempfile.TemporaryDirectory() as cache_dir,
            tempfile.TemporaryDirectory() as target_dir,
        ):
            cached = CachedS3Backend(inner=inner, cache_dir=cache_dir)

            target_path = os.path.join(target_dir, "output.pdf")

            try:
                cached.download_file("jobs/1/source.pdf", target_path)
                assert False, "Expected RuntimeError"
            except RuntimeError:
                pass

            # No .tmp files should remain in the cache directory
            for fname in os.listdir(cache_dir):
                assert not fname.endswith(".tmp"), f"Found leftover .tmp file: {fname}"

            # Cache file should not exist
            cache_path = cached._cache_path("jobs/1/source.pdf", inner.object_revision())
            assert not os.path.exists(cache_path)


def test_cache_copy_rejects_same_inode_mutation_without_publishing(tmp_path, monkeypatch):
    """A bounded cached read must never publish a mixture of two versions."""
    inner = MagicMock()
    inner.backend_name = "s3"
    inner.endpoint, inner.bucket, inner.region = "local", "bucket", "region"
    cache = CachedS3Backend(inner, str(tmp_path / "cache"))
    key = "jobs/example/output.txt"
    original = b"A" * (2 * 1024 * 1024)
    inner.object_revision.return_value = ("v1", '"etag-one"', len(original))
    cached = Path(cache._cache_path(key, inner.object_revision()))
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(original)
    target = tmp_path / "result.txt"
    target.write_bytes(b"previous-good-artifact")
    real_fdopen = os.fdopen
    changed = False

    class RacingReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size=-1):
            nonlocal changed
            result = self.stream.read(size)
            if not changed:
                changed = True
                cached.write_bytes(b"B" * len(original))
            return result

    def racing_open(fd, mode, *args, **kwargs):
        stream = real_fdopen(fd, mode, *args, **kwargs)
        return RacingReader(stream) if mode == "rb" else stream

    monkeypatch.setattr("jobs.storage.os.fdopen", racing_open)
    import pytest

    with pytest.raises(RuntimeError, match="changed during read"):
        cache.download_file(key, str(target), max_bytes=len(original))
    assert target.read_bytes() == b"previous-good-artifact"
    assert not list(tmp_path.glob(".ocr-copy-*"))


def test_independent_cache_instances_serialize_shared_publication(tmp_path):
    """Distinct worker instances cannot replace an inode during another read."""
    from threading import Barrier

    inner = MagicMock()
    inner.backend_name = "s3"
    inner.endpoint, inner.bucket, inner.region = "local", "bucket", "region"
    payload = b"immutable-cache-payload" * 100000
    inner.object_revision.return_value = ("v1", '"etag-one"', len(payload))

    def download(key, target):
        time.sleep(0.02)
        Path(target).write_bytes(payload)

    inner.download_file.side_effect = download
    caches = [CachedS3Backend(inner, str(tmp_path / "cache")) for _ in range(2)]
    barrier = Barrier(2)

    def read(number):
        barrier.wait(timeout=5)
        target = tmp_path / f"output-{number}.txt"
        caches[number].download_file("same-key", str(target))
        return target.read_bytes()

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(read, range(2))) == [payload, payload]
    assert inner.download_file.call_count == 1
    # Eviction must preserve the inode that coordinates every worker.
    caches[0].max_size_bytes = 1
    caches[0]._maybe_evict()
    assert (tmp_path / "cache" / ".cache-access.lock").is_file()


class MutableS3Client:
    """Small remote store with observable HEAD and download operations."""

    def __init__(self, data: bytes, revision: str = "v1"):
        self.data = data
        self.revision = revision
        self.head_error = None
        self.on_download = None
        self.downloads = 0

    def head_object(self, *, Bucket, Key):
        if self.head_error:
            raise self.head_error
        return {
            "VersionId": self.revision,
            "ETag": f'"{self.revision}"',
            "ContentLength": len(self.data),
        }

    def download_fileobj(self, bucket, key, destination):
        self.downloads += 1
        destination.write(self.data)
        if self.on_download:
            self.on_download()


def _cached_remote(tmp_path, *, endpoint, bucket, client):
    inner = S3Backend(endpoint, bucket, "key", "secret", "region", client)
    return CachedS3Backend(inner, str(tmp_path / "cache"))


def test_cache_separates_remote_namespaces_with_same_key(tmp_path):
    key = "jobs/one/page-1.txt"
    a = MutableS3Client(b"bucket-a")
    b = MutableS3Client(b"bucket-b")
    cache_a = _cached_remote(tmp_path, endpoint="http://a", bucket="shared", client=a)
    cache_b = _cached_remote(tmp_path, endpoint="http://b", bucket="shared", client=b)
    cache_c = _cached_remote(tmp_path, endpoint="http://a", bucket="other", client=b)

    assert len({cache._cache_path(key) for cache in (cache_a, cache_b, cache_c)}) == 3
    for number, cache in enumerate((cache_a, cache_b, cache_c)):
        cache.download_file(key, str(tmp_path / f"result-{number}.txt"))
    assert (tmp_path / "result-0.txt").read_bytes() == b"bucket-a"
    assert (tmp_path / "result-1.txt").read_bytes() == b"bucket-b"
    assert (tmp_path / "result-2.txt").read_bytes() == b"bucket-b"


def test_remote_overwrite_is_not_hidden_by_cached_page_bytes(tmp_path):
    client = MutableS3Client(b"old-page")
    cache = _cached_remote(tmp_path, endpoint="http://a", bucket="bucket", client=client)
    key = "jobs/one/pages/1.txt"
    target = tmp_path / "assembled-page.txt"
    cache.download_file(key, str(target))
    assert target.read_bytes() == b"old-page"

    # The replacement bypasses this worker's upload/invalidate methods.
    client.data, client.revision = b"new-page", "v2"
    cache.download_file(key, str(target))
    assert target.read_bytes() == b"new-page"
    assert client.downloads == 2
    cache.download_file(key, str(target))
    assert client.downloads == 2


def test_version_id_detects_same_content_aba(tmp_path):
    client = MutableS3Client(b"same-page")
    client.head_object = lambda **_kwargs: {
        "VersionId": client.revision,
        "ETag": '"same-etag"',
        "ContentLength": len(client.data),
    }
    cache = _cached_remote(tmp_path, endpoint="http://a", bucket="bucket", client=client)
    target = tmp_path / "assembled-page.txt"
    key = "jobs/one/pages/1.txt"
    cache.download_file(key, str(target))
    client.revision = "v2"
    cache.download_file(key, str(target))
    assert target.read_bytes() == b"same-page"
    assert client.downloads == 2


def test_head_failure_or_incomplete_metadata_cannot_serve_stale_cache(tmp_path):
    import pytest

    client = MutableS3Client(b"old-page")
    cache = _cached_remote(tmp_path, endpoint="http://a", bucket="bucket", client=client)
    key = "jobs/one/pages/1.txt"
    target = tmp_path / "assembled-page.txt"
    cache.download_file(key, str(target))
    client.head_error = RuntimeError("HEAD unavailable")
    with pytest.raises(RuntimeError, match="HEAD unavailable"):
        cache.download_file(key, str(target))
    assert target.read_bytes() == b"old-page"
    client.head_error = None
    client.head_object = lambda **_kwargs: {"ContentLength": 8}
    with pytest.raises(RuntimeError, match="revision metadata"):
        cache.download_file(key, str(target))


def test_remote_change_during_download_is_never_published(tmp_path):
    import pytest

    client = MutableS3Client(b"old-page")
    cache = _cached_remote(tmp_path, endpoint="http://a", bucket="bucket", client=client)
    target = tmp_path / "assembled-page.txt"
    target.write_bytes(b"previous-output")
    client.on_download = lambda: setattr(client, "revision", "v2")
    with pytest.raises(RuntimeError, match="revision changed"):
        cache.download_file("jobs/one/pages/1.txt", str(target))
    assert target.read_bytes() == b"previous-output"


def test_bounded_reader_caps_underlying_unbounded_reads():
    class RecordingSource(BytesIO):
        def __init__(self, data):
            super().__init__(data)
            self.sizes = []

        def read(self, size=-1):
            self.sizes.append(size)
            return super().read(size)

    import pytest

    source = RecordingSource(b"123456789" * 100)
    reader = _BoundedReader(source, 7)
    with pytest.raises(ValueError, match="S3_STORAGE_MAX_TRANSFER_BYTES"):
        reader.read()
    assert source.sizes == [8]


def test_upload_rechecks_file_growth_after_fstat(tmp_path, monkeypatch):
    import pytest

    source = tmp_path / "source.txt"
    source.write_bytes(b"1234")

    class GrowingClient:
        def upload_fileobj(self, stream, bucket, key):
            with source.open("ab") as handle:
                handle.write(b"56789" * 100)
            stream.read(-1)

    backend = S3Backend("http://a", "bucket", "key", "secret", client=GrowingClient())
    monkeypatch.setenv("S3_STORAGE_MAX_TRANSFER_BYTES", "7")
    with pytest.raises(ValueError, match="S3_STORAGE_MAX_TRANSFER_BYTES"):
        backend.upload_file(str(source), "jobs/one/source.txt")
