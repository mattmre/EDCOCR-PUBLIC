"""Tests for Phase 7 storage backend abstraction."""

import os
import tempfile
import time
from unittest.mock import MagicMock

from django.test import SimpleTestCase

from jobs.storage import CachedS3Backend, NFSBackend, S3Backend, create_storage_backend


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
            assert uploaded_path.endswith("jobs\\abc\\output\\test.txt") or uploaded_path.endswith(
                "jobs/abc/output/test.txt"
            )
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
                with open(os.path.join(tmpdir, "download.txt"), "r", encoding="utf-8") as f:
                    assert f.read() == "basename-target"
            finally:
                os.chdir(cwd)

    def test_list_objects_returns_all_files_under_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = NFSBackend(root=tmpdir)

            # Create test files
            for key in ["jobs/1/source.pdf", "jobs/1/temp/page1.pdf", "jobs/2/source.pdf"]:
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

        locator = backend.upload_file("/tmp/source.pdf", "jobs/1/source.pdf")
        assert locator == "s3://ocr-local/jobs/1/source.pdf"
        client.upload_file.assert_called_once_with("/tmp/source.pdf", "ocr-local", "jobs/1/source.pdf")

        backend.download_file("jobs/1/source.pdf", "/tmp/output/source.pdf")
        client.download_file.assert_called_once_with("ocr-local", "jobs/1/source.pdf", "/tmp/output/source.pdf")

        backend.delete("jobs/1/source.pdf")
        client.delete_object.assert_called_once_with(Bucket="ocr-local", Key="jobs/1/source.pdf")

    def test_download_allows_basename_target_path(self):
        backend, client = self._backend()

        backend.download_file("jobs/1/source.pdf", "download.pdf")

        client.download_file.assert_called_once_with("ocr-local", "jobs/1/source.pdf", "download.pdf")

    def test_exists_true_when_head_object_succeeds(self):
        backend, client = self._backend()
        client.head_object.return_value = {"ETag": "abc"}

        assert backend.exists("jobs/1/file.pdf") is True
        client.head_object.assert_called_once_with(Bucket="ocr-local", Key="jobs/1/file.pdf")

    def test_exists_false_when_head_object_raises(self):
        backend, client = self._backend()
        client.head_object.side_effect = RuntimeError("not found")

        assert backend.exists("jobs/1/file.pdf") is False

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
        mock_paginator.paginate.assert_called_once_with(Bucket="ocr-local", Prefix="jobs/1")

    def test_list_objects_empty_result(self):
        backend, client = self._backend()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": []}]
        client.get_paginator.return_value = mock_paginator

        keys = backend.list_objects("jobs/nonexistent")
        assert keys == []


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
        first_call_objects = client.delete_objects.call_args_list[0][1]["Delete"]["Objects"]
        assert len(first_call_objects) == 1000
        # Second batch should have 500 keys
        second_call_objects = client.delete_objects.call_args_list[1][1]["Delete"]["Objects"]
        assert len(second_call_objects) == 500
        assert result == 1500

    def test_batch_delete_returns_correct_count(self):
        """delete_many should subtract error count from total to get successful deletes."""
        backend, client = self._backend()
        client.delete_objects.return_value = {
            "Errors": [
                {"Key": "k2", "Code": "AccessDenied", "Message": "Denied"},
                {"Key": "k4", "Code": "InternalError", "Message": "Error"},
            ]
        }

        result = backend.delete_many(["k1", "k2", "k3", "k4", "k5"])

        assert result == 3  # 5 keys - 2 errors = 3 successful

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

    def _make_inner(self):
        """Create a mock S3Backend for use as the inner backend."""
        inner = MagicMock(spec=S3Backend)
        inner.backend_name = "s3"
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

        with tempfile.TemporaryDirectory() as cache_dir, tempfile.TemporaryDirectory() as target_dir:
            cached = CachedS3Backend(inner=inner, cache_dir=cache_dir)

            target_path = os.path.join(target_dir, "output.pdf")
            cached.download_file("jobs/1/source.pdf", target_path)

            # Inner download should have been called
            inner.download_file.assert_called_once()
            # Target file should exist with correct content
            assert os.path.isfile(target_path)
            with open(target_path, "rb") as f:
                assert f.read() == b"downloaded-content"

    def test_cache_hit_avoids_download(self):
        """On cache hit, should copy from cache without calling inner."""
        inner = self._make_inner()

        with tempfile.TemporaryDirectory() as cache_dir, tempfile.TemporaryDirectory() as target_dir:
            cached = CachedS3Backend(inner=inner, cache_dir=cache_dir)

            # Pre-populate cache entry for the key
            import hashlib
            cache_hash = hashlib.sha256("jobs/1/source.pdf".encode()).hexdigest()
            cache_path = os.path.join(cache_dir, cache_hash)
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

    def test_lru_eviction_removes_oldest(self):
        """When cache exceeds max_size_bytes, oldest files should be evicted."""
        inner = self._make_inner()

        with tempfile.TemporaryDirectory() as cache_dir:
            # Use a very small max_size to trigger eviction
            cached = CachedS3Backend(inner=inner, cache_dir=cache_dir, max_size_bytes=100)

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
            import hashlib
            cache_hash = hashlib.sha256("jobs/1/file.pdf".encode()).hexdigest()
            cache_path = os.path.join(cache_dir, cache_hash)
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
            import hashlib
            keys = ["jobs/1/a.pdf", "jobs/1/b.pdf"]
            cache_paths = []
            for key in keys:
                cache_hash = hashlib.sha256(key.encode()).hexdigest()
                cache_path = os.path.join(cache_dir, cache_hash)
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

        with tempfile.TemporaryDirectory() as cache_dir, tempfile.TemporaryDirectory() as target_dir:
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
            import hashlib
            cache_hash = hashlib.sha256("jobs/1/source.pdf".encode()).hexdigest()
            cache_path = os.path.join(cache_dir, cache_hash)
            assert not os.path.exists(cache_path)
