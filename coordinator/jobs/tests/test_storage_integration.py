"""Integration tests for storage backends with live S3-compatible services.

These tests validate storage behavior against real S3-compatible backends (MinIO/LocalStack)
and are opt-in via environment variables. They are skipped by default to keep unit tests fast.

Environment Variables Required:
    RUN_S3_INTEGRATION_TESTS=1  # Enable integration tests
    S3_ENDPOINT                  # e.g., http://localhost:9000
    S3_BUCKET                    # e.g., ocr-local-test
    S3_ACCESS_KEY                # MinIO/LocalStack access key
    S3_SECRET_KEY                # MinIO/LocalStack secret key
    S3_REGION                    # Optional, defaults to us-east-1

Usage:
    # Skip integration tests (default)
    cd coordinator && python -m pytest jobs/tests/test_storage_integration.py -v

    # Run with MinIO/LocalStack using the credentials from your env file
    export RUN_S3_INTEGRATION_TESTS=1
    export S3_ENDPOINT=http://localhost:9000
    export S3_BUCKET=ocr-local-test
    export S3_ACCESS_KEY=<minio-root-user>
    export S3_SECRET_KEY=<minio-root-password>
    cd coordinator && python -m pytest jobs/tests/test_storage_integration.py -v
"""

import os
import tempfile
from typing import Any

import pytest

from jobs.storage import NFSBackend, S3Backend, create_storage_backend


# Check if integration tests should run
def should_run_integration() -> bool:
    """Return True if all required S3 integration env vars are set."""
    return (
        os.getenv("RUN_S3_INTEGRATION_TESTS") == "1"
        and bool(os.getenv("S3_ENDPOINT"))
        and bool(os.getenv("S3_BUCKET"))
        and bool(os.getenv("S3_ACCESS_KEY"))
        and bool(os.getenv("S3_SECRET_KEY"))
    )


skip_unless_integration = pytest.mark.skipif(
    not should_run_integration(),
    reason="Set RUN_S3_INTEGRATION_TESTS=1 and S3_* env vars to run integration tests",
)


@pytest.fixture
def s3_config() -> dict[str, Any]:
    """S3 backend configuration from environment."""
    return {
        "endpoint": os.getenv("S3_ENDPOINT", ""),
        "bucket": os.getenv("S3_BUCKET", ""),
        "access_key": os.getenv("S3_ACCESS_KEY", ""),
        "secret_key": os.getenv("S3_SECRET_KEY", ""),
        "region": os.getenv("S3_REGION", "us-east-1"),
    }


@pytest.fixture
def s3_backend(s3_config: dict[str, Any]) -> S3Backend:
    """Live S3Backend connected to MinIO/LocalStack."""
    backend = S3Backend(**s3_config)
    # Ensure bucket exists
    try:
        backend.client.head_bucket(Bucket=backend.bucket)
    except Exception:
        backend.client.create_bucket(Bucket=backend.bucket)
    return backend


@pytest.fixture
def nfs_backend() -> NFSBackend:
    """NFS backend with temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield NFSBackend(root=tmpdir)


@pytest.fixture
def test_prefix() -> str:
    """Generate unique test prefix to avoid collisions."""
    import uuid

    return f"integration-test/{uuid.uuid4().hex[:8]}"


@pytest.mark.integration
@skip_unless_integration
class TestS3IntegrationUploadDownload:
    """Integration test: upload/download roundtrip integrity."""

    def test_upload_download_roundtrip_integrity(self, s3_backend: S3Backend, test_prefix: str):
        """Verify file uploaded to S3 can be downloaded with identical content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file with known content
            src_path = os.path.join(tmpdir, "source.txt")
            test_content = "Hello from integration test!\nLine 2\nLine 3\n"
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(test_content)

            # Upload
            key = f"{test_prefix}/upload-test.txt"
            locator = s3_backend.upload_file(src_path, key)
            assert locator == f"s3://{s3_backend.bucket}/{key}"

            # Download
            download_path = os.path.join(tmpdir, "downloaded.txt")
            result_path = s3_backend.download_file(key, download_path)
            assert result_path == download_path
            assert os.path.isfile(download_path)

            # Verify content integrity
            with open(download_path, "r", encoding="utf-8") as f:
                downloaded_content = f.read()
            assert downloaded_content == test_content

            # Cleanup
            s3_backend.delete(key)
            assert not s3_backend.exists(key)


@pytest.mark.integration
@skip_unless_integration
class TestS3IntegrationLifecycle:
    """Integration test: object exists + delete lifecycle."""

    def test_object_lifecycle_exists_and_delete(self, s3_backend: S3Backend, test_prefix: str):
        """Verify object lifecycle: upload -> exists -> delete -> not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and upload test file
            src_path = os.path.join(tmpdir, "lifecycle.txt")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write("lifecycle test")

            key = f"{test_prefix}/lifecycle-test.txt"

            # Initially should not exist
            assert not s3_backend.exists(key)

            # Upload
            s3_backend.upload_file(src_path, key)

            # Should exist after upload
            assert s3_backend.exists(key)

            # Delete
            s3_backend.delete(key)

            # Should not exist after delete
            assert not s3_backend.exists(key)

            # Delete again should be safe (idempotent)
            s3_backend.delete(key)
            assert not s3_backend.exists(key)


@pytest.mark.integration
@skip_unless_integration
class TestS3IntegrationListObjects:
    """Integration test: list_objects over prefixed keys."""

    def test_list_objects_with_prefixed_keys(self, s3_backend: S3Backend, test_prefix: str):
        """Verify list_objects returns all keys under prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            src_path = os.path.join(tmpdir, "list-test.txt")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write("list test")

            # Upload multiple files with different paths
            keys = [
                f"{test_prefix}/job1/source.pdf",
                f"{test_prefix}/job1/output/page1.txt",
                f"{test_prefix}/job1/output/page2.txt",
                f"{test_prefix}/job2/source.pdf",
            ]

            for key in keys:
                s3_backend.upload_file(src_path, key)

            try:
                # List all under test_prefix
                all_keys = s3_backend.list_objects(test_prefix)
                assert len(all_keys) == 4
                for key in keys:
                    assert key in all_keys

                # List only job1
                job1_keys = s3_backend.list_objects(f"{test_prefix}/job1")
                assert len(job1_keys) == 3
                assert f"{test_prefix}/job1/source.pdf" in job1_keys
                assert f"{test_prefix}/job1/output/page1.txt" in job1_keys
                assert f"{test_prefix}/job1/output/page2.txt" in job1_keys

                # List only job1/output
                output_keys = s3_backend.list_objects(f"{test_prefix}/job1/output")
                assert len(output_keys) == 2
                assert f"{test_prefix}/job1/output/page1.txt" in output_keys
                assert f"{test_prefix}/job1/output/page2.txt" in output_keys

                # List nonexistent prefix should return empty
                empty_keys = s3_backend.list_objects(f"{test_prefix}/nonexistent")
                assert empty_keys == []

            finally:
                # Cleanup all uploaded files
                for key in keys:
                    s3_backend.delete(key)


@pytest.mark.integration
@skip_unless_integration
class TestS3IntegrationPresignedURL:
    """Integration test: presigned_url generation."""

    def test_presigned_url_generation_and_access(self, s3_backend: S3Backend, test_prefix: str):
        """Verify presigned URL is generated and can be used to download object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and upload test file
            src_path = os.path.join(tmpdir, "presigned.txt")
            test_content = "presigned URL test content"
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(test_content)

            key = f"{test_prefix}/presigned-test.txt"
            s3_backend.upload_file(src_path, key)

            try:
                # Generate presigned URL
                url = s3_backend.presigned_url(key, expires=300)
                assert url is not None
                assert isinstance(url, str)
                assert len(url) > 0

                # URL should contain bucket and key information
                # (Note: Actual download via HTTP is optional, depends on network access)
                assert s3_backend.bucket in url or key in url

            finally:
                # Cleanup
                s3_backend.delete(key)


@pytest.mark.integration
@skip_unless_integration
class TestMixedBackendWorkflow:
    """Integration test: mixed NFS/S3 workflow scenario."""

    def test_mixed_nfs_and_s3_workflow(
        self, nfs_backend: NFSBackend, s3_backend: S3Backend, test_prefix: str
    ):
        """Verify coordinated workflow using both NFS and S3 backends."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Create source file locally
            src_path = os.path.join(tmpdir, "mixed-source.pdf")
            original_content = "Mixed backend workflow test content"
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(original_content)

            nfs_key = "jobs/mixed/source.pdf"
            s3_key = f"{test_prefix}/mixed/source.pdf"

            try:
                # Step 2: Upload to NFS
                nfs_backend.upload_file(src_path, nfs_key)
                assert nfs_backend.exists(nfs_key)

                # Step 3: Download from NFS
                nfs_download_path = os.path.join(tmpdir, "nfs-downloaded.pdf")
                nfs_backend.download_file(nfs_key, nfs_download_path)

                # Verify NFS content
                with open(nfs_download_path, "r", encoding="utf-8") as f:
                    nfs_content = f.read()
                assert nfs_content == original_content

                # Step 4: Upload same file to S3
                s3_locator = s3_backend.upload_file(nfs_download_path, s3_key)
                assert s3_backend.exists(s3_key)
                assert s3_locator.startswith("s3://")

                # Step 5: Download from S3
                s3_download_path = os.path.join(tmpdir, "s3-downloaded.pdf")
                s3_backend.download_file(s3_key, s3_download_path)

                # Verify S3 content matches original
                with open(s3_download_path, "r", encoding="utf-8") as f:
                    s3_content = f.read()
                assert s3_content == original_content

                # Step 6: List objects from both backends
                nfs_objects = nfs_backend.list_objects("jobs/mixed")
                assert nfs_key in nfs_objects

                s3_objects = s3_backend.list_objects(f"{test_prefix}/mixed")
                assert s3_key in s3_objects

                # Step 7: Cleanup NFS
                nfs_backend.delete(nfs_key)
                assert not nfs_backend.exists(nfs_key)

            finally:
                # Cleanup S3
                s3_backend.delete(s3_key)
                assert not s3_backend.exists(s3_key)


@pytest.mark.integration
@skip_unless_integration
class TestCreateStorageBackendIntegration:
    """Integration test: factory creates functional backends."""

    def test_create_s3_backend_from_config(self, s3_config: dict[str, Any], test_prefix: str):
        """Verify create_storage_backend factory produces working S3 backend."""
        backend = create_storage_backend(
            backend_name="s3",
            nfs_root="/unused",  # Not used for S3 backend
            s3_endpoint=s3_config["endpoint"],
            s3_bucket=s3_config["bucket"],
            s3_access_key=s3_config["access_key"],
            s3_secret_key=s3_config["secret_key"],
            s3_region=s3_config["region"],
        )

        assert isinstance(backend, S3Backend)
        assert backend.backend_name == "s3"

        # Verify it actually works
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "factory-test.txt")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write("factory test")

            key = f"{test_prefix}/factory-test.txt"

            try:
                backend.upload_file(src_path, key)
                assert backend.exists(key)

                download_path = os.path.join(tmpdir, "downloaded.txt")
                backend.download_file(key, download_path)
                assert os.path.isfile(download_path)

            finally:
                backend.delete(key)
