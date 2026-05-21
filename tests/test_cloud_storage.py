"""Tests for cloud storage connectors.

Covers StorageProvider enum, StorageObject, StorageConfig, LocalStorageBackend,
AzureBlobBackend, GcsBackend, and the create_backend factory function.
All cloud SDK interactions are mocked so tests pass without SDKs installed.
"""

import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from api.cloud_storage import (
    AzureBlobBackend,
    GcsBackend,
    LocalStorageBackend,
    StorageBackend,
    StorageConfig,
    StorageObject,
    StorageProvider,
    create_backend,
)

# ---------------------------------------------------------------------------
# StorageProvider enum
# ---------------------------------------------------------------------------


class TestStorageProvider:
    """Tests for the StorageProvider enum."""

    def test_has_local(self):
        assert StorageProvider.LOCAL.value == "local"

    def test_has_s3(self):
        assert StorageProvider.S3.value == "s3"

    def test_has_azure_blob(self):
        assert StorageProvider.AZURE_BLOB.value == "azure_blob"

    def test_has_gcs(self):
        assert StorageProvider.GCS.value == "gcs"

    def test_member_count(self):
        assert len(StorageProvider) == 4

    def test_from_value(self):
        assert StorageProvider("local") is StorageProvider.LOCAL
        assert StorageProvider("gcs") is StorageProvider.GCS


# ---------------------------------------------------------------------------
# StorageObject
# ---------------------------------------------------------------------------


class TestStorageObject:
    """Tests for the StorageObject dataclass."""

    def test_defaults(self):
        obj = StorageObject(key="test.txt")
        assert obj.key == "test.txt"
        assert obj.size == 0
        assert obj.last_modified == ""
        assert obj.content_type == ""
        assert obj.etag == ""
        assert obj.provider == ""

    def test_custom_values(self):
        obj = StorageObject(
            key="doc.pdf",
            size=1024,
            last_modified="2024-01-01",
            content_type="application/pdf",
            etag="abc123",
            provider="gcs",
        )
        assert obj.key == "doc.pdf"
        assert obj.size == 1024
        assert obj.provider == "gcs"

    def test_to_dict(self):
        obj = StorageObject(key="file.bin", size=42, provider="local")
        d = obj.to_dict()
        assert isinstance(d, dict)
        assert d["key"] == "file.bin"
        assert d["size"] == 42
        assert d["provider"] == "local"
        assert d["last_modified"] == ""
        assert d["content_type"] == ""
        assert d["etag"] == ""

    def test_to_dict_has_all_fields(self):
        obj = StorageObject(key="k")
        d = obj.to_dict()
        assert set(d.keys()) == {"key", "size", "last_modified", "content_type", "etag", "provider"}


# ---------------------------------------------------------------------------
# StorageConfig
# ---------------------------------------------------------------------------


class TestStorageConfig:
    """Tests for StorageConfig dataclass and from_env."""

    def test_defaults(self):
        cfg = StorageConfig()
        assert cfg.provider == StorageProvider.LOCAL
        assert cfg.bucket == ""
        assert cfg.prefix == ""
        assert cfg.base_path == ""
        assert cfg.endpoint_url == ""
        assert cfg.connection_string == ""
        assert cfg.project_id == ""

    def test_from_env_default_local(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = StorageConfig.from_env()
            assert cfg.provider == StorageProvider.LOCAL

    def test_from_env_explicit_provider_arg(self):
        cfg = StorageConfig.from_env(provider="gcs")
        assert cfg.provider == StorageProvider.GCS

    def test_from_env_s3(self):
        env = {
            "STORAGE_PROVIDER": "s3",
            "S3_ENDPOINT_URL": "http://minio:9000",
            "S3_ACCESS_KEY": "mykey",
            "S3_SECRET_KEY": "mysecret",
            "S3_BUCKET": "ocr-data",
            "S3_REGION": "us-east-1",
            "S3_PREFIX": "output",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = StorageConfig.from_env()
            assert cfg.provider == StorageProvider.S3
            assert cfg.endpoint_url == "http://minio:9000"
            assert cfg.access_key == "mykey"
            assert cfg.secret_key == "mysecret"
            assert cfg.bucket == "ocr-data"
            assert cfg.region == "us-east-1"
            assert cfg.prefix == "output"

    def test_from_env_azure(self):
        env = {
            "STORAGE_PROVIDER": "azure_blob",
            "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;...",
            "AZURE_STORAGE_ACCOUNT": "myaccount",
            "AZURE_STORAGE_KEY": "mykey==",
            "AZURE_CONTAINER_NAME": "docs",
            "AZURE_PREFIX": "scans",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = StorageConfig.from_env()
            assert cfg.provider == StorageProvider.AZURE_BLOB
            assert cfg.connection_string == "DefaultEndpointsProtocol=https;..."
            assert cfg.account_name == "myaccount"
            assert cfg.account_key == "mykey=="
            assert cfg.container_name == "docs"
            assert cfg.prefix == "scans"

    def test_from_env_gcs(self):
        env = {
            "STORAGE_PROVIDER": "gcs",
            "GCS_PROJECT_ID": "my-project",
            "GOOGLE_APPLICATION_CREDENTIALS": "/secrets/gcs.json",
            "GCS_BUCKET": "ocr-bucket",
            "GCS_PREFIX": "v2",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = StorageConfig.from_env()
            assert cfg.provider == StorageProvider.GCS
            assert cfg.project_id == "my-project"
            assert cfg.credentials_path == "/secrets/gcs.json"
            assert cfg.bucket == "ocr-bucket"
            assert cfg.prefix == "v2"

    def test_from_env_local_with_path(self):
        env = {
            "STORAGE_PROVIDER": "local",
            "LOCAL_STORAGE_PATH": "/data/ocr",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = StorageConfig.from_env()
            assert cfg.provider == StorageProvider.LOCAL
            assert cfg.base_path == "/data/ocr"


# ---------------------------------------------------------------------------
# StorageBackend ABC
# ---------------------------------------------------------------------------


class TestStorageBackendABC:
    """Verify StorageBackend is abstract and cannot be instantiated."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            StorageBackend()

    def test_subclass_must_implement_all(self):
        class Incomplete(StorageBackend):
            pass

        with pytest.raises(TypeError):
            Incomplete()


# ---------------------------------------------------------------------------
# LocalStorageBackend
# ---------------------------------------------------------------------------


class TestLocalStorageBackend:
    """Tests for the local filesystem backend using tmp_path."""

    @pytest.fixture
    def storage_dir(self, tmp_path):
        d = tmp_path / "storage"
        d.mkdir()
        return d

    @pytest.fixture
    def backend(self, storage_dir):
        return LocalStorageBackend(base_path=str(storage_dir))

    def test_upload_creates_file(self, backend, storage_dir):
        obj = backend.upload("hello.txt", b"world", content_type="text/plain")
        assert obj.key == "hello.txt"
        assert obj.size == 5
        assert obj.content_type == "text/plain"
        assert obj.provider == "local"
        assert (storage_dir / "hello.txt").read_bytes() == b"world"

    def test_download(self, backend):
        backend.upload("data.bin", b"\x00\x01\x02")
        result = backend.download("data.bin")
        assert result == b"\x00\x01\x02"

    def test_download_nonexistent_raises(self, backend):
        with pytest.raises(FileNotFoundError, match="Object not found"):
            backend.download("nonexistent.txt")

    def test_exists_true(self, backend):
        backend.upload("found.txt", b"yes")
        assert backend.exists("found.txt") is True

    def test_exists_false(self, backend):
        assert backend.exists("nope.txt") is False

    def test_delete_existing(self, backend):
        backend.upload("gone.txt", b"bye")
        assert backend.delete("gone.txt") is True
        assert backend.exists("gone.txt") is False

    def test_delete_nonexistent_returns_false(self, backend):
        assert backend.delete("nothing.txt") is False

    def test_list_objects_empty(self, backend):
        result = backend.list_objects()
        assert result == []

    def test_list_objects_flat(self, backend):
        backend.upload("a.txt", b"a")
        backend.upload("b.txt", b"bb")
        objs = backend.list_objects()
        keys = sorted(o.key for o in objs)
        assert keys == ["a.txt", "b.txt"]

    def test_list_objects_with_prefix(self, backend):
        backend.upload("docs/report.pdf", b"pdf")
        backend.upload("imgs/photo.jpg", b"jpg")
        objs = backend.list_objects(prefix="docs")
        assert len(objs) == 1
        assert objs[0].key == "docs/report.pdf"

    def test_list_objects_nonexistent_prefix(self, backend):
        result = backend.list_objects(prefix="no_such_dir")
        assert result == []

    def test_nested_directories(self, backend, storage_dir):
        backend.upload("a/b/c/deep.txt", b"nested")
        assert (storage_dir / "a" / "b" / "c" / "deep.txt").exists()
        data = backend.download("a/b/c/deep.txt")
        assert data == b"nested"

    def test_list_nested(self, backend):
        backend.upload("x/y/one.txt", b"1")
        backend.upload("x/y/two.txt", b"22")
        backend.upload("x/three.txt", b"333")
        objs = backend.list_objects(prefix="x/y")
        keys = sorted(o.key for o in objs)
        assert keys == ["x/y/one.txt", "x/y/two.txt"]

    def test_get_metadata(self, backend):
        backend.upload("meta.txt", b"hello")
        meta = backend.get_metadata("meta.txt")
        assert meta.key == "meta.txt"
        assert meta.size == 5
        assert meta.provider == "local"

    def test_get_metadata_nonexistent_raises(self, backend):
        with pytest.raises(FileNotFoundError, match="Object not found"):
            backend.get_metadata("missing.txt")

    def test_upload_overwrite(self, backend):
        backend.upload("dup.txt", b"first")
        backend.upload("dup.txt", b"second")
        assert backend.download("dup.txt") == b"second"

    def test_upload_empty_bytes(self, backend):
        obj = backend.upload("empty.bin", b"")
        assert obj.size == 0
        assert backend.download("empty.bin") == b""

    def test_list_objects_returns_sizes(self, backend):
        backend.upload("sized.txt", b"12345")
        objs = backend.list_objects()
        assert objs[0].size == 5


# ---------------------------------------------------------------------------
# AzureBlobBackend
# ---------------------------------------------------------------------------


class TestAzureBlobBackend:
    """Tests for AzureBlobBackend construction and key prefixing."""

    def test_init_stores_config(self):
        b = AzureBlobBackend(
            connection_string="conn",
            account_name="acct",
            account_key="key",
            container_name="ctr",
            prefix="pfx",
        )
        assert b._connection_string == "conn"
        assert b._account_name == "acct"
        assert b._account_key == "key"
        assert b.container_name == "ctr"
        assert b.prefix == "pfx"
        assert b._client is None

    def test_prefixed_key_with_prefix(self):
        b = AzureBlobBackend(prefix="data")
        assert b._prefixed_key("file.txt") == "data/file.txt"

    def test_prefixed_key_without_prefix(self):
        b = AzureBlobBackend(prefix="")
        assert b._prefixed_key("file.txt") == "file.txt"

    def test_get_client_raises_import_error(self):
        """Verify ImportError is raised when azure-storage-blob is missing."""
        b = AzureBlobBackend(connection_string="fake")
        with mock.patch.dict(sys.modules, {"azure": None, "azure.storage": None, "azure.storage.blob": None}):
            # Force _client to None so _get_client tries the import
            b._client = None
            with pytest.raises(ImportError, match="azure-storage-blob"):
                b._get_client()

    def test_init_default_values(self):
        b = AzureBlobBackend()
        assert b._connection_string == ""
        assert b._account_name == ""
        assert b._account_key == ""
        assert b.container_name == ""
        assert b.prefix == ""

    def test_prefixed_key_nested(self):
        b = AzureBlobBackend(prefix="a/b")
        assert b._prefixed_key("c/d.txt") == "a/b/c/d.txt"


# ---------------------------------------------------------------------------
# GcsBackend
# ---------------------------------------------------------------------------


class TestGcsBackend:
    """Tests for GcsBackend construction and key prefixing."""

    def test_init_stores_config(self):
        b = GcsBackend(
            project_id="proj",
            credentials_path="/creds.json",
            bucket_name="bkt",
            prefix="pfx",
        )
        assert b._project_id == "proj"
        assert b._credentials_path == "/creds.json"
        assert b.bucket_name == "bkt"
        assert b.prefix == "pfx"
        assert b._client is None

    def test_prefixed_key_with_prefix(self):
        b = GcsBackend(prefix="output")
        assert b._prefixed_key("scan.pdf") == "output/scan.pdf"

    def test_prefixed_key_without_prefix(self):
        b = GcsBackend(prefix="")
        assert b._prefixed_key("scan.pdf") == "scan.pdf"

    def test_get_client_raises_import_error(self):
        """Verify ImportError is raised when google-cloud-storage is missing."""
        b = GcsBackend(bucket_name="bkt")
        with mock.patch.dict(sys.modules, {"google": None, "google.cloud": None, "google.cloud.storage": None}):
            b._client = None
            with pytest.raises(ImportError, match="google-cloud-storage"):
                b._get_client()

    def test_init_default_values(self):
        b = GcsBackend()
        assert b._project_id == ""
        assert b._credentials_path == ""
        assert b.bucket_name == ""
        assert b.prefix == ""

    def test_prefixed_key_nested(self):
        b = GcsBackend(prefix="x/y")
        assert b._prefixed_key("z/w.bin") == "x/y/z/w.bin"


# ---------------------------------------------------------------------------
# create_backend factory
# ---------------------------------------------------------------------------


class TestCreateBackend:
    """Tests for the create_backend factory function."""

    def test_creates_local(self, tmp_path):
        cfg = StorageConfig(provider=StorageProvider.LOCAL, base_path=str(tmp_path))
        backend = create_backend(cfg)
        assert isinstance(backend, LocalStorageBackend)
        assert backend.base_path == str(tmp_path)

    def test_creates_azure(self):
        cfg = StorageConfig(
            provider=StorageProvider.AZURE_BLOB,
            connection_string="conn",
            container_name="ctr",
            prefix="p",
        )
        backend = create_backend(cfg)
        assert isinstance(backend, AzureBlobBackend)
        assert backend.container_name == "ctr"
        assert backend.prefix == "p"

    def test_creates_gcs(self):
        cfg = StorageConfig(
            provider=StorageProvider.GCS,
            project_id="proj",
            bucket="bkt",
            prefix="q",
        )
        backend = create_backend(cfg)
        assert isinstance(backend, GcsBackend)
        assert backend.bucket_name == "bkt"
        assert backend.prefix == "q"

    def test_unsupported_raises(self):
        cfg = StorageConfig(provider=StorageProvider.S3)
        with pytest.raises(ValueError, match="Unsupported storage provider"):
            create_backend(cfg)

    def test_local_roundtrip(self, tmp_path):
        cfg = StorageConfig(provider=StorageProvider.LOCAL, base_path=str(tmp_path))
        backend = create_backend(cfg)
        backend.upload("rt.txt", b"round-trip")
        assert backend.download("rt.txt") == b"round-trip"
