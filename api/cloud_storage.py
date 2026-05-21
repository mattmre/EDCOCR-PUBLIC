"""Cloud storage connectors for Azure Blob and Google Cloud Storage.

Provides a unified StorageBackend interface for uploading, downloading,
listing, and deleting files across multiple cloud providers. Extends
the existing S3/MinIO support with Azure Blob and GCS connectors.

All cloud SDK imports are lazy — modules work without SDKs installed.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credential manager integration (vault / KMS / env fallback chain)
# ---------------------------------------------------------------------------
try:
    from credential_manager import get_credential as _get_credential
except ImportError:

    def _get_credential(key, default=None):  # type: ignore[misc]
        return os.environ.get(key, default)


class StorageProvider(Enum):
    LOCAL = "local"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"


@dataclass
class StorageObject:
    """Metadata for a stored object."""
    key: str
    size: int = 0
    last_modified: str = ""
    content_type: str = ""
    etag: str = ""
    provider: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "size": self.size,
            "last_modified": self.last_modified,
            "content_type": self.content_type,
            "etag": self.etag,
            "provider": self.provider,
        }


@dataclass
class StorageConfig:
    """Configuration for a storage backend."""
    provider: StorageProvider = StorageProvider.LOCAL
    # Common
    bucket: str = ""
    prefix: str = ""
    # Local
    base_path: str = ""
    # S3
    endpoint_url: str = ""
    access_key: str = ""
    secret_key: str = ""
    region: str = ""
    # Azure
    connection_string: str = ""
    account_name: str = ""
    account_key: str = ""
    container_name: str = ""
    # GCS
    project_id: str = ""
    credentials_path: str = ""

    @classmethod
    def from_env(cls, provider: str = None) -> "StorageConfig":
        """Create config from environment variables."""
        prov = provider or os.environ.get("STORAGE_PROVIDER", "local")
        config = cls(provider=StorageProvider(prov))

        if config.provider == StorageProvider.S3:
            config.endpoint_url = os.environ.get("S3_ENDPOINT_URL", "")
            config.access_key = _get_credential("S3_ACCESS_KEY", "")
            config.secret_key = _get_credential("S3_SECRET_KEY", "")
            config.bucket = os.environ.get("S3_BUCKET", "")
            config.region = os.environ.get("S3_REGION", "")
            config.prefix = os.environ.get("S3_PREFIX", "")
        elif config.provider == StorageProvider.AZURE_BLOB:
            config.connection_string = _get_credential("AZURE_STORAGE_CONNECTION_STRING", "")
            config.account_name = os.environ.get("AZURE_STORAGE_ACCOUNT", "")
            config.account_key = _get_credential("AZURE_STORAGE_KEY", "")
            config.container_name = os.environ.get("AZURE_CONTAINER_NAME", "")
            config.prefix = os.environ.get("AZURE_PREFIX", "")
        elif config.provider == StorageProvider.GCS:
            config.project_id = os.environ.get("GCS_PROJECT_ID", "")
            config.credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            config.bucket = os.environ.get("GCS_BUCKET", "")
            config.prefix = os.environ.get("GCS_PREFIX", "")
        elif config.provider == StorageProvider.LOCAL:
            config.base_path = os.environ.get("LOCAL_STORAGE_PATH", "")

        return config


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def upload(self, key: str, data: bytes, content_type: str = "") -> StorageObject:
        """Upload data to storage."""
        ...

    @abstractmethod
    def download(self, key: str) -> bytes:
        """Download data from storage."""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete an object. Returns True if deleted."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if object exists."""
        ...

    @abstractmethod
    def list_objects(self, prefix: str = "") -> list:
        """List objects with optional prefix filter."""
        ...

    @abstractmethod
    def get_metadata(self, key: str) -> StorageObject:
        """Get object metadata without downloading."""
        ...


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: str):
        self.base_path = base_path

    def _full_path(self, key: str) -> str:
        return os.path.join(self.base_path, key.replace("/", os.sep))

    def upload(self, key: str, data: bytes, content_type: str = "") -> StorageObject:
        path = self._full_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return StorageObject(
            key=key,
            size=len(data),
            content_type=content_type,
            provider="local",
        )

    def download(self, key: str) -> bytes:
        path = self._full_path(key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Object not found: {key}")
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str) -> bool:
        path = self._full_path(key)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def exists(self, key: str) -> bool:
        return os.path.exists(self._full_path(key))

    def list_objects(self, prefix: str = "") -> list:
        results = []
        search_path = self._full_path(prefix) if prefix else self.base_path
        if not os.path.exists(search_path):
            return results
        for root, dirs, files in os.walk(search_path):
            for fname in files:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, self.base_path).replace(os.sep, "/")
                size = os.path.getsize(full)
                results.append(StorageObject(key=rel, size=size, provider="local"))
        return results

    def get_metadata(self, key: str) -> StorageObject:
        path = self._full_path(key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Object not found: {key}")
        return StorageObject(
            key=key,
            size=os.path.getsize(path),
            provider="local",
        )


class AzureBlobBackend(StorageBackend):
    """Azure Blob Storage backend. Requires azure-storage-blob package."""

    def __init__(self, connection_string: str = "", account_name: str = "",
                 account_key: str = "", container_name: str = "",
                 prefix: str = ""):
        self.container_name = container_name
        self.prefix = prefix
        self._client = None
        self._connection_string = connection_string
        self._account_name = account_name
        self._account_key = account_key

    def _get_client(self):
        if self._client is None:
            try:
                from azure.storage.blob import BlobServiceClient
            except ImportError:
                raise ImportError(
                    "azure-storage-blob is required. "
                    "Install with: pip install azure-storage-blob"
                )
            if self._connection_string:
                service = BlobServiceClient.from_connection_string(self._connection_string)
            else:
                url = f"https://{self._account_name}.blob.core.windows.net"
                service = BlobServiceClient(account_url=url, credential=self._account_key)
            self._client = service.get_container_client(self.container_name)
        return self._client

    def _prefixed_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def upload(self, key: str, data: bytes, content_type: str = "") -> StorageObject:
        client = self._get_client()
        blob_key = self._prefixed_key(key)
        kwargs = {}
        if content_type:
            from azure.storage.blob import ContentSettings
            kwargs["content_settings"] = ContentSettings(content_type=content_type)
        client.upload_blob(blob_key, data, overwrite=True, **kwargs)
        return StorageObject(key=key, size=len(data), content_type=content_type, provider="azure_blob")

    def download(self, key: str) -> bytes:
        client = self._get_client()
        blob = client.download_blob(self._prefixed_key(key))
        return blob.readall()

    def delete(self, key: str) -> bool:
        client = self._get_client()
        try:
            client.delete_blob(self._prefixed_key(key))
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        client = self._get_client()
        blob = client.get_blob_client(self._prefixed_key(key))
        try:
            blob.get_blob_properties()
            return True
        except Exception:
            return False

    def list_objects(self, prefix: str = "") -> list:
        client = self._get_client()
        search = self._prefixed_key(prefix) if prefix else self.prefix
        results = []
        for blob in client.list_blobs(name_starts_with=search):
            key = blob.name
            if self.prefix and key.startswith(f"{self.prefix}/"):
                key = key[len(self.prefix) + 1:]
            results.append(StorageObject(
                key=key,
                size=blob.size,
                last_modified=str(blob.last_modified) if blob.last_modified else "",
                content_type=blob.content_settings.content_type if blob.content_settings else "",
                etag=blob.etag or "",
                provider="azure_blob",
            ))
        return results

    def get_metadata(self, key: str) -> StorageObject:
        client = self._get_client()
        blob = client.get_blob_client(self._prefixed_key(key))
        props = blob.get_blob_properties()
        return StorageObject(
            key=key,
            size=props.size,
            last_modified=str(props.last_modified) if props.last_modified else "",
            content_type=props.content_settings.content_type if props.content_settings else "",
            etag=props.etag or "",
            provider="azure_blob",
        )


class GcsBackend(StorageBackend):
    """Google Cloud Storage backend. Requires google-cloud-storage package."""

    def __init__(self, project_id: str = "", credentials_path: str = "",
                 bucket_name: str = "", prefix: str = ""):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self._client = None
        self._project_id = project_id
        self._credentials_path = credentials_path

    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import storage
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required. "
                    "Install with: pip install google-cloud-storage"
                )
            kwargs = {}
            if self._project_id:
                kwargs["project"] = self._project_id
            if self._credentials_path:
                from google.oauth2 import service_account
                creds = service_account.Credentials.from_service_account_file(self._credentials_path)
                kwargs["credentials"] = creds
            client = storage.Client(**kwargs)
            self._client = client.bucket(self.bucket_name)
        return self._client

    def _prefixed_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def upload(self, key: str, data: bytes, content_type: str = "") -> StorageObject:
        bucket = self._get_client()
        blob = bucket.blob(self._prefixed_key(key))
        blob.upload_from_string(data, content_type=content_type or "application/octet-stream")
        return StorageObject(key=key, size=len(data), content_type=content_type, provider="gcs")

    def download(self, key: str) -> bytes:
        bucket = self._get_client()
        blob = bucket.blob(self._prefixed_key(key))
        return blob.download_as_bytes()

    def delete(self, key: str) -> bool:
        bucket = self._get_client()
        blob = bucket.blob(self._prefixed_key(key))
        try:
            blob.delete()
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        bucket = self._get_client()
        blob = bucket.blob(self._prefixed_key(key))
        return blob.exists()

    def list_objects(self, prefix: str = "") -> list:
        bucket = self._get_client()
        search = self._prefixed_key(prefix) if prefix else self.prefix
        results = []
        for blob in bucket.list_blobs(prefix=search):
            key = blob.name
            if self.prefix and key.startswith(f"{self.prefix}/"):
                key = key[len(self.prefix) + 1:]
            results.append(StorageObject(
                key=key,
                size=blob.size or 0,
                last_modified=str(blob.updated) if blob.updated else "",
                content_type=blob.content_type or "",
                etag=blob.etag or "",
                provider="gcs",
            ))
        return results

    def get_metadata(self, key: str) -> StorageObject:
        bucket = self._get_client()
        blob = bucket.blob(self._prefixed_key(key))
        blob.reload()
        return StorageObject(
            key=key,
            size=blob.size or 0,
            last_modified=str(blob.updated) if blob.updated else "",
            content_type=blob.content_type or "",
            etag=blob.etag or "",
            provider="gcs",
        )


def create_backend(config: StorageConfig) -> StorageBackend:
    """Factory function to create appropriate storage backend."""
    if config.provider == StorageProvider.LOCAL:
        return LocalStorageBackend(base_path=config.base_path)
    elif config.provider == StorageProvider.AZURE_BLOB:
        return AzureBlobBackend(
            connection_string=config.connection_string,
            account_name=config.account_name,
            account_key=config.account_key,
            container_name=config.container_name,
            prefix=config.prefix,
        )
    elif config.provider == StorageProvider.GCS:
        return GcsBackend(
            project_id=config.project_id,
            credentials_path=config.credentials_path,
            bucket_name=config.bucket,
            prefix=config.prefix,
        )
    else:
        raise ValueError(f"Unsupported storage provider: {config.provider}")
