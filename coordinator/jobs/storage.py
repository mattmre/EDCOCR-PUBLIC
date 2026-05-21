"""Storage backend abstractions for coordinator task I/O."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend interface."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return backend identifier."""

    @abstractmethod
    def upload_file(self, local_path: str, key: str) -> str:
        """Upload local file to backend and return remote locator."""

    @abstractmethod
    def download_file(self, key: str, local_path: str) -> str:
        """Download backend object to local path and return local path."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete backend object, if present."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether backend object exists."""

    @abstractmethod
    def list_objects(self, prefix: str) -> list[str]:
        """List all object keys matching prefix."""

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        """Return temporary download URL, if backend supports it."""
        raise NotImplementedError("Presigned URLs are not supported by this backend")

    def presigned_upload_url(self, key: str, expires: int = 3600) -> str:
        """Return temporary upload URL, if backend supports it."""
        raise NotImplementedError("Presigned upload URLs are not supported by this backend")

    def delete_many(self, keys: list[str]) -> int:
        """Delete multiple objects. Returns count of successfully deleted objects.

        Default implementation falls back to sequential delete().
        Subclasses may override for batch optimization.
        """
        deleted = 0
        for key in keys:
            try:
                self.delete(key)
                deleted += 1
            except Exception as exc:
                logger.warning("Failed to delete key %s: %s", key, exc)
        return deleted


@dataclass
class NFSBackend(StorageBackend):
    """NFS-backed storage backend."""

    root: str

    @property
    def backend_name(self) -> str:
        return "nfs"

    def to_absolute_path(self, key: str) -> str:
        safe_key = key.lstrip("/\\")
        resolved = os.path.normpath(os.path.join(self.root, safe_key))
        if not resolved.startswith(os.path.normpath(self.root)):
            raise ValueError(f"Path traversal detected in key: {key}")
        return resolved

    def upload_file(self, local_path: str, key: str) -> str:
        target = self.to_absolute_path(key)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(local_path, target)
        return target

    def download_file(self, key: str, local_path: str) -> str:
        source = self.to_absolute_path(key)
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        shutil.copy2(source, local_path)
        return local_path

    def delete(self, key: str) -> None:
        target = self.to_absolute_path(key)
        try:
            os.remove(target)
        except FileNotFoundError:
            return

    def exists(self, key: str) -> bool:
        return os.path.exists(self.to_absolute_path(key))

    def list_objects(self, prefix: str) -> list[str]:
        """List all files under prefix, returning keys relative to root."""
        prefix_path = self.to_absolute_path(prefix)
        if not os.path.exists(prefix_path):
            return []

        keys = []
        for dirpath, _, filenames in os.walk(prefix_path):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                # Convert to relative key from root
                rel_path = os.path.relpath(full_path, self.root)
                # Normalize to forward slashes
                key = rel_path.replace(os.sep, "/")
                keys.append(key)
        return keys


@dataclass
class S3Backend(StorageBackend):
    """S3-compatible storage backend."""

    endpoint: str
    bucket: str
    access_key: str
    secret_key: str
    region: str = ""
    client: object | None = None

    def __post_init__(self) -> None:
        if self.client is not None:
            return
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for S3 backend. Install it in coordinator dependencies."
            ) from exc
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint or None,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region or None,
        )

    @property
    def backend_name(self) -> str:
        return "s3"

    def upload_file(self, local_path: str, key: str) -> str:
        assert self.client is not None
        self.client.upload_file(local_path, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def download_file(self, key: str, local_path: str) -> str:
        assert self.client is not None
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        self.client.download_file(self.bucket, key, local_path)
        return local_path

    def delete(self, key: str) -> None:
        assert self.client is not None
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def exists(self, key: str) -> bool:
        assert self.client is not None
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as exc:
            # Only treat 404/NoSuchKey as "not found"; re-raise unexpected errors
            error_code = getattr(getattr(exc, "response", None), "get", lambda *a: None)
            if callable(error_code):
                resp = getattr(exc, "response", None) or {}
                code = resp.get("Error", {}).get("Code", "") if isinstance(resp, dict) else ""
                http_status = (
                    resp.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
                    if isinstance(resp, dict)
                    else 0
                )
                if code in ("404", "NoSuchKey") or http_status == 404:
                    return False
            # Fallback: check string representation for common "not found" indicators
            exc_str = str(exc).lower()
            if "404" in exc_str or "not found" in exc_str or "nosuchkey" in exc_str:
                return False
            raise

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        assert self.client is not None
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )

    def presigned_upload_url(self, key: str, expires: int = 3600) -> str:
        """Generate a presigned PUT URL for uploading to S3."""
        assert self.client is not None
        return self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )

    def list_objects(self, prefix: str) -> list[str]:
        """List all object keys matching prefix using paginator."""
        assert self.client is not None
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def delete_many(self, keys: list[str]) -> int:
        """Batch delete using S3 DeleteObjects API (max 1000 per request)."""
        assert self.client is not None
        if not keys:
            return 0

        deleted = 0
        batch_size = 1000
        for i in range(0, len(keys), batch_size):
            batch = keys[i:i + batch_size]
            try:
                response = self.client.delete_objects(
                    Bucket=self.bucket,
                    Delete={
                        'Objects': [{'Key': k} for k in batch],
                        'Quiet': True,
                    },
                )
                errors = response.get('Errors', [])
                deleted += len(batch) - len(errors)
                for error in errors:
                    logger.warning(
                        "S3 batch delete error: Key=%s Code=%s Message=%s",
                        error.get('Key'), error.get('Code'), error.get('Message'),
                    )
            except Exception as exc:
                logger.error("S3 batch delete failed for %d keys: %s", len(batch), exc)
                # Fall back to sequential for this batch
                for key in batch:
                    try:
                        self.delete(key)
                        deleted += 1
                    except Exception:
                        pass
        return deleted


@dataclass
class CachedS3Backend(StorageBackend):
    """S3 backend with worker-local LRU cache to avoid redundant downloads."""

    inner: S3Backend
    cache_dir: str = "/tmp/ocr-cache"
    max_size_bytes: int = 10 * 1024**3  # 10 GB default
    _eviction_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        os.makedirs(self.cache_dir, mode=0o700, exist_ok=True)

    @property
    def backend_name(self) -> str:
        return self.inner.backend_name

    def upload_file(self, local_path: str, key: str) -> str:
        return self.inner.upload_file(local_path, key)

    def download_file(self, key: str, local_path: str) -> str:
        cache_path = self._cache_path(key)
        if os.path.isfile(cache_path):
            # Cache hit: copy to target and touch for LRU
            os.utime(cache_path)
            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
            shutil.copy2(cache_path, local_path)
            return local_path
        # Cache miss: download to temp, rename to cache, copy to target
        cache_tmp = cache_path + ".tmp"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            self.inner.download_file(key, cache_tmp)
            # Atomic rename (same filesystem) to prevent partial cache entries
            os.replace(cache_tmp, cache_path)
        except Exception:
            # Clean up partial download
            if os.path.exists(cache_tmp):
                try:
                    os.remove(cache_tmp)
                except OSError:
                    pass
            raise
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        shutil.copy2(cache_path, local_path)
        self._maybe_evict()
        return local_path

    def delete(self, key: str) -> None:
        cache_path = self._cache_path(key)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except OSError:
                pass
        self.inner.delete(key)

    def delete_many(self, keys: list[str]) -> int:
        for key in keys:
            cache_path = self._cache_path(key)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
        return self.inner.delete_many(keys)

    def exists(self, key: str) -> bool:
        return self.inner.exists(key)

    def list_objects(self, prefix: str) -> list[str]:
        return self.inner.list_objects(prefix)

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        return self.inner.presigned_url(key, expires)

    def presigned_upload_url(self, key: str, expires: int = 3600) -> str:
        return self.inner.presigned_upload_url(key, expires)

    def _cache_path(self, key: str) -> str:
        hashed = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, hashed)

    def _maybe_evict(self) -> None:
        if not self._eviction_lock.acquire(blocking=False):
            return  # Another thread is already evicting
        try:
            entries: list[tuple[str, float, int]] = []
            total = 0
            for fname in os.listdir(self.cache_dir):
                fpath = os.path.join(self.cache_dir, fname)
                if os.path.isfile(fpath) and not fpath.endswith(".tmp"):
                    stat = os.stat(fpath)
                    entries.append((fpath, stat.st_mtime, stat.st_size))
                    total += stat.st_size
            if total <= self.max_size_bytes:
                return
            # Evict oldest first until below 80% of max
            target = int(self.max_size_bytes * 0.8)
            entries.sort(key=lambda e: e[1])  # oldest mtime first
            for fpath, _, size in entries:
                if total <= target:
                    break
                try:
                    os.remove(fpath)
                    total -= size
                except OSError:
                    pass
        finally:
            self._eviction_lock.release()


def create_storage_backend(
    *,
    backend_name: str,
    nfs_root: str,
    s3_endpoint: str = "",
    s3_bucket: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
    s3_region: str = "",
    s3_client: object | None = None,
) -> StorageBackend:
    """Create storage backend from normalized config values."""
    normalized = backend_name.strip().lower()
    if normalized == "nfs":
        return NFSBackend(root=nfs_root)
    if normalized == "s3":
        return S3Backend(
            endpoint=s3_endpoint,
            bucket=s3_bucket,
            access_key=s3_access_key,
            secret_key=s3_secret_key,
            region=s3_region,
            client=s3_client,
        )
    raise ValueError(f"Unsupported storage backend '{backend_name}'")
