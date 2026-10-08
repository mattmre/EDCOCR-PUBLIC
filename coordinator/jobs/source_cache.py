"""Cross-process read leases for immutable, authorized OCR source objects.

The cache never grants authority: callers must refresh/check the task's grant
on every lease, including warm hits.  A source revision and expected SHA-256
are mandatory, so a changed remote object cannot alias older cached bytes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterator
from urllib.parse import parse_qs, urlparse

from .storage import _ensure_local_directory, _open_local_parent


@dataclass(frozen=True)
class SourceIdentity:
    tenant_id: str
    job_id: str
    generation: int
    object_key: str
    sha256: str
    revision: str
    size_bytes: int

    def __post_init__(self) -> None:
        if any(not getattr(self, field) for field in ("tenant_id", "job_id", "object_key", "revision")):
            raise ValueError("source identity requires tenant, job, key, and revision")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0:
            raise ValueError("source generation must be a nonnegative integer")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            raise ValueError("source identity requires a positive byte size")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256.lower()):
            raise ValueError("source identity requires a SHA-256 digest")
        if self.object_key.startswith("/") or "\\" in self.object_key or any(
            part in {"", ".", ".."} for part in self.object_key.split("/")
        ):
            raise ValueError("source object key must be canonical and relative")

    def canonical(self) -> str:
        return json.dumps(
            [self.tenant_id, self.job_id, self.generation, self.object_key, self.sha256.lower(), self.revision, self.size_bytes],
            ensure_ascii=False,
            separators=(",", ":"),
        )


class SourceLeasePath(str):
    """A path whose original inode can be checked before output publication."""

    def __new__(cls, path: str, cache: ImmutableSourceCache, fence: list[int]):
        value = str.__new__(cls, path)
        value._cache = cache
        value._fence = fence
        return value

    def check(self) -> None:
        if self._cache._fence(self._cache._file_metadata(str(self))) != self._fence:
            raise RuntimeError("leased source changed during read")


class ImmutableSourceCache:
    """Share source bytes under a reader lock until the page releases them.

    Limits apply per cache instance/process.  Shared filesystem lock shards
    synchronize publication and eviction across prefork worker processes.
    ``fetch`` must write the named temporary path and enforce its own remote
    transfer limit; the cache verifies the resulting file before publication.
    """

    def __init__(
        self,
        cache_dir: str,
        *,
        max_size_bytes: int = 1 * 1024**3,
        max_object_bytes: int = 512 * 1024**2,
        max_active_leases: int = 32,
        max_fetches: int = 4,
        max_capacity_wait_seconds: float = 30.0,
        diagnostics_enabled: bool | None = None,
    ) -> None:
        if min(max_size_bytes, max_object_bytes, max_active_leases, max_fetches) < 1:
            raise ValueError("source cache limits must be positive")
        if max_capacity_wait_seconds <= 0:
            raise ValueError("source cache capacity wait must be positive")
        self.cache_dir = os.path.abspath(cache_dir)
        self.max_size_bytes = max_size_bytes
        self.max_object_bytes = min(max_object_bytes, max_size_bytes)
        self.max_active_leases = max_active_leases
        self.max_fetches = max_fetches
        self.max_capacity_wait_seconds = max_capacity_wait_seconds
        self.diagnostics_enabled = (
            os.environ.get("OCR_SOURCE_CACHE_DIAGNOSTICS", "false").lower() in {"1", "true", "yes"}
            if diagnostics_enabled is None else diagnostics_enabled
        )
        self._leases = threading.BoundedSemaphore(max_active_leases)
        self._fetches = threading.BoundedSemaphore(max_fetches)
        self._pid = os.getpid()
        self._prepare_directory(self.cache_dir)
        self._prepare_directory(os.path.join(self.cache_dir, "locks"))
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._reset_after_fork)

    def _reset_after_fork(self) -> None:
        # Semaphores inherited from a busy parent must never block a child.
        self._pid = os.getpid()
        self._leases = threading.BoundedSemaphore(self.max_active_leases)
        self._fetches = threading.BoundedSemaphore(self.max_fetches)

    @staticmethod
    def _prepare_directory(path: str) -> None:
        _ensure_local_directory(path)
        parent_fd, basename, _ = _open_local_parent(path, create=False)
        try:
            fd = os.open(basename, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            try:
                os.fchmod(fd, 0o700)
            finally:
                os.close(fd)
        finally:
            os.close(parent_fd)

    def _paths(self, identity: SourceIdentity) -> tuple[str, str, str]:
        digest = hashlib.sha256(identity.canonical().encode()).hexdigest()
        return (
            os.path.join(self.cache_dir, f"{digest}.blob"),
            os.path.join(self.cache_dir, f"{digest}.json"),
            os.path.join(self.cache_dir, "locks", f"{digest[:2]}.lock"),
        )

    def diagnostic_counters(self) -> dict[str, int]:
        """Return bounded cross-process payload counters for this cache root."""
        path = os.path.join(self.cache_dir, ".metrics.json")
        lock_fd = self._open_lock(os.path.join(self.cache_dir, ".metrics.lock"))
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            if not os.path.lexists(path):
                return self._empty_counters()
            with os.fdopen(self._open_regular(path), encoding="utf-8") as handle:
                counters = json.load(handle)
            return self._validate_counters(counters)
        finally:
            os.close(lock_fd)

    def _record_lease(self, *, fetched_bytes: int) -> None:
        self._record_metrics(
            leases=1,
            cache_hits=int(fetched_bytes == 0),
            successful_fetches=int(fetched_bytes > 0),
            successful_fetch_bytes=fetched_bytes,
        )

    @staticmethod
    def _empty_counters() -> dict[str, int]:
        return {
            "leases": 0, "cache_hits": 0, "successful_fetches": 0,
            "successful_fetch_bytes": 0, "cache_layer_copy_bytes": 0,
            "source_http_get_attempts": 0, "source_http_get_success_bytes": 0,
        }

    @classmethod
    def _validate_counters(cls, counters: object) -> dict[str, int]:
        defaults = cls._empty_counters()
        # A previous image may have written the original five fields into a
        # persistent cache. Preserve those measurements and initialize only
        # the newly introduced HTTP transport observations.
        old_fields = set(defaults) - {"source_http_get_attempts", "source_http_get_success_bytes"}
        if (
            not isinstance(counters, dict)
            or not old_fields.issubset(counters)
            or set(counters) - set(defaults)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters.values())
        ):
            raise RuntimeError("source cache diagnostic counters are malformed")
        return defaults | counters

    def _record_metrics(self, **increments: int) -> None:
        if not self.diagnostics_enabled:
            return
        if set(increments) - set(self._empty_counters()) or any(value < 0 for value in increments.values()):
            raise ValueError("invalid source cache diagnostic increment")
        lock_fd = self._open_lock(os.path.join(self.cache_dir, ".metrics.lock"))
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            path = os.path.join(self.cache_dir, ".metrics.json")
            counters = self.diagnostic_counters_unlocked(path)
            for name, value in increments.items():
                counters[name] += value
            temporary = os.path.join(self.cache_dir, f".metrics-{uuid.uuid4().hex}.tmp")
            parent_fd, temporary_name, _ = _open_local_parent(temporary, create=False)
            try:
                fd = os.open(temporary_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent_fd)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(counters, handle, separators=(",", ":"))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, ".metrics.json", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
                self._unlink(temporary, missing_ok=True)
        finally:
            os.close(lock_fd)

    def diagnostic_counters_unlocked(self, path: str) -> dict[str, int]:
        if not os.path.lexists(path):
            return self._empty_counters()
        with os.fdopen(self._open_regular(path), encoding="utf-8") as handle:
            counters = json.load(handle)
        return self._validate_counters(counters)

    @staticmethod
    def _open_lock(path: str) -> int:
        parent_fd, basename, _ = _open_local_parent(path, create=False)
        try:
            fd = os.open(basename, os.O_CREAT | os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(fd)
            raise RuntimeError("source cache lock is unsafe")
        return fd

    @staticmethod
    def _file_metadata(path: str) -> os.stat_result:
        fd = ImmutableSourceCache._open_regular(path)
        try:
            return os.fstat(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _open_regular(path: str) -> int:
        parent_fd, basename, _ = _open_local_parent(path, create=False)
        try:
            try:
                fd = os.open(basename, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
            except OSError as exc:
                raise RuntimeError("source cache file is unsafe") from exc
        finally:
            os.close(parent_fd)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(fd)
            raise RuntimeError("source cache file is unsafe")
        return fd

    @staticmethod
    def _unlink(path: str, *, missing_ok: bool = False) -> None:
        parent_fd, basename, _ = _open_local_parent(path, create=False)
        try:
            try:
                os.unlink(basename, dir_fd=parent_fd)
            except FileNotFoundError:
                if not missing_ok:
                    raise
        finally:
            os.close(parent_fd)

    def _list_cache(self) -> list[str]:
        parent_fd, basename, _ = _open_local_parent(self.cache_dir, create=False)
        try:
            directory_fd = os.open(basename, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        try:
            return os.listdir(directory_fd)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _fence(metadata: os.stat_result) -> list[int]:
        return [metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns]

    def _valid(self, identity: SourceIdentity, blob: str, manifest: str) -> bool:
        # Even an incomplete pair must not hide a linked or special file.
        if os.path.lexists(blob):
            self._file_metadata(blob)
        if os.path.lexists(manifest):
            self._file_metadata(manifest)
        if not os.path.lexists(blob) and not os.path.lexists(manifest):
            return False
        if not os.path.lexists(blob) or not os.path.lexists(manifest):
            return False  # Interrupted publication; exclusive miss repairs it.
        blob_stat = self._file_metadata(blob)
        with os.fdopen(self._open_regular(manifest), encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("identity") != identity.canonical() or record.get("fence") != self._fence(blob_stat):
            raise RuntimeError("source cache identity or file metadata changed")
        if blob_stat.st_size != identity.size_bytes or blob_stat.st_size > self.max_object_bytes:
            raise ValueError("cached source exceeds object limit")
        return True

    def _budget_usage(self) -> int:
        """Count published bytes and live in-flight reservations under budget lock."""
        total = 0
        for name in self._list_cache():
            path = os.path.join(self.cache_dir, name)
            if name.endswith(".blob"):
                total += self._file_metadata(path).st_size
            elif name.startswith(".reserve-"):
                fd = self._open_regular(path)
                try:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        with os.fdopen(os.dup(fd), encoding="ascii") as handle:
                            total += int(handle.read())
                    else:
                        # Owner died: remove its unfinished transfer too.
                        suffix = name.removeprefix(".reserve-")
                        for stale in (
                            path,
                            os.path.join(self.cache_dir, f".source-{suffix}.tmp"),
                            os.path.join(self.cache_dir, f".source-{suffix}.json.tmp"),
                        ):
                            try:
                                self._unlink(stale)
                            except FileNotFoundError:
                                pass
                finally:
                    os.close(fd)
        return total

    def _evict_idle(self, needed_bytes: int) -> int:
        """Free idle blobs while budget lock is held; never wait on a reader."""
        entries = []
        for name in self._list_cache():
            if name.endswith(".blob"):
                path = os.path.join(self.cache_dir, name)
                info = self._file_metadata(path)
                entries.append((info.st_mtime_ns, path, info.st_size))
        freed = 0
        for _, path, size in sorted(entries):
            if freed >= needed_bytes:
                break
            shard = os.path.basename(path)[:2]
            fd = self._open_lock(os.path.join(self.cache_dir, "locks", f"{shard}.lock"))
            try:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if os.path.lexists(path):
                    self._file_metadata(path)
                    self._unlink(path)
                    try:
                        self._unlink(path[:-5] + ".json")
                    except FileNotFoundError:
                        pass
                    freed += size
            finally:
                os.close(fd)
        return freed

    @contextmanager
    def _reserve(self, size: int, suffix: str) -> Iterator[None]:
        if size > self.max_object_bytes:
            raise ValueError("source exceeds cache object limit")
        reserve_path = os.path.join(self.cache_dir, f".reserve-{suffix}")
        reserve_fd = -1
        deadline = time.monotonic() + self.max_capacity_wait_seconds
        while reserve_fd < 0:
            budget_fd = self._open_lock(os.path.join(self.cache_dir, ".budget.lock"))
            try:
                fcntl.flock(budget_fd, fcntl.LOCK_EX)
                usage = self._budget_usage()
                if usage + size > self.max_size_bytes:
                    self._evict_idle(usage + size - self.max_size_bytes)
                    usage = self._budget_usage()
                if usage + size <= self.max_size_bytes:
                    parent_fd, basename, _ = _open_local_parent(reserve_path, create=False)
                    try:
                        reserve_fd = os.open(
                            basename,
                            os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
                            0o600,
                            dir_fd=parent_fd,
                        )
                    finally:
                        os.close(parent_fd)
                    fcntl.flock(reserve_fd, fcntl.LOCK_EX)
                    os.write(reserve_fd, str(size).encode("ascii"))
                    os.fsync(reserve_fd)
            finally:
                os.close(budget_fd)
            if reserve_fd < 0:
                if time.monotonic() >= deadline:
                    raise RuntimeError("source cache capacity exhausted by active readers or fetches")
                time.sleep(0.02)
        try:
            yield
        finally:
            budget_fd = self._open_lock(os.path.join(self.cache_dir, ".budget.lock"))
            try:
                fcntl.flock(budget_fd, fcntl.LOCK_EX)
                self._unlink(reserve_path)
            finally:
                os.close(budget_fd)
                os.close(reserve_fd)

    def _publish(self, identity: SourceIdentity, fetch: Callable[[str], object], blob: str, manifest: str) -> None:
        suffix = uuid.uuid4().hex
        with self._reserve(identity.size_bytes, suffix):
            self._publish_reserved(identity, fetch, blob, manifest, suffix)

    def _publish_reserved(
        self, identity: SourceIdentity, fetch: Callable[[str], object], blob: str, manifest: str, suffix: str
    ) -> None:
        temporary = os.path.join(self.cache_dir, f".source-{suffix}.tmp")
        manifest_tmp = os.path.join(self.cache_dir, f".source-{suffix}.json.tmp")
        try:
            with self._fetches:
                fetch(temporary)
            metadata = self._file_metadata(temporary)
            if metadata.st_size != identity.size_bytes or metadata.st_size > self.max_object_bytes:
                raise ValueError("downloaded source size conflicts with identity or limit")
            digest = hashlib.sha256()
            fd = self._open_regular(temporary)
            with os.fdopen(fd, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != identity.sha256.lower():
                raise RuntimeError("downloaded source digest conflicts with identity")
            if self._fence(self._file_metadata(temporary)) != self._fence(metadata):
                raise RuntimeError("downloaded source changed during verification")
            chmod_fd = self._open_regular(temporary)
            try:
                os.fchmod(chmod_fd, 0o600)
                verified = os.fstat(chmod_fd)
            finally:
                os.close(chmod_fd)
            parent_fd, source_name, _ = _open_local_parent(temporary, create=False)
            try:
                os.replace(source_name, os.path.basename(blob), src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
            blob_stat = self._file_metadata(blob)
            if blob_stat.st_ino != verified.st_ino or blob_stat.st_dev != verified.st_dev:
                raise RuntimeError("downloaded source changed before publication")
            record = {"identity": identity.canonical(), "fence": self._fence(blob_stat)}
            parent_fd, manifest_name, _ = _open_local_parent(manifest_tmp, create=False)
            try:
                manifest_fd = os.open(
                    manifest_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=parent_fd,
                )
            finally:
                os.close(parent_fd)
            with os.fdopen(manifest_fd, "w", encoding="utf-8") as handle:
                json.dump(record, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            parent_fd, manifest_name, _ = _open_local_parent(manifest_tmp, create=False)
            try:
                os.replace(manifest_name, os.path.basename(manifest), src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            for path in (temporary, manifest_tmp):
                try:
                    self._unlink(path)
                except FileNotFoundError:
                    pass

    @contextmanager
    def lease(
        self,
        identity: SourceIdentity,
        fetch: Callable[[str], object],
        authorize: Callable[[], object],
    ) -> Iterator[str]:
        """Yield a stable path after checking authority on this exact call."""
        if self._pid != os.getpid():
            self._reset_after_fork()
        authorize()
        blob, manifest, lock_path = self._paths(identity)
        entered = False
        fetched_bytes = 0
        try:
            with self._leases:
                fd = self._open_lock(lock_path)
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH)
                    if not self._valid(identity, blob, manifest):
                        fcntl.flock(fd, fcntl.LOCK_UN)
                        fcntl.flock(fd, fcntl.LOCK_EX)
                        if not self._valid(identity, blob, manifest):
                            self._publish(identity, fetch, blob, manifest)
                            fetched_bytes = identity.size_bytes
                        fcntl.flock(fd, fcntl.LOCK_SH)
                    authorize()  # Refresh after any wait for a cold fetch or reader.
                    leased_path = SourceLeasePath(blob, self, self._fence(self._file_metadata(blob)))
                    self._record_lease(fetched_bytes=fetched_bytes)
                    entered = True
                    try:
                        yield leased_path
                    finally:
                        leased_path.check()
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)
        finally:
            if entered:
                self.evict()

    def evict(self) -> None:
        """Remove oldest idle entries if external writes exceeded the cap."""
        lock_fd = self._open_lock(os.path.join(self.cache_dir, ".budget.lock"))
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            usage = self._budget_usage()
            if usage > self.max_size_bytes:
                self._evict_idle(usage - self.max_size_bytes)
        finally:
            os.close(lock_fd)


def _assert_presigned_unexpired(url: str) -> None:
    """Reject a warm hit unless its newly issued URL has checkable expiry."""
    params = parse_qs(urlparse(url).query, keep_blank_values=True)
    if "X-Amz-Date" in params and "X-Amz-Expires" in params:
        try:
            issued = datetime.strptime(params["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            expires = int(params["X-Amz-Expires"][0])
        except (ValueError, IndexError) as exc:
            raise ValueError("presigned URL expiry is invalid") from exc
        if expires <= 0 or datetime.now(timezone.utc).timestamp() >= issued.timestamp() + expires:
            raise ValueError("presigned URL has expired")
        return
    if "Expires" in params:
        try:
            expiry = int(params["Expires"][0])
        except (ValueError, IndexError) as exc:
            raise ValueError("presigned URL expiry is invalid") from exc
        if datetime.now(timezone.utc).timestamp() >= expiry:
            raise ValueError("presigned URL has expired")
        return
    raise ValueError("presigned URL expiry is unavailable for cache reuse")


@contextmanager
def lease_presigned_source(
    cache: ImmutableSourceCache,
    identity: SourceIdentity,
    url: str,
    *,
    authorize: Callable[[], object],
) -> Iterator[str]:
    """Use a freshly issued, job-bound URL to authorize an immutable source."""
    from .presigned_io import _validate_presigned_url, download_presigned

    def checked_authority() -> None:
        authorize()
        _validate_presigned_url(url, job_id=identity.job_id, expected_key=identity.object_key)
        _assert_presigned_unexpired(url)

    def observe_transfer(event: str, count: int) -> None:
        if event == "get_attempt":
            cache._record_metrics(source_http_get_attempts=1)
        elif event == "get_success_bytes":
            cache._record_metrics(source_http_get_success_bytes=count)
        else:
            raise ValueError("unknown source transfer observation")

    with cache.lease(
        identity,
        lambda temporary: download_presigned(
            url,
            temporary,
            job_id=identity.job_id,
            expected_key=identity.object_key,
            max_bytes=identity.size_bytes,
            transfer_observer=observe_transfer,
        ),
        checked_authority,
    ) as path:
        yield path
