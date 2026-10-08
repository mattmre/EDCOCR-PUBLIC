"""Job lifecycle manager — subprocess-based pipeline execution."""

from __future__ import annotations

import hashlib
import heapq
import logging
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import BinaryIO, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from api import config
from api.database import Batch, Job, Tenant
from api.job_log_writer import release_job_log_state, write_job_log
from api.path_safety import resolve_job_output_path, validate_source_path_input

_RESULT_DIR_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_RESULT_FILE_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


def _first_regular_file_under_root(
    root: Path, relative_parts: tuple[str, ...], suffix: str
) -> str | None:
    """Return a lexical path to the first regular file under *root*.

    The walk uses ``O_NOFOLLOW``. A swapped job directory raises ``ELOOP``
    and this helper returns ``None`` instead of advertising the victim tree.
    """
    try:
        current_fd = os.open(root, _RESULT_DIR_OPEN_FLAGS)
    except OSError:
        return None
    try:
        for part in relative_parts:
            try:
                child_fd = os.open(part, _RESULT_DIR_OPEN_FLAGS, dir_fd=current_fd)
            except OSError:
                return None
            os.close(current_fd)
            current_fd = child_fd
        try:
            names = os.listdir(current_fd)
        except OSError:
            return None
        for name in sorted(names):
            if not name.endswith(suffix) or name in {".", ".."}:
                continue
            try:
                file_fd = os.open(name, _RESULT_FILE_OPEN_FLAGS, dir_fd=current_fd)
            except OSError:
                continue
            try:
                metadata = os.fstat(file_fd)
            finally:
                os.close(file_fd)
            if stat.S_ISREG(metadata.st_mode):
                return str(root.joinpath(*relative_parts, name))
        return None
    finally:
        try:
            os.close(current_fd)
        except OSError:
            pass


logger = logging.getLogger(__name__)

# R4-RC-17/52: process-local admit lock (threads). Multi-process uses an
# OS file lock (fcntl) so SQLite multi-worker admits cannot TOCTOU even when
# nested BEGIN IMMEDIATE is unavailable inside an open SQLAlchemy transaction.
_ADMIT_CAPACITY_LOCK = threading.Lock()
_BATCH_RETRY_THREAD_LOCK = threading.Lock()
_ADMIT_ADVISORY_LOCK_KEY = 87421001  # stable pg_advisory_xact_lock key
_UPLOAD_COPY_CHUNK_BYTES = 1024 * 1024
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_SPAWNING_PID = -1  # persisted fence while Popen has no PID to publish yet


def _write_pipeline_heartbeat() -> None:
    """Refresh the API-managed pipeline heartbeat from measured child liveness.

    The standalone API launches one pipeline subprocess per job. Its own
    progress monitor already observes ``proc.poll()`` on every interval, so it
    is the authoritative liveness source for that execution mode. Refresh the
    same heartbeat consumed by detailed readiness while—and only while—the
    child is observed running. This avoids reporting a stale image-layer file
    during real work without weakening the stale-heartbeat failure boundary.
    """
    heartbeat_path = Path(os.environ.get("HEALTHCHECK_FILE", "/app/ocr_healthcheck"))
    try:
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        heartbeat_path.write_text(f"{time.time():.0f}\n", encoding="utf-8")
    except (OSError, ValueError):
        logger.warning(
            "API pipeline progress monitor could not refresh heartbeat %s",
            heartbeat_path,
            exc_info=True,
        )


def _expansion_limit(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Read a dynamic expansion budget with bounded, non-crashing parsing."""
    from ocr_distributed.ocr_utils import get_env_int

    return get_env_int(name, default, min_val=minimum, max_val=maximum)


def _compute_source_hash(filepath: Path | str) -> str:
    """Compute a content hash from the staged source without following links."""
    digest = hashlib.sha256()
    parent_fd = -1
    fd = -1
    path = Path(filepath)
    try:
        parent_fd = os.open(
            str(path.parent),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        fd = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_fd,
        )
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise OSError("source hash requires a regular single-link file")
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            for chunk in iter(lambda: stream.read(8192), b""):
                digest.update(chunk)
    finally:
        if fd >= 0:
            os.close(fd)
        if parent_fd >= 0:
            os.close(parent_fd)
    return digest.hexdigest()


def _kill_pipeline_process(proc: subprocess.Popen) -> None:
    """Kill a pipeline and its descendants when the pipeline owns a session.

    The API launches each pipeline in a fresh POSIX session.  Killing only the
    shell/parent leaves OCR children alive, which can continue writing output
    after a timeout or cancellation and corrupt the terminal state.  Keep the
    single-process fallback for platforms without process groups and for the
    lightweight process doubles used by unit tests.
    """
    popen_type = getattr(subprocess, "Popen", None)
    is_real_process = isinstance(popen_type, type) and isinstance(proc, popen_type)
    if os.name != "nt" and is_real_process:
        try:
            process_group = os.getpgid(proc.pid)
            if process_group == proc.pid:
                os.killpg(process_group, signal.SIGKILL)
                return
        except OSError:
            # The process may have exited between wait/lookup.  The fallback
            # below preserves the existing best-effort cleanup semantics.
            pass
    proc.kill()


def _kill_pipeline_pid(pid: int) -> None:
    """Kill a persisted pipeline PID without taking an unrelated group."""
    if os.name != "nt" and hasattr(os, "killpg"):
        try:
            process_group = os.getpgid(pid)
        except OSError:
            process_group = None
        if process_group == pid:
            os.killpg(process_group, signal.SIGKILL)
            return
    os.kill(pid, getattr(signal, "SIGKILL", 9))


def _pid_belongs_to_job(pid: int, job_id: str) -> bool:
    """Do not signal a reused persisted PID or an unrelated process."""
    if sys.platform != "linux":
        return False
    try:
        environment = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return False
    return f"OCR_JOB_ID={job_id}".encode() in environment.split(b"\0")


def _orphaned_pipeline_pids(job_id: str) -> set[int]:
    """Find live same-user children tagged by this API job, including unpublished PIDs.

    An unreadable same-user environment makes the search inconclusive. Never
    release a persisted PID fence when an orphan could still own the output.
    """
    if sys.platform != "linux":
        raise RuntimeError("Pipeline orphan inspection requires Linux /proc")
    matches = set()
    api_cgroup = Path("/proc/self/cgroup").read_bytes()
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            try:
                metadata = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if metadata.st_uid != os.geteuid():
                continue
            try:
                # A killed child awaiting its original parent's wait() is a
                # zombie: it cannot write even if /proc denies environ.
                proc_stat = Path(entry.path, "stat").read_text()
                if proc_stat.rsplit(") ", 1)[1].startswith("Z "):
                    continue
                environment = Path(entry.path, "environ").read_bytes()
            except FileNotFoundError:
                continue
            except PermissionError:
                # This host can expose same-UID processes from other service
                # cgroups while denying their environments. A pipeline child
                # inherits the API cgroup. Keep unknown local processes fenced.
                try:
                    if Path(entry.path, "cgroup").read_bytes() != api_cgroup:
                        continue
                except FileNotFoundError:
                    continue
                raise
            if f"OCR_JOB_ID={job_id}".encode() in environment.split(b"\0"):
                matches.add(int(entry.name))
    return matches


def _stop_orphaned_pipeline(job_id: str, pid: int | None) -> bool:
    """Stop tagged children before clearing an interrupted job's PID fence."""
    if pid is None:
        return True
    if pid > 0:
        try:
            proc_root = Path(f"/proc/{pid}")
            proc_stat = (proc_root / "stat").read_text()
            if not proc_stat.rsplit(") ", 1)[1].startswith("Z "):
                persisted_environment = (proc_root / "environ").read_bytes()
                if (
                    proc_root.stat().st_uid != os.geteuid()
                    and f"OCR_JOB_ID={job_id}".encode()
                    in persisted_environment.split(b"\0")
                ):
                    logger.error(
                        "Persisted pipeline %s at PID %s changed user; retaining fence",
                        job_id,
                        pid,
                    )
                    return False
        except FileNotFoundError:
            pass
        except OSError:
            logger.error("Cannot establish ownership of persisted PID %s for %s", pid, job_id)
            return False
    deadline = time.monotonic() + 10
    while True:
        try:
            live_pids = _orphaned_pipeline_pids(job_id)
        except OSError:
            logger.exception("Cannot inspect orphaned pipeline for %s", job_id)
            return False
        except RuntimeError:
            logger.exception("Cannot inspect orphaned pipeline for %s", job_id)
            return False
        if not live_pids:
            return True
        for live_pid in live_pids:
            # /proc ownership must be checked again immediately before signal.
            if not _pid_belongs_to_job(live_pid, job_id):
                continue
            try:
                _kill_pipeline_pid(live_pid)
            except ProcessLookupError:
                continue
            except OSError:
                logger.exception("Cannot stop orphaned pipeline %s at %s", job_id, live_pid)
                return False
        if time.monotonic() >= deadline:
            logger.error("Orphaned pipeline %s remained live after SIGKILL", job_id)
            return False
        time.sleep(0.05)


def _open_new_upload_destination(destination: Path):
    """Create a fresh upload file without following/replacing a leaf path."""
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(os.fspath(destination), flags, 0o600)
    return os.fdopen(fd, "wb")


def _max_upload_size_mb() -> int:
    """Return a usable upload cap even when isolated tests stub config."""
    value = getattr(config, "MAX_UPLOAD_SIZE_MB", 5120)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 5120
    return max(0, int(value))


def _max_outstanding_jobs() -> int:
    """Return the bounded admission ceiling for submitted + active jobs.

    Some isolated tests replace ``api.config`` with a partial mock. Fall back
    to the historical concurrency ceiling when the queue setting is absent or
    not a concrete integer, while never permitting a queue smaller than the
    worker pool.
    """
    concurrent = getattr(config, "MAX_CONCURRENT_JOBS", 4)
    if isinstance(concurrent, bool) or not isinstance(concurrent, int):
        concurrent = 4
    concurrent = max(1, concurrent)
    queued = getattr(config, "MAX_QUEUED_JOBS", concurrent)
    if isinstance(queued, bool) or not isinstance(queued, int):
        queued = concurrent
    return max(concurrent, queued)


def _copy_upload_stream(upload_file: BinaryIO, destination: Path) -> None:
    """Copy an upload to disk in bounded chunks and enforce the upload cap.

    FastAPI's ``UploadFile`` is backed by a spooled file, but calling
    ``read()`` without a size turns even disk-backed uploads into one giant
    in-memory ``bytes`` object.  Keep the request path bounded while retaining
    the configured disk-size policy for large documents.
    """
    max_upload_mb = _max_upload_size_mb()
    max_bytes = max_upload_mb * 1024 * 1024
    try:
        upload_file.seek(0)
        head = upload_file.read(8)
        if not isinstance(head, (bytes, bytearray)):
            raise ValueError("Upload stream did not return binary content")
        if head.startswith(b"MZ") or head.startswith(b"\x7fELF"):
            raise ValueError(
                "Executable binary upload refused (PE/ELF magic; B01/S1-05)"
            )
        upload_file.seek(0)
    except (AttributeError, OSError) as exc:
        raise ValueError("Upload stream must support seek/read") from exc

    total = 0
    try:
        with _open_new_upload_destination(destination) as output:
            while True:
                chunk = upload_file.read(_UPLOAD_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray)):
                    raise ValueError("Upload stream did not return binary content")
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Upload exceeds {max_upload_mb} MB limit.")
                output.write(chunk)
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise


def _copy_source_file(source: Path, destination: Path) -> None:
    """Copy a pre-staged source through a no-follow, regular-file handle.

    S3-12: source files are treated as untrusted upload inputs, not executable
    content; PE/ELF magic is rejected before this copy path is reached.
    ``validate_source_path_input`` resolves the path before this function is
    called, but a shared ingest directory can still be changed between that
    check and ``shutil.copy2``.  Open the validated path with ``O_NOFOLLOW``
    where available and copy from the resulting descriptor so a leaf-symlink
    swap cannot redirect the read.  Multi-hardlink inputs are rejected for
    the same foreign-inode reason as output downloads.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(os.fspath(source), flags)
    try:
        source_stat = os.fstat(fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("Source path must reference a regular file")
        if getattr(source_stat, "st_nlink", 1) > 1:
            raise ValueError("Source path has multiple hard links")
        with os.fdopen(fd, "rb") as source_file:
            fd = -1
            head = source_file.read(8)
            if head.startswith(b"MZ") or head.startswith(b"\x7fELF"):
                raise ValueError(
                    "Executable binary source refused (PE/ELF magic; S3-12)"
                )
            source_file.seek(0)
            _copy_upload_stream(source_file, destination)
    finally:
        if fd >= 0:
            os.close(fd)


# ---------------------------------------------------------------------------
# Application-wide JobManager singleton
# ---------------------------------------------------------------------------
_manager_instance: Optional["JobManager"] = None
_manager_instance_lock = threading.Lock()


def get_manager() -> "JobManager":
    """Return the application-wide JobManager singleton.

    The instance is lazily created on first call and reused for the lifetime
    of the process.  This ensures all API requests share a single
    ``PriorityJobQueue`` and worker-thread pool instead of leaking new ones
    on every request.
    """
    global _manager_instance
    if _manager_instance is None:
        with _manager_instance_lock:
            if _manager_instance is None:
                from api.database import get_session_factory

                _manager_instance = JobManager(get_session_factory())
    return _manager_instance


# Priority mapping: lower number = higher priority (for heapq min-heap)
_PRIORITY_MAP = {"urgent": 0, "normal": 1, "low": 2}
_VALID_PRIORITIES = frozenset(_PRIORITY_MAP)


def normalize_job_priority(
    priority: str | None,
    *,
    allow_urgent: bool = False,
) -> str:
    """Normalize client priority; demote unauthorized ``urgent`` (S2-04).

    ``urgent`` is reserved for platform_admin / ``priority_urgent`` permission
    or explicit ``OCR_ALLOW_CLIENT_URGENT=true`` lab waiver. Unknown strings
    fall back to ``normal``.
    """
    p = (priority or "normal").strip().lower()
    if p not in _VALID_PRIORITIES:
        return "normal"
    if p == "urgent" and not allow_urgent:
        return "normal"
    return p


class PriorityJobQueue:
    """Thread-safe priority queue for job processing.

    Jobs are ordered by priority (urgent > normal > low), then by
    submission timestamp (FIFO within the same priority level).
    """

    def __init__(self, max_concurrent: int = 4):
        self._queue: list[tuple[int, float, str]] = []  # (priority, timestamp, job_id)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._active = 0
        self._max_concurrent = max_concurrent
        self._shutdown = False

    def put(self, job_id: str, priority: str = "normal") -> None:
        """Add a job to the priority queue."""
        pri = _PRIORITY_MAP.get(priority, 1)
        with self._not_empty:
            heapq.heappush(self._queue, (pri, time.time(), job_id))
            self._not_empty.notify()

    def get(self) -> str | None:
        """Get the highest-priority job. Blocks until available or shutdown.

        R4-RC-17: honors ``_max_concurrent`` so active workers cannot exceed
        the configured capacity (previously ``_max_concurrent`` was dead).
        """
        with self._not_empty:
            while True:
                if self._shutdown and not self._queue:
                    return None
                # Wait for both a queued job AND free capacity.
                if self._queue and self._active < max(
                    1, int(self._max_concurrent or 1)
                ):
                    break
                if self._shutdown:
                    return None
                self._not_empty.wait(timeout=1.0)
            if not self._queue:
                return None
            _, _, job_id = heapq.heappop(self._queue)
            self._active += 1
            return job_id

    def done(self) -> None:
        """Mark a job as completed (decrement active counter)."""
        with self._not_empty:
            self._active = max(0, self._active - 1)
            self._not_empty.notify()

    def shutdown(self) -> None:
        """Signal shutdown to all waiting workers."""
        with self._not_empty:
            self._shutdown = True
            self._not_empty.notify_all()

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def active(self) -> int:
        with self._lock:
            return self._active


def _coerce_processing_timeout_minutes(raw_value) -> Optional[int]:
    """Return a positive timeout override in minutes, or None if unset/invalid."""
    if raw_value is None:
        return None
    try:
        minutes = int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid processing_timeout_minutes override: %r", raw_value
        )
        return None
    if minutes < 1:
        logger.warning(
            "Ignoring non-positive processing_timeout_minutes override: %r", raw_value
        )
        return None
    raw_max_minutes = getattr(config, "MAX_PROCESSING_TIMEOUT_MINUTES", 10_080)
    max_minutes = (
        int(raw_max_minutes)
        if isinstance(raw_max_minutes, int) and not isinstance(raw_max_minutes, bool)
        else 10_080
    )
    if minutes > max_minutes:
        # API models reject this value, but old durable settings and direct
        # callers still need a defensive bound before it reaches proc.wait().
        logger.warning(
            "Capping processing_timeout_minutes override at %d: %r",
            max_minutes,
            raw_value,
        )
        return max_minutes
    return minutes


def _resolve_processing_timeout_minutes(settings: dict) -> int:
    """Resolve the effective processing timeout in minutes for a job."""
    override = _coerce_processing_timeout_minutes(
        settings.get("processing_timeout_minutes")
    )
    if override is not None:
        return override
    try:
        default_minutes = int(getattr(config, "JOB_PROCESSING_TIMEOUT_MINUTES", 30))
    except (TypeError, ValueError):
        default_minutes = 30
    return max(1, default_minutes)


def _format_processing_timeout(minutes: int) -> str:
    unit = "minute" if minutes == 1 else "minutes"
    return f"{minutes} {unit}"


def _estimate_total_pages(path: Path) -> Optional[int]:
    """Best-effort page estimate for API progress reporting."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            import fitz

            with fitz.open(path) as doc:
                return max(1, int(doc.page_count))
        if suffix in {
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
            ".gif",
            ".webp",
        }:
            return 1
    except Exception:
        logger.debug("Failed to estimate page count for %s", path, exc_info=True)
    return None


def _check_image_pixel_budget(path: Path) -> None:
    """R2-RC-02: refuse oversized image pixel bombs at admit time."""
    suffix = path.suffix.lower()
    image_exts = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".gif",
        ".webp",
        ".jp2",
    }
    if suffix not in image_exts:
        return
    max_pixels = _expansion_limit(
        "MAX_IMAGE_PIXELS",
        100_000_000,
        minimum=0,
        maximum=2_000_000_000,
    )
    if max_pixels <= 0:
        return
    try:
        from PIL import Image

        # Do not lower PIL's bomb limit below ours before open — that raises
        # DecompressionBombError which must not be swallowed as a soft pass.
        try:
            from PIL.Image import DecompressionBombError
        except Exception:  # pragma: no cover
            DecompressionBombError = type("DecompressionBombError", (Exception,), {})

        try:
            with Image.open(path) as img:
                w, h = img.size
                pixels = int(w) * int(h)
                frames = max(1, int(getattr(img, "n_frames", 1) or 1))
                total = pixels * frames
                if total > max_pixels:
                    raise ValueError(
                        f"Image pixel budget exceeded: {total} > "
                        f"MAX_IMAGE_PIXELS={max_pixels} "
                        f"({w}x{h} x {frames} frames; R2-RC-02)"
                    )
        except DecompressionBombError as exc:
            raise ValueError(
                f"Image pixel budget exceeded (PIL bomb): MAX_IMAGE_PIXELS={max_pixels} "
                f"(R2-RC-02): {exc}"
            ) from exc
    except ValueError:
        raise
    except Exception as exc:
        # PB-A / Tier B: soft-pass on probe failure reopened expansion bombs.
        raise ValueError(
            f"Image pixel budget probe failed (fail-closed R2-RC-02): {path}: {exc}"
        ) from exc


@contextmanager
def _cross_process_admit_lock(suffix: str = ""):
    """OS-level exclusive lock for multi-process admit (R4-RC-17/52).

    Works for SQLite default API DB where nested BEGIN IMMEDIATE may be a no-op
    inside an already-open SQLAlchemy transaction. Uses fcntl on POSIX; on
    platforms without fcntl falls back to process lock only (caller also holds
    ``_ADMIT_CAPACITY_LOCK``).
    """
    lock_path = os.environ.get("ADMIT_CAPACITY_LOCK_PATH", "").strip()
    if not lock_path:
        out = getattr(config, "OUTPUT_FOLDER", None) or os.environ.get(
            "OCR_OUTPUT_DIR", tempfile.gettempdir()
        )
        lock_path = str(Path(out) / ".admit_capacity.lock")
    lock_path += suffix
    try:
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # A host-side control-plane operation can run before the container's
        # configured output volume exists (for example, tenant bootstrap or
        # key creation in a local lab).  Preserve a deterministic lock rather
        # than failing solely because the configured path is not writable.
        # Production shared-volume deployments still use their configured
        # path; the warning keeps the fallback from being mistaken for a
        # cross-host coordination proof.
        fallback_name = (
            "ocr-local-admit-"
            + hashlib.sha256(lock_path.encode("utf-8")).hexdigest()[:20]
            + ".lock"
        )
        fallback_path = Path(tempfile.gettempdir()) / fallback_name
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "Admission lock path %s is unavailable (%s); using local fallback %s",
            lock_path,
            exc,
            fallback_path,
        )
        lock_path = str(fallback_path)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            locked = True
        except Exception:
            locked = False
            logger.debug("fcntl admit lock unavailable; process lock only")
        try:
            yield
        finally:
            if locked:
                try:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
    finally:
        fh.close()


@contextmanager
def _batch_retry_serialization_lock():
    """Serialize batch retry, individual retry, and abandoned-claim recovery."""
    with _BATCH_RETRY_THREAD_LOCK:
        with _cross_process_admit_lock(".batch-retry"):
            yield


def _serialize_staging_and_admission(method):
    """Keep tenant purge behind uploads until staging is committed or removed.

    The bounded upload may delay admission on this standalone API. Holding the
    same lock before the first staging write prevents purge's DB snapshot from
    missing tenant-owned bytes that have no Job row yet.
    """

    @wraps(method)
    def serialized(self, *args, **kwargs):
        with _ADMIT_CAPACITY_LOCK:
            with _cross_process_admit_lock():
                tenant_id = kwargs.get("tenant_id")
                if tenant_id:
                    with self._session_factory() as session:
                        tenant = session.get(Tenant, tenant_id)
                        if tenant is None or tenant.status != "active":
                            raise ValueError("Tenant not found or inactive")
                retry_of = kwargs.get("retry_of")
                if retry_of:
                    with self._session_factory() as session:
                        original = session.get(Job, retry_of)
                        if original is None or original.tenant_id != tenant_id:
                            raise ValueError("Retry source no longer exists")
                        if original.retried_to:
                            replacement = session.get(Job, original.retried_to)
                            if replacement is None:
                                raise ValueError("Retry replacement no longer exists")
                            replacement_settings = (
                                replacement.settings
                                if isinstance(replacement.settings, dict)
                                else {}
                            )
                            if (
                                replacement.job_id == original.job_id
                                or replacement.retried_to is not None
                                or replacement_settings.get("retried_to")
                                or replacement_settings.get("superseded")
                            ):
                                raise ValueError(
                                    "Retry replacement link is self-referential or superseded"
                                )
                            if replacement.tenant_id != original.tenant_id:
                                raise ValueError(
                                    "Retry replacement tenant does not match source"
                                )
                            allowed_batch_ids = {
                                None,
                                original.batch_id,
                                kwargs.get("batch_id"),
                            }
                            if replacement.batch_id not in allowed_batch_ids:
                                raise ValueError(
                                    "Retry replacement belongs to another batch"
                                )
                            return replacement
                return method(self, *args, **kwargs)

    return serialized


def _acquire_admit_capacity_lock(session: Session):
    """DB-side exclusive lock when available (Postgres advisory; SQLite best-effort).

    Primary multi-process serialization is ``_cross_process_admit_lock`` (fcntl).
    """
    bind = session.get_bind()
    dialect = getattr(getattr(bind, "dialect", None), "name", "") or ""
    conn = session.connection()
    if dialect == "sqlite":
        try:
            conn.execute(text("BEGIN IMMEDIATE"))
        except Exception:
            logger.debug("sqlite BEGIN IMMEDIATE unavailable; fcntl lock is primary")
    elif dialect in ("postgresql", "postgres"):
        conn.execute(
            text("SELECT pg_advisory_xact_lock(:k)"),
            {"k": _ADMIT_ADVISORY_LOCK_KEY},
        )


def _job_output_root_is_symlink(output_root: Path) -> bool:
    """True when the job component itself was replaced with a directory symlink."""
    try:
        return output_root.is_symlink()
    except OSError:
        return True


_MAX_RECOVERY_OUTPUT_ENTRIES = 100_000
_MAX_RECOVERY_OUTPUT_DEPTH = 64


def _count_regular_suffix_from_fd(
    root_fd: int,
    relative_parts: tuple[str, ...],
    suffix: str,
) -> int:
    """Count owned artifacts below a held job root without reopening its path."""
    current_fd = os.dup(root_fd)
    try:
        for part in relative_parts:
            next_fd = os.open(part, _RESULT_DIR_OPEN_FLAGS, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd

        visited = 0

        def _walk(directory_fd: int, depth: int) -> int:
            nonlocal visited
            if depth > _MAX_RECOVERY_OUTPUT_DEPTH:
                raise RuntimeError("recovery output tree exceeds depth limit")
            count = 0
            names: list[str] = []
            with os.scandir(directory_fd) as iterator:
                for entry in iterator:
                    visited += 1
                    if visited > _MAX_RECOVERY_OUTPUT_ENTRIES:
                        raise RuntimeError("recovery output tree exceeds entry limit")
                    names.append(entry.name)
            for name in names:
                before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISLNK(before.st_mode):
                    continue
                if stat.S_ISDIR(before.st_mode):
                    child_fd = os.open(
                        name,
                        _RESULT_DIR_OPEN_FLAGS,
                        dir_fd=directory_fd,
                    )
                    try:
                        opened = os.fstat(child_fd)
                        if (before.st_dev, before.st_ino) != (
                            opened.st_dev,
                            opened.st_ino,
                        ):
                            raise RuntimeError("recovery output directory changed")
                        count += _walk(child_fd, depth + 1)
                        after = os.stat(
                            name,
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                        if (after.st_dev, after.st_ino) != (
                            opened.st_dev,
                            opened.st_ino,
                        ):
                            raise RuntimeError("recovery output directory changed")
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(before.st_mode) and name.endswith(suffix):
                    file_fd = os.open(
                        name, _RESULT_FILE_OPEN_FLAGS, dir_fd=directory_fd
                    )
                    try:
                        opened = os.fstat(file_fd)
                        if (before.st_dev, before.st_ino) != (
                            opened.st_dev,
                            opened.st_ino,
                        ) or opened.st_nlink != 1:
                            continue
                        after = os.stat(
                            name,
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                        if (after.st_dev, after.st_ino) != (
                            opened.st_dev,
                            opened.st_ino,
                        ):
                            raise RuntimeError("recovery output file changed")
                        count += 1
                    finally:
                        os.close(file_fd)
            return count

        return _walk(current_fd, 0)
    except FileNotFoundError:
        return 0
    finally:
        os.close(current_fd)


def _scan_recovery_outputs(output_dir: Path) -> tuple[int, int]:
    """Return PDF/text counts from one lexically verified job-root binding."""
    try:
        before = os.stat(output_dir, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
            return 0, 0
        root_fd = os.open(output_dir, _RESULT_DIR_OPEN_FLAGS)
    except OSError:
        return 0, 0
    try:
        opened = os.fstat(root_fd)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            return 0, 0
        pdf_count = _count_regular_suffix_from_fd(
            root_fd,
            ("EXPORT", "PDF"),
            ".pdf",
        )
        text_count = _count_regular_suffix_from_fd(
            root_fd,
            ("EXPORT", "TEXT"),
            ".txt",
        )
        after = os.stat(output_dir, follow_symlinks=False)
        if (
            not stat.S_ISDIR(after.st_mode)
            or stat.S_ISLNK(after.st_mode)
            or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            return 0, 0
        return pdf_count, text_count
    except (OSError, RuntimeError):
        return 0, 0
    finally:
        os.close(root_fd)


def _is_owned_output_file(path: Path, output_root: Path | None = None) -> bool:
    """Return true only for an owned regular file with no symlinked parent."""
    if output_root is not None:
        if _job_output_root_is_symlink(output_root):
            return False
        try:
            path.relative_to(output_root)
        except ValueError:
            return False
        cursor = path
        while True:
            if cursor.is_symlink():
                return False
            if cursor == output_root:
                break
            parent = cursor.parent
            if parent == cursor:
                return False
            cursor = parent
    elif path.is_symlink():
        return False
    try:
        file_stat = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(file_stat.st_mode) and file_stat.st_nlink == 1


def _has_completed_artifacts(output_dir: Path) -> bool:
    pdf_count, text_count = _scan_recovery_outputs(output_dir)
    return pdf_count > 0 and text_count > 0


def recover_interrupted_jobs(db_session_factory=None) -> int:
    """Reconcile jobs left active by an API restart or killed worker process."""
    if db_session_factory is None:
        from api.database import get_session_factory

        db_session_factory = get_session_factory()

    recovered = 0
    unresolved = 0
    session: Session = db_session_factory()
    try:
        # A crash can leave a large submitted/processing backlog. Recovery
        # must reconcile every row, but it must not first materialize the
        # entire backlog in the API process.
        recovery_query = (
            session.query(Job)
            .filter(
                (Job.status.in_(["submitted", "processing"]))
                | ((Job.status == "cancelled") & (Job.pid.is_not(None)))
            )
            .yield_per(1_000)
        )
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for job in recovery_query:
            if not _stop_orphaned_pipeline(job.job_id, job.pid):
                logger.error(
                    "Retaining PID fence for interrupted job %s; writer state is unknown",
                    job.job_id,
                )
                unresolved += 1
                continue
            if job.status == "cancelled":
                job.pid = None
                recovered += 1
                continue
            try:
                output_dir = resolve_job_output_path(
                    job.job_id,
                    job.result_path,
                    config.OUTPUT_FOLDER,
                )
            except ValueError:
                # A persisted path is not allowed to redirect restart
                # recovery into another job's output tree.
                logger.error(
                    "Refusing restart recovery for job %s with non-canonical output path",
                    job.job_id,
                )
                job.status = "failed"
                job.current_stage = "failed"
                job.error_message = "Job output path is invalid; recovery refused"
                job.pid = None
                job.completed_at = now
                recovered += 1
                continue
            if _has_completed_artifacts(output_dir):
                job.status = "completed"
                job.current_stage = "completed"
                job.error_message = None
                # The API writes one consolidated TEXT export per job, not one
                # text file per page. Once the complete PDF/TEXT pair exists,
                # use only the page count captured at admission; artifact count
                # cannot establish page progress.
                if job.total_pages:
                    job.pages_completed = max(
                        int(job.pages_completed or 0), int(job.total_pages)
                    )
            else:
                job.status = "failed"
                job.current_stage = "failed"
                job.error_message = "Job interrupted by API restart before completion"
            job.pid = None
            job.completed_at = now
            if job.started_at and job.processing_time is None:
                job.processing_time = max(0.0, (now - job.started_at).total_seconds())
            recovered += 1
        if recovered:
            session.commit()
            logger.info("Recovered %d interrupted OCR job(s) on startup", recovered)
        if unresolved:
            raise RuntimeError(
                f"Could not verify {unresolved} interrupted pipeline writer(s); "
                "refusing to serve stale job state"
            )
        return recovered
    finally:
        session.close()


class JobManager:
    """Manages OCR job submission, tracking, and result retrieval."""

    def __init__(self, db_session_factory):
        self._session_factory = db_session_factory
        self._priority_queue = PriorityJobQueue(
            max_concurrent=config.MAX_CONCURRENT_JOBS
        )
        self._workers_started = False
        self._workers_lock = threading.Lock()
        # maintenance review: track worker threads so shutdown() can join them gracefully.
        # Worker threads are non-daemon: relying on daemon=True killed in-flight
        # jobs without cleanup or audit trail at interpreter exit.
        self._worker_threads: list[threading.Thread] = []
        self._shutdown_called = False
        self._reconciler_stop = threading.Event()
        self._reconciler_thread: threading.Thread | None = None
        self._db_load_failures: dict[str, int] = {}
        self._db_load_failures_lock = threading.Lock()
        self._batch_reconcile_cursor = ""
        self._pipeline_condition = threading.Condition()
        self._active_pipeline_processes: dict[str, subprocess.Popen | None] = {}

    def start(self) -> None:
        """Start workers and repair loops for the application lifespan."""
        self._ensure_workers()

    def _requeue_after_db_failure(self, job_id: str) -> None:
        if self._shutdown_called:
            return
        with self._db_load_failures_lock:
            failures = min(self._db_load_failures.get(job_id, 0) + 1, 8)
            self._db_load_failures[job_id] = failures
        time.sleep(min(5.0, 0.25 * (2 ** (failures - 1))))
        self._priority_queue.put(job_id, priority="normal")

    def _ensure_workers(self) -> None:
        """Start background worker threads if not already started."""
        if self._workers_started:
            return
        with self._workers_lock:
            if self._workers_started:
                return
            self._workers_started = True
            max_workers = config.MAX_CONCURRENT_JOBS
            for i in range(max_workers):
                t = threading.Thread(
                    target=self._worker_loop,
                    name=f"job-worker-{i}",
                    daemon=False,
                )
                t.start()
                self._worker_threads.append(t)
            reconciler = threading.Thread(
                target=self._batch_reconciler_loop,
                name="batch-terminal-reconciler",
                daemon=False,
            )
            reconciler.start()
            self._reconciler_thread = reconciler

    def _batch_reconciler_loop(self) -> None:
        """Periodically repair parent batches after a transient final-child fault."""
        from api.background_work import register_writer
        from api.batch_manager import BatchManager

        while not self._reconciler_stop.is_set():
            session = None
            finish = register_writer()
            try:
                # Retry requests hold this process-shared lock for the full
                # multi-child admission window. Once acquired here, any
                # remaining retry_claimed rows are abandoned and can be
                # safely recovered before checking terminal batch state.
                with _batch_retry_serialization_lock():
                    session = self._session_factory()
                    query = session.query(Batch.batch_id).filter(
                        Batch.status.notin_(list(_TERMINAL_STATUSES))
                    )
                    if self._batch_reconcile_cursor:
                        query = query.filter(
                            Batch.batch_id > self._batch_reconcile_cursor
                        )
                    batch_ids = [
                        row[0]
                        for row in query.order_by(Batch.batch_id.asc())
                        .limit(1_000)
                        .all()
                    ]
                    if not batch_ids and self._batch_reconcile_cursor:
                        self._batch_reconcile_cursor = ""
                        continue
                    if batch_ids:
                        self._batch_reconcile_cursor = batch_ids[-1]
                    manager = BatchManager(self._session_factory, job_manager=self)
                    for batch_id in batch_ids:
                        if self._reconciler_stop.is_set():
                            break
                        manager.recover_abandoned_retry_claims(batch_id)
                        manager.check_batch_completion(batch_id)
                    self._expire_api_results(session)
            except Exception:
                logger.exception("Periodic batch terminal reconciliation failed")
            finally:
                if session is not None:
                    try:
                        session.close()
                    except Exception:
                        logger.exception("Batch reconciler failed to close DB session")
                finish()
            self._reconciler_stop.wait(30.0)

    def _expire_api_results(self, session: Session) -> None:
        """Remove terminal API result trees after RESULT_RETENTION_DAYS."""
        if os.environ.get("LITIGATION_HOLD", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
            days=max(1, int(config.RESULT_RETENTION_DAYS))
        )
        jobs = (
            session.query(Job)
            .filter(
                Job.status.in_(list(_TERMINAL_STATUSES)),
                Job.completed_at.is_not(None),
                Job.completed_at < cutoff,
                Job.result_path.is_not(None),
            )
            .limit(100)
            .all()
        )
        held_tenants = {
            value.strip()
            for value in os.environ.get("LITIGATION_HOLD_TENANTS", "").split(",")
            if value.strip()
        }
        for job in jobs:
            if str(job.tenant_id or "").strip() in held_tenants:
                continue
            try:
                output_path = resolve_job_output_path(
                    job.job_id, job.result_path, config.OUTPUT_FOLDER
                )
                if output_path.is_symlink():
                    raise ValueError("result path is symlinked")
                if output_path.exists():
                    shutil.rmtree(output_path)
                job.result_path = None
                session.commit()
                self._publish_job_event(job, "job.results_expired")
            except Exception:
                session.rollback()
                logger.exception("Failed to expire API results for %s", job.job_id)

    def shutdown(self, timeout: float = 30.0) -> bool:
        """Gracefully shut down worker threads.

        Signals the priority queue to stop, then waits up to ``timeout``
        seconds total for all worker threads to finish their in-flight jobs
        and exit.  Workers are non-daemon (maintenance review), so this method MUST be
        called at process shutdown to avoid orphaning in-flight jobs without
        cleanup or audit trail.

        Safe to call multiple times; subsequent calls recheck remaining writers.

        Args:
            timeout: Maximum total seconds to wait for all workers to exit.

        Returns:
            True if all worker threads exited within the timeout; False
            otherwise.  A False return indicates at least one worker is
            still running an in-flight job past the deadline.
        """
        with self._workers_lock:
            self._shutdown_called = True
            threads = list(self._worker_threads)
            reconciler = self._reconciler_thread

        deadline = time.monotonic() + max(timeout, 0.0)
        self._reconciler_stop.set()
        if reconciler is not None:
            reconciler.join(timeout=min(max(timeout, 0.0), 5.0))
        all_exited = reconciler is None or not reconciler.is_alive()

        if not threads:
            # Workers never started; nothing to do.
            self._priority_queue.shutdown()
            return all_exited

        logger.info(
            "JobManager.shutdown: signalling %d worker thread(s), timeout=%.1fs",
            len(threads),
            timeout,
        )
        # Signal the priority queue so any worker currently blocked in get()
        # wakes up and returns None, exiting the loop.
        self._priority_queue.shutdown()

        # Join each worker against a shared deadline so the aggregate wall
        # clock does not exceed ``timeout`` even if several workers need to
        # drain in-flight jobs.
        for t in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                remaining = 0
            t.join(timeout=remaining)
            if t.is_alive():
                all_exited = False
                logger.warning(
                    "JobManager.shutdown: worker %s did not exit within timeout",
                    t.name,
                )

        if all_exited:
            logger.info("JobManager.shutdown: all worker threads exited cleanly")
        return all_exited

    def _worker_loop(self) -> None:
        """Worker thread that processes jobs from the priority queue."""
        while True:
            job_id = self._priority_queue.get()
            if job_id is None:
                break
            try:
                # Look up the job's source/output/settings before running
                # pipeline.  Every branch remains inside the slot-release
                # guard below, including missing/cancelled jobs and DB errors.
                try:
                    session: Session = self._session_factory()
                except Exception:
                    logger.exception(
                        "Worker: database session unavailable for %s", job_id
                    )
                    self._requeue_after_db_failure(job_id)
                    continue
                job = None
                try:
                    job = session.get(Job, job_id)
                    if not job:
                        logger.warning("Worker: job %s not found, skipping", job_id)
                        continue
                    # R2-RC-05 sticky-cancel: never start pipeline after cancel committed.
                    if job.status == "cancelled":
                        logger.info(
                            "Worker: job %s already cancelled, skipping pipeline",
                            job_id,
                        )
                        continue
                    source_dir = str(Path(config.SOURCE_FOLDER) / job_id)
                    output_dir = str(
                        resolve_job_output_path(
                            job.job_id,
                            job.result_path,
                            config.OUTPUT_FOLDER,
                        )
                    )
                    settings = job.settings
                except ValueError:
                    # A persisted result_path is data, not an authority.  If
                    # it no longer resolves to this job's canonical output
                    # root, fail the job explicitly instead of leaving it in
                    # ``submitted`` forever after the queue slot is released.
                    if job is not None:
                        job.status = "failed"
                        job.current_stage = "failed"
                        job.error_message = (
                            "Job output path is invalid; processing refused"
                        )
                        job.pid = None
                        job.completed_at = datetime.now(timezone.utc).replace(
                            tzinfo=None
                        )
                        session.commit()
                        session.refresh(job)
                    logger.error(
                        "Worker: refusing job %s with invalid persisted output path",
                        job_id,
                        exc_info=True,
                    )
                    if job is not None:
                        try:
                            self._publish_job_event(job, "job.failed")
                        except Exception:
                            logger.exception(
                                "Worker: failed to publish invalid-output failure for %s",
                                job_id,
                            )
                    continue
                except Exception:
                    logger.exception("Worker: failed to load job %s", job_id)
                    self._requeue_after_db_failure(job_id)
                    continue
                finally:
                    try:
                        session.close()
                    except Exception:
                        logger.exception(
                            "Worker: failed to close DB session for %s", job_id
                        )

                with self._db_load_failures_lock:
                    self._db_load_failures.pop(job_id, None)

                try:
                    self._run_pipeline(job_id, source_dir, output_dir, settings)
                except Exception:
                    logger.exception("Worker error processing job %s", job_id)
            finally:
                # PriorityJobQueue.get() increments active before returning;
                # release that slot even when the lookup above continues.
                self._priority_queue.done()

    def _publish_job_event(self, job, event_type: str) -> None:
        """Persist and forward a job lifecycle event."""
        from api.events import publish_job_event

        publish_job_event(job, event_type)
        # maintenance review: signal waiting SSE generators to avoid polling delay
        try:
            from api.sse_notifier import notify_job_update

            notify_job_update(job.job_id)
        except Exception:
            pass  # Non-critical: SSE falls back to timeout polling

    def _fire_webhook(self, job) -> None:
        """Start webhook delivery for a job if a webhook URL is configured."""
        if not job.webhook_url:
            return
        from api.webhooks import start_webhook_delivery

        start_webhook_delivery(
            job.job_id,
            self._session_factory,
            webhook_timeout=config.WEBHOOK_TIMEOUT,
            webhook_max_retries=config.WEBHOOK_MAX_RETRIES,
            webhook_secret_default=config.WEBHOOK_SECRET,
        )

    def _check_batch_completion(self, job) -> None:
        """If the job belongs to a batch, check if the batch is complete."""
        batch_id = getattr(job, "batch_id", None)
        if not batch_id:
            return
        from api.batch_manager import BatchManager

        for attempt in range(3):
            try:
                batch_mgr = BatchManager(self._session_factory, job_manager=self)
                batch_mgr.check_batch_completion(batch_id)
                return
            except Exception:
                logger.exception(
                    "Error checking batch completion for batch %s (job %s), attempt %d/3",
                    batch_id,
                    job.job_id,
                    attempt + 1,
                )
                if attempt < 2:
                    time.sleep(0.25 * (2**attempt))

    # ------------------------------------------------------------------
    # Dashboard collector wiring
    # ------------------------------------------------------------------

    def _update_dashboard_on_complete(self, job, session: Session) -> None:
        """Push job completion data to in-memory dashboard collectors.

        All calls are guarded so dashboard failures never break the
        pipeline.
        """
        _tenant = getattr(job, "tenant_id", "") or ""
        try:
            from api.dashboard import get_collector

            collector = get_collector()
            collector.record_throughput(
                pages=job.pages_completed or 0,
                documents=1,
                bytes_processed=0,
                tenant_id=_tenant,
            )
            collector.record_latency(
                job_id=str(job.job_id),
                total_ms=int((job.processing_time or 0) * 1000),
                tenant_id=_tenant,
            )

            total = session.query(Job).count()
            active = (
                session.query(Job)
                .filter(Job.status.in_(["submitted", "processing"]))
                .count()
            )
            completed = session.query(Job).filter(Job.status == "completed").count()
            failed = session.query(Job).filter(Job.status == "failed").count()
            queued = session.query(Job).filter(Job.status == "submitted").count()
            collector.update_job_counts(
                total=total,
                active=active,
                completed=completed,
                failed=failed,
                queued=queued,
                tenant_id=_tenant,
            )
        except Exception:
            logger.debug("Dashboard metrics collector update failed", exc_info=True)

        try:
            from api.analytics import get_analytics_store

            get_analytics_store().record_job(
                job_id=str(job.job_id),
                pages=job.pages_completed or 0,
                duration_seconds=job.processing_time or 0.0,
                success=(job.status == "completed"),
                tenant_id=_tenant,
            )
        except Exception:
            logger.debug("Analytics store update failed", exc_info=True)

        try:
            from api.queue_alerting import get_queue_monitor

            active_count = (
                session.query(Job)
                .filter(Job.status.in_(["submitted", "processing"]))
                .count()
            )
            get_queue_monitor().record_depth("processing", active_count)
        except Exception:
            logger.debug("Queue monitor update failed", exc_info=True)

        try:
            from sla_monitoring import get_monitor

            monitor = get_monitor()
            if monitor:
                _sla_tenant = _tenant or "default"
                _success = job.status == "completed"
                _latency = job.processing_time or 0.0  # already seconds
                monitor.record_request(_sla_tenant, _success, _latency)
        except Exception:
            logger.debug("SLA monitor feed failed", exc_info=True)

    def _update_dashboard_on_submit(self, active_count: int) -> None:
        """Push submission data to in-memory dashboard collectors."""
        try:
            from api.dashboard import get_collector

            get_collector().update_stage(
                "processing", active_workers=1, queue_depth=active_count
            )
        except Exception:
            logger.debug("Dashboard stage update on submit failed", exc_info=True)

        try:
            from api.queue_alerting import get_queue_monitor

            get_queue_monitor().record_depth("processing", active_count)
        except Exception:
            logger.debug("Queue monitor update on submit failed", exc_info=True)

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    @_serialize_staging_and_admission
    def submit(
        self,
        *,
        source_path: Optional[str] = None,
        upload_filename: Optional[str] = None,
        upload_content: Optional[bytes] = None,
        upload_file: Optional[BinaryIO] = None,
        tenant_id: Optional[str] = None,
        priority: str = "normal",
        enable_docintel: bool = False,
        docintel_mode: str = "full",
        skip_ocr: bool = False,
        processing_timeout_minutes: Optional[int] = None,
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
        batch_id: Optional[str] = None,
        retry_of: Optional[str] = None,
    ) -> Job:
        """Create a job and prepare its source folder.

        ``batch_id`` is written in the same transaction as the child job.  A
        batch caller must not enqueue a child and attach it afterward: a fast
        worker can otherwise finish before completion accounting can see the
        membership.
        """
        if skip_ocr:
            # The standalone API launches ocr_gpu_async.py, whose verified
            # production entrypoint has no native-text bypass flag. Do not
            # accept-and-drop a forensic-affecting request option; the
            # coordinator's process_text_only task is a separate path.
            raise ValueError("skip_ocr is not supported by the standalone API pipeline")
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job_source = Path(config.SOURCE_FOLDER) / job_id
        job_output = Path(config.OUTPUT_FOLDER) / job_id
        job_source.mkdir(parents=True, exist_ok=True)
        job_output.mkdir(parents=True, exist_ok=True)

        def _staging_dir_is_owned(directory: Path, root: str) -> bool:
            """Require a job staging directory to be a real canonical dir."""
            try:
                expected = Path(root).expanduser().resolve(strict=False) / job_id
                return (
                    not directory.is_symlink()
                    and directory.is_dir()
                    and directory.resolve(strict=False) == expected
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                return False

        def _remove_staged_directory(directory: Path) -> None:
            """Remove a staging dir without ever traversing a symlink."""
            if directory.is_symlink():
                directory.unlink()
            else:
                shutil.rmtree(directory)

        def _cleanup_staged_dirs() -> None:
            """Remove directories created for a job that never committed."""
            for directory in (job_source, job_output):
                try:
                    _remove_staged_directory(directory)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning(
                        "Could not clean rejected job staging directory: %s",
                        directory,
                        exc_info=True,
                    )

        if not _staging_dir_is_owned(job_source, config.SOURCE_FOLDER) or not (
            _staging_dir_is_owned(job_output, config.OUTPUT_FOLDER)
        ):
            _cleanup_staged_dirs()
            raise ValueError("Job staging path is not canonical")

        # Resolve source file
        if upload_filename and (upload_content is not None or upload_file is not None):
            safe_name = os.path.basename(upload_filename)
            if not safe_name or safe_name.startswith("."):
                _cleanup_staged_dirs()
                raise ValueError(f"Invalid upload filename: {upload_filename!r}")
            dest = job_source / safe_name
            if upload_content is not None and upload_file is not None:
                _cleanup_staged_dirs()
                raise ValueError("Provide only one upload content source")
            if upload_content is not None:
                # S1-05 / B01: refuse PE/ELF before any durable write (do not
                # land PE-as-PDF on disk via extension-only names).
                head = (
                    upload_content[:8]
                    if isinstance(upload_content, (bytes, bytearray))
                    else b""
                )
                if head.startswith(b"MZ") or head.startswith(b"\x7fELF"):
                    _cleanup_staged_dirs()
                    raise ValueError(
                        "Executable binary upload refused (PE/ELF magic; B01/S1-05)"
                    )
                max_upload_mb = _max_upload_size_mb()
                max_bytes = max_upload_mb * 1024 * 1024
                if len(upload_content) > max_bytes:
                    _cleanup_staged_dirs()
                    raise ValueError(f"Upload exceeds {max_upload_mb} MB limit.")
                try:
                    with _open_new_upload_destination(dest) as output:
                        output.write(upload_content)
                except Exception:
                    _cleanup_staged_dirs()
                    raise
            else:
                try:
                    _copy_upload_stream(upload_file, dest)
                except Exception:
                    _cleanup_staged_dirs()
                    raise
            filename = safe_name
        elif source_path:
            try:
                src = validate_source_path_input(
                    path_value=source_path,
                    field_name="source_path",
                    allowed_roots=(config.SOURCE_FOLDER,),
                )
            except Exception:
                _cleanup_staged_dirs()
                raise
            if not src.exists():
                _cleanup_staged_dirs()
                logger.warning(
                    "Source path was not found during job submission: %s", source_path
                )
                raise FileNotFoundError("Source file not found")
            dest = job_source / src.name
            try:
                _copy_source_file(src, dest)
            except (OSError, ValueError) as exc:
                _cleanup_staged_dirs()
                if isinstance(exc, ValueError):
                    raise
                logger.warning(
                    "Source path could not be read during job submission: %s",
                    source_path,
                    exc_info=True,
                )
                raise ValueError("Source file could not be read") from exc
            filename = src.name
        else:
            _cleanup_staged_dirs()
            raise ValueError("Either source_path or file upload is required.")

        settings = {
            "enable_docintel": enable_docintel,
            "docintel_mode": docintel_mode,
        }
        if processing_timeout_minutes is not None:
            settings["processing_timeout_minutes"] = processing_timeout_minutes
        source_size = dest.stat().st_size
        try:
            # The standalone API does not run the coordinator ingest task.
            # Establish the content identity before admission so every API
            # job can export a truthful DocumentBundle source hash.
            source_hash = _compute_source_hash(dest)
        except OSError as exc:
            _cleanup_staged_dirs()
            raise ValueError("Source file could not be hashed") from exc

        session: Session = self._session_factory()
        job_persisted = False
        try:
            # Expansion budgets before any capacity slot is consumed.
            max_pages = _expansion_limit(
                "MAX_PDF_PAGES", 5000, minimum=1, maximum=1_000_000
            )
            estimated_pages = _estimate_total_pages(dest)
            # S3-11: PDF page budget must not skip when estimate fails.
            # Fail closed for PDF sources; non-PDF may still return None.
            if estimated_pages is None and dest.suffix.lower() == ".pdf":
                raise ValueError(
                    "Unable to estimate PDF page count; refusing admit "
                    f"(MAX_PDF_PAGES={max_pages}; S3-11 fail-closed)"
                )
            if estimated_pages is not None and estimated_pages > max_pages:
                raise ValueError(
                    f"Document page estimate {estimated_pages} exceeds "
                    f"MAX_PDF_PAGES={max_pages} (R2-RC-02 expansion budget)"
                )
            _check_image_pixel_budget(dest)

            # Tenant admission is intentionally resolved inside the same
            # serialization window as the capacity/quota checks.  Looking up
            # an active Tenant before the lock lets a concurrent suspend,
            # delete, or hard purge commit a terminal state while this session
            # still holds a stale ORM object.
            tenant = None

            # Encrypt webhook secret before storing at rest (SEC-001)
            encrypted_secret = None
            if webhook_secret:
                from api.config import encrypt_webhook_secret

                encrypted_secret = encrypt_webhook_secret(webhook_secret)

            # R4-RC-17/52: atomic capacity — thread lock + fcntl multi-process
            # lock + DB exclusive when available, then count→insert.
            _acquire_admit_capacity_lock(session)
            if tenant_id:
                tenant = session.get(Tenant, tenant_id)
                if not tenant or tenant.status != "active":
                    raise ValueError(f"Tenant not found or inactive: {tenant_id}")
            active = (
                session.query(Job)
                .filter(Job.status.in_(["submitted", "processing"]))
                .count()
            )
            if active >= _max_outstanding_jobs():
                raise ValueError("Job queue is at capacity")

            # Tenant quota reads must be inside the same admission
            # lock as the insert.  Checking them before the lock lets
            # concurrent submissions all observe the same headroom
            # and overshoot max_concurrent_jobs/storage limits.
            if tenant is not None:
                from api.quota import check_job_quota, check_storage_quota

                check_job_quota(tenant, session=session)
                check_storage_quota(tenant, source_size, session=session)

            # R4-RC-17/52: page quota at admit when estimate known
            if tenant is not None and estimated_pages:
                from api.quota import check_page_quota

                check_page_quota(tenant, estimated_pages, session=session)

            job = Job(
                job_id=job_id,
                status="submitted",
                priority=priority,
                source_file=filename,
                source_hash=source_hash,
                result_path=str(job_output),
                total_pages=estimated_pages,
                webhook_url=webhook_url,
                webhook_secret=encrypted_secret,
                tenant_id=tenant_id,
                batch_id=batch_id,
            )
            job.settings = settings
            session.add(job)
            session.flush()

            if tenant is not None:
                from api.usage import record_job_submitted, record_storage_used

                record_job_submitted(tenant_id, session=session)
                record_storage_used(tenant_id, source_size, session=session)

            if retry_of:
                # Lock the source row in the admission transaction. An
                # individual retry may have read an unclaimed row before a
                # BatchManager retry claimed it; the locked read makes the
                # claim check authoritative after concurrent updates commit.
                original = (
                    session.query(Job)
                    .filter(Job.job_id == retry_of)
                    .with_for_update()
                    .one_or_none()
                )
                if original is None:
                    raise ValueError("Retry source disappeared during admission")
                previous_batch = original.batch_id
                batch_retry_in_progress = bool(original.retry_claimed)
                if batch_retry_in_progress and batch_id != previous_batch:
                    raise ValueError(
                        "A batch retry is already in progress for this job"
                    )
                original.retried_to = job_id
                original.settings = {
                    **dict(original.settings or {}),
                    "retried_to": job_id,
                    "superseded": True,
                    "previous_batch_id": previous_batch,
                }
                # BatchManager holds this claim until the entire retry request
                # has admitted its replacements.  Releasing it here creates a
                # window where a fast first replacement can terminalize the
                # parent before later replacements have been admitted.
                if batch_retry_in_progress:
                    original.retry_claimed = 1
                else:
                    original.retry_claimed = 0
                if batch_id and not batch_retry_in_progress:
                    original.batch_id = None
            session.commit()
            job_persisted = True
            session.refresh(job)
            self._publish_job_event(job, "job.submitted")

            # Enqueue for priority-ordered background processing
            self._ensure_workers()
            self._priority_queue.put(job_id, priority=priority)

            self._update_dashboard_on_submit(active + 1)

            return job
        finally:
            session.close()
            if not job_persisted:
                _cleanup_staged_dirs()

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        job_id: str,
        source_dir: str,
        output_dir: str,
        settings: dict,
    ) -> None:
        """Run ocr_gpu_async.py as a subprocess for isolation."""
        with self._pipeline_condition:
            self._active_pipeline_processes[job_id] = None
            self._pipeline_condition.notify_all()
        session: Session = self._session_factory()
        try:
            job = session.get(Job, job_id)
            if not job:
                return
            started_at = datetime.now(timezone.utc).replace(tzinfo=None)
            transitioned = session.query(Job).filter(
                Job.job_id == job_id, Job.status == "submitted"
            ).update(
                {
                    "status": "processing",
                    "started_at": started_at,
                    "pid": _SPAWNING_PID,
                },
                synchronize_session=False,
            )
            session.commit()
            if not transitioned:
                logger.info("Pipeline skipped for %s: no longer submitted", job_id)
                return
            with self._pipeline_condition:
                session.expire_all()
                job = session.get(Job, job_id)
                if job is None or job.status != "processing":
                    if job is not None and job.status == "cancelled":
                        session.query(Job).filter(
                            Job.job_id == job_id,
                            Job.status == "cancelled",
                            Job.pid == _SPAWNING_PID,
                        ).update({"pid": None}, synchronize_session=False)
                        session.commit()
                    return
                self._publish_job_event(job, "job.processing")
                write_job_log(
                    job_id,
                    {
                        "level": "INFO",
                        "code": "JOB_STARTED",
                        "message": "Pipeline started",
                        "data": {
                            "source_dir": source_dir,
                            "output_dir": output_dir,
                            "enable_docintel": bool(settings.get("enable_docintel")),
                        },
                    },
                    base_dir=output_dir,
                )

            cmd = [
                # Use the interpreter that is running the API.  Production
                # images intentionally expose ``python3`` only; invoking a
                # bare ``python`` makes every accepted job fail before the
                # OCR pipeline starts.
                sys.executable,
                config.PIPELINE_SCRIPT,
                "--source",
                source_dir,
                "--output",
                output_dir,
            ]
            if settings.get("enable_docintel"):
                cmd.append("--enable-docintel")
                mode = settings.get("docintel_mode", "full")
                if mode != "full":
                    cmd.extend(["--docintel-mode", mode])

            logger.info("Starting pipeline for %s: %s", job_id, " ".join(cmd))

            # Inherit the API service's stdout/stderr so its runtime log
            # collector drains pipeline output. An unread PIPE can fill and
            # block an otherwise healthy OCR child while we wait for exit.
            popen_kwargs = {}
            if os.name != "nt":
                popen_kwargs["start_new_session"] = True
            pipeline_env = os.environ.copy()
            pipeline_env["OCR_JOB_ID"] = str(job_id)
            popen_kwargs["env"] = pipeline_env
            # Cancellation shares this lock from its terminal CAS through
            # child termination. It cannot miss a child between spawn and PID
            # persistence, and a cancelled row cannot launch a new child.
            with self._pipeline_condition:
                session.expire_all()
                current = session.get(Job, job_id)
                if current is None or current.status != "processing":
                    if current is not None and current.status == "cancelled":
                        session.query(Job).filter(
                            Job.job_id == job_id,
                            Job.status == "cancelled",
                            Job.pid == _SPAWNING_PID,
                        ).update({"pid": None}, synchronize_session=False)
                        session.commit()
                    return
                proc = subprocess.Popen(cmd, **popen_kwargs)
                self._active_pipeline_processes[job_id] = proc
                persisted = session.query(Job).filter(
                    Job.job_id == job_id, Job.status == "processing"
                ).update({"pid": proc.pid}, synchronize_session=False)
                session.commit()
                if not persisted:
                    _kill_pipeline_process(proc)
                    proc.wait(timeout=10)
                    session.query(Job).filter(
                        Job.job_id == job_id, Job.status == "cancelled"
                    ).update({"pid": None}, synchronize_session=False)
                    session.commit()
                    return
                self._pipeline_condition.notify_all()

            # Close the spawn-to-first-progress-poll readiness gap. This is
            # emitted only after Popen returns a measured live child.
            _write_pipeline_heartbeat()

            # Stream output to log, update progress via file system polling
            from api.background_work import start_writer

            monitor = start_writer(
                self._monitor_progress,
                args=(job_id, output_dir, proc),
                thread_factory=threading.Thread,
                name=f"monitor-{job_id}",
            )

            timeout_minutes = _resolve_processing_timeout_minutes(settings)
            timed_out = False
            try:
                proc.wait(timeout=timeout_minutes * 60)
            except subprocess.TimeoutExpired:
                timed_out = True
                logger.error(
                    "Pipeline for %s exceeded timeout of %s",
                    job_id,
                    _format_processing_timeout(timeout_minutes),
                )
                try:
                    _kill_pipeline_process(proc)
                except Exception:
                    logger.exception("Failed to kill timed-out pipeline for %s", job_id)
                try:
                    proc.wait(timeout=10)
                except Exception:
                    logger.exception(
                        "Timed-out pipeline for %s did not exit cleanly after kill",
                        job_id,
                    )
            monitor.join(timeout=10)

            # Claim finalization before quota, usage, events, or webhooks. A
            # cancellation committed in another session wins this CAS.
            session.expire_all()
            job = session.get(Job, job_id)
            if job is None or job.status != "processing":
                if job is not None and job.status == "cancelled":
                    session.query(Job).filter(
                        Job.job_id == job_id, Job.status == "cancelled"
                    ).update({"pid": None}, synchronize_session=False)
                    session.commit()
                return
            pages_completed = int(job.pages_completed or 0)
            completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
            updates = {"completed_at": completed_at, "pid": None}
            if timed_out:
                updates["status"] = "failed"
                updates["current_stage"] = "failed"
                updates["error_message"] = (
                    "Job exceeded processing timeout "
                    f"({_format_processing_timeout(timeout_minutes)})"
                )
            elif proc.returncode == 0 and not _has_completed_artifacts(
                Path(output_dir)
            ):
                updates["status"] = "failed"
                updates["current_stage"] = "failed"
                updates["error_message"] = (
                    "Pipeline exited successfully without complete PDF and text "
                    "artifacts"
                )
            elif proc.returncode == 0:
                # A fast pipeline may exit before the progress monitor's first
                # tick. Consolidated exports prove completion, not page count;
                # use the source count established at admission.
                if job.total_pages:
                    pages_completed = max(pages_completed, int(job.total_pages))
                updates.update(
                    status="completed",
                    current_stage="completed",
                    pages_completed=pages_completed,
                )
            else:
                updates.update(
                    status="failed",
                    current_stage="failed",
                    error_message=f"Pipeline exited with code {proc.returncode}",
                )

            if job.started_at:
                updates["processing_time"] = (
                    completed_at - job.started_at
                ).total_seconds()
            transitioned = session.query(Job).filter(
                Job.job_id == job_id, Job.status == "processing"
            ).update(updates, synchronize_session=False)
            if not transitioned:
                session.rollback()
                return
            session.refresh(job)
            if job.status == "completed":
                from api.quota import QuotaExceededError, consume_page_quota

                try:
                    if job.tenant_id and pages_completed > 0:
                        tenant = session.get(Tenant, job.tenant_id)
                        if tenant and tenant.status == "active":
                            consume_page_quota(tenant, pages_completed, session=session)
                except QuotaExceededError as exc:
                    job.status = "failed"
                    job.current_stage = "failed"
                    job.error_message = str(exc)
            if job.started_at:
                if job.tenant_id:
                    try:
                        from api.usage import record_processing_seconds

                        record_processing_seconds(
                            job.tenant_id,
                            job.processing_time,
                            period=job.started_at.strftime("%Y-%m"),
                            session=session,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to record tenant processing usage for %s",
                            job.tenant_id,
                        )
            session.commit()
            session.refresh(job)
            self._publish_job_event(job, f"job.{job.status}")

            if job.status == "completed":
                write_job_log(
                    job_id,
                    {
                        "level": "INFO",
                        "code": "JOB_COMPLETED",
                        "message": "Pipeline completed successfully",
                        "data": {
                            "pages_completed": job.pages_completed or 0,
                            "processing_time_seconds": job.processing_time,
                        },
                    },
                    base_dir=output_dir,
                )
            else:
                write_job_log(
                    job_id,
                    {
                        "level": "ERROR",
                        "code": "JOB_FAILED",
                        "message": job.error_message or "Pipeline failed",
                        "data": {"pages_completed": job.pages_completed or 0},
                    },
                    base_dir=output_dir,
                )

            self._update_dashboard_on_complete(job, session)

            self._fire_webhook(job)
            self._check_batch_completion(job)

        except Exception:
            logger.exception("Pipeline error for job %s", job_id)
            try:
                session.rollback()
                proc = self._active_pipeline_processes.get(job_id)
                if proc is not None and proc.poll() is None:
                    _kill_pipeline_process(proc)
                    proc.wait(timeout=10)
                session.query(Job).filter(
                    Job.job_id == job_id,
                    Job.status == "cancelled",
                    Job.pid == _SPAWNING_PID,
                ).update({"pid": None}, synchronize_session=False)
                session.commit()
                transitioned = session.query(Job).filter(
                    Job.job_id == job_id,
                    Job.status.in_(["submitted", "processing"]),
                ).update(
                    {
                        "status": "failed",
                        "current_stage": "failed",
                        "error_message": "Unexpected pipeline error",
                        "completed_at": datetime.now(timezone.utc).replace(tzinfo=None),
                        "pid": None,
                    },
                    synchronize_session=False,
                )
                session.commit()
                if transitioned:
                    job = session.get(Job, job_id)
                    session.refresh(job)
                    self._publish_job_event(job, "job.failed")
                    write_job_log(
                        job_id,
                        {
                            "level": "ERROR",
                            "code": "JOB_FAILED",
                            "message": "Unexpected pipeline error",
                        },
                        base_dir=output_dir,
                    )

                    self._update_dashboard_on_complete(job, session)

                    self._fire_webhook(job)
                    self._check_batch_completion(job)
            except Exception:
                logger.exception("Failed to update job status for %s", job_id)
        finally:
            session.close()
            release_job_log_state(job_id)
            with self._pipeline_condition:
                self._active_pipeline_processes.pop(job_id, None)
                self._pipeline_condition.notify_all()

    # ------------------------------------------------------------------
    # Progress monitoring
    # ------------------------------------------------------------------

    def _monitor_progress(
        self, job_id: str, output_dir: str, proc: subprocess.Popen
    ) -> None:
        """Poll output folder to estimate page progress."""
        output_path = Path(output_dir)
        session: Session = self._session_factory()
        try:
            while proc.poll() is None:
                _write_pipeline_heartbeat()
                time.sleep(config.PIPELINE_POLL_INTERVAL)
                if _job_output_root_is_symlink(output_path):
                    continue
                if not _has_completed_artifacts(output_path):
                    continue
                session.expire_all()
                job = session.get(Job, job_id)
                if not job or job.status != "processing":
                    break
                # ``EXPORT/TEXT`` is one consolidated text file for a whole
                # document. Its file count cannot be used as page progress.
                # The completed PDF/TEXT pair is only published after the
                # pipeline has accounted for the document, so use its admitted
                # page count when available.
                pages = int(job.total_pages or job.pages_completed or 0)
                if pages > int(job.pages_completed or 0):
                    from sqlalchemy import func

                    updated = session.query(Job).filter(
                        Job.job_id == job_id,
                        Job.status == "processing",
                        func.coalesce(Job.pages_completed, 0) < pages,
                    ).update(
                        {"pages_completed": pages, "current_stage": "processing"},
                        synchronize_session=False,
                    )
                    session.commit()
                    if not updated:
                        continue
                    session.refresh(job)
                    self._publish_job_event(job, "job.progress")
                    write_job_log(
                        job_id,
                        {
                            "level": "INFO",
                            "code": "JOB_PROGRESS_TICK",
                            "message": f"Completed {pages} page(s)",
                            "data": {"pages_completed": pages},
                        },
                        base_dir=output_dir,
                    )
        except Exception:
            logger.exception("Progress monitor error for %s", job_id)
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def check_queue_capacity(self) -> bool:
        """Return True if the job queue can accept new jobs."""
        session = self._session_factory()
        try:
            active = (
                session.query(Job)
                .filter(Job.status.in_(["submitted", "processing"]))
                .count()
            )
            return active < _max_outstanding_jobs()
        finally:
            session.close()

    def get_job(self, job_id: str, tenant_id: Optional[str] = None) -> Optional[Job]:
        session: Session = self._session_factory()
        try:
            query = session.query(Job).filter(Job.job_id == job_id)
            if tenant_id is not None:
                query = query.filter(Job.tenant_id == tenant_id)
            return query.first()
        finally:
            session.close()

    def list_jobs(
        self,
        status: Optional[str] = None,
        batch_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        # Backward-compatible aliases (deprecated)
        page: Optional[int] = None,
        per_page: Optional[int] = None,
        # Extended filters (D6)
        status_in: Optional[list[str]] = None,
        submitted_after: Optional[datetime] = None,
        submitted_before: Optional[datetime] = None,
        q: Optional[str] = None,
        sort: Optional[str] = None,
    ) -> tuple[list[Job], int]:
        from sqlalchemy import case, or_

        from api.deps import normalize_pagination, utc_naive_datetime

        # Keep internal callers under the same bounds as the HTTP dependency.
        limit, offset = normalize_pagination(limit, offset, page, per_page)
        submitted_after = utc_naive_datetime(submitted_after)
        submitted_before = utc_naive_datetime(submitted_before)

        session: Session = self._session_factory()
        try:
            query = session.query(Job)
            if status:
                query = query.filter(Job.status == status)
            if status_in:
                # Multi-status filter (D6).  Drop empty/invalid values.
                cleaned = list(
                    dict.fromkeys(s.strip() for s in status_in if s and s.strip())
                )
                if len(cleaned) > 16:
                    raise ValueError("at most 16 status filters are supported")
                if cleaned:
                    query = query.filter(Job.status.in_(cleaned))
            if batch_id:
                query = query.filter(Job.batch_id == batch_id)
            if tenant_id is not None:
                query = query.filter(Job.tenant_id == tenant_id)
            if submitted_after is not None:
                query = query.filter(Job.created_at >= submitted_after)
            if submitted_before is not None:
                query = query.filter(Job.created_at <= submitted_before)
            if q:
                # Sanitize for LIKE injection: escape % and _ then wrap with %.
                # Match against job_id and source_file (case-insensitive).
                escaped = (
                    q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                )
                pattern = f"%{escaped}%"
                query = query.filter(
                    or_(
                        Job.job_id.ilike(pattern, escape="\\"),
                        Job.source_file.ilike(pattern, escape="\\"),
                    )
                )
            total = query.count()

            # Sort selection
            if sort == "submitted_at_asc":
                ordering = (Job.created_at.asc(), Job.job_id.asc())
            elif sort == "duration_desc":
                ordering = (
                    Job.processing_time.desc().nullslast(),
                    Job.created_at.desc(),
                    Job.job_id.desc(),
                )
            elif sort == "status":
                ordering = (Job.status.asc(), Job.created_at.desc(), Job.job_id.desc())
            else:
                # Default: priority-then-time (urgent first, then newest first).
                # Also matches sort == "submitted_at_desc".
                priority_order = case(
                    (Job.priority == "urgent", 0),
                    (Job.priority == "normal", 1),
                    (Job.priority == "low", 2),
                    else_=1,
                )
                if sort == "submitted_at_desc":
                    ordering = (Job.created_at.desc(), Job.job_id.desc())
                else:
                    ordering = (
                        priority_order,
                        Job.created_at.desc(),
                        Job.job_id.desc(),
                    )

            jobs = query.order_by(*ordering).offset(offset).limit(limit).all()
            return jobs, total
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def _await_cancelled_writer(self, job_id: str, timeout: float = 10.0) -> None:
        """Wait for another runner to clear its persisted launch/PID fence."""
        deadline = time.monotonic() + timeout
        while True:
            session = self._session_factory()
            try:
                pid = session.query(Job.pid).filter(Job.job_id == job_id).scalar()
            finally:
                session.close()
            if pid is None:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Cancelled pipeline {job_id} has not stopped"
                )
            time.sleep(0.02)

    def cancel_job(self, job_id: str, tenant_id: Optional[str] = None) -> Optional[Job]:
        session: Session = self._session_factory()
        try:
            await_remote_writer = False
            with self._pipeline_condition:
                query = session.query(Job).filter(Job.job_id == job_id)
                if tenant_id is not None:
                    query = query.filter(Job.tenant_id == tenant_id)
                job = query.first()
                if not job:
                    return None
                if job.status in {"completed", "failed"}:
                    return job

                pid = job.pid
                already_cancelled = job.status == "cancelled"
                proc = self._active_pipeline_processes.get(job_id)
                if already_cancelled and pid is None and (
                    proc is None or proc.poll() is not None
                ):
                    return job
                if not already_cancelled:
                    transitioned = query.filter(
                        Job.status.in_(["submitted", "processing"])
                    ).update(
                        {
                            "status": "cancelled",
                            "completed_at": datetime.now(timezone.utc).replace(
                                tzinfo=None
                            ),
                        },
                        synchronize_session=False,
                    )
                    session.commit()
                    if not transitioned:
                        session.expire_all()
                        return query.first()

                if proc is not None:
                    if proc.poll() is None:
                        try:
                            _kill_pipeline_process(proc)
                            proc.wait(timeout=10)
                        except Exception:
                            logger.exception(
                                "Failed to stop cancelled pipeline %s", job_id
                            )
                            raise RuntimeError(
                                f"Cancelled pipeline {job_id} has not stopped"
                            ) from None
                    query.filter(Job.status == "cancelled").update(
                        {"pid": None}, synchronize_session=False
                    )
                    session.commit()
                elif pid == _SPAWNING_PID:
                    await_remote_writer = True
                elif pid and _pid_belongs_to_job(pid, job_id):
                    await_remote_writer = True
                    try:
                        _kill_pipeline_pid(pid)
                    except ProcessLookupError:
                        pass
                    except OSError:
                        logger.exception(
                            "Failed to signal remote pipeline %s at PID %s", job_id, pid
                        )
                elif pid and pid > 0:
                    # A reused or exited PID has no remote writer to clear
                    # the fence. Verify the tagged orphan set before releasing
                    # it, including a child spawned before PID publication.
                    if not _stop_orphaned_pipeline(job_id, pid):
                        raise RuntimeError(
                            f"Cancelled pipeline {job_id} writer state is unknown"
                        )
                    query.filter(Job.status == "cancelled", Job.pid == pid).update(
                        {"pid": None}, synchronize_session=False
                    )
                    session.commit()
                elif not pid:
                    # The runner has not spawned; the shared condition prevents
                    # it from doing so after this terminal transition.
                    pass

            if await_remote_writer:
                self._await_cancelled_writer(job_id)
            session.expire_all()
            job = session.get(Job, job_id)
            if job is None:
                return None
            if already_cancelled:
                return job
            self._publish_job_event(job, "job.cancelled")

            self._fire_webhook(job)

            return job
        finally:
            session.close()

    def quiesce_jobs_for_tenant(
        self,
        tenant_id: str,
        job_ids,
        *,
        timeout: float = 30.0,
    ) -> None:
        """Cancel and join all local pipeline writers before tenant erasure."""
        bounded_ids = {str(job_id) for job_id in job_ids}
        for job_id in bounded_ids:
            self.cancel_job(job_id, tenant_id=tenant_id)

        deadline = time.monotonic() + max(0.0, float(timeout))
        killed = set()
        with self._pipeline_condition:
            while True:
                active = {
                    job_id: proc
                    for job_id, proc in self._active_pipeline_processes.items()
                    if job_id in bounded_ids
                }
                if not active:
                    return
                for job_id, proc in active.items():
                    if proc is not None and job_id not in killed:
                        try:
                            _kill_pipeline_process(proc)
                        except Exception:
                            logger.exception(
                                "Failed to kill tenant-purge pipeline %s", job_id
                            )
                        killed.add(job_id)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "Tenant purge timed out waiting for pipeline writers: "
                        + ", ".join(sorted(active)[:5])
                    )
                self._pipeline_condition.wait(timeout=min(0.1, remaining))

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    def retry_job(
        self,
        job_id: str,
        tenant_id: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> Optional[Job]:
        """Retry a failed or cancelled job by re-submitting with the same source."""
        if batch_id is not None:
            # BatchManager already holds this fence for its whole request.
            return self._retry_job_unlocked(job_id, tenant_id, batch_id)
        # Hold the same fence from before the first row read through submit.
        # Otherwise recovery could clear a batch claim and finalize its parent
        # between this method's initial read and replacement admission.
        with _batch_retry_serialization_lock():
            return self._retry_job_unlocked(job_id, tenant_id, batch_id)

    def _retry_job_unlocked(
        self,
        job_id: str,
        tenant_id: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> Optional[Job]:
        """Perform retry admission while the caller holds any needed fence."""
        session: Session = self._session_factory()
        try:
            query = session.query(Job).filter(Job.job_id == job_id)
            if tenant_id is not None:
                query = query.filter(Job.tenant_id == tenant_id)
            original = query.first()
            if not original:
                return None

            if original.status not in ("failed", "cancelled"):
                raise ValueError(
                    f"Only failed or cancelled jobs can be retried (current: {original.status})"
                )
            if original.retry_claimed and batch_id != original.batch_id:
                raise ValueError("A batch retry is already in progress for this job")
            with self._pipeline_condition:
                active_proc = self._active_pipeline_processes.get(job_id)
                if original.pid is not None or (
                    active_proc is not None and active_proc.poll() is None
                ):
                    raise ValueError("Previous pipeline writer has not stopped")

            # Locate original source file
            original_source_dir = Path(config.SOURCE_FOLDER) / job_id
            if not original_source_dir.exists():
                raise FileNotFoundError(
                    f"Original source directory not found: {original_source_dir}"
                )

            source_files = list(original_source_dir.iterdir())
            if not source_files:
                raise FileNotFoundError("Original source file no longer available.")

            source_file = source_files[0]

            # Decrypt stored webhook secret before passing to submit()
            # (submit will re-encrypt for the new job record)
            retry_webhook_secret = None
            if original.webhook_secret:
                from api.config import decrypt_webhook_secret

                retry_webhook_secret = decrypt_webhook_secret(original.webhook_secret)

            # Submit as a new job, copying the source file and preserving webhook
            return self.submit(
                source_path=str(source_file),
                tenant_id=original.tenant_id,
                priority=original.priority,
                enable_docintel=original.settings.get("enable_docintel", False),
                docintel_mode=original.settings.get("docintel_mode", "full"),
                processing_timeout_minutes=original.settings.get(
                    "processing_timeout_minutes"
                ),
                webhook_url=original.webhook_url,
                webhook_secret=retry_webhook_secret,
                batch_id=batch_id,
                retry_of=job_id,
            )
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_result_artifacts(self, job_id: str) -> dict[str, str]:
        """Return dict of artifact_type -> file_path for completed job.

        Each job processes a single input file, so each output subdirectory
        will contain at most one artifact of each type.
        """
        job = self.get_job(job_id)
        if not job or not job.result_path:
            return {}

        try:
            result_dir = resolve_job_output_path(
                job.job_id,
                job.result_path,
                config.OUTPUT_FOLDER,
            )
        except ValueError:
            logger.error("Refusing non-canonical output path for job %s", job_id)
            return {}
        artifacts = {}
        pdf = _first_regular_file_under_root(result_dir, ("EXPORT", "PDF"), ".pdf")
        if pdf:
            artifacts["pdf"] = pdf
        text = _first_regular_file_under_root(result_dir, ("EXPORT", "TEXT"), ".txt")
        if text:
            artifacts["text"] = text
        structure = _first_regular_file_under_root(
            result_dir, ("EXPORT", "STRUCTURE"), ".json"
        )
        if structure:
            artifacts["structure"] = structure
        return artifacts
