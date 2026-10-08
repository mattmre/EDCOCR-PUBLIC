"""Durable Bates sequence allocator (R3-RC-01).

Process-local counters cannot prevent cross-job / cross-tenant collisions.
This module reserves exclusive numeric ranges keyed by
``(tenant_id or \"_global\", prefix)``.

Backends
--------
``file`` (default)
    Atomic JSON ledger on disk (``BATES_LEDGER_PATH``). Safe on a single
    shared filesystem; not multi-host without NFS/shared volume.

``redis``
    Atomic Redis counter via Lua (``BATES_LEDGER_REDIS_URL`` or
    ``REDIS_URL``). Correct for multi-host / multi-replica stamp APIs.

``auto``
    Prefer Redis when a Redis URL is available; fall back to file.

Environment
-----------
``BATES_LEDGER_BACKEND``
    ``auto`` | ``file`` | ``redis`` (default: ``auto``).
``BATES_LEDGER_PATH``
    JSON ledger path (default: ``{OCR_OUTPUT_DIR or /tmp}/bates_sequences.json``).
``BATES_LEDGER_REDIS_URL`` / ``REDIS_URL``
    Redis URL for multi-host allocator.
``BATES_LEDGER_REDIS_KEY_PREFIX``
    Redis key prefix (default: ``ocr:bates:seq:``).
"""

from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_MAX_COMPONENT_LENGTH = 256
_MAX_ALLOCATION_COUNT = 1_000_000
_MAX_LEDGER_BYTES = 1 * 1024 * 1024

# Lua: atomically reserve [start, start+count) under preferred floor.
_REDIS_ALLOCATE_LUA = """
local key = KEYS[1]
local count = tonumber(ARGV[1])
local preferred = tonumber(ARGV[2])
if count < 1 then count = 1 end
local cur = tonumber(redis.call('GET', key) or '0')
local next_val = cur + 1
if preferred > 0 and preferred > next_val then
  next_val = preferred
end
if next_val < 1 then next_val = 1 end
redis.call('SET', key, next_val + count - 1)
return next_val
"""


def _default_ledger_path() -> Path:
    env = os.environ.get("BATES_LEDGER_PATH", "").strip()
    if env:
        return Path(env)
    base = os.environ.get("OCR_OUTPUT_DIR") or os.environ.get("OUTPUT_FOLDER") or "/tmp"
    return Path(base) / "bates_sequences.json"


def _backend_name() -> str:
    raw = os.environ.get("BATES_LEDGER_BACKEND", "auto").strip().lower()
    if raw in ("file", "redis", "auto"):
        return raw
    return "auto"


def _redis_url() -> str:
    return (
        os.environ.get("BATES_LEDGER_REDIS_URL", "").strip()
        or os.environ.get("REDIS_URL", "").strip()
    )


def _redis_key(tenant_id: Optional[str], prefix: str) -> str:
    pfx = os.environ.get("BATES_LEDGER_REDIS_KEY_PREFIX", "ocr:bates:seq:").strip()
    if not pfx:
        pfx = "ocr:bates:seq:"
    t, p = _validate_components(tenant_id, prefix)
    # Length-prefix both components so delimiters cannot create a key
    # collision between different tenant/prefix tuples.
    return f"{pfx}{len(t)}:{t}|{len(p)}:{p}"


def _load(path: Path) -> dict:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return {}
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise RuntimeError(f"Bates ledger must be a regular non-symlink file: {path}")
    if path_stat.st_size > _MAX_LEDGER_BYTES:
        raise RuntimeError(
            f"Bates ledger exceeds the {_MAX_LEDGER_BYTES}-byte safety limit"
        )
    try:
        with path.open("rb") as handle:
            raw_bytes = handle.read(_MAX_LEDGER_BYTES + 1)
        if len(raw_bytes) > _MAX_LEDGER_BYTES:
            raise RuntimeError(
                f"Bates ledger exceeds the {_MAX_LEDGER_BYTES}-byte safety limit"
            )
        raw = raw_bytes.decode("utf-8")
        data = json.loads(raw) if raw.strip() else {}
        if not isinstance(data, dict):
            raise RuntimeError("Bates ledger must contain a JSON object")
        return data
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read Bates ledger {path}") from exc


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            logger.debug("Bates ledger directory fsync unavailable: %s", path.parent)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


@contextmanager
def _ledger_file_lock(path: Path):
    """Serialize file-ledger reads/writes across local processes."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(fd, "a+") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        try:
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:
                pass


def _validate_components(
    tenant_id: Optional[str],
    prefix: str,
) -> tuple[str, str]:
    if tenant_id is not None and not isinstance(tenant_id, str):
        raise ValueError("tenant_id and prefix must be strings")
    if not isinstance(prefix, str):
        raise ValueError("tenant_id and prefix must be strings")
    tenant = (tenant_id or "_global").strip() or "_global"
    pfx = prefix or ""
    if any(
        not value or len(value) > _MAX_COMPONENT_LENGTH or "\x00" in value
        for value in (tenant, pfx)
    ):
        raise ValueError("tenant_id and prefix must be bounded and NUL-free")
    return tenant, pfx


def _key(tenant_id: Optional[str], prefix: str) -> str:
    t, p = _validate_components(tenant_id, prefix)
    return f"{len(t)}:{t}\0{len(p)}:{p}"


def _allocate_file(
    *,
    prefix: str,
    tenant_id: Optional[str],
    count: int,
    preferred_start: Optional[int],
) -> int:
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("count must be an integer")
    if not 1 <= count <= _MAX_ALLOCATION_COUNT:
        raise ValueError("count is outside the supported range")
    _validate_components(tenant_id, prefix)
    path = _default_ledger_path()
    with _lock:
        with _ledger_file_lock(path):
            data = _load(path)
            k = _key(tenant_id, prefix)
            try:
                next_val = int(data.get(k, 1))
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("Bates ledger contains an invalid sequence") from exc
            if preferred_start is not None and preferred_start > next_val:
                next_val = int(preferred_start)
            if next_val < 1:
                next_val = 1
            start = next_val
            data[k] = start + count
            _save(path, data)
            logger.info(
                "Bates[file] allocated tenant=%s prefix=%r start=%s count=%s next=%s",
                tenant_id or "_global",
                prefix,
                start,
                count,
                data[k],
            )
            return start


def _allocate_redis(
    *,
    prefix: str,
    tenant_id: Optional[str],
    count: int,
    preferred_start: Optional[int],
    url: str,
) -> int:
    import redis  # type: ignore[import-untyped]

    client = redis.from_url(url, socket_connect_timeout=2, socket_timeout=5)
    try:
        preferred = int(preferred_start) if preferred_start is not None else 0
        if preferred < 0:
            preferred = 0
        start = int(
            client.eval(
                _REDIS_ALLOCATE_LUA,
                1,
                _redis_key(tenant_id, prefix),
                int(count),
                preferred,
            )
        )
        logger.info(
            "Bates[redis] allocated tenant=%s prefix=%r start=%s count=%s",
            tenant_id or "_global",
            prefix,
            start,
            count,
        )
        return start
    finally:
        try:
            client.close()
        except Exception:
            pass


def allocate_range(
    *,
    prefix: str,
    tenant_id: Optional[str],
    count: int,
    preferred_start: Optional[int] = None,
) -> int:
    """Reserve ``count`` consecutive numbers; return the start of the range.

    If ``preferred_start`` is provided and the ledger next value is still
    below it, the allocator advances to ``preferred_start`` (operator-chosen
    floor). Concurrent callers for the same key never receive overlapping
    ranges (file lock within process + fsync replace; Redis Lua multi-host).
    """
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("count must be an integer")
    if count < 1:
        count = 1
    if count > _MAX_ALLOCATION_COUNT:
        raise ValueError("count is outside the supported range")
    _validate_components(tenant_id, prefix)
    if preferred_start is not None and (
        not isinstance(preferred_start, int)
        or isinstance(preferred_start, bool)
        or preferred_start < 0
    ):
        raise ValueError("preferred_start must be a non-negative integer")

    backend = _backend_name()
    url = _redis_url()

    use_redis = backend == "redis" or (backend == "auto" and bool(url))
    if use_redis:
        if not url:
            raise RuntimeError(
                "BATES_LEDGER_BACKEND=redis requires BATES_LEDGER_REDIS_URL or REDIS_URL"
            )
        try:
            return _allocate_redis(
                prefix=prefix,
                tenant_id=tenant_id,
                count=count,
                preferred_start=preferred_start,
                url=url,
            )
        except Exception as exc:
            # A configured Redis allocator is the multi-host safety boundary;
            # falling back to a local file after an outage can overlap ranges
            # allocated by another host that still uses Redis.
            raise RuntimeError(
                "Bates Redis allocation failed; refusing unsafe file fallback"
            ) from exc

    return _allocate_file(
        prefix=prefix,
        tenant_id=tenant_id,
        count=count,
        preferred_start=preferred_start,
    )


def peek_next(*, prefix: str, tenant_id: Optional[str] = None) -> int:
    """Return the next number that would be allocated (no mutation)."""
    backend = _backend_name()
    url = _redis_url()
    use_redis = backend == "redis" or (backend == "auto" and bool(url))
    if use_redis and url:
        try:
            import redis  # type: ignore[import-untyped]

            client = redis.from_url(url, socket_connect_timeout=2, socket_timeout=5)
            try:
                cur = client.get(_redis_key(tenant_id, prefix))
                if cur is None:
                    return 1
                return int(cur) + 1
            finally:
                try:
                    client.close()
                except Exception:
                    pass
        except Exception:
            if backend == "redis":
                raise
            logger.exception("Bates Redis peek failed; falling back to file")

    _validate_components(tenant_id, prefix)
    path = _default_ledger_path()
    with _lock:
        with _ledger_file_lock(path):
            data = _load(path)
            try:
                return int(data.get(_key(tenant_id, prefix), 1))
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("Bates ledger contains an invalid sequence") from exc
