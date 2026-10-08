"""Translation model cache management -- Plan B Wave M2.

Provides a single source of truth for resolving on-disk paths to
CTranslate2 model weights used by the local translation engines
(OPUS-MT, NLLB-200, MADLAD-400 et al).  The cache supports:

* Atomic download + integrity verification via per-model
  ``manifest.json`` files containing SHA-256 hashes.
* LRU eviction with a configurable disk-pressure ceiling, with hot
  models exempted via :func:`pin_model`.
* Air-gapped mode -- when the pipeline config disables downloads the
  resolver refuses to fetch missing weights and raises
  :class:`ModelNotCachedError` instead.
* Forensic custody hooks via the translation custody adapter so
  download / integrity / eviction / NC-license-block events show up in
  the audit chain.
* Concurrency safety -- cache_state.json reads and writes are guarded
  by an OS-level file lock (``fcntl.flock`` on POSIX,
  ``msvcrt.locking`` on Windows) so multiple worker threads or
  processes can call :func:`get_translation_model_path` without
  corrupting the LRU bookkeeping.

This module is intentionally side-effect-free at import time.  It does
NOT register engines or touch the filesystem until a caller invokes one
of the public helpers.  This matches the gating in docs/release-candidate.md gotcha
``#86``: the cache code can be imported unconditionally; the engines
only call into it when ``ENABLE_TRANSLATION`` is on.

The actual download mechanism is intentionally pluggable.  Tests
inject a fake download fn via :func:`set_download_fn`; production
deployments will register a real registry connector in a follow-up
PR.  The default implementation raises ``NotImplementedError`` so that
calls falling through to a real download are loud rather than silent.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Optional

from ocr_local.translation.custody_adapter import ReasonCode
from ocr_local.translation.policy import PolicyDenied

if TYPE_CHECKING:
    from ocr_local.features.custody import CustodyChain
    from ocr_local.translation.policy import TenantPolicy


__all__ = [
    "CachedModelInfo",
    "ModelIntegrityError",
    "ModelNotCachedError",
    "get_translation_model_path",
    "verify_model_integrity",
    "evict_lru",
    "pin_model",
    "unpin_model",
    "list_cached_models",
    "register_model",
    "set_download_fn",
    "set_custody_chain",
    "DEFAULT_CACHE_DIR",
]


DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "ocr_local" / "translation"
_STATE_DIR_NAME = "_state"
_STATE_FILE_NAME = "cache_state.json"
_MANIFEST_FILE_NAME = "manifest.json"
_MAX_CACHE_COMPONENT_LENGTH = 256
_MAX_CACHE_JSON_BYTES = 1 * 1024 * 1024
_MAX_STATE_MODELS = 10_000
_MAX_MANIFEST_FILES = 100_000
_MAX_MANIFEST_FILENAME_LENGTH = 512


# ---------------------------------------------------------------------------
# Exceptions and dataclasses
# ---------------------------------------------------------------------------


class ModelNotCachedError(RuntimeError):
    """Raised when a model is not present and downloads are disallowed."""

    def __init__(self, engine: str, model_id: str, reason: str = "") -> None:
        self.engine = engine
        self.model_id = model_id
        self.reason = reason
        msg = f"Model not cached: engine={engine!r} model_id={model_id!r}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class ModelIntegrityError(RuntimeError):
    """Raised when a cached model fails SHA-256 manifest verification."""

    def __init__(
        self,
        engine: str,
        model_id: str,
        bad_files: Iterable[str] = (),
    ) -> None:
        self.engine = engine
        self.model_id = model_id
        self.bad_files = list(bad_files)
        super().__init__(
            f"Model integrity failure: engine={engine!r} "
            f"model_id={model_id!r} bad_files={self.bad_files!r}"
        )


@dataclasses.dataclass(frozen=True)
class CachedModelInfo:
    """Snapshot of one cached model entry."""

    engine: str
    model_id: str
    path: Path
    size_bytes: int
    license: str
    nc_licensed: bool
    pinned: bool
    access_time: float


# ---------------------------------------------------------------------------
# Module-level state -- guarded helpers below
# ---------------------------------------------------------------------------


_DownloadFn = Callable[[str, str, Path], dict]
"""Signature: ``(engine, model_id, dest_dir) -> manifest_dict``.

The download function is responsible for writing the model files into
``dest_dir`` and returning a manifest dict with at least the keys
``sha256`` (mapping filename -> hex digest), ``license`` (SPDX string)
and ``nc_licensed`` (bool).  The cache module writes the manifest to
disk after the download fn returns.
"""


def _default_download_fn(engine: str, model_id: str, dest_dir: Path) -> dict:
    raise NotImplementedError(
        "No download function registered. Call "
        "ocr_local.translation.cache.set_download_fn(fn) before requesting "
        f"missing model weights (engine={engine!r}, model_id={model_id!r})."
    )


_DOWNLOAD_FN: _DownloadFn = _default_download_fn

_CUSTODY_CHAIN: "Optional[CustodyChain]" = None

_THREAD_LOCK = threading.RLock()
"""Process-local lock guarding cache_state.json updates.

This is a reentrant lock so :func:`evict_lru` can call into helpers
that themselves take the lock without deadlocking.  Cross-process
serialization is handled separately by the OS-level file lock used
inside :func:`_locked_state`.
"""


def set_download_fn(fn: _DownloadFn) -> None:
    """Register the function used to fetch missing model weights.

    Pass ``None`` to reset back to the default (which raises
    ``NotImplementedError``).
    """
    global _DOWNLOAD_FN
    _DOWNLOAD_FN = fn if fn is not None else _default_download_fn


def set_custody_chain(chain: "Optional[CustodyChain]") -> None:
    """Register the custody chain used for cache events.

    When ``None`` (the default), cache events are silently dropped.
    Production callers should pass the same ``CustodyChain`` instance
    that the assembler uses so cache events land in the same audit
    log.  Tests typically pass ``MagicMock()`` and assert call counts.
    """
    global _CUSTODY_CHAIN
    _CUSTODY_CHAIN = chain


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _resolve_cache_root(cache_dir: Optional[str | Path] = None) -> Path:
    """Resolve the cache root, preferring the explicit arg over config.

    Lookup order: explicit arg -> ``PipelineConfig.translation_cache_dir``
    -> :data:`DEFAULT_CACHE_DIR`.  We import ``PipelineConfig`` lazily so
    importing this module does not pin the singleton early.
    """
    if cache_dir is not None:
        return Path(cache_dir)
    try:
        from pipeline_config import get_config  # type: ignore[import-not-found]

        cfg = get_config()
        if cfg is not None:
            override = getattr(cfg, "translation_cache_dir", None)
            if override:
                return Path(override)
    except (ImportError, AttributeError, RuntimeError):
        # ``get_config`` may not exist yet (PipelineConfig is still
        # being wired in maintenance review Ph3 follow-ups), or may raise when no
        # config has been set.  Fall back to the default cache dir.
        pass
    env_override = os.environ.get("TRANSLATION_MODEL_CACHE_DIR")
    if env_override:
        return Path(env_override)
    return DEFAULT_CACHE_DIR


def _model_dir(root: Path, engine: str, model_id: str) -> Path:
    return (
        root
        / _validate_cache_component(engine, "engine")
        / _validate_cache_component(model_id, "model_id")
    )


def _validate_cache_component(value: str, field_name: str) -> str:
    """Validate a cache path component before it reaches a filesystem join."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    value = value.strip()
    if len(value) > _MAX_CACHE_COMPONENT_LENGTH:
        raise ValueError(f"{field_name} exceeds the cache component limit")
    if value in {".", ".."} or "\x00" in value:
        raise ValueError(f"{field_name} is not a safe cache component")
    if "/" in value or "\\" in value:
        raise ValueError(f"{field_name} must not contain path separators")
    return value


def _state_path(root: Path) -> Path:
    return root / _STATE_DIR_NAME / _STATE_FILE_NAME


def _manifest_path(model_dir: Path) -> Path:
    return model_dir / _MANIFEST_FILE_NAME


# ---------------------------------------------------------------------------
# Atomic file IO + cross-platform file lock
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via temp-file + rename.

    The temp file is created in the same directory so the rename is a
    same-filesystem operation (atomic on POSIX, atomic-on-success on
    Windows).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise


def _lock_file(fh) -> None:
    """Acquire an exclusive lock on ``fh``.

    ``fcntl.flock`` is preferred on POSIX; ``msvcrt.locking`` is used on
    Windows.  Both fall back to a no-op if neither module is available
    (e.g. pyodide) -- in that case process-local ``_THREAD_LOCK`` is the
    only synchronization, which is still correct for single-process
    unit tests.
    """
    try:
        import fcntl  # type: ignore[import-not-found]

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        return
    except ImportError:
        pass
    if sys.platform.startswith("win"):
        try:
            import msvcrt  # type: ignore[import-not-found]

            # Lock first byte; sufficient because all writers serialize
            # on the same single-byte region.
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        except (ImportError, OSError):
            pass


def _unlock_file(fh) -> None:
    try:
        import fcntl  # type: ignore[import-not-found]

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return
    except ImportError:
        pass
    if sys.platform.startswith("win"):
        try:
            import msvcrt  # type: ignore[import-not-found]

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except (ImportError, OSError):
            pass


@contextlib.contextmanager
def _locked_state(root: Path):
    """Context manager yielding the cache_state dict under an OS-level lock.

    The yielded dict is the parsed contents of ``cache_state.json``.
    If the caller mutates it, they must call ``_save_state(root, state)``
    before exiting the context.  Read-only callers can ignore the
    return value.
    """
    state_dir = root / _STATE_DIR_NAME
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "cache_state.lock"
    # Open the lock file in read+write so we can acquire an exclusive
    # OS-level lock without truncating it.
    with _THREAD_LOCK:
        with open(lock_path, "a+", encoding="utf-8") as lock_fh:
            _lock_file(lock_fh)
            try:
                state = _read_state(root)
                yield state
            finally:
                _unlock_file(lock_fh)


def _read_state(root: Path) -> dict:
    path = _state_path(root)
    if not path.exists():
        return {"models": {}}
    try:
        if path.is_symlink() or path.stat().st_size > _MAX_CACHE_JSON_BYTES:
            return {"models": {}}
        with path.open("rb") as fh:
            raw = fh.read(_MAX_CACHE_JSON_BYTES + 1)
        if len(raw) > _MAX_CACHE_JSON_BYTES:
            return {"models": {}}
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError, ValueError):
        return {"models": {}}
    if not isinstance(data, dict) or "models" not in data:
        return {"models": {}}
    models = data.get("models")
    if not isinstance(models, dict):
        return {"models": {}}
    safe_models: dict[str, dict] = {}
    for key, entry in list(models.items())[:_MAX_STATE_MODELS]:
        if (
            not isinstance(key, str)
            or key.count("/") != 1
            or not isinstance(entry, dict)
        ):
            continue
        engine, model_id = key.split("/", 1)
        try:
            safe_key = _state_key(engine, model_id)
        except ValueError:
            continue
        safe_models[safe_key] = entry
    return {"models": safe_models}


def _save_state(root: Path, state: dict) -> None:
    _atomic_write_json(_state_path(root), state)


def _state_key(engine: str, model_id: str) -> str:
    return f"{_validate_cache_component(engine, 'engine')}/{_validate_cache_component(model_id, 'model_id')}"


# ---------------------------------------------------------------------------
# Hashing + manifest verification
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(model_dir: Path) -> Optional[dict]:
    path = _manifest_path(model_dir)
    if not path.exists():
        return None
    try:
        if (
            model_dir.is_symlink()
            or path.is_symlink()
            or path.stat().st_size > _MAX_CACHE_JSON_BYTES
        ):
            return None
        with path.open("rb") as fh:
            raw = fh.read(_MAX_CACHE_JSON_BYTES + 1)
        if len(raw) > _MAX_CACHE_JSON_BYTES:
            return None
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _safe_manifest_target(model_dir: Path, filename: object) -> Path | None:
    """Return a regular manifest target confined to ``model_dir``."""
    if (
        not isinstance(filename, str)
        or not filename
        or len(filename) > _MAX_MANIFEST_FILENAME_LENGTH
    ):
        return None
    if "\x00" in filename or "\\" in filename:
        return None
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        return None
    target = model_dir / relative
    try:
        target.resolve(strict=False).relative_to(model_dir.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return None
    cursor = target
    while cursor != model_dir:
        try:
            item = cursor.lstat()
        except OSError:
            return None
        if stat.S_ISLNK(item.st_mode):
            return None
        cursor = cursor.parent
    try:
        item = target.lstat()
    except OSError:
        return target
    if not stat.S_ISREG(item.st_mode) or getattr(item, "st_nlink", 1) > 1:
        return None
    return target


def _verify_model_files(model_dir: Path, manifest: dict) -> tuple[bool, list[str]]:
    """Return (ok, bad_files).

    ``bad_files`` includes every file in ``manifest["sha256"]`` whose
    contents do not match the recorded digest, or whose file is missing
    from the model directory.
    """
    try:
        if model_dir.is_symlink() or not model_dir.is_dir():
            return False, ["<unsafe model directory>"]
    except OSError:
        return False, ["<unsafe model directory>"]
    sha_map = manifest.get("sha256")
    if (
        not isinstance(sha_map, dict)
        or not sha_map
        or len(sha_map) > _MAX_MANIFEST_FILES
    ):
        return False, ["<missing sha256 map>"]
    bad: list[str] = []
    for fname, expected in sha_map.items():
        target = _safe_manifest_target(model_dir, fname)
        if target is None:
            bad.append(str(fname)[:_MAX_MANIFEST_FILENAME_LENGTH])
            continue
        if not isinstance(expected, str) or len(expected) != 64:
            bad.append(str(fname)[:_MAX_MANIFEST_FILENAME_LENGTH])
            continue
        try:
            int(expected, 16)
        except ValueError:
            bad.append(str(fname)[:_MAX_MANIFEST_FILENAME_LENGTH])
            continue
        if not target.exists():
            bad.append(fname)
            continue
        try:
            actual = _sha256_file(target)
        except OSError:
            bad.append(fname)
            continue
        if actual != expected:
            bad.append(fname)
    return (not bad), bad


# ---------------------------------------------------------------------------
# Custody event helpers
# ---------------------------------------------------------------------------


def _emit_custody(reason_code: ReasonCode, **payload: object) -> None:
    chain = _CUSTODY_CHAIN
    if chain is None:
        return
    try:
        chain.log_event(
            "TRANSLATION_MODEL_CACHE",
            {"reason_code": str(reason_code), **payload},
        )
    except Exception:
        # Custody emission must never crash the cache path.
        pass


# ---------------------------------------------------------------------------
# License helpers
# ---------------------------------------------------------------------------


def _is_nc_licensed(license_str: str) -> bool:
    """Detect non-commercial license strings.

    Matches anything containing ``nc`` (case-insensitive), with the
    underscore/hyphen separators that the SPDX strings used by Plan B
    use.  ``CC-BY-NC-4.0``, ``CC-BY-NC-SA-4.0``, etc. all match.
    """
    if not license_str:
        return False
    s = license_str.lower()
    return "-nc-" in s or s.endswith("-nc") or s.startswith("nc-")


def _tenant_allows_nc(policy: "Optional[TenantPolicy]") -> bool:
    """Return True when ``policy`` permits NC-licensed models.

    The cache module accepts either field name -- the actual codebase
    uses ``allow_nllb_commercial`` while docs/release-candidate.md names the
    field ``allow_nc_licensed``.  We honour either if present and
    default to True (permissive) when neither is defined, leaving
    license-policy enforcement to the caller in that case.
    """
    if policy is None:
        return True
    # New-style attribute (gotcha #90).
    if hasattr(policy, "allow_nc_licensed"):
        return bool(getattr(policy, "allow_nc_licensed"))
    # Existing-codebase attribute.
    if hasattr(policy, "allow_nllb_commercial"):
        return bool(getattr(policy, "allow_nllb_commercial"))
    return True


# ---------------------------------------------------------------------------
# Air-gapped helper
# ---------------------------------------------------------------------------


def _airgapped() -> bool:
    """Return True when the active PipelineConfig disables downloads."""
    try:
        from pipeline_config import get_config  # type: ignore[import-not-found]

        cfg = get_config()
        if cfg is not None:
            return bool(getattr(cfg, "translation_airgapped", False))
    except (ImportError, AttributeError, RuntimeError):
        pass
    raw = os.environ.get("TRANSLATION_AIRGAPPED", "")
    return raw.strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_translation_model_path(
    engine: str,
    model_id: str,
    *,
    allow_download: bool = False,
    tenant_policy: "Optional[TenantPolicy]" = None,
    cache_dir: Optional[str | Path] = None,
) -> Path:
    """Resolve the on-disk path to a translation model.

    See module docstring for the full state machine.  Briefly:

    * Hit + integrity-OK   -> update access_time, return path.
    * Hit + integrity-FAIL -> emit ``MODEL_INTEGRITY_FAILED``,
      raise :class:`ModelIntegrityError`.
    * Miss + airgapped     -> raise :class:`ModelNotCachedError`.
    * Miss + ``allow_download=False`` -> raise.
    * Miss + ``allow_download=True``  -> download via
      :data:`_DOWNLOAD_FN`, write manifest, emit ``MODEL_DOWNLOADED``,
      return path.

    The NC-licensed filter is enforced *both* on cached entries (so we
    don't hand out CC-BY-NC weights to a commercial tenant just because
    they happen to be downloaded) and after a fresh download.
    """
    root = _resolve_cache_root(cache_dir)
    model_dir = _model_dir(root, engine, model_id)
    key = _state_key(engine, model_id)

    # Air-gapped mode forces ``allow_download=False`` regardless of the
    # caller's request.  We never reach out to the network in this mode.
    if _airgapped():
        allow_download = False

    manifest = _read_manifest(model_dir) if model_dir.exists() else None

    # ----- Cache hit -----------------------------------------------------
    if manifest is not None:
        # Enforce NC license policy on the cached entry.
        nc_licensed = bool(manifest.get("nc_licensed", False))
        license_str = str(manifest.get("license", "unknown"))
        if nc_licensed and not _tenant_allows_nc(tenant_policy):
            tenant_id = getattr(tenant_policy, "tenant_id", "unknown")
            _emit_custody(
                ReasonCode.MODEL_LOAD_BLOCKED_NC_LICENSE,
                engine=engine,
                model_id=model_id,
                license=license_str,
                tenant_id=tenant_id,
            )
            raise PolicyDenied(
                ReasonCode.MODEL_LOAD_BLOCKED_NC_LICENSE,
                f"Tenant {tenant_id!r} cannot use NC-licensed model "
                f"{engine}/{model_id} ({license_str})",
            )

        ok, bad_files = _verify_model_files(model_dir, manifest)
        if not ok:
            _emit_custody(
                ReasonCode.MODEL_INTEGRITY_FAILED,
                engine=engine,
                model_id=model_id,
                bad_files=bad_files,
            )
            raise ModelIntegrityError(engine, model_id, bad_files)

        _emit_custody(
            ReasonCode.MODEL_INTEGRITY_VERIFIED,
            engine=engine,
            model_id=model_id,
        )

        # Touch access_time under the file lock.
        with _locked_state(root) as state:
            entry = state["models"].setdefault(key, {})
            entry["access_time"] = time.time()
            entry.setdefault("pinned", False)
            entry["size_bytes"] = manifest.get("size_bytes", entry.get("size_bytes", 0))
            entry["license"] = license_str
            entry["nc_licensed"] = nc_licensed
            _save_state(root, state)

        return model_dir

    # ----- Cache miss ---------------------------------------------------
    if not allow_download:
        reason = "airgapped" if _airgapped() else "downloads disabled"
        raise ModelNotCachedError(engine, model_id, reason=reason)

    # Download into a temp dir, then atomically swap into place.
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{model_id}.dl-", dir=str(model_dir.parent))
    )
    try:
        manifest_dict = _DOWNLOAD_FN(engine, model_id, staging)
        if not isinstance(manifest_dict, dict):
            raise RuntimeError(
                f"Download fn for {engine}/{model_id} returned "
                f"{type(manifest_dict).__name__}, expected dict"
            )
        # Compute size_bytes if missing.
        if "size_bytes" not in manifest_dict:
            total = 0
            for p in staging.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
            manifest_dict["size_bytes"] = total
        # Persist manifest.
        _atomic_write_json(_manifest_path(staging), manifest_dict)

        # Verify before swap.
        ok, bad_files = _verify_model_files(staging, manifest_dict)
        if not ok:
            _emit_custody(
                ReasonCode.MODEL_INTEGRITY_FAILED,
                engine=engine,
                model_id=model_id,
                bad_files=bad_files,
                stage="post_download",
            )
            raise ModelIntegrityError(engine, model_id, bad_files)

        # NC license check on freshly-downloaded weights.
        license_str = str(manifest_dict.get("license", "unknown"))
        nc_licensed = bool(manifest_dict.get("nc_licensed", False))
        if nc_licensed and not _tenant_allows_nc(tenant_policy):
            tenant_id = getattr(tenant_policy, "tenant_id", "unknown")
            _emit_custody(
                ReasonCode.MODEL_LOAD_BLOCKED_NC_LICENSE,
                engine=engine,
                model_id=model_id,
                license=license_str,
                tenant_id=tenant_id,
            )
            # Keep the downloaded files for the next caller who's
            # allowed to use NC weights -- swap into final path before
            # raising so we don't waste the bandwidth.
            if model_dir.exists():
                shutil.rmtree(model_dir)
            shutil.move(str(staging), str(model_dir))
            staging = None  # type: ignore[assignment]
            raise PolicyDenied(
                ReasonCode.MODEL_LOAD_BLOCKED_NC_LICENSE,
                f"Tenant {tenant_id!r} cannot use NC-licensed model "
                f"{engine}/{model_id} ({license_str})",
            )

        # Swap staging -> final.
        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.move(str(staging), str(model_dir))
        staging = None  # type: ignore[assignment]

        # Update LRU bookkeeping.
        with _locked_state(root) as state:
            entry = state["models"].setdefault(key, {})
            entry["access_time"] = time.time()
            entry.setdefault("pinned", False)
            entry["size_bytes"] = manifest_dict.get("size_bytes", 0)
            entry["license"] = license_str
            entry["nc_licensed"] = nc_licensed
            _save_state(root, state)

        _emit_custody(
            ReasonCode.MODEL_DOWNLOADED,
            engine=engine,
            model_id=model_id,
            license=license_str,
            size_bytes=manifest_dict.get("size_bytes", 0),
        )
        return model_dir
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def verify_model_integrity(
    engine: str,
    model_id: str,
    *,
    cache_dir: Optional[str | Path] = None,
) -> bool:
    """Recompute SHA-256 hashes and compare against the manifest."""
    root = _resolve_cache_root(cache_dir)
    model_dir = _model_dir(root, engine, model_id)
    if not model_dir.exists():
        return False
    manifest = _read_manifest(model_dir)
    if manifest is None:
        return False
    ok, _bad = _verify_model_files(model_dir, manifest)
    return ok


def evict_lru(
    target_free_bytes: int,
    *,
    cache_dir: Optional[str | Path] = None,
) -> list[str]:
    """Evict least-recently-used unpinned models to free disk space.

    Returns the list of state keys (``"<engine>/<model_id>"``) that
    were removed.  Pinned entries are never evicted.  The function
    walks LRU order until either ``target_free_bytes`` cumulative bytes
    have been freed or no more unpinned candidates remain.
    """
    if target_free_bytes <= 0:
        return []
    root = _resolve_cache_root(cache_dir)
    evicted: list[str] = []
    with _locked_state(root) as state:
        candidates: list[tuple[float, str, dict]] = []
        for key, entry in state["models"].items():
            if entry.get("pinned"):
                continue
            access_time = float(entry.get("access_time", 0.0))
            candidates.append((access_time, key, entry))
        candidates.sort()  # oldest first

        freed = 0
        for _ts, key, entry in candidates:
            if freed >= target_free_bytes:
                break
            engine, _, model_id = key.partition("/")
            model_dir = _model_dir(root, engine, model_id)
            size = int(entry.get("size_bytes", 0))
            try:
                if model_dir.exists():
                    shutil.rmtree(model_dir)
            except OSError:
                continue
            state["models"].pop(key, None)
            freed += size
            evicted.append(key)
            _emit_custody(
                ReasonCode.MODEL_EVICTED,
                engine=engine,
                model_id=model_id,
                size_bytes=size,
            )
        if evicted:
            _save_state(root, state)
    return evicted


def pin_model(
    engine: str,
    model_id: str,
    *,
    cache_dir: Optional[str | Path] = None,
) -> None:
    """Mark a model as pinned (LRU-exempt)."""
    root = _resolve_cache_root(cache_dir)
    key = _state_key(engine, model_id)
    with _locked_state(root) as state:
        entry = state["models"].setdefault(key, {})
        entry["pinned"] = True
        entry.setdefault("access_time", time.time())
        _save_state(root, state)
    _emit_custody(ReasonCode.MODEL_PINNED, engine=engine, model_id=model_id)


def unpin_model(
    engine: str,
    model_id: str,
    *,
    cache_dir: Optional[str | Path] = None,
) -> None:
    """Clear the pinned flag on a model."""
    root = _resolve_cache_root(cache_dir)
    key = _state_key(engine, model_id)
    with _locked_state(root) as state:
        entry = state["models"].get(key)
        if entry is not None:
            entry["pinned"] = False
            _save_state(root, state)
    _emit_custody(ReasonCode.MODEL_UNPINNED, engine=engine, model_id=model_id)


def list_cached_models(
    *,
    cache_dir: Optional[str | Path] = None,
) -> list[CachedModelInfo]:
    """Return a snapshot of all currently cached models."""
    root = _resolve_cache_root(cache_dir)
    out: list[CachedModelInfo] = []
    with _locked_state(root) as state:
        for key, entry in state["models"].items():
            engine, _, model_id = key.partition("/")
            model_dir = _model_dir(root, engine, model_id)
            if not model_dir.exists():
                continue
            manifest = _read_manifest(model_dir) or {}
            out.append(
                CachedModelInfo(
                    engine=engine,
                    model_id=model_id,
                    path=model_dir,
                    size_bytes=int(
                        entry.get("size_bytes", manifest.get("size_bytes", 0))
                    ),
                    license=str(
                        entry.get("license", manifest.get("license", "unknown"))
                    ),
                    nc_licensed=bool(
                        entry.get("nc_licensed", manifest.get("nc_licensed", False))
                    ),
                    pinned=bool(entry.get("pinned", False)),
                    access_time=float(entry.get("access_time", 0.0)),
                )
            )
    return out


def register_model(
    engine: str,
    model_id: str,
    source_dir: Path,
    *,
    license: str,
    nc_licensed: bool,
    sha256_manifest: dict[str, str],
    cache_dir: Optional[str | Path] = None,
) -> Path:
    """Register a pre-baked model directory into the cache.

    Used when building air-gapped Docker images: copy the model files
    into the build, then call this helper to write the manifest and
    LRU bookkeeping.  The ``source_dir`` is *copied* into the cache
    layout, so the caller is free to delete the original after this
    returns.
    """
    source = Path(source_dir)
    if source.is_symlink() or not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"source_dir does not exist: {source}")
    if not sha256_manifest:
        raise ValueError("sha256_manifest must be a non-empty dict")

    # Validate manifest paths before copytree so a malformed pre-baked bundle
    # cannot make the verification step inspect outside the cache root.
    for filename, digest in sha256_manifest.items():
        source_target = source / filename if isinstance(filename, str) else None
        if (
            source_target is not None
            and (source_target.exists() or source_target.is_symlink())
            and _safe_manifest_target(source, filename) is None
        ):
            raise ValueError(f"unsafe manifest filename: {filename!r}")
        if _safe_manifest_target(source, filename) is None:
            # The source file may not exist yet; validate the lexical path
            # independently while still rejecting traversal/separators.
            if (
                not isinstance(filename, str)
                or not filename
                or len(filename) > _MAX_MANIFEST_FILENAME_LENGTH
                or "\x00" in filename
                or "\\" in filename
                or Path(filename).is_absolute()
                or ".." in Path(filename).parts
                or "." in Path(filename).parts
            ):
                raise ValueError(f"unsafe manifest filename: {filename!r}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid SHA-256 manifest entry: {filename!r}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"invalid SHA-256 manifest entry: {filename!r}") from exc

    root = _resolve_cache_root(cache_dir)
    model_dir = _model_dir(root, engine, model_id)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, model_dir)

    total_bytes = 0
    for p in model_dir.rglob("*"):
        if p.is_file():
            total_bytes += p.stat().st_size

    manifest = {
        "sha256": dict(sha256_manifest),
        "license": license,
        "nc_licensed": bool(nc_licensed),
        "size_bytes": total_bytes,
    }
    _atomic_write_json(_manifest_path(model_dir), manifest)

    # Verify hashes after copy so a corrupt source-dir is caught early.
    ok, bad_files = _verify_model_files(model_dir, manifest)
    if not ok:
        shutil.rmtree(model_dir, ignore_errors=True)
        raise ModelIntegrityError(engine, model_id, bad_files)

    key = _state_key(engine, model_id)
    with _locked_state(root) as state:
        entry = state["models"].setdefault(key, {})
        entry["access_time"] = time.time()
        entry.setdefault("pinned", False)
        entry["size_bytes"] = total_bytes
        entry["license"] = license
        entry["nc_licensed"] = bool(nc_licensed)
        _save_state(root, state)

    return model_dir
