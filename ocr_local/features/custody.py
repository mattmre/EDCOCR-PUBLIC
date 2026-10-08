"""Forensic chain-of-custody logging with tamper-evident hash chains.

Each processing event is cryptographically linked to the previous event
via SHA-256 hashing, creating a verifiable audit trail.

Output: JSONL files (one JSON object per line) for append-only safety.
Location: ocr_output/custody/<document_hash>.custody.jsonl
"""

import contextlib
import datetime
import hashlib
import json
import logging
import os
import re
import stat
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Optional

__all__ = [
    "CustodyChain",
    "EVENT_TYPES",
    "MAX_CUSTODY_RETRIES",
    "CUSTODY_RETRY_DELAYS",
    "MAX_CUSTODY_EVENTS",
    "CANONICAL_TIMESTAMP_PATTERN",
    "compute_file_hash",
    "verify_custody_file",
    "validate_event_envelope",
    "validate_custody_document_id",
    "append_event_to_bound_chain",
    "open_bound_chain_transaction",
]

logger = logging.getLogger(__name__)

# Retry configuration for custody chain disk writes
MAX_CUSTODY_RETRIES = 3
CUSTODY_RETRY_DELAYS = [0.1, 0.5, 2.0]  # seconds
# Optional callers (for example API evidence bundles) can impose a bounded
# load budget without changing the forensic verifier's unrestricted offline
# behavior.  The limit is intentionally opt-in because historical chains may
# legitimately exceed an API response budget.
MAX_CUSTODY_EVENTS = 100_000
_SAFE_DOCUMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class _DuplicateCustodyJSONKeyError(ValueError):
    """Reject ambiguous custody records instead of accepting the last key."""


def _reject_duplicate_custody_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateCustodyJSONKeyError(
                f"duplicate JSON key in custody event: {key}"
            )
        result[key] = value
    return result


def validate_custody_document_id(document_id: Any) -> str:
    """Return an unmodified, filename-safe forensic authority id.

    Callers that select or accept an external custody identity must reject
    path-like values instead of basename-normalizing or truncating them onto
    another chain's file.
    """
    if not isinstance(document_id, str) or not _SAFE_DOCUMENT_ID_RE.fullmatch(
        document_id
    ):
        raise ValueError("document_id must be a conservative filename-safe ID")
    return document_id


_REQUIRED_EVENT_FIELDS = {
    "document_id",
    "event_type",
    "timestamp",
    "data",
    "prev_hash",
    "hash",
}
# Lowercase SHA-256 hex only — matches schemas/custody.schema.json patterns.
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
# Canonical custody timestamp subset (runtime + schemas/custody.schema.json):
#   YYYY-MM-DDTHH:MM:SS[.fraction](Z|+HH:MM|-HH:MM)
# Application-generated values use UTC with milliseconds and +00:00.
# Rejected: space separator, week dates, lowercase z, naive, leap seconds.
# Hours 00-23, minutes/seconds 00-59; timezone Z or +/-HH:MM with same bounds.
_CANONICAL_TIMESTAMP_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"T(?P<hour>(?:[01]\d|2[0-3])):(?P<minute>[0-5]\d):(?P<second>[0-5]\d)"
    r"(?:\.(?P<fraction>\d+))?"
    r"(?:Z|(?P<off_sign>[+-])(?P<off_hour>(?:[01]\d|2[0-3])):(?P<off_minute>[0-5]\d))$"
)
# Public schema pattern string — keep in lockstep with _CANONICAL_TIMESTAMP_RE
# lexical shape.  Calendar day validity (e.g. reject Feb 30) is enforced by
# runtime semantic checks *and* by schema format=date-time when the declared
# rfc3339-validator dependency is installed for Draft7 FormatChecker.
CANONICAL_TIMESTAMP_PATTERN = (
    r"^\d{4}-\d{2}-\d{2}T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d"
    r"(?:\.\d+)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)$"
)
_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()


def _process_lock_for(lock_key: str):
    """Return the shared in-process lock for one custody authority key."""
    key = os.path.normcase(lock_key)
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROCESS_LOCKS[key] = lock
        return lock


def _lock_file_strict(lock_file) -> None:
    """Acquire a blocking OS lock without silently degrading to a no-op."""
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        return

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - unsupported production OS
        raise RuntimeError(
            "Custody file locking is unsupported on this platform"
        ) from exc

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)


def _unlock_file_strict(lock_file) -> None:
    """Release the OS lock acquired by :func:`_lock_file_strict`."""
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        return

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - unsupported production OS
        raise RuntimeError(
            "Custody file locking is unsupported on this platform"
        ) from exc

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class _CustodyAuthorityError(OSError):
    """The custody pathname no longer identifies the locked evidence inode."""


_CUSTODY_OUTCOME_SAFE_UNCOMMITTED = "safe_uncommitted"
_CUSTODY_OUTCOME_COMMITTED = "committed"
_CUSTODY_OUTCOME_ROLLBACK_UNCERTAIN = "rollback_uncertain"


class _CustodyTransactionState:
    """Outcome shared by a bound transaction and its lock context.

    The state belongs to the transaction, but the exact-inode context owns the
    irreversible commit boundary. Sharing this small mutable object lets the
    context publish that boundary before unlock or descriptor cleanup can
    raise, including non-``Exception`` control-flow signals.
    """

    def __init__(self, *, preserve_original_failure: bool = False) -> None:
        self.outcome = _CUSTODY_OUTCOME_SAFE_UNCOMMITTED
        self.rollback: Callable[[], bool] | None = None
        self.preserve_original_failure = preserve_original_failure


def _finish_custody_cleanup(
    action,
    *,
    durable_commit: bool,
    label: str,
    propagate_postcommit_failure: bool = False,
) -> None:
    """Run post-context cleanup without turning one commit into two writes.

    Once the locked inode, canonical name, and append journal have passed the
    success boundary, an unlock/close error is resource-cleanup debt rather
    than evidence that the append failed. Generic append calls suppress that
    debt so it cannot enter their retry loop. Bound transactions may propagate
    the original signal after publishing their outcome, allowing an outer saga
    to preserve committed artifacts without swallowing control flow. During
    exception unwinding, cleanup errors are logged instead of replacing the
    primary failure.
    """
    unwinding = sys.exc_info()[0] is not None
    try:
        action()
    except BaseException as exc:
        if durable_commit and propagate_postcommit_failure:
            raise
        if not (durable_commit or unwinding):
            raise
        logger.critical(
            "Custody %s failed %s a durable commit; no blind retry was "
            "attempted: %s",
            label,
            "after" if durable_commit else "while unwinding",
            exc,
        )


def _rollback_failed_custody_context(
    *,
    custody_fd: int,
    append_journal: list[tuple[int, bytes]],
    transaction_state: _CustodyTransactionState,
    original_failure: BaseException,
) -> bool:
    """Try the registered exact rollback and classify an uncertain refusal.

    Bound transactions can register a post-PREPARED savepoint rollback.
    Ordinary single-event appends use the complete journal. A bound caller
    must receive its original exception/control-flow signal; the tri-state
    outcome carries rollback uncertainty without destroying that traceback.
    Generic append callers instead receive ``_CustodyAuthorityError`` so the
    retry loop cannot duplicate an append after an unproven rollback.
    """
    try:
        if transaction_state.rollback is not None:
            rolled_back = transaction_state.rollback()
        else:
            rolled_back = _rollback_append_journal(custody_fd, append_journal)
            if rolled_back:
                append_journal.clear()
    except BaseException as rollback_failure:
        transaction_state.outcome = _CUSTODY_OUTCOME_ROLLBACK_UNCERTAIN
        logger.critical(
            "Custody rollback failed; transaction outcome is rollback_uncertain: %s",
            rollback_failure,
        )
        if transaction_state.preserve_original_failure:
            if hasattr(original_failure, "add_note"):
                original_failure.add_note(
                    "custody rollback failed; outcome=rollback_uncertain: "
                    f"{rollback_failure}"
                )
            return False
        raise _CustodyAuthorityError(
            "custody transaction rollback could not be completed safely"
        ) from rollback_failure
    if rolled_back:
        transaction_state.outcome = _CUSTODY_OUTCOME_SAFE_UNCOMMITTED
        return True

    transaction_state.outcome = _CUSTODY_OUTCOME_ROLLBACK_UNCERTAIN
    logger.critical(
        "Custody rollback refused an unproven tail; transaction outcome is "
        "rollback_uncertain"
    )
    if transaction_state.preserve_original_failure:
        if hasattr(original_failure, "add_note"):
            original_failure.add_note(
                "custody rollback refused an unproven tail; "
                "outcome=rollback_uncertain"
            )
        return False
    raise _CustodyAuthorityError(
        "custody transaction rollback refused an unproven tail"
    ) from original_failure


def _assert_chain_name_inode(
    custody_dir_fd: int,
    filename: str,
    metadata: os.stat_result,
) -> None:
    try:
        current = os.stat(
            filename,
            dir_fd=custody_dir_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise _CustodyAuthorityError("canonical custody name disappeared") from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or current.st_nlink != 1
        or (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino)
    ):
        raise _CustodyAuthorityError("canonical custody name changed")


def _assert_directory_path_inode(path: str, directory_fd: int) -> None:
    """Require a lexical custody directory to still name the held inode."""
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise _CustodyAuthorityError("canonical custody directory disappeared") from exc
    opened = os.fstat(directory_fd)
    if (
        not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise _CustodyAuthorityError("canonical custody directory changed")


@contextlib.contextmanager
def _open_locked_chain_at(
    custody_dir_fd: int,
    filename: str,
    *,
    create: bool,
    writable: bool,
    final_authority_check: Callable[[], None] | None = None,
    transaction_state: _CustodyTransactionState | None = None,
):
    """Open and exclusively lock the authoritative custody JSONL inode.

    A replaceable sidecar lock pathname is not lock authority. All conforming
    writers lock the evidence inode itself and use the same descriptor for
    reload, verification, append, fsync, and rollback.
    """
    if Path(filename).name != filename or not filename.endswith(".custody.jsonl"):
        raise ValueError("custody filename must be a safe JSONL leaf")
    flags = ((os.O_RDWR | os.O_APPEND) if writable else os.O_RDONLY) | (
        getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    if create:
        if not writable:
            raise ValueError("creating a custody chain requires writable access")
        try:
            custody_fd = os.open(
                filename,
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=custody_dir_fd,
            )
        except FileExistsError:
            custody_fd = os.open(filename, flags, dir_fd=custody_dir_fd)
    else:
        custody_fd = os.open(filename, flags, dir_fd=custody_dir_fd)
    durable_commit = False
    state = transaction_state or _CustodyTransactionState()
    try:
        metadata = os.fstat(custody_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OSError("custody file must be a single-link regular file")
        process_lock = _process_lock_for(
            f"custody-inode-{metadata.st_dev}-{metadata.st_ino}"
        )
        with process_lock:
            lock_mode = "r+b" if writable else "rb"
            lock_file = os.fdopen(os.dup(custody_fd), lock_mode)
            lock_acquired = False
            try:
                _lock_file_strict(lock_file)
                lock_acquired = True
                locked_metadata = os.fstat(custody_fd)
                if (locked_metadata.st_dev, locked_metadata.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise _CustodyAuthorityError(
                        "canonical custody inode changed before lock"
                    )
                _assert_chain_name_inode(
                    custody_dir_fd,
                    filename,
                    locked_metadata,
                )
                append_journal: list[tuple[int, bytes]] = []
                try:
                    yield custody_fd, locked_metadata, append_journal
                except BaseException as exc:
                    if append_journal:
                        _rollback_failed_custody_context(
                            custody_fd=custody_fd,
                            append_journal=append_journal,
                            transaction_state=state,
                            original_failure=exc,
                        )
                    raise
                else:
                    try:
                        _assert_chain_name_inode(
                            custody_dir_fd,
                            filename,
                            locked_metadata,
                        )
                        if final_authority_check is not None:
                            final_authority_check()
                        if append_journal:
                            # The JSONL fd fsync makes event bytes durable;
                            # the directory fsync makes the canonical name
                            # durable as well (including first creation).
                            os.fsync(custody_dir_fd)
                    except BaseException as exc:
                        if append_journal:
                            _rollback_failed_custody_context(
                                custody_fd=custody_fd,
                                append_journal=append_journal,
                                transaction_state=state,
                                original_failure=exc,
                            )
                        raise
                    durable_commit = bool(append_journal)
                    if durable_commit:
                        # Publish the irreversible outcome before unlock or
                        # any descriptor cleanup can produce a signal.
                        state.outcome = _CUSTODY_OUTCOME_COMMITTED
            finally:
                try:
                    if lock_acquired:
                        _finish_custody_cleanup(
                            lambda: _unlock_file_strict(lock_file),
                            durable_commit=durable_commit,
                            label="evidence-inode unlock",
                            propagate_postcommit_failure=(
                                state.preserve_original_failure
                            ),
                        )
                finally:
                    _finish_custody_cleanup(
                        lock_file.close,
                        durable_commit=durable_commit,
                        label="evidence-lock descriptor close",
                        propagate_postcommit_failure=state.preserve_original_failure,
                    )
    finally:
        _finish_custody_cleanup(
            lambda: os.close(custody_fd),
            durable_commit=durable_commit,
            label="evidence descriptor close",
            propagate_postcommit_failure=state.preserve_original_failure,
        )


@contextlib.contextmanager
def _open_locked_chain_path(
    filepath: str,
    *,
    create: bool,
    writable: bool,
):
    """Path wrapper that binds the parent directory for one chain lifetime."""
    absolute = os.path.abspath(os.fspath(filepath))
    parent = os.path.dirname(absolute) or os.curdir
    filename = os.path.basename(absolute)
    directory_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    durable_commit = False
    try:
        _assert_directory_path_inode(parent, directory_fd)
        with _open_locked_chain_at(
            directory_fd,
            filename,
            create=create,
            writable=writable,
            final_authority_check=lambda: _assert_directory_path_inode(
                parent,
                directory_fd,
            ),
        ) as (custody_fd, metadata, append_journal):
            yield custody_fd, metadata, directory_fd, filename, append_journal
        durable_commit = bool(append_journal)
    finally:
        _finish_custody_cleanup(
            lambda: os.close(directory_fd),
            durable_commit=durable_commit,
            label="parent-directory descriptor close",
        )


def _read_locked_chain_bytes(
    custody_fd: int,
    *,
    max_bytes: int | None = None,
) -> bytes:
    """Read one stable snapshot from an already-locked custody inode."""
    before = os.fstat(custody_fd)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("Custody chain must be stored in a single-link regular file")
    if max_bytes is not None and before.st_size > max_bytes:
        raise ValueError(f"Custody chain exceeds the {max_bytes}-byte safety limit")
    chunks: list[bytes] = []
    offset = 0
    remaining = before.st_size
    while remaining:
        chunk = os.pread(custody_fd, min(remaining, 64 * 1024), offset)
        if not chunk:
            raise ValueError("Custody chain changed during locked read")
        chunks.append(chunk)
        offset += len(chunk)
        remaining -= len(chunk)
    after = os.fstat(custody_fd)

    def identity(item: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
            item.st_nlink,
        )

    if identity(before) != identity(after):
        raise ValueError("Custody chain changed during locked read")
    return b"".join(chunks)


def _rollback_exact_attempt(
    custody_fd: int,
    *,
    pre_append_size: int,
    encoded: bytes,
    bytes_written: int,
) -> bool:
    """Rollback only a tail proven byte-for-byte to belong to this attempt."""
    current = os.fstat(custody_fd)
    if current.st_size == pre_append_size and bytes_written == 0:
        return True
    if current.st_size != pre_append_size + bytes_written:
        return False
    if bytes_written:
        tail = os.pread(custody_fd, bytes_written, pre_append_size)
        if tail != encoded[:bytes_written]:
            return False
    os.ftruncate(custody_fd, pre_append_size)
    os.fsync(custody_fd)
    return True


def _rollback_append_journal(
    custody_fd: int,
    append_journal: list[tuple[int, bytes]],
) -> bool:
    """Rollback one contiguous, byte-proven series of locked appends."""
    if not append_journal:
        return True
    pre_append_size = append_journal[0][0]
    expected_offset = pre_append_size
    encoded_parts: list[bytes] = []
    for offset, encoded in append_journal:
        if offset != expected_offset:
            return False
        encoded_parts.append(encoded)
        expected_offset += len(encoded)
    encoded_tail = b"".join(encoded_parts)
    return _rollback_exact_attempt(
        custody_fd,
        pre_append_size=pre_append_size,
        encoded=encoded_tail,
        bytes_written=len(encoded_tail),
    )


def _validate_event_type(event_type: Any, *, context: str = "event_type") -> str:
    """Fail closed unless ``event_type`` is a registered registry/schema value."""
    if not isinstance(event_type, str) or event_type not in EVENT_TYPES:
        raise ValueError(f"unknown {context}: {event_type!r}")
    return event_type


def _validate_event_data(
    data: Any,
    *,
    allow_none: bool = False,
    context: str = "data",
) -> dict[str, Any]:
    """Fail closed unless ``data`` is mapping-shaped JSON object material.

    Append accepts ``None`` and normalizes it to ``{}``.  Load/verify require a
    real JSON object so non-mapping on-disk payloads cannot enter the chain.
    """
    if data is None:
        if allow_none:
            return {}
        raise ValueError(f"{context} must be a JSON object")
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a JSON object")
    return data


def _validate_timestamp(timestamp: Any, *, context: str = "timestamp") -> str:
    """Require the canonical custody timestamp subset.

    Accepted application/schema form:

    * ``YYYY-MM-DDTHH:MM:SS`` with optional fractional seconds
    * timezone: uppercase ``Z`` or numeric ``+/-HH:MM`` (including ``+00:00``)
    * seconds in ``00``–``59`` (leap seconds rejected)

    Lexical shape matches ``schemas/custody.schema.json``
    ``properties.timestamp.pattern`` / ``CANONICAL_TIMESTAMP_PATTERN``.
    Semantic calendar and clock ranges are validated here.  Public schema
    ``format=date-time`` plus the declared ``rfc3339-validator`` dependency
    also rejects calendar-invalid values under Draft7 FormatChecker; this
    runtime path remains authoritative even without the format checker.

    Rejected: space separators, week dates, lowercase ``z``, naive values,
    leap seconds, and invalid dates/offsets.
    """
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError(f"{context} must be a timezone-aware RFC3339 date-time string")
    match = _CANONICAL_TIMESTAMP_RE.fullmatch(timestamp)
    if match is None:
        raise ValueError(f"{context} must be a timezone-aware RFC3339 date-time string")

    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))

    if not (1 <= month <= 12):
        raise ValueError(f"{context} must be a timezone-aware RFC3339 date-time string")
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(f"{context} must be a timezone-aware RFC3339 date-time string")

    if match.group("off_sign") is not None:
        off_hour = int(match.group("off_hour"))
        off_minute = int(match.group("off_minute"))
        if not (0 <= off_hour <= 23 and 0 <= off_minute <= 59):
            raise ValueError(
                f"{context} must be a timezone-aware RFC3339 date-time string"
            )

    # Calendar validity (rejects 2026-02-31, 2026-04-31, etc.).
    try:
        datetime.date(year, month, day)
    except ValueError as exc:
        raise ValueError(
            f"{context} must be a timezone-aware RFC3339 date-time string"
        ) from exc

    return timestamp


def _validate_sha256_hex(
    value: Any,
    *,
    allow_null: bool = False,
    context: str = "hash",
) -> Optional[str]:
    """Require null (optional) or lowercase 64-char SHA-256 hex."""
    if value is None:
        if allow_null:
            return None
        raise ValueError(f"{context} must be lowercase 64-char SHA-256 hex")
    if not isinstance(value, str) or not _SHA256_HEX_RE.fullmatch(value):
        raise ValueError(f"{context} must be lowercase 64-char SHA-256 hex")
    return value


def validate_event_envelope(event: Any, *, context: str = "event") -> dict[str, Any]:
    """Validate a completed custody event against the public envelope contract.

    Enforced fields (aligned with ``schemas/custody.schema.json``):

    * object with all required keys
    * non-empty string ``document_id``
    * ``event_type`` in :data:`EVENT_TYPES`
    * timezone-aware RFC3339/date-time ``timestamp``
    * mapping ``data``
    * ``prev_hash`` null or lowercase 64-char SHA-256 hex
    * ``hash`` lowercase 64-char SHA-256 hex

    Additional properties are allowed (schema does not set
    ``additionalProperties: false``).  Hash-link / recomputation checks remain
    the caller's responsibility after envelope validation succeeds.
    """
    if not isinstance(event, dict):
        raise ValueError(f"{context} must be a JSON object")
    missing_fields = _REQUIRED_EVENT_FIELDS.difference(event)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"{context} missing fields: {missing}")

    document_id = event["document_id"]
    if not isinstance(document_id, str) or not document_id:
        raise ValueError(f"{context} document_id must be a non-empty string")

    # Field validators keep stable field-centric messages (unknown event_type,
    # data must be a JSON object, ...) so load/verify call sites and tests can
    # match them without encoding the outer context string.
    _validate_event_type(event["event_type"])
    _validate_timestamp(event["timestamp"])
    _validate_event_data(event["data"])
    _validate_sha256_hex(event["prev_hash"], allow_null=True, context="prev_hash")
    _validate_sha256_hex(event["hash"], context="hash")
    return event


# Public alias kept stable for tests/callers that prefer the underscore form.
_validate_event_envelope = validate_event_envelope


def _generated_event(
    *,
    document_id: str,
    prev_hash: str | None,
    event_type: str,
    data: dict[str, Any] | None,
) -> tuple[dict[str, Any], bytes]:
    event_type = _validate_event_type(event_type)
    event_data = _validate_event_data(data, allow_none=True)
    event = {
        "document_id": document_id,
        "event_type": event_type,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "data": event_data,
        "prev_hash": prev_hash,
    }
    event_bytes = json.dumps(event, sort_keys=True, default=str).encode("utf-8")
    event["hash"] = hashlib.sha256(event_bytes).hexdigest()
    validate_event_envelope(event, context="generated event")
    return event, json.dumps(event, default=str).encode("utf-8") + b"\n"


def _append_event_to_locked_fd(
    *,
    custody_fd: int,
    custody_dir_fd: int,
    filename: str,
    document_id: str,
    prev_hash: str | None,
    event_type: str,
    data: dict[str, Any] | None,
    max_bytes: int | None = None,
    append_journal: list[tuple[int, bytes]] | None = None,
) -> dict[str, Any]:
    """Append one event on the exact locked fd, with proof-bound rollback."""
    before = os.fstat(custody_fd)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise OSError("custody file must be a single-link regular file")
    _assert_chain_name_inode(custody_dir_fd, filename, before)
    event, encoded_event = _generated_event(
        document_id=document_id,
        prev_hash=prev_hash,
        event_type=event_type,
        data=data,
    )
    prefix = b""
    if before.st_size and os.pread(custody_fd, 1, before.st_size - 1) != b"\n":
        prefix = b"\n"
    encoded = prefix + encoded_event
    if max_bytes is not None and before.st_size + len(encoded) > max_bytes:
        raise ValueError("custody append would exceed the configured byte limit")
    bytes_written = 0
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(custody_fd, view)
            if written <= 0:
                raise OSError("custody append made no progress")
            bytes_written += written
            view = view[written:]
        os.fsync(custody_fd)
        # Durable bytes must be journaled before any later check can raise.
        # An empty savepoint suffix would otherwise treat this append as
        # never-written and authorize restoration.
        if append_journal is not None:
            append_journal.append((before.st_size, encoded))
        after = os.fstat(custody_fd)
        if (after.st_dev, after.st_ino) != (
            before.st_dev,
            before.st_ino,
        ) or after.st_size != before.st_size + len(encoded):
            raise _CustodyAuthorityError(
                "canonical custody inode changed during append"
            )
        _assert_chain_name_inode(custody_dir_fd, filename, before)
    except BaseException as exc:
        try:
            rolled_back = _rollback_exact_attempt(
                custody_fd,
                pre_append_size=before.st_size,
                encoded=encoded,
                bytes_written=bytes_written,
            )
        except BaseException as rollback_exc:
            raise _CustodyAuthorityError(
                "custody append rollback could not be completed safely"
            ) from rollback_exc
        if rolled_back and append_journal:
            recorded = (before.st_size, encoded)
            if append_journal[-1] == recorded:
                append_journal.pop()
        if not rolled_back:
            raise _CustodyAuthorityError(
                "custody append rollback refused an unproven tail"
            ) from exc
        raise
    return event


class CustodyChain:
    """Hash-chained event log for a single document's processing lifecycle.

    Each event contains:
    - event_type: Processing stage identifier
    - data: Stage-specific metadata
    - timestamp: ISO 8601 with millisecond precision
    - prev_hash: SHA-256 hash of previous event (None for genesis)
    - hash: SHA-256 hash of this event (computed from all fields)
    """

    def __init__(self, document_id: str, source_path: str, custody_dir: str = ""):
        """Initialize a new custody chain for a document.

        Args:
            document_id: Unique document identifier (e.g., SHA-256 hash prefix)
            source_path: Original source file path
            custody_dir: Directory for .custody.jsonl files
        """
        self.document_id = document_id
        self.source_path = source_path
        self.events: list[dict] = []
        self._prev_hash: Optional[str] = None
        self._custody_dir = custody_dir
        self._filepath: Optional[str] = None
        self._lock = threading.RLock()
        self.integrity_compromised = False

        if custody_dir:
            validate_custody_document_id(document_id)
            os.makedirs(custody_dir, exist_ok=True)
            self._filepath = os.path.join(
                custody_dir,
                f"{document_id}.custody.jsonl",
            )
            try:
                with _open_locked_chain_path(
                    self._filepath,
                    create=False,
                    writable=True,
                ) as (
                    custody_fd,
                    _metadata,
                    _directory_fd,
                    _filename,
                    _append_journal,
                ):
                    self._reload_existing_chain_from_fd_unlocked(custody_fd)
            except FileNotFoundError:
                pass

    def append_event(self, event_type: str, data: dict[str, Any] | None = None) -> dict:
        """Append a hash-chained event to the custody log.

        Args:
            event_type: One of the defined event types (see EVENT_TYPES)
            data: Stage-specific metadata dictionary

        Returns:
            The complete event dict with computed hash
        """
        if self._filepath:
            _validate_event_type(event_type)
            _validate_event_data(data, allow_none=True)
            for attempt in range(1, MAX_CUSTODY_RETRIES + 1):
                try:
                    with _open_locked_chain_path(
                        self._filepath,
                        create=True,
                        writable=True,
                    ) as (
                        custody_fd,
                        _metadata,
                        directory_fd,
                        filename,
                        append_journal,
                    ):
                        self._reload_existing_chain_from_fd_unlocked(custody_fd)
                        event = _append_event_to_locked_fd(
                            custody_fd=custody_fd,
                            custody_dir_fd=directory_fd,
                            filename=filename,
                            document_id=self.document_id,
                            prev_hash=self._prev_hash,
                            event_type=event_type,
                            data=data,
                            append_journal=append_journal,
                        )
                    self.events.append(event)
                    self._prev_hash = event["hash"]
                    return event
                except _CustodyAuthorityError:
                    self.integrity_compromised = True
                    raise
                except OSError as exc:
                    if attempt < MAX_CUSTODY_RETRIES:
                        delay = CUSTODY_RETRY_DELAYS[attempt - 1]
                        logger.warning(
                            "Custody write attempt %d failed: %s, retrying in %ss",
                            attempt,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                    logger.critical(
                        "Custody chain write FAILED after %d attempts for %s "
                        "-- forensic integrity compromised",
                        MAX_CUSTODY_RETRIES,
                        self.document_id,
                    )
                    self.integrity_compromised = True
                    raise OSError(
                        "custody chain write failed after "
                        f"{MAX_CUSTODY_RETRIES} attempts"
                    ) from exc

        with self._lock:
            return self._append_event_unlocked(event_type, data)

    def _reload_existing_chain_from_fd_unlocked(self, custody_fd: int) -> None:
        """Reload and verify the exact already-locked evidence inode."""
        raw = _read_locked_chain_bytes(custody_fd)
        if not raw:
            self.events = []
            self._prev_hash = None
            return
        try:
            existing = type(self).load_from_bytes(raw)
        except (UnicodeError, ValueError) as exc:
            raise ValueError(f"Existing custody chain unreadable: {exc}") from exc
        if existing.document_id != self.document_id:
            raise ValueError(
                "Existing custody document_id does not match requested chain"
            )
        is_valid, message = existing.verify_chain()
        if not is_valid:
            raise ValueError(f"Existing custody chain failed verification: {message}")
        self.events = existing.events
        self._prev_hash = existing._prev_hash

    def _append_event_unlocked(
        self,
        event_type: str,
        data: dict[str, Any] | None,
    ) -> dict:
        """Create one event for a memory-only chain while its lock is held."""
        event, _encoded = _generated_event(
            document_id=self.document_id,
            prev_hash=self._prev_hash,
            event_type=event_type,
            data=data,
        )
        self.events.append(event)
        self._prev_hash = event["hash"]

        return event

    def log_event(self, event_type: str, data: dict[str, Any] | None = None) -> dict:
        """Alias for :meth:`append_event` -- preferred name for new callers.

        Plan B M1-PR3 introduced the ``log_event`` name to align translation
        custody adapters with the language-detection / translation panel
        playbook.  ``append_event`` remains for backwards compatibility.
        """

        return self.append_event(event_type, data)

    def verify_chain(self) -> tuple[bool, str]:
        """Verify the integrity of the entire chain.

        Returns:
            (is_valid, message) tuple
        """
        if not self.events:
            return True, "Empty chain"

        prev_hash = None
        for i, event in enumerate(self.events):
            try:
                validate_event_envelope(event, context=f"event {i}")
            except ValueError as exc:
                return False, f"Malformed event {i}: {exc}"
            if event["document_id"] != self.document_id:
                return (
                    False,
                    f"Document ID mismatch at event {i}: "
                    f"expected {self.document_id}, got {event['document_id']}",
                )

            # Check prev_hash linkage
            if event["prev_hash"] != prev_hash:
                return (
                    False,
                    f"Broken chain at event {i}: expected prev_hash={prev_hash}, got {event['prev_hash']}",
                )

            # Recompute hash
            event_copy = {k: v for k, v in event.items() if k != "hash"}
            event_bytes = json.dumps(event_copy, sort_keys=True, default=str).encode(
                "utf-8"
            )
            computed_hash = hashlib.sha256(event_bytes).hexdigest()

            if computed_hash != event["hash"]:
                return (
                    False,
                    f"Tampered event {i}: computed hash {computed_hash} != stored {event['hash']}",
                )

            prev_hash = event["hash"]

        return True, f"Chain verified: {len(self.events)} events"

    @classmethod
    def load_from_file(
        cls,
        filepath: str,
        *,
        max_bytes: int | None = None,
        max_events: int | None = None,
    ) -> "CustodyChain":
        """Load a custody chain from a JSONL file.

        Args:
            filepath: Path to .custody.jsonl file

        Returns:
            CustodyChain instance with loaded events
        """
        if max_bytes is not None and (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or max_bytes < 1
        ):
            raise ValueError("max_bytes must be a positive integer")
        if max_events is not None and (
            not isinstance(max_events, int)
            or isinstance(max_events, bool)
            or max_events < 1
        ):
            raise ValueError("max_events must be a positive integer")

        if max_events is None:
            max_events = MAX_CUSTODY_EVENTS

        filepath = os.fspath(filepath)
        with _open_locked_chain_path(
            filepath,
            create=False,
            writable=False,
        ) as (
            custody_fd,
            _metadata,
            _directory_fd,
            _filename,
            _append_journal,
        ):
            try:
                raw = _read_locked_chain_bytes(custody_fd, max_bytes=max_bytes)
                chain = cls.load_from_bytes(
                    raw,
                    max_bytes=max_bytes,
                    max_events=max_events,
                )
            except UnicodeError as exc:
                raise ValueError(f"Custody chain unreadable: {exc}") from exc
        chain._filepath = filepath
        return chain

    @classmethod
    def load_from_bytes(
        cls,
        raw: bytes,
        *,
        max_bytes: int | None = None,
        max_events: int | None = None,
    ) -> "CustodyChain":
        """Parse a custody JSONL payload already read through a path jail."""
        if not isinstance(raw, bytes):
            raise TypeError("raw must be bytes")
        if max_events is not None and (
            not isinstance(max_events, int)
            or isinstance(max_events, bool)
            or max_events < 1
        ):
            raise ValueError("max_events must be a positive integer")
        if max_events is None:
            max_events = MAX_CUSTODY_EVENTS
        if max_bytes is not None and (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or max_bytes < 1
        ):
            raise ValueError("max_bytes must be a positive integer")
        if max_bytes is not None and len(raw) > max_bytes:
            raise ValueError(f"Custody chain exceeds the {max_bytes}-byte safety limit")
        chain = cls(document_id="", source_path="")
        saw_content = False
        bytes_read = 0
        for line_number, raw_line in enumerate(raw.splitlines(keepends=True), start=1):
            bytes_read += len(raw_line)
            if max_bytes is not None and bytes_read > max_bytes:
                raise ValueError(
                    f"Custody chain exceeds the {max_bytes}-byte safety limit"
                )
            line = raw_line.strip()
            if not line:
                continue
            saw_content = True
            try:
                event = json.loads(
                    line.decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_custody_json_keys,
                )
            except _DuplicateCustodyJSONKeyError as exc:
                raise ValueError(
                    f"Custody chain unreadable at line {line_number}: {exc}"
                ) from exc
            except UnicodeError as exc:
                raise ValueError(
                    f"Custody chain unreadable at line {line_number}: invalid UTF-8"
                ) from exc
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Custody chain unreadable at line {line_number}: {exc.msg}"
                ) from exc
            try:
                validate_event_envelope(event, context="event")
            except ValueError as exc:
                raise ValueError(
                    f"Custody chain unreadable at line {line_number}: {exc}"
                ) from exc
            if len(chain.events) >= max_events:
                raise ValueError(
                    f"Custody chain exceeds the {max_events}-event safety limit"
                )
            chain.events.append(event)
            chain._prev_hash = event["hash"]
            if not chain.document_id:
                chain.document_id = event["document_id"]
                chain.source_path = event["data"].get("source_path", "")
        if raw and not saw_content:
            raise ValueError("Custody chain unreadable: non-empty file has no events")
        return chain

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the custody chain for metadata/reporting."""
        compromised = bool(self.integrity_compromised)
        return {
            "document_id": self.document_id,
            "total_events": len(self.events),
            "event_types": list({e["event_type"] for e in self.events}),
            "first_event": self.events[0]["timestamp"] if self.events else None,
            "last_event": self.events[-1]["timestamp"] if self.events else None,
            "chain_hash": self._prev_hash,
            # A memory-only chain is useful for local inspection but is not a
            # durable forensic record.  A failed durable append also makes
            # the chain untrustworthy until an operator starts a new chain.
            "durable": bool(self._filepath) and not compromised,
            "integrity_compromised": compromised,
        }


# Defined event types for the processing pipeline
EVENT_TYPES = {
    "file_ingested": "Source file accepted for processing",
    "pages_extracted": "Document pages extracted and fan-out work prepared",
    "page_extracted": "Page image extracted from document",
    "text_extracted_natively": "Source text extracted without OCR from a native text or PDF page",
    "ocr_primary": "Primary OCR engine (PaddleOCR) processed page",
    "ocr_fallback": "Fallback OCR engine (Tesseract) processed page",
    "ocr_image_only": "OCR failed, page preserved as image",
    "language_detected": "Language detection completed",
    "language_reprocess": "Page re-processed with detected language model",
    "docintel_analysis": "Document Intelligence analysis completed",
    "assembly_complete": "Final document assembled from pages",
    "assembly_failed": "Final document assembly failed",
    "compression_complete": "PDF compression/optimization completed",
    "compression_skipped": (
        "PDF compression skipped (visual-validation budget); original bytes retained"
    ),
    "processing_resumed": "Processing resumed from durable page checkpoints",
    "compression_failed": "PDF compression or final metadata binding failed",
    "dpi_escalation": "Page re-extracted at higher DPI due to low OCR confidence",
    "processing_failed": "Processing stage failed",
    "temp_files_purged": "Orphaned temporary files purged from storage",
    "pii_purged": "PII/PHI entity records purged on demand",
    "tenant_purged": "All data for a tenant purged on demand",
    "output_cleaned": "Output files cleaned per retention policy",
    "audit_logs_rotated": "Audit log records archived and rotated",
    # Transform/stamp lifecycle events emitted by custody_hooks.py
    "transform_start": "Document transform started",
    "transform_complete": "Document transform completed",
    "transform_failed": "Document transform failed",
    "stamp_start": "Document stamp operation started",
    "stamp_complete": "Document stamp operation completed",
    "stamp_failed": "Document stamp operation failed",
    # Engine and language provenance emitted by production OCR paths
    "ENGINE_PROVENANCE": "OCR engine provenance and policy admission recorded",
    "LICENSE_BLOCKED": "OCR engine blocked by license or policy",
    "LANE_BOUNDARY_ENFORCED": "Forensic lane boundary blocked a generative engine",
    "LANGUAGE_DETECTED": "Per-page language detection completed",
    "LANGUAGE_MIXED_SCRIPT": "Mixed scripts detected on a page",
    "LANGUAGE_REDETECT_PREPARED": (
        "Language re-detection sidecar was staged and reserved for publication"
    ),
    "LANGUAGE_REDETECTED": "Language was re-detected after OCR processing",
    # Plan B M1-PR3 -- translation pipeline events
    "TRANSLATION_APPLIED": "Translation engine applied to OCR spans",
    "TRANSLATION_REJECTED": "Translation rejected by tenant policy or privilege guard",
    "QUALITY_BELOW_THRESHOLD": "Translation quality score below configured threshold",
    "TRANSLATION_FALLBACK": "Translation fell back from one engine to another",
    "TRANSLATION_REVIEWED": "Human reviewer approved/rejected a translation (forensic gate)",
    "GLOSSARY_APPLIED": "Glossary term overrides applied to translation output",
    "TRANSLATION_SKIPPED": "Translation pipeline skipped for this document/span",
    "TRANSLATION_MODEL_CACHE": "Translation model-cache lifecycle event recorded",
    "TRANSLATION_QUALITY_ESTIMATED": "Translation quality estimation completed",
    "TRANSLATION_COMPLETED": "Translation completed (legacy quality-event carrier)",
    "CUSTODY_TSA_WARNING": "Timestamp authority outage or warning recorded for custody anchoring",
    "TENANT_ENGINE_SELECTED": "Tenant-aware translation engine selected",
    "TENANT_ENGINE_NO_CANDIDATES": "No eligible tenant translation engine remained",
    "TENANT_POLICY_HYDRATED": "Tenant translation policy loaded and hashed",
    "BATCH_SUBMITTED": "Translation batch accepted and persisted",
    "BATCH_FAN_OUT": "Translation batch inputs dispatched",
    "BATCH_INPUT_COMPLETED": "Translation batch input completed",
    "BATCH_INPUT_FAILED": "Translation batch input failed",
    "BATCH_CANCELLED": "Translation batch cancelled",
    "BATCH_COMPLETED": "Translation batch reached a terminal state",
    "BATCH_REJECTED_CERTIFIED": "Certified translation batch rejected at submission",
    "PLUGIN_REGISTERED": "Enrichment plugin admitted to the Plan D plugin bus",
    "PLUGIN_REJECTED": "Enrichment plugin rejected by Plan D plugin-bus admission",
}


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_custody_file(
    filepath: str,
    *,
    max_bytes: int | None = None,
    max_events: int | None = None,
) -> tuple[bool, str]:
    """Convenience function to verify a custody JSONL file.

    Args:
        filepath: Path to .custody.jsonl file

    Returns:
        (is_valid, message) tuple
    """
    try:
        chain = CustodyChain.load_from_file(
            filepath,
            max_bytes=max_bytes,
            max_events=max_events,
        )
        return chain.verify_chain()
    except (OSError, UnicodeError, ValueError) as exc:
        return False, f"Failed to load custody file: {exc}"


class _BoundCustodyTransaction:
    """One exact-inode custody transaction held under the evidence-file lock."""

    def __init__(
        self,
        *,
        custody_fd: int,
        custody_dir_fd: int,
        filename: str,
        metadata: os.stat_result,
        chain: CustodyChain,
        max_bytes: int,
        max_events: int,
        append_journal: list[tuple[int, bytes]],
        transaction_state: _CustodyTransactionState,
    ) -> None:
        self.custody_fd = custody_fd
        self.custody_dir_fd = custody_dir_fd
        self.filename = filename
        self.metadata = metadata
        self.chain = chain
        self.max_bytes = max_bytes
        self.max_events = max_events
        self._append_journal = append_journal
        self._initial_event_count = len(chain.events)
        self._initial_size = metadata.st_size
        self._rollback_journal_index = 0
        self._rollback_event_count = self._initial_event_count
        self._transaction_state = transaction_state
        self._transaction_state.rollback = self.rollback_to_savepoint

    @property
    def document_id(self) -> str:
        return self.chain.document_id

    @property
    def chain_head(self) -> str:
        return str(self.chain._prev_hash or "")

    @property
    def size_bytes(self) -> int:
        return os.fstat(self.custody_fd).st_size

    @property
    def outcome(self) -> str:
        """Return ``safe_uncommitted``, ``committed``, or ``rollback_uncertain``."""
        return self._transaction_state.outcome

    def assert_authority(self) -> None:
        current = os.fstat(self.custody_fd)
        if (current.st_dev, current.st_ino) != (
            self.metadata.st_dev,
            self.metadata.st_ino,
        ):
            raise _CustodyAuthorityError("canonical custody inode changed")
        _assert_chain_name_inode(
            self.custody_dir_fd,
            self.filename,
            self.metadata,
        )

    def rollback_all_appends(self) -> bool:
        """Rollback all events appended through this still-locked transaction."""
        if not self._append_journal:
            return True
        if not _rollback_append_journal(self.custody_fd, self._append_journal):
            return False
        del self.chain.events[self._initial_event_count :]
        self.chain._prev_hash = (
            self.chain.events[-1]["hash"] if self.chain.events else None
        )
        self._append_journal.clear()
        return True

    def mark_rollback_savepoint(self) -> None:
        """Retain prior journal entries when a later commit append must roll back.

        Redetect calls this immediately after PREPARED is durable. A failure
        after that point may remove only the exact later COMMITTED append,
        leaving PREPARED as recovery evidence.
        """
        if self.outcome != _CUSTODY_OUTCOME_SAFE_UNCOMMITTED:
            raise RuntimeError("cannot move a finalized custody savepoint")
        self._rollback_journal_index = len(self._append_journal)
        self._rollback_event_count = len(self.chain.events)

    def _savepoint_expected_size(self) -> int:
        if self._rollback_journal_index <= 0:
            return self._initial_size
        offset, encoded = self._append_journal[self._rollback_journal_index - 1]
        return offset + len(encoded)

    def rollback_to_savepoint(self) -> bool:
        """Rollback the exact journal suffix after the current savepoint."""
        suffix = self._append_journal[self._rollback_journal_index :]
        expected_size = self._savepoint_expected_size()
        current_size = os.fstat(self.custody_fd).st_size
        if not suffix:
            # Unjournaled durable bytes after the savepoint are not a no-op.
            return current_size == expected_size
        if not _rollback_append_journal(self.custody_fd, suffix):
            return False
        if os.fstat(self.custody_fd).st_size != expected_size:
            return False
        del self._append_journal[self._rollback_journal_index :]
        del self.chain.events[self._rollback_event_count :]
        self.chain._prev_hash = (
            self.chain.events[-1]["hash"] if self.chain.events else None
        )
        return True

    def append_event(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if len(self.chain.events) >= self.max_events:
            raise ValueError("custody append would exceed the configured event limit")
        self.assert_authority()
        journal_index = len(self._append_journal)
        event = _append_event_to_locked_fd(
            custody_fd=self.custody_fd,
            custody_dir_fd=self.custody_dir_fd,
            filename=self.filename,
            document_id=self.chain.document_id,
            prev_hash=self.chain._prev_hash,
            event_type=event_type,
            data=data,
            max_bytes=self.max_bytes,
            append_journal=self._append_journal,
        )
        try:
            self.assert_authority()
        except OSError as exc:
            attempt_journal = self._append_journal[journal_index:]
            if not _rollback_append_journal(self.custody_fd, attempt_journal):
                raise _CustodyAuthorityError(
                    "custody append rollback refused an unproven tail"
                ) from exc
            del self._append_journal[journal_index:]
            raise
        self.chain.events.append(event)
        self.chain._prev_hash = event["hash"]
        return event


@contextlib.contextmanager
def open_bound_chain_transaction(
    *,
    custody_dir_fd: int,
    document_id: str,
    expected_device: int,
    expected_inode: int,
    expected_size: int,
    expected_chain_head: str,
    max_bytes: int = 16 * 1024 * 1024,
    max_events: int = MAX_CUSTODY_EVENTS,
    final_authority_check: Callable[[], None] | None = None,
):
    """Lock and verify the exact selected custody inode for multiple appends."""
    canonical_id = validate_custody_document_id(document_id)
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if (
        not isinstance(max_events, int)
        or isinstance(max_events, bool)
        or max_events <= 0
    ):
        raise ValueError("max_events must be a positive integer")
    directory_metadata = os.fstat(custody_dir_fd)
    if not stat.S_ISDIR(directory_metadata.st_mode):
        raise OSError("custody_dir_fd must reference a directory")
    filename = f"{canonical_id}.custody.jsonl"
    transaction_state = _CustodyTransactionState(preserve_original_failure=True)
    with _open_locked_chain_at(
        custody_dir_fd,
        filename,
        create=False,
        writable=True,
        final_authority_check=final_authority_check,
        transaction_state=transaction_state,
    ) as (custody_fd, metadata, append_journal):
        if (metadata.st_dev, metadata.st_ino) != (
            expected_device,
            expected_inode,
        ) or metadata.st_size != expected_size:
            raise _CustodyAuthorityError(
                "canonical custody inode changed after selection"
            )
        raw = _read_locked_chain_bytes(custody_fd, max_bytes=max_bytes)
        if len(raw) != expected_size:
            raise _CustodyAuthorityError(
                "canonical custody bytes changed after selection"
            )
        chain = CustodyChain.load_from_bytes(
            raw,
            max_bytes=max_bytes,
            max_events=max_events,
        )
        valid, message = chain.verify_chain()
        if not valid:
            raise _CustodyAuthorityError(
                f"canonical custody chain is invalid: {message}"
            )
        if chain.document_id != canonical_id:
            raise _CustodyAuthorityError("canonical custody document id changed")
        if str(chain._prev_hash or "") != expected_chain_head:
            raise _CustodyAuthorityError(
                "canonical custody chain tip changed after selection"
            )
        transaction = _BoundCustodyTransaction(
            custody_fd=custody_fd,
            custody_dir_fd=custody_dir_fd,
            filename=filename,
            metadata=metadata,
            chain=chain,
            max_bytes=max_bytes,
            max_events=max_events,
            append_journal=append_journal,
            transaction_state=transaction_state,
        )
        try:
            yield transaction
        except BaseException:
            raise
        else:
            # Let the enclosing exact-inode context perform any savepoint-
            # aware rollback so the transaction outcome is classified once.
            transaction.assert_authority()


def append_event_to_bound_chain(
    *,
    custody_dir_fd: int,
    document_id: str,
    expected_device: int,
    expected_inode: int,
    expected_size: int,
    expected_chain_head: str,
    event_type: str,
    data: dict[str, Any] | None = None,
    max_bytes: int = 16 * 1024 * 1024,
    max_events: int = MAX_CUSTODY_EVENTS,
) -> dict[str, Any]:
    """Append once through the shared exact-inode custody transaction."""
    _validate_event_type(event_type)
    _validate_event_data(data, allow_none=True)
    with open_bound_chain_transaction(
        custody_dir_fd=custody_dir_fd,
        document_id=document_id,
        expected_device=expected_device,
        expected_inode=expected_inode,
        expected_size=expected_size,
        expected_chain_head=expected_chain_head,
        max_bytes=max_bytes,
        max_events=max_events,
    ) as transaction:
        return transaction.append_event(event_type, data)
