"""Job management endpoints."""

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import stat
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Literal, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from api import config
from api.database import Job, get_session_factory
from api.deps import pagination_params, parse_job_submit_form, utc_naive_datetime
from api.identity import require_role
from api.job_manager import JobManager, get_manager
from api.limits import get_default_rate, get_submit_rate, limiter
from api.models import (
    ErrorResponse,
    JobLinks,
    JobListResponse,
    JobProgress,
    JobResultResponse,
    JobStatusResponse,
    JobSubmitRequest,
    JobSubmitResponse,
)
from api.output_manifest import OutputArtifact, build_manifest_from_root_fd
from api.path_safety import (
    _assert_directory_fd_matches_path,
    _bind_job_directory_under_root,
    _BoundJobDirectory,
    _directory_fd_alias,
    _open_directory_under_root_fd,
    _open_or_create_directory_under_root_fd,
    _open_regular_file_under_root,
    _open_regular_file_under_root_fd,
    _read_open_regular_file_bounded,
    read_regular_file_bounded_under_root_fd,
    resolve_job_output_path,
    secure_file_response_under_root_fd,
)
from api.quota import QuotaExceededError

logger = logging.getLogger(__name__)

_JOB_ID_RE = re.compile(r"^job_[0-9a-f]{12}$")
_REDETECT_MAX_BYTES = max(int(config.MAX_UPLOAD_SIZE_MB), 1) * 1024 * 1024
_REDETECT_MAX_CUSTODY_BYTES = 16 * 1024 * 1024
_REDETECT_MAX_CUSTODY_EVENTS = 100_000
_REDETECT_MAX_CUSTODY_DIRECTORY_ENTRIES = 1024
_REDETECT_MAX_CUSTODY_CANDIDATES = 128
_REDETECT_MAX_CUSTODY_AGGREGATE_BYTES = 64 * 1024 * 1024
_REDETECT_MAX_SIDECAR_BYTES = 16 * 1024 * 1024
_REDETECT_STAGE_NAME = "sidecar.stage"
_REDETECT_PRIOR_NAME = "prior.language.json"
_REDETECT_UNCOMMITTED_NAME = "uncommitted.language.json"
_REDETECT_REJECTED_RESTORE_NAME = "rejected-restore.language.json"
_RESULT_OUTPUT_TYPES = {
    "pdf": "searchable_pdf",
    "text": "ocr_text",
    "structure": "structure",
}

# Extended filter (D6) -- recognised values; unknown values are rejected.
_VALID_STATUS_FILTERS = {
    "submitted",
    "queued",
    "pending",
    "processing",
    "running",
    "completed",
    "failed",
    "cancelled",
}
_VALID_SORTS = {
    "submitted_at_desc",
    "submitted_at_asc",
    "duration_desc",
    "status",
}

_MAX_STATUS_FILTERS = 16
_SSE_STREAM_TIMEOUT_DEFAULT = 1800.0
_SSE_STREAM_TIMEOUT_MAX = 24 * 60 * 60.0
_SSE_POLL_INTERVAL_DEFAULT = 30.0
_SSE_POLL_INTERVAL_MAX = 300.0


def _bounded_sse_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """Parse an SSE timing setting without allowing NaN/infinite values."""
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value) or value < minimum or value > maximum:
        return default
    return value


router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def _result_artifact_from_manifest(
    manifest,
    result_type: str,
) -> OutputArtifact | None:
    output_type = _RESULT_OUTPUT_TYPES.get(result_type)
    if output_type is None:
        return None
    return next(
        (
            artifact
            for artifact in manifest.artifacts
            if artifact.output_type == output_type
        ),
        None,
    )


def _get_manager() -> JobManager:
    return get_manager()


def _request_tenant_id(request: Request) -> Optional[str]:
    """Return tenant scope for authenticated tenant keys."""
    return getattr(request.state, "tenant_id", None)


def _job_tenant_filter(request: Request) -> Optional[str]:
    """Tenant filter for job get/mutate: fail-closed under multitenancy."""
    from api.authz import require_tenant_scope_or_platform

    return require_tenant_scope_or_platform(request)


def _is_platform_admin(request: Request) -> bool:
    """True when server-attached tenant_permissions includes platform_admin.

    Trusts only ``request.state.tenant_permissions`` (API key path), never free-form
    JWT ``identity.claims["permissions"]`` (R4-RC-11 / RC-04 amp).
    """
    from api.authz import is_platform_admin

    return is_platform_admin(request)


def _job_status_response(job) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        priority=job.priority,
        source_file=job.source_file,
        progress=JobProgress(
            total_pages=job.total_pages or 0,
            pages_completed=job.pages_completed or 0,
            percent_complete=job.percent_complete(),
            current_stage=job.current_stage or "submitted",
        ),
        settings=job.settings,
        webhook_status=job.webhook_status,
        # The authenticated, tenant-scoped status contract must carry a
        # terminal reason so orchestrators can distinguish a provider restart
        # from an actual document failure. Never leak stale errors on active or
        # successful jobs.
        error_message=job.error_message if job.status == "failed" else None,
    )


def _safe_redetect_artifact_path(job, pdf_path: str) -> str:
    """Return a regular PDF path confined to the job's output root.

    Re-detection runs in a background thread and writes a derived sidecar, so
    it must apply the same path-jail and inode checks as a direct artifact
    download before handing the path to the worker.
    """
    try:
        output_root = resolve_job_output_path(
            job.job_id,
            job.result_path,
            config.OUTPUT_FOLDER,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "output_unavailable",
                "message": "Job output path is not valid.",
            },
        ) from exc
    try:
        relative = Path(pdf_path).relative_to(output_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "artifact_not_found",
                "message": "Re-detect PDF is not under the job output root.",
            },
        ) from exc
    try:
        file_fd, _metadata = _open_regular_file_under_root(
            output_root, relative.as_posix()
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "artifact_not_found",
                "message": "Re-detect PDF is not a safe regular file.",
            },
        ) from exc
    os.close(file_fd)
    return str(output_root / relative)


def _canonical_redetect_custody_document_id(
    *,
    job,
    custody_fd: int,
):
    """Select one verified, job-bound custody chain for REST re-detection.

    Standalone pipeline custody ids are resume ids, not ``source_hash[:16]``.
    Selection therefore inspects the existing chain under the already-bound
    job root. An exact job-id chain wins for older API outputs; otherwise a
    unique source-hash-attested chain wins, then a sole verified root chain.
    Ambiguity or corruption fails before a background mutation is queued.
    """
    from api.custody_authority import (
        CustodyAuthorityError,
        CustodyBytesCandidate,
        select_job_custody_chain,
    )

    candidates: list[CustodyBytesCandidate] = []

    names: list[str] = []
    try:
        with os.scandir(custody_fd) as iterator:
            for entry in iterator:
                if len(names) >= _REDETECT_MAX_CUSTODY_DIRECTORY_ENTRIES:
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "error": "custody_chain_bounds_exceeded",
                            "message": "Custody directory exceeds the scan bound.",
                        },
                    )
                names.append(entry.name)
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "custody_chain_invalid",
                "message": "Custody directory cannot be inspected safely.",
            },
        ) from exc

    aggregate_bytes = 0
    custody_count = 0
    for name in sorted(names):
        if Path(name).name != name or not name.endswith(".custody.jsonl"):
            continue
        custody_count += 1
        if custody_count > _REDETECT_MAX_CUSTODY_CANDIDATES:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "custody_chain_bounds_exceeded",
                    "message": "Custody candidate count exceeds the configured bound.",
                },
            )
        file_fd = -1
        try:
            file_fd, metadata = _open_regular_file_under_root_fd(custody_fd, name)
            raw = _read_open_regular_file_bounded(
                file_fd,
                metadata,
                max_bytes=_REDETECT_MAX_CUSTODY_BYTES,
            )
            os.close(file_fd)
            file_fd = -1
            aggregate_bytes += len(raw)
            if aggregate_bytes > _REDETECT_MAX_CUSTODY_AGGREGATE_BYTES:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "custody_chain_bounds_exceeded",
                        "message": (
                            "Custody candidates exceed the aggregate byte bound."
                        ),
                    },
                )
        except HTTPException:
            if file_fd >= 0:
                os.close(file_fd)
            raise
        except (OSError, TypeError, ValueError) as exc:
            if file_fd >= 0:
                os.close(file_fd)
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "custody_chain_invalid",
                    "message": "A custody candidate cannot be read safely.",
                },
            ) from exc
        candidates.append(
            CustodyBytesCandidate(
                relative_path=name,
                raw=raw,
                device=metadata.st_dev,
                inode=metadata.st_ino,
                size_bytes=metadata.st_size,
                nlink=metadata.st_nlink,
            )
        )

    try:
        selected = select_job_custody_chain(
            job_id=str(job.job_id),
            source_hash=str(getattr(job, "source_hash", "") or ""),
            candidates=candidates,
            max_bytes=_REDETECT_MAX_CUSTODY_BYTES,
            max_events=_REDETECT_MAX_CUSTODY_EVENTS,
            allow_legacy_job_without_hash=False,
            max_candidates=_REDETECT_MAX_CUSTODY_CANDIDATES,
            max_total_bytes=_REDETECT_MAX_CUSTODY_AGGREGATE_BYTES,
            max_total_events=_REDETECT_MAX_CUSTODY_EVENTS,
        )
    except CustodyAuthorityError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": exc.code,
                "message": str(exc),
            },
        ) from exc
    return selected


def _write_redetect_stage(operation_fd: int, payload: bytes) -> os.stat_result:
    """Durably stage canonical sidecar bytes below one hidden operation dir."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("redetect sidecar payload must be non-empty bytes")
    if len(payload) > _REDETECT_MAX_SIDECAR_BYTES:
        raise ValueError("redetect sidecar exceeds the configured byte limit")
    stage_fd = -1
    try:
        stage_fd = os.open(
            _REDETECT_STAGE_NAME,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=operation_fd,
        )
        view = memoryview(payload)
        while view:
            written = os.write(stage_fd, view)
            if written <= 0:
                raise OSError("redetect sidecar staging made no progress")
            view = view[written:]
        os.fsync(stage_fd)
        metadata = os.fstat(stage_fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
        ):
            raise OSError("redetect sidecar staging inode is not safe")
        os.fsync(operation_fd)
        return metadata
    except Exception:
        try:
            os.unlink(_REDETECT_STAGE_NAME, dir_fd=operation_fd)
        except OSError:
            pass
        raise
    finally:
        if stage_fd >= 0:
            os.close(stage_fd)


def _read_optional_redetect_target(
    language_fd: int,
    target_name: str,
) -> tuple[os.stat_result, bytes] | None:
    """Snapshot the exact pre-existing sidecar inode, or return ``None``."""
    target_fd = -1
    try:
        target_fd, metadata = _open_regular_file_under_root_fd(
            language_fd,
            target_name,
        )
    except FileNotFoundError:
        return None
    try:
        payload = _read_open_regular_file_bounded(
            target_fd,
            metadata,
            max_bytes=_REDETECT_MAX_SIDECAR_BYTES,
        )
        return metadata, payload
    finally:
        if target_fd >= 0:
            os.close(target_fd)


def _assert_redetect_name_inode(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
) -> None:
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or current.st_nlink != 1
        or (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino)
    ):
        raise OSError("redetect sidecar namespace changed")


def _assert_redetect_name_snapshot(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
) -> None:
    """Require the published name and mutation-sensitive metadata to be stable."""
    _assert_redetect_name_inode(directory_fd, name, expected)
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        current.st_size != expected.st_size
        or current.st_mtime_ns != expected.st_mtime_ns
        or current.st_ctime_ns != expected.st_ctime_ns
    ):
        raise OSError("published redetect sidecar changed")


def _snapshot_redetect_name_exact(
    directory_fd: int,
    name: str,
    *,
    expected: os.stat_result,
    expected_payload: bytes,
    allowed_link_counts: set[int],
) -> os.stat_result:
    """Verify exact bytes for a staged/published inode, including two-link state."""
    file_fd = -1
    try:
        before_name = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        file_fd = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before_name.st_mode)
            or (before.st_dev, before.st_ino) != (expected.st_dev, expected.st_ino)
            or (before_name.st_dev, before_name.st_ino)
            != (before.st_dev, before.st_ino)
            or before.st_nlink not in allowed_link_counts
            or before.st_size != len(expected_payload)
        ):
            raise OSError("redetect sidecar namespace changed")
        chunks: list[bytes] = []
        offset = 0
        remaining = before.st_size
        while remaining:
            chunk = os.pread(file_fd, min(remaining, 64 * 1024), offset)
            if not chunk:
                raise OSError("redetect sidecar changed during verification")
            chunks.append(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        after = os.fstat(file_fd)
        after_name = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)

        def signature(item: os.stat_result) -> tuple[int, ...]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
                item.st_nlink,
            )

        if (
            b"".join(chunks) != expected_payload
            or signature(after) != signature(before)
            or signature(after_name) != signature(after)
        ):
            raise OSError("redetect sidecar bytes changed")
        return after
    finally:
        if file_fd >= 0:
            os.close(file_fd)


def _restore_redetect_publish(
    *,
    language_fd: int,
    operation_fd: int,
    target_name: str,
    staged_metadata: os.stat_result,
    staged_payload: bytes,
    publish_attempted: bool,
    published: bool,
    prior_moved: bool,
    prior_snapshot: tuple[os.stat_result, bytes] | None,
) -> None:
    """Restore the exact prior target after a caught publish/commit failure."""
    target_present = False
    if publish_attempted:
        try:
            os.stat(
                target_name,
                dir_fd=language_fd,
                follow_symlinks=False,
            )
            target_present = True
        except FileNotFoundError:
            pass
    if target_present:
        if not published:
            # ``link`` itself may have completed before raising. Only undo a
            # target that is still provably the exact staged inode and bytes.
            published = True
        try:
            os.stat(
                _REDETECT_UNCOMMITTED_NAME,
                dir_fd=operation_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise OSError("redetect rollback quarantine already exists")
        # Move whichever inode currently owns the canonical name. A source
        # swap at this exact operation is preserved in quarantine, never
        # irreversibly unlinked.
        os.rename(
            target_name,
            _REDETECT_UNCOMMITTED_NAME,
            src_dir_fd=language_fd,
            dst_dir_fd=operation_fd,
        )
        os.fsync(language_fd)
        os.fsync(operation_fd)
        quarantined = os.stat(
            _REDETECT_UNCOMMITTED_NAME,
            dir_fd=operation_fd,
            follow_symlinks=False,
        )
        try:
            _snapshot_redetect_name_exact(
                operation_fd,
                _REDETECT_UNCOMMITTED_NAME,
                expected=staged_metadata,
                expected_payload=staged_payload,
                allowed_link_counts={1, 2},
            )
            if quarantined.st_nlink == 2:
                stage_metadata = _snapshot_redetect_name_exact(
                    operation_fd,
                    _REDETECT_STAGE_NAME,
                    expected=staged_metadata,
                    expected_payload=staged_payload,
                    allowed_link_counts={2},
                )
                if (stage_metadata.st_dev, stage_metadata.st_ino) != (
                    quarantined.st_dev,
                    quarantined.st_ino,
                ):
                    raise OSError("redetect stage authorities diverged")
            elif quarantined.st_nlink != 1:
                raise OSError("redetect published inode has an unsafe link count")
        except Exception:
            # A concurrent replacement was moved, not deleted. Restore that
            # exact inode under the canonical name without clobbering any
            # newer writer, and retain quarantine evidence for review.
            try:
                os.link(
                    _REDETECT_UNCOMMITTED_NAME,
                    target_name,
                    src_dir_fd=operation_fd,
                    dst_dir_fd=language_fd,
                    follow_symlinks=False,
                )
                restored = os.stat(
                    target_name,
                    dir_fd=language_fd,
                    follow_symlinks=False,
                )
                if (restored.st_dev, restored.st_ino) != (
                    quarantined.st_dev,
                    quarantined.st_ino,
                ):
                    raise OSError("redetect concurrent target restore diverged")
                os.fsync(language_fd)
            except FileExistsError:
                pass
            raise
    elif published:
        raise OSError("redetect published target disappeared before restore")
    if prior_moved:
        if prior_snapshot is None:
            raise OSError("redetect prior snapshot is unavailable")
        verified_prior = _read_optional_redetect_target(
            operation_fd,
            _REDETECT_PRIOR_NAME,
        )
        if (
            verified_prior is None
            or (verified_prior[0].st_dev, verified_prior[0].st_ino)
            != (prior_snapshot[0].st_dev, prior_snapshot[0].st_ino)
            or verified_prior[1] != prior_snapshot[1]
        ):
            raise OSError("redetect prior target changed before restore")
        try:
            os.stat(target_name, dir_fd=language_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise OSError(
                "redetect prior target cannot be restored without clobbering "
                "a concurrent file"
            )
        prior_metadata = os.stat(
            _REDETECT_PRIOR_NAME,
            dir_fd=operation_fd,
            follow_symlinks=False,
        )
        os.link(
            _REDETECT_PRIOR_NAME,
            target_name,
            src_dir_fd=operation_fd,
            dst_dir_fd=language_fd,
            follow_symlinks=False,
        )
        try:
            current = _snapshot_redetect_name_exact(
                language_fd,
                target_name,
                expected=prior_metadata,
                expected_payload=prior_snapshot[1],
                allowed_link_counts={2},
            )
            _snapshot_redetect_name_exact(
                operation_fd,
                _REDETECT_PRIOR_NAME,
                expected=prior_metadata,
                expected_payload=prior_snapshot[1],
                allowed_link_counts={2},
            )
        except Exception:
            try:
                os.stat(
                    _REDETECT_REJECTED_RESTORE_NAME,
                    dir_fd=operation_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise OSError("redetect rejected-restore quarantine exists")
            os.rename(
                target_name,
                _REDETECT_REJECTED_RESTORE_NAME,
                src_dir_fd=language_fd,
                dst_dir_fd=operation_fd,
            )
            moved = os.stat(
                _REDETECT_REJECTED_RESTORE_NAME,
                dir_fd=operation_fd,
                follow_symlinks=False,
            )
            if (moved.st_dev, moved.st_ino) != (
                prior_metadata.st_dev,
                prior_metadata.st_ino,
            ):
                try:
                    os.link(
                        _REDETECT_REJECTED_RESTORE_NAME,
                        target_name,
                        src_dir_fd=operation_fd,
                        dst_dir_fd=language_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    pass
            os.fsync(language_fd)
            os.fsync(operation_fd)
            raise
        _snapshot_redetect_name_exact(
            language_fd,
            target_name,
            expected=current,
            expected_payload=prior_snapshot[1],
            allowed_link_counts={2},
        )
    os.fsync(language_fd)
    os.fsync(operation_fd)


def _cleanup_redetect_operation(
    *,
    staging_root_fd: int,
    operation_fd: int,
    operation_id: str,
) -> None:
    """Best-effort cleanup of the fixed, hidden operation artifacts."""
    for name in (
        _REDETECT_STAGE_NAME,
        _REDETECT_PRIOR_NAME,
        _REDETECT_UNCOMMITTED_NAME,
        _REDETECT_REJECTED_RESTORE_NAME,
    ):
        try:
            os.unlink(name, dir_fd=operation_fd)
        except FileNotFoundError:
            pass
    os.fsync(operation_fd)
    try:
        os.rmdir(operation_id, dir_fd=staging_root_fd)
    except FileNotFoundError:
        pass
    os.fsync(staging_root_fd)


# ------------------------------------------------------------------
# POST /api/v1/jobs --- Submit job
# ------------------------------------------------------------------


# Submit `detail.error` codes (FE keys rendering on these, not on messages):
#   file_too_large (413), queue_full (429), page_budget_exceeded,
#   page_estimate_failed, pixel_budget_exceeded, pixel_budget_probe_failed,
#   invalid_staging_path, invalid_filename, conflicting_upload,
#   executable_refused, invalid_upload_stream, invalid_source_path,
#   unreadable_source, unhashable_source, missing_source, tenant_inactive,
#   unsupported_option, invalid_request (fallback for unclassified errors;
#   message intentionally generic to avoid leaking internal paths).
_SUBMIT_VALUE_ERROR_RULES: tuple[tuple[str, str, int, bool], ...] = (
    # (message needle, error code, status, echo str(exc))
    ("upload exceeds", "file_too_large", 413, True),
    ("capacity", "queue_full", 429, True),
    ("exceeds max_pdf_pages", "page_budget_exceeded", 400, True),
    ("unable to estimate pdf page count", "page_estimate_failed", 400, True),
    ("image pixel budget probe failed", "pixel_budget_probe_failed", 400, False),
    ("image pixel budget exceeded", "pixel_budget_exceeded", 400, True),
    ("pixel budget exceeded", "pixel_budget_exceeded", 400, True),
    ("staging path is not canonical", "invalid_staging_path", 400, True),
    ("invalid upload filename", "invalid_filename", 400, True),
    ("only one upload content source", "conflicting_upload", 400, True),
    ("executable binary upload refused", "executable_refused", 400, True),
    ("upload stream", "invalid_upload_stream", 400, True),
    ("regular file", "invalid_source_path", 400, True),
    ("multiple hard links", "invalid_source_path", 400, True),
    ("source file could not be read", "unreadable_source", 400, True),
    ("source file could not be hashed", "unhashable_source", 400, True),
    ("either source_path or file upload is required", "missing_source", 400, True),
    ("tenant not found or inactive", "tenant_inactive", 400, True),
    ("skip_ocr is not supported", "unsupported_option", 400, True),
)


def _classify_submit_value_error(exc: ValueError) -> tuple[int, str, str]:
    """Map a manager.submit ValueError to (status, error code, message)."""
    text = str(exc).lower()
    for needle, code, status, echo in _SUBMIT_VALUE_ERROR_RULES:
        if needle in text:
            if echo:
                return status, code, str(exc)
            return (
                status,
                code,
                "Image pixel budget probe failed; refusing admission.",
            )
    return (
        400,
        "invalid_request",
        "The job submission could not be accepted.",
    )


@router.post(
    "",
    name="submit_job",
    response_model=JobSubmitResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        501: {"model": ErrorResponse},
    },
)
@limiter.limit(get_submit_rate())
async def submit_job(
    request: Request,
    form_data: JobSubmitRequest = Depends(parse_job_submit_form),
    file: Optional[UploadFile] = File(None),
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Submit a document for OCR processing."""
    manager = _get_manager()
    tenant_id = _job_tenant_filter(request)

    if form_data.source_path:
        from api.authz import require_platform_admin_for_shared_paths

        require_platform_admin_for_shared_paths(request)

    if form_data.skip_ocr:
        raise HTTPException(
            status_code=501,
            detail={
                "error": "unsupported_feature",
                "message": (
                    "skip_ocr is not supported by the standalone API pipeline; "
                    "use the coordinator text-only path"
                ),
            },
        )

    # Check queue capacity
    if not manager.check_queue_capacity():
        raise HTTPException(
            status_code=429,
            detail={
                "error": "queue_full",
                "message": "Job queue is at capacity. Please try again later.",
            },
        )

    if not file and not form_data.source_path:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request",
                "message": "Provide file upload or source_path.",
            },
        )

    try:
        upload_filename = None
        upload_file = None
        if file:
            upload_filename = file.filename
            upload_file = file.file

        # S2-04: urgent is gated — platform_admin, priority_urgent perm, or
        # OCR_ALLOW_CLIENT_URGENT lab waiver. Others are demoted to normal.
        import os

        from api.authz import is_platform_admin
        from api.job_manager import normalize_job_priority

        perms = set(getattr(request.state, "tenant_permissions", None) or [])
        allow_urgent = (
            is_platform_admin(request)
            or "priority_urgent" in perms
            or os.environ.get("OCR_ALLOW_CLIENT_URGENT", "").lower()
            in ("1", "true", "yes")
        )
        requested = (form_data.priority or "normal").strip().lower()
        if requested == "urgent" and not allow_urgent:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "priority_forbidden",
                    "message": (
                        "priority=urgent requires platform_admin, "
                        "priority_urgent permission, or OCR_ALLOW_CLIENT_URGENT "
                        "(S2-04 QoS gate)"
                    ),
                },
            )
        effective_priority = normalize_job_priority(
            form_data.priority, allow_urgent=allow_urgent
        )

        job = manager.submit(
            source_path=form_data.source_path,
            upload_filename=upload_filename,
            upload_file=upload_file,
            tenant_id=tenant_id,
            priority=effective_priority,
            enable_docintel=form_data.enable_docintel,
            docintel_mode=form_data.docintel_mode,
            skip_ocr=form_data.skip_ocr,
            processing_timeout_minutes=form_data.processing_timeout_minutes,
            webhook_url=form_data.webhook_url,
            webhook_secret=form_data.webhook_secret,
        )
    except QuotaExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "quota_exceeded",
                "message": str(exc),
                "details": {
                    "tenant_id": exc.tenant_id,
                    "limit_type": exc.limit_type,
                    "current": exc.current,
                    "maximum": exc.maximum,
                },
            },
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "message": "Source file not found."},
        )
    except ValueError as exc:
        status, code, message = _classify_submit_value_error(exc)
        raise HTTPException(
            status_code=status,
            detail={"error": code, "message": message},
        )

    return JobSubmitResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        priority=job.priority,
        source_file=job.source_file,
        estimated_pages=job.total_pages,
        links=JobLinks(
            **{
                "self": str(request.url_for("get_job_status", job_id=job.job_id)),
                "result": str(request.url_for("get_job_result", job_id=job.job_id)),
            }
        ),
    )


# ------------------------------------------------------------------
# GET /api/v1/jobs --- List jobs
# ------------------------------------------------------------------


@router.get("", name="list_jobs", response_model=JobListResponse)
@limiter.limit(get_default_rate())
async def list_jobs(
    request: Request,
    status: Optional[list[str]] = Query(
        default=None,
        max_length=_MAX_STATUS_FILTERS,
        description=(
            "Filter by one or more statuses (repeatable: ?status=running&"
            "status=completed). A single value also works (?status=failed)."
        ),
    ),
    batch_id: Optional[str] = Query(
        default=None, max_length=64, description="Filter by batch ID"
    ),
    tenant_id: Optional[str] = Query(
        default=None,
        max_length=64,
        description="Filter by tenant ID (platform-admin only; non-admin callers are always scoped to their own tenant).",
    ),
    submitted_after: Optional[datetime] = Query(
        default=None,
        description="Created at or after this ISO 8601 timestamp; timezone-free values mean UTC.",
    ),
    submitted_before: Optional[datetime] = Query(
        default=None,
        description="Created at or before this ISO 8601 timestamp; timezone-free values mean UTC.",
    ),
    q: Optional[str] = Query(
        default=None,
        max_length=256,
        description="Substring search on source_filename and job_id (case-insensitive).",
    ),
    sort: Literal[
        "submitted_at_desc",
        "submitted_at_asc",
        "duration_desc",
        "status",
    ] = Query(
        default="submitted_at_desc",
        description="Sort order applied after filtering.",
    ),
    pagination: tuple[int, int] = Depends(pagination_params),
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """List jobs with optional filters, sorting, and pagination.

    Supports ``limit``/``offset`` query parameters (canonical) as well as
    the deprecated ``page``/``per_page`` pair for backward compatibility.

    Filter parameters (D6):

    - ``status``: single status (legacy) OR repeated multi-status filter.
      Both ``?status=failed`` and ``?status=running&status=completed`` are
      accepted; the repeated form takes precedence.
    - ``tenant_id``: only honoured for platform-admin callers; tenant-key
      callers are pinned to their own scope and a foreign ``?tenant_id=``
      is rejected with 403 ``tenant_mismatch`` (dashboard parity).
    - ``submitted_after`` / ``submitted_before``: ISO 8601 datetimes;
      offsets are converted to UTC and timezone-free values mean UTC.
    - ``q``: case-insensitive substring match on ``source_file`` and
      ``job_id`` with ``%``/``_`` escaped.
    - ``sort``: one of ``submitted_at_desc`` (default), ``submitted_at_asc``,
      ``duration_desc``, ``status``.
    """
    limit, offset = pagination
    manager = _get_manager()
    caller_tenant_id = _job_tenant_filter(request)

    # Normalise multi-status filter.  FastAPI maps repeated ?status=foo&
    # status=bar into a list; a single ?status=foo also arrives as a one-
    # element list because the parameter type is ``Optional[list[str]]``.
    if status is None:
        status_list: Optional[list[str]] = None
    else:
        if any(len(value) > 32 for value in status):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "status_filter_too_long",
                    "message": "Each status filter must be at most 32 characters.",
                },
            )
        cleaned = list(dict.fromkeys(s.strip() for s in status if s and s.strip()))
        if len(cleaned) > _MAX_STATUS_FILTERS:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "too_many_status_filters",
                    "message": (
                        "At most "
                        f"{_MAX_STATUS_FILTERS} distinct status filters are allowed."
                    ),
                },
            )
        status_list = cleaned or None

    if status_list:
        for value in status_list:
            if value not in _VALID_STATUS_FILTERS:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "invalid_status",
                        "message": f"Unknown status filter value: {value!r}",
                    },
                )

    if sort not in _VALID_SORTS:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_sort",
                "message": f"Unknown sort value: {sort!r}",
            },
        )

    try:
        submitted_after = utc_naive_datetime(submitted_after)
        submitted_before = utc_naive_datetime(submitted_before)
    except (ValueError, OverflowError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_date_range", "message": "Date bounds must be representable in UTC."},
        ) from exc

    if (
        submitted_after is not None
        and submitted_before is not None
        and submitted_after > submitted_before
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_date_range",
                "message": "submitted_after must be <= submitted_before",
            },
        )

    # Tenant isolation (gotcha #80 / RC-01): platform-admin callers may target a
    # specific tenant via ?tenant_id=; everyone else is forced to their
    # caller-tenant scope.  Under multitenancy, missing tenant binding is
    # fail-closed (not fleet-global list).
    from api.authz import require_tenant_scope_or_platform

    scoped = require_tenant_scope_or_platform(request)
    if scoped is not None:
        # Tenant key: pin to caller scope; a foreign ?tenant_id= is an
        # explicit mismatch (dashboard parity: 403, never silent rewrite).
        if tenant_id is not None and tenant_id != scoped:
            raise HTTPException(
                status_code=403,
                detail={"error": "tenant_mismatch", "message": "Tenant mismatch"},
            )
        tenant_id = scoped
    elif _is_platform_admin(request):
        # Platform: optional ?tenant_id= filter; None = global
        pass
    else:
        # Single-tenant mode (multitenancy off): drop untrusted filters.
        if tenant_id is not None and not _is_platform_admin(request):
            tenant_id = None
        if caller_tenant_id is not None:
            tenant_id = caller_tenant_id

    jobs, total = manager.list_jobs(
        batch_id=batch_id,
        tenant_id=tenant_id,
        limit=limit,
        offset=offset,
        status_in=status_list,
        submitted_after=submitted_after,
        submitted_before=submitted_before,
        q=q,
        sort=sort,
    )
    return JobListResponse(
        jobs=[_job_status_response(j) for j in jobs],
        total=total,
        limit=limit,
        offset=offset,
    )


# ------------------------------------------------------------------
# GET /api/v1/jobs/{job_id} --- Job status
# ------------------------------------------------------------------


@router.get(
    "/{job_id}",
    name="get_job_status",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def get_job_status(
    request: Request,
    job_id: str,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Get status and progress of a specific job."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_job_id", "message": "Invalid job ID format."},
        )
    manager = _get_manager()
    job = manager.get_job(job_id, tenant_id=_job_tenant_filter(request))
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
        )
    return _job_status_response(job)


# ------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/result --- Result metadata
# ------------------------------------------------------------------


@router.get(
    "/{job_id}/result",
    name="get_job_result",
    response_model=JobResultResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def get_job_result(
    request: Request,
    job_id: str,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Get result metadata and artifact links for a completed job."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_job_id", "message": "Invalid job ID format."},
        )
    manager = _get_manager()
    job = manager.get_job(job_id, tenant_id=_job_tenant_filter(request))
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
        )
    if job.status not in ("completed", "failed"):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "job_not_complete",
                "message": f"Job is {job.status}. Progress: {job.percent_complete()}%",
            },
        )

    artifacts = manager.get_result_artifacts(job_id)
    download_links = {
        k: str(request.url_for("download_artifact", job_id=job_id))
        + "?"
        + urlencode({"type": k})
        for k in artifacts
    }

    return JobResultResponse(
        job_id=job.job_id,
        status=job.status,
        completed_at=job.completed_at,
        processing_time_seconds=job.processing_time,
        artifacts=download_links,
        metadata={"pages_processed": job.pages_completed or 0},
    )


# ------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/result/download --- Download artifact
# ------------------------------------------------------------------


@router.get(
    "/{job_id}/result/download",
    name="download_artifact",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def download_artifact(
    request: Request,
    job_id: str,
    type: str = "pdf",
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Download a specific result artifact (pdf, text, structure)."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_job_id", "message": "Invalid job ID format."},
        )
    manager = _get_manager()
    job = manager.get_job(job_id, tenant_id=_job_tenant_filter(request))
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
        )
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail={"error": "job_not_complete", "message": f"Job is {job.status}."},
        )

    if not job.result_path:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "artifact_not_found",
                "message": f"No '{type}' artifact available.",
            },
        )

    binding: _BoundJobDirectory | None = None
    try:
        try:
            binding = _bind_job_directory_under_root(
                config.OUTPUT_FOLDER,
                job.job_id,
                persisted_result_path=job.result_path,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "output_unavailable",
                    "message": "Job output path is not valid.",
                },
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "artifact_not_found",
                    "message": f"No '{type}' artifact available.",
                },
            ) from exc

        output_root = binding.job_root
        root_fd = binding.job_fd
        manifest = build_manifest_from_root_fd(job_id, str(output_root), root_fd)
        artifact = _result_artifact_from_manifest(manifest, type)
        try:
            binding.assert_unchanged()
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "artifact_not_found",
                    "message": "Job output changed during download preparation.",
                },
            ) from exc
        if artifact is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "artifact_not_found",
                    "message": f"No '{type}' artifact available.",
                },
            )
        relative_artifact = artifact.relative_path

        try:
            bound_root = _directory_fd_alias(root_fd)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "output_unavailable",
                    "message": "Job output cannot be bound safely on this host.",
                },
            ) from exc

        # LT-13: refuse download while HITL deliverable hold marker is present.
        # Resolve diagnostic paths through the bound descriptor alias so a
        # parent-root swap cannot redirect the hold walk to another tenant.
        try:
            from exception_router import assert_deliverable_not_held
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "hold_check_unavailable",
                    "message": (
                        "LT-13 deliverable hold module unavailable; "
                        "refusing download."
                    ),
                },
            ) from exc
        try:
            bound_artifact = bound_root / relative_artifact
            pdf_artifact = _result_artifact_from_manifest(manifest, "pdf")
            bound_pdf = ""
            if pdf_artifact:
                bound_pdf = str(bound_root / pdf_artifact.relative_path)
            assert_deliverable_not_held(bound_pdf, str(bound_artifact))
        except ValueError as exc:
            raise HTTPException(
                status_code=423,
                detail={
                    "error": "deliverable_held",
                    "message": "Deliverable is under a litigation hold.",
                },
            ) from exc
        except OSError:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "hold_check_failed",
                    "message": "LT-13 hold check failed closed; output withheld.",
                },
            )

        media_types = {
            "pdf": "application/pdf",
            "text": "text/plain",
            "structure": "application/json",
        }
        try:
            binding.assert_unchanged()
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "artifact_not_found",
                    "message": "Job output changed during download preparation.",
                },
            ) from exc
        return secure_file_response_under_root_fd(
            root_fd,
            relative_artifact,
            media_type=media_types.get(type, "application/octet-stream"),
            filename=artifact.filename,
            expected_device=artifact.device,
            expected_inode=artifact.inode,
            expected_size=artifact.size_bytes,
            expected_mtime_ns=artifact.mtime_ns,
            expected_ctime_ns=artifact.ctime_ns,
        )
    finally:
        if binding is not None:
            binding.close()


# ------------------------------------------------------------------
# POST /api/v1/jobs/{job_id}/retry --- Retry failed job
# ------------------------------------------------------------------


@router.post(
    "/{job_id}/retry",
    name="retry_job",
    response_model=JobSubmitResponse,
    status_code=201,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
@limiter.limit(get_submit_rate())
async def retry_job(
    request: Request,
    job_id: str,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Retry a failed or cancelled job by re-submitting with the same source."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_job_id", "message": "Invalid job ID format."},
        )
    manager = _get_manager()

    if not manager.check_queue_capacity():
        raise HTTPException(
            status_code=429,
            detail={
                "error": "queue_full",
                "message": "Job queue is at capacity. Please try again later.",
            },
        )

    try:
        new_job = manager.retry_job(job_id, tenant_id=_job_tenant_filter(request))
    except QuotaExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "quota_exceeded",
                "message": str(exc),
                "details": {
                    "tenant_id": exc.tenant_id,
                    "limit_type": exc.limit_type,
                    "current": exc.current,
                    "maximum": exc.maximum,
                },
            },
        )
    except ValueError as exc:
        logger.info("Job retry rejected for job_id=%s: %s", job_id, exc)
        raise HTTPException(
            status_code=409,
            detail={
                "error": "invalid_state",
                "message": "Job is not currently retryable.",
            },
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={"error": "source_missing", "message": "Source file not found."},
        )

    if not new_job:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
        )

    return JobSubmitResponse(
        job_id=new_job.job_id,
        status=new_job.status,
        created_at=new_job.created_at,
        priority=new_job.priority,
        source_file=new_job.source_file,
        estimated_pages=new_job.total_pages,
        links=JobLinks(
            **{
                "self": str(request.url_for("get_job_status", job_id=new_job.job_id)),
                "result": str(request.url_for("get_job_result", job_id=new_job.job_id)),
            }
        ),
    )


# ------------------------------------------------------------------
# POST /api/v1/jobs/{job_id}/redetect-language --- Re-run language detection
# ------------------------------------------------------------------


@router.post(
    "/{job_id}/redetect-language",
    name="redetect_language",
    status_code=202,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def redetect_language(
    request: Request,
    job_id: str,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Re-run language detection on an already-completed job's output PDF.

    Enqueues a background thread that re-runs per-span language detection
    against the embedded text layer of the job's output PDF.  OCR is NOT
    re-executed; this is a cheap, detector-only pass that writes a new
    ``.language.json`` sidecar and emits a ``LANGUAGE_REDETECTED`` custody
    event.

    Responses
    ---------
    - ``202`` -- redetect queued; returns ``status``, ``job_id``, and an
      additive ``operation_id`` correlation identifier.  The queued daemon
      thread has no durable status/outbox or restart reconciliation; ``202``
      is not a completion signal.
    - ``404`` -- job not found (including when tenant scoping hides it)
    - ``409`` -- job is not in ``completed`` state
    """
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_job_id", "message": "Invalid job ID format."},
        )

    manager = _get_manager()
    job = manager.get_job(job_id, tenant_id=_job_tenant_filter(request))
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
        )
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "job_not_complete",
                "message": (
                    "Language re-detection requires a completed job; "
                    f"current status is {job.status}."
                ),
            },
        )

    if not job.result_path:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "pdf_missing",
                "message": "Completed job has no PDF artifact to re-detect.",
            },
        )
    binding: _BoundJobDirectory | None = None
    custody_fd = -1
    language_fd = -1
    owned_fds: list[int] = []

    def _close_owned_fds() -> None:
        while owned_fds:
            fd = owned_fds.pop()
            try:
                os.close(fd)
            except OSError:
                pass
        if binding is not None:
            binding.close()

    try:
        binding = _bind_job_directory_under_root(
            config.OUTPUT_FOLDER,
            job.job_id,
            persisted_result_path=job.result_path,
        )
        output_root = binding.job_root
        root_fd = binding.job_fd
    except (OSError, ValueError) as exc:
        _close_owned_fds()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "output_unavailable",
                "message": "Job output cannot be bound safely on this host.",
            },
        ) from exc

    try:
        manifest = build_manifest_from_root_fd(job_id, str(output_root), root_fd)
        binding.assert_unchanged()
    except (OSError, ValueError) as exc:
        _close_owned_fds()
        raise HTTPException(
            status_code=409,
            detail={
                "error": "output_changed",
                "message": (
                    "Job output changed or exceeded safe scan bounds during "
                    "re-detection preparation."
                ),
            },
        ) from exc

    pdf_artifact = _result_artifact_from_manifest(manifest, "pdf")
    if pdf_artifact is None:
        _close_owned_fds()
        raise HTTPException(
            status_code=409,
            detail={
                "error": "pdf_missing",
                "message": "Completed job has no PDF artifact to re-detect.",
            },
        )

    try:
        relative_pdf = pdf_artifact.relative_path
        # Retain the exact bytes read from the manifest-authorized inode. The
        # queued thread never reopens EXPORT/PDF after this point.
        pdf_bytes = read_regular_file_bounded_under_root_fd(
            root_fd,
            relative_pdf,
            max_bytes=_REDETECT_MAX_BYTES,
            expected_device=pdf_artifact.device,
            expected_inode=pdf_artifact.inode,
            expected_size=pdf_artifact.size_bytes,
            expected_mtime_ns=pdf_artifact.mtime_ns,
            expected_ctime_ns=pdf_artifact.ctime_ns,
        )
        custody_fd = _open_directory_under_root_fd(root_fd, "EXPORT/CUSTODY")
        owned_fds.append(custody_fd)
        custody_selection = _canonical_redetect_custody_document_id(
            job=job,
            custody_fd=custody_fd,
        )
        custody_document_id = custody_selection.document_id
        if (
            custody_selection.device is None
            or custody_selection.inode is None
            or custody_selection.size_bytes is None
        ):
            raise ValueError("canonical custody inode metadata is unavailable")
        if len(custody_selection.chain.events) > _REDETECT_MAX_CUSTODY_EVENTS - 2:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "custody_chain_bounds_exceeded",
                    "message": "Custody chain has no room for the redetect saga.",
                },
            )
        language_fd = _open_or_create_directory_under_root_fd(
            root_fd,
            "EXPORT/LANGUAGE",
        )
        owned_fds.append(language_fd)
        from ocr_distributed.ocr_utils import build_sidecar_base_name

        target_name = (
            f"{build_sidecar_base_name(Path(relative_pdf).name)}.language.json"
        )
        if Path(target_name).name != target_name or not target_name.endswith(
            ".language.json"
        ):
            raise ValueError("redetect target name is not safe")
        operation_id = f"redetect_{uuid.uuid4().hex}"
        source_pdf_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
        _assert_directory_fd_matches_path(
            output_root / "EXPORT" / "CUSTODY",
            custody_fd,
        )
        _assert_directory_fd_matches_path(
            output_root / "EXPORT" / "LANGUAGE",
            language_fd,
        )
        binding.assert_unchanged()
    except HTTPException:
        _close_owned_fds()
        raise
    except (OSError, ValueError) as exc:
        _close_owned_fds()
        raise HTTPException(
            status_code=409,
            detail={
                "error": "output_changed",
                "message": "Job output changed during re-detection preparation.",
            },
        ) from exc

    def _run() -> None:
        staging_root_fd = -1
        operation_fd = -1
        operation_created = False
        staged_metadata: os.stat_result | None = None
        publish_attempted = False
        published = False
        prior_moved = False
        cleanup_allowed = True
        control_flow_error: BaseException | None = None
        control_flow_traceback = None
        prior_snapshot: tuple[os.stat_result, bytes] | None = None
        custody_transaction = None
        try:
            from ocr_local.features.custody import open_bound_chain_transaction
            from ocr_local.features.language_detection import redetect_document

            # The request transfers ownership of ``root_fd`` to this thread.
            # Refuse a queued parent-name replacement before any mutation;
            # reads and writes then remain rooted through the held descriptor.
            binding.assert_unchanged()
            _assert_directory_fd_matches_path(
                output_root / "EXPORT" / "CUSTODY",
                custody_fd,
            )
            _assert_directory_fd_matches_path(
                output_root / "EXPORT" / "LANGUAGE",
                language_fd,
            )
            result = redetect_document(
                pdf_path=str(output_root / relative_pdf),
                output_json_path="",
                output_base_dir="",
                document_id=custody_document_id,
                pdf_bytes=pdf_bytes,
                emit_custody_event=False,
                write_sidecar=False,
            )
            _assert_directory_fd_matches_path(
                output_root / "EXPORT" / "CUSTODY",
                custody_fd,
            )
            _assert_directory_fd_matches_path(
                output_root / "EXPORT" / "LANGUAGE",
                language_fd,
            )
            binding.assert_unchanged()
            if result.get("status") != "ok":
                logger.warning(
                    "Background redetect returned an error for %s: %s",
                    job_id,
                    result.get("error", "unknown error"),
                )
                return
            sidecar_bytes = result.get("sidecar_bytes")
            if not isinstance(sidecar_bytes, bytes) or not sidecar_bytes:
                raise ValueError("redetect did not return canonical sidecar bytes")
            if result.get("sidecar_filename") not in {None, target_name}:
                raise ValueError("redetect returned an unexpected sidecar target")

            staging_root_fd = _open_or_create_directory_under_root_fd(
                root_fd,
                ".redetect",
                mode=0o700,
            )
            _assert_directory_fd_matches_path(
                output_root / ".redetect",
                staging_root_fd,
            )
            os.mkdir(operation_id, mode=0o700, dir_fd=staging_root_fd)
            operation_created = True
            operation_fd = os.open(
                operation_id,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=staging_root_fd,
            )
            staged_metadata = _write_redetect_stage(operation_fd, sidecar_bytes)
            staged_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()

            def _assert_final_custody_authority() -> None:
                binding.assert_unchanged()
                _assert_directory_fd_matches_path(
                    output_root / "EXPORT" / "CUSTODY",
                    custody_fd,
                )
                _assert_directory_fd_matches_path(
                    output_root / "EXPORT" / "LANGUAGE",
                    language_fd,
                )
                if published_snapshot is None:
                    raise OSError("redetect target was not published")
                current_target = _read_optional_redetect_target(
                    language_fd,
                    target_name,
                )
                if (
                    current_target is None
                    or (current_target[0].st_dev, current_target[0].st_ino)
                    != (
                        published_snapshot[0].st_dev,
                        published_snapshot[0].st_ino,
                    )
                    or current_target[1] != sidecar_bytes
                ):
                    raise OSError("published redetect sidecar changed")

            published_snapshot: tuple[os.stat_result, bytes] | None = None
            with open_bound_chain_transaction(
                custody_dir_fd=custody_fd,
                document_id=custody_selection.document_id,
                expected_device=custody_selection.device,
                expected_inode=custody_selection.inode,
                expected_size=custody_selection.size_bytes,
                expected_chain_head=custody_selection.chain_head,
                max_bytes=_REDETECT_MAX_CUSTODY_BYTES,
                max_events=_REDETECT_MAX_CUSTODY_EVENTS,
                final_authority_check=_assert_final_custody_authority,
            ) as custody_transaction:
                prior_snapshot = _read_optional_redetect_target(
                    language_fd,
                    target_name,
                )
                prior_metadata = prior_snapshot[0] if prior_snapshot else None
                prior_bytes = prior_snapshot[1] if prior_snapshot else None
                prior_description = {
                    "present": prior_snapshot is not None,
                    "sha256": (
                        hashlib.sha256(prior_bytes).hexdigest()
                        if prior_bytes is not None
                        else None
                    ),
                    "size": len(prior_bytes) if prior_bytes is not None else 0,
                }
                prepared_event = custody_transaction.append_event(
                    "LANGUAGE_REDETECT_PREPARED",
                    {
                        "operation_id": operation_id,
                        "phase": "prepared",
                        "document_id": custody_selection.document_id,
                        "source_file": Path(relative_pdf).name,
                        "source_pdf_sha256": source_pdf_sha256,
                        "target_file": target_name,
                        "staged_sha256": staged_sha256,
                        "staged_size": len(sidecar_bytes),
                        "prior_target": prior_description,
                    },
                )
                custody_transaction.mark_rollback_savepoint()

                _assert_directory_fd_matches_path(
                    output_root / "EXPORT" / "CUSTODY",
                    custody_fd,
                )
                _assert_directory_fd_matches_path(
                    output_root / "EXPORT" / "LANGUAGE",
                    language_fd,
                )
                binding.assert_unchanged()

                if prior_metadata is not None:
                    os.rename(
                        target_name,
                        _REDETECT_PRIOR_NAME,
                        src_dir_fd=language_fd,
                        dst_dir_fd=operation_fd,
                    )
                    prior_moved = True
                    _assert_redetect_name_inode(
                        operation_fd,
                        _REDETECT_PRIOR_NAME,
                        prior_metadata,
                    )
                    moved_prior = _read_optional_redetect_target(
                        operation_fd,
                        _REDETECT_PRIOR_NAME,
                    )
                    if (
                        moved_prior is None
                        or (moved_prior[0].st_dev, moved_prior[0].st_ino)
                        != (prior_metadata.st_dev, prior_metadata.st_ino)
                        or moved_prior[1] != prior_bytes
                    ):
                        raise OSError("redetect prior target changed before publish")
                else:
                    try:
                        os.stat(
                            target_name,
                            dir_fd=language_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        pass
                    else:
                        raise OSError("redetect target appeared before publish")

                publish_attempted = True
                try:
                    os.link(
                        _REDETECT_STAGE_NAME,
                        target_name,
                        src_dir_fd=operation_fd,
                        dst_dir_fd=language_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    current_target = os.stat(
                        target_name,
                        dir_fd=language_fd,
                        follow_symlinks=False,
                    )
                    publish_attempted = (
                        current_target.st_dev,
                        current_target.st_ino,
                    ) == (
                        staged_metadata.st_dev,
                        staged_metadata.st_ino,
                    )
                    raise
                published = True
                os.unlink(_REDETECT_STAGE_NAME, dir_fd=operation_fd)
                _assert_redetect_name_inode(
                    language_fd,
                    target_name,
                    staged_metadata,
                )
                os.fsync(language_fd)
                os.fsync(operation_fd)
                published_snapshot = _read_optional_redetect_target(
                    language_fd,
                    target_name,
                )
                if (
                    published_snapshot is None
                    or (published_snapshot[0].st_dev, published_snapshot[0].st_ino)
                    != (staged_metadata.st_dev, staged_metadata.st_ino)
                    or published_snapshot[1] != sidecar_bytes
                ):
                    raise OSError("published redetect sidecar changed")

                _assert_redetect_name_snapshot(
                    language_fd,
                    target_name,
                    published_snapshot[0],
                )
                custody_transaction.append_event(
                    "LANGUAGE_REDETECTED",
                    {
                        "operation_id": operation_id,
                        "phase": "committed",
                        "prepared_event_hash": prepared_event["hash"],
                        "artifact_sha256": staged_sha256,
                        "artifact_size": len(sidecar_bytes),
                        "target_file": target_name,
                        "source_pdf_sha256": source_pdf_sha256,
                        "document_id": custody_selection.document_id,
                        "source_file": Path(relative_pdf).name,
                        "primary_language": result.get("primary_language", "und"),
                        "page_count": int(result.get("page_count", 0) or 0),
                        "detection_engine": "fasttext",
                    },
                )

            _assert_directory_fd_matches_path(
                output_root / "EXPORT" / "CUSTODY",
                custody_fd,
            )
            _assert_directory_fd_matches_path(
                output_root / "EXPORT" / "LANGUAGE",
                language_fd,
            )
            binding.assert_unchanged()
            _assert_redetect_name_snapshot(
                language_fd,
                target_name,
                published_snapshot[0],
            )
            logger.info(
                "Background redetect committed for %s as operation %s",
                job_id,
                operation_id,
            )
        except BaseException as exc:  # pragma: no cover - defensive
            if not isinstance(exc, Exception):
                control_flow_error = exc
                control_flow_traceback = exc.__traceback__
            transaction_outcome = (
                custody_transaction.outcome
                if custody_transaction is not None
                else "safe_uncommitted"
            )
            if transaction_outcome in {"committed", "rollback_uncertain"}:
                cleanup_allowed = False
                logger.critical(
                    "Background redetect failed after custody outcome %s for %s "
                    "operation %s; canonical artifact and recovery evidence "
                    "were not mutated: %s",
                    transaction_outcome,
                    job_id,
                    operation_id,
                    exc,
                )
            elif (
                transaction_outcome == "safe_uncommitted"
                and operation_fd >= 0
                and staged_metadata is not None
            ):
                try:
                    _restore_redetect_publish(
                        language_fd=language_fd,
                        operation_fd=operation_fd,
                        target_name=target_name,
                        staged_metadata=staged_metadata,
                        staged_payload=sidecar_bytes,
                        publish_attempted=publish_attempted,
                        published=published,
                        prior_moved=prior_moved,
                        prior_snapshot=prior_snapshot,
                    )
                except BaseException as restore_exc:
                    cleanup_allowed = False
                    if (
                        not isinstance(restore_exc, Exception)
                        and control_flow_error is None
                    ):
                        control_flow_error = restore_exc
                        control_flow_traceback = restore_exc.__traceback__
                    logger.critical(
                        "Background redetect restore failed for %s operation %s: %s",
                        job_id,
                        operation_id,
                        restore_exc,
                    )
            elif transaction_outcome != "safe_uncommitted":
                cleanup_allowed = False
                logger.critical(
                    "Background redetect produced unknown custody outcome %r for "
                    "%s operation %s; recovery evidence was preserved",
                    transaction_outcome,
                    job_id,
                    operation_id,
                )
            logger.warning(
                "Background redetect failed for %s operation %s: %s",
                job_id,
                operation_id,
                exc,
            )
        finally:
            if operation_fd >= 0 and staging_root_fd >= 0 and cleanup_allowed:
                try:
                    _cleanup_redetect_operation(
                        staging_root_fd=staging_root_fd,
                        operation_fd=operation_fd,
                        operation_id=operation_id,
                    )
                except OSError as cleanup_exc:
                    logger.warning(
                        "Background redetect staging cleanup failed for %s "
                        "operation %s: %s",
                        job_id,
                        operation_id,
                        cleanup_exc,
                    )
            elif operation_created and staging_root_fd >= 0 and cleanup_allowed:
                try:
                    os.rmdir(operation_id, dir_fd=staging_root_fd)
                    os.fsync(staging_root_fd)
                except OSError as cleanup_exc:
                    logger.warning(
                        "Background redetect empty-operation cleanup failed for %s "
                        "operation %s: %s",
                        job_id,
                        operation_id,
                        cleanup_exc,
                    )
            for local_fd in (operation_fd, staging_root_fd):
                if local_fd >= 0:
                    try:
                        os.close(local_fd)
                    except OSError:
                        pass
            _close_owned_fds()
        if control_flow_error is not None:
            raise control_flow_error.with_traceback(control_flow_traceback)

    try:
        from api.background_work import start_writer

        start_writer(_run, thread_factory=Thread)
    except Exception:
        _close_owned_fds()
        raise
    return {
        "status": "queued",
        "job_id": job_id,
        "operation_id": operation_id,
    }


# ------------------------------------------------------------------
# DELETE /api/v1/jobs/{job_id} --- Cancel job
# ------------------------------------------------------------------


@router.delete(
    "/{job_id}",
    name="cancel_job",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
@limiter.limit(get_default_rate())
async def cancel_job(
    request: Request,
    job_id: str,
    _auth: None = Depends(require_role("admin", "operator")),
):
    """Cancel a running job."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_job_id", "message": "Invalid job ID format."},
        )
    manager = _get_manager()
    job = manager.cancel_job(job_id, tenant_id=_job_tenant_filter(request))
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
        )
    return _job_status_response(job)


# ------------------------------------------------------------------
# SSE helpers
# ------------------------------------------------------------------

# Terminal job states that end the SSE stream.
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


def _sse_event(event_type: str, data: dict) -> str:
    """Format a single Server-Sent Event frame.

    Returns a string in the SSE wire format:
        event: <type>\\ndata: <json>\\n\\n
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _sse_generator(
    job_id: str,
    request: Request,
    session_factory,
    tenant_id: Optional[str] = None,
):
    """Yield SSE frames tracking *job_id* until it reaches a terminal state.

    Uses an asyncio.Event notification hub (``api.sse_notifier``) so that
    job-status changes wake the generator immediately instead of requiring
    a fixed-interval poll.  A 30-second fallback timeout ensures the stream
    still advances even when a signal is missed (process restart, etc.).
    """
    from api.sse_notifier import create_job_event, remove_job_event

    last_status: Optional[str] = None
    last_pages: Optional[int] = None
    timeout = _bounded_sse_float(
        "SSE_STREAM_TIMEOUT",
        _SSE_STREAM_TIMEOUT_DEFAULT,
        minimum=0.01,
        maximum=_SSE_STREAM_TIMEOUT_MAX,
    )
    # maintenance review: default raised from 2s to 30s -- events fire instantly on change;
    # this interval is now only a safety-net fallback.
    fallback_interval = _bounded_sse_float(
        "SSE_POLL_INTERVAL",
        _SSE_POLL_INTERVAL_DEFAULT,
        minimum=0.01,
        maximum=_SSE_POLL_INTERVAL_MAX,
    )
    start = time.monotonic()

    _evt = create_job_event(job_id)
    try:
        while time.monotonic() - start < timeout:
            # Respect client disconnects
            if await request.is_disconnected():
                return

            try:
                session = session_factory()
                try:
                    query = session.query(Job).filter(Job.job_id == job_id)
                    if tenant_id is not None:
                        query = query.filter(Job.tenant_id == tenant_id)
                    job = query.first()
                finally:
                    session.close()

                if not job:
                    yield _sse_event(
                        "error", {"job_id": job_id, "error": "Job not found"}
                    )
                    yield _sse_event("done", {"job_id": job_id})
                    return

                current_status = job.status

                # Emit a status event whenever the status changes.
                if current_status != last_status:
                    yield _sse_event(
                        "status",
                        {
                            "job_id": job_id,
                            "status": current_status,
                            "previous_status": last_status,
                        },
                    )
                    last_status = current_status

                # Emit progress when in a non-terminal state and page counts changed.
                if current_status not in _TERMINAL_STATES:
                    pages_completed = job.pages_completed or 0
                    if pages_completed != last_pages:
                        progress_data: dict = {
                            "job_id": job_id,
                            "status": current_status,
                            "pages_completed": pages_completed,
                        }
                        if job.total_pages:
                            progress_data["total_pages"] = job.total_pages
                            progress_data["percent"] = round(
                                pages_completed / job.total_pages * 100,
                                1,
                            )
                        if job.current_stage:
                            progress_data["current_stage"] = job.current_stage
                        yield _sse_event("progress", progress_data)
                        last_pages = pages_completed

                # Terminal state handling
                if current_status in _TERMINAL_STATES:
                    if current_status == "completed":
                        result_data: dict = {
                            "job_id": job_id,
                            "status": "completed",
                            # Do not disclose the server filesystem path in a
                            # public stream. Use the authenticated artifact
                            # endpoint returned by the result API instead.
                            "result_available": bool(job.result_path),
                        }
                        if job.processing_time is not None:
                            result_data["processing_time_seconds"] = job.processing_time
                        if job.pages_completed is not None:
                            result_data["pages_processed"] = job.pages_completed
                        yield _sse_event("result", result_data)
                    elif current_status == "failed":
                        yield _sse_event(
                            "error",
                            {
                                "job_id": job_id,
                                "status": "failed",
                                "error": job.error_message or "Unknown error",
                            },
                        )
                    # cancelled falls through to done

                    yield _sse_event("done", {"job_id": job_id})
                    return

            except Exception as exc:
                logger.warning("SSE generator error for %s: %s", job_id, exc)
                # Never stream database/path/provider exception text to the
                # caller; it can disclose internal topology or PII.
                yield _sse_event(
                    "error",
                    {
                        "job_id": job_id,
                        "error": "Job progress is temporarily unavailable.",
                    },
                )

            # maintenance review: wait for event-driven notification or fallback timeout
            try:
                await asyncio.wait_for(
                    asyncio.shield(_evt.wait()),
                    timeout=fallback_interval,
                )
                _evt.clear()  # reset for next signal
            except asyncio.TimeoutError:
                pass  # fallback poll — query DB on next iteration

        # Stream timeout
        yield _sse_event("error", {"job_id": job_id, "error": "Stream timeout"})
        yield _sse_event("done", {"job_id": job_id})
    finally:
        remove_job_event(job_id, _evt)


# ------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}/stream --- SSE streaming
# ------------------------------------------------------------------


@router.get(
    "/{job_id}/stream",
    name="stream_job_progress",
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
@limiter.limit(get_default_rate())
async def stream_job_progress(
    request: Request,
    job_id: str,
    _auth: None = Depends(require_role("admin", "operator", "viewer")),
):
    """Stream job progress via Server-Sent Events (SSE).

    Sends events for status changes, page completion, and final result.

    Event types:
    - **status** -- emitted when job status changes (submitted, processing, etc.)
    - **progress** -- periodic progress with pages completed / total pages
    - **result** -- emitted when job completes successfully
    - **error** -- emitted on job failure or stream error
    - **done** -- final event, stream closes after this
    """
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_job_id", "message": "Invalid job ID format."},
        )

    session_factory = get_session_factory()
    tenant_id = _job_tenant_filter(request)

    # Verify job exists (and belongs to this tenant) before opening the stream.
    session = session_factory()
    try:
        query = session.query(Job).filter(Job.job_id == job_id)
        if tenant_id is not None:
            query = query.filter(Job.tenant_id == tenant_id)
        job = query.first()
    finally:
        session.close()

    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found."},
        )

    return StreamingResponse(
        _sse_generator(job_id, request, session_factory, tenant_id=tenant_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
