"""Management command to clean up output files for old completed jobs.

Removes output files (EXPORT/PDF, EXPORT/TEXT, EXPORT/NER, etc.) from NFS
storage for jobs that have completed beyond the retention window.  The Job
records and CustodyEvents are preserved -- only disk artifacts are deleted.

Supports:
- ``--retention-days N``: Override default from ``OUTPUT_RETENTION_DAYS``
  env var (default: 90 days).
- ``--dry-run``: Preview files that would be removed without deleting.
- ``--confirm``: Required safety guard for actual deletion.
- LITIGATION_HOLD env var blocks all automated deletions.

Emits a CustodyEvent (``output_cleaned``) recording the cleanup action.

Usage:
    python manage.py cleanup_output --dry-run
    python manage.py cleanup_output --retention-days 60 --confirm
"""

import getpass
import logging
import os
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from jobs.litigation_hold import check_litigation_hold
from jobs.models import CustodyEvent, Job
from jobs.rls import control_plane_rls_command
from ocr_distributed.tenant_purge_fs import remove_path_no_follow, safe_nfs_job_path

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_RETENTION_DAYS = 90
_OUTPUT_DELETION_FENCE_KEY = "_output_deletion_fence"

# All output directories that may contain retention-managed artifacts.  The
# generation pointer layout lives below ``output/generations`` rather than
# under ``output/EXPORT``; keep both layouts covered by the same job-age gate.
_OUTPUT_SUBDIRS = [
    "EXPORT/PDF",
    "EXPORT/TEXT",
    "EXPORT/NER",
    "EXPORT/STRUCTURE",
    "EXPORT/VALIDATION",
    "EXPORT/HANDWRITING",
    "EXPORT/SIGNATURE",
    "EXPORT/CLASSIFICATION",
    "EXPORT/EXTRACTION",
    "EXPORT/ENTITIES",
    "EXPORT/VERTICAL",
    "EXPORT/CUSTODY",
    "EXPORT/TRANSLATION",
    "EXPORT/RETRIEVAL",
    "EXPORT/LANGUAGE",
    "EXPORT/FORM",
    "EXPORT/TABLES",
    "EXPORT/TABLE_FALLBACK",
    "EXPORT/ROUTING",
    "EXPORT/AI_TEXT",
    "EXPORT/SYMBOLOGY",
    "generations",
]


def _output_deletion_fence(job):
    """Return validated persisted output-deletion fence metadata, if present."""
    settings_json = job.settings_json if isinstance(job.settings_json, dict) else {}
    raw = settings_json.get(_OUTPUT_DELETION_FENCE_KEY)
    if not isinstance(raw, dict):
        return None
    original_status = str(raw.get("original_status") or "")
    if original_status not in {Job.Status.COMPLETED, Job.Status.FAILED}:
        return None
    try:
        fenced_generation = int(raw["fenced_generation"])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "original_status": original_status,
        "fenced_generation": fenced_generation,
    }


def _set_output_deletion_fence(job, marker):
    """Copy-on-write the Job settings marker used for crash recovery."""
    settings_json = (
        dict(job.settings_json) if isinstance(job.settings_json, dict) else {}
    )
    if marker is None:
        settings_json.pop(_OUTPUT_DELETION_FENCE_KEY, None)
    else:
        settings_json[_OUTPUT_DELETION_FENCE_KEY] = marker
    job.settings_json = settings_json


def _get_retention_days():
    """Read OUTPUT_RETENTION_DAYS env var, returning the integer default on parse failure."""
    raw = os.environ.get("OUTPUT_RETENTION_DAYS")
    if raw is None:
        return _DEFAULT_OUTPUT_RETENTION_DAYS
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid value for OUTPUT_RETENTION_DAYS: %r -- using default %d",
            raw,
            _DEFAULT_OUTPUT_RETENTION_DAYS,
        )
        return _DEFAULT_OUTPUT_RETENTION_DAYS


def _compute_output_dir(job):
    """Determine the NFS output directory for a job.

    Returns the path string if one can be determined, else None.
    """
    nfs_root = getattr(settings, "NFS_ROOT", None)

    # Prefer the explicit path only when it is the job's canonical path under
    # NFS_ROOT.  A database row must not become an arbitrary deletion path.
    safe_job_path = safe_nfs_job_path(
        job.nfs_job_path,
        nfs_root,
        job_id=str(job.job_id),
    )
    if safe_job_path and safe_job_path.is_dir():
        output_dir = Path(safe_job_path) / "output"
        if os.path.lexists(output_dir):
            if output_dir.is_symlink():
                raise CommandError(
                    f"Output cleanup blocked: job {job.job_id} output path is symlinked"
                )
            resolved_root = Path(safe_job_path).resolve(strict=False)
            try:
                resolved_output = output_dir.resolve(strict=False)
                resolved_output.relative_to(resolved_root)
            except (OSError, ValueError) as exc:
                raise CommandError(
                    f"Output cleanup blocked: job {job.job_id} output path escapes job root"
                ) from exc
            if resolved_output != output_dir:
                raise CommandError(
                    f"Output cleanup blocked: job {job.job_id} output path is symlinked"
                )
            if output_dir.is_dir():
                return str(output_dir)

    # Fall back to convention: {NFS_ROOT}/jobs/{job_id}/output
    if nfs_root:
        safe_candidate = safe_nfs_job_path(
            os.path.join(nfs_root, "jobs", str(job.job_id)),
            nfs_root,
            job_id=str(job.job_id),
        )
        if safe_candidate:
            candidate = safe_candidate / "output"
            if os.path.lexists(candidate):
                if candidate.is_symlink():
                    raise CommandError(
                        f"Output cleanup blocked: job {job.job_id} output path is symlinked"
                    )
                resolved_root = safe_candidate.resolve(strict=False)
                try:
                    resolved_candidate = candidate.resolve(strict=False)
                    resolved_candidate.relative_to(resolved_root)
                except (OSError, ValueError) as exc:
                    raise CommandError(
                        f"Output cleanup blocked: job {job.job_id} output path escapes job root"
                    ) from exc
                if resolved_candidate != candidate:
                    raise CommandError(
                        f"Output cleanup blocked: job {job.job_id} output path is symlinked"
                    )
                if candidate.is_dir():
                    return str(candidate)

    return None


def _count_files_and_size(directory):
    """Walk *directory* and return (file_count, total_bytes)."""
    file_count = 0
    total_bytes = 0
    for dirpath, _, filenames in os.walk(directory):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                total_bytes += os.path.getsize(fpath)
            except OSError:
                pass
            file_count += 1
    return file_count, total_bytes


def _remove_output_subdirs(output_dir):
    """Remove EXPORT sub-directories from *output_dir*.

    Returns (files_removed, bytes_freed, failures).
    """
    files_removed = 0
    bytes_freed = 0
    failures = []
    for subdir in _OUTPUT_SUBDIRS:
        full_path = os.path.join(output_dir, subdir)
        if os.path.lexists(full_path) and not os.path.isdir(full_path):
            failures.append(full_path)
            continue
        if os.path.isdir(full_path):
            fc, sz = _count_files_and_size(full_path)
            files_removed += fc
            bytes_freed += sz
            try:
                removed = remove_path_no_follow(full_path)
            except (OSError, RuntimeError):
                removed = False
            if not removed or os.path.lexists(full_path):
                failures.append(full_path)
    return files_removed, bytes_freed, failures


def _record_cleanup_custody_event(job_count, file_count, bytes_freed, retention_days):
    """Create a CustodyEvent recording an output cleanup action.

    This is a control-plane event and is intentionally not attached to an
    arbitrary Job row.
    """
    data = {
        "action": "cleanup_output",
        "retention_policy": f"OUTPUT_RETENTION_DAYS={retention_days}",
        "jobs_cleaned": job_count,
        "files_removed": file_count,
        "bytes_freed": bytes_freed,
        "reason": "retention_policy",
        "operator": getpass.getuser(),
    }

    try:
        from jobs.tasks import _record_control_plane_custody_event

        _record_control_plane_custody_event(
            tenant_id="__control_plane__",
            document_id="output-retention-control-plane",
            event_type="output_cleaned",
            data=data,
            worker_hostname="management-command",
        )
    except Exception:
        logger.exception("Failed to record output cleanup custody event")
        return False
    return True


def _record_output_job_custody_event(job, event_type, data):
    """Seal a per-job output deletion transition outside storage rollback."""
    try:
        from jobs.tasks import _record_control_plane_custody_event

        _record_control_plane_custody_event(
            tenant_id=str(job.tenant_id or "").strip() or "__unscoped__",
            document_id=f"output-retention-{job.job_id}",
            event_type=event_type,
            data={"job_id": str(job.job_id), **data},
            worker_hostname="management-command",
        )
    except Exception:
        logger.exception(
            "Failed to record %s custody for output cleanup job %s",
            event_type,
            job.job_id,
        )
        return False
    return True


def _has_unfinished_output_deletion(job) -> bool:
    """Return whether a committed START lacks a terminal custody transition."""
    # The persisted lifecycle fence is the authoritative crash-recovery marker.
    # A later discovery failure is an audit observation, not a terminal deletion
    # transition, and must not hide an already committed START.
    if _output_deletion_fence(job) is not None:
        return True
    latest = (
        CustodyEvent.objects.filter(
            job__isnull=True,
            tenant_id=str(job.tenant_id or "").strip() or "__unscoped__",
            document_id=f"output-retention-{job.job_id}",
        )
        .order_by("-id")
        .values_list("event_type", flat=True)
        .first()
    )
    return latest == "output_deletion_started"


def _commit_output_deletion_fence(
    job,
    *,
    cutoff,
    retention_days,
    backend_name,
    record_start=True,
):
    """Commit a generation/retry fence before irreversible output deletion."""
    from jobs.models import PresignedTaskCapability, TaskDispatchOutbox

    with transaction.atomic():
        locked_job = (
            Job.objects.select_for_update()
            .filter(
                job_id=job.job_id,
                completed_at__lt=cutoff,
            )
            .first()
        )
        if locked_job is None:
            return None
        existing_fence = _output_deletion_fence(locked_job)
        if existing_fence is not None:
            original_status = existing_fence["original_status"]
            expected_status = (
                Job.Status.CANCELLED
                if original_status == Job.Status.FAILED
                else Job.Status.COMPLETED
            )
            if (
                locked_job.status != expected_status
                or int(locked_job.dispatch_generation or 0)
                != existing_fence["fenced_generation"]
            ):
                return None
            return (
                locked_job,
                locked_job.status,
                existing_fence["fenced_generation"],
                original_status,
            )
        if locked_job.status not in {Job.Status.COMPLETED, Job.Status.FAILED}:
            return None
        original_status = locked_job.status
        # FAILED is otherwise eligible for the admin retry action.  COMPLETED
        # stays semantically completed, but both states receive a new dispatch
        # generation so already-issued work loses publication authority.
        if locked_job.status == Job.Status.FAILED:
            locked_job.status = Job.Status.CANCELLED
        locked_job.dispatch_generation = int(locked_job.dispatch_generation or 0) + 1
        locked_job.celery_task_id = ""
        _set_output_deletion_fence(
            locked_job,
            {
                "original_status": original_status,
                "fenced_generation": locked_job.dispatch_generation,
            },
        )
        locked_job.save(
            update_fields=[
                "status",
                "dispatch_generation",
                "celery_task_id",
                "settings_json",
            ]
        )
        fenced_status = locked_job.status
        fenced_generation = locked_job.dispatch_generation
        TaskDispatchOutbox.objects.filter(
            job=locked_job,
            status__in=[
                TaskDispatchOutbox.Status.PENDING,
                TaskDispatchOutbox.Status.FAILED,
            ],
        ).update(
            status=TaskDispatchOutbox.Status.DEAD,
            dead_at=timezone.now(),
            last_error="output retention fenced dispatch",
        )
        PresignedTaskCapability.objects.filter(job=locked_job).delete()
        if record_start and not _record_output_job_custody_event(
            locked_job,
            "output_deletion_started",
            {
                "retention_days": retention_days,
                "storage_backend": backend_name,
                "original_status": original_status,
                "fenced_generation": fenced_generation,
            },
        ):
            transaction.set_rollback(True)
            return None
    return locked_job, fenced_status, fenced_generation, original_status


def _commit_output_deletion_success(
    job,
    *,
    cutoff,
    fenced_status,
    fenced_generation,
    original_status,
    data,
):
    """Record SUCCESS only while the exact committed deletion fence holds."""
    with transaction.atomic():
        locked_job = (
            Job.objects.select_for_update()
            .filter(
                job_id=job.job_id,
                status=fenced_status,
                dispatch_generation=fenced_generation,
                completed_at__lt=cutoff,
            )
            .first()
        )
        if locked_job is None:
            raise RuntimeError("output deletion fence changed during storage cleanup")
        marker = _output_deletion_fence(locked_job)
        if marker != {
            "original_status": original_status,
            "fenced_generation": fenced_generation,
        }:
            raise RuntimeError("output deletion fence metadata changed")
        if not _record_output_job_custody_event(
            locked_job,
            "output_deleted",
            {**data, "fenced_generation": fenced_generation},
        ):
            raise RuntimeError("deletion-completion custody failed")
        locked_job.status = original_status
        _set_output_deletion_fence(locked_job, None)
        locked_job.save(update_fields=["status", "settings_json"])


def _commit_output_deletion_failure(
    job,
    *,
    cutoff,
    fenced_status,
    fenced_generation,
    original_status,
    data,
):
    """Commit terminal FAIL and restore the historical terminal job status."""
    with transaction.atomic():
        locked_job = (
            Job.objects.select_for_update()
            .filter(
                job_id=job.job_id,
                status=fenced_status,
                dispatch_generation=fenced_generation,
                completed_at__lt=cutoff,
            )
            .first()
        )
        if locked_job is None:
            raise RuntimeError("output deletion fence changed during failed cleanup")
        marker = _output_deletion_fence(locked_job)
        if marker != {
            "original_status": original_status,
            "fenced_generation": fenced_generation,
        }:
            raise RuntimeError("output deletion fence metadata changed")
        if not _record_output_job_custody_event(
            locked_job,
            "output_deletion_failed",
            {**data, "fenced_generation": fenced_generation},
        ):
            raise RuntimeError("deletion-failure custody failed")
        locked_job.status = original_status
        _set_output_deletion_fence(locked_job, None)
        locked_job.save(update_fields=["status", "settings_json"])


class Command(BaseCommand):
    help = "Remove output files for completed jobs older than N days (respects LITIGATION_HOLD)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--retention-days",
            type=int,
            default=_get_retention_days(),
            help=(
                "Remove output files for completed jobs older than this many days "
                f"(default: {_DEFAULT_OUTPUT_RETENTION_DAYS}, overridable via OUTPUT_RETENTION_DAYS env)"
            ),
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview what would be removed without actually deleting",
        )
        parser.add_argument(
            "--confirm",
            action="store_true",
            help="Required for actual deletion (safety guard)",
        )

    @control_plane_rls_command
    def handle(self, *args, **options):
        # ---- Litigation hold gate -----------------------------------------
        if check_litigation_hold(self):
            return

        retention_days = options["retention_days"]
        dry_run = options["dry_run"]
        confirm = options["confirm"]

        if retention_days <= 0:
            raise CommandError("Output retention days must be positive")

        cutoff = timezone.now() - timezone.timedelta(days=retention_days)

        # Include a FAILED job that was durably moved to CANCELLED as a
        # deletion fence before a prior process crashed.  The settings marker
        # distinguishes that internal fence from an ordinary cancellation.
        old_jobs = list(
            Job.objects.filter(
                status__in=[Job.Status.COMPLETED, Job.Status.FAILED],
                completed_at__lt=cutoff,
            )
        )
        fenced_jobs = Job.objects.filter(
            status=Job.Status.CANCELLED,
            completed_at__lt=cutoff,
            settings_json__has_key=_OUTPUT_DELETION_FENCE_KEY,
        )
        seen_job_ids = {job.job_id for job in old_jobs}
        old_jobs.extend(job for job in fenced_jobs if job.job_id not in seen_job_ids)
        old_jobs.sort(key=lambda job: (job.completed_at, str(job.job_id)))
        job_count = len(old_jobs)

        if job_count == 0:
            self.stdout.write(
                f"No completed/failed jobs older than {retention_days} days."
            )
            return

        # ---- Scan output directories for size estimates -------------------
        jobs_with_output = []
        s3_jobs_with_output = []
        recovery_jobs = []
        discovery_failures = []
        total_files = 0
        total_bytes = 0
        from jobs.tasks import _get_backend_for_job

        for job in old_jobs:
            try:
                backend = _get_backend_for_job(job)
                if backend.backend_name == "s3":
                    keys = backend.list_objects(f"jobs/{job.job_id}/output/")
                    if keys:
                        s3_jobs_with_output.append((job, backend, keys))
                        total_files += len(keys)
                    elif _has_unfinished_output_deletion(job):
                        recovery_jobs.append((job, "s3"))
                    continue
                output_dir = _compute_output_dir(job)
                if output_dir:
                    fc, sz = _count_files_and_size(output_dir)
                    if fc > 0:
                        jobs_with_output.append((job, output_dir, fc, sz))
                        total_files += fc
                        total_bytes += sz
                    elif _has_unfinished_output_deletion(job):
                        recovery_jobs.append((job, "nfs"))
                elif _has_unfinished_output_deletion(job):
                    recovery_jobs.append((job, "nfs"))
            except Exception as exc:
                error = (
                    f"job {job.job_id}: output discovery failed "
                    f"({type(exc).__name__}: {exc})"
                )
                discovery_failures.append(error)
                if (
                    confirm
                    and not dry_run
                    and not _record_output_job_custody_event(
                        job,
                        "output_deletion_discovery_failed",
                        {
                            "phase": "discovery",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                ):
                    discovery_failures.append(
                        f"job {job.job_id}: discovery-failure custody failed"
                    )
                continue

        if not jobs_with_output and not s3_jobs_with_output and not recovery_jobs:
            if discovery_failures:
                raise CommandError(
                    "Output cleanup blocked: one or more jobs could not be inspected; "
                    + "; ".join(discovery_failures[:5])
                )
            self.stdout.write(
                f"Found {job_count} old jobs but none have output files on disk."
            )
            return

        size_mb = total_bytes / (1024 * 1024)

        # ---- Display summary ----------------------------------------------
        self.stdout.write(
            f"Found {len(jobs_with_output) + len(s3_jobs_with_output)} jobs with output files "
            f"({total_files} files, {size_mb:.1f} MB) and "
            f"{len(recovery_jobs)} interrupted deletion saga(s)."
        )

        # ---- Dry run stops here -------------------------------------------
        if dry_run:
            self.stdout.write("DRY RUN: No files removed.")
            preview = (
                [(job, fc, sz) for job, _, fc, sz in jobs_with_output]
                + [(job, len(keys), 0) for job, _, keys in s3_jobs_with_output]
                + [(job, 0, 0) for job, _ in recovery_jobs]
            )
            for job, fc, sz in preview[:10]:
                self.stdout.write(
                    f"  {job.job_id}: {fc} files ({sz / (1024 * 1024):.1f} MB)"
                )
            if len(preview) > 10:
                self.stdout.write(f"  ... and {len(preview) - 10} more")
            if discovery_failures:
                raise CommandError(
                    "Output cleanup dry run incomplete: one or more jobs could not be "
                    "inspected; " + "; ".join(discovery_failures[:5])
                )
            return

        # ---- Confirm gate -------------------------------------------------
        if not confirm:
            self.stdout.write(
                self.style.WARNING("Use --confirm to actually delete these files.")
            )
            if discovery_failures:
                raise CommandError(
                    "Output cleanup preview incomplete: one or more jobs could not be "
                    "inspected; " + "; ".join(discovery_failures[:5])
                )
            return

        # ---- Perform cleanup -----------------------------------------------
        cleaned_jobs = 0
        cleaned_files = 0
        cleaned_bytes = 0
        cleanup_failures = list(discovery_failures)

        # A prior process may have committed START, removed every object, and
        # crashed before SUCCESS. The verified-empty scan above is sufficient
        # to close that durable saga without fabricating another deletion.
        for job, backend_name in recovery_jobs:
            fence = _commit_output_deletion_fence(
                job,
                cutoff=cutoff,
                retention_days=retention_days,
                backend_name=backend_name,
                record_start=False,
            )
            if fence is None:
                cleanup_failures.append(
                    f"job {job.job_id}: output deletion fence unavailable"
                )
                continue
            fenced_job, fenced_status, fenced_generation, original_status = fence
            try:
                _commit_output_deletion_success(
                    fenced_job,
                    cutoff=cutoff,
                    fenced_status=fenced_status,
                    fenced_generation=fenced_generation,
                    original_status=original_status,
                    data={
                        "files_removed": 0,
                        "reconciled_after_interruption": True,
                        "storage_backend": backend_name,
                    },
                )
                cleaned_jobs += 1
            except Exception as exc:
                cleanup_failures.append(
                    f"job {job.job_id}: interrupted deletion reconciliation failed ({exc})"
                )

        for job, output_dir, _, _ in jobs_with_output:
            fence = _commit_output_deletion_fence(
                job,
                cutoff=cutoff,
                retention_days=retention_days,
                backend_name="nfs",
            )
            if fence is None:
                cleanup_failures.append(
                    f"job {job.job_id}: deletion fence/start custody failed"
                )
                continue
            fenced_job, fenced_status, fenced_generation, original_status = fence
            fc, sz, failures = _remove_output_subdirs(output_dir)
            cleaned_files += fc
            cleaned_bytes += sz
            if failures:
                cleanup_failures.extend(
                    f"job {job.job_id}: {path}" for path in failures
                )
                try:
                    _commit_output_deletion_failure(
                        fenced_job,
                        cutoff=cutoff,
                        fenced_status=fenced_status,
                        fenced_generation=fenced_generation,
                        original_status=original_status,
                        data={
                            "remaining_paths": failures[:10],
                        },
                    )
                except Exception as exc:
                    cleanup_failures.append(
                        f"job {job.job_id}: deletion-failure custody failed ({exc})"
                    )
            else:
                try:
                    _commit_output_deletion_success(
                        fenced_job,
                        cutoff=cutoff,
                        fenced_status=fenced_status,
                        fenced_generation=fenced_generation,
                        original_status=original_status,
                        data={"files_removed": fc, "bytes_freed": sz},
                    )
                    cleaned_jobs += 1
                except Exception as exc:
                    cleanup_failures.append(
                        f"job {job.job_id}: deletion completion failed ({exc})"
                    )

        for job, backend, keys in s3_jobs_with_output:
            fence = _commit_output_deletion_fence(
                job,
                cutoff=cutoff,
                retention_days=retention_days,
                backend_name="s3",
            )
            if fence is None:
                cleanup_failures.append(
                    f"job {job.job_id}: deletion fence/start custody failed"
                )
                continue
            fenced_job, fenced_status, fenced_generation, original_status = fence
            try:
                deleted = backend.delete_many(keys)
                remaining = backend.list_objects(f"jobs/{job.job_id}/output/")
                cleaned_files += deleted
                if deleted != len(keys) or remaining:
                    cleanup_failures.append(
                        f"job {job.job_id}: deleted {deleted} of {len(keys)} S3 "
                        f"output objects; remaining={len(remaining)}"
                    )
                    try:
                        _commit_output_deletion_failure(
                            fenced_job,
                            cutoff=cutoff,
                            fenced_status=fenced_status,
                            fenced_generation=fenced_generation,
                            original_status=original_status,
                            data={
                                "objects_deleted": deleted,
                                "objects_expected": len(keys),
                                "objects_remaining": len(remaining),
                            },
                        )
                    except Exception as exc:
                        cleanup_failures.append(
                            f"job {job.job_id}: deletion-failure custody failed ({exc})"
                        )
                else:
                    try:
                        _commit_output_deletion_success(
                            fenced_job,
                            cutoff=cutoff,
                            fenced_status=fenced_status,
                            fenced_generation=fenced_generation,
                            original_status=original_status,
                            data={"objects_deleted": deleted},
                        )
                        cleaned_jobs += 1
                    except Exception as exc:
                        cleanup_failures.append(
                            f"job {job.job_id}: deletion completion failed ({exc})"
                        )
            except Exception as exc:
                cleanup_failures.append(
                    f"job {job.job_id}: S3 output cleanup failed ({exc})"
                )
                try:
                    _commit_output_deletion_failure(
                        fenced_job,
                        cutoff=cutoff,
                        fenced_status=fenced_status,
                        fenced_generation=fenced_generation,
                        original_status=original_status,
                        data={
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                except Exception as custody_exc:
                    cleanup_failures.append(
                        f"job {job.job_id}: deletion-failure custody failed ({custody_exc})"
                    )

        if cleanup_failures:
            raise CommandError(
                "Output cleanup blocked: one or more directories remain; "
                + "; ".join(cleanup_failures[:5])
            )

        cleaned_mb = cleaned_bytes / (1024 * 1024)

        # ---- Record custody event before deletion is considered complete --
        if not _record_cleanup_custody_event(
            job_count=cleaned_jobs,
            file_count=cleaned_files,
            bytes_freed=cleaned_bytes,
            retention_days=retention_days,
        ):
            raise CommandError(
                "Output cleanup blocked: custody event could not be recorded"
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Cleaned output for {cleaned_jobs} jobs: "
                f"{cleaned_files} files removed ({cleaned_mb:.1f} MB recovered)."
            )
        )
