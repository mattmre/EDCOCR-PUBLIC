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
import shutil

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from jobs.litigation_hold import check_litigation_hold
from jobs.models import CustodyEvent, Job

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_RETENTION_DAYS = 90

# All EXPORT sub-directories that may contain output artifacts.
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
    "EXPORT/VERTICAL",
    "EXPORT/CUSTODY",
]


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

    # Prefer the explicit nfs_job_path if set
    if job.nfs_job_path and os.path.isdir(job.nfs_job_path):
        output_dir = os.path.join(job.nfs_job_path, "output")
        if os.path.isdir(output_dir):
            return output_dir

    # Fall back to convention: {NFS_ROOT}/jobs/{job_id}/output
    if nfs_root:
        candidate = os.path.join(nfs_root, "jobs", str(job.job_id), "output")
        if os.path.isdir(candidate):
            return candidate

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

    Returns (files_removed, bytes_freed).
    """
    files_removed = 0
    bytes_freed = 0
    for subdir in _OUTPUT_SUBDIRS:
        full_path = os.path.join(output_dir, subdir)
        if os.path.isdir(full_path):
            fc, sz = _count_files_and_size(full_path)
            files_removed += fc
            bytes_freed += sz
            shutil.rmtree(full_path, ignore_errors=True)
    return files_removed, bytes_freed


def _record_cleanup_custody_event(job_count, file_count, bytes_freed, retention_days):
    """Create a CustodyEvent recording an output cleanup action.

    Anchors to the first available Job (same pattern as cleanup_old_jobs.py).
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
        first_job = Job.objects.first()
        if first_job is not None:
            CustodyEvent.objects.create(
                document_id=f"output-cleanup-{timezone.now().isoformat()}",
                job=first_job,
                event_type="output_cleaned",
                data=data,
            )
        else:
            logger.info(
                "No Job available to anchor custody event -- logging instead: "
                "output_cleaned jobs=%d files=%d bytes=%d",
                job_count,
                file_count,
                bytes_freed,
            )
    except Exception:
        # Custody event recording must never prevent the cleanup itself.
        logger.exception("Failed to record output cleanup custody event")


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

    def handle(self, *args, **options):
        # ---- Litigation hold gate -----------------------------------------
        if check_litigation_hold(self):
            return

        retention_days = options["retention_days"]
        dry_run = options["dry_run"]
        confirm = options["confirm"]

        cutoff = timezone.now() - timezone.timedelta(days=retention_days)

        # Find completed jobs older than the retention period
        old_jobs = Job.objects.filter(
            status__in=[Job.Status.COMPLETED, Job.Status.FAILED],
            completed_at__lt=cutoff,
        )
        job_count = old_jobs.count()

        if job_count == 0:
            self.stdout.write(
                f"No completed/failed jobs older than {retention_days} days."
            )
            return

        # ---- Scan output directories for size estimates -------------------
        jobs_with_output = []
        total_files = 0
        total_bytes = 0

        for job in old_jobs:
            output_dir = _compute_output_dir(job)
            if output_dir:
                fc, sz = _count_files_and_size(output_dir)
                if fc > 0:
                    jobs_with_output.append((job, output_dir, fc, sz))
                    total_files += fc
                    total_bytes += sz

        if not jobs_with_output:
            self.stdout.write(
                f"Found {job_count} old jobs but none have output files on disk."
            )
            return

        size_mb = total_bytes / (1024 * 1024)

        # ---- Display summary ----------------------------------------------
        self.stdout.write(
            f"Found {len(jobs_with_output)} jobs with output files "
            f"({total_files} files, {size_mb:.1f} MB)."
        )

        # ---- Dry run stops here -------------------------------------------
        if dry_run:
            self.stdout.write("DRY RUN: No files removed.")
            for job, output_dir, fc, sz in jobs_with_output[:10]:
                self.stdout.write(
                    f"  {job.job_id}: {fc} files ({sz / (1024 * 1024):.1f} MB)"
                )
            if len(jobs_with_output) > 10:
                self.stdout.write(
                    f"  ... and {len(jobs_with_output) - 10} more"
                )
            return

        # ---- Confirm gate -------------------------------------------------
        if not confirm:
            self.stdout.write(
                self.style.WARNING(
                    "Use --confirm to actually delete these files."
                )
            )
            return

        # ---- Perform cleanup -----------------------------------------------
        cleaned_jobs = 0
        cleaned_files = 0
        cleaned_bytes = 0

        for job, output_dir, _, _ in jobs_with_output:
            fc, sz = _remove_output_subdirs(output_dir)
            cleaned_files += fc
            cleaned_bytes += sz
            cleaned_jobs += 1

        cleaned_mb = cleaned_bytes / (1024 * 1024)

        # ---- Record custody event AFTER successful cleanup ----------------
        _record_cleanup_custody_event(
            job_count=cleaned_jobs,
            file_count=cleaned_files,
            bytes_freed=cleaned_bytes,
            retention_days=retention_days,
        )

        self.stdout.write(self.style.SUCCESS(
            f"Cleaned output for {cleaned_jobs} jobs: "
            f"{cleaned_files} files removed ({cleaned_mb:.1f} MB recovered)."
        ))
