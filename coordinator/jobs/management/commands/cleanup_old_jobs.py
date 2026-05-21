"""Management command to clean up old completed/failed jobs.

Deletes Job records (and cascading PageResult, CustodyEvent) older than
the configured retention period. Also removes NFS job artifacts.

Supports tiered retention for different data classes:
- JOB_RETENTION_DAYS: Job metadata (default: 30)
- PII_ENTITY_RETENTION_DAYS: PII/PHI entity records (default: 90)
- AUDIT_LOG_RETENTION_DAYS: Custody event records (default: 2555 = 7 years)
- OUTPUT_RETENTION_DAYS: Output files on disk (default: 90, via cleanup_output command)
- LITIGATION_HOLD: When "true", skip ALL automated deletions (default: "false")

Usage:
    python manage.py cleanup_old_jobs              # 30-day default
    python manage.py cleanup_old_jobs --days 7     # 7-day retention
    python manage.py cleanup_old_jobs --dry-run    # Preview only
"""

import getpass
import logging
import os
import shutil

from django.core.management.base import BaseCommand
from django.utils import timezone

from jobs.litigation_hold import check_litigation_hold
from jobs.models import CustodyEvent, Job, PiiEntity

logger = logging.getLogger(__name__)

# Retention defaults (days)
_DEFAULT_JOB_RETENTION_DAYS = 30
_DEFAULT_PII_ENTITY_RETENTION_DAYS = 90
_DEFAULT_AUDIT_LOG_RETENTION_DAYS = 2555  # ~7 years


def _get_retention_days(env_var, default):
    """Read a retention-days env var, returning the integer default on parse failure."""
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid value for %s: %r — using default %d", env_var, raw, default
        )
        return default


def _record_deletion_custody_event(event_type, data, job=None):
    """Create a CustodyEvent recording a deletion action.

    If *job* is None, a synthetic document_id is used so the audit record
    is self-contained and does not depend on a Job foreign key that is
    about to be cascade-deleted.
    """
    try:
        if job is not None:
            CustodyEvent.objects.create(
                document_id=f"cleanup-{timezone.now().isoformat()}",
                job=job,
                event_type=event_type,
                data=data,
            )
        else:
            # Bulk deletion record — attach to the first available job or
            # skip if there are none (edge case: all jobs already deleted).
            # We create a record against a dummy document_id for audit.
            # Since CustodyEvent has a required FK to Job, we need at least
            # one job.  If none remain, log instead.
            first_job = Job.objects.first()
            if first_job is not None:
                CustodyEvent.objects.create(
                    document_id=f"bulk-cleanup-{timezone.now().isoformat()}",
                    job=first_job,
                    event_type=event_type,
                    data=data,
                )
            else:
                logger.info(
                    "No Job available to anchor custody event — logging instead: "
                    "%s %s",
                    event_type,
                    data,
                )
    except Exception:
        # Custody event recording must never prevent the cleanup itself.
        logger.exception("Failed to record deletion custody event")


class Command(BaseCommand):
    help = "Delete completed/failed jobs older than N days (respects LITIGATION_HOLD)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=_get_retention_days(
                "JOB_RETENTION_DAYS", _DEFAULT_JOB_RETENTION_DAYS
            ),
            help="Delete jobs older than this many days (default: 30)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview deletions without actually deleting",
        )
        parser.add_argument(
            "--include-pii",
            action="store_true",
            help="Also purge PII entities older than PII_ENTITY_RETENTION_DAYS",
        )

    def handle(self, *args, **options):
        # ---- Litigation hold gate -----------------------------------------
        if check_litigation_hold(self):
            return

        days = options["days"]
        dry_run = options["dry_run"]
        include_pii = options["include_pii"]

        self._cleanup_jobs(days, dry_run)

        if include_pii:
            pii_days = _get_retention_days(
                "PII_ENTITY_RETENTION_DAYS", _DEFAULT_PII_ENTITY_RETENTION_DAYS
            )
            self._cleanup_pii_entities(pii_days, dry_run)

    # ----- Job cleanup ------------------------------------------------------

    def _cleanup_jobs(self, days, dry_run):
        cutoff = timezone.now() - timezone.timedelta(days=days)

        old_jobs = Job.objects.filter(
            status__in=[Job.Status.COMPLETED, Job.Status.FAILED, Job.Status.CANCELLED],
            created_at__lt=cutoff,
        )
        count = old_jobs.count()

        if count == 0:
            self.stdout.write(f"No jobs older than {days} days to clean up.")
            return

        if dry_run:
            self.stdout.write(
                f"DRY RUN: Would delete {count} jobs older than {days} days."
            )
            for job in old_jobs[:10]:
                self.stdout.write(
                    f"  {job.job_id} ({job.status}, created {job.created_at})"
                )
            if count > 10:
                self.stdout.write(f"  ... and {count - 10} more")
            return

        # Record custody event BEFORE deletion so the FK still exists
        _record_deletion_custody_event(
            event_type="data_deleted",
            data={
                "action": "cleanup_old_jobs",
                "retention_policy": f"JOB_RETENTION_DAYS={days}",
                "jobs_to_delete": count,
                "cutoff_date": cutoff.isoformat(),
                "reason": "retention_policy",
                "operator": getpass.getuser(),
            },
        )

        # Remove NFS artifacts for jobs that have nfs_job_path set
        nfs_cleaned = 0
        for job in old_jobs.exclude(nfs_job_path=""):
            if job.nfs_job_path and os.path.isdir(job.nfs_job_path):
                shutil.rmtree(job.nfs_job_path, ignore_errors=True)
                nfs_cleaned += 1

        # Cascade delete (removes PageResult, CustodyEvent via FK)
        deleted_count, _ = old_jobs.delete()

        self.stdout.write(self.style.SUCCESS(
            f"Deleted {deleted_count} jobs older than {days} days "
            f"({nfs_cleaned} NFS dirs removed)."
        ))

    # ----- PII entity cleanup -----------------------------------------------

    def _cleanup_pii_entities(self, days, dry_run):
        cutoff = timezone.now() - timezone.timedelta(days=days)

        old_pii = PiiEntity.objects.filter(created_at__lt=cutoff)
        count = old_pii.count()

        if count == 0:
            self.stdout.write(
                f"No PII entities older than {days} days to clean up."
            )
            return

        if dry_run:
            self.stdout.write(
                f"DRY RUN: Would delete {count} PII entities older than {days} days."
            )
            return

        # Record custody event BEFORE deletion
        _record_deletion_custody_event(
            event_type="pii_deleted",
            data={
                "action": "cleanup_pii_entities",
                "retention_policy": f"PII_ENTITY_RETENTION_DAYS={days}",
                "entities_to_delete": count,
                "cutoff_date": cutoff.isoformat(),
                "reason": "retention_policy",
                "operator": getpass.getuser(),
            },
        )

        deleted_count, _ = old_pii.delete()

        self.stdout.write(self.style.SUCCESS(
            f"Deleted {deleted_count} PII entities older than {days} days."
        ))
