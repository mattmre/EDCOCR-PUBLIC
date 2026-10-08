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
    python manage.py cleanup_old_jobs --confirm    # 30-day default
    python manage.py cleanup_old_jobs --days 7 --confirm
    python manage.py cleanup_old_jobs --dry-run    # Preview only
"""

import getpass
import logging
import os

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from jobs.litigation_hold import check_litigation_hold
from jobs.models import Job, PiiEntity
from jobs.rls import control_plane_rls_command
from ocr_distributed.tenant_purge_fs import remove_path_no_follow, safe_nfs_job_path

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
        from jobs.tasks import _record_control_plane_custody_event

        job_id = str(getattr(job, "job_id", "") or data.get("job_id") or "").strip()
        tenant_id = str(
            getattr(job, "tenant_id", "")
            or data.get("tenant_id")
            or "__control_plane__"
        ).strip()
        document_id = f"retention-{job_id}" if job_id else "retention-control-plane"
        _record_control_plane_custody_event(
            tenant_id=tenant_id,
            document_id=document_id,
            event_type=event_type,
            data=data,
            worker_hostname="management-command",
        )
    except Exception:
        logger.exception("Failed to record deletion custody event")
        return False
    return True


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
            "--confirm",
            action="store_true",
            help="Acknowledge that matching jobs and stored artifacts will be deleted",
        )
        parser.add_argument(
            "--include-pii",
            action="store_true",
            help="Also purge PII entities older than PII_ENTITY_RETENTION_DAYS",
        )

    @control_plane_rls_command
    def handle(self, *args, **options):
        # ---- Litigation hold gate -----------------------------------------
        if check_litigation_hold(self):
            return

        days = options["days"]
        dry_run = options["dry_run"]
        confirm = options.get("confirm", False)
        include_pii = options["include_pii"]

        if days <= 0:
            raise CommandError("Retention days must be positive")
        if not dry_run and not confirm:
            raise CommandError(
                "Refusing destructive cleanup without --confirm; "
                "use --dry-run to preview candidates"
            )

        self._cleanup_jobs(days, dry_run)

        if include_pii:
            pii_days = _get_retention_days(
                "PII_ENTITY_RETENTION_DAYS", _DEFAULT_PII_ENTITY_RETENTION_DAYS
            )
            self._cleanup_pii_entities(pii_days, dry_run)

    # ----- Job cleanup ------------------------------------------------------

    def _cleanup_jobs(self, days, dry_run):
        cleanup_failures = self._cleanup_jobs_atomic(days, dry_run)
        if cleanup_failures:
            raise CommandError(
                "Cleanup blocked: artifact removal was incomplete; "
                + "; ".join(cleanup_failures[:5])
            )

    def _cleanup_jobs_atomic(self, days, dry_run):
        """Run one committed retention saga per job.

        No database transaction spans external storage I/O.  Each job gets a
        committed START event before bytes are touched and a committed
        SUCCESS/FAIL event before the next candidate.  A later custody, row
        deletion, commit, or process failure therefore cannot erase the audit
        boundary for bytes that were already removed.
        """
        cutoff = timezone.now() - timezone.timedelta(days=days)

        old_jobs = Job.objects.filter(
            status__in=[Job.Status.COMPLETED, Job.Status.FAILED, Job.Status.CANCELLED],
            completed_at__isnull=False,
            completed_at__lt=cutoff,
        )
        candidate_ids = list(old_jobs.values_list("job_id", flat=True))
        count = len(candidate_ids)

        if count == 0:
            self.stdout.write(f"No jobs older than {days} days to clean up.")
            return

        if dry_run:
            self.stdout.write(
                f"DRY RUN: Would delete {count} jobs older than {days} days."
            )
            for job in old_jobs[:10]:
                self.stdout.write(
                    f"  {job.job_id} ({job.status}, completed {job.completed_at})"
                )
            if count > 10:
                self.stdout.write(f"  ... and {count - 10} more")
            return

        # Remove NFS artifacts for jobs with an explicit path or a canonical
        # path left behind after a restart.  Both forms remain job-bound.
        nfs_cleaned = 0
        s3_jobs_cleaned = 0
        s3_objects_cleaned = 0
        cleanup_failures = []
        nfs_root = getattr(settings, "NFS_ROOT", None)
        # Resolve the backend captured at ingest for every job.  Cleanup must
        # not use today's global backend after an operator changes storage
        # configuration or when a tenant contains mixed-backend jobs.
        from jobs.tasks import _get_backend_for_job

        jobs_deleted = 0
        for candidate_id in candidate_ids:
            # Revalidate and lock only while committing START. The lock must
            # be released before filesystem/object-store calls.
            with transaction.atomic():
                job = (
                    Job.objects.select_for_update()
                    .filter(
                        job_id=candidate_id,
                        status__in=[
                            Job.Status.COMPLETED,
                            Job.Status.FAILED,
                            Job.Status.CANCELLED,
                        ],
                        completed_at__isnull=False,
                        completed_at__lt=cutoff,
                    )
                    .first()
                )
                if job is None:
                    continue
                original_status = job.status
                job.status = Job.Status.CANCELLED
                job.dispatch_generation = int(job.dispatch_generation or 0) + 1
                job.celery_task_id = ""
                job.last_failure_reason = "retention_deletion_fence"
                job.save(
                    update_fields=[
                        "status",
                        "dispatch_generation",
                        "celery_task_id",
                        "last_failure_reason",
                    ]
                )
                fenced_generation = job.dispatch_generation
                from jobs.models import (
                    PresignedTaskCapability,
                    TaskDispatchOutbox,
                )

                TaskDispatchOutbox.objects.filter(
                    job=job,
                    status__in=[
                        TaskDispatchOutbox.Status.PENDING,
                        TaskDispatchOutbox.Status.FAILED,
                    ],
                ).update(
                    status=TaskDispatchOutbox.Status.DEAD,
                    dead_at=timezone.now(),
                    last_error="retention deletion fenced dispatch",
                )
                PresignedTaskCapability.objects.filter(job=job).delete()
                if not _record_deletion_custody_event(
                    event_type="data_deletion_started",
                    data={
                        "action": "cleanup_old_jobs",
                        "job_id": str(job.job_id),
                        "tenant_id": str(job.tenant_id or "").strip(),
                        "retention_policy": f"JOB_RETENTION_DAYS={days}",
                        "operator": getpass.getuser(),
                        "original_status": original_status,
                        "fenced_generation": fenced_generation,
                    },
                    job=job,
                ):
                    cleanup_failures.append(
                        f"job {job.job_id}: deletion-start custody failed"
                    )
                    transaction.set_rollback(True)
                    continue

            job_failure = None
            backend = None
            try:
                backend = _get_backend_for_job(job)
                if backend.backend_name == "nfs":
                    raw_path = job.nfs_job_path or (
                        os.path.join(nfs_root, "jobs", str(job.job_id))
                        if nfs_root
                        else ""
                    )
                    if raw_path:
                        safe_path = safe_nfs_job_path(
                            raw_path,
                            nfs_root,
                            job_id=str(job.job_id),
                        )
                        if safe_path is None:
                            job_failure = f"job {job.job_id}: untrusted nfs_job_path"
                        elif os.path.lexists(safe_path):
                            if remove_path_no_follow(safe_path):
                                nfs_cleaned += 1
                            else:
                                job_failure = f"job {job.job_id}: NFS directory remains"
                else:
                    prefix = f"jobs/{job.job_id}/"
                    keys = backend.list_objects(prefix)
                    if keys:
                        deleted = backend.delete_many(keys)
                        remaining = backend.list_objects(prefix)
                        if deleted != len(keys) or remaining:
                            job_failure = (
                                f"job {job.job_id}: deleted {deleted} of {len(keys)} "
                                f"objects; remaining={len(remaining)}"
                            )
                        else:
                            s3_jobs_cleaned += 1
                            s3_objects_cleaned += deleted
            except Exception as exc:
                job_failure = f"job {job.job_id}: storage cleanup failed ({exc})"

            if job_failure is not None:
                cleanup_failures.append(job_failure)
                if not _record_deletion_custody_event(
                    event_type="data_deletion_failed",
                    data={
                        "action": "cleanup_old_jobs",
                        "job_id": str(job.job_id),
                        "tenant_id": str(job.tenant_id or "").strip(),
                        "failure": job_failure,
                        "operator": getpass.getuser(),
                    },
                    job=job,
                ):
                    cleanup_failures.append(
                        f"job {job.job_id}: deletion-failure custody failed"
                    )
                continue

            # Commit completion custody and row deletion only if the exact
            # cancellation/generation fence still holds.  Retry/admission
            # paths increment the generation, so they cannot race storage
            # deletion and later be misreported as a successful retention.
            try:
                with transaction.atomic():
                    locked_job = (
                        Job.objects.select_for_update()
                        .filter(
                            job_id=job.job_id,
                            status=Job.Status.CANCELLED,
                            dispatch_generation=fenced_generation,
                            completed_at__isnull=False,
                            completed_at__lt=cutoff,
                        )
                        .first()
                    )
                    if locked_job is None:
                        raise RuntimeError(
                            "retention deletion fence changed after storage cleanup"
                        )
                    if not _record_deletion_custody_event(
                        event_type="data_deleted",
                        data={
                            "action": "cleanup_old_jobs",
                            "job_id": str(job.job_id),
                            "tenant_id": str(job.tenant_id or "").strip(),
                            "storage_backend": backend.backend_name,
                            "reason": "retention_policy",
                            "retention_policy": f"JOB_RETENTION_DAYS={days}",
                            "operator": getpass.getuser(),
                            "fenced_generation": fenced_generation,
                        },
                        job=locked_job,
                    ):
                        raise RuntimeError("deletion-completion custody failed")
                    locked_job.delete()
                    jobs_deleted += 1
            except Exception as exc:
                job_failure = f"job {job.job_id}: database cleanup failed ({exc})"
                cleanup_failures.append(job_failure)
                if not _record_deletion_custody_event(
                    event_type="data_deletion_failed",
                    data={
                        "action": "cleanup_old_jobs",
                        "job_id": str(job.job_id),
                        "tenant_id": str(job.tenant_id or "").strip(),
                        "failure": job_failure,
                        "phase": "database_delete",
                        "operator": getpass.getuser(),
                    },
                    job=job,
                ):
                    cleanup_failures.append(
                        f"job {job.job_id}: database-failure custody failed"
                    )

        self.stdout.write(
            self.style.SUCCESS(
                f"Deleted {jobs_deleted} jobs older than {days} days "
                f"({nfs_cleaned} NFS dirs removed, "
                f"{s3_objects_cleaned} S3 objects removed)."
            )
        )
        return cleanup_failures

    # ----- PII entity cleanup -----------------------------------------------

    def _cleanup_pii_entities(self, days, dry_run):
        if days <= 0:
            raise CommandError("PII retention days must be positive")
        cutoff = timezone.now() - timezone.timedelta(days=days)
        from jobs.extraction_models import (
            DocumentChunk,
            ExtractedEntity,
            ExtractedFormValue,
        )
        from jobs.management.commands.purge_pii import erase_pii_for_job

        terminal_statuses = [
            Job.Status.COMPLETED,
            Job.Status.FAILED,
            Job.Status.CANCELLED,
        ]
        candidate_ids = set(
            PiiEntity.objects.filter(
                created_at__lt=cutoff,
                job__status__in=terminal_statuses,
            ).values_list("job_id", flat=True)
        )
        candidate_ids.update(
            ExtractedEntity.objects.filter(
                created_at__lt=cutoff,
                job__status__in=terminal_statuses,
            ).values_list("job_id", flat=True)
        )
        candidate_ids.update(
            ExtractedFormValue.objects.filter(
                created_at__lt=cutoff,
                job__status__in=terminal_statuses,
            ).values_list("job_id", flat=True)
        )
        candidate_ids.update(
            DocumentChunk.objects.filter(
                created_at__lt=cutoff,
                contains_pii=True,
                job__status__in=terminal_statuses,
            ).values_list("job_id", flat=True)
        )
        if not candidate_ids:
            self.stdout.write(
                f"No PII-bearing jobs older than {days} days to clean up."
            )
            return

        if dry_run:
            self.stdout.write(
                "DRY RUN: Would erase all PII derivatives for "
                f"{len(candidate_ids)} jobs with PII older than {days} days."
            )
            return

        totals = {
            "pii_entities": 0,
            "extracted_entities": 0,
            "form_values": 0,
            "pii_chunks": 0,
        }
        failures = []
        jobs = Job.objects.filter(job_id__in=candidate_ids).order_by("job_id")
        for job in jobs:
            result = erase_pii_for_job(
                job,
                purge_scope=f"retention_days={days}",
                source="cleanup_old_jobs",
            )
            failures.extend(result["failures"])
            for key in totals:
                totals[key] += result["deleted"].get(key, 0)
        if failures:
            raise CommandError("PII cleanup incomplete; " + "; ".join(failures[:5]))

        self.stdout.write(
            self.style.SUCCESS(
                f"Erased PII derivatives for {len(candidate_ids)} jobs: {totals}."
            )
        )
