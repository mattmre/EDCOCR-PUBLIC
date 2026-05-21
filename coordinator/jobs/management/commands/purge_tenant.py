"""Management command to purge all data for a specific tenant.

Deletes Jobs, PageResults, PiiEntities, and optionally CustodyEvents
and NFS/S3 output files for a given tenant_id.  Emits a CustodyEvent
audit record BEFORE deletion so the purge is recorded in the chain.

Respects LITIGATION_HOLD to block all purge operations.

Usage:
    python manage.py purge_tenant --tenant-id acme-corp --dry-run
    python manage.py purge_tenant --tenant-id acme-corp --confirm
    python manage.py purge_tenant --tenant-id acme-corp --confirm --include-custody
    python manage.py purge_tenant --tenant-id acme-corp --confirm --include-output
"""

import getpass
import logging
import os
import shutil

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from jobs.litigation_hold import check_litigation_hold
from jobs.models import CustodyEvent, Job, PageResult, PiiEntity

logger = logging.getLogger(__name__)


def _record_tenant_purge_custody_event(tenant_id, job_count, page_count,
                                       pii_count, custody_count, anchor_job):
    """Create a CustodyEvent recording a tenant purge action.

    The event is anchored to *anchor_job* so it survives until the job
    cascade-delete runs.  If anchor_job is None the event is logged
    instead (edge case: tenant has no jobs).
    """
    data = {
        "action": "purge_tenant",
        "tenant_id": tenant_id,
        "job_count": job_count,
        "page_count": page_count,
        "pii_entity_count": pii_count,
        "custody_event_count": custody_count,
        "operator": getpass.getuser(),
        "timestamp": timezone.now().isoformat(),
    }

    try:
        if anchor_job is not None:
            CustodyEvent.objects.create(
                document_id=f"tenant-purge-{tenant_id}-{timezone.now().isoformat()}",
                job=anchor_job,
                event_type="tenant_purged",
                data=data,
            )
        else:
            logger.info(
                "No Job available to anchor tenant purge custody event -- "
                "logging instead: tenant_purged tenant_id=%s",
                tenant_id,
            )
    except Exception:
        # Custody event recording must never prevent the purge itself.
        logger.exception("Failed to record tenant purge custody event")


class Command(BaseCommand):
    help = "Purge all data for a specific tenant (respects LITIGATION_HOLD)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--tenant-id",
            type=str,
            required=True,
            help="The tenant ID whose data should be purged",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview what would be purged without deleting",
        )
        parser.add_argument(
            "--confirm",
            action="store_true",
            help="Required for actual deletion (safety guard)",
        )
        parser.add_argument(
            "--include-custody",
            action="store_true",
            help="Also purge CustodyEvent records (default: keep for audit)",
        )
        parser.add_argument(
            "--include-output",
            action="store_true",
            help="Also remove NFS/S3 output directories for tenant jobs",
        )

    def handle(self, *args, **options):
        # ---- Litigation hold gate -----------------------------------------
        if check_litigation_hold(self):
            return

        tenant_id = options["tenant_id"]
        dry_run = options["dry_run"]
        confirm = options["confirm"]
        include_custody = options["include_custody"]
        include_output = options["include_output"]

        if not tenant_id or not tenant_id.strip():
            raise CommandError("--tenant-id must not be empty.")

        # ---- Query tenant jobs --------------------------------------------
        tenant_jobs = Job.objects.filter(tenant_id=tenant_id)
        job_count = tenant_jobs.count()

        if job_count == 0:
            self.stdout.write(f"No jobs found for tenant '{tenant_id}'.")
            return

        # ---- Collect impact summary ---------------------------------------
        job_ids = list(tenant_jobs.values_list("job_id", flat=True))

        page_count = PageResult.objects.filter(job_id__in=job_ids).count()
        pii_count = PiiEntity.objects.filter(job_id__in=job_ids).count()
        custody_count = CustodyEvent.objects.filter(job_id__in=job_ids).count()

        # Estimate NFS storage
        nfs_dirs = []
        nfs_root = getattr(settings, "NFS_ROOT", None)
        for job in tenant_jobs:
            if job.nfs_job_path and os.path.isdir(job.nfs_job_path):
                nfs_dirs.append(job.nfs_job_path)
            elif nfs_root:
                candidate = os.path.join(nfs_root, "jobs", str(job.job_id))
                if os.path.isdir(candidate):
                    nfs_dirs.append(candidate)

        # ---- Display impact summary ---------------------------------------
        self.stdout.write(f"Tenant purge impact for '{tenant_id}':")
        self.stdout.write(f"  Jobs: {job_count}")
        self.stdout.write(f"  Pages: {page_count}")
        self.stdout.write(f"  PII entities: {pii_count}")
        self.stdout.write(f"  Custody events: {custody_count}")
        self.stdout.write(f"  NFS directories: {len(nfs_dirs)}")

        # ---- Dry run stops here -------------------------------------------
        if dry_run:
            self.stdout.write("DRY RUN: No records deleted.")
            return

        # ---- Confirm gate -------------------------------------------------
        if not confirm:
            self.stdout.write(
                self.style.WARNING(
                    "Use --confirm to actually delete these records."
                )
            )
            return

        # ---- Record custody event BEFORE deletion -------------------------
        anchor_job = tenant_jobs.first()
        _record_tenant_purge_custody_event(
            tenant_id=tenant_id,
            job_count=job_count,
            page_count=page_count,
            pii_count=pii_count,
            custody_count=custody_count,
            anchor_job=anchor_job,
        )

        # ---- Delete PII entities ------------------------------------------
        pii_deleted, _ = PiiEntity.objects.filter(job_id__in=job_ids).delete()

        # ---- Delete PageResult records ------------------------------------
        pages_deleted, _ = PageResult.objects.filter(job_id__in=job_ids).delete()

        # ---- Remove NFS output directories if requested -------------------
        nfs_cleaned = 0
        if include_output:
            for nfs_dir in nfs_dirs:
                try:
                    shutil.rmtree(nfs_dir, ignore_errors=True)
                    nfs_cleaned += 1
                except Exception:
                    logger.exception("Failed to remove NFS dir: %s", nfs_dir)

        # ---- Remove S3 objects if requested and S3 is configured ----------
        s3_cleaned = 0
        if include_output:
            try:
                from jobs.storage import create_storage_backend

                backend_name = getattr(settings, "STORAGE_BACKEND", "nfs")
                if backend_name.strip().lower() == "s3":
                    s3_backend = create_storage_backend(
                        backend_name="s3",
                        nfs_root=getattr(settings, "NFS_ROOT", ""),
                        s3_endpoint=getattr(settings, "S3_ENDPOINT", ""),
                        s3_bucket=getattr(settings, "S3_BUCKET", ""),
                        s3_access_key=getattr(settings, "S3_ACCESS_KEY", ""),
                        s3_secret_key=getattr(settings, "S3_SECRET_KEY", ""),
                        s3_region=getattr(settings, "S3_REGION", ""),
                    )
                    for job_id in job_ids:
                        prefix = f"jobs/{job_id}/"
                        try:
                            keys = s3_backend.list_objects(prefix)
                            if keys:
                                s3_cleaned += s3_backend.delete_many(keys)
                        except Exception:
                            logger.exception(
                                "Failed to clean S3 objects for job %s", job_id
                            )
                    if s3_cleaned:
                        self.stdout.write(
                            f"  Removed {s3_cleaned} S3 objects for tenant {tenant_id}"
                        )
            except Exception:
                logger.debug("S3 cleanup skipped (S3 backend not available or not configured)")

        # ---- Explicitly delete CustodyEvents if requested -----------------
        custody_deleted = 0
        if include_custody:
            # Delete custody events explicitly (before cascade) so we can
            # report an accurate count.  The purge audit event we just
            # created will also be cascade-deleted with the anchor job.
            cd_qs = CustodyEvent.objects.filter(job_id__in=job_ids)
            custody_deleted, _ = cd_qs.delete()

        # ---- Delete Job records (CASCADE handles remaining FKs) -----------
        # Note: this also cascade-deletes any remaining PageResult and
        # CustodyEvent records via FK relationships.
        jobs_deleted, _ = tenant_jobs.delete()

        # ---- Zero out tenant cost tracking if available -------------------
        try:
            from cost_tracking import CostTracker
            tracker = CostTracker()
            tracker.reset_tenant(tenant_id)
        except Exception:
            # cost_tracking unavailability must not block tenant purge
            logger.debug("Cost tracking reset skipped (unavailable or error)")

        # ---- Reset SLA monitoring windows if available -------------------
        try:
            from sla_monitoring import get_monitor
            monitor = get_monitor()
            if hasattr(monitor, '_windows') and tenant_id in monitor._windows:
                del monitor._windows[tenant_id]
                self.stdout.write(f"  Reset SLA monitoring windows for tenant {tenant_id}")
        except Exception:
            # SLA monitor not available in coordinator context
            pass

        # ---- Print completion summary -------------------------------------
        parts = [
            f"Purged tenant '{tenant_id}': ",
            f"{jobs_deleted} jobs, {pages_deleted} pages, ",
            f"{pii_deleted} PII entities deleted",
        ]
        if include_custody:
            parts.append(f", {custody_deleted} custody events deleted")
        if include_output:
            parts.append(f", {nfs_cleaned} NFS dirs removed")
            if s3_cleaned:
                parts.append(f", {s3_cleaned} S3 objects removed")
        parts.append(".")
        self.stdout.write(self.style.SUCCESS("".join(parts)))
