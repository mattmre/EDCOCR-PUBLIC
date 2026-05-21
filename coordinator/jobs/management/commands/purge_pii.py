"""Management command to purge PII/PHI entity records on demand.

Supports targeted purge by job ID, subject name, or entity type.
Emits CustodyEvent audit records for every purge operation.
Respects LITIGATION_HOLD to block all purges when active.

Usage:
    python manage.py purge_pii --job-id <uuid>                     # Purge all PII for a job
    python manage.py purge_pii --subject "Jane Doe"                # Purge by subject name
    python manage.py purge_pii --subject "Jane" --entity-type SSN  # Filter by entity type
    python manage.py purge_pii --job-id <uuid> --dry-run           # Preview only
    python manage.py purge_pii --job-id <uuid> --confirm           # Actually delete
"""

import getpass
import logging
import os
import shutil

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from jobs.extraction_models import ExtractedEntity
from jobs.litigation_hold import check_litigation_hold
from jobs.models import CustodyEvent, Job, PiiEntity

logger = logging.getLogger(__name__)


def _record_pii_purge_custody_event(purge_scope, entity_count, entity_types,
                                    affected_job_ids, job=None):
    """Create a CustodyEvent recording a PII purge action.

    If *job* is provided, anchors the event to that specific job.
    Otherwise, anchors to the first available Job (same pattern as
    cleanup_old_jobs.py).
    """
    data = {
        "action": "purge_pii",
        "purge_scope": purge_scope,
        "entity_count": entity_count,
        "entity_types": sorted(entity_types),
        "affected_jobs": [str(jid) for jid in affected_job_ids],
        "operator": getpass.getuser(),
    }

    try:
        anchor_job = job
        if anchor_job is None:
            anchor_job = Job.objects.first()
        if anchor_job is not None:
            CustodyEvent.objects.create(
                document_id=f"pii-purge-{timezone.now().isoformat()}",
                job=anchor_job,
                event_type="pii_purged",
                data=data,
            )
        else:
            logger.info(
                "No Job available to anchor custody event -- logging instead: "
                "pii_purged scope=%s count=%d",
                purge_scope,
                entity_count,
            )
    except Exception:
        # Custody event recording must never prevent the purge itself.
        logger.exception("Failed to record PII purge custody event")


def _delete_ner_files_for_job(job_id):
    """Remove PII-bearing sidecar output dirs associated with a job from NFS.

    Covers NER, EXTRACTION, HANDWRITING, and SIGNATURE directories under
    {NFS_ROOT}/jobs/{job_id}/output/EXPORT/<dir>.

    Returns the number of directories removed.
    """
    removed = 0
    nfs_root = getattr(settings, "NFS_ROOT", None)

    sidecar_dirs = ("NER", "EXTRACTION", "HANDWRITING", "SIGNATURE")

    if nfs_root:
        for subdir in sidecar_dirs:
            target = os.path.join(
                nfs_root, "jobs", str(job_id), "output", "EXPORT", subdir,
            )
            if os.path.isdir(target):
                try:
                    shutil.rmtree(target, ignore_errors=True)
                    removed += 1
                except Exception:
                    logger.exception("Failed to remove %s dir: %s", subdir, target)

    return removed


def _clear_redis_cache_for_jobs(job_ids):
    """Attempt to clear Redis chunk cache entries for affected jobs.

    Returns the number of keys deleted, or 0 if Redis is unavailable.
    """
    deleted = 0
    try:
        from jobs.redis_streams import RedisStreamClient
        client = RedisStreamClient()
        for job_id in job_ids:
            # Convention: chunk keys use job_id as prefix
            deleted += client.delete_chunk_cache(str(job_id))
    except Exception:
        # Redis unavailability must not block PII purge
        logger.warning("Redis cache cleanup skipped (unavailable or error)")
    return deleted


class Command(BaseCommand):
    help = "Purge PII/PHI entity records (respects LITIGATION_HOLD)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--job-id",
            type=str,
            default=None,
            help="Purge all PII for a specific job UUID",
        )
        parser.add_argument(
            "--subject",
            type=str,
            default=None,
            help="Purge PII matching entity_value containing this subject name (case-insensitive)",
        )
        parser.add_argument(
            "--entity-type",
            type=str,
            default=None,
            help="Filter by entity type (SSN, DOB, EMAIL, PHONE, NAME, etc.)",
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

    def handle(self, *args, **options):
        # ---- Litigation hold gate -----------------------------------------
        if check_litigation_hold(self):
            return

        job_id = options["job_id"]
        subject = options["subject"]
        entity_type = options["entity_type"]
        dry_run = options["dry_run"]
        confirm = options["confirm"]

        # Must provide at least one filter
        if not job_id and not subject:
            raise CommandError(
                "At least one of --job-id or --subject must be provided."
            )

        # ---- Build queryset -----------------------------------------------
        qs = PiiEntity.objects.all()

        if job_id:
            qs = qs.filter(job_id=job_id)

        if subject:
            qs = qs.filter(entity_value__icontains=subject)

        if entity_type:
            qs = qs.filter(entity_type=entity_type)

        count = qs.count()

        if count == 0:
            self.stdout.write("No matching PII entities found.")
            return

        # ---- Collect summary info -----------------------------------------
        entity_types_found = set()
        affected_job_ids = set()
        for entity in qs:
            entity_types_found.add(entity.entity_type)
            affected_job_ids.add(entity.job_id)

        # ---- Display summary ----------------------------------------------
        self.stdout.write(f"Found {count} PII entities to purge:")
        self.stdout.write(f"  Entity types: {', '.join(sorted(entity_types_found))}")
        self.stdout.write(f"  Affected jobs: {len(affected_job_ids)}")

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

        # ---- Build purge scope label --------------------------------------
        scope_parts = []
        if job_id:
            scope_parts.append(f"job_id={job_id}")
        if subject:
            scope_parts.append(f"subject={subject}")
        if entity_type:
            scope_parts.append(f"entity_type={entity_type}")
        purge_scope = "; ".join(scope_parts)

        # ---- Record custody event BEFORE deletion -------------------------
        anchor_job = None
        if job_id:
            try:
                anchor_job = Job.objects.filter(job_id=job_id).first()
            except Exception:
                pass

        _record_pii_purge_custody_event(
            purge_scope=purge_scope,
            entity_count=count,
            entity_types=entity_types_found,
            affected_job_ids=affected_job_ids,
            job=anchor_job,
        )

        # ---- Delete PII records from database -----------------------------
        deleted_count, _ = qs.delete()

        # ---- Delete matching ExtractedEntity records ----------------------
        extracted_qs = ExtractedEntity.objects.none()
        if job_id:
            extracted_qs = ExtractedEntity.objects.filter(job_id=job_id)
        if subject:
            extracted_qs = extracted_qs | ExtractedEntity.objects.filter(
                entity_text__icontains=subject,
            )
        extracted_deleted = 0
        if extracted_qs.exists():
            extracted_deleted, _ = extracted_qs.delete()

        # ---- Clean up sidecar files from NFS ------------------------------
        sidecar_cleaned = 0
        for jid in affected_job_ids:
            sidecar_cleaned += _delete_ner_files_for_job(jid)

        # ---- Clear Redis chunk cache --------------------------------------
        redis_cleaned = _clear_redis_cache_for_jobs(affected_job_ids)

        self.stdout.write(self.style.SUCCESS(
            f"Purged {deleted_count} PII entities, "
            f"{extracted_deleted} extracted entities "
            f"({sidecar_cleaned} sidecar dirs removed, "
            f"{redis_cleaned} cache keys cleared)."
        ))
