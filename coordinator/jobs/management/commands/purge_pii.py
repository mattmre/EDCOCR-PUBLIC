"""Management command to purge PII/PHI entity records on demand.

Supports targeted purge by job ID, subject name, or entity type.
Emits CustodyEvent audit records for every purge operation.
Respects LITIGATION_HOLD to block all purges when active.

Usage:
    python manage.py purge_pii --job-id <uuid>                     # Purge all PII for a job
    python manage.py purge_pii --subject "Jane Doe"                # Purge PII-bearing jobs matching subject
    python manage.py purge_pii --subject "Jane" --entity-type SSN  # Select jobs by subject/type
    python manage.py purge_pii --job-id <uuid> --dry-run           # Preview only
    python manage.py purge_pii --job-id <uuid> --confirm           # Actually delete
"""

import getpass
import logging
import os
import shutil

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from jobs.extraction_models import ExtractedEntity
from jobs.litigation_hold import check_litigation_hold
from jobs.models import (
    Job,
    PiiEntity,
    PresignedTaskCapability,
    TaskDispatchOutbox,
)
from jobs.rls import control_plane_rls_command
from ocr_distributed.tenant_purge_fs import safe_nfs_job_path

logger = logging.getLogger(__name__)


def _record_pii_purge_custody_event(
    purge_scope,
    entity_count,
    entity_types,
    affected_job_ids,
    job=None,
    *,
    event_type="pii_purged",
    extra_data=None,
):
    """Create a CustodyEvent recording a PII purge action.

    If *job* is provided, anchors the event to that specific job. A global
    purge remains a control-plane event and is not attached to an arbitrary
    first Job row.
    """
    data = {
        "action": "purge_pii",
        "purge_scope": purge_scope,
        "entity_count": entity_count,
        "entity_types": sorted(entity_types),
        "affected_jobs": [str(jid) for jid in affected_job_ids],
        "operator": getpass.getuser(),
        **(extra_data or {}),
    }

    try:
        from jobs.tasks import _record_control_plane_custody_event

        job_id = str(getattr(job, "job_id", "") or "").strip()
        tenant_id = str(getattr(job, "tenant_id", "") or "").strip()
        _record_control_plane_custody_event(
            tenant_id=tenant_id or "__control_plane__",
            document_id=f"pii-purge:{job_id}" if job_id else "pii-purge-control-plane",
            event_type=event_type,
            data=data,
            worker_hostname="management-command",
        )
    except Exception:
        logger.exception("Failed to record PII purge custody event")
        return False
    return True


def _job_nfs_path(job_id):
    """Return the existing, canonical NFS directory for *job_id*.

    PII purge is destructive, so a job identifier is the authority for the
    path.  A path assembled from a database value must not be allowed to
    redirect deletion to another job or outside ``NFS_ROOT``.
    """
    nfs_root = getattr(settings, "NFS_ROOT", None)
    if not nfs_root:
        return None

    for raw_path in (
        os.path.join(nfs_root, "jobs", str(job_id)),
        os.path.join(nfs_root, str(job_id)),
    ):
        safe_path = safe_nfs_job_path(
            raw_path,
            nfs_root,
            job_id=str(job_id),
        )
        if safe_path is not None and safe_path.is_dir():
            return safe_path
    return None


def _delete_ner_files_for_job_checked(job_id):
    """Remove PII-bearing sidecar output dirs associated with a job from NFS.

    Covers NER, EXTRACTION, HANDWRITING, and SIGNATURE directories under
    {NFS_ROOT}/jobs/{job_id}/output/EXPORT/<dir>.

    Returns ``(directories_removed, failures)``.  Failure is explicit: a
    purge must not delete database entities while PII-bearing files remain.
    """
    removed = 0
    failures = []

    sidecar_dirs = (
        "NER",
        "ENTITIES",
        "EXTRACTION",
        "FORM",
        "HANDWRITING",
        "SIGNATURE",
    )
    job_root = _job_nfs_path(job_id)

    if job_root is None:
        return removed, failures

    for subdir in sidecar_dirs:
        target = job_root / "output" / "EXPORT" / subdir
        if not os.path.lexists(target):
            continue
        try:
            resolved_target = target.resolve(strict=False)
            resolved_target.relative_to(job_root.resolve(strict=False))
        except (OSError, ValueError):
            failures.append(f"job {job_id}: untrusted {subdir} sidecar path")
            continue
        if target.is_symlink() or resolved_target != target:
            failures.append(f"job {job_id}: symlinked {subdir} sidecar path")
            continue
        if not target.is_dir():
            failures.append(f"job {job_id}: {subdir} sidecar is not a directory")
            continue
        try:
            shutil.rmtree(target)
        except Exception:
            logger.exception("Failed to remove %s dir: %s", subdir, target)
            failures.append(f"job {job_id}: failed to remove {subdir} sidecar")
            continue
        if os.path.lexists(target):
            failures.append(f"job {job_id}: {subdir} sidecar remains")
        else:
            removed += 1

    return removed, failures


def _delete_ner_files_for_job(job_id):
    """Compatibility wrapper returning only the successful removal count."""
    removed, failures = _delete_ner_files_for_job_checked(job_id)
    if failures:
        logger.error("PII sidecar cleanup incomplete: %s", "; ".join(failures))
    return removed


def _delete_pii_sidecars_for_job_checked(job):
    """Delete all PII-bearing sidecars from the job's persisted backend."""
    from jobs.tasks import _get_backend_for_job

    backend = _get_backend_for_job(job)
    if backend.backend_name == "nfs":
        removed, failures = _delete_ner_files_for_job_checked(job.job_id)
        return removed, failures

    removed = 0
    failures = []
    for subdir in (
        "NER",
        "ENTITIES",
        "EXTRACTION",
        "FORM",
        "HANDWRITING",
        "SIGNATURE",
    ):
        prefix = f"jobs/{job.job_id}/output/EXPORT/{subdir}/"
        try:
            keys = list(backend.list_objects(prefix))
            if not keys:
                continue
            deleted = int(backend.delete_many(keys) or 0)
            remaining = list(backend.list_objects(prefix))
            if deleted != len(keys) or remaining:
                failures.append(
                    f"job {job.job_id}: S3 {subdir} deleted {deleted} of "
                    f"{len(keys)}; remaining={len(remaining)}"
                )
            else:
                removed += deleted
        except Exception as exc:
            failures.append(f"job {job.job_id}: S3 {subdir} cleanup failed ({exc})")
    return removed, failures


def _clear_redis_cache_for_jobs(job_ids):
    """Attempt to clear Redis chunk cache entries for affected jobs.

    Returns ``(keys_deleted, failures)``.  Redis is part of the temporary
    PII-bearing data plane, so an unavailable cache blocks the purge instead
    of being reported as a successful zero-key deletion.
    """
    deleted = 0
    failures = []
    try:
        from jobs.redis_streams import RedisStreamClient

        client = RedisStreamClient()
        for job_id in job_ids:
            # Convention: chunk keys use job_id as prefix
            deleted += int(client.delete_chunk_cache(str(job_id)) or 0)
            cache_error = getattr(client, "last_cache_error", None)
            if cache_error is not None:
                failures.append(f"job {job_id}: Redis cache cleanup failed")
    except Exception:
        logger.exception("Redis cache cleanup failed")
        failures.extend(f"job {job_id}: Redis cache unavailable" for job_id in job_ids)
    return deleted, failures


def erase_pii_for_job(
    job,
    *,
    purge_scope,
    source="management-command",
):
    """Run one durable, fenced PII-erasure saga for a terminal job.

    Every data plane below is job-scoped: sidecars and Redis cannot prove a
    subject-level rewrite. A subject/type filter therefore selects affected
    jobs, then the saga erases every PII-bearing row for each selected job.
    This keeps DB, sidecar, and cache planes consistent instead of deleting
    another subject's sidecar evidence while retaining its database rows.
    """
    from jobs.extraction_models import DocumentChunk, ExtractedFormValue

    terminal_statuses = {
        Job.Status.COMPLETED,
        Job.Status.FAILED,
        Job.Status.CANCELLED,
    }
    pii_targets = PiiEntity.objects.filter(job_id=job.job_id)
    extracted_targets = ExtractedEntity.objects.filter(job_id=job.job_id)
    form_targets = ExtractedFormValue.objects.filter(job_id=job.job_id)
    chunk_targets = DocumentChunk.objects.filter(
        job_id=job.job_id,
        contains_pii=True,
    )
    planned = {
        "pii_entities": pii_targets.count(),
        "extracted_entities": extracted_targets.count(),
        "form_values": form_targets.count(),
        "pii_chunks": chunk_targets.count(),
    }
    entity_types = set(pii_targets.values_list("entity_type", flat=True)) | set(
        extracted_targets.values_list("entity_type", flat=True)
    )
    job_ids = {job.job_id}

    try:
        with transaction.atomic():
            locked_job = Job.objects.select_for_update().get(pk=job.pk)
            if locked_job.status not in terminal_statuses:
                raise RuntimeError("job is no longer terminal")
            settings_json = dict(locked_job.settings_json or {})
            prior_fence = settings_json.get("_pii_erasure_fence")
            if isinstance(prior_fence, dict):
                original_status = str(
                    prior_fence.get("original_status") or Job.Status.CANCELLED
                )
                original_failure_reason = str(
                    prior_fence.get("original_failure_reason") or ""
                )
            else:
                original_status = str(locked_job.status)
                original_failure_reason = str(locked_job.last_failure_reason or "")
            settings_json["_pii_erasure_fence"] = {
                "original_status": original_status,
                "original_failure_reason": original_failure_reason,
                "started_at": timezone.now().isoformat(),
            }
            locked_job.status = Job.Status.CANCELLED
            locked_job.dispatch_generation = (
                int(locked_job.dispatch_generation or 0) + 1
            )
            locked_job.celery_task_id = ""
            locked_job.last_failure_reason = "pii_erasure_fence"
            locked_job.settings_json = settings_json
            locked_job.save(
                update_fields=[
                    "status",
                    "dispatch_generation",
                    "celery_task_id",
                    "last_failure_reason",
                    "settings_json",
                ]
            )
            fenced_generation = locked_job.dispatch_generation
            TaskDispatchOutbox.objects.filter(
                job=locked_job,
                status__in=[
                    TaskDispatchOutbox.Status.PENDING,
                    TaskDispatchOutbox.Status.FAILED,
                    TaskDispatchOutbox.Status.CLAIMED,
                ],
            ).update(
                status=TaskDispatchOutbox.Status.DEAD,
                dead_at=timezone.now(),
                last_error="PII erasure fenced dispatch",
            )
            PresignedTaskCapability.objects.filter(job=locked_job).delete()
            if not _record_pii_purge_custody_event(
                purge_scope,
                planned["pii_entities"],
                entity_types,
                job_ids,
                job=locked_job,
                event_type="pii_deletion_started",
                extra_data={
                    "phase": "start",
                    "source": source,
                    "fenced_generation": fenced_generation,
                    "deletion_plan": planned,
                    "original_status": original_status,
                },
            ):
                raise RuntimeError("PII deletion-start custody failed")
    except Exception as exc:
        return {
            "ok": False,
            "failures": [f"job {job.job_id}: PII deletion fence failed ({exc})"],
            "deleted": {},
            "sidecars_deleted": 0,
            "redis_deleted": 0,
        }

    sidecars_deleted, failures = _delete_pii_sidecars_for_job_checked(job)
    redis_deleted, redis_failures = _clear_redis_cache_for_jobs(job_ids)
    failures.extend(redis_failures)
    if failures:
        _record_pii_purge_custody_event(
            purge_scope,
            planned["pii_entities"],
            entity_types,
            job_ids,
            job=job,
            event_type="pii_deletion_failed",
            extra_data={
                "phase": "storage",
                "source": source,
                "failure": "; ".join(failures[:5]),
            },
        )
        return {
            "ok": False,
            "failures": failures,
            "deleted": {},
            "sidecars_deleted": sidecars_deleted,
            "redis_deleted": redis_deleted,
        }

    try:
        with transaction.atomic():
            locked_job = (
                Job.objects.select_for_update()
                .filter(
                    pk=job.pk,
                    status=Job.Status.CANCELLED,
                    dispatch_generation=fenced_generation,
                    last_failure_reason="pii_erasure_fence",
                )
                .first()
            )
            if locked_job is None:
                raise RuntimeError("PII erasure fence changed after storage cleanup")
            deleted = {
                "pii_entities": pii_targets.delete()[0],
                "extracted_entities": extracted_targets.delete()[0],
                "form_values": form_targets.delete()[0],
                "pii_chunks": chunk_targets.delete()[0],
            }
            if not _record_pii_purge_custody_event(
                purge_scope,
                planned["pii_entities"],
                entity_types,
                job_ids,
                job=locked_job,
                event_type="pii_purged",
                extra_data={
                    "phase": "complete",
                    "source": source,
                    "fenced_generation": fenced_generation,
                    "deleted": deleted,
                },
            ):
                raise RuntimeError("PII deletion-success custody failed")
            restored_settings = dict(locked_job.settings_json or {})
            restored_settings.pop("_pii_erasure_fence", None)
            locked_job.status = original_status
            locked_job.last_failure_reason = original_failure_reason
            locked_job.settings_json = restored_settings
            locked_job.save(
                update_fields=["status", "last_failure_reason", "settings_json"]
            )
    except Exception as exc:
        failure = f"job {job.job_id}: database PII cleanup failed ({exc})"
        _record_pii_purge_custody_event(
            purge_scope,
            planned["pii_entities"],
            entity_types,
            job_ids,
            job=job,
            event_type="pii_deletion_failed",
            extra_data={"phase": "database", "source": source, "failure": failure},
        )
        return {
            "ok": False,
            "failures": [failure],
            "deleted": {},
            "sidecars_deleted": sidecars_deleted,
            "redis_deleted": redis_deleted,
        }

    return {
        "ok": True,
        "failures": [],
        "deleted": deleted,
        "sidecars_deleted": sidecars_deleted,
        "redis_deleted": redis_deleted,
    }


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
            help=(
                "Select jobs containing this subject name (case-insensitive); "
                "all job-scoped PII planes are then erased"
            ),
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

    @control_plane_rls_command
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

        from jobs.extraction_models import DocumentChunk, ExtractedFormValue

        # Build one discovery plan across every persisted PII-bearing model.
        # Encrypted fields are intentionally not SQL-searchable, so subject
        # matching streams their decrypted values and records stable job IDs.
        affected_job_ids = set()
        entity_types_found = set()
        pii_target_ids = set()
        extracted_target_ids = set()
        form_target_ids = set()
        chunk_target_ids = set()
        if job_id:
            requested_job = Job.objects.filter(job_id=job_id).first()
            if requested_job is not None:
                affected_job_ids.add(requested_job.job_id)
                pii_targets = PiiEntity.objects.filter(job=requested_job)
                extracted_targets = ExtractedEntity.objects.filter(job=requested_job)
                if entity_type:
                    pii_targets = pii_targets.filter(entity_type=entity_type)
                    extracted_targets = extracted_targets.filter(
                        entity_type=entity_type
                    )
                pii_target_ids.update(pii_targets.values_list("pk", flat=True))
                extracted_target_ids.update(
                    extracted_targets.values_list("pk", flat=True)
                )
                if not entity_type:
                    form_target_ids.update(
                        ExtractedFormValue.objects.filter(
                            job=requested_job
                        ).values_list("pk", flat=True)
                    )
                    chunk_target_ids.update(
                        DocumentChunk.objects.filter(
                            job=requested_job, contains_pii=True
                        ).values_list("pk", flat=True)
                    )

        if subject:
            needle = subject.casefold()
            pii_candidates = PiiEntity.objects.all()
            extracted_candidates = ExtractedEntity.objects.all()
            if entity_type:
                pii_candidates = pii_candidates.filter(entity_type=entity_type)
                extracted_candidates = extracted_candidates.filter(
                    entity_type=entity_type
                )
            for entity in pii_candidates.iterator(chunk_size=500):
                if needle in str(entity.entity_value or "").casefold():
                    affected_job_ids.add(entity.job_id)
                    entity_types_found.add(entity.entity_type)
                    pii_target_ids.add(entity.pk)
            for entity in extracted_candidates.iterator(chunk_size=500):
                if needle in str(entity.entity_text or "").casefold():
                    affected_job_ids.add(entity.job_id)
                    entity_types_found.add(entity.entity_type)
                    extracted_target_ids.add(entity.pk)
            if not entity_type:
                for value in ExtractedFormValue.objects.all().iterator(chunk_size=500):
                    if needle in str(value.field_value or "").casefold():
                        affected_job_ids.add(value.job_id)
                        form_target_ids.add(value.pk)
                for chunk in DocumentChunk.objects.filter(contains_pii=True).iterator(
                    chunk_size=500
                ):
                    if needle in str(chunk.chunk_text or "").casefold():
                        affected_job_ids.add(chunk.job_id)
                        chunk_target_ids.add(chunk.pk)

        if not affected_job_ids:
            self.stdout.write("No matching PII records found.")
            return

        jobs = list(Job.objects.filter(job_id__in=affected_job_ids).order_by("job_id"))
        terminal_statuses = {
            Job.Status.COMPLETED,
            Job.Status.FAILED,
            Job.Status.CANCELLED,
        }
        active = [job for job in jobs if job.status not in terminal_statuses]
        if active:
            raise CommandError(
                "PII purge blocked: affected jobs must be terminal before erasure: "
                + ", ".join(str(job.job_id) for job in active[:5])
            )

        # Subject/type matching is a job selector. Sidecars and cache entries
        # are job-scoped and cannot be rewritten with a provable per-subject
        # contract, so expand the database plan to the same whole-job scope.
        pii_target_ids = set(
            PiiEntity.objects.filter(job_id__in=affected_job_ids).values_list(
                "pk", flat=True
            )
        )
        extracted_target_ids = set(
            ExtractedEntity.objects.filter(job_id__in=affected_job_ids).values_list(
                "pk", flat=True
            )
        )
        form_target_ids = set(
            ExtractedFormValue.objects.filter(job_id__in=affected_job_ids).values_list(
                "pk", flat=True
            )
        )
        chunk_target_ids = set(
            DocumentChunk.objects.filter(
                job_id__in=affected_job_ids,
                contains_pii=True,
            ).values_list("pk", flat=True)
        )

        count = len(pii_target_ids)
        extracted_count = len(extracted_target_ids)
        form_count = len(form_target_ids)
        chunk_count = len(chunk_target_ids)
        for entity in PiiEntity.objects.filter(pk__in=pii_target_ids):
            entity_types_found.add(entity.entity_type)
        for entity in ExtractedEntity.objects.filter(pk__in=extracted_target_ids):
            entity_types_found.add(entity.entity_type)

        # ---- Display summary ----------------------------------------------
        self.stdout.write(
            f"Found {count + extracted_count + form_count + chunk_count} "
            "PII-bearing records to purge:"
        )
        self.stdout.write(f"  Entity types: {', '.join(sorted(entity_types_found))}")
        self.stdout.write(f"  Affected jobs: {len(affected_job_ids)}")

        # ---- Dry run stops here -------------------------------------------
        if dry_run:
            self.stdout.write("DRY RUN: No records deleted.")
            return

        # ---- Confirm gate -------------------------------------------------
        if not confirm:
            self.stdout.write(
                self.style.WARNING("Use --confirm to actually delete these records.")
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
        purge_scope += "; erasure_scope=job-wide"

        # One shared committed saga per terminal job. Retention entrypoints use
        # this same primitive with an unrestricted per-job target set.
        sidecar_cleaned = 0
        redis_cleaned = 0
        deleted_count = 0
        extracted_deleted = 0
        form_values_deleted = 0
        chunks_deleted = 0
        cleanup_failures = []
        for job in jobs:
            result = erase_pii_for_job(
                job,
                purge_scope=purge_scope,
            )
            sidecar_cleaned += result["sidecars_deleted"]
            redis_cleaned += result["redis_deleted"]
            cleanup_failures.extend(result["failures"])
            deleted_count += result["deleted"].get("pii_entities", 0)
            extracted_deleted += result["deleted"].get("extracted_entities", 0)
            form_values_deleted += result["deleted"].get("form_values", 0)
            chunks_deleted += result["deleted"].get("pii_chunks", 0)

        if cleanup_failures:
            raise CommandError(
                "PII purge incomplete; " + "; ".join(cleanup_failures[:5])
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Purged {deleted_count} PII entities, "
                f"{extracted_deleted} extracted entities, "
                f"{form_values_deleted} form values, {chunks_deleted} PII chunks "
                f"({sidecar_cleaned} sidecar dirs removed, "
                f"{redis_cleaned} cache keys cleared)."
            )
        )
