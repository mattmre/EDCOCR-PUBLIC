"""Management command to purge all data for a specific tenant.

Deletes Jobs, PageResults, PiiEntities, and optionally CustodyEvents
and NFS/S3/EXPORT/TEMP filesystem trees for a given tenant_id.  Emits a
CustodyEvent audit record BEFORE deletion so the purge is recorded in the
chain.

R4-RC-16 dual-plane: confirmed purge defaults to filesystem wipe using the
same ``ocr_distributed.tenant_purge_fs`` helper as the FastAPI
``purge_tenant_data`` path (EXPORT / TEMP / resume doc_id trees, not only
NFS job dirs).

Respects LITIGATION_HOLD to block all purge operations.

Usage:
    python manage.py purge_tenant --tenant-id acme-corp --dry-run
    python manage.py purge_tenant --tenant-id acme-corp --confirm
    python manage.py purge_tenant --tenant-id acme-corp --confirm --include-custody
    python manage.py purge_tenant --tenant-id acme-corp --confirm --skip-output
"""

import getpass
import hashlib
import logging
import os

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import models, transaction
from django.utils import timezone

from jobs.litigation_hold import check_litigation_hold
from jobs.models import (
    CustodyEvent,
    Job,
    PageResult,
    PiiEntity,
    TaskDispatchOutbox,
)
from jobs.rls import tenant_rls_command
from ocr_distributed.tenant_purge_fs import (
    purge_fs_job_artifacts,
    remove_path_no_follow,
    safe_nfs_job_path,
)

logger = logging.getLogger(__name__)


def _tenant_purge_document_id(tenant_id):
    tenant_digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:24]
    return f"tenant-purge:{tenant_digest}"


def _record_tenant_purge_custody_event(
    tenant_id,
    job_count,
    page_count,
    pii_count,
    custody_count,
    *,
    event_type,
    extra=None,
):
    """Create a CustodyEvent recording a tenant purge action.

    Purge custody is deliberately jobless so deleting the tenant's job rows
    cannot erase the audit record that proves the destructive operation.
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
        **(extra or {}),
    }
    from jobs.tasks import _record_control_plane_custody_event

    return _record_control_plane_custody_event(
        tenant_id=tenant_id,
        document_id=_tenant_purge_document_id(tenant_id),
        event_type=event_type,
        data=data,
        worker_hostname="purge-tenant-command",
    )


def _optional_tenant_model(name):
    """Return an independently tenant-owned jobs model when available.

    The command is also imported by lightweight API/test environments that
    only provide the core Job models, so these imports stay lazy.  Translation
    batches/config/glossaries are not foreign-key children of Job and must be
    explicitly included in a coordinator purge.
    """
    try:
        from jobs import models as job_models
    except Exception:
        return None
    return getattr(job_models, name, None)


def _tenant_model_count(model, tenant_id):
    """Count rows for a model with a direct tenant_id column."""
    if model is None:
        return 0
    try:
        return model.objects.filter(tenant_id=tenant_id).count()
    except (AttributeError, TypeError):
        return 0


def _tenant_model_delete(model, tenant_id):
    """Delete rows for a model with a direct tenant_id column."""
    if model is None:
        return 0
    try:
        deleted, _ = model.objects.filter(tenant_id=tenant_id).delete()
        return deleted
    except (AttributeError, TypeError):
        return 0


def _safe_nfs_job_path(raw_path, nfs_root, *, job_id=None):
    """Return a root-contained NFS path suitable for direct deletion.

    The shared purge helper validates database paths too, but this management
    command historically had a second direct ``rmtree`` loop.  Keep that
    legacy cleanup only for a resolved descendant of NFS_ROOT; a tampered
    database row must never become an unrestricted deletion authority.
    """
    return safe_nfs_job_path(raw_path, nfs_root, job_id=job_id)


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
            default=None,
            help=(
                "Deprecated alias: filesystem purge is ON by default for "
                "--confirm (R4-RC-16 dual-plane). Prefer omitting this flag."
            ),
        )
        parser.add_argument(
            "--skip-output",
            action="store_true",
            help=(
                "Skip filesystem / NFS / S3 wipe (DB-only). Not recommended; "
                "leaves EXPORT/TEMP PII on disk."
            ),
        )

    @tenant_rls_command
    def handle(self, *args, **options):
        # ---- Litigation hold gate -----------------------------------------
        if check_litigation_hold(self):
            return

        tenant_id = options["tenant_id"]
        dry_run = options["dry_run"]
        confirm = options["confirm"]
        include_custody = options["include_custody"]
        # R4-RC-16: confirmed purge defaults to FS wipe (parity with API plane).
        # --skip-output is the opt-out. Legacy --include-output=False (tests /
        # old callers) still means skip FS wipe.
        skip_output = bool(options.get("skip_output"))
        legacy_include = options.get("include_output")
        if legacy_include is False:
            skip_output = True
        elif legacy_include is True:
            skip_output = False
        include_output = not skip_output

        if not tenant_id or not tenant_id.strip():
            raise CommandError("--tenant-id must not be empty.")

        # ---- Query tenant jobs --------------------------------------------
        tenant_jobs = Job.objects.filter(tenant_id=tenant_id)
        job_count = tenant_jobs.count()
        from jobs.tasks import _get_backend_for_job

        job_backends = {}
        backend_failures = []
        for job in tenant_jobs:
            try:
                job_backends[job.job_id] = _get_backend_for_job(job)
            except Exception as exc:
                backend_failures.append(f"job {job.job_id}: {exc}")
        if backend_failures:
            raise CommandError(
                "Tenant purge blocked: persisted storage binding unavailable; "
                + "; ".join(backend_failures[:5])
            )

        # These models are independent of Job and therefore are not removed
        # by the Job delete cascade (translation batches cascade only to their
        # inputs).  Count them before the confirmation gate for an honest
        # dry-run impact report.
        batch_model = _optional_tenant_model("BatchTranslationJob")
        config_model = _optional_tenant_model("TranslationTenantConfig")
        glossary_model = _optional_tenant_model("GlossaryEntry")
        translation_batch_count = _tenant_model_count(batch_model, tenant_id)
        translation_config_count = _tenant_model_count(config_model, tenant_id)
        glossary_count = _tenant_model_count(glossary_model, tenant_id)
        custody_scope = models.Q(tenant_id=tenant_id) | models.Q(
            job__tenant_id=tenant_id
        )
        custody_count = (
            CustodyEvent.objects.filter(custody_scope)
            .exclude(document_id=_tenant_purge_document_id(tenant_id))
            .count()
        )

        if (
            job_count == 0
            and translation_batch_count == 0
            and translation_config_count == 0
            and glossary_count == 0
            and not (include_custody and custody_count)
        ):
            self.stdout.write(
                f"No jobs found or independent tenant rows found for '{tenant_id}'."
            )
            return

        # ---- Collect impact summary ---------------------------------------
        # Keep tenant scoping in SQL.  Materializing every job UUID into an
        # IN-list made a large-tenant purge both memory-heavy and vulnerable
        # to database parameter limits.
        page_count = PageResult.objects.filter(job__tenant_id=tenant_id).count()
        pii_count = PiiEntity.objects.filter(job__tenant_id=tenant_id).count()

        # Estimate NFS storage
        nfs_dirs = []
        nfs_root = getattr(settings, "NFS_ROOT", None)
        for job in tenant_jobs:
            if job_backends.get(job.job_id, None) is None or (
                job_backends[job.job_id].backend_name != "nfs"
            ):
                continue
            safe_job_path = _safe_nfs_job_path(
                getattr(job, "nfs_job_path", None),
                nfs_root,
                job_id=str(job.job_id),
            )
            if safe_job_path and safe_job_path.is_dir():
                nfs_dirs.append(str(safe_job_path))
            elif nfs_root:
                candidate = _safe_nfs_job_path(
                    os.path.join(nfs_root, "jobs", str(job.job_id)),
                    nfs_root,
                    job_id=str(job.job_id),
                )
                if candidate and candidate.is_dir():
                    nfs_dirs.append(str(candidate))

        # ---- Display impact summary ---------------------------------------
        self.stdout.write(f"Tenant purge impact for '{tenant_id}':")
        self.stdout.write(f"  Jobs: {job_count}")
        self.stdout.write(f"  Pages: {page_count}")
        self.stdout.write(f"  PII entities: {pii_count}")
        self.stdout.write(f"  Custody events: {custody_count}")
        self.stdout.write(f"  Translation batches: {translation_batch_count}")
        self.stdout.write(f"  Translation tenant configs: {translation_config_count}")
        self.stdout.write(f"  Glossary entries: {glossary_count}")
        self.stdout.write(f"  NFS directories: {len(nfs_dirs)}")
        self.stdout.write(
            f"  Filesystem wipe: {'ON (default)' if include_output else 'SKIPPED (--skip-output)'}"
        )

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

        # Commit a tenant-wide cancellation/generation fence and START custody
        # before deleting bytes.  Terminal FAILED rows are fenced too, so an
        # admin retry cannot reactivate a job during the external-I/O window.
        with transaction.atomic():
            fenced_jobs = Job.objects.select_for_update().filter(tenant_id=tenant_id)
            fenced_ids = list(fenced_jobs.values_list("pk", flat=True))
            # Admission and stage retries require the parent Job lock. While
            # holding it, refuse to cascade away a live raster's cleanup and
            # byte-reservation authority. A remote PUT can finish after a
            # storage-prefix sweep; deleting its stage would orphan both.
            stage_model = _optional_tenant_model("PageOCRStage")
            if stage_model is not None and stage_model.objects.filter(
                job_id__in=fenced_ids
            ).filter(
                ~models.Q(status__in=("done", "obsolete", "failed"))
                | models.Q(reserved_bytes__gt=0)
                | ~models.Q(lease_token="")
                | models.Q(lease_expires_at__isnull=False)
                | models.Q(publish_safe_after__isnull=False)
            ).exists():
                raise CommandError(
                    "Tenant purge blocked: staged OCR writers or cleanup are "
                    "not drained. Stop admission, drain or cancel the jobs, "
                    "run stage recovery, then retry; preserve stage rows and "
                    "byte reservations."
                )
            fenced_jobs.update(
                status=Job.Status.CANCELLED,
                completed_at=timezone.now(),
                last_failure_reason="tenant_purge_fence",
                celery_task_id="",
                dispatch_generation=models.F("dispatch_generation") + 1,
            )
            TaskDispatchOutbox.objects.filter(
                job_id__in=fenced_ids,
                status__in=[
                    TaskDispatchOutbox.Status.PENDING,
                    TaskDispatchOutbox.Status.FAILED,
                ],
            ).update(
                status=TaskDispatchOutbox.Status.DEAD,
                dead_at=timezone.now(),
                last_error="tenant purge fenced dispatch",
            )
            capability_model = _optional_tenant_model("PresignedTaskCapability")
            if capability_model is not None:
                capability_model.objects.filter(job_id__in=fenced_ids).delete()
            try:
                _record_tenant_purge_custody_event(
                    tenant_id,
                    job_count,
                    page_count,
                    pii_count,
                    custody_count,
                    event_type="tenant_purge_started",
                    extra={"filesystem_wipe": include_output},
                )
            except Exception as exc:
                raise CommandError(
                    "Tenant purge blocked: start custody event could not be recorded"
                ) from exc

        # ---- FS wipe (shared with API plane) + NFS paths -------------------
        fs_cleaned = 0
        nfs_cleaned = 0
        cleanup_failures = []
        if include_output:
            for job in tenant_jobs:
                backend = job_backends.get(job.job_id)
                if backend is None or backend.backend_name != "nfs":
                    continue
                try:
                    raw_nfs_path = getattr(job, "nfs_job_path", None) or None
                    if (
                        raw_nfs_path
                        and _safe_nfs_job_path(
                            raw_nfs_path,
                            nfs_root,
                            job_id=str(job.job_id),
                        )
                        is None
                    ):
                        cleanup_failures.append(
                            f"filesystem job {job.job_id}: untrusted nfs_job_path"
                        )
                        continue
                    fs_cleaned += purge_fs_job_artifacts(
                        job_id=str(job.job_id),
                        tenant_id=tenant_id,
                        source_file=getattr(job, "source_file", None) or None,
                        nfs_job_path=raw_nfs_path,
                        output_bases=(
                            [getattr(settings, "NFS_ROOT", "")]
                            if getattr(settings, "NFS_ROOT", None)
                            else None
                        ),
                        extra_dirs=(
                            [os.path.join(nfs_root, "jobs", str(job.job_id))]
                            if nfs_root
                            else None
                        ),
                        strict=True,
                    )
                    _record_tenant_purge_custody_event(
                        tenant_id,
                        job_count,
                        page_count,
                        pii_count,
                        custody_count,
                        event_type="tenant_purge_job_storage_deleted",
                        extra={"job_id": str(job.job_id), "backend": "nfs"},
                    )
                except Exception as exc:
                    logger.exception("Failed shared FS purge for job %s", job.job_id)
                    cleanup_failures.append(f"filesystem job {job.job_id}: {exc}")
            for nfs_dir in nfs_dirs:
                try:
                    if os.path.lexists(nfs_dir):
                        if remove_path_no_follow(nfs_dir):
                            nfs_cleaned += 1
                        else:
                            cleanup_failures.append(f"NFS directory {nfs_dir} remains")
                except Exception as exc:
                    logger.exception("Failed to remove NFS dir: %s", nfs_dir)
                    cleanup_failures.append(f"NFS directory {nfs_dir}: {exc}")

        # ---- Remove S3 objects using each job's persisted binding ----------
        s3_cleaned = 0
        if include_output:
            for job in tenant_jobs:
                backend = job_backends.get(job.job_id)
                if backend is None or backend.backend_name != "s3":
                    continue
                prefix = f"jobs/{job.job_id}/"
                try:
                    keys = backend.list_objects(prefix)
                    if not keys:
                        continue
                    deleted = backend.delete_many(keys)
                    remaining = backend.list_objects(prefix)
                    s3_cleaned += deleted
                    if deleted != len(keys) or remaining:
                        cleanup_failures.append(
                            f"S3 job {job.job_id}: deleted {deleted} of {len(keys)} "
                            f"objects; remaining={len(remaining)}"
                        )
                    else:
                        _record_tenant_purge_custody_event(
                            tenant_id,
                            job_count,
                            page_count,
                            pii_count,
                            custody_count,
                            event_type="tenant_purge_job_storage_deleted",
                            extra={
                                "job_id": str(job.job_id),
                                "backend": "s3",
                                "objects_deleted": deleted,
                            },
                        )
                except Exception as exc:
                    logger.exception(
                        "Failed to clean S3 objects for job %s", job.job_id
                    )
                    cleanup_failures.append(f"S3 job {job.job_id}: {exc}")
            if s3_cleaned:
                self.stdout.write(
                    f"  Removed {s3_cleaned} S3 objects for tenant {tenant_id}"
                )

        if cleanup_failures:
            # Stop before deleting database rows.  A successful exit must not
            # claim a complete purge when an output data plane remains.
            raise CommandError(
                "Tenant purge blocked: output cleanup failed; "
                + "; ".join(cleanup_failures[:5])
            )

        # Every locally transactional data plane and completion custody commit
        # together.  A DB/FK/custody error leaves all SQL rows retryable rather
        # than partially erasing the tenant after external storage was swept.
        with transaction.atomic():
            locked_jobs = Job.objects.select_for_update().filter(tenant_id=tenant_id)
            if locked_jobs.exclude(status=Job.Status.CANCELLED).exists():
                raise CommandError(
                    "Tenant purge fence changed before database deletion"
                )

            from jobs.artifact_retirement import retain_cleanup_authority

            retired_prefixes = 0
            if include_output:
                for job in locked_jobs:
                    retired_prefixes += retain_cleanup_authority(job, job_backends[job.pk])

            pii_deleted, _ = PiiEntity.objects.filter(job__tenant_id=tenant_id).delete()
            pages_deleted, _ = PageResult.objects.filter(
                job__tenant_id=tenant_id
            ).delete()

            custody_deleted = 0
            if include_custody:
                # The purge's own durable START/SUCCESS chain is immutable
                # audit evidence, not tenant content selected for erasure.
                cd_qs = CustodyEvent.objects.filter(custody_scope).exclude(
                    document_id=_tenant_purge_document_id(tenant_id)
                )
                custody_deleted, _ = cd_qs.delete()

            translation_batches_deleted = _tenant_model_delete(batch_model, tenant_id)
            translation_configs_deleted = _tenant_model_delete(config_model, tenant_id)
            glossary_deleted = _tenant_model_delete(glossary_model, tenant_id)
            jobs_deleted, _ = locked_jobs.delete()

            _record_tenant_purge_custody_event(
                tenant_id,
                job_count,
                page_count,
                pii_count,
                custody_count,
                event_type="tenant_purged",
                extra={
                    "jobs_deleted": jobs_deleted,
                    "pages_deleted": pages_deleted,
                    "pii_deleted": pii_deleted,
                    "late_s3_cleanup_prefixes": retired_prefixes,
                    "storage_scope": "initial visible-object sweep; late S3 cleanup continues",
                },
            )

        if retired_prefixes:
            self.stdout.write(
                f"Initial S3 sweep complete; {retired_prefixes} retired job prefixes "
                "remain under periodic cleanup for late uploads. This does not "
                "certify historical versions or immediate physical erasure."
            )

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
            if hasattr(monitor, "_windows") and tenant_id in monitor._windows:
                del monitor._windows[tenant_id]
                self.stdout.write(
                    f"  Reset SLA monitoring windows for tenant {tenant_id}"
                )
        except Exception:
            # SLA monitor not available in coordinator context
            pass

        # ---- Print completion summary -------------------------------------
        parts = [
            f"Purged tenant '{tenant_id}': ",
            f"{jobs_deleted} jobs, {pages_deleted} pages, ",
            f"{pii_deleted} PII entities deleted",
        ]
        parts.append(
            f", {translation_batches_deleted} translation batches, "
            f"{translation_configs_deleted} translation configs, "
            f"{glossary_deleted} glossary entries deleted"
        )
        if include_custody:
            parts.append(f", {custody_deleted} custody events deleted")
        if include_output:
            parts.append(
                f", {fs_cleaned} shared-fs paths + {nfs_cleaned} NFS dirs removed"
            )
            if s3_cleaned:
                parts.append(f", {s3_cleaned} S3 objects removed")
        else:
            parts.append(" (filesystem wipe skipped)")
        parts.append(".")
        self.stdout.write(self.style.SUCCESS("".join(parts)))
