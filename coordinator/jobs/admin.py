import logging
import os
from functools import partial

from django.contrib import admin
from django.db import transaction
from django.utils import timezone
from django.utils.html import format_html
from django_otp.plugins.otp_totp.admin import TOTPDeviceAdmin
from django_otp.plugins.otp_totp.models import TOTPDevice

from .admin_tenant_scope import apply_tenant_scope
from .models import (
    ApiKeyRecord,
    CustodyEvent,
    GlossaryEntry,
    Job,
    PageResult,
    PresignedTaskCapability,
    TaskDispatchOutbox,
    TranslationTenantConfig,
    Worker,
)

logger = logging.getLogger(__name__)

# Register TOTP device management in the default admin site if not already
# registered.  This lets administrators manage TOTP devices for users
# directly through the Django admin interface.
if not admin.site.is_registered(TOTPDevice):
    admin.site.register(TOTPDevice, TOTPDeviceAdmin)

# RC-01: ensure extraction ModelAdmins are registered (jobs.admin is what
# Django autodiscover loads; extraction_admin is not auto-imported otherwise).
from . import extraction_admin as _extraction_admin  # noqa: E402,F401


class PageResultInline(admin.TabularInline):
    model = PageResult
    extra = 0
    can_delete = False
    readonly_fields = [
        "page_num",
        "status",
        "ocr_method",
        "ocr_confidence",
        "text_length",
        "processing_time_ms",
        "worker_hostname",
    ]
    fields = readonly_fields

    def has_add_permission(self, request, obj=None):
        return False


class CustodyEventInline(admin.TabularInline):
    model = CustodyEvent
    extra = 0
    can_delete = False
    readonly_fields = [
        "timestamp",
        "event_type",
        "event_hash_short",
        "chain_finalized",
        "worker_hostname",
    ]
    fields = readonly_fields

    def has_add_permission(self, request, obj=None):
        return False

    @admin.display(description="Hash")
    def event_hash_short(self, obj):
        return obj.event_hash[:12] + "..." if obj.event_hash else "-"


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = [
        "job_id_short",
        "status_badge",
        "priority",
        "source_file_name",
        "progress_bar",
        "detected_language",
        "tenant_id",
        "assigned_worker",
        "created_at",
    ]
    list_filter = ["status", "priority", "detected_language", "tenant_id"]
    search_fields = ["source_file", "job_id"]
    date_hierarchy = "created_at"
    readonly_fields = [
        "job_id",
        "created_at",
        "started_at",
        "completed_at",
        "processing_time_seconds",
        "storage_backend_used",
    ]
    # ``webhook_secret`` is encrypted at rest but is still bearer material
    # after decryption.  ModelAdmin's implicit all-field form would otherwise
    # render it to staff and allow accidental edits/copying.
    exclude = ["webhook_secret"]
    inlines = [PageResultInline, CustodyEventInline]
    actions = ["retry_failed_jobs", "cancel_running_jobs"]

    def get_readonly_fields(self, request, obj=None):
        """Prevent ordinary staff from mass-assigning execution policy.

        ``settings_json`` contains engine/egress/forensic policy and
        ``tenant_id`` is an authorization boundary. Superusers may still use
        the existing admin workflow for controlled break-glass operations.
        """
        if request.user.is_superuser:
            return self.readonly_fields
        field_names = {
            field.name
            for field in self.model._meta.get_fields()
            if getattr(field, "concrete", False)
            and not getattr(field, "many_to_many", False)
        }
        return tuple(sorted(field_names | set(self.readonly_fields)))

    def get_queryset(self, request):
        """RC-01 / C01: under multi-tenancy, refuse unscoped fleet JobAdmin lists.

        Superusers may pass ``?fleet=1`` for explicit cross-tenant listing.
        Non-superuser staff must supply ``tenant_id`` query or ``X-Tenant-Id``.
        """
        qs = super().get_queryset(request)
        return apply_tenant_scope(qs, request, tenant_field="tenant_id")

    def _tenant_on_litigation_hold(self, tenant_id: str) -> bool:
        """R4-RC-04 / PB-A03: refuse mutate when tenant (or unscoped job) is held.

        Empty ``tenant_id`` previously bypassed hold checks (fleet staff residual).
        When any litigation-hold set is configured, unscoped jobs fail closed.
        """
        try:
            import os

            if os.environ.get("LITIGATION_HOLD", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                return True
            from django.conf import settings as dj_settings

            holds = getattr(dj_settings, "LITIGATION_HOLD_TENANTS", None)
            if holds is None:
                raw = os.environ.get("LITIGATION_HOLD_TENANTS", "")
                holds = {t.strip() for t in raw.split(",") if t.strip()}
            if isinstance(holds, str):
                holds = {t.strip() for t in holds.split(",") if t.strip()}
            hold_set = set(holds or ())
            if not hold_set:
                return False
            tid = (tenant_id or "").strip()
            # Unscoped / empty tenant cannot bypass an active hold list.
            if not tid:
                return True
            return tid in hold_set
        except Exception:
            # Fail closed on lookup errors when holds may be active.
            import os

            if os.environ.get("LITIGATION_HOLD_TENANTS", "").strip():
                return True
            return False

    def has_delete_permission(self, request, obj=None):
        if obj is not None and self._tenant_on_litigation_hold(
            getattr(obj, "tenant_id", "") or ""
        ):
            return False
        return super().has_delete_permission(request, obj)

    def delete_model(self, request, obj):
        if self._tenant_on_litigation_hold(getattr(obj, "tenant_id", "") or ""):
            from django.contrib import messages

            messages.error(
                request,
                f"Refusing delete of job {obj.job_id}: tenant on litigation hold "
                "(R4-RC-04).",
            )
            return
        return super().delete_model(request, obj)

    def delete_queryset(self, request, queryset):
        blocked = [
            obj
            for obj in queryset
            if self._tenant_on_litigation_hold(getattr(obj, "tenant_id", "") or "")
        ]
        if blocked:
            from django.contrib import messages

            messages.error(
                request,
                f"Refusing bulk delete: {len(blocked)} job(s) on litigation hold "
                "(R4-RC-04).",
            )
            queryset = queryset.exclude(pk__in=[b.pk for b in blocked])
        return super().delete_queryset(request, queryset)

    @admin.display(description="Job ID")
    def job_id_short(self, obj):
        return str(obj.job_id)[:8]

    @admin.display(description="File")
    def source_file_name(self, obj):
        return os.path.basename(obj.source_file)

    @admin.display(description="Status")
    def status_badge(self, obj):
        colors = {
            "submitted": "#6c757d",
            "ingesting": "#17a2b8",
            "processing": "#007bff",
            "assembling": "#ffc107",
            "completed": "#28a745",
            "failed": "#dc3545",
            "cancelled": "#6c757d",
        }
        color = colors.get(obj.status, "#6c757d")
        return format_html(
            '<span style="background:{}; color:white; padding:2px 8px; '
            'border-radius:3px;">{}</span>',
            color,
            obj.get_status_display(),
        )

    @admin.display(description="Progress")
    def progress_bar(self, obj):
        if obj.total_pages == 0:
            return "-"
        pct = int(obj.pages_completed / obj.total_pages * 100)
        return format_html(
            '<div style="width:100px;background:#eee;border-radius:3px;">'
            '<div style="width:{}%;background:#28a745;height:16px;'
            "border-radius:3px;text-align:center;color:white;font-size:11px;"
            'line-height:16px;">{}/{}</div></div>',
            pct,
            obj.pages_completed,
            obj.total_pages,
        )

    @admin.action(description="Retry failed jobs")
    def retry_failed_jobs(self, request, queryset):
        # S3-A07: litigation hold blocks retry as well as delete.
        allowed_ids = []
        blocked = 0
        for obj in queryset.filter(status="failed"):
            if self._tenant_on_litigation_hold(getattr(obj, "tenant_id", "") or ""):
                blocked += 1
                continue
            allowed_ids.append(obj.pk)
        updated = 0
        if allowed_ids:
            with transaction.atomic():
                retry_jobs = list(
                    queryset.select_for_update().filter(
                        pk__in=allowed_ids, status="failed"
                    )
                )
                retry_ids = [job.pk for job in retry_jobs]
                PageResult.objects.filter(job_id__in=retry_ids).delete()
                from .outbox import enqueue_task_dispatch, publish_task_dispatch_outbox

                for job in retry_jobs:
                    job.status = Job.Status.SUBMITTED
                    job.error_message = ""
                    job.last_failure_reason = ""
                    job.celery_task_id = ""
                    job.completed_at = None
                    job.custody_published_at = None
                    job.started_at = None
                    job.fanout_dispatched = False
                    job.pages_completed = 0
                    job.pages_failed = 0
                    job.dispatch_generation = int(job.dispatch_generation or 0) + 1
                    job.save(
                        update_fields=[
                            "status",
                            "error_message",
                            "last_failure_reason",
                            "celery_task_id",
                            "completed_at",
                            "custody_published_at",
                            "started_at",
                            "fanout_dispatched",
                            "pages_completed",
                            "pages_failed",
                            "dispatch_generation",
                        ]
                    )
                    # Retire every prior-generation contract before creating
                    # the new ingest handoff.  Published rows cannot be pulled
                    # back from the broker, but generation-aware consumers and
                    # relay bookkeeping no longer grant them current authority.
                    TaskDispatchOutbox.objects.filter(job=job).exclude(
                        generation=str(job.dispatch_generation)
                    ).update(
                        status=TaskDispatchOutbox.Status.DEAD,
                        dead_at=timezone.now(),
                        claim_token="",
                        lease_expires_at=None,
                        last_error="superseded by admin retry generation",
                    )
                    PresignedTaskCapability.objects.filter(job=job).delete()
                    kwargs = {}
                    if str(job.tenant_id or "").strip():
                        kwargs["tenant_id"] = str(job.tenant_id).strip()
                    row = enqueue_task_dispatch(
                        job=job,
                        stage="ingest_document",
                        generation=str(job.dispatch_generation),
                        task_name="jobs.tasks.ingest_document",
                        args=[str(job.job_id)],
                        kwargs=kwargs,
                        queue="coordinator",
                        priority=5,
                    )
                    transaction.on_commit(
                        partial(
                            publish_task_dispatch_outbox,
                            dispatch_id=row.dispatch_id,
                        )
                    )
                    updated += 1
        msg = f"{updated} jobs queued for retry."
        if blocked:
            msg += f" Refused {blocked} on litigation hold."
        self.message_user(request, msg)

    @admin.action(description="Cancel running jobs")
    def cancel_running_jobs(self, request, queryset):
        # S3-A07: litigation hold blocks cancel mutate.
        allowed_ids = []
        blocked = 0
        active = queryset.filter(
            status__in=["submitted", "ingesting", "processing", "assembling"]
        )
        for obj in active:
            if self._tenant_on_litigation_hold(getattr(obj, "tenant_id", "") or ""):
                blocked += 1
                continue
            allowed_ids.append(obj.pk)
        updated = 0
        if allowed_ids:
            with transaction.atomic():
                active_rows = queryset.filter(
                    pk__in=allowed_ids,
                    status__in=["submitted", "ingesting", "processing", "assembling"],
                )
                active_ids = list(active_rows.values_list("pk", flat=True))
                updated = active_rows.update(
                    status="cancelled", completed_at=timezone.now()
                )
                TaskDispatchOutbox.objects.filter(
                    job_id__in=active_ids,
                    status__in=[
                        TaskDispatchOutbox.Status.PENDING,
                        TaskDispatchOutbox.Status.FAILED,
                    ],
                ).update(
                    status=TaskDispatchOutbox.Status.DEAD,
                    dead_at=timezone.now(),
                    last_error="job cancelled before dispatch",
                )
                PresignedTaskCapability.objects.filter(job_id__in=active_ids).delete()
        msg = f"{updated} jobs cancelled."
        if blocked:
            msg += f" Refused {blocked} on litigation hold."
        self.message_user(request, msg)


@admin.register(Worker)
class WorkerAdmin(admin.ModelAdmin):
    list_display = [
        "hostname",
        "status_badge",
        "capabilities_display",
        "gpu_info",
        "tasks_completed",
        "tasks_failed",
        "last_heartbeat",
    ]
    list_filter = ["status", "gpu_available"]
    actions = ["drain_workers", "mark_workers_offline", "ping_workers"]

    def _fleet_actions_allowed(self, request) -> bool:
        """PH2-13: drain/offline/ping are fleet control — superuser only under MT/prod.

        Workers have no tenant column; unscoped staff must not run fleet
        mutate actions. Superusers retain control. Lab without multi-tenancy
        keeps prior staff behavior unless DEPLOYMENT_ENV=production.
        """
        user = getattr(request, "user", None)
        if bool(getattr(user, "is_superuser", False)):
            return True
        mt = os.environ.get("ENABLE_MULTITENANCY", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        prod = os.environ.get("DEPLOYMENT_ENV", "").strip().lower() in (
            "production",
            "prod",
        )
        if mt or prod:
            return False
        return bool(getattr(user, "is_staff", False))

    def get_actions(self, request):
        actions = super().get_actions(request)
        if not self._fleet_actions_allowed(request):
            for name in ("drain_workers", "mark_workers_offline", "ping_workers"):
                actions.pop(name, None)
        return actions

    @admin.display(description="Status")
    def status_badge(self, obj):
        colors = {
            "online": "#28a745",
            "busy": "#ffc107",
            "offline": "#dc3545",
            "draining": "#6c757d",
        }
        color = colors.get(obj.status, "#6c757d")
        return format_html(
            '<span style="background:{}; color:white; padding:2px 8px; '
            'border-radius:3px;">{}</span>',
            color,
            obj.get_status_display(),
        )

    @admin.display(description="Capabilities")
    def capabilities_display(self, obj):
        return ", ".join(obj.capabilities) if obj.capabilities else "-"

    @admin.display(description="GPU")
    def gpu_info(self, obj):
        if not obj.gpu_available:
            return "CPU only"
        return f"{obj.gpu_model} ({obj.gpu_vram_mb}MB)"

    @admin.action(description="Drain selected workers (stop accepting new tasks)")
    def drain_workers(self, request, queryset):
        if not self._fleet_actions_allowed(request):
            self.message_user(
                request,
                "Refusing drain: fleet WorkerAdmin actions require superuser "
                "under multi-tenancy/production (PH2-13).",
                level="error",
            )
            return
        online_workers = queryset.filter(
            status__in=[Worker.Status.ONLINE, Worker.Status.BUSY]
        )
        hostnames = list(online_workers.values_list("hostname", flat=True))
        updated = online_workers.update(status=Worker.Status.DRAINING)

        # Tell Celery to cancel queue consumers for these workers
        if hostnames:
            try:
                from coordinator.celery import app

                errors = []
                for worker in Worker.objects.filter(hostname__in=hostnames):
                    for queue_name in worker.queues or []:
                        try:
                            app.control.cancel_consumer(
                                queue_name, destination=[worker.hostname]
                            )
                        except Exception as exc:
                            errors.append(f"{worker.hostname}/{queue_name}: {exc}")
                if errors:
                    self.message_user(
                        request,
                        f"{updated} workers set to draining, "
                        f"but some Celery controls failed: {'; '.join(errors)}",
                        level="warning",
                    )
                    return
            except Exception as exc:
                self.message_user(
                    request,
                    f"{updated} workers set to draining, "
                    f"but Celery control failed: {exc}",
                    level="warning",
                )
                return

        self.message_user(request, f"{updated} workers set to draining.")

    @admin.action(description="Force selected workers offline")
    def mark_workers_offline(self, request, queryset):
        if not self._fleet_actions_allowed(request):
            self.message_user(
                request,
                "Refusing offline: fleet WorkerAdmin actions require superuser "
                "under multi-tenancy/production (PH2-13).",
                level="error",
            )
            return
        updated = queryset.exclude(status=Worker.Status.OFFLINE).update(
            status=Worker.Status.OFFLINE,
            current_task_id="",
        )
        self.message_user(request, f"{updated} workers marked offline.")

    @admin.action(description="Ping selected workers")
    def ping_workers(self, request, queryset):
        if not self._fleet_actions_allowed(request):
            self.message_user(
                request,
                "Refusing ping: fleet WorkerAdmin actions require superuser "
                "under multi-tenancy/production (PH2-13).",
                level="error",
            )
            return
        hostnames = list(queryset.values_list("hostname", flat=True))
        try:
            from coordinator.celery import app

            responses = app.control.ping(destination=hostnames, timeout=5.0)
        except Exception as exc:
            self.message_user(
                request,
                f"Ping failed: {exc}",
                level="error",
            )
            return

        # Responses are usually keyed by Celery node names such as
        # ``cpu-worker@host`` while the Worker table stores the host part.
        # Compare both forms so a healthy worker is not reported dead merely
        # because the control-plane identity has a role prefix.
        selected_hostnames = set(hostnames)
        responsive = set()
        for resp_dict in responses or []:
            for node_name in resp_dict:
                node_name = str(node_name)
                host = node_name.rsplit("@", 1)[-1]
                if node_name in selected_hostnames:
                    responsive.add(node_name)
                elif host in selected_hostnames:
                    responsive.add(host)

        # Update heartbeat for responsive workers
        if responsive:
            Worker.objects.filter(hostname__in=responsive).update(
                last_heartbeat=timezone.now()
            )

        alive = len(responsive & set(hostnames))
        dead = queryset.count() - alive

        self.message_user(
            request,
            f"Ping results: {alive} responsive, {dead} unresponsive.",
        )


@admin.register(PageResult)
class PageResultAdmin(admin.ModelAdmin):
    list_display = [
        "job",
        "page_num",
        "status",
        "ocr_method",
        "ocr_confidence",
        "worker_hostname",
        "processing_time_ms",
    ]
    list_filter = ["status", "ocr_method"]
    raw_id_fields = ["job"]

    def get_queryset(self, request):
        """RC-01: PageResult lists scope via job.tenant_id under multi-tenancy."""
        qs = super().get_queryset(request)
        return apply_tenant_scope(qs, request, via_job=True)


@admin.register(CustodyEvent)
class CustodyEventAdmin(admin.ModelAdmin):
    list_display = [
        "document_id",
        "event_type",
        "timestamp",
        "worker_hostname",
        "event_hash_short",
        "chain_finalized",
    ]
    list_filter = ["event_type", "chain_finalized"]
    search_fields = ["document_id"]
    raw_id_fields = ["job"]
    readonly_fields = [field.name for field in CustodyEvent._meta.concrete_fields]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def get_queryset(self, request):
        """RC-01: CustodyEvent lists scope via job.tenant_id under multi-tenancy."""
        qs = super().get_queryset(request)
        return apply_tenant_scope(qs, request, via_job=True)

    @admin.display(description="Hash")
    def event_hash_short(self, obj):
        return obj.event_hash[:12] + "..." if obj.event_hash else "-"


@admin.register(ApiKeyRecord)
class ApiKeyRecordAdmin(admin.ModelAdmin):
    list_display = [
        "key_id_short",
        "description",
        "is_active",
        "use_count",
        "last_used_at",
        "created_at",
    ]
    list_filter = ["is_active"]
    search_fields = ["key_id", "description"]
    readonly_fields = ["key_id", "created_at", "last_used_at", "use_count"]

    @admin.display(description="Key ID")
    def key_id_short(self, obj):
        return obj.key_id[:12] + "..." if len(obj.key_id) > 12 else obj.key_id


@admin.register(TranslationTenantConfig)
class TranslationTenantConfigAdmin(admin.ModelAdmin):
    list_display = [
        "tenant_id",
        "allow_nc_licensed",
        "require_certified",
        "residency_region",
        "default_quality_tier",
        "updated_at",
    ]
    list_filter = [
        "allow_nc_licensed",
        "require_certified",
        "default_quality_tier",
    ]
    search_fields = ["tenant_id", "residency_region"]
    readonly_fields = ["created_at", "updated_at"]

    def get_queryset(self, request):
        """RC-01: translation tenant config is tenant-owned."""
        qs = super().get_queryset(request)
        return apply_tenant_scope(qs, request, tenant_field="tenant_id")


@admin.register(GlossaryEntry)
class GlossaryEntryAdmin(admin.ModelAdmin):
    list_display = [
        "tenant_id",
        "source_term",
        "target_term",
        "source_lang",
        "target_lang",
        "priority",
        "case_sensitive",
        "is_regex",
        "updated_at",
    ]
    list_filter = ["source_lang", "target_lang", "case_sensitive", "is_regex"]
    search_fields = ["tenant_id", "source_term", "target_term"]
    readonly_fields = ["created_at", "updated_at"]

    def get_queryset(self, request):
        """RC-01: glossary rows are tenant-owned."""
        qs = super().get_queryset(request)
        return apply_tenant_scope(qs, request, tenant_field="tenant_id")
