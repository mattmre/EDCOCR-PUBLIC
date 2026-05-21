import logging
import os

from django.contrib import admin
from django.utils import timezone
from django.utils.html import format_html
from django_otp.plugins.otp_totp.admin import TOTPDeviceAdmin
from django_otp.plugins.otp_totp.models import TOTPDevice

from .models import (
    ApiKeyRecord,
    CustodyEvent,
    GlossaryEntry,
    Job,
    PageResult,
    TranslationTenantConfig,
    Worker,
)

logger = logging.getLogger(__name__)

# Register TOTP device management in the default admin site if not already
# registered.  This lets administrators manage TOTP devices for users
# directly through the Django admin interface.
if not admin.site.is_registered(TOTPDevice):
    admin.site.register(TOTPDevice, TOTPDeviceAdmin)


class PageResultInline(admin.TabularInline):
    model = PageResult
    extra = 0
    can_delete = False
    readonly_fields = [
        'page_num', 'status', 'ocr_method', 'ocr_confidence',
        'text_length', 'processing_time_ms', 'worker_hostname',
    ]
    fields = readonly_fields

    def has_add_permission(self, request, obj=None):
        return False


class CustodyEventInline(admin.TabularInline):
    model = CustodyEvent
    extra = 0
    can_delete = False
    readonly_fields = [
        'timestamp', 'event_type', 'event_hash_short', 'chain_finalized',
        'worker_hostname',
    ]
    fields = readonly_fields

    def has_add_permission(self, request, obj=None):
        return False

    @admin.display(description='Hash')
    def event_hash_short(self, obj):
        return obj.event_hash[:12] + '...' if obj.event_hash else '-'


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = [
        'job_id_short', 'status_badge', 'priority', 'source_file_name',
        'progress_bar', 'detected_language', 'tenant_id', 'assigned_worker',
        'created_at',
    ]
    list_filter = ['status', 'priority', 'detected_language', 'tenant_id']
    search_fields = ['source_file', 'job_id']
    date_hierarchy = 'created_at'
    readonly_fields = [
        'job_id', 'created_at', 'started_at', 'completed_at',
        'processing_time_seconds', 'storage_backend_used',
    ]
    inlines = [PageResultInline, CustodyEventInline]
    actions = ['retry_failed_jobs', 'cancel_running_jobs']

    @admin.display(description='Job ID')
    def job_id_short(self, obj):
        return str(obj.job_id)[:8]

    @admin.display(description='File')
    def source_file_name(self, obj):
        return os.path.basename(obj.source_file)

    @admin.display(description='Status')
    def status_badge(self, obj):
        colors = {
            'submitted': '#6c757d',
            'ingesting': '#17a2b8',
            'processing': '#007bff',
            'assembling': '#ffc107',
            'completed': '#28a745',
            'failed': '#dc3545',
            'cancelled': '#6c757d',
        }
        color = colors.get(obj.status, '#6c757d')
        return format_html(
            '<span style="background:{}; color:white; padding:2px 8px; '
            'border-radius:3px;">{}</span>',
            color,
            obj.get_status_display(),
        )

    @admin.display(description='Progress')
    def progress_bar(self, obj):
        if obj.total_pages == 0:
            return '-'
        pct = int(obj.pages_completed / obj.total_pages * 100)
        return format_html(
            '<div style="width:100px;background:#eee;border-radius:3px;">'
            '<div style="width:{}%;background:#28a745;height:16px;'
            'border-radius:3px;text-align:center;color:white;font-size:11px;'
            'line-height:16px;">{}/{}</div></div>',
            pct,
            obj.pages_completed,
            obj.total_pages,
        )

    @admin.action(description='Retry failed jobs')
    def retry_failed_jobs(self, request, queryset):
        updated = queryset.filter(status='failed').update(
            status='submitted', error_message=''
        )
        self.message_user(request, f'{updated} jobs queued for retry.')

    @admin.action(description='Cancel running jobs')
    def cancel_running_jobs(self, request, queryset):
        updated = queryset.filter(
            status__in=['submitted', 'ingesting', 'processing', 'assembling']
        ).update(status='cancelled')
        self.message_user(request, f'{updated} jobs cancelled.')


@admin.register(Worker)
class WorkerAdmin(admin.ModelAdmin):
    list_display = [
        'hostname', 'status_badge', 'capabilities_display', 'gpu_info',
        'tasks_completed', 'tasks_failed', 'last_heartbeat',
    ]
    list_filter = ['status', 'gpu_available']
    actions = ['drain_workers', 'mark_workers_offline', 'ping_workers']

    @admin.display(description='Status')
    def status_badge(self, obj):
        colors = {
            'online': '#28a745',
            'busy': '#ffc107',
            'offline': '#dc3545',
            'draining': '#6c757d',
        }
        color = colors.get(obj.status, '#6c757d')
        return format_html(
            '<span style="background:{}; color:white; padding:2px 8px; '
            'border-radius:3px;">{}</span>',
            color,
            obj.get_status_display(),
        )

    @admin.display(description='Capabilities')
    def capabilities_display(self, obj):
        return ', '.join(obj.capabilities) if obj.capabilities else '-'

    @admin.display(description='GPU')
    def gpu_info(self, obj):
        if not obj.gpu_available:
            return 'CPU only'
        return f'{obj.gpu_model} ({obj.gpu_vram_mb}MB)'

    @admin.action(description='Drain selected workers (stop accepting new tasks)')
    def drain_workers(self, request, queryset):
        online_workers = queryset.filter(
            status__in=[Worker.Status.ONLINE, Worker.Status.BUSY]
        )
        hostnames = list(online_workers.values_list('hostname', flat=True))
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
                            errors.append(f'{worker.hostname}/{queue_name}: {exc}')
                if errors:
                    self.message_user(
                        request,
                        f'{updated} workers set to draining, '
                        f'but some Celery controls failed: {"; ".join(errors)}',
                        level='warning',
                    )
                    return
            except Exception as exc:
                self.message_user(
                    request,
                    f'{updated} workers set to draining, '
                    f'but Celery control failed: {exc}',
                    level='warning',
                )
                return

        self.message_user(request, f'{updated} workers set to draining.')

    @admin.action(description='Force selected workers offline')
    def mark_workers_offline(self, request, queryset):
        updated = queryset.exclude(
            status=Worker.Status.OFFLINE
        ).update(
            status=Worker.Status.OFFLINE,
            current_task_id='',
        )
        self.message_user(request, f'{updated} workers marked offline.')

    @admin.action(description='Ping selected workers')
    def ping_workers(self, request, queryset):
        hostnames = list(queryset.values_list('hostname', flat=True))
        try:
            from coordinator.celery import app
            responses = app.control.ping(
                destination=hostnames, timeout=5.0
            )
        except Exception as exc:
            self.message_user(
                request,
                f'Ping failed: {exc}',
                level='error',
            )
            return

        # responses is a list of dicts: [{'worker@host': {'ok': 'pong'}}]
        responsive = set()
        for resp_dict in responses or []:
            responsive.update(resp_dict.keys())

        # Update heartbeat for responsive workers
        if responsive:
            Worker.objects.filter(hostname__in=responsive).update(
                last_heartbeat=timezone.now()
            )

        alive = len(responsive & set(hostnames))
        dead = queryset.count() - alive

        self.message_user(
            request,
            f'Ping results: {alive} responsive, {dead} unresponsive.',
        )


@admin.register(PageResult)
class PageResultAdmin(admin.ModelAdmin):
    list_display = [
        'job', 'page_num', 'status', 'ocr_method', 'ocr_confidence',
        'worker_hostname', 'processing_time_ms',
    ]
    list_filter = ['status', 'ocr_method']
    raw_id_fields = ['job']


@admin.register(CustodyEvent)
class CustodyEventAdmin(admin.ModelAdmin):
    list_display = [
        'document_id', 'event_type', 'timestamp', 'worker_hostname',
        'event_hash_short', 'chain_finalized',
    ]
    list_filter = ['event_type', 'chain_finalized']
    search_fields = ['document_id']
    raw_id_fields = ['job']

    @admin.display(description='Hash')
    def event_hash_short(self, obj):
        return obj.event_hash[:12] + '...' if obj.event_hash else '-'


@admin.register(ApiKeyRecord)
class ApiKeyRecordAdmin(admin.ModelAdmin):
    list_display = [
        'key_id_short', 'description', 'is_active', 'use_count',
        'last_used_at', 'created_at',
    ]
    list_filter = ['is_active']
    search_fields = ['key_id', 'description']
    readonly_fields = ['key_id', 'created_at', 'last_used_at', 'use_count']

    @admin.display(description='Key ID')
    def key_id_short(self, obj):
        return obj.key_id[:12] + '...' if len(obj.key_id) > 12 else obj.key_id


@admin.register(TranslationTenantConfig)
class TranslationTenantConfigAdmin(admin.ModelAdmin):
    list_display = [
        'tenant_id', 'allow_nc_licensed', 'require_certified',
        'default_quality_tier', 'updated_at',
    ]
    list_filter = ['allow_nc_licensed', 'require_certified', 'default_quality_tier']
    search_fields = ['tenant_id']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(GlossaryEntry)
class GlossaryEntryAdmin(admin.ModelAdmin):
    list_display = [
        'tenant_id', 'source_term', 'target_term',
        'source_lang', 'target_lang', 'priority',
        'case_sensitive', 'is_regex', 'updated_at',
    ]
    list_filter = ['source_lang', 'target_lang', 'case_sensitive', 'is_regex']
    search_fields = ['tenant_id', 'source_term', 'target_term']
    readonly_fields = ['created_at', 'updated_at']
