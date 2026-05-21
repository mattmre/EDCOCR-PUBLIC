import uuid

from django.db import models
from django.utils import timezone

from .encrypted_field import EncryptedTextField


class Job(models.Model):
    """Represents an OCR processing job."""

    class Status(models.TextChoices):
        SUBMITTED = 'submitted', 'Submitted'
        INGESTING = 'ingesting', 'Ingesting'
        PROCESSING = 'processing', 'Processing'
        ASSEMBLING = 'assembling', 'Assembling'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'
        CANCELLED = 'cancelled', 'Cancelled'

    class Priority(models.TextChoices):
        URGENT = 'urgent', 'Urgent'
        NORMAL = 'normal', 'Normal'
        LOW = 'low', 'Low'

    job_id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.SUBMITTED,
        db_index=True,
    )
    priority = models.CharField(
        max_length=10,
        choices=Priority.choices,
        default=Priority.NORMAL,
    )

    # Source info
    source_file = models.CharField(max_length=1024)
    source_hash = models.CharField(max_length=64, blank=True, default='')
    source_type = models.CharField(max_length=10, blank=True, default='')
    detected_language = models.CharField(max_length=20, blank=True, default='en')

    # Progress
    total_pages = models.IntegerField(default=0)
    pages_completed = models.IntegerField(default=0)
    pages_failed = models.IntegerField(default=0)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    @property
    def processing_time_seconds(self):
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    # Configuration
    settings_json = models.JSONField(default=dict, blank=True)
    nfs_job_path = models.CharField(max_length=1024, blank=True, default='')
    storage_backend_used = models.CharField(
        max_length=10,
        blank=True,
        default='',
        help_text='Storage backend locked at ingest time (nfs or s3)',
    )

    # Results
    result_summary = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True, default='')

    # Tenant tracking (optional, empty string means no tenant)
    tenant_id = models.CharField(max_length=128, blank=True, default='', db_index=True)

    # Cross-reference to API server (nullable — only set when job originates from API)
    api_job_id = models.CharField(max_length=64, blank=True, default='', db_index=True)

    # Celery tracking
    celery_task_id = models.CharField(max_length=255, blank=True, default='')
    assigned_worker = models.CharField(max_length=255, blank=True, default='')

    # Federation tracking (Plan C Phase 1, item C5 -- failover)
    # ``assigned_cluster`` records which federation peer is currently
    # responsible for this job; the failover engine uses it to identify
    # in-flight work that needs to be re-dispatched when a peer goes
    # unhealthy. Empty string means "local cluster" / "not federated".
    assigned_cluster = models.CharField(
        max_length=128, blank=True, default='', db_index=True,
    )
    # ``last_failure_reason`` is a short, machine-readable tag explaining
    # the last failure observed for this job (e.g. ``cluster_unhealthy``,
    # ``s3_download_error``).  Operators read it from the dashboard; the
    # field is purely informational and never user-facing.
    last_failure_reason = models.CharField(
        max_length=64, blank=True, default='',
    )

    # Webhook
    webhook_url = models.URLField(max_length=2048, blank=True, default='')
    webhook_secret = models.CharField(max_length=255, blank=True, default='')
    webhook_status = models.CharField(max_length=20, blank=True, default='')

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Job {self.job_id} ({self.status})"


class Worker(models.Model):
    """Represents a Celery worker node."""

    class Status(models.TextChoices):
        ONLINE = 'online', 'Online'
        BUSY = 'busy', 'Busy'
        OFFLINE = 'offline', 'Offline'
        DRAINING = 'draining', 'Draining'

    hostname = models.CharField(max_length=255, primary_key=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.OFFLINE,
    )
    queues = models.JSONField(default=list)
    capabilities = models.JSONField(default=list)
    concurrency = models.IntegerField(default=4)

    # Hardware
    gpu_available = models.BooleanField(default=False)
    gpu_model = models.CharField(max_length=255, blank=True, default='')
    gpu_vram_mb = models.IntegerField(default=0)
    gpu_index = models.IntegerField(null=True, blank=True, default=None)
    cpu_cores = models.IntegerField(default=0)
    ram_mb = models.IntegerField(default=0)

    # Status tracking
    last_heartbeat = models.DateTimeField(null=True, blank=True)
    tasks_completed = models.IntegerField(default=0)
    tasks_failed = models.IntegerField(default=0)
    current_task_id = models.CharField(max_length=255, blank=True, default='')

    # Version
    pipeline_version = models.CharField(max_length=20, blank=True, default='')
    registered_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['hostname']

    def __str__(self):
        return f"{self.hostname} ({self.status})"


class PageResult(models.Model):
    """Per-page OCR processing result."""

    job = models.ForeignKey(
        Job, on_delete=models.CASCADE, related_name='page_results'
    )
    document_id = models.CharField(max_length=64)
    page_num = models.IntegerField()

    # OCR details
    ocr_method = models.CharField(max_length=50, blank=True, default='')
    ocr_language = models.CharField(max_length=20, blank=True, default='')
    ocr_confidence = models.FloatField(default=0.0)
    text_length = models.IntegerField(default=0)
    status = models.CharField(max_length=20, default='pending')

    # Worker tracking
    worker_hostname = models.CharField(max_length=255, blank=True, default='')
    processing_time_ms = models.IntegerField(default=0)
    celery_task_id = models.CharField(max_length=255, blank=True, default='')
    temp_pdf_path = models.CharField(max_length=1024, blank=True, default='')

    class Meta:
        ordering = ['job', 'page_num']
        constraints = [
            models.UniqueConstraint(
                fields=['job', 'page_num'],
                name='unique_job_page_num',
            ),
        ]

    def __str__(self):
        return f"Page {self.page_num} of {self.job_id} ({self.status})"


class CustodyEvent(models.Model):
    """Chain-of-custody event for forensic audit trail."""

    document_id = models.CharField(max_length=64, db_index=True)
    job = models.ForeignKey(
        Job, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='custody_events',
    )
    event_type = models.CharField(max_length=50)
    timestamp = models.DateTimeField(default=timezone.now)
    worker_hostname = models.CharField(max_length=255, blank=True, default='')
    data = models.JSONField(default=dict)
    prev_hash = models.CharField(max_length=64, blank=True, default='')
    event_hash = models.CharField(max_length=64, blank=True, default='')
    chain_finalized = models.BooleanField(default=False)

    class Meta:
        ordering = ['document_id', 'timestamp']

    def __str__(self):
        return f"{self.event_type} - {self.document_id} @ {self.timestamp}"


class PiiEntity(models.Model):
    """Spatial mapping for recognized PII/PHI entities."""

    job = models.ForeignKey(
        Job, on_delete=models.CASCADE, related_name='pii_entities'
    )
    page = models.ForeignKey(
        PageResult, on_delete=models.CASCADE, related_name='pii_entities'
    )
    
    entity_type = models.CharField(max_length=50, help_text="e.g., SSN, DOB, EMAIL, PHONE, NAME")
    entity_value = EncryptedTextField(help_text="The extracted PII/PHI text")
    
    # Spatial coordinates
    bounding_box = models.JSONField(
        help_text="Spatial coordinates [x1, y1, x2, y2]",
        default=list
    )
    
    confidence = models.FloatField(default=0.0, help_text="Confidence score from 0.0 to 1.0")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['page', 'entity_type']
        indexes = [
            models.Index(fields=['job', 'entity_type']),
            models.Index(fields=['page', 'entity_type']),
        ]

    def __str__(self):
        return f"{self.entity_type} on Page {self.page.page_num} ({self.confidence:.2f})"


class ApiKeyRecord(models.Model):
    """Tracks API key usage for access review and audit.

    Stores a hash of the API key (never the raw key) along with usage
    metadata.  Management commands use this model to list, audit, and
    revoke API keys.
    """

    key_id = models.CharField(
        max_length=64,
        unique=True,
        help_text="SHA-256 hash of the API key (never store raw keys)",
    )
    description = models.CharField(max_length=255, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)
    use_count = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True, db_index=True)
    permissions = models.JSONField(
        default=list,
        help_text="List of permission strings, e.g. ['read', 'write', 'admin']",
    )

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        truncated = self.key_id[:12] + '...' if len(self.key_id) > 12 else self.key_id
        status = 'active' if self.is_active else 'revoked'
        return f"ApiKey {truncated} ({status})"


class TranslationTenantConfig(models.Model):
    """Per-tenant translation policy and defaults (Plan B Wave M2).

    One row per tenant.  Hydrated into ``ocr_local.translation.policy.TenantPolicy``
    via ``get_tenant_policy(tenant_id)`` at routing time.  When no row exists
    a safe default is returned (deny NC-licensed weights, no certified flag).
    """

    tenant_id = models.CharField(
        max_length=128,
        unique=True,
        db_index=True,
        help_text="Tenant identifier; must be unique.",
    )
    target_languages = models.JSONField(
        default=list,
        blank=True,
        help_text="Default target BCP-47 codes, e.g. ['en', 'fr'].",
    )
    preferred_engines = models.JSONField(
        default=list,
        blank=True,
        help_text="Engine ids in priority order, e.g. ['opus_mt', 'nllb_200'].",
    )
    allow_nc_licensed = models.BooleanField(
        default=False,
        help_text="When True, NC-licensed engines (e.g. NLLB CC-BY-NC-4.0) are eligible.",
    )
    require_certified = models.BooleanField(
        default=False,
        help_text=(
            "Tenant policy flag: requires translations to be reviewed before "
            "any 'certified' attestation can be applied.  This does NOT bypass "
            "the certified=False enforcement at the sidecar write path."
        ),
    )
    default_quality_tier = models.CharField(
        max_length=20,
        default='standard',
        blank=True,
        help_text="One of: draft, standard, legal.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['tenant_id']

    def __str__(self):
        return f"TranslationTenantConfig({self.tenant_id})"


class GlossaryEntry(models.Model):
    """Per-tenant glossary entry for translation term overrides (Plan B Wave M2).

    Entries are applied in ``priority`` ascending order at translation time.
    Literal entries do exact-substring replacement (case-sensitive when
    ``case_sensitive=True``); regex entries use compiled ``re.sub``.
    """

    tenant_id = models.CharField(
        max_length=128,
        db_index=True,
        help_text="Tenant identifier (matches TranslationTenantConfig.tenant_id).",
    )
    source_term = models.CharField(max_length=500)
    target_term = models.CharField(max_length=500)
    source_lang = models.CharField(max_length=10, help_text="BCP-47 code of source.")
    target_lang = models.CharField(max_length=10, help_text="BCP-47 code of target.")
    case_sensitive = models.BooleanField(default=False)
    is_regex = models.BooleanField(
        default=False,
        help_text="When True, source_term is interpreted as a Python regex.",
    )
    priority = models.IntegerField(
        default=100,
        help_text="Lower priority is applied first (10 < 100).",
    )
    notes = models.TextField(blank=True, default='', null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['tenant_id', 'priority', 'id']
        indexes = [
            models.Index(fields=['tenant_id']),
            models.Index(fields=['tenant_id', 'source_lang', 'target_lang']),
        ]
        constraints = [
            # Unique on (tenant_id, source_term, source_lang, target_lang) for
            # literal (non-regex) entries.  Regex entries can have duplicate
            # source_term values (different patterns may overlap).
            models.UniqueConstraint(
                fields=['tenant_id', 'source_term', 'source_lang', 'target_lang'],
                condition=models.Q(is_regex=False),
                name='unique_literal_glossary_entry',
            ),
        ]

    def __str__(self):
        return (
            f"GlossaryEntry(tenant={self.tenant_id}, "
            f"{self.source_term}->{self.target_term}, "
            f"{self.source_lang}->{self.target_lang})"
        )


class BatchTranslationJob(models.Model):
    """Batch translation job header (Plan B Wave M2 -- B17).

    Persists the high-level batch state.  Individual inputs live in
    ``BatchTranslationInput`` rows and are dispatched as Celery subtasks
    on the dedicated ``translation_batch`` queue.
    """

    STATUS_PENDING = 'pending'
    STATUS_RUNNING = 'running'
    STATUS_COMPLETED = 'completed'
    STATUS_FAILED = 'failed'
    STATUS_CANCELLED = 'cancelled'

    TERMINAL_STATUSES = frozenset({
        STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED,
    })

    batch_id = models.CharField(
        primary_key=True,
        max_length=64,
        help_text='UUID4 hex of the batch.',
    )
    tenant_id = models.CharField(
        max_length=128,
        db_index=True,
        help_text='Tenant identifier.',
    )
    source_lang = models.CharField(
        max_length=10,
        help_text='BCP-47 source language.',
    )
    target_lang = models.CharField(
        max_length=10,
        help_text='BCP-47 target language.',
    )
    status = models.CharField(
        max_length=20,
        default=STATUS_PENDING,
        db_index=True,
        help_text='One of: pending, running, completed, failed, cancelled.',
    )
    priority = models.IntegerField(
        default=0,
        help_text='Celery priority 0..9 (lower = lower priority).',
    )
    glossary_enabled = models.BooleanField(default=True)
    total_inputs = models.IntegerField(default=0)
    completed_inputs = models.IntegerField(default=0)
    failed_inputs = models.IntegerField(default=0)
    submitted_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        ordering = ['-submitted_at']

    def __str__(self):
        return (
            f"BatchTranslationJob({self.batch_id} tenant={self.tenant_id} "
            f"{self.source_lang}->{self.target_lang} {self.status})"
        )


class BatchTranslationInput(models.Model):
    """Single input within a :class:`BatchTranslationJob`.

    ``client_ref`` is unique per batch (enforced at submit time and via a
    UniqueConstraint).  Result fields (``target_text``, ``engine_id``,
    ``confidence``, ``glossary_hits_json``, ``error``) are populated by
    the worker once the Celery subtask completes.
    """

    batch = models.ForeignKey(
        BatchTranslationJob,
        on_delete=models.CASCADE,
        related_name='inputs',
    )
    client_ref = models.CharField(
        max_length=128,
        help_text='Tenant-supplied opaque id, unique per batch.',
    )
    input_index = models.IntegerField(
        default=0,
        help_text='Position in the original submitted list.',
    )
    text = models.TextField()
    status = models.CharField(
        max_length=20,
        default='pending',
        db_index=True,
    )
    target_text = models.TextField(blank=True, default='')
    engine_id = models.CharField(max_length=64, blank=True, default='')
    confidence = models.FloatField(blank=True, null=True)
    glossary_hits_json = models.JSONField(blank=True, default=list)
    error = models.TextField(blank=True, default='')
    celery_task_id = models.CharField(max_length=128, blank=True, default='')
    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    optional_metadata_json = models.JSONField(blank=True, default=dict)

    class Meta:
        ordering = ['batch', 'input_index']
        indexes = [
            models.Index(
                fields=['batch', 'status'],
                name='jobs_btinp_batch_s_idx',
            ),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=('batch', 'client_ref'),
                name='unique_batch_client_ref',
            ),
        ]

    def __str__(self):
        return (
            f"BatchTranslationInput(batch={self.batch_id} "
            f"client_ref={self.client_ref} status={self.status})"
        )


# Import secondary model modules so Django registers them under the jobs app.
from .extraction_models import DocumentChunk, ExtractedEntity, ExtractedFormValue  # noqa: E402,F401
