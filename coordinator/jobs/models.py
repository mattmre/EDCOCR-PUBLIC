import uuid

from django.db import models
from django.utils import timezone

from .encrypted_field import EncryptedTextField
from .webhook_secret_field import WebhookSecretField


class Job(models.Model):
    """Represents an OCR processing job."""

    class Status(models.TextChoices):
        SUBMITTED = "submitted", "Submitted"
        INGESTING = "ingesting", "Ingesting"
        PROCESSING = "processing", "Processing"
        ASSEMBLING = "assembling", "Assembling"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    class Priority(models.TextChoices):
        URGENT = "urgent", "Urgent"
        NORMAL = "normal", "Normal"
        LOW = "low", "Low"

    job_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
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
    source_hash = models.CharField(max_length=64, blank=True, default="")
    source_type = models.CharField(max_length=10, blank=True, default="")
    detected_language = models.CharField(max_length=20, blank=True, default="und")
    # Durable service-API admission. The raw idempotency key is never stored.
    # ``submission_ready`` becomes true only after the exact source bytes are
    # committed to object storage. The root ingest outbox is then created
    # idempotently before a successful admission response is returned.
    submission_key_hash = models.CharField(max_length=64, blank=True, default="")
    submission_ready = models.BooleanField(default=False, db_index=True)
    source_object_key = models.CharField(max_length=1024, blank=True, default="")
    source_size_bytes = models.BigIntegerField(default=0)

    # Progress
    total_pages = models.IntegerField(default=0)
    pages_completed = models.IntegerField(default=0)
    pages_failed = models.IntegerField(default=0)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    custody_published_at = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Last terminal custody chain durably published to job storage.",
    )

    @property
    def processing_time_seconds(self):
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    # Configuration
    settings_json = models.JSONField(default=dict, blank=True)
    nfs_job_path = models.CharField(max_length=1024, blank=True, default="")
    storage_backend_used = models.CharField(
        max_length=10,
        blank=True,
        default="",
        help_text="Storage backend locked at ingest time (nfs or s3)",
    )

    # Results
    result_summary = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True, default="")

    # Tenant tracking (optional, empty string means no tenant)
    tenant_id = models.CharField(max_length=128, blank=True, default="", db_index=True)

    # Cross-reference to API server (nullable — only set when job originates from API)
    api_job_id = models.CharField(max_length=64, blank=True, default="", db_index=True)

    # Celery tracking
    celery_task_id = models.CharField(max_length=255, blank=True, default="")
    # Native/skip_ocr document owner is a fresh execution nonce, independent
    # of the broker task id (which is reused on at-least-once redelivery).
    processing_attempt_token = models.CharField(max_length=64, blank=True, default="")
    processing_attempt_identity = models.CharField(max_length=64, blank=True, default="")
    processing_lease_expires_at = models.DateTimeField(null=True, blank=True)
    assigned_worker = models.CharField(max_length=255, blank=True, default="")
    # Durable fan-out idempotency guard. A duplicate chord dispatch must not
    # create a second PageResult fan-out for the same parent job.
    fanout_dispatched = models.BooleanField(default=False)
    last_page_admission_at = models.DateTimeField(null=True, blank=True)
    # Monotonic root-dispatch generation.  The generation is paired with a
    # stage in TaskDispatchOutbox so a retry republishes the same logical
    # delivery rather than creating a second work contract.
    dispatch_generation = models.PositiveIntegerField(default=0)

    # Federation tracking (Plan C Phase 1, item C5 -- failover)
    # ``assigned_cluster`` records which federation peer is currently
    # responsible for this job; the failover engine uses it to identify
    # in-flight work that needs to be re-dispatched when a peer goes
    # unhealthy. Empty string means "local cluster" / "not federated".
    assigned_cluster = models.CharField(
        max_length=128,
        blank=True,
        default="",
        db_index=True,
    )
    # ``last_failure_reason`` is a short, machine-readable tag explaining
    # the last failure observed for this job (e.g. ``cluster_unhealthy``,
    # ``s3_download_error``).  Operators read it from the dashboard; the
    # field is purely informational and never user-facing.
    last_failure_reason = models.CharField(
        max_length=64,
        blank=True,
        default="",
    )

    # Webhook
    webhook_url = models.URLField(max_length=2048, blank=True, default="")
    webhook_secret = WebhookSecretField(blank=True, default="")
    webhook_status = models.CharField(max_length=20, blank=True, default="")

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["tenant_id", "submission_key_hash"],
                condition=~models.Q(submission_key_hash=""),
                name="unique_job_tenant_submission_key",
            ),
        ]

    def __str__(self):
        return f"Job {self.job_id} ({self.status})"


class TaskDispatchOutbox(models.Model):
    """Durable DB-to-broker handoff for coordinator-owned task dispatch.

    A row is created in the same transaction as the state transition that
    makes the task eligible.  A bounded relay claims the row with a lease and
    publishes it outside the transaction.  If the relay dies after broker
    publication, the row can be retried; the consumer's job/page lease fences
    duplicate deliveries.

    The first wired stage is the root processing dispatch.  Fan-out/chords
    and post-processing intentionally remain separate contracts until this
    primitive has crash-injection evidence.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        CLAIMED = "claimed", "Claimed"
        PUBLISHED = "published", "Published"
        FAILED = "failed", "Failed"
        DEAD = "dead", "Dead letter"

    dispatch_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    job = models.ForeignKey(
        Job,
        on_delete=models.CASCADE,
        related_name="task_dispatches",
    )
    stage = models.CharField(max_length=64)
    generation = models.CharField(max_length=64)
    task_name = models.CharField(max_length=255)
    args = models.JSONField(default=list, blank=True)
    kwargs = models.JSONField(default=dict, blank=True)
    queue = models.CharField(max_length=128, blank=True, default="")
    priority = models.PositiveSmallIntegerField(default=0)
    max_attempts = models.PositiveSmallIntegerField(
        default=12,
        help_text="Bounded broker publish attempts before dead-lettering.",
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
    )
    attempts = models.PositiveIntegerField(default=0)
    claim_token = models.CharField(max_length=64, blank=True, default="")
    lease_expires_at = models.DateTimeField(null=True, blank=True, db_index=True)
    available_at = models.DateTimeField(default=timezone.now, db_index=True)
    broker_task_id = models.CharField(max_length=255, blank=True, default="")
    last_error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_at = models.DateTimeField(null=True, blank=True)
    dead_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["job", "stage", "generation"],
                name="unique_task_dispatch_job_stage_generation",
            ),
        ]
        indexes = [
            models.Index(
                fields=["status", "available_at"],
                name="jobs_dispatch_status_avail_idx",
            ),
            models.Index(
                fields=["job", "stage", "status"],
                name="jobs_dispatch_job_stage_idx",
            ),
        ]

    def __str__(self):
        return (
            f"TaskDispatchOutbox({self.job_id} {self.stage} "
            f"gen={self.generation} {self.status})"
        )


class PresignedTaskCapability(models.Model):
    """Job-bound, short-lived storage capability for a worker task.

    The capability identifier may travel through Celery, but the bearer URLs
    never do.  The encrypted payload is resolved by a worker after it has
    loaded the job row, which gives the coordinator a durable job/operation/
    expiry binding and keeps URL material out of broker messages and task
    history.
    """

    capability_id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    job = models.ForeignKey(
        Job,
        on_delete=models.CASCADE,
        related_name="presigned_capabilities",
    )
    operation = models.CharField(max_length=64)
    dispatch_generation = models.PositiveIntegerField(
        default=0,
        help_text="Job dispatch generation authorized when the capability was issued.",
    )
    payload = EncryptedTextField(
        help_text="Encrypted presigned URL map; never serialize this field into Celery.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(db_index=True)

    class Meta:
        indexes = [
            models.Index(
                fields=["job", "operation", "expires_at"],
                name="jobs_presigned_job_op_exp_idx",
            ),
        ]

    def __str__(self):
        return f"Presigned capability {self.capability_id} ({self.operation})"


class Worker(models.Model):
    """Represents a Celery worker node."""

    class Status(models.TextChoices):
        ONLINE = "online", "Online"
        BUSY = "busy", "Busy"
        OFFLINE = "offline", "Offline"
        DRAINING = "draining", "Draining"

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
    gpu_model = models.CharField(max_length=255, blank=True, default="")
    gpu_vram_mb = models.IntegerField(default=0)
    gpu_index = models.IntegerField(null=True, blank=True, default=None)
    cpu_cores = models.IntegerField(default=0)
    ram_mb = models.IntegerField(default=0)

    # Status tracking
    last_heartbeat = models.DateTimeField(null=True, blank=True)
    tasks_completed = models.IntegerField(default=0)
    tasks_failed = models.IntegerField(default=0)
    current_task_id = models.CharField(max_length=255, blank=True, default="")

    # Version
    pipeline_version = models.CharField(max_length=20, blank=True, default="")
    registered_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["hostname"]

    def __str__(self):
        return f"{self.hostname} ({self.status})"


class PageResult(models.Model):
    """Per-page OCR processing result."""

    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="page_results")
    document_id = models.CharField(max_length=64)
    page_num = models.IntegerField()

    # OCR details
    ocr_method = models.CharField(max_length=50, blank=True, default="")
    ocr_language = models.CharField(max_length=20, blank=True, default="")
    ocr_confidence = models.FloatField(default=0.0)
    needs_review = models.BooleanField(default=True)
    ocr_provenance = models.JSONField(blank=True, default=dict)
    text_length = models.IntegerField(default=0)
    status = models.CharField(max_length=20, default="pending")

    # Worker tracking
    worker_hostname = models.CharField(max_length=255, blank=True, default="")
    processing_time_ms = models.IntegerField(default=0)
    celery_task_id = models.CharField(max_length=255, blank=True, default="")
    lease_expires_at = models.DateTimeField(null=True, blank=True, db_index=True)
    temp_pdf_path = models.CharField(max_length=1024, blank=True, default="")
    admitted_generation = models.PositiveIntegerField(null=True, blank=True, db_index=True)

    class Meta:
        ordering = ["job", "page_num"]
        constraints = [
            models.UniqueConstraint(
                fields=["job", "page_num"],
                name="unique_job_page_num",
            ),
        ]

    def __str__(self):
        return f"Page {self.page_num} of {self.job_id} ({self.status})"


class PageOCRStage(models.Model):
    """Durable, generation-bound CPU render / GPU OCR / CPU finish handoff."""

    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="ocr_stages")
    page_num = models.PositiveIntegerField()
    generation = models.PositiveIntegerField()
    tenant_id = models.CharField(max_length=128)
    source_sha256 = models.CharField(max_length=64)
    status = models.CharField(max_length=20, default="pending", db_index=True)
    lease_token = models.CharField(max_length=64, blank=True, default="")
    lease_expires_at = models.DateTimeField(null=True, blank=True, db_index=True)
    raster_locator = models.CharField(max_length=1024, blank=True, default="")
    # A crashed renderer can leave an HTTP PUT still completing after its
    # lease is reclaimed. Keep the exact old locator and reservation until
    # this bounded in-flight publication window has passed.
    publish_safe_after = models.DateTimeField(null=True, blank=True, db_index=True)
    raster_sha256 = models.CharField(max_length=64, blank=True, default="")
    raster_size_bytes = models.PositiveBigIntegerField(default=0)
    raster_width = models.PositiveIntegerField(default=0)
    raster_height = models.PositiveIntegerField(default=0)
    raster_dpi = models.PositiveSmallIntegerField(default=300)
    retry_dpi = models.PositiveSmallIntegerField(default=300)
    geometry = models.JSONField(default=dict, blank=True)
    script_evidence = models.JSONField(default=list, blank=True)
    reading_direction = models.JSONField(default=dict, blank=True)
    inference = models.JSONField(default=dict, blank=True)
    reserved_bytes = models.PositiveBigIntegerField(default=0)
    attempts = models.PositiveSmallIntegerField(default=0)
    render_attempts = models.PositiveSmallIntegerField(default=0)
    infer_attempts = models.PositiveSmallIntegerField(default=0)
    redrive_count = models.PositiveIntegerField(default=0)
    last_error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["job", "page_num", "generation"],
                name="unique_page_ocr_stage_generation",
            ),
        ]


class PageOCRByteBudget(models.Model):
    """Single DB-locked counter for fleet-wide staged raster reservations."""

    id = models.PositiveSmallIntegerField(primary_key=True, default=1)
    reserved_bytes = models.PositiveBigIntegerField(default=0)


class PageOCRRasterRetirement(models.Model):
    """Durable exact-key cleanup for a raster PUT whose owner disappeared."""

    job_uuid = models.UUIDField(db_index=True)
    tenant_id = models.CharField(max_length=128)
    generation = models.PositiveIntegerField()
    page_num = models.PositiveIntegerField()
    locator = models.CharField(max_length=1024, unique=True)
    backend_binding = models.CharField(max_length=64)
    reserved_bytes = models.PositiveBigIntegerField(default=0)
    safe_to_release_after = models.DateTimeField()
    next_sweep_at = models.DateTimeField(db_index=True)
    last_swept_at = models.DateTimeField(null=True, blank=True)
    sweep_count = models.PositiveIntegerField(default=0)
    last_error = models.CharField(max_length=256, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)


class JobArtifactRetirement(models.Model):
    """Cleanup authority for late uploads, independent of the deleted Job."""

    job_uuid = models.UUIDField(primary_key=True)
    tenant_id = models.CharField(max_length=128)
    backend_binding = models.CharField(max_length=64)
    next_sweep_at = models.DateTimeField(db_index=True)
    last_swept_at = models.DateTimeField(null=True, blank=True)
    sweep_count = models.PositiveIntegerField(default=0)
    last_error = models.CharField(max_length=256, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)


class CustodyEvent(models.Model):
    """Chain-of-custody event for forensic audit trail."""

    document_id = models.CharField(max_length=64, db_index=True)
    job = models.ForeignKey(
        Job,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="custody_events",
    )
    tenant_id = models.CharField(
        max_length=128,
        blank=True,
        default="",
        db_index=True,
        help_text=(
            "Persisted tenant binding for jobless control-plane events; "
            "job-backed events inherit their parent tenant."
        ),
    )
    chain_id = models.CharField(
        max_length=64,
        blank=True,
        default="",
        db_index=True,
        help_text="Immutable custody-chain identity independent of the Job foreign key.",
    )
    hash_version = models.PositiveSmallIntegerField(
        default=1,
        help_text="Custody envelope version (1=legacy, 2=chain-bound).",
    )
    event_type = models.CharField(max_length=50)
    timestamp = models.DateTimeField(default=timezone.now)
    worker_hostname = models.CharField(max_length=255, blank=True, default="")
    data = models.JSONField(default=dict)
    prev_hash = models.CharField(max_length=64, blank=True, default="")
    event_hash = models.CharField(max_length=64, blank=True, default="")
    chain_finalized = models.BooleanField(default=False)

    class Meta:
        ordering = ["document_id", "timestamp"]

    def __str__(self):
        return f"{self.event_type} - {self.document_id} @ {self.timestamp}"

    def save(self, *args, **kwargs):
        if not self.chain_id:
            import hashlib

            if self.job_id:
                identity = f"job\0{self.job_id}\0{self.document_id}"
            else:
                identity = f"control\0{self.tenant_id}\0{self.document_id}"
            self.chain_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if not self.tenant_id and self.job_id:
            parent = getattr(self, "job", None)
            self.tenant_id = str(getattr(parent, "tenant_id", "") or "").strip()
        super().save(*args, **kwargs)


class CustodyArchiveHead(models.Model):
    """Durable predecessor manifest for each rotated custody chain."""

    chain_id = models.CharField(max_length=64, primary_key=True)
    event_hash = models.CharField(max_length=64)
    last_event_id = models.PositiveBigIntegerField(default=0)
    archive_path = models.CharField(max_length=2048, blank=True, default="")
    archive_checksum = models.CharField(max_length=64, blank=True, default="")
    legacy_unverified = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)


class PiiEntity(models.Model):
    """Spatial mapping for recognized PII/PHI entities."""

    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="pii_entities")
    page = models.ForeignKey(
        PageResult, on_delete=models.CASCADE, related_name="pii_entities"
    )

    entity_type = models.CharField(
        max_length=50, help_text="e.g., SSN, DOB, EMAIL, PHONE, NAME"
    )
    entity_value = EncryptedTextField(help_text="The extracted PII/PHI text")

    # Spatial coordinates
    bounding_box = models.JSONField(
        help_text="Spatial coordinates [x1, y1, x2, y2]", default=list
    )

    confidence = models.FloatField(
        default=0.0, help_text="Confidence score from 0.0 to 1.0"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["page", "entity_type"]
        indexes = [
            models.Index(fields=["job", "entity_type"]),
            models.Index(fields=["page", "entity_type"]),
        ]

    def __str__(self):
        return (
            f"{self.entity_type} on Page {self.page.page_num} ({self.confidence:.2f})"
        )


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
    description = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)
    use_count = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True, db_index=True)
    permissions = models.JSONField(
        default=list,
        help_text="List of permission strings, e.g. ['read', 'write', 'admin']",
    )

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        truncated = self.key_id[:12] + "..." if len(self.key_id) > 12 else self.key_id
        status = "active" if self.is_active else "revoked"
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
            "the certified=False enforcement at the sidecar write path. "
            "When True, delivery of uncertified batch/job results is blocked (S2-04)."
        ),
    )
    residency_region = models.CharField(
        max_length=64,
        blank=True,
        default="",
        help_text=(
            "Optional data-residency pin (e.g. 'eu-west-1', 'eu'). When set, "
            "cloud translation engines are rejected with RESIDENCY_VIOLATION "
            "until region-affinity cloud adapters exist (S2-04)."
        ),
    )
    default_quality_tier = models.CharField(
        max_length=20,
        default="standard",
        blank=True,
        help_text="One of: draft, standard, legal.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["tenant_id"]

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
    notes = models.TextField(blank=True, default="", null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["tenant_id", "priority", "id"]
        indexes = [
            models.Index(fields=["tenant_id"]),
            models.Index(fields=["tenant_id", "source_lang", "target_lang"]),
        ]
        constraints = [
            # Unique on (tenant_id, source_term, source_lang, target_lang) for
            # literal (non-regex) entries.  Regex entries can have duplicate
            # source_term values (different patterns may overlap).
            models.UniqueConstraint(
                fields=["tenant_id", "source_term", "source_lang", "target_lang"],
                condition=models.Q(is_regex=False),
                name="unique_literal_glossary_entry",
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

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"

    TERMINAL_STATUSES = frozenset(
        {
            STATUS_COMPLETED,
            STATUS_FAILED,
            STATUS_CANCELLED,
        }
    )

    batch_id = models.CharField(
        primary_key=True,
        max_length=64,
        help_text="UUID4 hex of the batch.",
    )
    tenant_id = models.CharField(
        max_length=128,
        db_index=True,
        help_text="Tenant identifier.",
    )
    source_lang = models.CharField(
        max_length=10,
        help_text="BCP-47 source language.",
    )
    target_lang = models.CharField(
        max_length=10,
        help_text="BCP-47 target language.",
    )
    status = models.CharField(
        max_length=20,
        default=STATUS_PENDING,
        db_index=True,
        help_text="One of: pending, running, completed, failed, cancelled.",
    )
    priority = models.IntegerField(
        default=0,
        help_text="Celery priority 0..9 (lower = lower priority).",
    )
    glossary_enabled = models.BooleanField(default=True)
    total_inputs = models.IntegerField(default=0)
    completed_inputs = models.IntegerField(default=0)
    failed_inputs = models.IntegerField(default=0)
    # S2-04: delivery gate. Write path never sets this True; only review
    # certify (strong-auth + real custody_event_id) may promote it.
    certified = models.BooleanField(
        default=False,
        help_text=(
            "True only after review certify with real custody linkage. "
            "When tenant require_certified=True, result delivery is blocked "
            "until this flag is True (S2-04)."
        ),
    )
    submitted_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        ordering = ["-submitted_at"]

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
        related_name="inputs",
    )
    client_ref = models.CharField(
        max_length=128,
        help_text="Tenant-supplied opaque id, unique per batch.",
    )
    input_index = models.IntegerField(
        default=0,
        help_text="Position in the original submitted list.",
    )
    text = models.TextField()
    status = models.CharField(
        max_length=20,
        default="pending",
        db_index=True,
    )
    target_text = models.TextField(blank=True, default="")
    engine_id = models.CharField(max_length=64, blank=True, default="")
    confidence = models.FloatField(blank=True, null=True)
    glossary_hits_json = models.JSONField(blank=True, default=list)
    error = models.TextField(blank=True, default="")
    celery_task_id = models.CharField(max_length=128, blank=True, default="")
    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    optional_metadata_json = models.JSONField(blank=True, default=dict)

    class Meta:
        ordering = ["batch", "input_index"]
        indexes = [
            models.Index(
                fields=["batch", "status"],
                name="jobs_btinp_batch_s_idx",
            ),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=("batch", "client_ref"),
                name="unique_batch_client_ref",
            ),
        ]

    def __str__(self):
        return (
            f"BatchTranslationInput(batch={self.batch_id} "
            f"client_ref={self.client_ref} status={self.status})"
        )


# Import secondary model modules so Django registers them under the jobs app.
from .extraction_models import (  # noqa: E402,F401
    DocumentChunk,
    ExtractedEntity,
    ExtractedFormValue,
)
